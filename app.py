import os, json, re, unicodedata
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
import time, random

# ---------- Config ----------
load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # fast & cheap


# support multiple API keys via comma-separated env: GROQ_API_KEYS="key1,key2,key3"
def _load_groq_keys() -> list[str]:
    keys_env = os.getenv("GROQ_API_KEYS", "").strip()
    if keys_env:
        keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        if keys:
            return keys
    # fallback to single key for backward-compat
    single = os.getenv("GROQ_API_KEY", "").strip()
    return [single] if single else []


GROQ_API_KEYS: list[str] = _load_groq_keys()

# --- Key selection state (primary + secondary rotation) ---

PRIMARY_KEY: str = (os.getenv("GROQ_API_KEY", "") or "").strip()
SECONDARY_KEYS: list[str] = [k for k in GROQ_API_KEYS if k and k != PRIMARY_KEY]

# Per-key cooldowns (unix timestamps) + permanently bad keys
KEY_BACKOFFS: dict[str, float] = {}
BAD_KEYS: set[str] = set()

# Round-robin pointer for SECONDARY_KEYS
RR_IDX: int = 0


def _select_key_order() -> list[tuple[str, str]]:
    """
    Returns a prioritized list of (kind, key):
      - 'primary' first if ready (not bad & beyond cooldown)
      - then secondary keys in round-robin order if ready
      - if nothing is ready, returns the earliest-cooling key as a last resort
    """
    now = time.time()
    order: list[tuple[str, str]] = []

    # 1) Primary first when ready
    if (
        PRIMARY_KEY
        and PRIMARY_KEY not in BAD_KEYS
        and KEY_BACKOFFS.get(PRIMARY_KEY, 0) <= now
    ):
        order.append(("primary", PRIMARY_KEY))

    # 2) Secondary keys (ready) in round-robin order
    if SECONDARY_KEYS:
        n = len(SECONDARY_KEYS)
        start = RR_IDX % n
        rotated = SECONDARY_KEYS[start:] + SECONDARY_KEYS[:start]
        for k in rotated:
            if k in BAD_KEYS:
                continue
            if KEY_BACKOFFS.get(k, 0) <= now:
                order.append(("secondary", k))

    # 3) If nothing ready, add the earliest-cooling candidate as a last resort
    if not order:
        candidates: list[tuple[float, tuple[str, str]]] = []
        if PRIMARY_KEY and PRIMARY_KEY not in BAD_KEYS:
            candidates.append(
                (KEY_BACKOFFS.get(PRIMARY_KEY, 0), ("primary", PRIMARY_KEY))
            )
        for k in SECONDARY_KEYS:
            if k in BAD_KEYS:
                continue
            candidates.append((KEY_BACKOFFS.get(k, 0), ("secondary", k)))
        if candidates:
            candidates.sort(key=lambda t: t[0])
            order.append(candidates[0][1])

    return order


DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "corpus.json"
XP_PATH = DATA_DIR / "experience.json"  # used for experiences list + retrieval merge

# Hard-coded contacts
CONTACT_LINKS = {
    "linkedin": "http://www.linkedin.com/in/pankit-shah13",
    "github": "https://github.com/Shah-Pankit",
    "phone": "+91 722- 901-3335",
    "email": "pankitshah493@gmail.com",
}

# ---------- App ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Load corpus ----------
def load_corpus() -> list[str]:
    if not CORPUS_PATH.exists():
        raise RuntimeError(
            "data/corpus.json not found. Put your corpus.json in the data/ folder."
        )
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


CORPUS = load_corpus()

# ---------- Experience helpers (also used for retrieval) ----------
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _normalize_dash(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace("–", "-").replace("—", "-")


def _parse_start(duration: str) -> tuple[int, int]:
    if not duration:
        return (0, 0)
    d = _normalize_dash(duration.lower())
    parts = d.split("-")
    start = parts[0].strip() if parts else ""
    toks = start.split()
    if len(toks) < 2:
        return (0, 0)
    mon = MONTHS.get(toks[0][:3], 0)
    yr = 0
    try:
        yr = int("".join(ch for ch in toks[1] if ch.isdigit()))
    except:
        pass
    return (yr, mon)


def _load_exp_entries():
    if not XP_PATH.exists():
        return []
    try:
        items = json.loads(XP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    seen = set()
    out = []
    for e in items:
        title = (e.get("title") or "").strip()
        company = (e.get("company") or "").strip()
        duration = (e.get("duration") or "").strip()
        description = (e.get("description") or "").strip()
        key = (title.lower(), company.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "title": title,
                "company": company,
                "duration": duration,
                "description": description,
                "sort_key": _parse_start(duration),
            }
        )
    out.sort(key=lambda x: (x["sort_key"][0], x["sort_key"][1]), reverse=True)
    return out


def _experience_texts_for_retrieval(exps: list[dict]) -> list[str]:
    # Convert each experience entry into a retrieval-friendly text chunk
    texts = []
    for e in exps:
        t = f'{e["title"]} at {e["company"]} ({e["duration"]}): {e["description"]}'
        texts.append(t)
    return texts


def format_experiences_ul(exps) -> str:
    if not exps:
        return "<ul><li>No experience data found.</li></ul>"
    items = []
    for e in exps:
        t = f'{e["title"]} at {e["company"]} ({e["duration"]}): {e["description"]}'
        items.append(f"<li>{t}</li>")
    return f"<ul>{''.join(items)}</ul>"


# ---------- Retrieval (keyword-only, merged corpus + experiences) ----------
def retrieve_merged(query: str, k: int = 8) -> List[str]:
    """Ultra-light keyword retrieval over corpus + experience entries."""
    if not query.strip():
        return []
    exps = _load_exp_entries()
    exp_texts = _experience_texts_for_retrieval(exps)
    docs = CORPUS + exp_texts

    q = query.lower()
    words = set(re.findall(r"[a-z0-9]+", q))
    scored = []
    for i, txt in enumerate(docs):
        t = txt.lower()
        score = sum(t.count(w) for w in words)
        scored.append((score, i))
    scored.sort(reverse=True)
    idxs = [i for s, i in scored[:k] if s > 0]
    if not idxs:
        idxs = list(range(min(k, len(docs))))
    return [(CORPUS + exp_texts)[i] for i in idxs]


# ---------- Response guards / formatting ----------
LINK_TRAIL = re.compile(r"(https?://\S+?)([.,;:!?])(\s|$)")


def clean_links(s: str) -> str:
    return LINK_TRAIL.sub(r"\1\3", s)


def inject_contacts(text: str) -> str:
    for key, val in CONTACT_LINKS.items():
        text = text.replace(f"{{{key}}}", val)
    return clean_links(text)


def format_contacts_ul() -> str:
    github = CONTACT_LINKS["github"]
    linkedin = CONTACT_LINKS["linkedin"]
    email = CONTACT_LINKS["email"]
    phone = CONTACT_LINKS["phone"]
    return (
        "<ul>"
        f"<li>GitHub: {github}</li>"
        f"<li>LinkedIn: {linkedin}</li>"
        f"<li>Email: {email}</li>"
        f"<li>Phone: {phone}</li>"
        "</ul>"
    )


# --- Markdown stripping + bullet forcing (for clean frontend rendering) ---
_MD_BOLD = re.compile(r"\*\*(.*?)\*\*")
_MD_ITAL = re.compile(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)")
_MD_ULINE = re.compile(r"__(.*?)__")
_MD_EMPH = re.compile(r"_(.*?)_")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _strip_markdown(s: str) -> str:
    s = _MD_LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", s)
    s = _MD_BOLD.sub(r"\1", s)
    s = _MD_ULINE.sub(r"\1", s)
    s = _MD_ITAL.sub(r"\1", s)
    s = _MD_EMPH.sub(r"\1", s)
    return s


def markdown_to_ul(text: str) -> str:
    if not text:
        return "<ul><li></li></ul>"
    if "<ul" in text and "<li>" in text:
        return text
    s = _strip_markdown(text).replace("\r", "\n").strip()
    bullet_pat = re.compile(
        r"(?:^|\n)\s*(?:[-*•]|[\d]{1,2}[.)])\s+(.*?)(?=(?:\n\s*(?:[-*•]|[\d]{1,2}[.)])\s+)|$)",
        flags=re.S,
    )
    items = [itm.strip() for itm in bullet_pat.findall(s) if itm.strip()]
    if not items:
        chunks = [c.strip() for c in re.split(r"\n{2,}", s) if c.strip()]
        if len(chunks) > 1:
            items = chunks
    if not items:
        sentences = [t.strip() for t in re.split(r"(?<=[.!?])\s+", s) if t.strip()]
        if len(sentences) <= 6:
            items = sentences
        else:
            step = max(1, len(sentences) // 5)
            items = [
                " ".join(sentences[i : i + step]).strip()
                for i in range(0, len(sentences), step)
            ]
            items = [i for i in items if i][:6]
    if not items:
        items = [s]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"





def force_ul_list(text: str) -> str:
    return markdown_to_ul(text)


SYSTEM = (
    "You are Pankit Shah (AI/ML Engineer). Speak in first person as Pankit.\n"
    "Use ONLY the provided CONTEXT to answer. If the info is not in CONTEXT, reply:\n"
    '"I don’t have that in my portfolio context."\n'
    "For lists, use valid HTML <ul><li>…</li></ul>."
    "if asked for whatsapp number or number reply like.. Sure, here is my Number : +91 722-901-3335 "
)
CONTACT_INSTR = (
    "If asked for my GitHub, LinkedIn, or phone, respond with the exact values:\n"
    f"GitHub: {CONTACT_LINKS['github']}\n"
    f"LinkedIn: {CONTACT_LINKS['linkedin']}\n"
    f"Phone: {CONTACT_LINKS['phone']}\n"
    "Do NOT place punctuation immediately after any link."
)


# # ---------- Groq call with fallback over multiple API keys ----------
# async def groq_chat_completion(
#     messages: list[dict], temperature: float = 0.2
# ) -> Optional[str]:
#     if not GROQ_API_KEYS:
#         return None
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         for key in GROQ_API_KEYS:
#             try:
#                 r = await client.post(
#                     "https://api.groq.com/openai/v1/chat/completions",
#                     headers={"Authorization": f"Bearer {key}"},
#                     json={
#                         "model": GROQ_MODEL,
#                         "messages": messages,
#                         "temperature": temperature,
#                     },
#                 )
#             except httpx.HTTPError:
#                 continue
#             if r.status_code in (401, 403, 408, 409, 429, 500, 502, 503, 504):
#                 continue
#             if r.status_code != 200:
#                 continue
#             data = r.json()
#             reply = (data.get("choices") or [{}])[0].get("message", {}).get("content")
#             if reply:
#                 return reply
#     return None


# ---------- Groq call with cooldown + primary re-probe + RR secondaries ----------
async def groq_chat_completion(
    messages: list[dict], temperature: float = 0.2
) -> Optional[str]:
    global RR_IDX

    # If you only set GROQ_API_KEY (primary), that still works.
    if not (PRIMARY_KEY or SECONDARY_KEYS):
        return None

    # Tight, sane timeouts
    timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for kind, key in _select_key_order():
            # Skip if still in cooldown (guard; _select_key_order tries to avoid these)
            if KEY_BACKOFFS.get(key, 0) > time.time():
                continue

            try:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": messages,
                        "temperature": float(temperature),
                    },
                )
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError):
                # transient network → short backoff
                KEY_BACKOFFS[key] = time.time() + 15
                continue

            # Status handling
            if r.status_code in (401, 403):
                # permanently invalid key → never try again this process
                BAD_KEYS.add(key)
                continue

            if r.status_code == 429:
                # rate limited → respect Retry-After when possible
                ra = r.headers.get("Retry-After")
                try:
                    ra_s = int(ra) if ra else 20
                except Exception:
                    ra_s = 20
                KEY_BACKOFFS[key] = time.time() + max(10, min(60, ra_s))
                continue

            if r.status_code in (408, 409, 500, 502, 503, 504):
                # temporary → brief cooldown
                KEY_BACKOFFS[key] = time.time() + 10
                continue

            if r.status_code != 200:
                # other client errors → try next key
                continue

            # Parse success
            try:
                data = r.json()
                txt = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if txt:
                    # Advance RR when a secondary succeeds
                    if kind == "secondary" and SECONDARY_KEYS:
                        try:
                            idx = SECONDARY_KEYS.index(key)
                            RR_IDX = (idx + 1) % len(SECONDARY_KEYS)
                        except ValueError:
                            pass
                    return txt
            except Exception:
                # malformed → try next key
                continue

    # exhausted all candidates
    return None


# ---------- Endpoints ----------

@app.get("/")
def health():
    return {
        "status": "ok",
        "chunks": len(CORPUS),
        "groq_keys": len(GROQ_API_KEYS),
    }


@app.get("/health")
def health_alias():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    user_msg = (body.get("message") or "").strip()
    history = body.get("history") or ""

    # 1) Hardcoded intents (no LLM)
    if re.search(r"\b(whatsapp|whatapp|whats-app)\b", user_msg, flags=re.I):
        phone = CONTACT_LINKS["phone"]
        reply = f"<ul><li>WhatsApp: {phone}</li></ul>"
        return {
            "reply": reply,
            "updatedHistory": f"{history}\nUser: {user_msg}\nBot: {reply}",
        }

    if re.search(
        r"\b(contact|contacts|contact details|email|e-?mail|linkedin|github)\b",
        user_msg,
        flags=re.I,
    ):
        reply = format_contacts_ul()
        return {
            "reply": reply,
            "updatedHistory": f"{history}\nUser: {user_msg}\nBot: {reply}",
        }


    # 2) Chunk-based RAG for EVERYTHING ELSE
    ctx_chunks = retrieve_merged(user_msg, k=8)
    ctx = "\n\n".join(ctx_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM}\n\n{CONTACT_INSTR}\n\n"
                "Write a warm, concise, first-person answer. "
                "If the answer contains multiple points, FORMAT STRICTLY AS VALID HTML: "
                "return a single <ul> with 3–8 <li> items. "
                "No markdown, no extra text outside the list.\n\n"
                f"CONTEXT:\n{ctx}"
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    reply = await groq_chat_completion(messages, temperature=0.25)
    if not reply:
        return {
            "reply": "Sorry—my model call failed (all keys). Please try again later.",
            "updatedHistory": history,
        }

    # Force bullet list for clean frontend rendering
    reply = force_ul_list(inject_contacts(reply))
    return {
        "reply": reply,
        "updatedHistory": f"{history}\nUser: {user_msg}\nBot: {reply}",
    }



