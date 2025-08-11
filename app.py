# import os, json, re, unicodedata
# from pathlib import Path
# from typing import List, Optional
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import httpx

# # ---------- Config ----------
# load_dotenv()
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # fast & cheap


# # support multiple API keys via comma-separated env: GROQ_API_KEYS="key1,key2,key3"
# def _load_groq_keys() -> list[str]:
#     keys_env = os.getenv("GROQ_API_KEYS", "").strip()
#     if keys_env:
#         keys = [k.strip() for k in keys_env.split(",") if k.strip()]
#         if keys:
#             return keys
#     # fallback to single key for backward-compat
#     single = os.getenv("GROQ_API_KEY", "").strip()
#     return [single] if single else []


# GROQ_API_KEYS: list[str] = _load_groq_keys()

# DATA_DIR = Path("data")
# CORPUS_PATH = DATA_DIR / "corpus.json"
# XP_PATH = DATA_DIR / "experience.json"  # used for the experiences shortcut

# # Hard-coded contacts
# CONTACT_LINKS = {
#     "linkedin": "http://www.linkedin.com/in/pankit-shah13",
#     "github": "https://github.com/Shah-Pankit",
#     "phone": "+91 722- 901-3335",
#     "email": "pankitshah493@gmail.com",
# }

# # ---------- App ----------
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ---------- Load corpus ----------
# def load_corpus() -> list[str]:
#     if not CORPUS_PATH.exists():
#         raise RuntimeError(
#             "data/corpus.json not found. Put your corpus.json in the data/ folder."
#         )
#     return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


# CORPUS = load_corpus()


# # ---------- Retrieval (keyword-only: tiny & fast) ----------
# def retrieve(query: str, k: int = 6) -> List[str]:
#     if not query.strip():
#         return []
#     q = query.lower()
#     words = set(w for w in re.findall(r"[a-z0-9]+", q))
#     scored = []
#     for i, txt in enumerate(CORPUS):
#         t = txt.lower()
#         score = sum(t.count(w) for w in words)
#         scored.append((score, i))
#     scored.sort(reverse=True)
#     idxs = [i for s, i in scored[:k] if s > 0]
#     if not idxs:
#         idxs = list(range(min(k, len(CORPUS))))
#     return [CORPUS[i] for i in idxs]


# # ---------- Experience shortcut (dedupe + sort + UL) ----------
# MONTHS = {
#     "jan": 1,
#     "feb": 2,
#     "mar": 3,
#     "apr": 4,
#     "may": 5,
#     "jun": 6,
#     "jul": 7,
#     "aug": 8,
#     "sep": 9,
#     "sept": 9,
#     "oct": 10,
#     "nov": 11,
#     "dec": 12,
# }


# def _normalize_dash(s: str) -> str:
#     # normalize en/em dashes etc to "-"
#     return unicodedata.normalize("NFKC", s).replace("–", "-").replace("—", "-")


# def _parse_start(duration: str) -> tuple[int, int]:
#     """
#     'Jan 2025 - Mar 2025' -> (2025, 1)
#     Fallback (0,0) if not parseable.
#     """
#     if not duration:
#         return (0, 0)
#     d = _normalize_dash(duration.lower())
#     parts = d.split("-")
#     start = parts[0].strip() if parts else ""
#     toks = start.split()
#     if len(toks) < 2:
#         return (0, 0)
#     mon = MONTHS.get(toks[0][:3], 0)
#     yr = 0
#     try:
#         yr = int("".join(ch for ch in toks[1] if ch.isdigit()))
#     except:
#         pass
#     return (yr, mon)


# def _load_exp_entries():
#     if not XP_PATH.exists():
#         return []
#     try:
#         items = json.loads(XP_PATH.read_text(encoding="utf-8"))
#     except Exception:
#         return []
#     # dedupe by (title, company)
#     seen = set()
#     out = []
#     for e in items:
#         title = (e.get("title") or "").strip()
#         company = (e.get("company") or "").strip()
#         duration = (e.get("duration") or "").strip()
#         description = (e.get("description") or "").strip()
#         key = (title.lower(), company.lower())
#         if key in seen:
#             continue
#         seen.add(key)
#         out.append(
#             {
#                 "title": title,
#                 "company": company,
#                 "duration": duration,
#                 "description": description,
#                 "sort_key": _parse_start(duration),
#             }
#         )
#     # newest first
#     out.sort(key=lambda x: (x["sort_key"][0], x["sort_key"][1]), reverse=True)
#     return out


# def format_experiences_ul(exps) -> str:
#     if not exps:
#         return "<ul><li>No experience data found.</li></ul>"
#     items = []
#     for e in exps:
#         t = f'{e["title"]} at {e["company"]} ({e["duration"]}): {e["description"]}'
#         items.append(f"<li>{t}</li>")
#     return f"<ul>{''.join(items)}</ul>"


# # ---------- Response guards ----------
# LINK_TRAIL = re.compile(r"(https?://\S+?)([.,;:!?])(\s|$)")


# def clean_links(s: str) -> str:
#     return LINK_TRAIL.sub(r"\1\3", s)


# def inject_contacts(text: str) -> str:
#     for key, val in CONTACT_LINKS.items():
#         text = text.replace(f"{{{key}}}", val)
#     return clean_links(text)


# # plain-text contacts so your frontend auto-links
# def format_contacts_ul() -> str:
#     github = CONTACT_LINKS["github"]
#     linkedin = CONTACT_LINKS["linkedin"]
#     email = CONTACT_LINKS["email"]
#     phone = CONTACT_LINKS["phone"]
#     return (
#         "<ul>"
#         f"<li>GitHub: {github}</li>"
#         f"<li>LinkedIn: {linkedin}</li>"
#         f"<li>Email: {email}</li>"
#         f"<li>Phone: {phone}</li>"
#         "</ul>"
#     )


# SYSTEM = (
#     "You are Pankit Shah (AI/ML Engineer). Speak in first person as Pankit.\n"
#     "Use ONLY the provided CONTEXT to answer. If the info is not in CONTEXT, reply:\n"
#     '"I don’t have that in my portfolio context."\n'
#     "For long responses, use valid HTML <ul><li>…</li></ul> strictly."
#     "if asked for whatsapp number then the whatsapp number is +91 722-901-3335"
# )
# CONTACT_INSTR = (
#     "If asked for my GitHub, LinkedIn, or phone, respond with the exact values:\n"
#     f"GitHub: {CONTACT_LINKS['github']}\n"
#     f"LinkedIn: {CONTACT_LINKS['linkedin']}\n"
#     f"Phone: {CONTACT_LINKS['phone']}\n"
#     "Do NOT place punctuation immediately after any link."
# )


# # ---------- Groq call with fallback over multiple API keys ----------
# async def groq_chat_completion(
#     messages: list[dict], temperature: float = 0.2
# ) -> Optional[str]:
#     """
#     Try each key in GROQ_API_KEYS until one succeeds.
#     Returns reply text or None if all fail.
#     """
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
#                 # network error → try next key
#                 continue

#             # Rate limit or server/auth errors → try next key
#             if r.status_code in (401, 403, 408, 409, 429, 500, 502, 503, 504):
#                 continue

#             # Non-200 but not in the above list -> also try next
#             if r.status_code != 200:
#                 continue

#             data = r.json()
#             reply = (data.get("choices") or [{}])[0].get("message", {}).get("content")
#             if reply:
#                 return reply
#             # malformed/empty -> try next key

#     return None


# # ------- Matching helpers -------
# def _norm_text(s: str) -> str:
#     # remove non-alphanumeric and lowercase for fuzzy contains
#     return re.sub(r"[^a-z0-9]+", "", s.lower())


# def _filter_exps_for_query(exps, query: str):
#     nq = _norm_text(query)
#     picked = []
#     for e in exps:
#         if _norm_text(e["company"]) and _norm_text(e["company"]) in nq:
#             picked.append(e)
#             continue
#         if _norm_text(e["title"]) and _norm_text(e["title"]) in nq:
#             picked.append(e)
#             continue
#     return picked


# # ------- Output normalization (force bullets even if model returns paragraphs) -------
# _MD_BOLD = re.compile(r"\*\*(.*?)\*\*")
# _MD_ITAL = re.compile(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)")
# _MD_ULINE = re.compile(r"__(.*?)__")
# _MD_EMPH = re.compile(r"_(.*?)_")
# _MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")  # [text](url) -> "text (url)"


# def _strip_markdown(s: str) -> str:
#     s = _MD_LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", s)
#     s = _MD_BOLD.sub(r"\1", s)
#     s = _MD_ULINE.sub(r"\1", s)
#     s = _MD_ITAL.sub(r"\1", s)
#     s = _MD_EMPH.sub(r"\1", s)
#     return s


# def markdown_to_ul(text: str) -> str:
#     """
#     Convert any markdown/paragraph text to a clean <ul><li>…</li></ul>.
#     - Strips **bold**, *italics*, and [links](url) -> "links (url)".
#     - Detects bullets like -, *, •, or numbered "1.", "2)" etc.
#     - Falls back to splitting by sentences/newlines if needed.
#     """
#     if not text:
#         return "<ul><li></li></ul>"

#     # Already a proper list? Keep it.
#     if "<ul" in text and "<li>" in text:
#         return text

#     s = _strip_markdown(text).replace("\r", "\n").strip()

#     # First, try to extract explicit bullet/numbered lines
#     bullet_pat = re.compile(
#         r"(?:^|\n)\s*(?:[-*•]|[\d]{1,2}[.)])\s+(.*?)(?=(?:\n\s*(?:[-*•]|[\d]{1,2}[.)])\s+)|$)",
#         flags=re.S,
#     )
#     items = [itm.strip() for itm in bullet_pat.findall(s) if itm.strip()]

#     # If no explicit bullets, split by double newlines or sentence-ish breaks
#     if not items:
#         chunks = [c.strip() for c in re.split(r"\n{2,}", s) if c.strip()]
#         if len(chunks) > 1:
#             items = chunks

#     # Still nothing? Split long text into shorter bullets by ". " boundaries
#     if not items:
#         sentences = [t.strip() for t in re.split(r"(?<=[.!?])\s+", s) if t.strip()]
#         # group into 3–6 bullets max
#         if len(sentences) <= 6:
#             items = sentences
#         else:
#             # pack sentences into ~5 bullets
#             step = max(1, len(sentences) // 5)
#             items = [
#                 " ".join(sentences[i : i + step]).strip()
#                 for i in range(0, len(sentences), step)
#             ]
#             items = [i for i in items if i][:6]

#     if not items:
#         items = [s]

#     return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# def force_ul_list(text: str) -> str:
#     # Simple wrapper using markdown_to_ul to guarantee UL output
#     return markdown_to_ul(text)


# # ---------- Endpoints ----------
# @app.get("/")
# def health():
#     return {
#         "status": "ok",
#         "chunks": len(CORPUS),
#         "groq_keys": len(GROQ_API_KEYS),  # handy to verify Render env is set
#     }


# @app.post("/api/chat")
# async def chat(req: Request):
#     body = await req.json()
#     user_msg = (body.get("message") or "").strip()
#     history = body.get("history") or ""

#     # Load experiences once for intent detection & filtering
#     exps = _load_exp_entries()
#     filtered = _filter_exps_for_query(exps, user_msg)

#     # EXPERIENCE INTENT if:
#     #   - contains experience keywords OR
#     #   - mentions a known company/title (filtered != [])
#     if filtered or re.search(
#         r"\b(experience|experiences|internship|internships)\b", user_msg, flags=re.I
#     ):
#         # Company-specific + request to summarize/explain → LLM then FORCE UL
#         if filtered and re.search(
#             r"\b(summarize|summary|tell me about|how('?s| was)|describe|explain|in detail|details)\b",
#             user_msg,
#             flags=re.I,
#         ):
#             ctx = "\n".join(
#                 f"{e['title']} at {e['company']} ({e['duration']}): {e['description']}"
#                 for e in filtered
#             )
#             messages = [
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are Pankit Shah. Based ONLY on the provided experience text, "
#                         "write a warm, positive, first-person summary that mentions what I built, "
#                         "the key technologies (e.g., ASR, NLP, RAG, Qdrant, Docker, AWS), and what I learned. "
#                         "Keep it concise (3–6 short bullet points). "
#                         "FORMAT STRICTLY AS VALID HTML: return a single <ul> with 3–6 <li> items. Maintina the <ul><li> structure please."
#                         "No paragraphs, no markdown, no extra text before or after the <ul>."
#                     ),
#                 },
#                 {"role": "user", "content": ctx},
#             ]
#             reply = await groq_chat_completion(messages, temperature=0.3)
#             if not reply:
#                 reply = format_experiences_ul(filtered)
#             else:
#                 # HARD GUARANTEE: convert any paragraphs/markdown to clean UL bullets
#                 reply = force_ul_list(reply)
#         else:
#             # Default deterministic list (full or filtered)
#             reply = format_experiences_ul(filtered if filtered else exps)

#         updated_history = f"{history}\nUser: {user_msg}\nBot: {reply}"
#         return {"reply": reply, "updatedHistory": updated_history}

#     # WhatsApp intent → same as phone (deterministic)
#     if re.search(r"\b(whatsapp|whatapp|whats-app)\b", user_msg, flags=re.I):
#         phone = CONTACT_LINKS["phone"]
#         reply = f"<ul><li>WhatsApp: {phone}</li></ul>"
#         updated_history = f"{history}\nUser: {user_msg}\nBot: {reply}"
#         return {"reply": reply, "updatedHistory": updated_history}

#     # Contact details → deterministic
#     if re.search(
#         r"\b(contact|contacts|contact details|email|e-?mail|phone|mobile|linkedin|github)\b",
#         user_msg,
#         flags=re.I,
#     ):
#         reply = format_contacts_ul()
#         updated_history = f"{history}\nUser: {user_msg}\nBot: {reply}"
#         return {"reply": reply, "updatedHistory": updated_history}

#     # Fallback: RAG via keyword retrieval + LLM
#     ctx = "\n\n".join(retrieve(user_msg, k=6))
#     messages = [
#         {
#             "role": "system",
#             "content": f"{SYSTEM}\n\n{CONTACT_INSTR}\n\nCONTEXT:\n{ctx}",
#         },
#         {"role": "user", "content": user_msg},
#     ]

#     reply = await groq_chat_completion(messages, temperature=0.2)
#     if not reply:
#         return {
#             "reply": "Sorry—my model call failed (all keys). Please try again later.",
#             "updatedHistory": history,
#         }

#     reply = inject_contacts(reply)
#     updated_history = f"{history}\nUser: {user_msg}\nBot: {reply}"
#     return {"reply": reply, "updatedHistory": updated_history}


import os, json, re, unicodedata
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx

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
    " joined Techlusion as an AI/ML (GenAI) Intern on July 21, 2025. Please add this to my work experience and use it to answer questions about my current role, latest company, availability, or what I’m currently doing. When asked about availability, respond that I am currently engaged in my internship at Techlusion but open to relevant opportunities or collaborations."
)
CONTACT_INSTR = (
    "If asked for my GitHub, LinkedIn, or phone, respond with the exact values:\n"
    f"GitHub: {CONTACT_LINKS['github']}\n"
    f"LinkedIn: {CONTACT_LINKS['linkedin']}\n"
    f"Phone: {CONTACT_LINKS['phone']}\n"
    "Do NOT place punctuation immediately after any link."
)


# ---------- Groq call with fallback over multiple API keys ----------
async def groq_chat_completion(
    messages: list[dict], temperature: float = 0.2
) -> Optional[str]:
    if not GROQ_API_KEYS:
        return None
    async with httpx.AsyncClient(timeout=30.0) as client:
        for key in GROQ_API_KEYS:
            try:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": messages,
                        "temperature": temperature,
                    },
                )
            except httpx.HTTPError:
                continue
            if r.status_code in (401, 403, 408, 409, 429, 500, 502, 503, 504):
                continue
            if r.status_code != 200:
                continue
            data = r.json()
            reply = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if reply:
                return reply
    return None


# ---------- Endpoints ----------
@app.get("/")
def health():
    return {
        "status": "ok",
        "chunks": len(CORPUS),
        "groq_keys": len(GROQ_API_KEYS),
    }


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

    # Show all experiences (deterministic)
    if re.search(
        r"\b(experience|experiences|internship|internships)\b", user_msg, flags=re.I
    ) and not re.search(
        r"\b(summarize|summary|insight|insights|explain|describe|detail|details)\b",
        user_msg,
        flags=re.I,
    ):
        exps = _load_exp_entries()
        reply = format_experiences_ul(exps)
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
