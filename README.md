
Pankit’s Portfolio Chatbot — Backend (FastAPI + Groq)

Ultra-lightweight RAG-ish backend for a personal portfolio chatbot.
Loads a tiny local corpus, does fast keyword retrieval (no heavy vector DB),
and calls Groq for responses. Returns HTML lists (<ul><li>…</li></ul>) so the
frontend can render clean bullets. Designed for Render Free.

------------------------------------------------------------
FEATURES
------------------------------------------------------------
- Tiny & fast: keyword retrieval over small corpus (no embeddings required).
- Accurate & scoped: system prompt restricts answers to portfolio context.
- Deterministic branches:
  * Contacts (GitHub/LinkedIn/Email/Phone/WhatsApp) → static HTML list.
  * “What are your experiences?” → deterministic list from experience.json.
- Smart fallback: For all other queries, retrieves top context chunks
  (from corpus.json + experience.json) and asks Groq to answer concisely.
- HTML guarantee: Backend post-processes any LLM output to a single <ul>.
- Multi-key failover: Rotate through multiple Groq API keys automatically
  when rate-limited.

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------
app.py
requirements.txt
render.yaml                # optional (Render IaC)
data/
    corpus.json            # small chunked knowledge base
    experience.json        # list of experience entries
    projects.json          # optional
    resume.txt             # optional
    site_chunks.txt        # optional

------------------------------------------------------------
QUICK START (LOCAL)
------------------------------------------------------------
1. Create Python environment:
   python -m venv .venv
   source .venv/bin/activate   (Windows: .venv\Scripts\activate)

2. Install dependencies:
   pip install -r requirements.txt

3. Create .env file:
   GROQ_API_KEY=gsk_...
   or
   GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3

   Optional:
   GROQ_MODEL=llama-3.1-8b-instant

4. Run:
   uvicorn app:app --reload

5. Health check:
   GET http://127.0.0.1:8000/

------------------------------------------------------------
API ENDPOINTS
------------------------------------------------------------
GET /
- Health check: returns status, chunk count, loaded keys.

POST /api/chat
- Body JSON:
  { "message": "What did you do at Lintel Technologies?", "history": "" }
- Returns:
  { "reply": "<ul><li>…</li></ul>", "updatedHistory": "..." }

------------------------------------------------------------
DEPLOYMENT ON RENDER
------------------------------------------------------------
Build Command:
  pip install -r requirements.txt

Start Command:
  uvicorn app:app --host 0.0.0.0 --port $PORT

Environment Variables:
  GROQ_API_KEY or GROQ_API_KEYS
  GROQ_MODEL (optional)

Leave "Root Directory" empty unless app.py is in a subfolder.

------------------------------------------------------------
RESPONSE LOGIC
------------------------------------------------------------
1. Hardcoded routes:
   - Contacts/WhatsApp
   - "What are your experiences?" → experience.json

2. General queries:
   - Merge corpus.json + experience.json
   - Keyword-score to pick top chunks
   - Send chunks to Groq with strict HTML list instruction
   - Post-process to ensure valid HTML list

3. Failover:
   - Rotate keys if rate-limited

------------------------------------------------------------
TROUBLESHOOTING
------------------------------------------------------------
- ModuleNotFoundError: No module named 'api':
  Use correct start command:
    uvicorn app:app --host 0.0.0.0 --port $PORT

- 500 on startup:
  Ensure corpus.json and experience.json exist and are valid.

- 401/403:
  Bad or missing Groq keys.

- 429:
  Use GROQ_API_KEYS with multiple keys.

- CORS:
  CORS is open by default. Restrict in app.py if needed.

------------------------------------------------------------
SECURITY NOTES
------------------------------------------------------------
- Keep API keys out of repo.
- Serve only sanitized HTML.

------------------------------------------------------------
LICENSE
MIT 
------------------------------------------------------------
url : https://my-portfolio-website-bot-backend.onrender.com
------------------------------------------------------------
Personal use for portfolio chatbot.
