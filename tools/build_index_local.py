# tools/build_index_local.py
# Dev-only script to build the small corpus and (optionally) embeddings.
# Requires: pip install sentence-transformers numpy
import json
from pathlib import Path
import numpy as np

# 1) Build corpus from your files (paste your data here or read from your repo paths)
DATA = Path("data")
DATA.mkdir(exist_ok=True)

# --- Load your existing files (adapt paths if needed) ---
resume = Path("./data/resume.txt").read_text(encoding="utf-8")
projects = json.loads(Path("./data/projects.json").read_text(encoding="utf-8"))
experiences = json.loads(Path("./data/experience.json").read_text(encoding="utf-8"))

corpus = [resume.strip()]
for p in projects:
    title = p.get("title") or p.get("name")
    desc = p.get("description") or p.get("desc")
    link = p.get("github") or (p.get("links") or {}).get("code", "")
    corpus.append(f"Project: {title}\nDescription: {desc}\nGitHub Link: {link}")

for e in experiences:
    corpus.append(
        f"{e.get('title')} at {e.get('company')} ({e.get('duration')}): {e.get('description')}"
    )

# Save corpus
(Path("data/corpus.json")).write_text(
    json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
)

# 2) OPTIONAL: create embeddings (better retrieval). Requires sentence-transformers locally.
try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(corpus, normalize_embeddings=True)
    emb = emb.astype("float32")
    np.save("data/embeddings.npy", emb)
    Path("data/meta.json").write_text(
        json.dumps({"dim": emb.shape[1], "count": emb.shape[0]}), encoding="utf-8"
    )
    print("Wrote data/embeddings.npy and data/meta.json")
except Exception as e:
    print(
        "Skipped embeddings build (install sentence-transformers to enable). Only corpus.json was created."
    )
