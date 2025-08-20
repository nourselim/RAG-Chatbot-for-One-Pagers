import os, argparse, re
from pathlib import Path
import numpy as np, ujson
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

load_dotenv() 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
EMB_NPY = OUT_DIR / "embeddings.npy"
META_JSONL = OUT_DIR / "chunks_meta.jsonl"

MODEL = "text-embedding-3-large"
BATCH = 100
EMB_DIM = 3072

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def load_chunks():
    rows = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(ujson.loads(line))
    return rows

def embed_texts(texts):
    vecs = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding", unit="batch"):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=MODEL, input=batch)
        for d in resp.data:
            vecs.append(l2_normalize(np.array(d.embedding, dtype=np.float32)))
    return np.vstack(vecs) if vecs else np.zeros((0, EMB_DIM), dtype=np.float32)

def embed_query(q: str) -> np.ndarray:
    v = np.array(client.embeddings.create(model=MODEL, input=[q]).data[0].embedding, dtype=np.float32)
    return l2_normalize(v).reshape(1, -1)

def cosine_batch(qv: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (qv @ X.T).flatten()  

def pct_from_cos(c: float) -> float:
    return (c + 1.0) / 2.0 * 100.0  

def cmd_embed():
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"Missing {CHUNKS_FILE}. Run convert_json_to_chunks.py first.")
    rows = load_chunks()
    texts = [r["text"] for r in rows]
    X = embed_texts(texts)
    np.save(EMB_NPY, X)
    with open(META_JSONL, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(ujson.dumps({
                "uuid": r["uuid"],
                "profile_id": r.get("profile_id"),
                "employee_name": r.get("employee_name"),
                "title": r.get("title"),
                "email": r.get("email"),
                "chunk_type": r.get("chunk_type"),
                "text": r.get("text")
            }, ensure_ascii=False) + "\n")
    print(f"Saved embeddings -> {EMB_NPY}")
    print(f"Saved metadata   -> {META_JSONL}")

def cmd_search(query: str, top: int, pool: int):
    if not EMB_NPY.exists() or not META_JSONL.exists():
        raise SystemExit("Embeddings or metadata missing. Run: python embed_only.py embed")

    X = np.load(EMB_NPY)  
    metas = [ujson.loads(l) for l in open(META_JSONL, "r", encoding="utf-8")]
    if X.shape[0] != len(metas):
        raise SystemExit("Mismatch between embeddings and metadata counts.")

    qv = embed_query(query)
    sims = cosine_batch(qv, X) 

    q = query.lower()
    wants_cert = any(w in q for w in ["certified", "certification", "certificate"])
    exact_phrases = [
        "microsoft certified: azure administrator",
        "azure administrator",
        "aws certified solutions architect",
        "sap s/4hana",
    ]
    q_terms = {"microsoft", "azure", "administrator", "certified", "certification"}

    boosted_per_emp = {}
    per_emp_chunks = {}

    for sim, m in zip(sims, metas):
        txt = (m.get("text") or "").lower()
        ctype = (m.get("chunk_type") or "").lower()

        score = float(sim)

        phrase_bonus = 0.0
        for ph in exact_phrases:
            if ph in q and ph in txt:
                phrase_bonus += 0.10  
                break

        cert_bonus = 0.0
        if wants_cert and ctype == "certifications":
            cert_bonus += 0.05

        cover_bonus = 0.0
        if q_terms & set(q.split()):  
            hits = sum(1 for t in q_terms if t in q and t in txt)
            if hits:
                cover_bonus += min(0.02 * hits, 0.08)  

        boosted = score + phrase_bonus + cert_bonus + cover_bonus

        emp = m.get("profile_id") or f"unknown-{m['uuid'][-8:]}"
        per_emp_chunks.setdefault(emp, []).append((boosted, score, m))

    for emp, arr in per_emp_chunks.items():
        arr.sort(key=lambda t: t[0], reverse=True) 
        top2 = arr[:2]
        avg_boosted = sum(b for b, _, _ in top2) / len(top2)
        best_boosted, best_cos, best_meta = top2[0]
        boosted_per_emp[emp] = (avg_boosted, best_cos, best_meta)

    ranked = sorted(boosted_per_emp.items(), key=lambda kv: kv[1][0], reverse=True)[:top]
    if not ranked:
        print("No matches."); return

    best_final = ranked[0][1][0]
    print(f"\nQuery: {query}\n")
    for emp, (final_score, best_cos, m) in ranked[:pool]:
        rel = (final_score / best_final) if best_final > 0 else 0.0
        pct = (best_cos + 1.0) / 2.0 * 100.0  
        label = "High" if best_cos >= 0.60 else ("Medium" if best_cos >= 0.40 else "Low")
        name = m.get("employee_name") or m.get("email") or emp
        role = f" — {m.get('title')}" if m.get("title") else ""
        mail = f" — {m.get('email')}" if m.get("email") else ""
        snippet = re.sub(r"\s+", " ", m.get("text",""))[:180] + ("…" if len(m.get("text",""))>180 else "")
        print(f"final {final_score:.2f} | cos {best_cos:.2f} | {pct:.1f}% | rel {rel*100:.0f}% | {label} | {name}{role}{mail} | via {m.get('chunk_type')}\n{snippet}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed")
    s = sub.add_parser("search")
    s.add_argument("query", nargs="+")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--pool", type=int, default=5)
    args = ap.parse_args()
    if args.cmd == "embed":
        cmd_embed()
    else:
        cmd_search(" ".join(args.query), top=args.top, pool=args.pool)
