import json, hashlib, uuid, re
from pathlib import Path

BASE = Path(__file__).parent
DATA_IN = BASE / "data" / "all_employees.json"
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_OUT = DATA_DIR / "chunks.jsonl"

def normalize_list(x):
    if x is None: return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    return [str(x).strip()]

def safe_join(items, sep=", "): return sep.join([s for s in items if s])
def hash_str(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

def norm_name(s):
    if not s: return None
    return re.sub(r"\s+", " ", s).strip()

def canonical_profile_id(name, email):
    if email: return email.strip().lower()
    if name:  return norm_name(name).lower()
    return "unknown-" + uuid.uuid4().hex[:8]

def make_chunk(p, chunk_type, text):
    t = (text or "").strip()
    if not t: return None
    base_id = f"{p.get('profile_id','unknown')}::{chunk_type}::{hash_str(t)[:12]}"
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, base_id)),
        "profile_id": p.get("profile_id"),
        "employee_name": p.get("name"),
        "title": p.get("title"),
        "email": p.get("email"),
        "chunk_type": chunk_type,
        "text": t
    }

def profile_to_chunks(p):
    p = dict(p)
    p["name"] = norm_name(p.get("name"))
    email = (p.get("contact") or {}).get("email")
    p["email"] = email.strip().lower() if isinstance(email, str) else None
    p["profile_id"] = canonical_profile_id(p.get("name"), p.get("email"))

    chunks = []
    if p.get("summary"): chunks.append(make_chunk(p, "summary", p["summary"]))
    ks = p.get("key_skills") or {}
    skills = []
    for _, items in ks.items(): skills += normalize_list(items)
    if skills: chunks.append(make_chunk(p, "skills", f"Skills: {safe_join(skills)}"))
    certs = normalize_list(p.get("certifications"))
    if certs: chunks.append(make_chunk(p, "certifications", f"Certifications: {safe_join(certs)}"))

    for exp in (p.get("experience") or []):
        role = exp.get("role") or ""; company = exp.get("company") or ""
        duration = exp.get("duration") or ""; ach = normalize_list(exp.get("achievements"))
        txt = "\n".join([
            f"Role: {role}",
            f"Company: {company}",
            f"Duration: {duration}",
            f"Achievements: {safe_join(ach, '; ')}"
        ])
        chunks.append(make_chunk(p, "experience", txt))

    for key, label in [("languages","Languages"),("education","Education"),("clients","Clients")]:
        vals = normalize_list(p.get(key))
        if vals: chunks.append(make_chunk(p, key, f"{label}: {safe_join(vals)}"))
    return [c for c in chunks if c]

def main():
    with open(DATA_IN, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    tot = 0
    with open(CHUNKS_OUT, "w", encoding="utf-8") as out:
        for p in profiles:
            for ch in profile_to_chunks(p):
                out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                tot += 1
    print(f"Wrote {tot} chunks to {CHUNKS_OUT}")

if __name__ == "__main__":
    main()
