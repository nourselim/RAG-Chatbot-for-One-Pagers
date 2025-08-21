# #!/usr/bin/env python3
# """
# Build a FAISS vector store (LangChain) from all_employees.json.
# Assumes the JSON is already cleaned/sanitized.

# Usage:
#   python build_faiss_from_all_employees.py \
#       --in data/json_output/all_employees.json \
#       --out frontend/faiss_store \
#       --model text-embedding-3-small \
#       --emit-jsonl frontend/output/chunks_with_vecs.jsonl \
#       --query "Who has AWS certification and Kubernetes experience?" --k 5

# Outputs:
#   - <out>/index.faiss  +  <out>/index.pkl       (LangChain FAISS store)
#   - <emit-jsonl> (optional)                     (one JSON per stored chunk with precomputed embeddings)
# """

# from __future__ import annotations
# import os
# import json
# import argparse
# from pathlib import Path
# from typing import List, Dict, Any

# from dotenv import load_dotenv, find_dotenv
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document


# # ------------------------- helpers -------------------------

# def _as_list(x):
#     if not x:
#         return []
#     return x if isinstance(x, list) else [x]

# def load_all_employees(path: Path) -> List[Dict[str, Any]]:
#     """Load a list of employee dicts from all_employees.json (handles common shapes)."""
#     data = json.loads(path.read_text(encoding="utf-8"))
#     if isinstance(data, list):
#         return data
#     if isinstance(data, dict) and "employees" in data and isinstance(data["employees"], list):
#         return data["employees"]
#     if isinstance(data, dict):
#         items = list(data.values())
#         if items and isinstance(items[0], dict):
#             return items
#     raise ValueError("Could not interpret all_employees.json; expected a list or {employees: [...]}")

# def record_to_documents(rec: Dict[str, Any]) -> List[Document]:
#     """
#     Turn one employee record into multiple Documents:
#       - Summary (with Name/Title)
#       - One Document per Experience (role/company/duration + bulleted achievements)
#       - Skills / Certifications / Clients (each in its own block)
#     Assumes text is already clean.
#     """
#     docs: List[Document] = []
#     name: str | None = rec.get("name")
#     title: str | None = rec.get("title") or rec.get("deloitte_title")
#     summary: str | None = rec.get("summary")

#     # Header on every doc for grounding
#     header_lines = []
#     if name:  header_lines.append(f"- Name: {name}")
#     if title: header_lines.append(f"- Title: {title}")
#     header = "\n".join(header_lines)

#     # Summary doc
#     if summary and summary.strip():
#         content = (f"{header}\nSummary:\n{summary}".strip() if header else f"Summary:\n{summary}".strip())
#         docs.append(Document(
#             page_content=content,
#             metadata={"type": "summary", "name": name, "title": title}
#         ))

#     # Experience docs
#     for xp in _as_list(rec.get("experience")):
#         if not isinstance(xp, dict):
#             continue
#         role = xp.get("role")
#         company = xp.get("company")
#         duration = xp.get("duration")
#         head_parts = [p for p in [role, company, duration] if p]
#         head = " - ".join(head_parts)
#         ach_lines = [f"- {a}" for a in _as_list(xp.get("achievements")) if a]
#         content = "\n".join([s for s in [header, head, *ach_lines] if s]).strip()
#         if content:
#             docs.append(Document(
#                 page_content=content,
#                 metadata={
#                     "type": "experience",
#                     "name": name,
#                     "title": title,
#                     "role": role,
#                     "company": company,
#                 }
#             ))

#     # Skills (flatten buckets)
#     skills: List[str] = []
#     ks = rec.get("key_skills") or {}
#     for bucket in ("business", "technology", "industry"):
#         skills += [s for s in _as_list(ks.get(bucket)) if s]
#     if skills:
#         sk_block = "Skills:\n" + "\n".join(f"- {s}" for s in skills)
#         content = "\n".join([header, sk_block]).strip() if header else sk_block
#         docs.append(Document(
#             page_content=content,
#             metadata={"type": "skills", "name": name, "title": title}
#         ))

#     # Certifications
#     certs = [c for c in _as_list(rec.get("certifications")) if c]
#     if certs:
#         cert_block = "Certifications:\n" + "\n".join(f"- {c}" for c in certs)
#         content = "\n".join([header, cert_block]).strip() if header else cert_block
#         docs.append(Document(
#             page_content=content,
#             metadata={"type": "certifications", "name": name, "title": title}
#         ))

#     # Clients
#     clients = [c for c in _as_list(rec.get("clients")) if c]
#     if clients:
#         client_block = "Clients:\n" + "\n".join(f"- {c}" for c in clients)
#         content = "\n".join([header, client_block]).strip() if header else client_block
#         docs.append(Document(
#             page_content=content,
#             metadata={"type": "clients", "name": name, "title": title}
#         ))

#     return docs


# # --------------------- build / save FAISS ------------------

# def build_faiss_from_all_employees(in_json: Path, out_dir: Path, embed_model: str,
#                                    emit_jsonl: Path | None = None) -> FAISS:
#     """Load employees, create Documents, embed with OpenAI, persist FAISS store."""
#     employees = load_all_employees(in_json)

#     all_docs: List[Document] = []
#     for rec in employees:
#         all_docs.extend(record_to_documents(rec))

#     if not all_docs:
#         raise RuntimeError("No Documents were generated from all_employees.json")

#     out_dir.mkdir(parents=True, exist_ok=True)

#     embeddings = OpenAIEmbeddings(model=embed_model)
#     vs = FAISS.from_documents(all_docs, embedding=embeddings)
#     vs.save_local(out_dir.as_posix())

#     # Optional: emit a JSONL with precomputed vectors for reuse/debugging
#     if emit_jsonl:
#         texts = [d.page_content for d in all_docs]
#         vecs = embeddings.embed_documents(texts)  # extra embedding pass (fine for small corpora)
#         emit_jsonl.parent.mkdir(parents=True, exist_ok=True)
#         with emit_jsonl.open("w", encoding="utf-8") as f:
#             for doc, vec in zip(all_docs, vecs):
#                 row = {
#                     "id": doc.metadata.get("name"),         # change to a stable id if you have one
#                     "name": doc.metadata.get("name"),
#                     "chunk_type": doc.metadata.get("type"),
#                     "text": doc.page_content,
#                     "vector": vec,
#                 }
#                 f.write(json.dumps(row, ensure_ascii=False) + "\n")

#     return vs


# # def quick_demo(vs: FAISS, query: str, k: int = 5):
# #     retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 25})
# #     results = retriever.get_relevant_documents(query)
# #     print("\nTop results:")
# #     for i, d in enumerate(results, 1):
# #         nm = d.metadata.get("name"); tp = d.metadata.get("type")
# #         prev = d.page_content[:140].replace("\n", " ")
# #         print(f"{i}. {nm or 'Unknown'} | {tp} | {prev}...")


# # ----------------------------- CLI ------------------------

# def main():
#     parser = argparse.ArgumentParser(description="Build LangChain+FAISS store from all_employees.json (clean data assumed).")
#     parser.add_argument("--in", dest="in_json", required=True, help="Path to all_employees.json")
#     parser.add_argument("--out", dest="out_dir", default="frontend/faiss_store", help="Output dir for FAISS store")
#     parser.add_argument("--model", dest="embed_model", default="text-embedding-3-small",
#                         help="OpenAI embedding model (e.g., text-embedding-3-small or -large)")
#     parser.add_argument("--emit-jsonl", dest="emit_jsonl", default=None, help="Optional: write chunks_with_vecs.jsonl")
#     parser.add_argument("--query", dest="query", default=None, help="Optional: test query to print results")
#     parser.add_argument("--k", dest="k", type=int, default=5, help="Top-k for test query")
#     args = parser.parse_args()

#     load_dotenv(find_dotenv(), override=False)
#     if not os.getenv("OPENAI_API_KEY"):
#         raise RuntimeError("OPENAI_API_KEY not set. Put it in your .env at repo root.")

#     in_path = Path(args.in_json)
#     out_dir = Path(args.out_dir)
#     emit_path = Path(args.emit_jsonl) if args.emit_jsonl else None

#     print(f"Embedding from: {in_path}")
#     print(f"Saving FAISS store to: {out_dir}")
#     if emit_path:
#         print(f"Also emitting JSONL with vectors to: {emit_path}")

#     vs = build_faiss_from_all_employees(in_path, out_dir, args.embed_model, emit_jsonl=emit_path)
#     print("✅ FAISS store built and persisted.")

#     if args.query:
#         quick_demo(vs, args.query, k=args.k)

# if __name__ == "__main__":
#     main()
