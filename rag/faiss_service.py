import numpy as np
import json as ujson
from pathlib import Path
import faiss
from embed_only import embed_query, cosine_batch, pct_from_cos

class FaissCandidateSearch:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_path = index_dir / "faiss_index.bin"
        self.meta_path = index_dir / "chunks_meta.jsonl"
        self.index = None
        self.metas = []

    def build_index(self, embeddings_npy_path: Path, chunks_meta_path: Path):
        """
        Loads embeddings and metadata from the files created by embed_only.py
        and builds a FAISS index, saving it to disk.
        """
        print("Loading embeddings and metadata...")
        X = np.load(embeddings_npy_path)
        self.metas = [ujson.loads(line) for line in open(chunks_meta_path, 'r', encoding='utf-8')]

        if X.shape[0] != len(self.metas):
            raise ValueError(f"Mismatch: {X.shape[0]} embeddings but {len(self.metas)} metadata entries.")

        print(f"Building FAISS index for {X.shape[0]} vectors...")
        # Create a flat IP (Inner Product) index. Since our vectors are normalized, IP = Cosine Similarity.
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X.astype(np.float32)) # FAISS requires float32

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        # Save the metadata for this index
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            for meta in self.metas:
                f.write(ujson.dumps(meta) + '\n')
        print(f"Index built and saved to {self.index_path}")

    def load_index(self):
        """Loads the FAISS index and corresponding metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found at {self.index_path}. Run 'build_index' first.")
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))
        self.metas = [ujson.loads(line) for line in open(self.meta_path, 'r', encoding='utf-8')]
        print(f"Index loaded. Ready to search {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 50, pool_size: int = 5):
        """
        Searches the index for the best matching candidates.
        Applies the same boosting and per-employee pooling logic as embed_only.py.

        Args:
            query: The project description or search query.
            top_k: How many raw chunks to retrieve from the vector store.
            pool_size: How many final candidate profiles to return.

        Returns:
            A list of ranked candidate profiles with their scores and best matching chunk.
        """
        if self.index is None or not self.metas:
            raise ValueError("Index not loaded. Call load_index() first.")

        # 1. Get query vector
        qv = embed_query(query)
        q = query.lower()

        # 2. Perform initial FAISS search (returns top_k matches)
        distances, indices = self.index.search(qv.astype(np.float32), top_k)
        # For IndexFlatIP, 'distances' are actually cosine similarities
        sims = distances[0] 
        retrieved_metas = [self.metas[i] for i in indices[0]]

        # 3. Define boosting rules (copied from your original logic)
        wants_cert = any(w in q for w in ["certified", "certification", "certificate"])
        exact_phrases = [
            "microsoft certified: azure administrator",
            "azure administrator",
            "aws certified solutions architect",
            "sap s/4hana",
        ]
        q_terms = set(q.split())

        # 4. Apply boosting and pool by employee
        boosted_per_emp = {}
        per_emp_chunks = {}

        for sim, m in zip(sims, retrieved_metas):
            txt = (m.get("text") or "").lower()
            ctype = (m.get("chunk_type") or "").lower()

            score = float(sim)

            # Apply bonuses
            phrase_bonus = 0.0
            for ph in exact_phrases:
                if ph in q and ph in txt:
                    phrase_bonus += 0.10
                    break

            cert_bonus = 0.0
            if wants_cert and ctype == "certifications":
                cert_bonus += 0.05

            cover_bonus = 0.0
            if q_terms:
                hits = sum(1 for t in q_terms if t in txt)
                if hits:
                    cover_bonus += min(0.02 * hits, 0.08)

            boosted_score = score + phrase_bonus + cert_bonus + cover_bonus

            emp_id = m.get("profile_id") or f"unknown-{m['uuid'][-8:]}"
            per_emp_chunks.setdefault(emp_id, []).append((boosted_score, score, m))

        # 5. For each employee, take their top 2 chunks and average the boosted score
        for emp_id, chunks in per_emp_chunks.items():
            chunks.sort(key=lambda x: x[0], reverse=True)
            top_2_chunks = chunks[:2]
            avg_boosted = sum(s[0] for s in top_2_chunks) / len(top_2_chunks)
            best_original_cos = top_2_chunks[0][1] # The best original cosine sim
            best_chunk_meta = top_2_chunks[0][2]   # The meta for the best chunk
            boosted_per_emp[emp_id] = (avg_boosted, best_original_cos, best_chunk_meta)

        # 6. DEDUPLICATION FIX - Ensure only one entry per candidate
        unique_candidates = {}
        for emp_id, score_data in boosted_per_emp.items():
            # Use a consistent key - email is better than profile_id if available
            key = score_data[2].get('email') or emp_id  # Prefer email, fallback to profile_id
            if key not in unique_candidates or score_data[0] > unique_candidates[key][0]:
                unique_candidates[key] = score_data

        # 7. Rank employees by their averaged boosted score
        ranked_candidates = []
        for key, score_data in unique_candidates.items():
            ranked_candidates.append((key, score_data))
        ranked_candidates.sort(key=lambda x: x[1][0], reverse=True)

        # 8. Return the top pool_size candidates
        return ranked_candidates[:pool_size]

    def search_filtered(
        self,
        query: str,
        target_names: list[str] | None = None,
        chunk_types: list[str] | None = None,
        top_k: int = 50,
        pool_size: int = 5,
    ):
        """
        History-aware filtered search. If target_names and/or chunk_types are provided,
        restrict the results to those employees and/or sections while retaining the
        boosting and per-employee pooling.

        Args:
            query: natural language query
            target_names: list of employee names to include (case-insensitive)
            chunk_types: list of chunk types to include, e.g., ["skills", "summary"]
            top_k: number of raw chunks to fetch from FAISS
            pool_size: number of final employees to return
        """
        if self.index is None or not self.metas:
            raise ValueError("Index not loaded. Call load_index() first.")

        name_set = {n.strip().lower() for n in (target_names or []) if n and n.strip()}
        type_set = {t.strip().lower() for t in (chunk_types or []) if t and t.strip()}

        qv = embed_query(query)
        q = query.lower()

        distances, indices = self.index.search(qv.astype(np.float32), top_k)
        sims = distances[0]
        retrieved_metas = [self.metas[i] for i in indices[0]]

        wants_cert = any(w in q for w in ["certified", "certification", "certificate"])
        exact_phrases = [
            "microsoft certified: azure administrator",
            "azure administrator",
            "aws certified solutions architect",
            "sap s/4hana",
        ]
        q_terms = set(q.split())

        boosted_per_emp = {}
        per_emp_chunks = {}

        for sim, m in zip(sims, retrieved_metas):
            txt = (m.get("text") or "").lower()
            ctype = (m.get("chunk_type") or "").lower()
            nm = (m.get("employee_name") or "").strip().lower()

            # Apply name/type filters if provided
            if name_set and nm not in name_set:
                continue
            if type_set and ctype not in type_set:
                continue

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
            if q_terms:
                hits = sum(1 for t in q_terms if t in txt)
                if hits:
                    cover_bonus += min(0.02 * hits, 0.08)

            boosted_score = score + phrase_bonus + cert_bonus + cover_bonus

            emp_id = m.get("profile_id") or f"unknown-{m['uuid'][-8:]}"
            per_emp_chunks.setdefault(emp_id, []).append((boosted_score, score, m))

        # If nothing passed filters, fall back to normal search
        if not per_emp_chunks:
            return self.search(query, top_k=top_k, pool_size=pool_size)

        for emp_id, chunks in per_emp_chunks.items():
            chunks.sort(key=lambda x: x[0], reverse=True)
            top_2_chunks = chunks[:2]
            avg_boosted = sum(s[0] for s in top_2_chunks) / len(top_2_chunks)
            best_original_cos = top_2_chunks[0][1]
            best_chunk_meta = top_2_chunks[0][2]
            boosted_per_emp[emp_id] = (avg_boosted, best_original_cos, best_chunk_meta)

        unique_candidates = {}
        for emp_id, score_data in boosted_per_emp.items():
            key = score_data[2].get('email') or emp_id
            if key not in unique_candidates or score_data[0] > unique_candidates[key][0]:
                unique_candidates[key] = score_data

        ranked_candidates = []
        for key, score_data in unique_candidates.items():
            ranked_candidates.append((key, score_data))
        ranked_candidates.sort(key=lambda x: x[1][0], reverse=True)

        return ranked_candidates[:pool_size]

    def search_employee_details(self, employee_name: str, query: str | None = None, top_k: int = 12):
        """
        Return the top matching chunks for a specific employee, suitable for
        synthesis into a friendly narrative answer.

        Args:
            employee_name: full display name to match against metadata.employee_name
            query: optional focused query; if None, use a generic profile query
            top_k: number of chunks to return

        Returns:
            List of dictionaries containing 'chunk_type', 'text', and other metadata
            ordered by descending similarity.
        """
        if self.index is None or not self.metas:
            raise ValueError("Index not loaded. Call load_index() first.")

        name_low = (employee_name or "").strip().lower()
        if not name_low:
            return []

        if not query or not str(query).strip():
            query = f"{employee_name} profile summary key skills experience certifications clients"  # generic profile query

        qv = embed_query(query)
        distances, indices = self.index.search(qv.astype(np.float32), min(top_k * 10, max(50, top_k * 5)))
        sims = distances[0]
        retrieved_metas = [self.metas[i] for i in indices[0]]

        rows = []
        for sim, m in zip(sims, retrieved_metas):
            nm = (m.get("employee_name") or "").strip().lower()
            if nm != name_low:
                continue
            rows.append((float(sim), m))
        rows.sort(key=lambda t: t[0], reverse=True)

        out = []
        for sim, m in rows[:top_k]:
            rec = dict(m)
            rec["similarity"] = sim
            out.append(rec)
        return out


# Example standalone execution for building the index
if __name__ == "__main__":
    from embed_only import EMB_NPY, META_JSONL, OUT_DIR
    service = FaissCandidateSearch(OUT_DIR)
    service.build_index(EMB_NPY, META_JSONL)