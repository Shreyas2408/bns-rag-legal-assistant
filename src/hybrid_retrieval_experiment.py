"""
BNS Hybrid Retrieval Experiment
=================================
Dense (MPNet-Base-v2 FAISS) + Sparse (BM25) with Min-Max score fusion.

Hybrid Formula:
    score_hybrid = α · norm(dense_score) + (1 − α) · norm(bm25_score)
    where α = 0.5 and norm = Min-Max Normalization

Metrics: Accuracy@K, MRR@K for K ∈ {3, 5}

Output:
  - results/hybrid_results.csv
  - results/hybrid_misses.json  (queries where hybrid failed but dense succeeded)

Environment: conda run -n bns_rag python src/hybrid_retrieval_experiment.py
"""

import json
import os
import sys
import re
import string
import time
import csv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "bns_chunks.json")
QUERIES_PATH = os.path.join(BASE_DIR, "data", "evaluation_queries_augmented.json")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
DENSE_INDEX_FILE = "mpnet.index"
META_FILE = "chunk_metadata.json"

TOP_K_VALUES = [3, 5]
ALPHA = 0.5  # Dense weight; BM25 weight = 1 - ALPHA
CANDIDATE_POOL = 20  # Retrieve top-N from each retriever before fusion

# Legal stop words to remove during BM25 tokenization
LEGAL_STOP_WORDS = {
    "the", "of", "and", "in", "to", "is", "a", "an", "or", "for",
    "by", "on", "at", "be", "as", "it", "that", "this", "with",
    "from", "are", "was", "were", "been", "has", "have", "had",
    "shall", "may", "will", "can", "such", "any", "which", "who",
    "whom", "where", "when", "if", "not", "no", "but", "so",
    "than", "into", "upon", "under", "above", "below", "between",
    "through", "during", "before", "after", "about", "against",
    "other", "also", "being", "its", "their", "his", "her",
    "he", "she", "they", "them", "him", "his",
}


# ── Text Pre-processing ──────────────────────────────────────────────────

def clean_text(text: str) -> list:
    """
    Tokenize text for BM25 with legal-domain awareness.

    - Preserves "Section 302" as "section_302" (not just "302")
    - Removes legal stop words
    - Handles punctuation while keeping numeric references
    """
    text = text.lower()

    # Preserve "section X" as a single token "section_X"
    text = re.sub(r'\bsection\s+(\d+)', r'section_\1', text)
    text = re.sub(r'\bs\.?\s*(\d+)', r'section_\1', text)
    text = re.sub(r'\b§\s*(\d+)', r'section_\1', text)

    # Remove punctuation except underscores (for section_NNN tokens)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Remove stop words but keep legal-meaningful tokens
    tokens = [t for t in tokens if t not in LEGAL_STOP_WORDS and len(t) > 1]

    return tokens


# ── Min-Max Normalization ─────────────────────────────────────────────────

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range using Min-Max scaling."""
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


# ── Metrics ───────────────────────────────────────────────────────────────

def get_correct_sections(query: dict) -> set:
    """Extract correct_section as a set of strings (handles str and list)."""
    cs = query["correct_section"]
    if isinstance(cs, list):
        return {str(s) for s in cs}
    return {str(cs)}


def compute_hit(retrieved_sections: list, correct: set) -> bool:
    return bool(set(str(s) for s in retrieved_sections) & correct)


def compute_rr(retrieved_sections: list, correct: set) -> float:
    for i, sid in enumerate(retrieved_sections):
        if str(sid) in correct:
            return 1.0 / (i + 1)
    return 0.0


# ── Dense Retrieval ───────────────────────────────────────────────────────

def load_dense_components():
    """Load FAISS index, metadata, and SentenceTransformer model."""
    idx_path = os.path.join(INDEX_DIR, DENSE_INDEX_FILE)
    meta_path = os.path.join(INDEX_DIR, META_FILE)

    if not os.path.exists(idx_path):
        print(f"  ❌ Dense index not found: {idx_path}")
        print(f"     Run: conda run -n bns_rag python src/build_indices.py")
        sys.exit(1)

    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer(DENSE_MODEL)
    return index, metadata, model


# ── BM25 Setup ────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list):
    """Build BM25 index from chunk texts."""
    print("  ⏳ Building BM25 index...")
    tokenized_corpus = [clean_text(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  ✅ BM25 index built — {len(tokenized_corpus)} documents")
    return bm25


# ── Hybrid Retrieval ──────────────────────────────────────────────────────

def hybrid_retrieve(
    query_text: str,
    dense_model: SentenceTransformer,
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    metadata: list,
    top_k: int,
) -> list:
    """
    Perform hybrid retrieval with Min-Max score fusion.

    1. Get top-CANDIDATE_POOL from Dense (FAISS)
    2. Get top-CANDIDATE_POOL from BM25
    3. Union candidate set
    4. Min-Max normalize both score arrays
    5. Fuse: α * dense_norm + (1-α) * bm25_norm
    6. Return top-K by fused score
    """
    pool = CANDIDATE_POOL

    # ── Dense retrieval ───────────────────────────────────────────
    q_emb = dense_model.encode([query_text], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    dense_scores_raw, dense_indices = faiss_index.search(q_emb, pool)
    dense_scores_raw = dense_scores_raw[0]
    dense_indices = dense_indices[0]

    # ── BM25 retrieval ────────────────────────────────────────────
    query_tokens = clean_text(query_text)
    bm25_scores_all = bm25_index.get_scores(query_tokens)

    # Get top-pool BM25 indices
    bm25_top_indices = np.argsort(bm25_scores_all)[::-1][:pool]

    # ── Build candidate union ─────────────────────────────────────
    candidate_set = set(dense_indices.tolist()) | set(bm25_top_indices.tolist())
    # Remove any -1 indices (FAISS padding)
    candidate_set.discard(-1)
    candidates = sorted(candidate_set)

    # ── Score vectors for all candidates ──────────────────────────
    dense_score_map = dict(zip(dense_indices.tolist(), dense_scores_raw.tolist()))
    bm25_score_map = {int(i): float(bm25_scores_all[i]) for i in candidates}

    # Build aligned arrays
    d_scores = np.array([dense_score_map.get(c, 0.0) for c in candidates])
    b_scores = np.array([bm25_score_map.get(c, 0.0) for c in candidates])

    # ── Min-Max Normalize ─────────────────────────────────────────
    d_norm = min_max_normalize(d_scores)
    b_norm = min_max_normalize(b_scores)

    # ── Fuse ──────────────────────────────────────────────────────
    fused = ALPHA * d_norm + (1 - ALPHA) * b_norm

    # ── Rank and return top-K ─────────────────────────────────────
    ranked_indices = np.argsort(fused)[::-1][:top_k]
    results = []
    for ri in ranked_indices:
        doc_idx = candidates[ri]
        if 0 <= doc_idx < len(metadata):
            results.append(str(metadata[doc_idx]["section_id"]))
    return results


# ── Load Dense Retrieval Results (for negative interference check) ────────

def load_dense_hits(queries: list, faiss_index, metadata, dense_model, top_k_values):
    """
    Pre-compute which queries were hits under pure dense retrieval
    to later identify negative interference from hybridization.
    """
    max_k = max(top_k_values)

    query_texts = [q["query"] for q in queries]
    print(f"  ⏳ Encoding {len(query_texts)} queries for dense baseline...")
    q_embs = dense_model.encode(
        query_texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True
    )
    q_embs = np.array(q_embs, dtype=np.float32)

    scores, indices = faiss_index.search(q_embs, max_k)

    # dense_hits[top_k] = set of query indices that were hits
    dense_hits = {}
    for top_k in top_k_values:
        hits = set()
        for qi, query in enumerate(queries):
            correct = get_correct_sections(query)
            retrieved = [
                str(metadata[idx]["section_id"])
                for idx in indices[qi][:top_k]
                if 0 <= idx < len(metadata)
            ]
            if compute_hit(retrieved, correct):
                hits.add(qi)
        dense_hits[top_k] = hits
    return dense_hits


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("BNS HYBRID RETRIEVAL EXPERIMENT")
    print(f"Dense: {DENSE_MODEL}  |  Sparse: BM25Okapi  |  α = {ALPHA}")
    print("=" * 70)

    # Load data
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"  Corpus:  {len(chunks)} chunks")
    print(f"  Queries: {len(queries)} (full benchmark)")
    print(f"  Top-K:   {TOP_K_VALUES}")
    print(f"  Fusion:  α={ALPHA} (dense) + {1-ALPHA} (BM25), Min-Max normalized\n")

    # Load components
    faiss_index, metadata, dense_model = load_dense_components()
    print(f"  ✅ Dense: FAISS {faiss_index.ntotal} vectors, dim={faiss_index.d}")

    bm25_index = build_bm25_index(chunks)

    # Pre-compute dense-only hits for negative interference detection
    print("\n─── DENSE BASELINE (for interference analysis) ───")
    dense_hits = load_dense_hits(queries, faiss_index, metadata, dense_model, TOP_K_VALUES)
    for k in TOP_K_VALUES:
        print(f"  Dense @{k}: {len(dense_hits[k])}/{len(queries)} hits")

    # ── Hybrid Experiment ─────────────────────────────────────────
    print("\n─── HYBRID RETRIEVAL ───")
    results_rows = []
    all_misses = []

    max_k = max(TOP_K_VALUES)

    # Cache hybrid results per query (retrieve at max_k, slice for each k)
    print(f"\n  ⏳ Running hybrid retrieval for {len(queries)} queries (pool={CANDIDATE_POOL})...")
    hybrid_results_cache = []
    for qi, query in enumerate(tqdm(queries, desc="  Hybrid retrieval")):
        retrieved = hybrid_retrieve(
            query["query"], dense_model, faiss_index, bm25_index, metadata, max_k
        )
        hybrid_results_cache.append(retrieved)

    # Evaluate at each K
    for top_k in TOP_K_VALUES:
        hits = 0
        mrr_total = 0.0
        k_misses = []

        for qi, query in enumerate(queries):
            correct = get_correct_sections(query)
            retrieved = hybrid_results_cache[qi][:top_k]

            is_hit = compute_hit(retrieved, correct)
            rr = compute_rr(retrieved, correct)

            if is_hit:
                hits += 1
            else:
                miss_entry = {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "category": query.get("category", "unknown"),
                    "correct_section": query["correct_section"],
                    "retrieved_sections": retrieved,
                    "top_k": top_k,
                    "dense_was_hit": qi in dense_hits[top_k],
                }
                k_misses.append(miss_entry)

            mrr_total += rr

        n = len(queries)
        accuracy = hits / n
        mrr = mrr_total / n

        # Count negative interference (dense hit → hybrid miss)
        neg_interference = sum(1 for m in k_misses if m["dense_was_hit"])

        results_rows.append({
            "Model": "Hybrid (MPNet + BM25)",
            "TopK": top_k,
            "Accuracy": round(accuracy, 4),
            "Mean_Reciprocal_Rank": round(mrr, 4),
        })
        all_misses.extend(k_misses)

        print(f"\n  📊 Hybrid @ Top-{top_k}:")
        print(f"     Hit Rate (Accuracy@K): {accuracy:.4f}  ({hits}/{n})")
        print(f"     MRR@K:                 {mrr:.4f}")
        print(f"     Misses:                {len(k_misses)}")
        print(f"     Negative interference: {neg_interference} "
              f"(dense hit → hybrid miss)")

    # ── CSV Output ────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "hybrid_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Model", "TopK", "Accuracy", "Mean_Reciprocal_Rank"]
        )
        writer.writeheader()
        writer.writerows(results_rows)

    # ── Misses JSON ───────────────────────────────────────────────
    misses_path = os.path.join(RESULTS_DIR, "hybrid_misses.json")
    with open(misses_path, "w", encoding="utf-8") as f:
        json.dump(all_misses, f, indent=2, ensure_ascii=False)

    # Separate negative-interference misses
    neg_only = [m for m in all_misses if m["dense_was_hit"]]

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<28} {'Top-K':<8} {'Accuracy':<14} {'MRR':<14}")
    print(f"  {'─'*28} {'─'*8} {'─'*14} {'─'*14}")
    for r in results_rows:
        print(f"  {r['Model']:<28} {r['TopK']:<8} "
              f"{r['Accuracy']:<14.4f} {r['Mean_Reciprocal_Rank']:<14.4f}")
    print()
    print(f"  ✅ CSV:    {csv_path}")
    print(f"  ✅ Misses: {misses_path} ({len(all_misses)} total, "
          f"{len(neg_only)} negative interference)")
    print("=" * 70)


if __name__ == "__main__":
    main()
