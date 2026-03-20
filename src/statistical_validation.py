"""
BNS Statistical Validation
============================
Paired t-tests comparing retrieval system variants:
  Test A: Dense-only vs Hybrid (α=0.5)
  Test B: Hybrid (α=0.5) vs Hybrid + Cross-Encoder Reranking

Uses per-query binary hit vectors (1=hit, 0=miss) at Top-K = {3, 5}.

Environment: conda run -n bns_rag python src/statistical_validation.py
"""

import json
import os
import re
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from scipy.stats import ttest_rel
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "bns_chunks.json")
QUERIES_PATH = os.path.join(BASE_DIR, "data", "evaluation_queries_augmented.json")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DENSE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE_INDEX_FILE = "mpnet.index"
META_FILE = "chunk_metadata.json"

TOP_K_VALUES = [3, 5]
ALPHA = 0.5
CANDIDATE_POOL = 20

LEGAL_STOP_WORDS = {
    "the", "of", "and", "in", "to", "is", "a", "an", "or", "for",
    "by", "on", "at", "be", "as", "it", "that", "this", "with",
    "from", "are", "was", "were", "been", "has", "have", "had",
    "shall", "may", "will", "can", "such", "any", "which", "who",
    "whom", "where", "when", "if", "not", "no", "but", "so",
    "than", "into", "upon", "under", "above", "below", "between",
    "through", "during", "before", "after", "about", "against",
    "other", "also", "being", "its", "their", "his", "her",
    "he", "she", "they", "them", "him",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\bsection\s+(\d+)', r'section_\1', text)
    text = re.sub(r'\bs\.?\s*(\d+)', r'section_\1', text)
    text = re.sub(r'\b§\s*(\d+)', r'section_\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in LEGAL_STOP_WORDS and len(t) > 1]


def min_max_normalize(scores):
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def get_correct(q):
    cs = q["correct_section"]
    return {str(s) for s in cs} if isinstance(cs, list) else {str(cs)}


def is_hit(retrieved, correct):
    return 1 if bool({str(s) for s in retrieved} & correct) else 0


# ── System A: Dense-Only ──────────────────────────────────────────────────

def compute_dense_hits(queries, faiss_index, metadata, dense_model, max_k):
    print("  ⏳ Computing Dense-only hits...")
    q_texts = [q["query"] for q in queries]
    q_embs = dense_model.encode(q_texts, show_progress_bar=True, batch_size=64,
                                normalize_embeddings=True)
    q_embs = np.array(q_embs, dtype=np.float32)
    _, indices = faiss_index.search(q_embs, max_k)

    hits = {}
    for k in TOP_K_VALUES:
        vec = []
        for qi, q in enumerate(queries):
            correct = get_correct(q)
            retrieved = [str(metadata[idx]["section_id"]) for idx in indices[qi][:k]
                         if 0 <= idx < len(metadata)]
            vec.append(is_hit(retrieved, correct))
        hits[k] = np.array(vec)
        print(f"    Dense @{k}: {sum(vec)}/{len(vec)} hits")
    return hits


# ── System B: Hybrid (α=0.5) ─────────────────────────────────────────────

def hybrid_retrieve(query_text, dense_model, faiss_index, bm25_index, metadata,
                    n_chunks, alpha, pool):
    q_emb = dense_model.encode([query_text], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    d_scores, d_idx = faiss_index.search(q_emb, pool)
    d_scores, d_idx = d_scores[0], d_idx[0]

    bm25_all = bm25_index.get_scores(clean_text(query_text))
    bm25_top = np.argsort(bm25_all)[::-1][:pool]

    cands = sorted(set(d_idx.tolist()) | set(bm25_top.tolist()) - {-1})
    cands = [c for c in cands if 0 <= c < n_chunks]

    d_map = dict(zip(d_idx.tolist(), d_scores.tolist()))
    d_arr = np.array([d_map.get(c, 0.0) for c in cands])
    b_arr = np.array([float(bm25_all[c]) for c in cands])

    fused = alpha * min_max_normalize(d_arr) + (1 - alpha) * min_max_normalize(b_arr)
    ranked = sorted(zip(cands, fused.tolist()), key=lambda x: -x[1])
    return ranked[:pool]


def compute_hybrid_hits(queries, dense_model, faiss_index, bm25_index, metadata,
                        n_chunks, max_k):
    print("  ⏳ Computing Hybrid hits...")
    hits = {k: [] for k in TOP_K_VALUES}
    hybrid_candidates = []  # Cache for reranker

    for q in tqdm(queries, desc="    Hybrid"):
        correct = get_correct(q)
        ranked = hybrid_retrieve(q["query"], dense_model, faiss_index, bm25_index,
                                 metadata, n_chunks, ALPHA, CANDIDATE_POOL)
        hybrid_candidates.append(ranked)

        for k in TOP_K_VALUES:
            retrieved = [str(metadata[c[0]]["section_id"]) for c in ranked[:k]]
            hits[k].append(is_hit(retrieved, correct))

    for k in TOP_K_VALUES:
        hits[k] = np.array(hits[k])
        print(f"    Hybrid @{k}: {hits[k].sum()}/{len(queries)} hits")
    return hits, hybrid_candidates


# ── System C: Hybrid + Cross-Encoder ─────────────────────────────────────

def compute_rerank_hits(queries, hybrid_candidates, chunks, cross_encoder, metadata, max_k):
    print("  ⏳ Computing Hybrid + CrossEncoder hits...")
    hits = {k: [] for k in TOP_K_VALUES}

    for qi, q in enumerate(tqdm(queries, desc="    Rerank")):
        correct = get_correct(q)
        cand_indices = [c[0] for c in hybrid_candidates[qi]]

        if not cand_indices:
            for k in TOP_K_VALUES:
                hits[k].append(0)
            continue

        pairs = [(q["query"], chunks[idx]["text"]) for idx in cand_indices]
        ce_scores = cross_encoder.predict(pairs)
        scored = sorted(zip(cand_indices, ce_scores.tolist()), key=lambda x: -x[1])

        for k in TOP_K_VALUES:
            retrieved = [str(metadata[r[0]]["section_id"]) for r in scored[:k]]
            hits[k].append(is_hit(retrieved, correct))

    for k in TOP_K_VALUES:
        hits[k] = np.array(hits[k])
        print(f"    Rerank @{k}: {hits[k].sum()}/{len(queries)} hits")
    return hits


# ── T-Test Helper ─────────────────────────────────────────────────────────

def run_ttest(name, vec_a, vec_b, label_a, label_b):
    t_stat, p_val = ttest_rel(vec_a, vec_b)

    if p_val < 0.01:
        sig = "HIGHLY SIGNIFICANT (p < 0.01) ✅✅"
    elif p_val < 0.05:
        sig = "SIGNIFICANT (p < 0.05) ✅"
    else:
        sig = "NOT significant (p ≥ 0.05) ❌"

    print(f"\n  {name}")
    print(f"    {label_a} mean: {vec_a.mean():.4f}  |  {label_b} mean: {vec_b.mean():.4f}")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value:     {p_val:.6f}")
    print(f"    ▸ {sig}")
    return {"test": name, "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6), "significance": sig}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STATISTICAL VALIDATION — PAIRED T-TESTS")
    print("=" * 70)

    # Load data
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(os.path.join(INDEX_DIR, META_FILE), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n_chunks = len(chunks)
    max_k = max(TOP_K_VALUES)
    print(f"  Corpus: {n_chunks} | Queries: {len(queries)}\n")

    # Load models
    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, DENSE_INDEX_FILE))
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    tokenized = [clean_text(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(tokenized)
    print("  ✅ All models loaded\n")

    # Compute per-query hit vectors
    print("─── System A: Dense-Only ───")
    dense_hits = compute_dense_hits(queries, faiss_index, metadata, dense_model, max_k)

    print("\n─── System B: Hybrid (α=0.5) ───")
    hybrid_hits, hybrid_cands = compute_hybrid_hits(
        queries, dense_model, faiss_index, bm25_index, metadata, n_chunks, max_k)

    print("\n─── System C: Hybrid + CrossEncoder ───")
    rerank_hits = compute_rerank_hits(queries, hybrid_cands, chunks, cross_encoder,
                                     metadata, max_k)

    # ── Paired T-Tests ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PAIRED T-TEST RESULTS")
    print("=" * 70)

    results = []
    for k in TOP_K_VALUES:
        results.append(run_ttest(
            f"Test A @{k}: Dense vs Hybrid",
            dense_hits[k], hybrid_hits[k], "Dense", "Hybrid",
        ))
        results.append(run_ttest(
            f"Test B @{k}: Hybrid vs Hybrid+Rerank",
            hybrid_hits[k], rerank_hits[k], "Hybrid", "Rerank",
        ))

    # ── Summary Table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)
    print(f"\n| {'Test':<40} | {'t-stat':>8} | {'p-value':>10} | {'Result':<35} |")
    print(f"|{'-'*42}|{'-'*10}|{'-'*12}|{'-'*37}|")
    for r in results:
        print(f"| {r['test']:<40} | {r['t_stat']:>8.4f} | {r['p_value']:>10.6f} | {r['significance']:<35} |")

    # ── Save CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "statistical_validation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test", "t_stat", "p_value", "significance"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ CSV saved: {csv_path}")
    print("=" * 70)


import csv  # at top-level for the save

if __name__ == "__main__":
    main()
