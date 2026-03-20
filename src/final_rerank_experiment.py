"""
BNS Two-Stage Retrieval: Hybrid + Cross-Encoder Re-ranking
============================================================
Stage 1: Hybrid (MPNet Dense + BM25 Sparse) → Top-20 candidate shortlist
Stage 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2) re-scores the shortlist

Metrics: Accuracy@K, MRR@K for K ∈ {3, 5}
Statistical: Paired t-test (Hybrid vs Hybrid+Rerank)
Comparison: Dense baseline vs Hybrid vs Hybrid+Rerank

Output:
  - results/final_benchmark_results.csv
  - logs/final_system_misses.json

Environment: conda run -n bns_rag python src/final_rerank_experiment.py
"""

import json
import os
import sys
import re
import time
import csv
import numpy as np
import faiss
from collections import defaultdict
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
LOGS_DIR = os.path.join(BASE_DIR, "logs")

DENSE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DENSE_INDEX_FILE = "mpnet.index"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
META_FILE = "chunk_metadata.json"

TOP_K_VALUES = [3, 5]
ALPHA = 0.5
CANDIDATE_POOL = 20

CATEGORIES = ["direct", "paraphrased", "scenario", "multi-section", "confusing"]
CATEGORY_LABELS = {
    "direct": "Direct",
    "paraphrased": "Paraphrased",
    "scenario": "Scenario-based",
    "multi-section": "Multi-section",
    "confusing": "Confusing",
}

# Previous baselines (from existing CSVs)
BASELINE_RESULTS = {
    ("Dense (MPNet)", 3): {"Accuracy": 0.6133, "MRR": 0.5200},
    ("Dense (MPNet)", 5): {"Accuracy": 0.6967, "MRR": 0.5377},
    ("Hybrid (MPNet+BM25)", 3): {"Accuracy": 0.6833, "MRR": 0.5656},
    ("Hybrid (MPNet+BM25)", 5): {"Accuracy": 0.7367, "MRR": 0.5771},
}

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


# ── Text Pre-processing ──────────────────────────────────────────────────

def clean_text(text: str) -> list:
    """Legal-aware BM25 tokenization preserving section references."""
    text = text.lower()
    text = re.sub(r'\bsection\s+(\d+)', r'section_\1', text)
    text = re.sub(r'\bs\.?\s*(\d+)', r'section_\1', text)
    text = re.sub(r'\b§\s*(\d+)', r'section_\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in LEGAL_STOP_WORDS and len(t) > 1]


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


# ── Metrics ───────────────────────────────────────────────────────────────

def get_correct_sections(query: dict) -> set:
    cs = query["correct_section"]
    return {str(s) for s in cs} if isinstance(cs, list) else {str(cs)}


def compute_hit(retrieved: list, correct: set) -> bool:
    return bool({str(s) for s in retrieved} & correct)


def compute_rr(retrieved: list, correct: set) -> float:
    for i, sid in enumerate(retrieved):
        if str(sid) in correct:
            return 1.0 / (i + 1)
    return 0.0


# ── Stage 1: Hybrid Candidate Generation ─────────────────────────────────

def hybrid_retrieve_candidates(
    query_text: str,
    dense_model: SentenceTransformer,
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    n_chunks: int,
    pool: int = CANDIDATE_POOL,
) -> list:
    """
    Returns list of (doc_index, fused_score) tuples, sorted by fused score.
    """
    # Dense
    q_emb = dense_model.encode([query_text], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    d_scores_raw, d_indices = faiss_index.search(q_emb, pool)
    d_scores_raw, d_indices = d_scores_raw[0], d_indices[0]

    # BM25
    q_tokens = clean_text(query_text)
    bm25_all = bm25_index.get_scores(q_tokens)
    bm25_top = np.argsort(bm25_all)[::-1][:pool]

    # Union candidates
    candidates = sorted(set(d_indices.tolist()) | set(bm25_top.tolist()) - {-1})
    candidates = [c for c in candidates if 0 <= c < n_chunks]

    # Score maps
    d_map = dict(zip(d_indices.tolist(), d_scores_raw.tolist()))
    b_map = {int(c): float(bm25_all[c]) for c in candidates}

    d_arr = np.array([d_map.get(c, 0.0) for c in candidates])
    b_arr = np.array([b_map.get(c, 0.0) for c in candidates])

    # Min-Max + fuse
    d_norm = min_max_normalize(d_arr)
    b_norm = min_max_normalize(b_arr)
    fused = ALPHA * d_norm + (1 - ALPHA) * b_norm

    # Sort by fused score descending
    ranked = sorted(zip(candidates, fused.tolist()), key=lambda x: -x[1])
    return ranked[:pool]


# ── Stage 2: Cross-Encoder Re-ranking ─────────────────────────────────────

def rerank_with_cross_encoder(
    query_text: str,
    candidate_doc_indices: list,
    chunks: list,
    cross_encoder: CrossEncoder,
    top_k: int,
) -> list:
    """
    Re-rank candidate chunks using Cross-Encoder.
    Returns list of (doc_index, ce_score) sorted by CE score descending.
    """
    if not candidate_doc_indices:
        return []

    # Build (query, passage) pairs
    pairs = [(query_text, chunks[idx]["text"]) for idx in candidate_doc_indices]

    # Score with Cross-Encoder
    ce_scores = cross_encoder.predict(pairs)

    # Sort by CE score descending
    scored = sorted(
        zip(candidate_doc_indices, ce_scores.tolist()),
        key=lambda x: -x[1],
    )
    return scored[:top_k]


# ── Dense-only retrieval (for baseline per-query scores) ──────────────────

def dense_retrieve_per_query(queries, faiss_index, metadata, dense_model, max_k):
    """Returns per-query hit arrays for t-test comparison."""
    q_texts = [q["query"] for q in queries]
    q_embs = dense_model.encode(q_texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    q_embs = np.array(q_embs, dtype=np.float32)
    _, indices = faiss_index.search(q_embs, max_k)

    per_query_hits = {}
    for top_k in TOP_K_VALUES:
        hits = []
        for qi, q in enumerate(queries):
            correct = get_correct_sections(q)
            retrieved = [str(metadata[idx]["section_id"]) for idx in indices[qi][:top_k] if 0 <= idx < len(metadata)]
            hits.append(1 if compute_hit(retrieved, correct) else 0)
        per_query_hits[top_k] = np.array(hits)
    return per_query_hits


# ── Hybrid-only retrieval per-query (for t-test) ─────────────────────────

def hybrid_retrieve_per_query(queries, dense_model, faiss_index, bm25_index, metadata, n_chunks, max_k):
    """Returns per-query hit arrays for hybrid (no reranking)."""
    per_query_hits = {k: [] for k in TOP_K_VALUES}

    for qi, q in enumerate(queries):
        correct = get_correct_sections(q)
        candidates = hybrid_retrieve_candidates(q["query"], dense_model, faiss_index, bm25_index, n_chunks)

        for top_k in TOP_K_VALUES:
            retrieved = [str(metadata[c[0]]["section_id"]) for c in candidates[:top_k]]
            per_query_hits[top_k].append(1 if compute_hit(retrieved, correct) else 0)

    return {k: np.array(v) for k, v in per_query_hits.items()}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 75)
    print("BNS TWO-STAGE RETRIEVAL: Hybrid + Cross-Encoder Re-ranking")
    print("=" * 75)

    # Load data
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    meta_path = os.path.join(INDEX_DIR, META_FILE)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n_chunks = len(chunks)
    print(f"  Corpus:       {n_chunks} chunks")
    print(f"  Queries:      {len(queries)}")
    print(f"  Top-K:        {TOP_K_VALUES}")
    print(f"  Hybrid pool:  {CANDIDATE_POOL}")
    print(f"  Fusion alpha: {ALPHA}\n")

    # Load models
    print("─── Loading Models ───")
    idx_path = os.path.join(INDEX_DIR, DENSE_INDEX_FILE)
    faiss_index = faiss.read_index(idx_path)
    print(f"  ✅ FAISS: {faiss_index.ntotal} vectors, dim={faiss_index.d}")

    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    print(f"  ✅ Dense: {DENSE_MODEL_NAME}")

    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"  ✅ Cross-Encoder: {CROSS_ENCODER_MODEL}")

    print("  ⏳ Building BM25 index...")
    tokenized = [clean_text(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(tokenized)
    print(f"  ✅ BM25: {n_chunks} documents\n")

    # ── Compute dense baseline per-query hits (for t-test) ────────
    print("─── Dense Baseline (per-query) ───")
    max_k = max(TOP_K_VALUES)
    dense_pq = dense_retrieve_per_query(queries, faiss_index, metadata, dense_model, max_k)
    for k in TOP_K_VALUES:
        print(f"  Dense @{k}: {dense_pq[k].sum()}/{len(queries)} hits")

    # ── Run Two-Stage Pipeline ────────────────────────────────────
    print("\n─── Two-Stage Pipeline: Hybrid → Cross-Encoder Re-rank ───")

    rerank_results = {k: [] for k in TOP_K_VALUES}  # per-query hit (0/1)
    rerank_rr = {k: [] for k in TOP_K_VALUES}  # per-query reciprocal rank
    hybrid_only_hits = {k: [] for k in TOP_K_VALUES}
    all_misses = []
    category_stats = {k: defaultdict(lambda: {"total": 0, "hits": 0}) for k in TOP_K_VALUES}

    for qi, query in enumerate(tqdm(queries, desc="  Two-stage retrieval")):
        correct = get_correct_sections(query)
        cat = query.get("category", "unknown")

        # Stage 1: Hybrid candidates
        candidates = hybrid_retrieve_candidates(
            query["query"], dense_model, faiss_index, bm25_index, n_chunks
        )
        candidate_indices = [c[0] for c in candidates]

        # Track hybrid-only performance (before rerank)
        for top_k in TOP_K_VALUES:
            hybrid_secs = [str(metadata[c[0]]["section_id"]) for c in candidates[:top_k]]
            hybrid_only_hits[top_k].append(1 if compute_hit(hybrid_secs, correct) else 0)

        # Stage 2: Cross-Encoder re-rank
        reranked = rerank_with_cross_encoder(
            query["query"], candidate_indices, chunks, cross_encoder, max_k
        )

        for top_k in TOP_K_VALUES:
            final_secs = [str(metadata[r[0]]["section_id"]) for r in reranked[:top_k]]
            hit = compute_hit(final_secs, correct)
            rr = compute_rr(final_secs, correct)

            rerank_results[top_k].append(1 if hit else 0)
            rerank_rr[top_k].append(rr)

            # Category tracking
            category_stats[top_k][cat]["total"] += 1
            if hit:
                category_stats[top_k][cat]["hits"] += 1

            if not hit:
                all_misses.append({
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "category": cat,
                    "correct_section": query["correct_section"],
                    "retrieved_sections": final_secs,
                    "top_k": top_k,
                })

    # Convert to numpy
    for k in TOP_K_VALUES:
        rerank_results[k] = np.array(rerank_results[k])
        rerank_rr[k] = np.array(rerank_rr[k])
        hybrid_only_hits[k] = np.array(hybrid_only_hits[k])

    # ── Compute Final Metrics ─────────────────────────────────────
    print("\n")
    final_rows = []

    for k in TOP_K_VALUES:
        acc = rerank_results[k].mean()
        mrr = rerank_rr[k].mean()
        final_rows.append({
            "Model": "Hybrid + CrossEncoder",
            "TopK": k,
            "Accuracy": round(float(acc), 4),
            "Mean_Reciprocal_Rank": round(float(mrr), 4),
        })

    # ── Statistical Significance (Paired t-test) ──────────────────
    print("─── Statistical Significance (Paired t-test) ───")
    ttest_results = {}
    for k in TOP_K_VALUES:
        # Hybrid vs Hybrid+Rerank
        t_stat, p_val = ttest_rel(hybrid_only_hits[k], rerank_results[k])
        ttest_results[k] = {"t_stat": round(float(t_stat), 4), "p_value": round(float(p_val), 6)}
        sig = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ NOT significant"
        print(f"  @Top-{k}: t={t_stat:.4f}, p={p_val:.6f}  {sig}")

    # ── Categorical Breakdown ─────────────────────────────────────
    print("\n─── Categorical Performance (Hybrid + CrossEncoder) ───")
    for k in TOP_K_VALUES:
        print(f"\n  @ Top-{k}:")
        print(f"  | {'Category':<22} | {'Total':>6} | {'Hits':>6} | {'Acc%':>8} |")
        print(f"  |{'-'*24}|{'-'*8}|{'-'*8}|{'-'*10}|")
        for cat in CATEGORIES:
            s = category_stats[k][cat]
            total = s["total"]
            hits = s["hits"]
            pct = (hits / total * 100) if total > 0 else 0.0
            label = CATEGORY_LABELS.get(cat, cat)
            print(f"  | {label:<22} | {total:>6} | {hits:>6} | {pct:>7.2f}% |")

    # ── Full Comparison Table ─────────────────────────────────────
    print("\n" + "=" * 75)
    print("FINAL COMPARISON: All Systems")
    print("=" * 75)
    print(f"  {'System':<30} {'K':>3}  {'Accuracy':>10}  {'MRR':>10}")
    print(f"  {'─'*30} {'─'*3}  {'─'*10}  {'─'*10}")

    # System A: Dense
    for k in TOP_K_VALUES:
        b = BASELINE_RESULTS.get(("Dense (MPNet)", k), {})
        print(f"  {'Dense (MPNet)':<30} {k:>3}  {b.get('Accuracy',0):>10.4f}  {b.get('MRR',0):>10.4f}")

    # System B: Hybrid
    for k in TOP_K_VALUES:
        b = BASELINE_RESULTS.get(("Hybrid (MPNet+BM25)", k), {})
        print(f"  {'Hybrid (MPNet+BM25)':<30} {k:>3}  {b.get('Accuracy',0):>10.4f}  {b.get('MRR',0):>10.4f}")

    # System C: Hybrid + CrossEncoder
    for r in final_rows:
        print(f"  {'Hybrid + CrossEncoder':<30} {r['TopK']:>3}  {r['Accuracy']:>10.4f}  {r['Mean_Reciprocal_Rank']:>10.4f}")

    # Deltas
    print()
    for k in TOP_K_VALUES:
        dense_acc = BASELINE_RESULTS.get(("Dense (MPNet)", k), {}).get("Accuracy", 0)
        hybrid_acc = BASELINE_RESULTS.get(("Hybrid (MPNet+BM25)", k), {}).get("Accuracy", 0)
        final_acc = [r["Accuracy"] for r in final_rows if r["TopK"] == k][0]
        print(f"  @Top-{k} improvement over Dense:  +{(final_acc - dense_acc)*100:.1f}pp")
        print(f"  @Top-{k} improvement over Hybrid: +{(final_acc - hybrid_acc)*100:.1f}pp")
        t_info = ttest_results[k]
        sig_label = "p < 0.05 ✅" if t_info["p_value"] < 0.05 else "p >= 0.05 ❌"
        print(f"  @Top-{k} t-test (Hybrid vs Rerank): p={t_info['p_value']:.6f} ({sig_label})")
        print()

    # ── Save CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "final_benchmark_results.csv")
    all_csv_rows = []

    for k in TOP_K_VALUES:
        b = BASELINE_RESULTS.get(("Dense (MPNet)", k), {})
        all_csv_rows.append({"System": "Dense (MPNet)", "TopK": k,
                             "Accuracy": b.get("Accuracy", 0), "MRR": b.get("MRR", 0)})

    for k in TOP_K_VALUES:
        b = BASELINE_RESULTS.get(("Hybrid (MPNet+BM25)", k), {})
        all_csv_rows.append({"System": "Hybrid (MPNet+BM25)", "TopK": k,
                             "Accuracy": b.get("Accuracy", 0), "MRR": b.get("MRR", 0)})

    for r in final_rows:
        all_csv_rows.append({"System": "Hybrid + CrossEncoder", "TopK": r["TopK"],
                             "Accuracy": r["Accuracy"], "MRR": r["Mean_Reciprocal_Rank"]})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["System", "TopK", "Accuracy", "MRR"])
        writer.writeheader()
        writer.writerows(all_csv_rows)

    # ── Save Misses ───────────────────────────────────────────────
    misses_path = os.path.join(LOGS_DIR, "final_system_misses.json")
    with open(misses_path, "w", encoding="utf-8") as f:
        json.dump(all_misses, f, indent=2, ensure_ascii=False)

    print(f"  ✅ CSV:    {csv_path} ({len(all_csv_rows)} rows)")
    print(f"  ✅ Misses: {misses_path} ({len(all_misses)} entries)")
    print("=" * 75)


if __name__ == "__main__":
    main()
