"""
BNS Retrieval Experiment
=========================
Benchmarks dual-model FAISS retrieval over the full 300-query BNS evaluation dataset.

Models:
  - intfloat/e5-large-v2       → query prefix: "query: "
  - all-mpnet-base-v2          → no prefix

Metrics (per model, per K ∈ {3, 5}):
  - Hit Rate (Accuracy@K)
  - Mean Reciprocal Rank (MRR@K)

Output:
  - results/retrieval_results.csv
  - logs/retrieval_errors.json

Environment: conda run -n bns_rag python src/retrieval_experiment.py
"""

import json
import os
import sys
import time
import csv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES_PATH = os.path.join(BASE_DIR, "data", "evaluation_queries_augmented.json")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

TOP_K_VALUES = [3, 5]

MODEL_CONFIGS = [
    {
        "label": "E5-Large-v2",
        "model_name": "intfloat/e5-large-v2",
        "index_file": "e5_large.index",
        "query_prefix": "query: ",
    },
    {
        "label": "MPNet-Base-v2",
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "index_file": "mpnet.index",
        "query_prefix": "",
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────

def get_correct_sections(query: dict) -> set:
    """Extract correct_section as a set (handles both str and list)."""
    cs = query["correct_section"]
    return set(cs) if isinstance(cs, list) else {cs}


def evaluate_retrieval(
    queries: list,
    query_embeddings: np.ndarray,
    index: faiss.Index,
    metadata: list,
    model_label: str,
    top_k_values: list,
) -> tuple:
    """
    Evaluate retrieval over all queries for multiple Top-K values.

    Returns:
        results_rows: list of dicts for CSV output
        error_entries: list of miss records for JSON audit log
    """
    max_k = max(top_k_values)

    # Batch FAISS search
    scores, indices = index.search(query_embeddings, max_k)

    results_rows = []
    error_entries = []

    for top_k in top_k_values:
        hits = 0
        mrr_total = 0.0
        k_errors = []

        for qi, query in enumerate(queries):
            correct = get_correct_sections(query)

            # Get retrieved section IDs for this query at this K
            retrieved_idx = indices[qi][:top_k]
            retrieved_sections = [
                metadata[idx]["section_id"]
                for idx in retrieved_idx
                if 0 <= idx < len(metadata)
            ]

            # Hit: any correct section in retrieved set
            is_hit = bool(set(retrieved_sections) & correct)

            # MRR: 1/rank of first correct result
            rr = 0.0
            for rank, sid in enumerate(retrieved_sections, start=1):
                if sid in correct:
                    rr = 1.0 / rank
                    break

            if is_hit:
                hits += 1
            else:
                k_errors.append({
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "category": query.get("category", "unknown"),
                    "correct_section": query["correct_section"],
                    "retrieved_sections": retrieved_sections,
                    "model": model_label,
                    "top_k": top_k,
                })

            mrr_total += rr

        n = len(queries)
        accuracy = hits / n
        mrr = mrr_total / n

        results_rows.append({
            "Model": model_label,
            "TopK": top_k,
            "Accuracy": round(accuracy, 4),
            "Mean_Reciprocal_Rank": round(mrr, 4),
        })
        error_entries.extend(k_errors)

        print(f"  📊 {model_label} @ Top-{top_k}:")
        print(f"     Hit Rate (Accuracy@K): {accuracy:.4f}  ({hits}/{n})")
        print(f"     MRR@K:                 {mrr:.4f}")
        print(f"     Misses:                {len(k_errors)}")

    return results_rows, error_entries


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 70)
    print("BNS RETRIEVAL EXPERIMENT — DUAL MODEL BENCHMARK")
    print("=" * 70)

    # Load queries
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"  Queries:  {len(queries)}  ({os.path.basename(QUERIES_PATH)})")

    # Load chunk metadata
    meta_path = os.path.join(INDEX_DIR, "chunk_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  Metadata: {len(metadata)} chunks")
    print(f"  Top-K:    {TOP_K_VALUES}\n")

    all_results = []
    all_errors = []

    for cfg in MODEL_CONFIGS:
        label = cfg["label"]
        print("─" * 70)
        print(f"  MODEL: {label}  ({cfg['model_name']})")
        print("─" * 70)

        # Load FAISS index
        idx_path = os.path.join(INDEX_DIR, cfg["index_file"])
        if not os.path.exists(idx_path):
            print(f"  ❌ Index not found: {idx_path}")
            print(f"     Run: conda run -n bns_rag python src/build_indices.py")
            sys.exit(1)

        index = faiss.read_index(idx_path)
        print(f"  ✅ Index loaded — {index.ntotal} vectors, dim={index.d}")

        # Load model
        print(f"  ⏳ Loading model...")
        model = SentenceTransformer(cfg["model_name"])

        # Encode queries
        prefix = cfg["query_prefix"]
        query_texts = [f"{prefix}{q['query']}" for q in queries]

        print(f"  ⏳ Encoding {len(query_texts)} queries" +
              (f' (prefix: "{prefix}")' if prefix else " (no prefix)") + "...")
        t0 = time.time()
        q_embeddings = model.encode(
            query_texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )
        q_embeddings = np.array(q_embeddings, dtype=np.float32)
        print(f"  ✅ Encoded in {time.time() - t0:.1f}s — shape: {q_embeddings.shape}\n")

        # Evaluate
        rows, errors = evaluate_retrieval(
            queries, q_embeddings, index, metadata, label, TOP_K_VALUES
        )
        all_results.extend(rows)
        all_errors.extend(errors)
        print()

        del model, q_embeddings, index

    # ── Output: CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "retrieval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Model", "TopK", "Accuracy", "Mean_Reciprocal_Rank"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    # ── Output: Error Log ─────────────────────────────────────────────
    err_path = os.path.join(LOGS_DIR, "retrieval_errors.json")
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<20} {'Top-K':<8} {'Accuracy':<14} {'MRR':<14}")
    print(f"  {'─'*20} {'─'*8} {'─'*14} {'─'*14}")
    for r in all_results:
        print(f"  {r['Model']:<20} {r['TopK']:<8} {r['Accuracy']:<14.4f} {r['Mean_Reciprocal_Rank']:<14.4f}")
    print()
    print(f"  ✅ CSV:    {csv_path}")
    print(f"  ✅ Errors: {err_path} ({len(all_errors)} misses)")
    print("=" * 70)


if __name__ == "__main__":
    main()
