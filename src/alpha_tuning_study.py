"""
BNS Alpha Sensitivity Study
=============================
Sweeps α ∈ {0.3, 0.5, 0.7} for the Hybrid (MPNet Dense + BM25) system.
Computes Accuracy@3, Accuracy@5, MRR@5 overall and per-category Accuracy@5.

Generates:
  - results/alpha_sensitivity.csv
  - results/alpha_vs_accuracy.png  (line plot)
  - results/alpha_category_bars.png  (grouped bar chart)

Environment: conda run -n bns_rag python src/alpha_tuning_study.py
"""

import json
import os
import re
import sys
import time
import csv
import numpy as np
import faiss
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "bns_chunks.json")
QUERIES_PATH = os.path.join(BASE_DIR, "data", "evaluation_queries_augmented.json")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DENSE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DENSE_INDEX_FILE = "mpnet.index"
META_FILE = "chunk_metadata.json"

ALPHA_VALUES = [0.3, 0.5, 0.7]
TOP_K_VALUES = [3, 5]
CANDIDATE_POOL = 20

CATEGORIES = ["direct", "paraphrased", "scenario", "multi-section", "confusing"]
CAT_LABELS = {
    "direct": "Direct", "paraphrased": "Paraphrased", "scenario": "Scenario",
    "multi-section": "Multi-section", "confusing": "Confusing",
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


def hit(retrieved, correct):
    return bool({str(s) for s in retrieved} & correct)


def rr(retrieved, correct):
    for i, s in enumerate(retrieved):
        if str(s) in correct:
            return 1.0 / (i + 1)
    return 0.0


# ── Hybrid Retrieval at given Alpha ──────────────────────────────────────

def hybrid_search_one(query_text, dense_model, faiss_index, bm25_index, metadata,
                      n_chunks, alpha, top_k):
    pool = CANDIDATE_POOL

    # Dense
    q_emb = dense_model.encode([query_text], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    d_scores, d_idx = faiss_index.search(q_emb, pool)
    d_scores, d_idx = d_scores[0], d_idx[0]

    # BM25
    bm25_all = bm25_index.get_scores(clean_text(query_text))
    bm25_top = np.argsort(bm25_all)[::-1][:pool]

    # Union
    cands = sorted(set(d_idx.tolist()) | set(bm25_top.tolist()) - {-1})
    cands = [c for c in cands if 0 <= c < n_chunks]

    d_map = dict(zip(d_idx.tolist(), d_scores.tolist()))
    d_arr = np.array([d_map.get(c, 0.0) for c in cands])
    b_arr = np.array([float(bm25_all[c]) for c in cands])

    fused = alpha * min_max_normalize(d_arr) + (1 - alpha) * min_max_normalize(b_arr)
    ranked = sorted(zip(cands, fused.tolist()), key=lambda x: -x[1])

    return [str(metadata[c[0]]["section_id"]) for c in ranked[:top_k]]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("ALPHA SENSITIVITY STUDY — Hybrid (MPNet + BM25)")
    print(f"Alpha values: {ALPHA_VALUES}")
    print("=" * 70)

    # Load data
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(os.path.join(INDEX_DIR, META_FILE), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n_chunks = len(chunks)
    print(f"  Corpus: {n_chunks} | Queries: {len(queries)}\n")

    # Load models
    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, DENSE_INDEX_FILE))
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    tokenized = [clean_text(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(tokenized)
    print(f"  ✅ Models loaded\n")

    max_k = max(TOP_K_VALUES)
    all_results = []  # For CSV
    plot_data = []    # For line plot: (alpha, accuracy@5)
    cat_plot_data = []  # For bar chart: (alpha, category, accuracy@5)

    for alpha in ALPHA_VALUES:
        print(f"─── α = {alpha} ───")

        # Per-query metrics
        hits_k = {k: 0 for k in TOP_K_VALUES}
        mrr_5_total = 0.0
        cat_stats = {k: defaultdict(lambda: {"total": 0, "hits": 0}) for k in TOP_K_VALUES}

        for qi, q in enumerate(tqdm(queries, desc=f"  α={alpha}")):
            correct = get_correct(q)
            cat = q.get("category", "unknown")
            retrieved = hybrid_search_one(
                q["query"], dense_model, faiss_index, bm25_index, metadata,
                n_chunks, alpha, max_k,
            )

            for k in TOP_K_VALUES:
                is_hit = hit(retrieved[:k], correct)
                cat_stats[k][cat]["total"] += 1
                if is_hit:
                    hits_k[k] += 1
                    cat_stats[k][cat]["hits"] += 1

            mrr_5_total += rr(retrieved[:5], correct)

        n = len(queries)
        acc3 = hits_k[3] / n
        acc5 = hits_k[5] / n
        mrr5 = mrr_5_total / n

        print(f"  Accuracy@3: {acc3:.4f}  |  Accuracy@5: {acc5:.4f}  |  MRR@5: {mrr5:.4f}")

        all_results.append({
            "Alpha": alpha, "Accuracy@3": round(acc3, 4),
            "Accuracy@5": round(acc5, 4), "MRR@5": round(mrr5, 4),
        })
        plot_data.append((alpha, acc5))

        # Category breakdown @5
        for cat in CATEGORIES:
            s = cat_stats[5][cat]
            cat_acc = s["hits"] / s["total"] if s["total"] > 0 else 0.0
            label = CAT_LABELS.get(cat, cat)
            all_results[-1][f"Cat_{label}_Acc@5"] = round(cat_acc, 4)
            cat_plot_data.append({"Alpha": str(alpha), "Category": label,
                                  "Accuracy@5": round(cat_acc * 100, 2)})
            print(f"    {label:<15} Acc@5: {cat_acc:.4f} ({s['hits']}/{s['total']})")
        print()

    # ── CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "alpha_sensitivity.csv")
    fieldnames = ["Alpha", "Accuracy@3", "Accuracy@5", "MRR@5"] + \
                 [f"Cat_{CAT_LABELS[c]}_Acc@5" for c in CATEGORIES]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✅ CSV saved: {csv_path}")

    # ── Plot 1: Alpha vs Accuracy@5 (line plot) ──────────────────
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 5))
    alphas = [p[0] for p in plot_data]
    accs = [p[1] * 100 for p in plot_data]
    ax.plot(alphas, accs, "o-", color="#2196F3", linewidth=2.5, markersize=10)
    for a, acc in zip(alphas, accs):
        ax.annotate(f"{acc:.1f}%", (a, acc), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Alpha (α) — Dense Weight", fontsize=13)
    ax.set_ylabel("Accuracy@5 (%)", fontsize=13)
    ax.set_title("Alpha Sensitivity: Hybrid Retrieval Accuracy@5", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(alphas)
    ax.set_ylim(min(accs) - 5, max(accs) + 5)
    plt.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "alpha_vs_accuracy.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"✅ Plot saved: {p1}")

    # ── Plot 2: Category-wise grouped bar chart ──────────────────
    import pandas as pd
    df = pd.DataFrame(cat_plot_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.22
    x = np.arange(len(CATEGORIES))

    colors = ["#FF7043", "#42A5F5", "#66BB6A"]
    for i, alpha in enumerate(ALPHA_VALUES):
        subset = df[df["Alpha"] == str(alpha)]
        # Ensure order matches CATEGORIES
        vals = [subset[subset["Category"] == CAT_LABELS[c]]["Accuracy@5"].values[0]
                for c in CATEGORIES]
        bars = ax.bar(x + i * bar_width, vals, bar_width, label=f"α={alpha}",
                      color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Category", fontsize=13)
    ax.set_ylabel("Accuracy@5 (%)", fontsize=13)
    ax.set_title("Category-wise Accuracy@5 by Alpha", fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES], fontsize=11)
    ax.legend(title="Alpha", fontsize=11)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "alpha_category_bars.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"✅ Plot saved: {p2}")

    # ── Markdown Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALPHA SENSITIVITY — PAPER-READY TABLE")
    print("=" * 70)
    print(f"\n| {'α':>4} | {'Acc@3':>8} | {'Acc@5':>8} | {'MRR@5':>8} |"
          + "".join(f" {CAT_LABELS[c]:>12} |" for c in CATEGORIES))
    print(f"|{'-'*6}|{'-'*10}|{'-'*10}|{'-'*10}|"
          + "".join(f"{'-'*14}|" for _ in CATEGORIES))
    for r in all_results:
        line = f"| {r['Alpha']:>4} | {r['Accuracy@3']:>8.4f} | {r['Accuracy@5']:>8.4f} | {r['MRR@5']:>8.4f} |"
        for c in CATEGORIES:
            line += f" {r.get(f'Cat_{CAT_LABELS[c]}_Acc@5', 0):>12.4f} |"
        print(line)
    print("=" * 70)


if __name__ == "__main__":
    main()
