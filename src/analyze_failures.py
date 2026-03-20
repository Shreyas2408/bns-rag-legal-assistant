"""
BNS Categorical Failure Analysis
==================================
Processes logs/retrieval_errors.json against the full 300-query benchmark
to generate a per-category failure report with expert commentary.

Output:
  - Console: Markdown table + Technical Insights
  - results/failure_analysis.csv

Environment: conda run -n bns_rag python src/analyze_failures.py
"""

import json
import os
import csv
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERRORS_PATH = os.path.join(BASE_DIR, "logs", "retrieval_errors.json")
QUERIES_PATH = os.path.join(BASE_DIR, "data", "evaluation_queries_augmented.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Category display order
CATEGORIES = ["direct", "paraphrased", "scenario", "multi-section", "confusing"]
CATEGORY_LABELS = {
    "direct": "Direct",
    "paraphrased": "Paraphrased",
    "scenario": "Scenario-based",
    "multi-section": "Multi-section Reasoning",
    "confusing": "Confusing/Misleading",
}

# ── Expert Commentary ─────────────────────────────────────────────────────
INSIGHTS = {
    "direct": (
        "Insight: High Direct-query failure indicates poor lexical-semantic "
        "alignment. Consider fine-tuning the embedding model on BNS-specific "
        "terminology or augmenting chunks with section titles."
    ),
    "paraphrased": (
        "Insight: Paraphrased query failures reveal weak semantic generalization. "
        "A Cross-Encoder Reranker (e.g., ms-marco-MiniLM) on the top-K shortlist "
        "can significantly improve recall for synonym-heavy legal queries."
    ),
    "scenario": (
        "Insight: Scenario-based failures suggest the model struggles with "
        "fact-pattern-to-statute mapping. Query Decomposition (breaking "
        "scenarios into sub-queries) or Hypothetical Document Embeddings "
        "(HyDE) may improve performance."
    ),
    "multi-section": (
        "Insight: High Multi-section failure confirms the need for a "
        "Cross-Encoder Reranker or Query Expansion strategy. Single-vector "
        "retrieval fundamentally struggles when an act triggers multiple "
        "statutes — consider multi-hop retrieval or graph-based approaches."
    ),
    "confusing": (
        "Insight: Confusing/Edge-case failures are expected at moderate rates. "
        "These queries test General Exceptions (§14-§44) which are semantically "
        "distant from offence definitions. A dedicated exception-aware retrieval "
        "layer or chain-of-thought prompting may help."
    ),
}

INSIGHT_THRESHOLD = 50.0  # Failure % above which an insight is triggered


# ── Data Loading ──────────────────────────────────────────────────────────

def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Analysis ──────────────────────────────────────────────────────────────

def compute_category_totals(queries: list) -> dict:
    """Count total queries per category from the full 300-query benchmark."""
    totals = defaultdict(int)
    for q in queries:
        cat = q.get("category", "unknown")
        totals[cat] += 1
    return dict(totals)


def compute_failure_counts(errors: list) -> dict:
    """
    Count failures per (model, top_k, category).
    Returns: { (model, top_k): { category: count } }
    """
    failures = defaultdict(lambda: defaultdict(int))
    for e in errors:
        key = (e["model"], e["top_k"])
        cat = e.get("category", "unknown")
        failures[key][cat] += 1
    return failures


def generate_report(queries: list, errors: list):
    """Generate the full categorical failure report."""
    totals = compute_category_totals(queries)
    failures = compute_failure_counts(errors)

    # Collect all (model, top_k) combinations, sorted
    model_k_combos = sorted(failures.keys(), key=lambda x: (x[0], x[1]))

    # ── Console Markdown Table ────────────────────────────────────────
    print("=" * 80)
    print("CATEGORICAL FAILURE REPORT — BNS Retrieval Evaluation")
    print("=" * 80)

    all_csv_rows = []

    for model, top_k in model_k_combos:
        cat_failures = failures[(model, top_k)]

        print(f"\n### {model} @ Top-{top_k}")
        print()
        print(f"| {'Category':<25} | {'Total':>7} | {'Failed':>7} | {'Failure %':>10} |")
        print(f"|{'-'*27}|{'-'*9}|{'-'*9}|{'-'*12}|")

        triggered_insights = []
        total_all = 0
        failed_all = 0

        for cat in CATEGORIES:
            label = CATEGORY_LABELS.get(cat, cat)
            total = totals.get(cat, 0)
            failed = cat_failures.get(cat, 0)
            pct = (failed / total * 100) if total > 0 else 0.0

            print(f"| {label:<25} | {total:>7} | {failed:>7} | {pct:>9.2f}% |")

            total_all += total
            failed_all += failed

            all_csv_rows.append({
                "Model": model,
                "TopK": top_k,
                "Category": label,
                "Total_Queries": total,
                "Failed_Queries": failed,
                "Failure_Pct": round(pct, 2),
            })

            if pct > INSIGHT_THRESHOLD:
                triggered_insights.append((cat, pct))

        # Aggregate row
        agg_pct = (failed_all / total_all * 100) if total_all > 0 else 0.0
        print(f"|{'-'*27}|{'-'*9}|{'-'*9}|{'-'*12}|")
        print(f"| {'TOTAL':<25} | {total_all:>7} | {failed_all:>7} | {agg_pct:>9.2f}% |")

        all_csv_rows.append({
            "Model": model,
            "TopK": top_k,
            "Category": "TOTAL",
            "Total_Queries": total_all,
            "Failed_Queries": failed_all,
            "Failure_Pct": round(agg_pct, 2),
        })

        # Expert Commentary
        if triggered_insights:
            print()
            print("  ⚠️  TECHNICAL INSIGHTS (categories exceeding "
                  f"{INSIGHT_THRESHOLD:.0f}% failure rate):")
            for cat, pct in triggered_insights:
                label = CATEGORY_LABELS.get(cat, cat)
                print(f"\n  [{label} — {pct:.1f}%]")
                print(f"  {INSIGHTS.get(cat, 'No insight available.')}")

    # ── CSV Export ────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "failure_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Model", "TopK", "Category", "Total_Queries",
                         "Failed_Queries", "Failure_Pct"],
        )
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print()
    print("=" * 80)
    print(f"✅ CSV exported: {csv_path}")
    print(f"   ({len(all_csv_rows)} rows: {len(model_k_combos)} configs × "
          f"{len(CATEGORIES) + 1} categories)")
    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    queries = load_json(QUERIES_PATH)
    errors = load_json(ERRORS_PATH)
    print(f"  Queries:  {len(queries)}")
    print(f"  Errors:   {len(errors)}")
    generate_report(queries, errors)


if __name__ == "__main__":
    main()
