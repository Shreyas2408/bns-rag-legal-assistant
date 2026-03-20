"""
BNS Publication Figures
=========================
Generates 5 publication-ready Seaborn figures from experiment results.

Fig 1: Retrieval System Comparison (Dense vs Hybrid vs Reranked)
Fig 2: Alpha Tuning Sensitivity (α vs Accuracy@5)
Fig 3: Prompt Template Comparison (P1 vs P2 vs P3)
Fig 4: Category Difficulty Heatmap (sensitive/criminal categories)
Fig 5: Model Comparison — Accuracy vs Safety Refusal Rate

Input: results/*.csv from prior experiments + matrix_results.csv
Output: results/figures/fig_1..5.png

Environment: conda run -n bns_rag python src/generate_figures.py
"""

import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  ⚠️  {name} not found, skipping dependent figure.")
        return None
    return pd.read_csv(path)


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.15,
                  rc={"figure.dpi": 150, "savefig.dpi": 150,
                      "axes.edgecolor": ".3", "grid.color": ".9"})
    plt.rcParams["font.family"] = "sans-serif"


# ── Fig 1: Retrieval System Comparison ────────────────────────────────────

def fig1_retrieval_comparison():
    df = load_csv("final_benchmark_results.csv")
    if df is None:
        return
    print("  📊 Fig 1: Retrieval System Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, metric in enumerate(["Accuracy", "MRR"]):
        ax = axes[i]
        data = df.pivot(index="TopK", columns="System", values=metric)
        cols_order = ["Dense (MPNet)", "Hybrid (MPNet+BM25)", "Hybrid + CrossEncoder"]
        cols_present = [c for c in cols_order if c in data.columns]
        data = data[cols_present]

        colors = ["#78909C", "#42A5F5", "#AB47BC"][:len(cols_present)]
        data.plot(kind="bar", ax=ax, color=colors, edgecolor="white", linewidth=1.2,
                  width=0.7)
        ax.set_title(f"{metric} by System", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Top-K", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_ylim(0, 1.0)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=3)

    plt.suptitle("Retrieval System Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_1_retrieval_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("    ✅ Saved")


# ── Fig 2: Alpha Sensitivity ─────────────────────────────────────────────

def fig2_alpha_tuning():
    df = load_csv("alpha_sensitivity.csv")
    if df is None:
        return
    print("  📊 Fig 2: Alpha Tuning Sensitivity")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for metric, color, marker in [("Accuracy@3", "#EF5350", "s"),
                                   ("Accuracy@5", "#42A5F5", "o"),
                                   ("MRR@5", "#66BB6A", "D")]:
        ax.plot(df["Alpha"], df[metric], f"{marker}-", color=color,
                linewidth=2.5, markersize=10, label=metric)
        for x, y in zip(df["Alpha"], df[metric]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Alpha (α) — Dense Weight", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Alpha Sensitivity: Hybrid Retrieval", fontsize=14, fontweight="bold")
    ax.set_xticks(df["Alpha"].values)
    ax.legend(fontsize=11)
    ax.set_ylim(0.45, 0.80)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_2_alpha_sensitivity.png"), bbox_inches="tight")
    plt.close(fig)
    print("    ✅ Saved")


# ── Fig 3: Prompt Comparison ─────────────────────────────────────────────

def fig3_prompt_comparison():
    df = load_csv("generation_results.csv")
    if df is None:
        return
    print("  📊 Fig 3: Prompt Template Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Overall accuracy + latency
    ax1 = axes[0]
    colors = ["#42A5F5", "#AB47BC", "#FF7043"]
    bars = ax1.bar(df["Template"], df["Accuracy"], color=colors, edgecolor="white",
                   linewidth=1.2)
    ax1.set_ylabel("Generation Accuracy", fontsize=12)
    ax1.set_title("Overall Accuracy by Template", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 0.8)
    for bar, v in zip(bars, df["Accuracy"]):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax1b = ax1.twinx()
    ax1b.plot(df["Template"], df["Avg_Latency_s"], "D--", color="#EF5350",
              markersize=8, linewidth=2)
    ax1b.set_ylabel("Avg Latency (s)", fontsize=12, color="#EF5350")
    ax1b.tick_params(axis="y", labelcolor="#EF5350")

    # Right: Category breakdown
    ax2 = axes[1]
    cat_cols = [c for c in df.columns if c.startswith("Cat_")]
    cat_names = [c.replace("Cat_", "") for c in cat_cols]
    x = np.arange(len(cat_names))
    w = 0.25
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[c] * 100 for c in cat_cols]
        ax2.bar(x + i * w, vals, w, label=row["Template"], color=colors[i],
                edgecolor="white")
    ax2.set_xlabel("Category", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Category Accuracy by Template", fontsize=13, fontweight="bold")
    ax2.set_xticks(x + w)
    ax2.set_xticklabels(cat_names, fontsize=10, rotation=15)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 100)

    plt.suptitle("Prompt Engineering Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_3_prompt_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("    ✅ Saved")


# ── Fig 4: Category Difficulty Heatmap ────────────────────────────────────

def fig4_category_difficulty():
    df = load_csv("matrix_results.csv")
    if df is None:
        return
    print("  📊 Fig 4: Category Difficulty Heatmap")

    cat_cols = [c for c in df.columns if c.startswith("Cat_")]
    cat_names = [c.replace("Cat_", "") for c in cat_cols]

    # Average across all configs per category
    pivot_data = []
    for _, row in df.iterrows():
        label = f"{row['Model']}|{row['Prompt']}"
        vals = {cat_names[i]: row[c] for i, c in enumerate(cat_cols)}
        vals["Config"] = label
        pivot_data.append(vals)

    pdf = pd.DataFrame(pivot_data)
    # Aggregate by model for cleaner heatmap
    agg = df.groupby("Model")[cat_cols].mean()
    agg.columns = cat_names

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(agg, annot=True, fmt=".3f", cmap="RdYlGn", linewidths=1,
                ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Accuracy"})
    ax.set_title("Category Difficulty by Model (Avg. across configs)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_4_category_difficulty.png"), bbox_inches="tight")
    plt.close(fig)
    print("    ✅ Saved")


# ── Fig 5: Model Comparison — Accuracy vs Refusal Rate ────────────────────

def fig5_model_comparison():
    df = load_csv("matrix_results.csv")
    if df is None:
        return
    print("  📊 Fig 5: Model Comparison — Accuracy vs Refusal Rate")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Accuracy distribution
    ax1 = axes[0]
    palette = {"llama3:latest": "#42A5F5", "mistral:7b-instruct": "#FF7043"}
    sns.boxplot(data=df, x="Model", y="Accuracy", ax=ax1, palette=palette,
                width=0.5, linewidth=1.5)
    sns.stripplot(data=df, x="Model", y="Accuracy", ax=ax1, color=".3",
                  size=4, alpha=0.5, jitter=True)
    ax1.set_title("Accuracy Distribution", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 0.9)

    # Right: Refusal rate
    ax2 = axes[1]
    refusal_agg = df.groupby("Model")["Refusal_Rate"].agg(["mean", "std"]).reset_index()
    bars = ax2.bar(refusal_agg["Model"], refusal_agg["mean"] * 100,
                   yerr=refusal_agg["std"] * 100, capsize=5,
                   color=[palette.get(m, "#999") for m in refusal_agg["Model"]],
                   edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, refusal_agg["mean"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("Safety Refusal Rate", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Refusal Rate (%)", fontsize=12)

    plt.suptitle("Model Comparison: LLaMA-3 vs Mistral", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_5_model_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # Statistical test: Refusal rate comparison
    llama_df = df[df["Model"] == "llama3"].sort_values(["Prompt", "TopK", "Temperature"])
    mistral_df = df[df["Model"] == "mistral"].sort_values(["Prompt", "TopK", "Temperature"])
    if len(llama_df) == len(mistral_df) and len(llama_df) > 1:
        t, p = ttest_rel(llama_df["Refusal_Rate"].values,
                         mistral_df["Refusal_Rate"].values)
        sig = "Significant" if p < 0.05 else "Not significant"
        print(f"    📈 Refusal t-test: t={t:.4f}, p={p:.6f} ({sig})")
    print("    ✅ Saved")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_style()

    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    fig1_retrieval_comparison()
    fig2_alpha_tuning()
    fig3_prompt_comparison()
    fig4_category_difficulty()
    fig5_model_comparison()

    print(f"\n✅ All figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
