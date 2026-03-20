"""
Fig 5 — Model Comparison: Accuracy vs Safety Refusal Rate (IEEE Standard)
==========================================================================
Standalone, publication-ready figure evaluating LLaMA-3 vs Mistral-7B on
generation accuracy and safety refusal behaviour across the full 36-config
matrix experiment.

Subplot 1: Accuracy distribution (boxplot + swarmplot)
Subplot 2: Mean safety refusal rate (error-bar chart)
Footer:    Paired t-test significance annotation

Input:  results/matrix_results.csv
Output: results/figures/fig_5_model_comparison.png (300 DPI)

Run:    conda run -n bns_rag python src/fig5_ModelComparison_AccuracyvsRefusal_Rate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CSV_PATH = os.path.join(RESULTS_DIR, "matrix_results.csv")
OUTPUT_PATH = os.path.join(FIG_DIR, "fig_5_model_comparison.png")

# ── Model filter & palette ────────────────────────────────────────────────
VALID_MODELS = ["llama3:latest", "mistral:7b-instruct"]
MODEL_PALETTE = {
    "llama3:latest":       "#42A5F5",   # Material Blue 400
    "mistral:7b-instruct": "#FF7043",   # Material Deep Orange 400
}
# Human-readable labels for axis ticks and legends
MODEL_LABELS = {
    "llama3:latest":       "LLaMA-3 (8B)",
    "mistral:7b-instruct": "Mistral (7B-Instruct)",
}


# ── IEEE academic styling ─────────────────────────────────────────────────
def setup_ieee_style():
    """Configure matplotlib/seaborn for IEEE two-column journal figures."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        # Fallback for older seaborn/mpl builds
        sns.set_theme(style="whitegrid")

    sns.set_theme(
        style="whitegrid",
        font_scale=1.1,
        rc={
            "figure.dpi":      150,
            "savefig.dpi":     300,
            "axes.edgecolor":  "0.25",
            "axes.linewidth":  0.8,
            "grid.color":      "0.92",
            "grid.linewidth":  0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
        },
    )
    # IEEE journals prefer serif fonts (Times New Roman / similar)
    plt.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
        "mathtext.fontset": "stix",         # STIX matches Times
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  9,
    })


# ── Data loading & cleaning ──────────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load matrix CSV, filter models, cast types."""
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  📄 Loaded {len(df)} rows from matrix_results.csv")

    # Strict model filter
    df["Model"] = df["Model"].astype(str).str.strip()
    df = df[df["Model"].isin(VALID_MODELS)].copy()
    if df.empty:
        print(f"❌ No rows match models {VALID_MODELS}. "
              f"Found: {pd.read_csv(csv_path)['Model'].unique().tolist()}",
              file=sys.stderr)
        sys.exit(1)

    # Explicit type casts
    df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce").fillna(0.0)
    df["Refusal_Rate"] = pd.to_numeric(df["Refusal_Rate"], errors="coerce").fillna(0.0)
    df["Model"] = pd.Categorical(df["Model"], categories=VALID_MODELS, ordered=True)

    print(f"  ✅ Filtered to {len(df)} rows  "
          f"({df['Model'].value_counts().to_dict()})")
    return df


# ── Statistical test ─────────────────────────────────────────────────────
def run_paired_ttest(df: pd.DataFrame) -> dict:
    """Paired t-test on Refusal_Rate between the two models.

    Pairs are matched on (Prompt, TopK, Temperature) so each configuration
    has exactly one observation per model.
    """
    sort_cols = ["Prompt", "TopK", "Temperature"]
    llama = (df[df["Model"] == "llama3:latest"]
             .sort_values(sort_cols)["Refusal_Rate"].values)
    mistral = (df[df["Model"] == "mistral:7b-instruct"]
               .sort_values(sort_cols)["Refusal_Rate"].values)

    result = {"valid": False, "t": None, "p": None, "sig": False, "label": ""}
    if len(llama) == len(mistral) and len(llama) > 1:
        t_stat, p_val = ttest_rel(llama, mistral)
        sig = p_val < 0.05
        if p_val < 0.001:
            label = "$p < 0.001$"
        elif p_val < 0.01:
            label = f"$p = {p_val:.3f}$"
        elif p_val < 0.05:
            label = f"$p = {p_val:.3f}$"
        else:
            label = f"$p = {p_val:.3f}$ (n.s.)"
        result.update(valid=True, t=t_stat, p=p_val, sig=sig, label=label)
        print(f"  📈 Paired t-test: t = {t_stat:.4f}, p = {p_val:.6f} "
              f"({'Significant' if sig else 'Not significant'})")
    else:
        print("  ⚠️  Cannot run paired t-test (unequal n or n ≤ 1)")
    return result


# ── Plotting ──────────────────────────────────────────────────────────────
def generate_figure(df: pd.DataFrame, ttest: dict):
    """Create the two-panel IEEE figure and save to disk."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)

    # ── Map internal model names → display labels for plotting ────
    label_map = MODEL_LABELS
    df_plot = df.copy()
    df_plot["Model_Label"] = df_plot["Model"].map(label_map)
    ordered_labels = [label_map[m] for m in VALID_MODELS]
    palette_labels = {label_map[k]: v for k, v in MODEL_PALETTE.items()}

    # ================================================================
    # Subplot 1: Accuracy Distribution (Boxplot + Swarmplot)
    # ================================================================
    ax1 = axes[0]
    sns.boxplot(
        data=df_plot, x="Model_Label", y="Accuracy", ax=ax1,
        hue="Model_Label", palette=palette_labels, legend=False,
        order=ordered_labels,
        width=0.50, linewidth=1.4, fliersize=3,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(edgecolor="0.3"),
        whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"),
    )
    sns.swarmplot(
        data=df_plot, x="Model_Label", y="Accuracy", ax=ax1,
        hue="Model_Label", palette=palette_labels, legend=False,
        order=ordered_labels,
        size=4.5, alpha=0.55, edgecolor="0.3", linewidth=0.4,
    )

    ax1.set_title("Accuracy Distribution", fontweight="bold", pad=10)
    ax1.set_xlabel("")
    ax1.set_ylabel("Generation Accuracy", fontsize=12)
    ax1.set_ylim(0.0, 1.0)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Annotate medians
    for i, model in enumerate(VALID_MODELS):
        med = df_plot[df_plot["Model"] == model]["Accuracy"].median()
        ax1.annotate(
            f"Md = {med:.3f}",
            xy=(i, med), xytext=(25, -8),
            textcoords="offset points", fontsize=9, fontstyle="italic",
            color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7),
        )

    # ================================================================
    # Subplot 2: Safety Refusal Rate (Error-Bar Chart)
    # ================================================================
    ax2 = axes[1]
    refusal_agg = (
        df_plot.groupby("Model_Label", observed=True)["Refusal_Rate"]
        .agg(["mean", "std"])
        .reindex(ordered_labels)
        .reset_index()
    )
    refusal_agg["pct_mean"] = refusal_agg["mean"] * 100
    refusal_agg["pct_std"]  = refusal_agg["std"]  * 100

    bar_colors = [palette_labels[m] for m in refusal_agg["Model_Label"]]
    bars = ax2.bar(
        refusal_agg["Model_Label"], refusal_agg["pct_mean"],
        yerr=refusal_agg["pct_std"], capsize=6,
        color=bar_colors, edgecolor="0.3", linewidth=0.8,
        error_kw=dict(elinewidth=1.2, capthick=1.2, color="0.3"),
        width=0.50,
    )

    # Percentage labels on bars
    for bar, mean_pct, std_pct in zip(bars, refusal_agg["pct_mean"],
                                       refusal_agg["pct_std"]):
        label_y = bar.get_height() + std_pct + 0.4
        ax2.text(
            bar.get_x() + bar.get_width() / 2, label_y,
            f"{mean_pct:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="0.15",
        )

    ax2.set_title("Safety Refusal Rate", fontweight="bold", pad=10)
    ax2.set_xlabel("")
    ax2.set_ylabel("Refusal Rate (%)", fontsize=12)

    # Dynamic y-axis: fit data + headroom for labels
    max_y = (refusal_agg["pct_mean"] + refusal_agg["pct_std"]).max()
    ax2.set_ylim(0, max(max_y * 1.45, 2.0))  # at least 2 % ceiling

    # ── Significance annotation ───────────────────────────────────
    if ttest["valid"] and ttest["sig"]:
        # Draw bracket between the two bars
        y_brack = max_y * 1.20
        x0, x1 = 0, 1
        ax2.plot([x0, x0, x1, x1],
                 [y_brack - 0.15, y_brack, y_brack, y_brack - 0.15],
                 lw=1.0, color="0.25")
        ax2.text(
            (x0 + x1) / 2, y_brack + 0.1, ttest["label"],
            ha="center", va="bottom", fontsize=10, color="0.15",
        )

    # ── Suptitle ──────────────────────────────────────────────────
    fig.suptitle(
        "Model Comparison: LLaMA-3 vs Mistral-7B",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # ── Footer: t-test summary ────────────────────────────────────
    if ttest["valid"]:
        footer = (f"Paired t-test (refusal rate): "
                  f"t = {ttest['t']:.4f}, {ttest['label']}")
        fig.text(
            0.5, -0.03, footer,
            ha="center", fontsize=9, fontstyle="italic", color="0.35",
        )

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_ieee_style()

    print("=" * 65)
    print("  FIG 5 — Model Comparison: Accuracy vs Refusal Rate (IEEE)")
    print("=" * 65)

    df = load_and_clean(CSV_PATH)
    ttest = run_paired_ttest(df)
    fig = generate_figure(df, ttest)

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"\n  ✅ Saved: {OUTPUT_PATH}  (300 DPI)")
    print("=" * 65)


if __name__ == "__main__":
    main()
