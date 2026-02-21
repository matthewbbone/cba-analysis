"""Compare GABRIEL vs Original healthcare generosity scores."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

SHARED_DIMS = ["score_employer_cost", "score_coverage_breadth", "score_dependent", "score_extras"]


def load_and_align():
    orig = pd.read_csv(os.path.join(SCRIPT_DIR, "healthcare_generosity_scores.csv"))
    gabriel = pd.read_csv(os.path.join(SCRIPT_DIR, "health_generosity_results.csv"))

    # Build employer lookup and short display names
    employer_map = orig.set_index("contract_id")["employer"].to_dict()

    # Keep only relevant columns
    orig_cols = ["contract_id"] + SHARED_DIMS + ["composite_score"]
    gabriel_cols = ["contract_id"] + SHARED_DIMS + ["composite_score"]

    orig = orig[orig_cols].copy()
    gabriel = gabriel[gabriel_cols].copy()

    # Rename for merge
    orig = orig.rename(columns={c: f"orig_{c}" for c in orig.columns if c != "contract_id"})
    gabriel = gabriel.rename(columns={c: f"gab_{c}" for c in gabriel.columns if c != "contract_id"})

    df = pd.merge(orig, gabriel, on="contract_id")

    # Normalize GABRIEL 0-100 scores to 1-5 scale
    for dim in SHARED_DIMS + ["composite_score"]:
        df[f"gab_{dim}_norm"] = df[f"gab_{dim}"] / 20.0

    # Rank each dimension individually (1 = most generous)
    for dim in SHARED_DIMS:
        df[f"orig_{dim}_rank"] = df[f"orig_{dim}"].rank(ascending=False, method="average").astype(float)
        df[f"gab_{dim}_rank"] = df[f"gab_{dim}"].rank(ascending=False, method="average").astype(float)

    # Composite rank = average of per-dimension ranks, then re-ranked
    orig_dim_ranks = [f"orig_{d}_rank" for d in SHARED_DIMS]
    gab_dim_ranks = [f"gab_{d}_rank" for d in SHARED_DIMS]
    df["orig_avg_dim_rank"] = df[orig_dim_ranks].mean(axis=1)
    df["gab_avg_dim_rank"] = df[gab_dim_ranks].mean(axis=1)
    df["orig_rank"] = df["orig_avg_dim_rank"].rank(method="min").astype(int)
    df["gab_rank"] = df["gab_avg_dim_rank"].rank(method="min").astype(int)

    # Add employer display name (shortened to first meaningful part)
    df["employer"] = df["contract_id"].map(employer_map)
    df["label"] = df["employer"].apply(_short_name)

    return df


def _short_name(name):
    """Shorten employer name for chart labels."""
    # Strip common suffixes/parentheticals
    name = name.split(",")[0].strip()
    name = name.split("(")[0].strip()
    # Truncate if still long
    if len(name) > 30:
        name = name[:27] + "..."
    return name


def print_comparison_table(df):
    print("\n" + "=" * 90)
    print("PER-DIMENSION RANKS (avg rank across dimensions -> composite rank)")
    print("=" * 90)

    dim_labels = [d.replace("score_", "") for d in SHARED_DIMS]

    # Per-dimension rank table
    print("\n--- Original per-dimension ranks ---")
    orig_rank_cols = ["label"] + [f"orig_{d}_rank" for d in SHARED_DIMS] + ["orig_avg_dim_rank", "orig_rank"]
    orig_table = df[orig_rank_cols].copy()
    orig_table.columns = ["Employer"] + dim_labels + ["Avg Rank", "Composite Rank"]
    orig_table = orig_table.sort_values("Composite Rank")
    print(orig_table.to_string(index=False, float_format="{:.1f}".format))

    print("\n--- GABRIEL per-dimension ranks ---")
    gab_rank_cols = ["label"] + [f"gab_{d}_rank" for d in SHARED_DIMS] + ["gab_avg_dim_rank", "gab_rank"]
    gab_table = df[gab_rank_cols].copy()
    gab_table.columns = ["Employer"] + dim_labels + ["Avg Rank", "Composite Rank"]
    gab_table = gab_table.sort_values("Composite Rank")
    print(gab_table.to_string(index=False, float_format="{:.1f}".format))

    print("\n--- Composite rank comparison ---")
    table = df[["label", "orig_avg_dim_rank", "gab_avg_dim_rank", "orig_rank", "gab_rank"]].copy()
    table.columns = ["Employer", "Orig Avg Rank", "GABRIEL Avg Rank", "Orig Rank", "GABRIEL Rank"]
    table["Rank Diff"] = table["Orig Rank"] - table["GABRIEL Rank"]
    table = table.sort_values("Orig Rank")
    print(table.to_string(index=False, float_format="{:.1f}".format))


def print_correlation_stats(df):
    print("\n" + "=" * 80)
    print("RANK CORRELATION STATISTICS")
    print("=" * 80)

    # Composite rank
    rho, p_rho = spearmanr(df["orig_rank"], df["gab_rank"])
    tau, p_tau = kendalltau(df["orig_rank"], df["gab_rank"])
    print(f"\n{'composite_rank':>25s}:  Spearman rho={rho:+.3f} (p={p_rho:.4f})  Kendall tau={tau:+.3f} (p={p_tau:.4f})")

    # Per dimension rank
    for dim in SHARED_DIMS:
        rho, p_rho = spearmanr(df[f"orig_{dim}_rank"], df[f"gab_{dim}_rank"])
        tau, p_tau = kendalltau(df[f"orig_{dim}_rank"], df[f"gab_{dim}_rank"])
        print(f"{dim:>25s}:  Spearman rho={rho:+.3f} (p={p_rho:.4f})  Kendall tau={tau:+.3f} (p={p_tau:.4f})")


def plot_rank_comparison(df):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["orig_rank"], df["gab_rank"], s=60, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["label"], (row["orig_rank"], row["gab_rank"]),
                     fontsize=7, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    lims = [0.5, len(df) + 0.5]
    ax.plot(lims, lims, "--", color="grey", alpha=0.6, label="Perfect agreement")
    r, _ = pearsonr(df["orig_rank"], df["gab_rank"])
    rho, _ = spearmanr(df["orig_rank"], df["gab_rank"])
    ax.annotate(f"Pearson r = {r:.3f}\nSpearman rho = {rho:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=10, va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.set_xlabel("Original Rank")
    ax.set_ylabel("GABRIEL Rank")
    ax.set_title("Rank Comparison: Original vs GABRIEL")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.invert_yaxis()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rank_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved rank_comparison.png")


def plot_composite_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["orig_composite_score"], df["gab_composite_score_norm"], s=60, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["label"], (row["orig_composite_score"], row["gab_composite_score_norm"]),
                     fontsize=7, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    # Identity line
    lo = min(df["orig_composite_score"].min(), df["gab_composite_score_norm"].min()) - 0.2
    hi = max(df["orig_composite_score"].max(), df["gab_composite_score_norm"].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.6)
    r, p_r = pearsonr(df["orig_composite_score"], df["gab_composite_score_norm"])
    rho, _ = spearmanr(df["orig_composite_score"], df["gab_composite_score_norm"])
    ax.annotate(f"Pearson r = {r:.3f}\nSpearman rho = {rho:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=10, va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.set_xlabel("Original Composite (1-5)")
    ax.set_ylabel("GABRIEL Composite (normalized 1-5)")
    ax.set_title("Composite Score Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "composite_scatter.png"), dpi=150)
    plt.close(fig)
    print("  Saved composite_scatter.png")


def plot_dimension_heatmaps(df):
    dims = SHARED_DIMS
    df_sorted = df.sort_values("orig_rank")
    labels = df_sorted["label"].values

    orig_data = df_sorted[[f"orig_{d}_rank" for d in dims]].values
    gab_data = df_sorted[[f"gab_{d}_rank" for d in dims]].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    dim_labels = [d.replace("score_", "") for d in dims]
    n = len(df)
    # Reverse colormap: rank 1 (best) = dark/hot, rank 20 (worst) = light/cool
    cmap = plt.cm.YlOrRd_r

    im1 = ax1.imshow(orig_data, aspect="auto", cmap=cmap, vmin=1, vmax=n)
    ax1.set_xticks(range(len(dims)))
    ax1.set_xticklabels(dim_labels, rotation=45, ha="right")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_title("Original Ranks")

    im2 = ax2.imshow(gab_data, aspect="auto", cmap=cmap, vmin=1, vmax=n)
    ax2.set_xticks(range(len(dims)))
    ax2.set_xticklabels(dim_labels, rotation=45, ha="right")
    ax2.set_title("GABRIEL Ranks")

    fig.suptitle("Per-Dimension Rank Heatmaps (1 = best)", fontsize=14)
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.22, right=0.82)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax, label="Rank (1 = most generous)")
    fig.savefig(os.path.join(PLOTS_DIR, "dimension_heatmaps.png"), dpi=150)
    plt.close(fig)
    print("  Saved dimension_heatmaps.png")


def plot_rank_difference(df):
    df_sorted = df.sort_values("orig_rank")
    rank_diff = df_sorted["orig_rank"] - df_sorted["gab_rank"]
    labels = df_sorted["label"].values

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2ecc71" if d > 0 else "#e74c3c" if d < 0 else "#95a5a6" for d in rank_diff]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, rank_diff, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Rank Displacement (positive = GABRIEL ranked higher)")
    ax.set_title("Rank Difference: Original Rank - GABRIEL Rank")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rank_difference.png"), dpi=150)
    plt.close(fig)
    print("  Saved rank_difference.png")


def main():
    df = load_and_align()
    print(f"Loaded {len(df)} CBAs with matched scores.\n")

    print_comparison_table(df)
    print_correlation_stats(df)

    print("\nGenerating charts...")
    plot_rank_comparison(df)
    plot_composite_scatter(df)
    plot_dimension_heatmaps(df)
    plot_rank_difference(df)

    print(f"\nAll charts saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
