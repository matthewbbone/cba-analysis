from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DEFAULT_PILOT_SCORES = Path("references/cba_generosity_scores_100doc_pilot.csv")
DEFAULT_DAVIDSON_RANKINGS = Path(
    "cache/06_generosity_output/gpt_5_4_nano/davidson_rankings.json"
)
DEFAULT_LINKED_OUTPUT = Path(
    "review/cba_generosity_scores_100doc_pilot_linked_davidson.csv"
)
DEFAULT_CORRELATION_OUTPUT = Path(
    "review/cba_generosity_pilot_davidson_category_correlations.csv"
)
DEFAULT_FIGURE_OUTPUT = Path(
    "figures/cba_generosity_pilot_davidson_category_correlations.png"
)


def normalize_pilot_document_id(document_id: object) -> str | None:
    if document_id is None or pd.isna(document_id):
        return None

    raw = str(document_id).strip()
    if not raw:
        return None

    if raw.startswith("DOL_"):
        return f"dol_archive/document_{raw.removeprefix('DOL_')}_res"

    if "/" in raw:
        return raw

    return raw


def load_davidson_rankings(rankings_path: Path) -> pd.DataFrame:
    if not rankings_path.exists():
        raise FileNotFoundError(f"Davidson rankings file does not exist: {rankings_path}")

    with rankings_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rankings_by_category = payload.get("rankings_by_category") or {}
    rows: list[dict[str, Any]] = []
    for category, rankings in rankings_by_category.items():
        for document_id, scores in rankings.items():
            rows.append(
                {
                    "normalized_document_id": document_id,
                    "provision_category": category,
                    "davidson_strength": scores.get("strength"),
                    "davidson_log_strength": scores.get("log_strength"),
                    "davidson_wins": scores.get("wins"),
                    "davidson_losses": scores.get("losses"),
                    "davidson_ties": scores.get("ties"),
                    "davidson_decisive_comparisons": scores.get(
                        "decisive_comparisons"
                    ),
                    "davidson_total_comparisons": scores.get("total_comparisons"),
                }
            )

    return pd.DataFrame(rows)


def z_score_by_category(
    df: pd.DataFrame,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[value_column] = pd.to_numeric(out[value_column], errors="coerce")

    def z_score(values: pd.Series) -> pd.Series:
        mean = values.mean(skipna=True)
        std = values.std(skipna=True, ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=values.index)
        return (values - mean) / std

    out[output_column] = out.groupby("provision_category", dropna=False)[
        value_column
    ].transform(z_score)
    return out


def load_linked_scores(
    pilot_scores_path: Path,
    davidson_rankings_path: Path,
) -> pd.DataFrame:
    if not pilot_scores_path.exists():
        raise FileNotFoundError(f"Pilot score file does not exist: {pilot_scores_path}")

    pilot = pd.read_csv(pilot_scores_path)
    pilot["document_id"] = pilot["document_id"].map(
        lambda value: str(value).strip() if not pd.isna(value) else value
    )
    pilot["normalized_document_id"] = pilot["document_id"].map(
        normalize_pilot_document_id
    )
    pilot["generosity_score_0_1"] = pd.to_numeric(
        pilot["generosity_score_0_1"],
        errors="coerce",
    )
    pilot["score_uncalibrated_0_1"] = pd.to_numeric(
        pilot["score_uncalibrated_0_1"],
        errors="coerce",
    )

    rankings = load_davidson_rankings(davidson_rankings_path)
    linked = pilot.merge(
        rankings,
        on=["normalized_document_id", "provision_category"],
        how="inner",
    )
    linked = z_score_by_category(
        linked,
        value_column="davidson_log_strength",
        output_column="davidson_log_strength_z",
    )
    return linked.sort_values(
        ["provision_category", "document_id"],
        kind="stable",
    ).reset_index(drop=True)


def correlation_stats(
    rows: pd.DataFrame,
    x_column: str = "generosity_score_0_1",
    y_column: str = "davidson_log_strength_z",
) -> dict[str, float | int | None]:
    paired = rows[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(paired) < 2 or paired[x_column].nunique() < 2 or paired[y_column].nunique() < 2:
        return {
            "n": int(len(paired)),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }

    pearson_r, pearson_p = pearsonr(paired[x_column], paired[y_column])
    spearman_rho, spearman_p = spearmanr(paired[x_column], paired[y_column])
    return {
        "n": int(len(paired)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def build_correlation_summary(linked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"provision_category": "All categories", **correlation_stats(linked)})

    for category in sorted(linked["provision_category"].dropna().unique()):
        category_rows = linked[linked["provision_category"] == category]
        rows.append(
            {
                "provision_category": category,
                **correlation_stats(category_rows),
            }
        )

    return pd.DataFrame(rows)


def add_scatter_with_fit(
    ax: plt.Axes,
    rows: pd.DataFrame,
    category: str,
    stats: dict[str, Any],
) -> None:
    x = pd.to_numeric(rows["generosity_score_0_1"], errors="coerce")
    y = pd.to_numeric(rows["davidson_log_strength_z"], errors="coerce")
    paired = pd.DataFrame({"x": x, "y": y}).dropna()

    ax.scatter(paired["x"], paired["y"], s=28, alpha=0.75)
    if len(paired) >= 2 and paired["x"].nunique() > 1:
        slope, intercept = np.polyfit(paired["x"], paired["y"], 1)
        line_x = np.linspace(float(paired["x"].min()), float(paired["x"].max()), 100)
        ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.2)

    ax.set_title(category, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.02, 1.02)

    pearson = stats.get("pearson_r")
    spearman = stats.get("spearman_rho")
    text = (
        f"n={stats.get('n', 0)}\n"
        f"Pearson r={format_stat(pearson)}\n"
        f"Spearman rho={format_stat(spearman)}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )


def format_stat(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def plot_category_correlations(
    linked: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    categories = sorted(linked["provision_category"].dropna().unique())
    if not categories:
        raise ValueError("No linked category scores to plot.")

    n_cols = 3
    n_rows = ceil(len(categories) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    stats_by_category = correlation_summary.set_index("provision_category").to_dict(
        orient="index"
    )
    for index, category in enumerate(categories):
        ax = axes[index // n_cols][index % n_cols]
        add_scatter_with_fit(
            ax,
            linked[linked["provision_category"] == category],
            category,
            stats_by_category.get(category, {}),
        )

    for index in range(len(categories), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    for ax in axes[-1]:
        ax.set_xlabel("Pilot generosity score (0-1)")
    for row_axes in axes:
        row_axes[0].set_ylabel("Davidson category log-strength z-score")

    fig.suptitle(
        "Pilot Category Scores vs Davidson Category-Specific Generosity",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_analysis(
    pilot_scores_path: Path = DEFAULT_PILOT_SCORES,
    davidson_rankings_path: Path = DEFAULT_DAVIDSON_RANKINGS,
    linked_output: Path = DEFAULT_LINKED_OUTPUT,
    correlation_output: Path = DEFAULT_CORRELATION_OUTPUT,
    figure_output: Path = DEFAULT_FIGURE_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    linked = load_linked_scores(pilot_scores_path, davidson_rankings_path)
    if linked.empty:
        raise ValueError("No matched document/category scores after linking inputs.")

    summary = build_correlation_summary(linked)

    linked_output.parent.mkdir(parents=True, exist_ok=True)
    correlation_output.parent.mkdir(parents=True, exist_ok=True)
    linked.to_csv(linked_output, index=False)
    summary.to_csv(correlation_output, index=False)
    plot_category_correlations(linked, summary, figure_output)

    return linked, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Link pilot category generosity scores to Davidson category rankings "
            "and plot category-level correlations."
        )
    )
    parser.add_argument("--pilot-scores", type=Path, default=DEFAULT_PILOT_SCORES)
    parser.add_argument(
        "--davidson-rankings",
        type=Path,
        default=DEFAULT_DAVIDSON_RANKINGS,
        help="Path to davidson_rankings.json.",
    )
    parser.add_argument("--linked-output", type=Path, default=DEFAULT_LINKED_OUTPUT)
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=DEFAULT_CORRELATION_OUTPUT,
    )
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        linked, summary = run_analysis(
            pilot_scores_path=args.pilot_scores,
            davidson_rankings_path=args.davidson_rankings,
            linked_output=args.linked_output,
            correlation_output=args.correlation_output,
            figure_output=args.figure_output,
        )
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Linked rows: {len(linked)}")
    print(f"Documents: {linked['document_id'].nunique()}")
    print(f"Categories: {linked['provision_category'].nunique()}")
    print(f"Wrote linked scores to {args.linked_output}.")
    print(f"Wrote correlation summary to {args.correlation_output}.")
    print(f"Wrote scatter plot to {args.figure_output}.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
