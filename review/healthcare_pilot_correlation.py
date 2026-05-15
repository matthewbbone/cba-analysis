import argparse
import csv
import json
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.utils.llm import model_slug


MODEL_NAME = "gpt-5.4-nano"


def load_elo_calculator():
    calculator_path = (
        Path(__file__).resolve().parents[1]
        / "pipeline"
        / "06_generosity"
        / "elo_calculator.py"
    )
    spec = importlib.util.spec_from_file_location("elo_calculator", calculator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE_MAP = {
    "Cornell-DoL": "cornell_dol",
    "Cornell-RetailEducation": "cornell_retail_educ",
    "DOL Archive": "dol_archive",
}


def document_id_from_pilot_row(row):
    source = SOURCE_MAP.get(row["source"], row["source"])
    file_stem = Path(row["file_path"]).stem
    return f"{source}/{file_stem}_res"


def load_scalar_scores(scalar_scores_path):
    with scalar_scores_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {
            row["Doc"]: {
                "scalar_score": float(row["Scalar 1-5"]),
                "scalar_elo_rank": float(row["ELO rank"]),
            }
            for row in reader
            if row.get("Doc") and row.get("Scalar 1-5")
        }


def load_matched_scores(
    pilot_scores_path,
    scalar_scores_path,
    comparison_results_path,
    method,
):
    with comparison_results_path.open("r", encoding="utf-8") as f:
        comparison_results = json.load(f)

    elo_calculator = load_elo_calculator()
    fitted_results = elo_calculator.build_output(comparison_results, method)
    healthcare_rankings = fitted_results["rankings_by_category"]["Healthcare"]
    scalar_scores = load_scalar_scores(scalar_scores_path)
    rows = []

    with pilot_scores_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = document_id_from_pilot_row(row)
            ranking = healthcare_rankings.get(doc_id)
            scalar = scalar_scores.get(row["contract_id"])
            if not ranking or ranking.get("log_strength") is None:
                continue
            if not scalar:
                continue
            rows.append(
                {
                    "contract_id": row["contract_id"],
                    "document_id": doc_id,
                    "employer": row.get("employer", ""),
                    "pilot_score": float(row["composite_score"]),
                    "scalar_score": scalar["scalar_score"],
                    "scalar_elo_rank": scalar["scalar_elo_rank"],
                    "healthcare_log_strength": float(ranking["log_strength"]),
                }
            )

    if not rows:
        raise ValueError(
            "No matched Claude Pilot, Codex Pilot, and non-null ELO Generosity scores."
        )

    z_normalize_rows(rows, "healthcare_log_strength")
    return rows


def z_normalize_rows(rows, key):
    values = [row[key] for row in rows]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_dev = variance ** 0.5
    for row in rows:
        row[key] = (row[key] - mean_value) / std_dev if std_dev else 0.0


def add_scatter_panel(ax, rows, x_key, y_key):
    x = np.array([row[x_key] for row in rows], dtype=float)
    y = np.array([row[y_key] for row in rows], dtype=float)

    ax.scatter(x, y, s=38, alpha=0.8)
    if len(rows) >= 2 and len(set(x)) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color="black", linewidth=1.2)
    ax.grid(alpha=0.25)


def add_correlation_panel(ax, rows, x_key, y_key):
    x = np.array([row[x_key] for row in rows], dtype=float)
    y = np.array([row[y_key] for row in rows], dtype=float)
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.25)
    ax.text(
        0.5,
        0.60,
        f"Pearson r={pearson_r:.2f}\np={pearson_p:.3g}",
        ha="center",
        va="center",
        fontsize=16,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.36,
        f"Spearman rho={spearman_rho:.2f}\np={spearman_p:.3g}",
        ha="center",
        va="center",
        fontsize=16,
        transform=ax.transAxes,
    )
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def print_rank_disagreements(rows, left_key, right_key, left_label, right_label):
    left_scores = np.array([row[left_key] for row in rows], dtype=float)
    right_scores = np.array([row[right_key] for row in rows], dtype=float)
    left_ranks = rankdata(-left_scores, method="average")
    right_ranks = rankdata(-right_scores, method="average")
    rank_disagreements = right_ranks - left_ranks

    print(f"\nLargest rank-order disagreements: {left_label} vs {right_label}")
    outlier_order = np.argsort(np.abs(rank_disagreements))[::-1]
    for rank, index in enumerate(outlier_order[:5], start=1):
        row = rows[index]
        rank_gap = float(rank_disagreements[index])
        direction = "lower" if rank_gap > 0 else "higher"
        label = row["employer"] or row["document_id"]
        print(
            f"{rank}. {label} ({row['contract_id']}, {row['document_id']}): "
            f"{left_label}={left_scores[index]:.3f}, "
            f"{right_label}={right_scores[index]:.3f}, "
            f"{left_label}_rank={left_ranks[index]:.1f}, "
            f"{right_label}_rank={right_ranks[index]:.1f}, "
            f"rank_gap={rank_gap:+.1f} ({right_label} ranks it {direction})"
        )


def plot_correlation_grid(rows, output_path):
    variables = [
        ("Claude Pilot", "pilot_score"),
        ("Codex Pilot", "scalar_score"),
        ("ELO Generosity", "healthcare_log_strength"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    correlations = {}

    for row_index, (row_label, row_key) in enumerate(variables):
        for col_index, (col_label, col_key) in enumerate(variables):
            ax = axes[row_index, col_index]

            if row_index == col_index:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_alpha(0.25)
                ax.text(
                    0.5,
                    0.5,
                    row_label,
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    transform=ax.transAxes,
                )
            elif row_index > col_index:
                add_scatter_panel(ax, rows, col_key, row_key)
            else:
                correlations[f"{col_label}_vs_{row_label}"] = add_correlation_panel(
                    ax,
                    rows,
                    col_key,
                    row_key,
                )

            if row_index != len(variables) - 1:
                ax.set_xticklabels([])
            if col_index != 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Matched observations: {len(rows)}")
    for label, values in correlations.items():
        print(
            f"{label}: Pearson r={values['pearson_r']:.4f}, "
            f"p={values['pearson_p']:.4g}; "
            f"Spearman rho={values['spearman_rho']:.4f}, "
            f"p={values['spearman_p']:.4g}"
        )

    print_rank_disagreements(
        rows,
        "pilot_score",
        "healthcare_log_strength",
        "Claude Pilot",
        "ELO Generosity",
    )
    print_rank_disagreements(
        rows,
        "scalar_score",
        "healthcare_log_strength",
        "Codex Pilot",
        "ELO Generosity",
    )
    print(
        "\nScore ranges: "
        + "; ".join(
            f"{label} {min(row[key] for row in rows):.2f}-{max(row[key] for row in rows):.2f}"
            for label, key in variables
        )
    )
    print(f"Saved figure to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare pilot healthcare scores against fitted ELO-style generosity scores."
    )
    parser.add_argument(
        "--method",
        choices=["bradley-terry", "davidson"],
        default="davidson",
        help="Ranking model to fit before plotting.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Model cache name used to locate default input paths.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to runner output JSON with saved comparisons.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/healthcare_pilot_correlation.png"),
        help="Path to write the correlation figure.",
    )
    parser.add_argument(
        "--pilot-scores",
        type=Path,
        default=Path("references/pilot_healthcare_scores.csv"),
    )
    parser.add_argument(
        "--scalar-scores",
        type=Path,
        default=Path("references/doc_scalar_elo_rank.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_model = model_slug(args.model_name)
    input_path = args.input or (
        Path("cache/06_generosity_output")
        / cache_model
        / "bradley_terry_results.json"
    )
    rows = load_matched_scores(
        args.pilot_scores,
        args.scalar_scores,
        input_path,
        args.method,
    )
    plot_correlation_grid(rows, args.output)


if __name__ == "__main__":
    main()
