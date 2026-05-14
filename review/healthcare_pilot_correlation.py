import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.utils.llm import model_slug


MODEL_NAME = "gpt-5.4-nano"


SOURCE_MAP = {
    "Cornell-DoL": "cornell_dol",
    "Cornell-RetailEducation": "cornell_retail_educ",
    "DOL Archive": "dol_archive",
}


def document_id_from_pilot_row(row):
    source = SOURCE_MAP.get(row["source"], row["source"])
    file_stem = Path(row["file_path"]).stem
    return f"{source}/{file_stem}"


def load_matched_scores(pilot_scores_path, bradley_terry_path):
    with bradley_terry_path.open("r", encoding="utf-8") as f:
        bt_results = json.load(f)

    healthcare_rankings = bt_results["rankings_by_category"]["Healthcare"]
    rows = []

    with pilot_scores_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = document_id_from_pilot_row(row)
            ranking = healthcare_rankings.get(doc_id)
            if not ranking or ranking.get("log_strength") is None:
                continue
            rows.append(
                {
                    "document_id": doc_id,
                    "employer": row.get("employer", ""),
                    "pilot_score": float(row["composite_score"]),
                    "healthcare_log_strength": float(ranking["log_strength"]),
                }
            )

    if not rows:
        raise ValueError("No matched pilot scores with non-null Healthcare log strengths.")

    return rows


def plot_correlation(rows, output_path):
    pilot_scores = np.array([row["pilot_score"] for row in rows], dtype=float)
    healthcare_scores = np.array(
        [row["healthcare_log_strength"] for row in rows], dtype=float
    )

    pearson_r, pearson_p = pearsonr(pilot_scores, healthcare_scores)
    spearman_rho, spearman_p = spearmanr(pilot_scores, healthcare_scores)

    slope, intercept = np.polyfit(pilot_scores, healthcare_scores, 1)
    fitted_scores = slope * pilot_scores + intercept
    residuals = healthcare_scores - fitted_scores
    line_x = np.linspace(float(np.min(pilot_scores)), float(np.max(pilot_scores)), 100)
    line_y = slope * line_x + intercept

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pilot_scores, healthcare_scores, s=45, alpha=0.8)
    ax.plot(line_x, line_y, color="black", linewidth=1.5, label="Best fit")

    ax.set_xlabel("Pilot healthcare composite score")
    ax.set_ylabel("Pipeline Healthcare log strength")
    ax.set_title(
        "Pilot vs Pipeline Healthcare Generosity\n"
        f"n={len(rows)} | Pearson r={pearson_r:.2f} (p={pearson_p:.3g}) | "
        f"Spearman rho={spearman_rho:.2f} (p={spearman_p:.3g})"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Matched observations: {len(rows)}")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman rho={spearman_rho:.4f}, p={spearman_p:.4g}")
    print("\nLargest outliers from linear fit:")
    outlier_order = np.argsort(np.abs(residuals))[::-1]
    for rank, index in enumerate(outlier_order[:5], start=1):
        row = rows[index]
        residual = float(residuals[index])
        direction = "higher" if residual > 0 else "lower"
        label = row["employer"] or row["document_id"]
        print(
            f"{rank}. {label} ({row['document_id']}): "
            f"pilot={pilot_scores[index]:.2f}, "
            f"log_strength={healthcare_scores[index]:.3f}, "
            f"residual={residual:+.3f} ({direction} than fit)"
        )
    print(f"Saved figure to {output_path}")


def main():
    rows = load_matched_scores(
        Path("references/pilot_healthcare_scores.csv"),
        Path("cache/06_generosity_output")
        / model_slug(MODEL_NAME)
        / "bradley_terry_results.json",
    )
    plot_correlation(rows, Path("figures/healthcare_pilot_correlation.png"))


if __name__ == "__main__":
    main()
