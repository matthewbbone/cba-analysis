import argparse
import json
from pathlib import Path
import sys
import csv
import importlib.util

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

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


def load_results(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def short_label(document_id, max_length=32):
    label = document_id.split("/", 1)[-1].replace("_", " ")
    if len(label) <= max_length:
        return label
    return label[: max_length - 3] + "..."


def document_id_from_pilot_row(row):
    source = SOURCE_MAP.get(row["source"], row["source"])
    file_stem = Path(row["file_path"]).stem
    return f"{source}/{file_stem}_res"


def load_contract_id_map(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {
            document_id_from_pilot_row(row): row["contract_id"]
            for row in reader
            if row.get("contract_id") and row.get("file_path")
        }


def z_scores(values_by_doc):
    valid_values = [
        value
        for value in values_by_doc.values()
        if value is not None
    ]
    if not valid_values:
        return {doc_id: None for doc_id in values_by_doc}

    mean_value = sum(valid_values) / len(valid_values)
    variance = sum((value - mean_value) ** 2 for value in valid_values) / len(valid_values)
    std_dev = variance ** 0.5
    if std_dev == 0:
        return {
            doc_id: 0.0 if value is not None else None
            for doc_id, value in values_by_doc.items()
        }

    return {
        doc_id: (
            (value - mean_value) / std_dev
            if value is not None
            else None
        )
        for doc_id, value in values_by_doc.items()
    }


def build_plot_data(results):
    documents = [item["document_id"] for item in results["documents"]]
    categories = results["categories"]

    category_scores = {}
    for category in categories:
        log_strengths = {
            doc_id: results["rankings_by_category"][category][doc_id]["log_strength"]
            for doc_id in documents
        }
        category_scores[category] = z_scores(log_strengths)

    document_scores = {}
    document_variances = {}
    for doc_id in documents:
        scores = [
            category_scores[category][doc_id]
            for category in categories
            if category_scores[category][doc_id] is not None
        ]
        if not scores:
            document_scores[doc_id] = None
            document_variances[doc_id] = None
            continue

        mean_score = sum(scores) / len(scores)
        document_scores[doc_id] = mean_score
        document_variances[doc_id] = (
            sum((score - mean_score) ** 2 for score in scores) / len(scores)
        )

    return documents, categories, category_scores, document_scores, document_variances


def print_variance_score_correlation(document_scores, document_variances):
    paired = [
        (document_scores[doc_id], document_variances[doc_id])
        for doc_id in document_scores
        if document_scores[doc_id] is not None
        and document_variances[doc_id] is not None
    ]
    if len(paired) < 2:
        print("Not enough documents to correlate category variance with composite score.")
        return

    composite_scores = [score for score, _ in paired]
    variances = [variance for _, variance in paired]
    pearson_r, pearson_p = pearsonr(composite_scores, variances)
    spearman_rho, spearman_p = spearmanr(composite_scores, variances)
    print(
        "Category-score variance vs composite score: "
        f"Pearson r={pearson_r:.4f}, p={pearson_p:.4g}; "
        f"Spearman rho={spearman_rho:.4f}, p={spearman_p:.4g}"
    )


def select_display_documents(document_scores, n_top=5, n_bottom=5):
    ranked = [
        (doc_id, score)
        for doc_id, score in document_scores.items()
        if score is not None
    ]
    ranked.sort(key=lambda item: item[1])
    selected = ranked[:n_bottom] + ranked[-n_top:]
    deduped = {}
    for doc_id, score in selected:
        deduped[doc_id] = score
    return sorted(deduped, key=lambda doc_id: deduped[doc_id], reverse=True)


def plot_generosity_scatter(
    comparison_results_path,
    output_path,
    contract_id_path,
    method,
    n_top=5,
    n_bottom=5,
):
    comparison_results = load_results(comparison_results_path)
    elo_calculator = load_elo_calculator()
    results = elo_calculator.build_output(comparison_results, method)
    contract_ids = load_contract_id_map(contract_id_path)
    _, categories, category_scores, document_scores, document_variances = build_plot_data(results)
    print_variance_score_correlation(document_scores, document_variances)
    display_docs = select_display_documents(document_scores, n_top=n_top, n_bottom=n_bottom)

    y_positions = {doc_id: idx for idx, doc_id in enumerate(display_docs)}

    fig_height = max(4.5, 0.38 * len(display_docs) + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    colors = plt.get_cmap("tab20").colors
    for category_index, category in enumerate(categories):
        xs = []
        ys = []
        for doc_id in display_docs:
            score = category_scores[category][doc_id]
            if score is None:
                continue
            xs.append(score)
            ys.append(y_positions[doc_id])
        ax.scatter(
            xs,
            ys,
            s=38,
            alpha=0.8,
            color=colors[category_index % len(colors)],
            label=category,
            linewidths=0,
        )

    composite_xs = [document_scores[doc_id] for doc_id in display_docs]
    composite_ys = [y_positions[doc_id] for doc_id in display_docs]
    ax.scatter(
        composite_xs,
        composite_ys,
        s=55,
        color="black",
        label="Document composite",
        zorder=5,
    )

    for doc_id, x, y in zip(display_docs, composite_xs, composite_ys):
        ax.text(x, y + 0.18, f"{x:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(
        [contract_ids.get(doc_id, short_label(doc_id)) for doc_id in display_docs],
        fontsize=9,
    )
    ax.set_xlabel("Standardized ELO-Style Generosity")
    ax.set_title("Generosity by CBA")
    plotted_values = [
        score
        for category in categories
        for doc_id in display_docs
        for score in [category_scores[category][doc_id]]
        if score is not None
    ]
    plotted_values.extend(x for x in composite_xs if x is not None)
    x_min = min(plotted_values)
    x_max = max(plotted_values)
    x_pad = 0.05 * (x_max - x_min) if x_max != x_min else 1.0
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(n_top - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(x_max, n_top - 0.75, f"Top {n_top}", ha="right", va="top", fontsize=8, color="gray")
    ax.text(x_max, n_top - 0.25, f"Bottom {n_bottom}", ha="right", va="bottom", fontsize=8, color="gray")

    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = [handles[-1], *handles[:-1]]
    ordered_labels = [labels[-1], *labels[:-1]]
    ax.legend(
        ordered_handles,
        ordered_labels,
        title="Scores",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CBA generosity rankings from saved pairwise comparisons."
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
        default=Path("figures/generosity_scatter.png"),
        help="Path to write the scatter plot.",
    )
    parser.add_argument(
        "--contract-ids",
        type=Path,
        default=Path("references/pilot_healthcare_scores.csv"),
        help="CSV used to map document ids to CBA ids.",
    )
    parser.add_argument("--n-top", type=int, default=5)
    parser.add_argument("--n-bottom", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    cache_model = model_slug(args.model_name)
    input_path = args.input or (
        Path("cache/06_generosity_output")
        / cache_model
        / "bradley_terry_results.json"
    )
    plot_generosity_scatter(
        input_path,
        args.output,
        args.contract_ids,
        args.method,
        n_top=args.n_top,
        n_bottom=args.n_bottom,
    )


if __name__ == "__main__":
    main()
