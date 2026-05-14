import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.utils.llm import model_slug


MODEL_NAME = "gpt-5.4-nano"


def load_results(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def short_label(document_id, max_length=32):
    label = document_id.split("/", 1)[-1].replace("_", " ")
    if len(label) <= max_length:
        return label
    return label[: max_length - 3] + "..."


def percentile_scores(values_by_doc):
    valid_items = [
        (doc_id, value)
        for doc_id, value in values_by_doc.items()
        if value is not None
    ]
    if not valid_items:
        return {doc_id: None for doc_id in values_by_doc}

    valid_items.sort(key=lambda item: item[1])
    if len(valid_items) == 1:
        return {valid_items[0][0]: 3.0}

    scores = {}
    max_rank = len(valid_items) - 1
    for rank, (doc_id, _) in enumerate(valid_items):
        scores[doc_id] = 1.0 + 4.0 * (rank / max_rank)

    for doc_id in values_by_doc:
        scores.setdefault(doc_id, None)
    return scores


def build_plot_data(results):
    documents = [item["document_id"] for item in results["documents"]]
    categories = results["categories"]

    category_scores = {}
    for category in categories:
        strengths = {
            doc_id: results["rankings_by_category"][category][doc_id]["strength"]
            for doc_id in documents
        }
        category_scores[category] = percentile_scores(strengths)

    document_scores = {}
    for doc_id in documents:
        scores = [
            category_scores[category][doc_id]
            for category in categories
            if category_scores[category][doc_id] is not None
        ]
        document_scores[doc_id] = sum(scores) / len(scores) if scores else None

    return documents, categories, category_scores, document_scores


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


def plot_generosity_scatter(results_path, output_path, n_top=5, n_bottom=5):
    results = load_results(results_path)
    _, categories, category_scores, document_scores = build_plot_data(results)
    display_docs = select_display_documents(document_scores, n_top=n_top, n_bottom=n_bottom)

    y_positions = {doc_id: idx for idx, doc_id in enumerate(display_docs)}

    fig_height = max(5, 0.45 * len(display_docs) + 1.5)
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
            s=24,
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
    ax.set_yticklabels([short_label(doc_id) for doc_id in display_docs], fontsize=9)
    ax.set_xlabel("Document composite score (1-5)")
    ax.set_title("Generosity by CBA")
    ax.set_xlim(0.8, 5.2)
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(3.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(n_top - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(5.15, n_top - 0.75, f"Top {n_top}", ha="right", va="top", fontsize=8, color="gray")
    ax.text(5.15, n_top - 0.25, f"Bottom {n_bottom}", ha="right", va="bottom", fontsize=8, color="gray")

    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = [handles[-1], *handles[:-1]]
    ordered_labels = [labels[-1], *labels[:-1]]
    ax.legend(
        ordered_handles,
        ordered_labels,
        title="Scores",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    plot_generosity_scatter(
        Path("cache/06_generosity_output")
        / model_slug(MODEL_NAME)
        / "bradley_terry_results.json",
        Path("figures/generosity_scatter.png"),
    )


if __name__ == "__main__":
    main()
