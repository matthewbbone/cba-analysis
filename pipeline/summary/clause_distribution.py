from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Keep this list aligned with pipeline/04_generosity_llm/runner.py.
EXCLUDED_PROCEDURAL_CLAUSE_TYPES = [
    "Recognition Clause",
    "Recognition",
    "Parties to Agreement and Preamble",
    "Parties to Agreement",
    "Preamble",
    "Bargaining Unit",
    "",
]
EXCLUDED_PROCEDURAL_CLAUSE_TYPE_KEYS = {
    re.sub(r"[^a-z0-9]+", " ", str(label).strip().lower()).strip()
    for label in EXCLUDED_PROCEDURAL_CLAUSE_TYPES
    if str(label).strip()
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cache_dir() -> str:
    return os.environ.get("CACHE_DIR", "").strip()


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _default_classification_dir() -> Path:
    root = _project_root()
    cache_dir = _default_cache_dir()
    if cache_dir:
        return (_resolve_path(cache_dir, root) / "03_classification_output" / "dol_archive").resolve()
    return (root / "outputs" / "03_classification_output" / "dol_archive").resolve()


def _default_output_dir() -> Path:
    # Keep outputs local to the repository, independent of CACHE_DIR.
    return (_project_root() / "figures").resolve()


def _default_figure_dir() -> Path:
    return (_project_root() / "figures").resolve()


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _document_sort_key(path_or_name: str | Path) -> tuple[int, str]:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    match = re.fullmatch(r"document_(\d+)", name)
    if not match:
        return (10**12, name)
    return (int(match.group(1)), name)


def _segment_sort_key(path_or_name: str | Path) -> tuple[int, str]:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    match = re.fullmatch(r"segment_(\d+)\.json", name)
    if not match:
        return (10**12, name)
    return (int(match.group(1)), name)


def _extract_clause_type(payload: dict[str, Any]) -> str:
    labels = payload.get("labels", [])
    if isinstance(labels, list):
        for label in labels:
            clause = str(label).strip()
            if clause:
                return clause
    label = str(payload.get("label", "")).strip()
    return label if label else "OTHER"


def _normalize_clause_type_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _is_excluded_procedural_clause_type(clause_type: Any) -> bool:
    return _normalize_clause_type_key(clause_type) in EXCLUDED_PROCEDURAL_CLAUSE_TYPE_KEYS


def _display_clause_label(clause_type: Any) -> str:
    text = str(clause_type or "").strip()
    return text.replace(" Clause", "")


def _list_document_dirs(classification_dir: Path) -> list[Path]:
    if not classification_dir.exists() or not classification_dir.is_dir():
        return []
    doc_dirs = [
        path
        for path in classification_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"document_\d+", path.name)
    ]
    return sorted(doc_dirs, key=_document_sort_key)


def run(
    *,
    classification_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    include_other: bool = False,
) -> dict[str, Any]:
    classification_dir = classification_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    figure_dir = figure_dir.expanduser().resolve()

    if not classification_dir.exists():
        raise FileNotFoundError(f"Classification directory not found: {classification_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    doc_dirs = _list_document_dirs(classification_dir)
    total_documents = 0
    total_segments_counted = 0
    doc_clause_counter: Counter[str] = Counter()
    segment_clause_counter: Counter[str] = Counter()

    for doc_dir in doc_dirs:
        segment_paths = sorted(
            [p for p in doc_dir.glob("segment_*.json") if p.is_file()],
            key=_segment_sort_key,
        )
        if not segment_paths:
            continue

        total_documents += 1
        clauses_in_doc: set[str] = set()
        for segment_path in segment_paths:
            payload = _safe_read_json(segment_path)
            if not isinstance(payload, dict):
                continue
            clause_type = _extract_clause_type(payload)
            if not include_other and clause_type == "OTHER":
                continue
            if _is_excluded_procedural_clause_type(clause_type):
                continue
            total_segments_counted += 1
            segment_clause_counter[clause_type] += 1
            clauses_in_doc.add(clause_type)

        for clause_type in clauses_in_doc:
            doc_clause_counter[clause_type] += 1

    clause_types = sorted(
        segment_clause_counter.keys(),
        key=lambda clause: (
            -doc_clause_counter.get(clause, 0),
            -segment_clause_counter.get(clause, 0),
            clause.lower(),
        ),
    )

    rows: list[dict[str, Any]] = []
    for rank, clause_type in enumerate(clause_types, start=1):
        docs_with_clause = int(doc_clause_counter.get(clause_type, 0))
        segments_with_clause = int(segment_clause_counter.get(clause_type, 0))
        share_docs = (float(docs_with_clause) / float(total_documents)) if total_documents > 0 else 0.0
        share_segments = (
            float(segments_with_clause) / float(total_segments_counted)
            if total_segments_counted > 0
            else 0.0
        )
        rows.append(
            {
                "rank": rank,
                "clause_type": clause_type,
                "documents_with_clause": docs_with_clause,
                "share_documents_with_clause": share_docs,
                "pct_documents_with_clause": share_docs * 100.0,
                "segments_with_clause": segments_with_clause,
                "share_segments_with_clause": share_segments,
                "pct_segments_with_clause": share_segments * 100.0,
            }
        )
    all_rows = list(rows)
    # Keep only top 10 most common clause types for charting.
    chart_rows = all_rows[:10]

    csv_path = output_dir / "clause_type_distribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "clause_type",
                "documents_with_clause",
                "share_documents_with_clause",
                "pct_documents_with_clause",
                "segments_with_clause",
                "share_segments_with_clause",
                "pct_segments_with_clause",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    chart_html_path = None
    chart_png_path = None
    png_path = figure_dir / "clause_type_document_prevalence.png"
    if chart_rows:
        try:
            import plotly.graph_objects as go

            chart_y_labels = [_display_clause_label(row["clause_type"]) for row in chart_rows]
            customdata = [
                [
                    int(row["documents_with_clause"]),
                    int(total_documents),
                    int(row["segments_with_clause"]),
                    int(total_segments_counted),
                    float(row["pct_segments_with_clause"]),
                ]
                for row in chart_rows
            ]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[float(row["pct_documents_with_clause"]) for row in chart_rows],
                        y=chart_y_labels,
                        orientation="h",
                        marker={"color": "#d1d5db"},
                        text=[f"{float(row['pct_documents_with_clause']):.0f}%" for row in chart_rows],
                        textposition="inside",
                        textfont={"color": "#111827"},
                        cliponaxis=False,
                        customdata=customdata,
                        hovertemplate=(
                            "Clause type: %{y}<br>"
                            "Documents: %{customdata[0]} / %{customdata[1]} "
                            "(%{x:.2f}%)<br>"
                            "Segments: %{customdata[2]} / %{customdata[3]} "
                            "(%{customdata[4]:.2f}%)<extra></extra>"
                        ),
                    )
                ]
            )
            fig.update_layout(
                title=None,
                xaxis_title="CBA Coverage (%)",
                yaxis_title=None,
                template="plotly_white",
                width=300,
                height=max(320, min(900, 40 + (20 * len(chart_rows)))),
                margin={"l": 220, "r": 30, "t": 60, "b": 50},
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_yaxes(tickfont={"size": 8})

            html_path = figure_dir / "clause_type_document_prevalence.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            chart_html_path = str(html_path)

            try:
                fig.write_image(
                    str(png_path),
                    width=300,
                    height=max(320, min(900, 40 + (20 * len(chart_rows)))),
                    scale=2,
                )
                chart_png_path = str(png_path)
            except Exception:
                chart_png_path = None
        except Exception:
            chart_html_path = None
            chart_png_path = None

        if chart_png_path is None:
            try:
                import matplotlib.pyplot as plt

                y_labels = [_display_clause_label(row["clause_type"]) for row in chart_rows]
                x_values = [float(row["pct_documents_with_clause"]) for row in chart_rows]

                fig_mpl_h = max(3.2, min(9.0, 0.9 + 0.24 * len(chart_rows)))
                fig_mpl, ax = plt.subplots(figsize=(3.0, fig_mpl_h))
                bars = ax.barh(y_labels, x_values, color="#d1d5db")
                ax.invert_yaxis()
                ax.set_xlabel("CBA Coverage (%)")
                ax.set_ylabel("")
                ax.set_title("")
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="x", alpha=0.25, linewidth=0.6)
                x_max = max(x_values) if x_values else 0.0
                label_offset = max(0.25, x_max * 0.015)
                for bar, value in zip(bars, x_values):
                    if value <= 0:
                        continue
                    if value > (label_offset * 1.3):
                        x_text = value - label_offset
                        ha = "right"
                    else:
                        x_text = max(0.1, value * 0.5)
                        ha = "center"
                    ax.text(
                        x_text,
                        bar.get_y() + (bar.get_height() / 2.0),
                        f"{value:.0f}%",
                        va="center",
                        ha=ha,
                        fontsize=8,
                        color="#111827",
                    )
                if x_max > 0:
                    ax.set_xlim(0.0, x_max + (label_offset * 2.0))
                fig_mpl.tight_layout()
                fig_mpl.savefig(png_path, dpi=220)
                plt.close(fig_mpl)
                chart_png_path = str(png_path)
            except Exception:
                chart_png_path = None

    summary = {
        "classification_dir": str(classification_dir),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "include_other": bool(include_other),
        "excluded_procedural_clause_types": EXCLUDED_PROCEDURAL_CLAUSE_TYPES,
        "documents_with_segments": int(total_documents),
        "segments_counted": int(total_segments_counted),
        "clause_types_counted": int(len(all_rows)),
        "clause_types_plotted": int(len(chart_rows)),
        "top_n_clause_types": 10,
        "outputs": {
            "clause_distribution_csv": str(csv_path),
            "plotly_clause_distribution_html": chart_html_path,
            "plotly_clause_distribution_png": chart_png_path,
        },
    }
    (output_dir / "clause_type_distribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clause type prevalence summary across documents and save Plotly chart to ./figures."
    )
    parser.add_argument("--classification-dir", type=Path, default=_default_classification_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--figure-dir", type=Path, default=_default_figure_dir())
    parser.add_argument("--include-other", action="store_true", help="Include OTHER clause type in counts and chart.")
    args = parser.parse_args()

    summary = run(
        classification_dir=args.classification_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        include_other=args.include_other,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
