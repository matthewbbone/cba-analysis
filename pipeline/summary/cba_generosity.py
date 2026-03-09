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

# Keep aligned with pipeline/04_generosity_llm/runner.py.
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


def _default_llm_output_dir() -> Path:
    root = _project_root()
    cache_dir = _default_cache_dir()
    if cache_dir:
        return (_resolve_path(cache_dir, root) / "04_generosity_llm_output" / "dol_archive").resolve()
    return (root / "outputs" / "04_generosity_llm_output" / "dol_archive").resolve()


def _default_figure_dir() -> Path:
    return (_project_root() / "figures").resolve()


def _default_dol_archive_dir() -> Path:
    return (_project_root() / "dol_archive").resolve()


def _safe_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN
        return None
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_clause_type_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _is_excluded_procedural_clause_type(clause_type: Any) -> bool:
    return _normalize_clause_type_key(clause_type) in EXCLUDED_PROCEDURAL_CLAUSE_TYPE_KEYS


def _parse_document_num(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"document_(\d+)", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _to_document_id_from_cbafile(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    numeric = _safe_float_or_none(raw_value)
    if numeric is None:
        text = str(raw_value).strip()
        if not text:
            return None
        m = re.search(r"(\d+)", text)
        if not m:
            return None
        return f"document_{int(m.group(1))}"
    return f"document_{int(numeric)}"


def _load_document_company_names(dol_archive_dir: Path) -> tuple[dict[str, str], str | None]:
    try:
        import pandas as pd
    except Exception:
        return {}, None

    candidates = [
        dol_archive_dir / "CBAList_with_statefips.dta",
        dol_archive_dir / "CBAList_fixed.dta",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_stata(path)
        except Exception:
            continue

        if df is None or df.empty:
            continue

        lower_to_col = {str(col).strip().lower(): str(col) for col in df.columns}
        cbafile_col = lower_to_col.get("cbafile")
        employer_col = lower_to_col.get("employername")
        if not cbafile_col or not employer_col:
            continue

        out: dict[str, str] = {}
        for _, row in df.iterrows():
            doc_id = _to_document_id_from_cbafile(row.get(cbafile_col))
            employer_name = str(row.get(employer_col, "")).strip()
            if not doc_id or not employer_name or employer_name.lower() == "nan":
                continue
            if doc_id not in out:
                out[doc_id] = employer_name
        if out:
            return out, str(path)

    return {}, None


def _select_top_bottom_documents(
    rows: list[dict[str, Any]],
    count: int,
    score_key: str,
) -> tuple[list[dict[str, Any]], int, int]:
    if not rows:
        return [], 0, 0
    n = max(1, int(count))
    top_rows = list(rows[:n])
    top_ids = {str(row.get("document_id", "")).strip() for row in top_rows}
    bottom_rows = [
        row
        for row in rows[-n:]
        if str(row.get("document_id", "")).strip()
        and str(row.get("document_id", "")).strip() not in top_ids
    ]

    out = [*top_rows, *bottom_rows]
    # Keep explicit score ordering within the displayed set (highest -> lowest).
    out.sort(
        key=lambda row: (
            -float(row.get(score_key, float("-inf"))),
            _parse_document_num(str(row.get("document_id", ""))),
        )
    )

    top_count = len([row for row in out if str(row.get("document_id", "")).strip() in top_ids])
    bottom_count = max(0, len(out) - top_count)
    return out, top_count, bottom_count


def _build_document_display_labels(doc_ids: list[str], company_map: dict[str, str]) -> list[str]:
    max_len = 30

    def _truncate(text: str, limit: int = max_len) -> str:
        raw = str(text or "").strip()
        if len(raw) <= limit:
            return raw
        if limit <= 3:
            return "." * max(0, limit)
        return raw[: limit - 3].rstrip() + "..."

    base_labels = [str(company_map.get(doc_id, "")).strip() or str(doc_id) for doc_id in doc_ids]
    duplicates = Counter(base_labels)
    labels = []
    for doc_id, base in zip(doc_ids, base_labels):
        if duplicates.get(base, 0) > 1:
            labels.append(_truncate(f"{base} ({doc_id})"))
        else:
            labels.append(_truncate(base))
    return labels


def _display_clause_label(clause_type: Any) -> str:
    return str(clause_type or "").strip().replace(" Clause", "")


def _llm_score_or_none(value: Any) -> float | None:
    score = _safe_float_or_none(value)
    if isinstance(score, (int, float)) and score == score:
        return float(score)
    return None


def _load_document_composite_rows(llm_output_dir: Path) -> list[dict[str, Any]]:
    csv_path = llm_output_dir / "document_composite_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"LLM document composite CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not isinstance(raw, dict):
                continue
            document_id = str(raw.get("document_id", "")).strip()
            score = _llm_score_or_none(raw.get("document_composite_score"))
            if not document_id or score is None:
                continue
            rows.append(
                {
                    "provider": str(raw.get("provider", "")).strip(),
                    "model": str(raw.get("model", "")).strip(),
                    "document_id": document_id,
                    "document_composite_score": float(score),
                    "clause_count_scored": max(0, _safe_int(raw.get("clause_count_scored", 0))),
                }
            )

    rows.sort(
        key=lambda row: (
            -float(row.get("document_composite_score", float("-inf"))),
            _parse_document_num(str(row.get("document_id", ""))),
        )
    )
    rank = 0
    for row in rows:
        score = _llm_score_or_none(row.get("document_composite_score"))
        if score is None:
            row["document_rank"] = None
            continue
        rank += 1
        row["document_rank"] = rank
    return rows


def _load_document_clause_rows(llm_output_dir: Path) -> list[dict[str, Any]]:
    csv_path = llm_output_dir / "document_clause_composite_scores.csv"
    if not csv_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not isinstance(raw, dict):
                continue
            document_id = str(raw.get("document_id", "")).strip()
            clause_type = str(raw.get("clause_type", "")).strip() or "OTHER"
            score = _llm_score_or_none(raw.get("clause_composite_score"))
            if not document_id or score is None:
                continue
            if clause_type == "OTHER":
                continue
            if _is_excluded_procedural_clause_type(clause_type):
                continue
            rows.append(
                {
                    "document_id": document_id,
                    "clause_type": clause_type,
                    "clause_composite_score": float(score),
                }
            )
    return rows


def run(
    *,
    llm_output_dir: Path,
    figure_dir: Path,
    dol_archive_dir: Path,
    top_bottom_count: int = 5,
) -> dict[str, Any]:
    llm_output_dir = llm_output_dir.expanduser().resolve()
    figure_dir = figure_dir.expanduser().resolve()
    dol_archive_dir = dol_archive_dir.expanduser().resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_document_composite_rows(llm_output_dir)
    if not rows:
        raise RuntimeError("No usable document composite rows found in document_composite_scores.csv.")

    plot_rows, top_group_count, bottom_group_count = _select_top_bottom_documents(
        rows,
        top_bottom_count,
        score_key="document_composite_score",
    )
    doc_ids = [str(row["document_id"]) for row in plot_rows]
    if not doc_ids:
        raise RuntimeError("No documents selected for plotting.")

    company_name_map, company_source_path = _load_document_company_names(dol_archive_dir)
    y_labels = _build_document_display_labels(doc_ids, company_name_map)
    doc_id_set = set(doc_ids)

    clause_rows = [
        row
        for row in _load_document_clause_rows(llm_output_dir)
        if str(row.get("document_id", "")).strip() in doc_id_set
    ]
    clause_types = sorted(
        {str(row.get("clause_type", "")).strip() for row in clause_rows if str(row.get("clause_type", "")).strip()},
        key=lambda label: label.lower(),
    )
    clause_scores_by_doc: dict[str, list[tuple[str, float]]] = {doc_id: [] for doc_id in doc_ids}
    for row in clause_rows:
        doc_id = str(row["document_id"])
        clause_type = str(row["clause_type"])
        score = float(row["clause_composite_score"])
        clause_scores_by_doc.setdefault(doc_id, []).append((clause_type, score))

    mpl_cache_dir = figure_dir / ".mpl_cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
    except Exception as exc:
        raise RuntimeError("matplotlib and numpy are required to render this figure.") from exc

    y = np.arange(len(plot_rows))
    scores = np.array(
        [float(row.get("document_composite_score", 0.0)) for row in plot_rows],
        dtype=float,
    )

    fig_height = max(3.8, min(8.5, 1.7 + (0.34 * len(doc_ids))))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    cmap = plt.get_cmap("tab20")
    clause_color_map = {clause_type: cmap(idx % 20) for idx, clause_type in enumerate(clause_types)}

    for idx, doc_id in enumerate(doc_ids):
        clause_points = clause_scores_by_doc.get(doc_id, [])
        for clause_type, clause_score in clause_points:
            ax.scatter(
                float(clause_score),
                idx,
                s=26,
                color=clause_color_map.get(clause_type, "#9ca3af"),
                edgecolors="white",
                linewidths=0.4,
                alpha=0.95,
                zorder=3,
            )
        ax.scatter(
            float(scores[idx]),
            idx,
            s=90,
            color="black",
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    if len(plot_rows) > 0:
        # Add a little top padding so annotations above the first point are visible.
        ax.set_ylim(len(plot_rows) - 0.5, -0.7)
    ax.set_xlabel("Document composite score (1-5)")
    ax.set_ylabel("")
    ax.set_title("Generosity by CBA")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    total_values = [float(row["document_composite_score"]) for row in plot_rows]
    clause_values = [float(row.get("clause_composite_score", 0.0)) for row in clause_rows]
    all_x_values = [*total_values, *clause_values]
    max_dev = max((abs(value - 3.0) for value in all_x_values), default=1.0)
    half_range = max(1.0, max_dev + 0.2)
    x_min = 3.0 - half_range
    x_max = 3.0 + half_range
    ax.set_xlim(x_min, x_max)
    ax.axvline(3.0, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.9, zorder=1)
    offset = max(0.12, 0.045 * half_range)
    for idx, total in enumerate(total_values):
        ax.text(
            total,
            idx - offset,
            f"{total:.2f}",
            va="bottom",
            ha="center",
            fontsize=8,
            clip_on=False,
        )

    # Visual divider between top-N and bottom-N groups.
    if top_group_count > 0 and bottom_group_count > 0:
        divider_y = float(top_group_count) - 0.5
        ax.axhline(divider_y, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.9)
        label_x = x_max - (0.03 * half_range)
        ax.text(
            label_x,
            divider_y - 0.18,
            f"Top {top_group_count}",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#374151",
        )
        ax.text(
            label_x,
            divider_y + 0.18,
            f"Bottom {bottom_group_count}",
            ha="right",
            va="top",
            fontsize=8,
            color="#374151",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=8,
            label="Document composite",
        )
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=clause_color_map[clause_type],
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=5.5,
                label=_display_clause_label(clause_type),
            )
            for clause_type in clause_types
        ]
    )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Scores",
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            frameon=True,
            facecolor="white",
            edgecolor="#d1d5db",
            framealpha=0.5,
            fontsize=8,
            title_fontsize=9,
        )
    fig.subplots_adjust(left=0.3, right=0.97, top=0.92, bottom=0.12)
    png_path = figure_dir / "cba_generosity_llm_composite_scatter.png"
    fig.savefig(png_path, dpi=240)
    plt.close(fig)

    summary = {
        "llm_output_dir": str(llm_output_dir),
        "dol_archive_dir": str(dol_archive_dir),
        "company_name_source_dta": company_source_path,
        "figure_dir": str(figure_dir),
        "source_csv": str(llm_output_dir / "document_composite_scores.csv"),
        "documents_plotted": len(doc_ids),
        "documents_available": len(rows),
        "top_bottom_count": int(top_bottom_count),
        "top_group_count": int(top_group_count),
        "bottom_group_count": int(bottom_group_count),
        "score_field": "document_composite_score",
        "clause_score_field": "clause_composite_score",
        "clause_points_plotted": len(clause_rows),
        "score_source_logic": "matches review_ui/app3.py _load_llm_document_composite_scores_csv",
        "display_order": "document_composite_score_desc",
        "group_divider_line": bool(top_group_count > 0 and bottom_group_count > 0),
        "x_axis_center": 3.0,
        "x_axis_limits": [float(x_min), float(x_max)],
        "label_mode": "employername_from_dta_fallback_document_id",
        "output_png": str(png_path),
    }
    summary_path = figure_dir / "cba_generosity_llm_composite_scatter_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a horizontal bar chart of 04_generosity_llm document composite scores "
            "(top and bottom CBAs) and export PNG to ./figures."
        )
    )
    parser.add_argument("--llm-output-dir", type=Path, default=_default_llm_output_dir())
    parser.add_argument("--figure-dir", type=Path, default=_default_figure_dir())
    parser.add_argument("--dol-archive-dir", type=Path, default=_default_dol_archive_dir())
    parser.add_argument("--top-bottom-count", type=int, default=5)
    args = parser.parse_args()

    summary = run(
        llm_output_dir=args.llm_output_dir,
        figure_dir=args.figure_dir,
        dol_archive_dir=args.dol_archive_dir,
        top_bottom_count=args.top_bottom_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
