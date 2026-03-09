from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()


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


def _default_gab_output_dir() -> Path:
    root = _project_root()
    cache_dir = _default_cache_dir()
    if cache_dir:
        return (_resolve_path(cache_dir, root) / "04_generosity_gab_output" / "dol_archive").resolve()
    return (root / "outputs" / "04_generosity_gab_output" / "dol_archive").resolve()


def _default_output_dir() -> Path:
    return (_project_root() / "figures").resolve()


def _safe_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if np.isnan(numeric):
        return None
    return float(numeric)


def _parse_document_num(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"document_(\d+)", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _pearson_corr(x_vals: list[float], y_vals: list[float]) -> float | None:
    if len(x_vals) != len(y_vals) or len(x_vals) < 2:
        return None
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom <= 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denom)


def _average_tie_ranks(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)

    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank_value = ((i + j) / 2.0) + 1.0
        ranks[order[i : j + 1]] = rank_value
        i = j + 1
    return ranks


def _spearman_corr(x_vals: list[float], y_vals: list[float]) -> float | None:
    if len(x_vals) != len(y_vals) or len(x_vals) < 2:
        return None
    x_ranks = _average_tie_ranks(x_vals)
    y_ranks = _average_tie_ranks(y_vals)
    return _pearson_corr(x_ranks.tolist(), y_ranks.tolist())


def _load_llm_document_composite_map(llm_output_dir: Path) -> dict[str, float]:
    csv_path = llm_output_dir / "document_composite_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"LLM composite CSV not found: {csv_path}")

    out: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            doc_id = str(row.get("document_id", "")).strip()
            score = _safe_float_or_none(row.get("document_composite_score"))
            if not doc_id or score is None:
                continue
            out[doc_id] = float(score)
    return out


def _load_gab_document_composite_map(gab_output_dir: Path) -> dict[str, float]:
    csv_path = gab_output_dir / "document_composite_rankings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"GAB composite CSV not found: {csv_path}")

    out: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            doc_id = str(row.get("document_id", "")).strip()
            score = _safe_float_or_none(row.get("composite_score"))
            if not doc_id or score is None:
                continue
            out[doc_id] = float(score)
    return out


def run(
    *,
    llm_output_dir: Path,
    gab_output_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    llm_output_dir = llm_output_dir.expanduser().resolve()
    gab_output_dir = gab_output_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_map = _load_llm_document_composite_map(llm_output_dir)
    gab_map = _load_gab_document_composite_map(gab_output_dir)

    overlap_doc_ids = sorted(set(llm_map.keys()).intersection(gab_map.keys()), key=_parse_document_num)
    if len(overlap_doc_ids) < 2:
        raise RuntimeError(
            "Need at least 2 overlapping documents with numeric scores to build the validation scatter."
        )

    x_vals = [float(llm_map[doc_id]) for doc_id in overlap_doc_ids]
    y_vals = [float(gab_map[doc_id]) for doc_id in overlap_doc_ids]
    pearson = _pearson_corr(x_vals, y_vals)
    spearman = _spearman_corr(x_vals, y_vals)

    overlap_csv_path = output_dir / "validation_llm_vs_gab_overlap.csv"
    with overlap_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "document_id",
                "llm_document_composite_score",
                "gab_composite_score",
                "difference_llm_minus_gab",
            ],
        )
        writer.writeheader()
        for doc_id in overlap_doc_ids:
            x_val = float(llm_map[doc_id])
            y_val = float(gab_map[doc_id])
            writer.writerow(
                {
                    "document_id": doc_id,
                    "llm_document_composite_score": round(x_val, 6),
                    "gab_composite_score": round(y_val, 6),
                    "difference_llm_minus_gab": round(x_val - y_val, 6),
                }
            )

    mpl_cache_dir = output_dir / ".mpl_cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to create the validation scatter plot.") from exc

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(x_vals, y_vals, s=42, alpha=0.8, color="#2563eb", edgecolors="white", linewidths=0.5)

    if len(x_vals) >= 2:
        try:
            slope, intercept = np.polyfit(np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float), deg=1)
            x_min = min(x_vals)
            x_max = max(x_vals)
            ax.plot(
                [x_min, x_max],
                [slope * x_min + intercept, slope * x_max + intercept],
                color="#dc2626",
                linewidth=1.3,
            )
        except Exception:
            pass

    annotation_lines = [
        f"Pearson: {pearson:.3f}" if isinstance(pearson, (int, float)) else "Pearson: n/a",
        f"Spearman: {spearman:.3f}" if isinstance(spearman, (int, float)) else "Spearman: n/a",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
    )

    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Rubric-based Generosity")
    ax.set_ylabel("GABRIEL Generosity")

    png_path = output_dir / "validation_llm_vs_gab_scatter.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=240)
    plt.close(fig)

    summary = {
        "llm_output_dir": str(llm_output_dir),
        "gab_output_dir": str(gab_output_dir),
        "output_dir": str(output_dir),
        "llm_docs_with_score": len(llm_map),
        "gab_docs_with_score": len(gab_map),
        "overlap_docs": len(overlap_doc_ids),
        "pearson": pearson,
        "spearman": spearman,
        "outputs": {
            "scatter_png": str(png_path),
            "overlap_csv": str(overlap_csv_path),
        },
    }
    summary_path = output_dir / "validation_llm_vs_gab_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create LLM-vs-GAB validation scatter with Pearson and Spearman correlation "
            "(LLM on x-axis, GAB on y-axis)."
        )
    )
    parser.add_argument("--llm-output-dir", type=Path, default=_default_llm_output_dir())
    parser.add_argument("--gab-output-dir", type=Path, default=_default_gab_output_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()

    summary = run(
        llm_output_dir=args.llm_output_dir,
        gab_output_dir=args.gab_output_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
