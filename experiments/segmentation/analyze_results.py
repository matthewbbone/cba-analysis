"""Analyze segmentation experiment results and write report.md."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.csv"
REPORT_FILE = EXPERIMENT_DIR / "report.md"


def _to_float(val: str) -> float | None:
    try:
        return float(val)
    except Exception:
        return None


def _to_int(val: str) -> int | None:
    try:
        return int(val)
    except Exception:
        return None


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def main() -> None:
    if not RESULTS_FILE.exists():
        REPORT_FILE.write_text(
            "# Segmentation Report\n\nNo results found. Run `run_experiment.py` first.\n",
            encoding="utf-8",
        )
        print(f"Wrote {REPORT_FILE}")
        return

    rows = []
    with RESULTS_FILE.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    run_rows = [r for r in rows if (r.get("row_type", "run").strip() == "run")]
    overlap_rows = [r for r in rows if (r.get("row_type", "").strip() == "overlap")]

    lines: list[str] = []
    lines.append("# Segmentation Experiment Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows in results: {len(rows)}")
    lines.append(f"- Run rows: {len(run_rows)}")
    lines.append(f"- Overlap rows: {len(overlap_rows)}")
    lines.append(
        f"- Documents evaluated: {len(sorted(set(r.get('document_id', '') for r in run_rows if r.get('document_id'))))}"
    )
    lines.append("")

    if run_rows:
        lines.append("## Run Metrics")
        lines.append("")
        run_table = []
        for r in sorted(run_rows, key=lambda x: (x.get("document_id", ""), x.get("method", ""), x.get("version", ""))):
            runtime = _to_float(r.get("runtime_sec", "")) or 0.0
            segments = _to_int(r.get("segments", "")) or 0
            coverage = _to_float(r.get("coverage_ratio", "")) or 0.0
            run_table.append([
                r.get("document_id", ""),
                r.get("method", ""),
                r.get("version", ""),
                str(segments),
                f"{coverage:.4f}",
                f"{runtime:.2f}",
            ])
        lines.extend(_md_table(
            ["document_id", "method", "version", "segments", "coverage_ratio", "runtime_sec"],
            run_table,
        ))
        lines.append("")

        lines.append("## Method Comparison")
        lines.append("")
        lines.append("Comparison of `llm_segment_v2` against `llm_segment` for the same document/version.")
        lines.append("")

        by_doc_version: dict[tuple[str, str], dict[str, dict]] = {}
        for r in run_rows:
            doc_id = str(r.get("document_id", ""))
            version = str(r.get("version", ""))
            method = str(r.get("method", ""))
            if not doc_id or not version or not method:
                continue
            by_doc_version.setdefault((doc_id, version), {})[method] = r

        cmp_rows: list[list[str]] = []
        for (doc_id, version), methods in sorted(by_doc_version.items(), key=lambda x: (x[0][0], x[0][1])):
            v1 = methods.get("llm_segment")
            v2 = methods.get("llm_segment_v2")
            if not v1 or not v2:
                continue

            v1_segments = _to_int(v1.get("segments", "")) or 0
            v2_segments = _to_int(v2.get("segments", "")) or 0
            v1_cov = _to_float(v1.get("coverage_ratio", "")) or 0.0
            v2_cov = _to_float(v2.get("coverage_ratio", "")) or 0.0
            cmp_rows.append([
                doc_id,
                version,
                str(v1_segments),
                str(v2_segments),
                str(v2_segments - v1_segments),
                f"{v1_cov:.4f}",
                f"{v2_cov:.4f}",
                f"{(v2_cov - v1_cov):.4f}",
            ])

        if cmp_rows:
            lines.extend(_md_table(
                [
                    "document_id",
                    "version",
                    "v1_segments",
                    "v2_segments",
                    "delta_segments",
                    "v1_coverage_ratio",
                    "v2_coverage_ratio",
                    "delta_coverage_ratio",
                ],
                cmp_rows,
            ))
            lines.append("")
        else:
            lines.append("No document/version pairs contain both methods yet.")
            lines.append("")

    if overlap_rows:
        lines.append("## Overlap KPI")
        lines.append("")
        lines.append("Higher is better for both overlap metrics.")
        lines.append("")
        overlap_table = []
        for r in sorted(overlap_rows, key=lambda x: (x.get("document_id", ""), x.get("method", ""), x.get("version", ""))):
            sec = _to_float(r.get("section_overlap_mean", "")) or 0.0
            cov = _to_float(r.get("coverage_iou", "")) or 0.0
            overlap_table.append([
                r.get("document_id", ""),
                f"{r.get('method', '')}:{r.get('version', '')}",
                f"{r.get('method_b', '')}:{r.get('version_b', '')}",
                f"{sec:.4f}",
                f"{cov:.4f}",
            ])
        lines.extend(_md_table(
            ["document_id", "run_a", "run_b", "section_overlap_mean", "coverage_iou"],
            overlap_table,
        ))
        lines.append("")
    else:
        lines.append("## Overlap KPI")
        lines.append("")
        lines.append("Not enough distinct runs yet to compute pairwise overlap.")
        lines.append("")

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_FILE}")


if __name__ == "__main__":
    main()
