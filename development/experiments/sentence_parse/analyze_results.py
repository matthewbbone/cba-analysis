"""Analyze sentence_parse experiment runs and write a markdown report."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.csv"
REPORT_FILE = EXPERIMENT_DIR / "report.md"
PLOTS_DIR = EXPERIMENT_DIR / "plots"


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _to_int(v: str) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fmt_avg(values: list[float]) -> str:
    if not values:
        return "0.0000"
    return f"{sum(values) / len(values):.4f}"


def _maybe_write_plots(rows: list[dict[str, str]]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    methods = sorted({r.get("method", "") for r in rows if r.get("method", "")})
    if methods:
        runtime_by_method = []
        sentences_by_method = []
        for method in methods:
            subset = [r for r in rows if r.get("method", "") == method and r.get("status", "") == "ok"]
            runtime_by_method.append(sum(_to_float(r.get("runtime_sec", "0")) for r in subset))
            sentences_by_method.append(sum(_to_int(r.get("sentences_processed", "0")) for r in subset))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(methods, runtime_by_method)
        ax.set_title("Total Runtime by Method (successful runs)")
        ax.set_ylabel("seconds")
        ax.set_xlabel("method")
        fig.tight_layout()
        p1 = PLOTS_DIR / "runtime_by_method.png"
        fig.savefig(p1)
        plt.close(fig)
        out_paths.append(p1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(methods, sentences_by_method)
        ax.set_title("Total Sentences Parsed by Method (successful runs)")
        ax.set_ylabel("sentences")
        ax.set_xlabel("method")
        fig.tight_layout()
        p2 = PLOTS_DIR / "sentences_by_method.png"
        fig.savefig(p2)
        plt.close(fig)
        out_paths.append(p2)

    return out_paths


def build_report(rows: list[dict[str, str]], plot_paths: list[Path]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    total_runs = len(rows)
    ok_rows = [r for r in rows if r.get("status", "") == "ok"]
    err_rows = [r for r in rows if r.get("status", "") != "ok"]

    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_method[r.get("method", "unknown")].append(r)

    lines: list[str] = []
    lines.append("# sentence_parse experiment report")
    lines.append("")
    lines.append(f"- Generated (UTC): {now}")
    lines.append(f"- Results file: `{RESULTS_FILE}`")
    lines.append("")
    lines.append("## Overall summary")
    lines.append("")
    lines.append(f"- Total runs: {total_runs}")
    lines.append(f"- Successful runs: {len(ok_rows)}")
    lines.append(f"- Failed runs: {len(err_rows)}")
    lines.append(
        f"- Total sentences parsed (successful runs): "
        f"{sum(_to_int(r.get('sentences_processed', '0')) for r in ok_rows)}"
    )
    lines.append(
        f"- Total segments processed (successful runs): "
        f"{sum(_to_int(r.get('segments_processed', '0')) for r in ok_rows)}"
    )
    lines.append("")

    lines.append("## Method summary")
    lines.append("")
    lines.append("| method | runs | success | failures | avg runtime (s) | total sentences | total segments |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in sorted(by_method):
        subset = by_method[method]
        ok_subset = [r for r in subset if r.get("status", "") == "ok"]
        runtimes = [_to_float(r.get("runtime_sec", "0")) for r in subset]
        total_sentences = sum(_to_int(r.get("sentences_processed", "0")) for r in ok_subset)
        total_segments = sum(_to_int(r.get("segments_processed", "0")) for r in ok_subset)
        lines.append(
            f"| {method} | {len(subset)} | {len(ok_subset)} | {len(subset) - len(ok_subset)} | "
            f"{_fmt_avg(runtimes)} | {total_sentences} | {total_segments} |"
        )
    lines.append("")

    if err_rows:
        lines.append("## Recent errors")
        lines.append("")
        for r in err_rows[-5:]:
            lines.append(
                f"- `{r.get('timestamp_utc', '')}` | `{r.get('method', '')}` | "
                f"{r.get('error', '').strip() or 'Unknown error'}"
            )
        lines.append("")

    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for p in plot_paths:
            rel = p.relative_to(EXPERIMENT_DIR)
            lines.append(f"![{p.stem}]({rel.as_posix()})")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This report is generated from `results.csv` only.")
    lines.append("- Re-run `analyze_results.py` after new experiment runs to refresh this report.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = _read_rows(RESULTS_FILE)
    plots = _maybe_write_plots(rows)
    report = build_report(rows, plots)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"Wrote report: {REPORT_FILE}")
    if plots:
        for p in plots:
            print(f"Wrote plot: {p}")


if __name__ == "__main__":
    main()
