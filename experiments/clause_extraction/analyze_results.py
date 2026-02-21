"""Analyze clause extraction experiment results.

Reads results.csv and generates summary statistics and a report.
"""

import csv
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.csv"
REPORT_FILE = EXPERIMENT_DIR / "report.md"


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        print(f"No results file found at {RESULTS_FILE}")
        return []
    with open(RESULTS_FILE, "r") as f:
        return list(csv.DictReader(f))


def analyze(rows: list[dict]) -> dict:
    """Compute summary statistics from results."""
    # Group by method pair
    pair_stats = defaultdict(lambda: {"char_ious": [], "token_jaccards": [], "label_agreement": 0, "total": 0})

    for row in rows:
        key = (row["method_a"], row["method_b"])
        char_iou = float(row["char_iou"])
        token_jac = float(row["token_jaccard"])
        present_a = row["present_a"] == "True"
        present_b = row["present_b"] == "True"

        pair_stats[key]["char_ious"].append(char_iou)
        pair_stats[key]["token_jaccards"].append(token_jac)
        if present_a and present_b:
            pair_stats[key]["label_agreement"] += 1
        pair_stats[key]["total"] += 1

    # Group by clause label
    clause_stats = defaultdict(lambda: {"char_ious": [], "token_jaccards": [], "count": 0})
    for row in rows:
        label = row["clause_label"]
        clause_stats[label]["char_ious"].append(float(row["char_iou"]))
        clause_stats[label]["token_jaccards"].append(float(row["token_jaccard"]))
        clause_stats[label]["count"] += 1

    return {
        "pair_stats": dict(pair_stats),
        "clause_stats": dict(clause_stats),
        "total_rows": len(rows),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def generate_report(stats: dict) -> str:
    lines = ["# Clause Extraction Experiment Report\n"]

    lines.append("## Overall Pairwise Agreement\n")
    lines.append("| Method A | Method B | Avg Char IoU | Avg Token Jaccard | Label Agreement | N |")
    lines.append("|----------|----------|-------------|-------------------|----------------|---|")

    for (m_a, m_b), s in sorted(stats["pair_stats"].items()):
        avg_iou = _mean(s["char_ious"])
        avg_jac = _mean(s["token_jaccards"])
        agree_pct = (s["label_agreement"] / s["total"] * 100) if s["total"] > 0 else 0
        lines.append(
            f"| {m_a} | {m_b} | {avg_iou:.3f} | {avg_jac:.3f} | "
            f"{s['label_agreement']}/{s['total']} ({agree_pct:.0f}%) | {s['total']} |"
        )

    lines.append("\n## Per-Clause Agreement (across all method pairs)\n")
    lines.append("| Clause | Avg Char IoU | Avg Token Jaccard | Occurrences |")
    lines.append("|--------|-------------|-------------------|-------------|")

    sorted_clauses = sorted(
        stats["clause_stats"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )
    for label, s in sorted_clauses:
        avg_iou = _mean(s["char_ious"])
        avg_jac = _mean(s["token_jaccards"])
        lines.append(f"| {label} | {avg_iou:.3f} | {avg_jac:.3f} | {s['count']} |")

    lines.append(f"\n**Total comparison rows:** {stats['total_rows']}")
    return "\n".join(lines)


def main():
    rows = load_results()
    if not rows:
        return

    stats = analyze(rows)
    report = generate_report(stats)

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
