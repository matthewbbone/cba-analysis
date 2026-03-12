"""Analyze OCR experiment results — pairwise CER/WER comparisons."""

import json
import csv
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
OUTPUT_DIR = EXPERIMENT_DIR / "output"
RESULTS_FILE = EXPERIMENT_DIR / "results.csv"
REPORT_FILE = EXPERIMENT_DIR / "report.md"


def _load_method_text(pdf_stem: str, method: str, page: int) -> str:
    """Load cached OCR text for a specific method/pdf/page."""
    cache_file = OUTPUT_DIR / pdf_stem / f"{method}.json"
    if not cache_file.exists():
        return "(no cached output)"
    data = json.loads(cache_file.read_text())
    return data.get(str(page), "(page not found)")


def compute_agreement_scores(rows: list[dict]) -> dict[tuple[str, int], float]:
    """Compute composite agreement score per (pdf, page).

    Score = 1 - mean of (CER + WER) / 2 across all method pairs.
    Higher score = higher agreement (1.0 = perfect agreement).
    """
    page_scores = defaultdict(list)
    for row in rows:
        key = (row["pdf"], row["page"])
        page_scores[key].append((row["cer"] + row["wer"]) / 2)
    return {k: 1 - sum(v) / len(v) for k, v in page_scores.items()}


def load_results() -> list[dict]:
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["cer"] = float(row["cer"])
            row["wer"] = float(row["wer"])
            row["page"] = int(row["page"])
            row["len_a"] = int(row["len_a"])
            row["len_b"] = int(row["len_b"])
            rows.append(row)
    return rows


def analyze(rows: list[dict]) -> str:
    lines = []
    lines.append("# OCR Experiment Report\n")
    lines.append(f"Total comparisons: {len(rows)}\n")

    # --- Overall pairwise summary ---
    pair_metrics = defaultdict(lambda: {"cer": [], "wer": []})
    for row in rows:
        key = f"{row['method_a']} vs {row['method_b']}"
        pair_metrics[key]["cer"].append(row["cer"])
        pair_metrics[key]["wer"].append(row["wer"])

    lines.append("## Pairwise Summary\n")
    lines.append("| Pair | Mean CER | Median CER | Mean WER | Median WER | Pages |")
    lines.append("|------|----------|------------|----------|------------|-------|")
    for pair, metrics in sorted(pair_metrics.items()):
        cers = sorted(metrics["cer"])
        wers = sorted(metrics["wer"])
        n = len(cers)
        mean_cer = sum(cers) / n
        med_cer = cers[n // 2]
        mean_wer = sum(wers) / n
        med_wer = wers[n // 2]
        lines.append(f"| {pair} | {mean_cer:.4f} | {med_cer:.4f} | {mean_wer:.4f} | {med_wer:.4f} | {n} |")

    # --- Per-PDF summary ---
    lines.append("\n## Per-PDF Summary\n")
    pdf_metrics = defaultdict(lambda: defaultdict(lambda: {"cer": [], "wer": []}))
    for row in rows:
        key = f"{row['method_a']} vs {row['method_b']}"
        pdf_metrics[row["pdf"]][key]["cer"].append(row["cer"])
        pdf_metrics[row["pdf"]][key]["wer"].append(row["wer"])

    for pdf in sorted(pdf_metrics.keys()):
        lines.append(f"### {pdf}\n")
        lines.append("| Pair | Mean CER | Mean WER | Pages |")
        lines.append("|------|----------|----------|-------|")
        for pair, metrics in sorted(pdf_metrics[pdf].items()):
            cers = metrics["cer"]
            wers = metrics["wer"]
            n = len(cers)
            lines.append(f"| {pair} | {sum(cers)/n:.4f} | {sum(wers)/n:.4f} | {n} |")
        lines.append("")

    # --- Top disagreements per pair with text comparison ---
    pairs = sorted({(row["method_a"], row["method_b"]) for row in rows})
    for method_a, method_b in pairs:
        pair_label = f"{method_a} vs {method_b}"
        pair_rows = [r for r in rows if r["method_a"] == method_a and r["method_b"] == method_b]
        pair_rows.sort(key=lambda r: r["cer"], reverse=True)
        top5 = pair_rows[:5]

        lines.append(f"## Top 5 Disagreements — {pair_label}\n")
        for row in top5:
            pdf_stem = Path(row["pdf"]).stem
            text_a = _load_method_text(pdf_stem, method_a, row["page"])
            text_b = _load_method_text(pdf_stem, method_b, row["page"])
            lines.append(f"### {row['pdf']} — Page {row['page']} (CER: {row['cer']:.4f}, WER: {row['wer']:.4f})\n")
            lines.append(f"<details><summary><b>{method_a}</b> ({len(text_a)} chars)</summary>\n")
            lines.append(f"```\n{text_a}\n```\n")
            lines.append("</details>\n")
            lines.append(f"<details><summary><b>{method_b}</b> ({len(text_b)} chars)</summary>\n")
            lines.append(f"```\n{text_b}\n```\n")
            lines.append("</details>\n")
            lines.append("---\n")

    # --- Agreement ranking ---
    agreement = compute_agreement_scores(rows)
    ranked = sorted(agreement.items(), key=lambda x: x[1], reverse=True)

    lines.append("\n## Agreement Ranking (all pages)\n")
    lines.append("Score = 1 - mean of (CER + WER) / 2 across all pairs. Higher = more agreement.\n")
    lines.append("| Rank | PDF | Page | Agreement Score |")
    lines.append("|------|-----|------|-----------------|")
    for i, ((pdf, page), score) in enumerate(ranked, 1):
        lines.append(f"| {i} | {pdf} | {page} | {score:.4f} |")

    high_agreement = [x for x in ranked if x[1] > 0.95]
    lines.append(f"\nPages with strong agreement (score > 0.95): {len(high_agreement)} / {len(ranked)}\n")

    return "\n".join(lines)


def main():
    if not RESULTS_FILE.exists():
        print(f"No results file found at {RESULTS_FILE}. Run the experiment first.")
        return

    rows = load_results()
    report = analyze(rows)

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
