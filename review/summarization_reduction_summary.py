import json
from collections import defaultdict
from pathlib import Path


CACHE_MODEL = "gpt_5_4_nano"
INPUT_DIR = Path("cache/05_summarize_output") / CACHE_MODEL


def empty_stats():
    return {
        "count": 0,
        "input_chars": 0,
        "output_chars": 0,
        "chars_reduced": 0,
        "compression_ratios": [],
    }


def add_summary(stats, category_summary):
    input_chars = category_summary.get("provisions_input_chars") or 0
    output_chars = category_summary.get("summary_output_chars") or 0
    if input_chars <= 0:
        return

    stats["count"] += 1
    stats["input_chars"] += input_chars
    stats["output_chars"] += output_chars
    stats["chars_reduced"] += input_chars - output_chars
    stats["compression_ratios"].append(output_chars / input_chars)


def finalize_stats(stats):
    count = stats["count"]
    input_chars = stats["input_chars"]
    output_chars = stats["output_chars"]
    ratios = stats["compression_ratios"]
    return {
        "count": count,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "chars_reduced": stats["chars_reduced"],
        "mean_input_chars": input_chars / count if count else 0,
        "mean_output_chars": output_chars / count if count else 0,
        "mean_chars_reduced": stats["chars_reduced"] / count if count else 0,
        "mean_compression_ratio": sum(ratios) / len(ratios) if ratios else 0,
        "overall_compression_ratio": output_chars / input_chars if input_chars else 0,
        "overall_reduction_pct": (
            100 * (1 - (output_chars / input_chars)) if input_chars else 0
        ),
    }


def collect_reduction_stats(input_dir):
    total = empty_stats()
    by_source = defaultdict(empty_stats)
    by_category = defaultdict(empty_stats)
    documents = 0

    for path in sorted(input_dir.glob("*/*.json")):
        documents += 1
        source = path.parent.name
        with path.open("r", encoding="utf-8") as f:
            document = json.load(f)

        for category_summary in document.get("category_summaries", []):
            if not category_summary.get("summary"):
                continue
            category = category_summary.get("category") or "unknown"
            add_summary(total, category_summary)
            add_summary(by_source[source], category_summary)
            add_summary(by_category[category], category_summary)

    return {
        "documents": documents,
        "total": finalize_stats(total),
        "by_source": {
            source: finalize_stats(stats)
            for source, stats in sorted(by_source.items())
        },
        "by_category": {
            category: finalize_stats(stats)
            for category, stats in sorted(by_category.items())
        },
    }


def print_stats(name, stats):
    print(
        f"{name}: {stats['count']:,} summaries, "
        f"mean input={stats['mean_input_chars']:,.1f} chars, "
        f"mean output={stats['mean_output_chars']:,.1f} chars, "
        f"mean reduction={stats['mean_chars_reduced']:,.1f} chars, "
        f"mean compression={stats['mean_compression_ratio']:.3f}, "
        f"overall reduction={stats['overall_reduction_pct']:.1f}%"
    )


def main():
    stats = collect_reduction_stats(INPUT_DIR)

    print(f"Summarization reduction summary: {CACHE_MODEL}")
    print(f"Documents scanned: {stats['documents']:,}")
    print_stats("Total", stats["total"])

    print("\nBy source")
    for source, source_stats in stats["by_source"].items():
        print_stats(source, source_stats)

    print("\nBy category")
    for category, category_stats in stats["by_category"].items():
        print_stats(category, category_stats)


if __name__ == "__main__":
    main()
