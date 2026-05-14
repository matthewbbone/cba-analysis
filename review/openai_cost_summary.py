import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.utils.llm import MODEL_PRICING


TOKEN_FIELDS = [
    "input_tokens",
    "cached_input_tokens",
    "billable_input_tokens",
    "output_tokens",
    "total_tokens",
]


def empty_totals():
    return {
        "api_calls": 0,
        **{field: 0 for field in TOKEN_FIELDS},
    }


def add_usage(totals, usage):
    if not isinstance(usage, dict):
        return

    totals["api_calls"] += 1
    for field in TOKEN_FIELDS:
        totals[field] += usage.get(field, 0) or 0


def estimate_costs(totals):
    estimates = {}
    for model, pricing in MODEL_PRICING.items():
        input_cost_usd = (
            totals["billable_input_tokens"] / 1_000_000
        ) * pricing["input"]
        cached_input_cost_usd = (
            totals["cached_input_tokens"] / 1_000_000
        ) * pricing["cached_input"]
        output_cost_usd = (totals["output_tokens"] / 1_000_000) * pricing["output"]
        estimates[model] = {
            "input_cost_usd": input_cost_usd,
            "cached_input_cost_usd": cached_input_cost_usd,
            "output_cost_usd": output_cost_usd,
            "total_cost_usd": input_cost_usd + cached_input_cost_usd + output_cost_usd,
        }
    return estimates


def with_cost_estimates(totals):
    totals = dict(totals)
    totals["cost_estimates_usd"] = estimate_costs(totals)
    return totals


def finalize_grouped_totals(grouped_totals):
    return {
        name: with_cost_estimates(totals)
        for name, totals in sorted(grouped_totals.items())
    }


def usage_model(usage):
    if not isinstance(usage, dict):
        return "unknown"
    return usage.get("model") or "unknown"


def collect_provisions_usage(root):
    by_source = defaultdict(empty_totals)
    by_cache_model = defaultdict(empty_totals)
    by_model = defaultdict(empty_totals)
    total = empty_totals()

    for path in sorted(root.glob("*/*/*.json")):
        cache_model, source = path.relative_to(root).parts[:2]
        with path.open("r", encoding="utf-8") as f:
            sections = json.load(f)

        for section in sections:
            usage = section.get("provision_extraction_usage")
            add_usage(total, usage)
            add_usage(by_source[source], usage)
            add_usage(by_cache_model[cache_model], usage)
            add_usage(by_model[usage_model(usage)], usage)

    return {
        "total": with_cost_estimates(total),
        "by_source": finalize_grouped_totals(by_source),
        "by_cache_model": finalize_grouped_totals(by_cache_model),
        "by_model": finalize_grouped_totals(by_model),
    }


def collect_classification_usage(root):
    by_source = defaultdict(empty_totals)
    by_cache_model = defaultdict(empty_totals)
    by_model = defaultdict(empty_totals)
    total = empty_totals()

    for path in sorted(root.glob("*/*/*.json")):
        cache_model, source = path.relative_to(root).parts[:2]
        with path.open("r", encoding="utf-8") as f:
            sections = json.load(f)

        for section in sections:
            for provision in section.get("extracted_provisions", []):
                usage = provision.get("classification_usage")
                add_usage(total, usage)
                add_usage(by_source[source], usage)
                add_usage(by_cache_model[cache_model], usage)
                add_usage(by_model[usage_model(usage)], usage)

    return {
        "total": with_cost_estimates(total),
        "by_source": finalize_grouped_totals(by_source),
        "by_cache_model": finalize_grouped_totals(by_cache_model),
        "by_model": finalize_grouped_totals(by_model),
    }


def collect_summarization_usage(root):
    by_source = defaultdict(empty_totals)
    by_cache_model = defaultdict(empty_totals)
    by_category = defaultdict(empty_totals)
    by_model = defaultdict(empty_totals)
    total = empty_totals()

    for path in sorted(root.glob("*/*/*.json")):
        cache_model, source = path.relative_to(root).parts[:2]
        with path.open("r", encoding="utf-8") as f:
            document = json.load(f)

        for category_summary in document.get("category_summaries", []):
            category = category_summary.get("category") or "unknown"
            usage = category_summary.get("summarization_usage")
            add_usage(total, usage)
            add_usage(by_source[source], usage)
            add_usage(by_cache_model[cache_model], usage)
            add_usage(by_category[category], usage)
            add_usage(by_model[usage_model(usage)], usage)

    return {
        "total": with_cost_estimates(total),
        "by_source": finalize_grouped_totals(by_source),
        "by_cache_model": finalize_grouped_totals(by_cache_model),
        "by_category": finalize_grouped_totals(by_category),
        "by_model": finalize_grouped_totals(by_model),
    }


def collect_generosity_usage(root):
    by_cache_model = defaultdict(empty_totals)
    by_category = defaultdict(empty_totals)
    by_model = defaultdict(empty_totals)
    total = empty_totals()

    for path in sorted(root.glob("*/bradley_terry_results.json")):
        cache_model = path.relative_to(root).parts[0]
        with path.open("r", encoding="utf-8") as f:
            results = json.load(f)

        for comparison in results.get("comparisons", []):
            category = comparison.get("category") or "unknown"
            directional_comparisons = comparison.get("comparisons", {})
            for direction in directional_comparisons.values():
                usage = direction.get("usage")
                add_usage(total, usage)
                add_usage(by_cache_model[cache_model], usage)
                add_usage(by_category[category], usage)
                add_usage(by_model[usage_model(usage)], usage)

    return {
        "total": with_cost_estimates(total),
        "by_cache_model": finalize_grouped_totals(by_cache_model),
        "by_category": finalize_grouped_totals(by_category),
        "by_model": finalize_grouped_totals(by_model),
    }


def combine_totals(*totals):
    combined = empty_totals()
    for totals_item in totals:
        combined["api_calls"] += totals_item.get("api_calls", 0)
        for field in TOKEN_FIELDS:
            combined[field] += totals_item.get(field, 0) or 0
    return with_cost_estimates(combined)


def print_summary(summary):
    step_names = [
        "03_provisions",
        "04_classification",
        "05_summarize",
        "06_generosity",
    ]
    table_rows = []

    for model in MODEL_PRICING:
        step_costs = [
            summary[step_name]["total"]["cost_estimates_usd"][model]["total_cost_usd"]
            for step_name in step_names
        ]
        total_cost = summary["pipeline_total"]["cost_estimates_usd"][model][
            "total_cost_usd"
        ]
        table_rows.append([model, *step_costs, total_cost])

    headers = ["Model", *step_names, "Total"]
    formatted_rows = [
        [row[0], *[f"${value:,.6f}" for value in row[1:]]]
        for row in table_rows
    ]
    widths = [
        max(len(str(item)) for item in [header, *[row[index] for row in formatted_rows]])
        for index, header in enumerate(headers)
    ]

    print("OpenAI API cost summary")
    print("=======================")
    print(f"API calls: {summary['pipeline_total']['api_calls']:,}")
    print(f"Input tokens: {summary['pipeline_total']['input_tokens']:,}")
    print(f"Cached input tokens: {summary['pipeline_total']['cached_input_tokens']:,}")
    print(f"Billable input tokens: {summary['pipeline_total']['billable_input_tokens']:,}")
    print(f"Output tokens: {summary['pipeline_total']['output_tokens']:,}")
    print(f"Total tokens: {summary['pipeline_total']['total_tokens']:,}")
    print()
    print("Estimated cost by model and step")
    print(
        " | ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        )
    )
    print("-+-".join("-" * width for width in widths))
    for row in formatted_rows:
        print(
            " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )
    print()
    for step_name in step_names:
        step_total = summary[step_name]["total"]
        print(
            f"{step_name}: {step_total['api_calls']:,} calls, "
            f"{step_total['total_tokens']:,} tokens"
        )


def main():
    provisions = collect_provisions_usage(Path("cache/03_provisions_output"))
    classification = collect_classification_usage(Path("cache/04_classification_output"))
    summarization = collect_summarization_usage(Path("cache/05_summarize_output"))
    generosity = collect_generosity_usage(Path("cache/06_generosity_output"))

    summary = {
        "03_provisions": provisions,
        "04_classification": classification,
        "05_summarize": summarization,
        "06_generosity": generosity,
        "pipeline_total": combine_totals(
            provisions["total"],
            classification["total"],
            summarization["total"],
            generosity["total"],
        ),
    }

    output_path = Path("cache/openai_cost_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print_summary(summary)
    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
