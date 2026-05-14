import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.utils.llm import model_slug


def collect_provisions_by_category(sections: list[dict], category: str) -> list[str]:
    provisions = []
    for section in sections:
        for provision in section.get("extracted_provisions", []):
            if provision.get("category") == category:
                provisions.append(provision.get("span", ""))
    return provisions


def format_summarization_prompt(category: str, provisions: list[str]) -> str:
    return " ".join(
        [
            f"Summarize the following provisions related to {category}:\n\n -",
            "\n - ".join(provisions),
        ]
    )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_category_summary(summarized_document: dict, category: str) -> dict | None:
    for category_summary in summarized_document.get("category_summaries", []):
        if category_summary.get("category") == category:
            return category_summary
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print summarization prompt provisions and saved summary for one document category."
    )
    parser.add_argument("source", help="Source directory, e.g. cornell_retail_educ")
    parser.add_argument("document_id", help="Document id without .json")
    parser.add_argument("category", help="Provision category to review")
    parser.add_argument(
        "--model-name",
        default="gpt-5.4-nano",
        help="Model name used for cache lookup. Defaults to gpt-5.4-nano.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_cache_dir = model_slug(args.model_name)
    classified_path = (
        Path("cache/04_classification_output")
        / model_cache_dir
        / args.source
        / f"{args.document_id}.json"
    )
    summarized_path = (
        Path("cache/05_summarize_output")
        / model_cache_dir
        / args.source
        / f"{args.document_id}.json"
    )

    if not classified_path.exists():
        raise FileNotFoundError(f"Classified document not found: {classified_path}")
    if not summarized_path.exists():
        raise FileNotFoundError(f"Summarized document not found: {summarized_path}")

    sections = load_json(classified_path)
    summarized_document = load_json(summarized_path)
    provisions = collect_provisions_by_category(sections, args.category)
    category_summary = find_category_summary(summarized_document, args.category)

    print(f"Source: {args.source}")
    print(f"Document: {args.document_id}")
    print(f"Category: {args.category}")
    print(f"Model: {args.model_name} ({model_cache_dir})")
    print(f"Provision count: {len(provisions)}")
    print()
    print("=== Summarization Prompt Provision Text ===")
    print(format_summarization_prompt(args.category, provisions))
    print()
    print("=== Saved Summary ===")
    if category_summary is None:
        print(f"No saved summary found for category: {args.category}")
        return
    print(category_summary.get("summary", ""))
    print()
    print("=== Summary Metadata ===")
    for key in [
        "provision_count",
        "provisions_input_chars",
        "summary_output_chars",
        "compression_ratio",
        "chars_reduced",
    ]:
        print(f"{key}: {category_summary.get(key)}")


if __name__ == "__main__":
    main()
