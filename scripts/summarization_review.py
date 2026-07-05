import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.utils.llm import model_slug


def collect_provisions_by_category(sections: list[dict], category: str) -> list[str]:
    provisions = []
    for section in sections:
        if section.get("category") == category:
            provisions.append(section.get("content", ""))
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


def resolve_document_path(root: Path, document_id: str) -> Path:
    candidates = [
        root / f"{document_id}.json",
        root / f"{document_id}_res.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Document not found under {root}: tried "
        + ", ".join(candidate.name for candidate in candidates)
    )


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
    classified_root = Path("cache/04_classification_output") / model_cache_dir / args.source
    summarized_root = Path("cache/05_summarize_output") / model_cache_dir / args.source
    classified_path = resolve_document_path(classified_root, args.document_id)
    summarized_path = resolve_document_path(summarized_root, args.document_id)

    sections = load_json(classified_path)
    summarized_document = load_json(summarized_path)
    provisions = collect_provisions_by_category(sections, args.category)
    category_summary = find_category_summary(summarized_document, args.category)

    print(f"Source: {args.source}")
    print(f"Document: {args.document_id}")
    print(f"Classified file: {classified_path.name}")
    print(f"Summarized file: {summarized_path.name}")
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
