import json
from collections import defaultdict
from pathlib import Path


CACHE_MODEL = "gpt_5_4_nano"
INPUT_DIR = Path("cache/04_classification_output") / CACHE_MODEL
TAXONOMY_PATH = Path("references/provision_taxonomy.json")


def load_categories():
    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    return [category["name"] for category in taxonomy["categories"]]


def collect_category_coverage(input_dir, categories):
    category_chars = {category: 0 for category in categories}
    category_doc_pct_sum = {category: 0.0 for category in categories}
    category_docs = {category: set() for category in categories}

    total_chars = 0
    total_documents = 0

    for path in sorted(input_dir.glob("*/*.json")):
        total_documents += 1
        source = path.parent.name
        document_id = f"{source}/{path.stem}"
        document_categories = set()
        document_total_chars = 0
        document_category_chars = {category: 0 for category in categories}

        with path.open("r", encoding="utf-8") as f:
            sections = json.load(f)

        for section in sections:
            content = section.get("content") or ""
            char_count = len(content)
            total_chars += char_count
            document_total_chars += char_count

            category = section.get("category")
            if category not in category_chars:
                continue

            category_chars[category] += char_count
            document_category_chars[category] += char_count
            if char_count:
                document_categories.add(category)

        for category in document_categories:
            category_docs[category].add(document_id)
        if document_total_chars:
            for category in categories:
                category_doc_pct_sum[category] += (
                    100 * document_category_chars[category] / document_total_chars
                )

    rows = []
    for category in categories:
        chars = category_chars[category]
        rows.append(
            {
                "category": category,
                "chars": chars,
                "mean_doc_pct_cba_text": (
                    category_doc_pct_sum[category] / total_documents
                    if total_documents
                    else 0
                ),
                "pooled_pct_cba_text": (100 * chars / total_chars) if total_chars else 0,
                "cba_count": len(category_docs[category]),
                "pct_cbas": (
                    100 * len(category_docs[category]) / total_documents
                    if total_documents
                    else 0
                ),
            }
        )

    return {
        "total_documents": total_documents,
        "total_chars": total_chars,
        "rows": sorted(rows, key=lambda row: row["mean_doc_pct_cba_text"], reverse=True),
    }


def print_table(rows):
    headers = [
        "Category",
        "Chars",
        "Mean doc % text",
        "Pooled % text",
        "CBAs",
        "% CBAs",
    ]
    formatted_rows = [
        [
            row["category"],
            f"{row['chars']:,}",
            f"{row['mean_doc_pct_cba_text']:.2f}%",
            f"{row['pooled_pct_cba_text']:.2f}%",
            f"{row['cba_count']:,}",
            f"{row['pct_cbas']:.1f}%",
        ]
        for row in rows
    ]
    widths = [
        max(len(str(item)) for item in [header, *[row[index] for row in formatted_rows]])
        for index, header in enumerate(headers)
    ]

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


def main():
    categories = load_categories()
    coverage = collect_category_coverage(INPUT_DIR, categories)

    print(f"Classification category coverage: {CACHE_MODEL}")
    print(f"Documents scanned: {coverage['total_documents']:,}")
    print(f"Total CBA text chars: {coverage['total_chars']:,}")
    print()
    print_table(coverage["rows"])


if __name__ == "__main__":
    main()
