"""OlmoOCR 2 method - adapted from clause_extraction/ocr/runner.py."""

import asyncio
import json
from pathlib import Path

from openai import OpenAI
from olmocr.pipeline import PageResult, build_page_query
from olmocr.prompts import PageResponse

MODEL_NAME = "allenai_olmOCR-2-7B-1025-GGUF"


def _parse_response(raw: str) -> PageResponse:
    """Parse model output, handling JSON or YAML-frontmatter formats."""
    try:
        return PageResponse(**json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        pass

    parts = raw.split("---")
    if len(parts) >= 3:
        meta_str = parts[1].strip()
        natural_text = "---".join(parts[2:]).strip()
        meta = {}
        for line in meta_str.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip()
                if val in ("True", "true"):
                    val = True
                elif val in ("False", "false"):
                    val = False
                elif val.isdigit():
                    val = int(val)
                meta[key.strip()] = val

        return PageResponse(
            primary_language=meta.get("primary_language", "en"),
            is_rotation_valid=meta.get("is_rotation_valid", True),
            rotation_correction=meta.get("rotation_correction", 0),
            is_table=meta.get("is_table", False),
            is_diagram=meta.get("is_diagram", False),
            natural_text=natural_text,
        )

    return PageResponse(
        primary_language="en",
        is_rotation_valid=True,
        rotation_correction=0,
        is_table=False,
        is_diagram=False,
        natural_text=raw.strip(),
    )


async def extract_page(pdf_path: str, page: int, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio") -> str:
    """Extract text from a single PDF page using OlmoOCR 2.

    Args:
        pdf_path: Path to the PDF file.
        page: 1-indexed page number.
        base_url: OpenAI-compatible API base URL.
        api_key: API key for the server.

    Returns:
        Extracted text as a string.
    """
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    query = await build_page_query(pdf_path, page=page, target_longest_image_dim=1024)
    query["model"] = MODEL_NAME

    response = client.chat.completions.create(**query)
    raw = response.choices[0].message.content or ""
    page_response = _parse_response(raw)
    return page_response.natural_text


async def extract_document(pdf_path: str, pages: list[int] | None = None, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio") -> dict[int, str]:
    """Extract text from specified pages of a PDF.

    Args:
        pdf_path: Path to the PDF file.
        pages: List of 1-indexed page numbers. If None, processes all pages.

    Returns:
        Dict mapping 1-indexed page numbers to extracted text.
    """
    if pages is None:
        from pypdf import PdfReader
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f, strict=False)
            pages = list(range(1, len(reader.pages) + 1))

    from tqdm import tqdm

    results = {}
    for page in tqdm(pages, desc="    OlmoOCR pages", leave=False):
        try:
            text = await extract_page(pdf_path, page, base_url, api_key)
            results[page] = text
        except Exception as e:
            print(f"  OlmoOCR page {page} failed: {e}")
            results[page] = ""
    return results


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    if pdf:
        result = asyncio.run(extract_document(pdf))
        for page_num, text in result.items():
            print(f"--- Page {page_num} ---")
            print(text[:500])
    else:
        print("Usage: python method.py <pdf_path>")
