"""pdftotext method - use poppler-utils pdftotext to extract embedded text."""

import asyncio
import subprocess
from pathlib import Path

from pypdf import PdfReader


async def extract_page(pdf_path: str, page: int) -> str:
    """Extract text from a single PDF page using pdftotext.

    Args:
        pdf_path: Path to the PDF file.
        page: 1-indexed page number.

    Returns:
        Extracted text as a string.
    """
    result = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-layout", pdf_path, "-"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    return result.stdout


async def extract_document(pdf_path: str, pages: list[int] | None = None) -> dict[int, str]:
    """Extract text from specified pages of a PDF.

    Args:
        pdf_path: Path to the PDF file.
        pages: List of 1-indexed page numbers. If None, processes all pages.

    Returns:
        Dict mapping 1-indexed page numbers to extracted text.
    """
    if pages is None:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f, strict=False)
            pages = list(range(1, len(reader.pages) + 1))

    from tqdm import tqdm

    results = {}
    for page in tqdm(pages, desc="    pdftotext pages", leave=False):
        try:
            text = await extract_page(pdf_path, page)
            results[page] = text
        except Exception as e:
            print(f"  pdftotext page {page} failed: {e}")
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
