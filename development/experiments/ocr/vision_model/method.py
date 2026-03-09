"""Vision model method - pass PDF pages as PNG to a model via OpenRouter."""

import asyncio
import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

VERSIONS = {
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "gpt-5-mini": "openai/gpt-5-mini",
}
DEFAULT_VERSION = "gemini-3-flash"

PROMPT = "Extract all text from this document image exactly as it appears. Preserve the original formatting, line breaks, and structure as closely as possible. Output only the extracted text, nothing else."


def _render_page_to_png(pdf_path: str, page: int, dpi: int = 200) -> bytes:
    """Render a single PDF page to PNG bytes using pypdfium2."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    page_obj = pdf[page - 1]  # 0-indexed
    bitmap = page_obj.render(scale=dpi / 72)
    pil_image = bitmap.to_pil()

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.read()

    page_obj.close()
    pdf.close()
    return png_bytes


async def extract_page(pdf_path: str, page: int, version: str = DEFAULT_VERSION) -> str:
    """Extract text from a single PDF page using a vision model via OpenRouter.

    Args:
        pdf_path: Path to the PDF file.
        page: 1-indexed page number.
        version: Key from VERSIONS dict selecting the model.

    Returns:
        Extracted text as a string.
    """
    model = VERSIONS[version]
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=120)

    png_bytes = _render_page_to_png(pdf_path, page)
    b64_image = base64.b64encode(png_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


async def extract_document(pdf_path: str, pages: list[int] | None = None, version: str = DEFAULT_VERSION) -> dict[int, str]:
    """Extract text from specified pages of a PDF.

    Args:
        pdf_path: Path to the PDF file.
        pages: List of 1-indexed page numbers. If None, processes all pages.
        version: Key from VERSIONS dict selecting the model.

    Returns:
        Dict mapping 1-indexed page numbers to extracted text.
    """
    if pages is None:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f, strict=False)
            pages = list(range(1, len(reader.pages) + 1))

    from tqdm import tqdm

    model_name = VERSIONS[version]
    results = {}
    for page in tqdm(pages, desc=f"    {version} pages", leave=False):
        try:
            text = await extract_page(pdf_path, page, version)
            results[page] = text
        except Exception as e:
            print(f"  {version} page {page} failed: {e}")
            results[page] = ""
    return results


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    version = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VERSION
    if pdf:
        if version not in VERSIONS:
            print(f"Unknown version: {version}. Available: {list(VERSIONS.keys())}")
            sys.exit(1)
        result = asyncio.run(extract_document(pdf, version=version))
        for page_num, text in result.items():
            print(f"--- Page {page_num} ---")
            print(text[:500])
    else:
        print(f"Usage: python method.py <pdf_path> [{'/'.join(VERSIONS.keys())}]")
