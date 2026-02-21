"""Direct LLM segmentation method for clause extraction.

Feeds OCR text to an LLM via OpenRouter and asks it to identify
clause boundaries and classify segments against the taxonomy.
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_VERSION = "google/gemini-3-flash-preview"


def _build_prompt(taxonomy: list[dict]) -> str:
    lines = []
    for t in taxonomy:
        name = t["name"]
        if name == "OTHER":
            continue
        tldr = t.get("tldr", "")
        if tldr:
            lines.append(f"- {name}: {tldr}")
        else:
            lines.append(f"- {name}")

    return "\n".join([
        "You are a legal document analyst. Given a page of OCR text from a Collective Bargaining Agreement (CBA),",
        "identify all clause segments present on this page.",
        "",
        "For each clause you find:",
        "1. Classify it using ONLY the allowed labels below",
        "2. Extract the EXACT text span from the page that belongs to this clause (as much text as relevant, not just a snippet)",
        "",
        "Allowed clause labels:",
        *lines,
        "- OTHER: none of the above apply.",
        "",
        "Rules:",
        "- Use exact text from the page for extraction_text — do not paraphrase or summarize",
        "- A page may contain multiple clauses or parts of clauses",
        "- If no clauses are identifiable, return an empty hits array",
        "- Include as much of the relevant clause text as possible",
        "",
        "Return strict JSON:",
        '{"hits": [{"clause_label": "<label>", "extraction_text": "<exact text from page>"}]}',
    ])


def _parse_json_loose(text: str):
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


def _find_span(text: str, extraction_text: str) -> tuple[int | None, int | None]:
    snippet = extraction_text.strip()
    if not snippet:
        return None, None
    idx = text.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    idx = text.lower().find(snippet.lower())
    if idx >= 0:
        return idx, idx + len(snippet)
    return None, None


def _normalize_label(raw: str, canonical: dict[str, str], names: set[str]) -> str:
    s = raw.strip()
    if not s:
        return "OTHER"
    k = s.lower()
    if k in canonical:
        return canonical[k]
    k2 = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", k)).strip()
    if k2 in canonical:
        return canonical[k2]
    for candidate in names:
        if candidate.lower() in k:
            return candidate
    return "OTHER"


async def extract_page(
    text: str,
    taxonomy: list[dict],
    version: str = DEFAULT_VERSION,
) -> list[dict]:
    """Extract clause mentions from a single page of OCR text.

    Args:
        text: The OCR text content of the page.
        taxonomy: List of dicts with 'name' and 'tldr' keys.
        version: OpenRouter model identifier.

    Returns:
        List of extraction dicts with clause_label, extraction_text, start_pos, end_pos.
    """
    if not text.strip():
        return []

    feature_names = [t["name"] for t in taxonomy]
    canonical = {}
    for name in feature_names:
        canonical[name.lower()] = name
        canonical[re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()] = name
    names = set(feature_names)

    prompt = _build_prompt(taxonomy)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=120)

    response = client.chat.completions.create(
        model=version,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    output_text = response.choices[0].message.content or ""
    if not output_text.strip():
        return []

    data = _parse_json_loose(output_text)
    hits = data.get("hits", []) if isinstance(data, dict) else []

    extractions = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        raw_label = item.get("clause_label", "") or item.get("feature_name", "")
        extraction_text = item.get("extraction_text", "")
        label = _normalize_label(str(raw_label), canonical, names)
        start, end = _find_span(text, str(extraction_text))
        extractions.append({
            "clause_label": label,
            "extraction_text": str(extraction_text),
            "start_pos": start,
            "end_pos": end,
        })

    return extractions


async def extract_document(
    doc_dir: str,
    pages: list[int] | None = None,
    version: str = DEFAULT_VERSION,
    taxonomy: list[dict] | None = None,
) -> dict[int, list[dict]]:
    """Extract clauses from all pages in a document directory.

    Args:
        doc_dir: Path to ocr_output/document_* directory.
        pages: List of 1-indexed page numbers. If None, processes all pages.
        version: OpenRouter model identifier.
        taxonomy: Clause taxonomy list. If None, loads from default path.

    Returns:
        Dict mapping page numbers to lists of extraction dicts.
    """
    from pathlib import Path

    doc_path = Path(doc_dir)
    if taxonomy is None:
        from experiments.clause_extraction.langextract.method import _load_default_taxonomy
        taxonomy = _load_default_taxonomy()

    if pages is None:
        page_files = sorted(doc_path.glob("page_*.txt"))
        pages = []
        for pf in page_files:
            m = re.match(r"page_(\d+)\.txt$", pf.name)
            if m:
                pages.append(int(m.group(1)))

    from tqdm import tqdm

    results = {}
    for page_num in tqdm(pages, desc=f"    llm_segmentation pages", leave=False):
        page_file = doc_path / f"page_{page_num:04d}.txt"
        if not page_file.exists():
            results[page_num] = []
            continue
        text = page_file.read_text(encoding="utf-8", errors="replace").strip()
        try:
            extractions = await extract_page(text, taxonomy, version)
            results[page_num] = extractions
        except Exception as e:
            print(f"  llm_segmentation page {page_num} failed: {e}")
            results[page_num] = []

    return results
