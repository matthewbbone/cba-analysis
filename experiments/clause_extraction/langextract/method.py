"""LangExtract method for clause extraction.

Uses Google's langextract library (https://github.com/google/langextract)
to perform structured extraction of CBA clause mentions from OCR text.
Simplified from clause_extraction/extraction/runner.py.
"""

import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()

import langextract as lx

DEFAULT_VERSION = "gpt-5-nano"


def parse_taxonomy(path: str | Path) -> list[dict]:
    """Parse clause taxonomy from markdown file."""
    text = Path(path).read_text(encoding="utf-8")
    rows = []
    current_name = None
    current_tldr = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("--###"):
            continue
        m = re.match(r"^###\s+\d+\.\s+(.+)$", line)
        if m:
            if current_name:
                rows.append({"name": current_name, "tldr": current_tldr})
            current_name = m.group(1).strip()
            current_tldr = ""
            continue
        if current_name and line.startswith("**TLDR**"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_tldr = parts[1].strip()
            continue

    if current_name:
        rows.append({"name": current_name, "tldr": current_tldr})

    if not any(r["name"] == "OTHER" for r in rows):
        rows.append({"name": "OTHER", "tldr": "Other clause not covered by taxonomy."})

    return rows


def _load_default_taxonomy() -> list[dict]:
    taxonomy_path = Path(__file__).parent.parent.parent.parent / "references" / "feature_taxonomy_final.md"
    return parse_taxonomy(taxonomy_path)


def _build_prompt(taxonomy: list[dict]) -> str:
    """Build a langextract prompt_description listing allowed clause labels."""
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
        "Extract contract clause mentions from the provided CBA page text.",
        "Only return clauses from the allowed list below. If none match, return OTHER.",
        "Use exact source spans from the text for extraction_text.",
        "Set extraction_class to the clause label name from the list below.",
        "Allowed clauses:",
        *lines,
        "- OTHER: use only when no listed clause applies.",
    ])


def _build_examples() -> list[Any]:
    """Build few-shot examples for langextract.

    Uses extraction_class as the clause label (not attributes) to avoid
    langextract resolver issues with OpenAI models returning dicts.
    """
    return [
        lx.data.ExampleData(
            text=(
                "The Employer recognizes the Union as the exclusive bargaining "
                "representative for all employees in the bargaining unit."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="Recognition Clause",
                    extraction_text="recognizes the Union as the exclusive bargaining representative",
                ),
            ],
        ),
        lx.data.ExampleData(
            text="No employee shall be paid less than the rates listed in Appendix A wage schedule.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="Wages Clause",
                    extraction_text="paid less than the rates listed in Appendix A wage schedule",
                ),
            ],
        ),
    ]


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


def _extract_items(result: Any) -> list[Any]:
    """Return extraction objects from langextract result across schema variants."""
    if result is None:
        return []
    if hasattr(result, "extractions") and result.extractions is not None:
        return list(result.extractions)
    docs = getattr(result, "documents", None)
    if docs:
        items: list[Any] = []
        for d in docs:
            xs = getattr(d, "extractions", None)
            if xs:
                items.extend(xs)
        return items
    return []


def _parse_result(
    result: Any,
    text: str,
    canonical: dict[str, str],
    names: set[str],
) -> list[dict]:
    """Convert langextract result to our standard extraction format."""
    extractions = []

    for ex in _extract_items(result):
        data = ex.model_dump() if hasattr(ex, "model_dump") else {}

        extraction_text = (
            data.get("extraction_text")
            or getattr(ex, "extraction_text", "")
            or ""
        )

        # Get char_interval if langextract provides it
        char_interval = data.get("char_interval") or getattr(ex, "char_interval", None)
        start_pos = None
        end_pos = None
        if isinstance(char_interval, dict):
            start_pos = char_interval.get("start_pos")
            end_pos = char_interval.get("end_pos")
        elif char_interval is not None:
            start_pos = getattr(char_interval, "start_pos", None)
            end_pos = getattr(char_interval, "end_pos", None)

        # Fallback span lookup if langextract didn't align
        if (start_pos is None or end_pos is None) and extraction_text:
            start_pos, end_pos = _find_span(text, str(extraction_text))

        # Label is stored in extraction_class (not attributes) to avoid
        # langextract resolver bugs with OpenAI models and attribute dicts
        attributes = data.get("attributes") or getattr(ex, "attributes", {}) or {}
        raw_label = (
            getattr(ex, "extraction_class", "")
            or attributes.get("feature_name")
            or attributes.get("clause")
            or attributes.get("label")
            or ""
        )
        label = _normalize_label(str(raw_label), canonical, names)

        extractions.append({
            "clause_label": label,
            "extraction_text": str(extraction_text),
            "start_pos": start_pos,
            "end_pos": end_pos,
        })

    return extractions


def _build_canonical(taxonomy: list[dict]) -> tuple[dict[str, str], set[str]]:
    """Build canonical label lookup from taxonomy."""
    feature_names = [t["name"] for t in taxonomy]
    canonical = {}
    for name in feature_names:
        canonical[name.lower()] = name
        canonical[re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()] = name
    return canonical, set(feature_names)


async def extract_document(
    doc_dir: str,
    pages: list[int] | None = None,
    version: str = DEFAULT_VERSION,
    taxonomy: list[dict] | None = None,
) -> dict[int, list[dict]]:
    """Extract clauses from a whole document using a single langextract call.

    Concatenates all page texts into one document, runs lx.extract() once,
    then maps extractions back to their source pages via character offsets.

    Args:
        doc_dir: Path to ocr_output/document_* directory.
        pages: List of 1-indexed page numbers. If None, processes all pages.
        version: langextract model_id.
        taxonomy: Clause taxonomy list. If None, loads from default path.

    Returns:
        Dict mapping page numbers to lists of extraction dicts.
    """
    doc_path = Path(doc_dir)
    if taxonomy is None:
        taxonomy = _load_default_taxonomy()

    if pages is None:
        page_files = sorted(doc_path.glob("page_*.txt"))
        pages = []
        for pf in page_files:
            m = re.match(r"page_(\d+)\.txt$", pf.name)
            if m:
                pages.append(int(m.group(1)))

    # Read and concatenate all page texts, tracking page boundaries
    PAGE_SEP = "\n\n"
    page_texts: dict[int, str] = {}
    # (page_num, start_offset, end_offset) in the combined text
    page_spans: list[tuple[int, int, int]] = []
    combined_parts: list[str] = []
    offset = 0

    for page_num in pages:
        page_file = doc_path / f"page_{page_num:04d}.txt"
        if not page_file.exists():
            page_texts[page_num] = ""
            continue
        text = page_file.read_text(encoding="utf-8", errors="replace").strip()
        page_texts[page_num] = text
        if not text:
            continue

        if combined_parts:
            offset += len(PAGE_SEP)
            combined_parts.append(PAGE_SEP)

        start = offset
        combined_parts.append(text)
        offset += len(text)
        page_spans.append((page_num, start, offset))

    combined_text = "".join(combined_parts)

    # Initialize results for all pages
    results: dict[int, list[dict]] = {p: [] for p in pages}

    if not combined_text.strip():
        return results

    canonical, names = _build_canonical(taxonomy)
    prompt = _build_prompt(taxonomy)
    examples = _build_examples()

    print(f"  [langextract] Extracting from combined document ({len(combined_text)} chars, {len(page_spans)} pages)...")
    # Gemini models work natively; OpenAI models need fence_output workaround
    is_openai = version.startswith("gpt")
    extract_kwargs = dict(
        text_or_documents=combined_text,
        prompt_description=prompt,
        examples=examples,
        model_id=version,
        resolver_params={"suppress_parse_errors": True},
    )
    if is_openai:
        extract_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
        extract_kwargs["fence_output"] = True
        extract_kwargs["use_schema_constraints"] = False
    else:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            extract_kwargs["api_key"] = api_key

    result = lx.extract(**extract_kwargs)

    all_extractions = _parse_result(result, combined_text, canonical, names)

    # Map each extraction back to its source page using character offsets
    for ext in all_extractions:
        start = ext.get("start_pos")
        end = ext.get("end_pos")

        if start is not None and end is not None:
            # Find which page this span falls in
            assigned = False
            for page_num, page_start, page_end in page_spans:
                # Extraction overlaps this page
                if start < page_end and end > page_start:
                    # Convert offsets to page-local coordinates
                    local_ext = {
                        "clause_label": ext["clause_label"],
                        "extraction_text": ext["extraction_text"],
                        "start_pos": max(0, start - page_start),
                        "end_pos": min(page_end - page_start, end - page_start),
                    }
                    results[page_num].append(local_ext)
                    assigned = True
            if not assigned:
                # Fallback: assign to first page
                if page_spans:
                    results[page_spans[0][0]].append(ext)
        else:
            # No span info — try to find the extraction text in each page
            ext_text = ext.get("extraction_text", "")
            assigned = False
            for page_num, page_text in page_texts.items():
                if not page_text:
                    continue
                local_start, local_end = _find_span(page_text, ext_text)
                if local_start is not None:
                    results[page_num].append({
                        "clause_label": ext["clause_label"],
                        "extraction_text": ext_text,
                        "start_pos": local_start,
                        "end_pos": local_end,
                    })
                    assigned = True
                    break
            if not assigned and page_spans:
                results[page_spans[0][0]].append(ext)

    total = sum(len(v) for v in results.values())
    print(f"  [langextract] {total} extractions mapped across {len(pages)} pages")
    return results
