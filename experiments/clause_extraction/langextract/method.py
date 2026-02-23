"""Simple langextract-style method backed by OpenRouter.

This implementation keeps the same experiment interface while replacing the
`langextract` dependency with direct OpenRouter chat-completions calls.
It is optimized for long documents by:
- chunking combined document text with overlap,
- giving each chunk bounded left/right context,
- running a second focused pass on candidate regions,
- deduplicating overlap hits,
- mapping chunk-global spans back to page-local spans.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_VERSION = "openai/gpt-5-mini"
DEFAULT_MAX_CHAR_BUFFER = 10_000
DEFAULT_OVERLAP_FRACTION = 0.1
DEFAULT_CONTEXT_CHARS = 1000
DEFAULT_EXTRACTION_PASSES = 2
DEFAULT_REFINEMENT_MARGIN_CHARS = 500

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


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


def _build_canonical(taxonomy: list[dict]) -> tuple[dict[str, str], set[str]]:
    feature_names = [t["name"] for t in taxonomy]
    canonical = {}
    for name in feature_names:
        canonical[name.lower()] = name
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()
        canonical[key] = name
    return canonical, set(feature_names)


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


def _build_system_prompt(taxonomy: list[dict]) -> str:
    allowed = []
    for t in taxonomy:
        if t["name"] == "OTHER":
            continue
        tldr = t.get("tldr", "")
        if tldr:
            allowed.append(f"- {t['name']}: {tldr}")
        else:
            allowed.append(f"- {t['name']}")

    return "\n".join([
        "You are a legal contract clause extraction engine.",
        "Given a text chunk from a Collective Bargaining Agreement, extract clause mentions.",
        "",
        "Rules:",
        "- Use ONLY allowed clause labels.",
        "- extraction_text must be copied exactly from CURRENT_CHUNK (no paraphrase).",
        "- If no clause is present, return an empty hits array.",
        "- Keep hits high precision; do not emit vague or speculative clauses.",
        "",
        "Allowed clause labels:",
        *allowed,
        "- OTHER: only when a clear clause exists but none of the labels match.",
        "",
        "Return strict JSON with this schema:",
        '{"hits":[{"clause_label":"<label>","extraction_text":"<exact substring from CURRENT_CHUNK>"}]}',
    ])


def _make_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120)


def _parse_json_loose(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _chunk_text(
    text: str,
    max_char_buffer: int,
    overlap_fraction: float,
) -> list[tuple[int, int, str]]:
    """Split text into separator-aware chunks with overlap."""
    if not text:
        return []

    n = len(text)
    if n <= max_char_buffer:
        return [(0, n, text)]

    max_char_buffer = max(1000, max_char_buffer)
    overlap_fraction = min(0.5, max(0.0, overlap_fraction))
    overlap_chars = int(max_char_buffer * overlap_fraction)

    chunks: list[tuple[int, int, str]] = []
    cursor = 0

    while cursor < n:
        max_end = min(n, cursor + max_char_buffer)
        min_end = min(n, cursor + int(max_char_buffer * 0.6))
        end = max_end

        if max_end < n:
            window = text[cursor:max_end]
            for sep in _SEPARATORS:
                sep_idx = window.rfind(sep)
                if sep_idx >= 0:
                    candidate = cursor + sep_idx + len(sep)
                    if candidate >= min_end:
                        end = candidate
                        break

        if end <= cursor:
            end = max_end

        chunk_text = text[cursor:end]
        chunks.append((cursor, end, chunk_text))

        if end >= n:
            break

        next_cursor = end - overlap_chars
        if next_cursor <= cursor:
            next_cursor = end
        cursor = next_cursor

    return chunks


def _dedupe_extractions(extractions: list[dict]) -> list[dict]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict] = []

    for ext in extractions:
        label = ext.get("clause_label", "OTHER")
        text = str(ext.get("extraction_text", "")).strip()
        start = ext.get("start_pos")
        end = ext.get("end_pos")

        if start is not None and end is not None:
            key = (label, int(start), int(end))
        else:
            text_key = re.sub(r"\s+", " ", text.lower()).strip()
            key = (label, text_key)

        if key in seen:
            continue
        seen.add(key)
        deduped.append(ext)

    return deduped


def _merge_regions(
    regions: list[tuple[int, int]],
    max_gap: int = 200,
) -> list[tuple[int, int]]:
    """Merge overlapping or near-adjacent regions."""
    if not regions:
        return []

    regions = sorted(regions)
    merged: list[list[int]] = [[regions[0][0], regions[0][1]]]
    for start, end in regions[1:]:
        tail = merged[-1]
        if start <= tail[1] + max_gap:
            tail[1] = max(tail[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _build_refinement_chunks(
    text: str,
    seed_extractions: list[dict],
    refinement_margin_chars: int,
    refinement_chunk_size: int,
    overlap_fraction: float,
) -> list[tuple[int, int, str]]:
    """Build second-pass chunks around first-pass extraction spans."""
    n = len(text)
    if n == 0 or not seed_extractions:
        return []

    refinement_margin_chars = max(200, refinement_margin_chars)
    refinement_chunk_size = max(1000, refinement_chunk_size)

    seed_regions: list[tuple[int, int]] = []
    for ext in seed_extractions:
        start = ext.get("start_pos")
        end = ext.get("end_pos")

        if start is None or end is None:
            snippet = str(ext.get("extraction_text", "")).strip()
            if not snippet:
                continue
            start, end = _find_span(text, snippet)
            if start is None or end is None:
                continue

        start = int(start)
        end = int(end)
        if end <= start:
            continue

        seed_regions.append((
            max(0, start - refinement_margin_chars),
            min(n, end + refinement_margin_chars),
        ))

    merged_regions = _merge_regions(seed_regions, max_gap=refinement_margin_chars // 3)
    if not merged_regions:
        return []

    chunks: list[tuple[int, int, str]] = []
    seen_offsets: set[tuple[int, int]] = set()

    for region_start, region_end in merged_regions:
        region_text = text[region_start:region_end]
        region_chunks = _chunk_text(
            text=region_text,
            max_char_buffer=refinement_chunk_size,
            overlap_fraction=overlap_fraction,
        )
        for local_start, local_end, chunk_text in region_chunks:
            global_start = region_start + local_start
            global_end = region_start + local_end
            key = (global_start, global_end)
            if key in seen_offsets:
                continue
            seen_offsets.add(key)
            chunks.append((global_start, global_end, chunk_text))

    return sorted(chunks, key=lambda x: (x[0], x[1]))


def _extract_chunk_sync(
    client: OpenAI,
    chunk_text: str,
    left_context: str,
    right_context: str,
    taxonomy: list[dict],
    version: str,
    canonical: dict[str, str],
    names: set[str],
) -> list[dict]:
    if not chunk_text.strip():
        return []

    system_prompt = _build_system_prompt(taxonomy)
    user_prompt = "\n".join([
        "Use left/right context only for disambiguation.",
        "Extract only text from CURRENT_CHUNK.",
        "",
        "LEFT_CONTEXT:",
        left_context or "[none]",
        "",
        "CURRENT_CHUNK:",
        chunk_text,
        "",
        "RIGHT_CONTEXT:",
        right_context or "[none]",
    ])

    kwargs = dict(
        model=version,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0,
    )

    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content or ""
    payload = _parse_json_loose(content)
    hits = payload.get("hits", []) if isinstance(payload, dict) else []

    chunk_extractions = []
    for item in hits:
        if not isinstance(item, dict):
            continue

        raw_label = str(item.get("clause_label", "") or item.get("feature_name", ""))
        extraction_text = str(item.get("extraction_text", "")).strip()
        if not extraction_text:
            continue

        label = _normalize_label(raw_label, canonical, names)
        local_start, local_end = _find_span(chunk_text, extraction_text)

        chunk_extractions.append({
            "clause_label": label,
            "extraction_text": extraction_text,
            "start_pos": local_start,
            "end_pos": local_end,
        })

    return chunk_extractions


async def _run_chunk_pass(
    client: OpenAI,
    combined_text: str,
    chunks: list[tuple[int, int, str]],
    taxonomy: list[dict],
    version: str,
    canonical: dict[str, str],
    names: set[str],
    context_chars: int,
    pass_name: str,
) -> list[dict]:
    """Run extraction on a set of chunks and return global-span extractions."""
    pass_extractions: list[dict] = []
    for i, (start, end, chunk_text) in enumerate(chunks, start=1):
        left = combined_text[max(0, start - context_chars):start]
        right = combined_text[end:min(len(combined_text), end + context_chars)]
        try:
            chunk_hits = await asyncio.to_thread(
                _extract_chunk_sync,
                client,
                chunk_text,
                left,
                right,
                taxonomy,
                version,
                canonical,
                names,
            )
        except Exception as exc:
            print(f"  [langextract] {pass_name} chunk {i}/{len(chunks)} failed: {exc}")
            continue

        for hit in chunk_hits:
            local_start = hit.get("start_pos")
            local_end = hit.get("end_pos")
            if local_start is None or local_end is None:
                global_start = None
                global_end = None
            else:
                global_start = start + int(local_start)
                global_end = start + int(local_end)

            pass_extractions.append({
                "clause_label": hit["clause_label"],
                "extraction_text": hit["extraction_text"],
                "start_pos": global_start,
                "end_pos": global_end,
            })

    return _dedupe_extractions(pass_extractions)


async def extract_document(
    doc_dir: str,
    pages: list[int] | None = None,
    version: str = DEFAULT_VERSION,
    taxonomy: list[dict] | None = None,
    max_char_buffer: int = DEFAULT_MAX_CHAR_BUFFER,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
    extraction_passes: int = DEFAULT_EXTRACTION_PASSES,
    refinement_margin_chars: int = DEFAULT_REFINEMENT_MARGIN_CHARS,
) -> dict[int, list[dict]]:
    """Extract clauses from a whole document using OpenRouter-backed chunking.

    Supports a simple 2-pass mode:
    - pass 1: coarse extraction over full-document chunks
    - pass 2: focused extraction around pass-1 candidate spans
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

    page_texts: dict[int, str] = {}
    page_spans: list[tuple[int, int, int]] = []
    combined_parts: list[str] = []
    offset = 0
    page_sep = "\n\n"

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
            combined_parts.append(page_sep)
            offset += len(page_sep)

        start = offset
        combined_parts.append(text)
        offset += len(text)
        page_spans.append((page_num, start, offset))

    combined_text = "".join(combined_parts)
    results: dict[int, list[dict]] = {p: [] for p in pages}

    if not combined_text.strip():
        return results

    canonical, names = _build_canonical(taxonomy)
    chunks = _chunk_text(
        text=combined_text,
        max_char_buffer=max_char_buffer,
        overlap_fraction=overlap_fraction,
    )

    print(
        "  [langextract] "
        f"OpenRouter chunk extraction on {len(combined_text)} chars "
        f"across {len(chunks)} chunks (pass 1)"
    )

    client = _make_client()
    pass1_extractions = await _run_chunk_pass(
        client=client,
        combined_text=combined_text,
        chunks=chunks,
        taxonomy=taxonomy,
        version=version,
        canonical=canonical,
        names=names,
        context_chars=context_chars,
        pass_name="pass1",
    )
    global_extractions = list(pass1_extractions)

    if extraction_passes >= 2:
        refinement_chunk_size = max(2000, max_char_buffer // 2)
        pass2_chunks = _build_refinement_chunks(
            text=combined_text,
            seed_extractions=pass1_extractions,
            refinement_margin_chars=refinement_margin_chars,
            refinement_chunk_size=refinement_chunk_size,
            overlap_fraction=overlap_fraction,
        )
        if pass2_chunks:
            print(
                "  [langextract] "
                f"Running focused pass 2 on {len(pass2_chunks)} chunks "
                f"(around {len(pass1_extractions)} seed hits)"
            )
            pass2_extractions = await _run_chunk_pass(
                client=client,
                combined_text=combined_text,
                chunks=pass2_chunks,
                taxonomy=taxonomy,
                version=version,
                canonical=canonical,
                names=names,
                context_chars=context_chars,
                pass_name="pass2",
            )
            global_extractions = _dedupe_extractions(global_extractions + pass2_extractions)
        else:
            print("  [langextract] Skipping pass 2 (no candidate regions from pass 1)")

    for ext in global_extractions:
        start = ext.get("start_pos")
        end = ext.get("end_pos")

        if start is not None and end is not None:
            assigned = False
            for page_num, page_start, page_end in page_spans:
                if start < page_end and end > page_start:
                    results[page_num].append({
                        "clause_label": ext["clause_label"],
                        "extraction_text": ext["extraction_text"],
                        "start_pos": max(0, start - page_start),
                        "end_pos": min(page_end - page_start, end - page_start),
                    })
                    assigned = True

            if not assigned and page_spans:
                results[page_spans[0][0]].append(ext)
            continue

        # No global span info: best-effort assignment by local text match.
        ext_text = ext.get("extraction_text", "")
        assigned = False
        for page_num, page_text in page_texts.items():
            if not page_text:
                continue
            local_start, local_end = _find_span(page_text, ext_text)
            if local_start is None or local_end is None:
                continue
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

    for page_num in pages:
        deduped = _dedupe_extractions(results.get(page_num, []))
        results[page_num] = sorted(
            deduped,
            key=lambda x: (
                x.get("start_pos") is None,
                x.get("start_pos") if x.get("start_pos") is not None else 10**12,
                x.get("end_pos") if x.get("end_pos") is not None else 10**12,
                x.get("clause_label", ""),
            ),
        )

    total = sum(len(v) for v in results.values())
    print(f"  [langextract] {total} extractions mapped across {len(pages)} pages")
    return results
