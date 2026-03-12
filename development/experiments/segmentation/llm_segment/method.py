"""LLM-based document segmentation for CBA hierarchy extraction.

Pass 1: infer section/subsection hierarchy from the first 10-25 pages.
Pass 2: scan the full document in overlapping chunks and extract subsection spans.
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

DEFAULT_MODEL = "openai/gpt-5-mini"
DEFAULT_PLANNING_MODEL = "openai/gpt-5-mini"
DEFAULT_PLANNING_PAGES = 20
DEFAULT_PLANNING_MAX_CHARS = 80_000
DEFAULT_MAX_CHUNK_CHARS = 10_000
DEFAULT_OVERLAP_FRACTION = 0.12
DEFAULT_CONTEXT_CHARS = 1200

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


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
    max_chars: int,
    overlap_fraction: float,
) -> list[tuple[int, int, str]]:
    if not text:
        return []

    n = len(text)
    if n <= max_chars:
        return [(0, n, text)]

    max_chars = max(1000, max_chars)
    overlap_fraction = min(0.5, max(0.0, overlap_fraction))
    overlap_chars = int(max_chars * overlap_fraction)

    chunks: list[tuple[int, int, str]] = []
    cursor = 0

    while cursor < n:
        max_end = min(n, cursor + max_chars)
        min_end = min(n, cursor + int(max_chars * 0.6))
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

        chunks.append((cursor, end, text[cursor:end]))
        if end >= n:
            break

        next_cursor = end - overlap_chars
        if next_cursor <= cursor:
            next_cursor = end
        cursor = next_cursor

    return chunks


def _norm_header(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def _find_span(text: str, snippet: str) -> tuple[int | None, int | None]:
    target = snippet.strip()
    if not target:
        return None, None

    idx = text.find(target)
    if idx >= 0:
        return idx, idx + len(target)

    idx = text.lower().find(target.lower())
    if idx >= 0:
        return idx, idx + len(target)

    return None, None


def _dedupe_segments(segments: list[dict]) -> list[dict]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict] = []

    for seg in segments:
        start = seg.get("start_pos")
        end = seg.get("end_pos")
        parent = _norm_header(seg.get("parent", ""))
        title = _norm_header(seg.get("title", ""))
        text = re.sub(r"\s+", " ", str(seg.get("text", "")).strip().lower())

        if isinstance(start, int) and isinstance(end, int):
            key = (start, end)
        else:
            key = (parent.lower(), title.lower(), text)

        if key in seen:
            continue
        seen.add(key)
        out.append(seg)

    return out


def _enforce_non_overlap(segments: list[dict]) -> list[dict]:
    with_pos = [
        s for s in segments
        if isinstance(s.get("start_pos"), int) and isinstance(s.get("end_pos"), int)
    ]
    no_pos = [s for s in segments if s not in with_pos]

    with_pos = sorted(with_pos, key=lambda x: (x["start_pos"], x["end_pos"]))
    filtered: list[dict] = []

    for seg in with_pos:
        if not filtered:
            filtered.append(seg)
            continue

        prev = filtered[-1]
        if seg["start_pos"] < prev["end_pos"]:
            prev_len = prev["end_pos"] - prev["start_pos"]
            seg_len = seg["end_pos"] - seg["start_pos"]
            if seg_len > prev_len:
                filtered[-1] = seg
            continue

        filtered.append(seg)

    return filtered + no_pos


def _build_planning_prompt() -> str:
    return "\n".join([
        "You analyze Collective Bargaining Agreement structure.",
        "Infer the practical two-tier hierarchy used by this document.",
        "Top tier is usually ARTICLE/SECTION; lower tier is subsection.",
        "Do not extract all provisions yet; only infer organization rules.",
        "",
        "Return strict JSON:",
        "{",
        '  "top_level_name": "...",',
        '  "subsection_level_name": "...",',
        '  "header_patterns": ["..."],',
        '  "rules": ["..."]',
        "}",
    ])


def _build_extraction_prompt(plan: dict[str, Any]) -> str:
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    return "\n".join([
        "Use the inferred hierarchy plan to extract subsection segments.",
        "Return subsection-level records only.",
        "For each segment, return:",
        "- parent: nearest top-level section/article heading",
        "- title: subsection heading/title",
        "- text: exact subsection text from CURRENT_CHUNK",
        "- start_pos/end_pos: 0-indexed character offsets in CURRENT_CHUNK (end exclusive)",
        "",
        "Rules:",
        "- text must be copied exactly from CURRENT_CHUNK.",
        "- No overlapping segments.",
        "- Skip tiny header-only fragments without substantive body text.",
        "- If no segment exists, return an empty segments array.",
        "",
        "HIERARCHY_PLAN:",
        plan_json,
        "",
        "Return strict JSON with this schema:",
        '{"segments":[{"parent":"...","title":"...","text":"...","start_pos":0,"end_pos":10}]}',
    ])


def _plan_hierarchy_sync(
    client: OpenAI,
    sample_text: str,
    model: str,
) -> dict[str, Any]:
    if not sample_text.strip():
        return {
            "top_level_name": "Section",
            "subsection_level_name": "Subsection",
            "header_patterns": [],
            "rules": [],
        }

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": _build_planning_prompt()},
            {"role": "user", "content": sample_text},
        ],
        max_tokens=1200,
        temperature=0,
    )

    schema_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "segmentation_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "top_level_name": {"type": "string"},
                    "subsection_level_name": {"type": "string"},
                    "header_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "rules": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "top_level_name",
                    "subsection_level_name",
                    "header_patterns",
                    "rules",
                ],
            },
        },
    }

    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format=schema_response_format,
        )
    except Exception:
        try:
            response = client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = client.chat.completions.create(**kwargs)

    payload = _parse_json_loose(response.choices[0].message.content or "")
    if not payload:
        payload = {}

    return {
        "top_level_name": str(payload.get("top_level_name", "Section")).strip() or "Section",
        "subsection_level_name": str(payload.get("subsection_level_name", "Subsection")).strip() or "Subsection",
        "header_patterns": [
            str(x).strip()
            for x in payload.get("header_patterns", [])
            if str(x).strip()
        ],
        "rules": [
            str(x).strip()
            for x in payload.get("rules", [])
            if str(x).strip()
        ],
    }


def _extract_chunk_sync(
    client: OpenAI,
    chunk_text: str,
    left_context: str,
    right_context: str,
    plan: dict[str, Any],
    model: str,
) -> list[dict]:
    if not chunk_text.strip():
        return []

    prompt = _build_extraction_prompt(plan)
    user_prompt = "\n".join([
        "Use LEFT/RIGHT context only for boundary disambiguation.",
        "Extract text from CURRENT_CHUNK only.",
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
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0,
    )

    schema_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "segmentation_hits",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "parent": {"type": "string"},
                                "title": {"type": "string"},
                                "text": {"type": "string"},
                                "start_pos": {"type": "integer"},
                                "end_pos": {"type": "integer"},
                            },
                            "required": ["parent", "title", "text", "start_pos", "end_pos"],
                        },
                    },
                },
                "required": ["segments"],
            },
        },
    }

    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format=schema_response_format,
        )
    except Exception:
        try:
            response = client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = client.chat.completions.create(**kwargs)

    payload = _parse_json_loose(response.choices[0].message.content or "")
    raw_segments = payload.get("segments", []) if isinstance(payload, dict) else []

    segments: list[dict] = []
    n = len(chunk_text)
    for item in raw_segments:
        if not isinstance(item, dict):
            continue

        parent = _norm_header(item.get("parent", ""))
        title = _norm_header(item.get("title", ""))
        body = str(item.get("text", "")).strip()
        start = item.get("start_pos")
        end = item.get("end_pos")

        try:
            start = int(start)
            end = int(end)
        except Exception:
            start = None
            end = None

        if start is None or end is None or start < 0 or end <= start or end > n:
            if body:
                start, end = _find_span(chunk_text, body)

        if start is None or end is None or start < 0 or end <= start or end > n:
            continue

        exact = chunk_text[start:end]
        if not exact.strip():
            continue

        segments.append({
            "parent": parent or "Unknown Parent",
            "title": title or "Untitled",
            "text": exact,
            "start_pos": start,
            "end_pos": end,
        })

    return segments


def _assign_pages(
    segments: list[dict],
    page_spans: list[tuple[int, int, int]],
) -> list[dict]:
    out: list[dict] = []
    for seg in segments:
        start = seg.get("start_pos")
        end = seg.get("end_pos")
        start_page = None
        end_page = None

        if isinstance(start, int) and isinstance(end, int):
            touched = [
                page_num
                for page_num, page_start, page_end in page_spans
                if start < page_end and end > page_start
            ]
            if touched:
                start_page = touched[0]
                end_page = touched[-1]

        out.append({
            "parent": seg.get("parent", ""),
            "title": seg.get("title", ""),
            "text": seg.get("text", ""),
            "start_pos": start,
            "end_pos": end,
            "start_page": start_page,
            "end_page": end_page,
        })
    return out


async def extract_document(
    doc_dir: str,
    pages: list[int] | None = None,
    version: str = "default",
    model: str = DEFAULT_MODEL,
    planning_model: str | None = None,
    planning_pages: int = DEFAULT_PLANNING_PAGES,
    planning_max_chars: int = DEFAULT_PLANNING_MAX_CHARS,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Segment a document into subsection records.

    Returns a dict with `segments` list where each item has at least:
    `{parent, title, text}` and also span/page metadata.
    """
    planning_model = planning_model or model or DEFAULT_PLANNING_MODEL
    model = model or DEFAULT_MODEL

    doc_path = Path(doc_dir)
    if pages is None:
        pages = []
        for pf in sorted(doc_path.glob("page_*.txt")):
            m = re.match(r"page_(\d+)\.txt$", pf.name)
            if m:
                pages.append(int(m.group(1)))

    page_texts: dict[int, str] = {}
    page_spans: list[tuple[int, int, int]] = []
    parts: list[str] = []
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

        if parts:
            parts.append(page_sep)
            offset += len(page_sep)
        start = offset
        parts.append(text)
        offset += len(text)
        page_spans.append((page_num, start, offset))

    combined_text = "".join(parts)
    if not combined_text.strip():
        return {
            "document_id": doc_path.name,
            "version": version,
            "model": model,
            "planning_model": planning_model,
            "hierarchy_plan": {},
            "segments": [],
            "stats": {"input_chars": 0, "chunks": 0, "planning_pages_used": 0},
        }

    # Planning pass on first 10-25 pages (or less if document is short).
    non_empty_pages = [p for p in pages if page_texts.get(p)]
    planning_pages = max(10, min(25, int(planning_pages)))
    planning_count = min(len(non_empty_pages), planning_pages)
    planning_text_parts = []
    for p in non_empty_pages[:planning_count]:
        planning_text_parts.append(f"[Page {p}]")
        planning_text_parts.append(page_texts[p])
    planning_text = "\n\n".join(planning_text_parts)
    planning_text = planning_text[: max(5_000, int(planning_max_chars))]

    client = _make_client()
    print(
        "  [llm_segment] planning pass "
        f"model={planning_model} on {planning_count} pages ({len(planning_text)} chars)"
    )
    hierarchy_plan = await asyncio.to_thread(
        _plan_hierarchy_sync,
        client,
        planning_text,
        planning_model,
    )

    chunks = _chunk_text(
        text=combined_text,
        max_chars=max_chunk_chars,
        overlap_fraction=overlap_fraction,
    )
    print(
        "  [llm_segment] extraction pass "
        f"model={model} on {len(combined_text)} chars across {len(chunks)} chunks"
    )

    raw_segments: list[dict] = []
    for i, (start, end, chunk_text) in enumerate(chunks, start=1):
        left = combined_text[max(0, start - context_chars):start]
        right = combined_text[end:min(len(combined_text), end + context_chars)]
        try:
            chunk_segments = await asyncio.to_thread(
                _extract_chunk_sync,
                client,
                chunk_text,
                left,
                right,
                hierarchy_plan,
                model,
            )
        except Exception as exc:
            print(f"  [llm_segment] chunk {i}/{len(chunks)} failed: {exc}")
            continue

        for seg in chunk_segments:
            raw_segments.append({
                "parent": seg["parent"],
                "title": seg["title"],
                "text": seg["text"],
                "start_pos": start + int(seg["start_pos"]),
                "end_pos": start + int(seg["end_pos"]),
            })

    deduped = _dedupe_segments(raw_segments)
    non_overlap = _enforce_non_overlap(deduped)
    final_segments = _assign_pages(non_overlap, page_spans=page_spans)

    print(
        "  [llm_segment] "
        f"{len(final_segments)} segments extracted after dedupe/non-overlap"
    )
    return {
        "document_id": doc_path.name,
        "version": version,
        "model": model,
        "planning_model": planning_model,
        "hierarchy_plan": hierarchy_plan,
        "segments": final_segments,
        "stats": {
            "input_chars": len(combined_text),
            "chunks": len(chunks),
            "planning_pages_used": planning_count,
            "planning_chars": len(planning_text),
        },
    }
