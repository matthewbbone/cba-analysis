"""Boundary-first LLM segmentation for CBA hierarchy extraction.

Pipeline:
1) Planning pass on first 10-25 pages to infer hierarchy hints.
2) Deterministic boundary candidate generation (regex + line heuristics).
3) LLM verification/normalization of boundary candidates.
4) Deterministic subsection span assembly from verified boundaries.
5) Optional fallback to coarse chunk extraction if no segments survive.
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
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []

load_dotenv()

DEFAULT_MODEL = "openai/gpt-5-mini"
DEFAULT_PLANNING_MODEL = "openai/gpt-5-mini"

DEFAULT_PLANNING_PAGES = 20
DEFAULT_PLANNING_MAX_CHARS = 80_000

DEFAULT_MAX_CHUNK_CHARS = 5_000
DEFAULT_OVERLAP_FRACTION = 0.1
DEFAULT_CONTEXT_CHARS = 500

DEFAULT_CANDIDATE_CONTEXT_CHARS = 320
DEFAULT_CANDIDATE_BATCH_SIZE = 1
DEFAULT_MIN_BOUNDARY_CONFIDENCE = 0.25
DEFAULT_MIN_SEGMENT_CHARS = 80

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")

_TOP_PATTERNS = [
    re.compile(r"^(ARTICLE|Art\.?)\b(?:[\s\-:]*[A-Z0-9IVXLCM.\-]+)?", re.IGNORECASE),
    re.compile(r"^(SECTION|Sec\.?)\b(?:[\s\-:]*[A-Z0-9IVXLCM.\-]+)?", re.IGNORECASE),
    re.compile(
        r"^(CHAPTER|PART|TITLE|APPENDIX|EXHIBIT|ADDENDUM|SCHEDULE|ATTACHMENT)\b(?:[\s\-:]*[A-Z0-9IVXLCM.\-]+)?",
        re.IGNORECASE,
    ),
    re.compile(r"^(PREAMBLE|RECITALS?|INTRODUCTION|PURPOSE|DEFINITIONS?)\b", re.IGNORECASE),
]

_SUB_PATTERNS = [
    re.compile(r"^(\d+(?:\.\d+){1,5})(?:\s|$)"),
    re.compile(r"^(\d+\.\d+|\d+\.|\d+\)|\(\d+\))\s+"),
    re.compile(r"^([A-Z]\.|[A-Z]\)|\([a-z]\)|[a-z]\.)\s+"),
    re.compile(r"^(\([ivxlcdm]+\)|[ivxlcdm]+\.|[ivxlcdm]+\))\s+", re.IGNORECASE),
    re.compile(r"^(§+\s*\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"^([A-Z]{1,3}\d+(?:\.\d+)*)\b"),  # e.g., A1.03
    re.compile(r"^(Section\s+\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"^(Subsection|Paragraph|Para\.|Clause)\s+[A-Za-z0-9]+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"^(Sec\.?\s+\d+(?:\.\d+)*)\b", re.IGNORECASE),
]

_ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9\s,&\-]{6,120}$")
_TOP_KEYWORDS = {
    "preamble",
    "recitals",
    "introduction",
    "recognition",
    "purpose",
    "scope",
    "definitions",
    "term of agreement",
    "duration",
    "management rights",
    "grievance procedure",
    "arbitration",
}


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


def _trim_span_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


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
                    "header_patterns": {"type": "array", "items": {"type": "string"}},
                    "rules": {"type": "array", "items": {"type": "string"}},
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


def _build_line_index(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    cursor = 0
    for raw in text.splitlines(keepends=True):
        clean = raw.rstrip("\r\n")
        start = cursor
        end = start + len(clean)
        lines.append({
            "start": start,
            "end": end,
            "raw": clean,
            "stripped": clean.strip(),
        })
        cursor += len(raw)

    if not lines and text:
        lines.append({
            "start": 0,
            "end": len(text),
            "raw": text,
            "stripped": text.strip(),
        })
    return lines


def _is_all_caps_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    if len(s.split()) > 16:
        return False
    if s.endswith("."):
        return False
    if not _ALL_CAPS_HEADING.match(s):
        return False
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return upper_ratio >= 0.9


def _is_title_case_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    if s.endswith("."):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*", s)
    if not words:
        return False
    if len(words) > 14:
        return False

    title_like = 0
    for w in words:
        if w[0].isupper() and not w.isupper():
            title_like += 1
    ratio = title_like / len(words)
    return ratio >= 0.65


def _contains_top_keyword(line: str) -> bool:
    low = _norm_header(line).lower()
    return low in _TOP_KEYWORDS or any(low.startswith(k + " ") for k in _TOP_KEYWORDS)


def _is_heading_like_line(line: str) -> bool:
    """High-recall heuristic for lines that look like structural headings."""
    s = line.strip()
    if not s:
        return False
    if len(s) < 3 or len(s) > 120:
        return False
    if len(s.split()) > 24:
        return False
    if s.endswith(";"):
        return False
    if "," in s and len(s.split()) > 18:
        return False
    if _contains_top_keyword(s):
        return True
    if any(p.search(s) for p in _TOP_PATTERNS):
        return True
    if any(p.search(s) for p in _SUB_PATTERNS):
        return True
    if _is_all_caps_heading(s):
        return True
    if _is_title_case_heading(s):
        return True
    # Common heading syntax: "Xxx Yyy:".
    if s.endswith(":") and len(s.split()) <= 18:
        return True
    # Numeric prefixed lines are frequently structural labels.
    if re.match(r"^[A-Za-z]?\d+(?:\.\d+){0,4}\b", s):
        return True
    # Single short line ending with period can still be a heading+micro-body.
    if s.endswith(".") and len(s.split()) <= 10:
        return True
    return False


def _line_has_boundary_context(lines: list[dict[str, Any]], idx: int) -> bool:
    """Return True when a line is visually isolated, suggesting a heading boundary."""
    prev_blank = idx == 0 or not str(lines[idx - 1].get("stripped", "")).strip()
    next_blank = idx == len(lines) - 1 or not str(lines[idx + 1].get("stripped", "")).strip()
    if prev_blank and next_blank:
        return True
    if prev_blank or next_blank:
        return True
    return False


def _compile_plan_patterns(header_patterns: list[str]) -> list[re.Pattern]:
    compiled = []
    for pat in header_patterns:
        p = str(pat or "").strip()
        if not p:
            continue
        if len(p) > 180:
            continue
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except Exception:
            continue
    return compiled


def _add_candidate(
    candidates: list[dict[str, Any]],
    seen: set[tuple[int, str]],
    line_start: int,
    heading_text: str,
    level_guess: str,
    source: str,
) -> None:
    heading_text = _norm_header(heading_text)
    if not heading_text:
        return
    norm = heading_text.lower()
    key = (line_start, norm)
    if key in seen:
        return
    seen.add(key)
    candidates.append({
        "start_pos": line_start,
        "end_pos": line_start + len(heading_text),
        "heading_text": heading_text,
        "level_guess": level_guess,
        "source": source,
    })


def _generate_boundary_candidates(
    text: str,
    hierarchy_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    _ = hierarchy_plan  # Candidate generation is intentionally text-structure-first for high recall.
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    if not text:
        return []

    starts = {0}
    for m in re.finditer(r"\n{2,}", text):
        start = m.end()
        # Move to first non-whitespace character of the next block.
        while start < len(text) and text[start] in {"\n", "\r", "\t", " "}:
            start += 1
        if start < len(text):
            starts.add(start)

    sorted_starts = sorted(starts)
    for i, start in enumerate(sorted_starts):
        block_end = len(text)
        if i + 1 < len(sorted_starts):
            block_end = sorted_starts[i + 1]
        block = text[start:block_end].strip()
        if not block:
            continue

        # Keep candidate payload short; verification gets additional surrounding context.
        first_line = block.splitlines()[0].strip() if block.splitlines() else ""
        heading = first_line or block[:160]
        _add_candidate(candidates, seen, start, heading, "sub", "double_newline_block")

    candidates = sorted(candidates, key=lambda x: (x["start_pos"], x["end_pos"]))
    for i, cand in enumerate(candidates, start=1):
        cand["candidate_id"] = str(i)
    return candidates


def _build_verify_prompt() -> str:
    return "\n".join([
        "You verify and normalize heading boundary candidates in a CBA.",
        "Each candidate is one possible heading line with nearby context.",
        "Decide if it should be used as a hierarchy boundary.",
        "Bias toward recall: keep plausible boundaries rather than dropping them.",
        "Examples that should usually be kept when structural: PREAMBLE, ARTICLE headings, short SECTION headings.",
        "If uncertain-but-plausible, keep=true with lower confidence instead of dropping.",
        "",
        "Rules:",
        "- keep=true only when candidate is a real structural heading boundary.",
        "- level must be `top` or `sub`.",
        "- top corresponds to Article/Section-level headings.",
        "- sub corresponds to subsection-level headings.",
        "- title should be the normalized heading text.",
        "- confidence is in [0,1].",
        "",
        "Return strict JSON:",
        '{"candidates":[{"candidate_id":"1","keep":true,"level":"top","title":"...","parent_hint":"","confidence":0.91}]}',
    ])


def _verify_candidates_sync(
    client: OpenAI,
    text: str,
    candidates: list[dict[str, Any]],
    model: str,
    candidate_batch_size: int,
    candidate_context_chars: int,
    min_boundary_confidence: float,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    candidate_batch_size = max(1, int(candidate_batch_size))
    candidate_context_chars = max(80, int(candidate_context_chars))
    min_boundary_confidence = max(0.0, min(1.0, float(min_boundary_confidence)))
    system_prompt = _build_verify_prompt()
    n = len(text)
    batch_starts = list(range(0, len(candidates), candidate_batch_size))

    verified: list[dict[str, Any]] = []
    for batch_start in tqdm(
        batch_starts,
        desc="  [llm_segment_v2] verifying candidate batches",
        leave=False,
    ):
        batch = candidates[batch_start: batch_start + candidate_batch_size]
        prompt_items = []
        for c in batch:
            start = int(c["start_pos"])
            end = int(c["end_pos"])
            win_start = max(0, start - candidate_context_chars)
            win_end = min(n, end + candidate_context_chars)
            context = text[win_start:win_end]
            prompt_items.append({
                "candidate_id": c["candidate_id"],
                "heading_text": c["heading_text"],
                "level_guess": c["level_guess"],
                "source": c["source"],
                "context": context,
            })

        user_prompt = "CANDIDATES_JSON:\n" + json.dumps(prompt_items, ensure_ascii=False)

        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
            temperature=0,
        )
        schema_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "verified_boundaries",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "candidate_id": {"type": "string"},
                                    "keep": {"type": "boolean"},
                                    "level": {"type": "string"},
                                    "title": {"type": "string"},
                                    "parent_hint": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["candidate_id", "keep", "level", "title", "parent_hint", "confidence"],
                            },
                        },
                    },
                    "required": ["candidates"],
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
            except Exception as exc:
                print(f"  [llm_segment_v2] verify batch failed: {exc}")
                continue

        payload = _parse_json_loose(response.choices[0].message.content or "")
        rows = payload.get("candidates", []) if isinstance(payload, dict) else []
        by_id = {c["candidate_id"]: c for c in batch}

        for item in rows:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id", "")).strip()
            if not cid or cid not in by_id:
                continue
            keep = bool(item.get("keep"))
            if not keep:
                continue
            level = str(item.get("level", "")).strip().lower()
            if level not in {"top", "sub"}:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            if confidence < min_boundary_confidence:
                continue

            src = by_id[cid]
            title = _norm_header(item.get("title", "")) or _norm_header(src.get("heading_text", ""))
            parent_hint = _norm_header(item.get("parent_hint", ""))
            verified.append({
                "candidate_id": cid,
                "start_pos": int(src["start_pos"]),
                "end_pos": int(src["end_pos"]),
                "heading_text": _norm_header(src.get("heading_text", "")),
                "level": level,
                "title": title,
                "parent_hint": parent_hint,
                "confidence": confidence,
            })

    # Deduplicate start collisions; prefer top boundaries, then higher confidence.
    best_by_start: dict[int, dict[str, Any]] = {}
    for b in verified:
        start = int(b["start_pos"])
        prev = best_by_start.get(start)
        if prev is None:
            best_by_start[start] = b
            continue
        prev_level = prev.get("level")
        cur_level = b.get("level")
        if prev_level != "top" and cur_level == "top":
            best_by_start[start] = b
            continue
        if cur_level == prev_level and float(b.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            best_by_start[start] = b

    out = sorted(best_by_start.values(), key=lambda x: (int(x["start_pos"]), int(x["end_pos"])))
    return out


def _has_structured_bullets(text: str) -> bool:
    return bool(re.search(r"\n\s*(?:[-*•]|\d+[.)]|\([a-z]\)|[A-Z][.)])\s+", text))


def _looks_like_short_sentence_segment(text: str) -> bool:
    """Allow short one-sentence substantive segments to improve recall."""
    s = _norm_header(text)
    if len(s) < 28:
        return False
    words = re.findall(r"[A-Za-z0-9]+", s)
    if len(words) < 5:
        return False
    if re.search(r"\b(shall|will|must|may|agree|entitled|prohibited|required)\b", s, flags=re.IGNORECASE):
        return True
    sentence_breaks = re.findall(r"[.!?;:]", s)
    if sentence_breaks and len(words) <= 40:
        return True
    return False


def _assemble_segments_from_boundaries(
    text: str,
    boundaries: list[dict[str, Any]],
    min_segment_chars: int,
) -> list[dict[str, Any]]:
    min_segment_chars = max(1, int(min_segment_chars))
    n = len(text)
    boundaries = sorted(boundaries, key=lambda x: int(x["start_pos"]))

    segments: list[dict[str, Any]] = []
    active_top = ""
    for i, b in enumerate(boundaries):
        level = b.get("level")
        if level == "top":
            active_top = _norm_header(b.get("title", "")) or _norm_header(b.get("heading_text", ""))
            continue
        if level != "sub":
            continue

        seg_start = int(b["start_pos"])
        seg_end = n
        for nxt in boundaries[i + 1:]:
            ns = int(nxt["start_pos"])
            if ns > seg_start:
                seg_end = ns
                break

        seg_start, seg_end = _trim_span_whitespace(text, seg_start, seg_end)
        if seg_end <= seg_start:
            continue

        seg_text = text[seg_start:seg_end]
        if len(seg_text) < min_segment_chars:
            if not (
                (_has_structured_bullets(seg_text) and len(seg_text) >= 35)
                or _looks_like_short_sentence_segment(seg_text)
            ):
                continue

        parent = active_top or _norm_header(b.get("parent_hint", "")) or "Unknown Parent"
        title = _norm_header(b.get("title", "")) or _norm_header(b.get("heading_text", "")) or "Untitled"
        segments.append({
            "parent": parent,
            "title": title,
            "text": seg_text,
            "start_pos": seg_start,
            "end_pos": seg_end,
            "boundary_source": "regex+llm",
            "boundary_confidence": float(b.get("confidence", 0.0)),
            "heading_text": _norm_header(b.get("heading_text", "")),
            "heading_level": "sub",
        })

    return segments


def _dedupe_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for seg in segments:
        start = seg.get("start_pos")
        end = seg.get("end_pos")
        if isinstance(start, int) and isinstance(end, int):
            key = ("pos", start, end)
        else:
            key = (
                "txt",
                _norm_header(seg.get("parent", "")).lower(),
                _norm_header(seg.get("title", "")).lower(),
                re.sub(r"\s+", " ", str(seg.get("text", "")).strip().lower()),
            )
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
    return out


def _enforce_non_overlap(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with_pos = [
        s for s in segments
        if isinstance(s.get("start_pos"), int) and isinstance(s.get("end_pos"), int)
    ]
    no_pos = [s for s in segments if s not in with_pos]
    with_pos = sorted(with_pos, key=lambda x: (x["start_pos"], x["end_pos"]))

    filtered: list[dict[str, Any]] = []
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


def _assign_pages(
    segments: list[dict[str, Any]],
    page_spans: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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

        enriched = dict(seg)
        enriched["start_page"] = start_page
        enriched["end_page"] = end_page
        out.append(enriched)
    return out


def _build_fallback_prompt(plan: dict[str, Any]) -> str:
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    return "\n".join([
        "Fallback segmentation pass.",
        "Extract subsection-level segments only.",
        "Return exact text spans from CURRENT_CHUNK.",
        "",
        "For each segment return:",
        "- parent",
        "- title",
        "- text",
        "- start_pos",
        "- end_pos",
        "",
        "HIERARCHY_PLAN:",
        plan_json,
        "",
        '{"segments":[{"parent":"...","title":"...","text":"...","start_pos":0,"end_pos":12}]}',
    ])


def _fallback_extract_chunk_sync(
    client: OpenAI,
    chunk_text: str,
    left_context: str,
    right_context: str,
    plan: dict[str, Any],
    model: str,
) -> list[dict[str, Any]]:
    if not chunk_text.strip():
        return []

    system_prompt = _build_fallback_prompt(plan)
    user_prompt = "\n".join([
        "Use left/right context only for boundary disambiguation.",
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0,
    )

    schema_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "fallback_segments",
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

    out: list[dict[str, Any]] = []
    n = len(chunk_text)
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        parent = _norm_header(item.get("parent", "")) or "Unknown Parent"
        title = _norm_header(item.get("title", "")) or "Untitled"
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
        start, end = _trim_span_whitespace(chunk_text, start, end)
        if end <= start:
            continue
        exact = chunk_text[start:end]
        if not exact.strip():
            continue
        out.append({
            "parent": parent,
            "title": title,
            "text": exact,
            "start_pos": int(start),
            "end_pos": int(end),
            "boundary_source": "fallback_v1",
            "heading_level": "sub",
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
    candidate_context_chars: int = DEFAULT_CANDIDATE_CONTEXT_CHARS,
    candidate_batch_size: int = DEFAULT_CANDIDATE_BATCH_SIZE,
    min_boundary_confidence: float = DEFAULT_MIN_BOUNDARY_CONFIDENCE,
    min_segment_chars: int = DEFAULT_MIN_SEGMENT_CHARS,
    enable_v1_fallback: bool = True,
) -> dict[str, Any]:
    """Segment a document into subsection records using boundary-first v2."""
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
            "stats": {
                "input_chars": 0,
                "planning_pages_used": 0,
                "planning_chars": 0,
                "candidate_count_raw": 0,
                "candidate_count_verified": 0,
                "segments_before_dedupe": 0,
                "segments_after_dedupe": 0,
                "used_fallback": False,
            },
        }

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
        "  [llm_segment_v2] planning pass "
        f"model={planning_model} on {planning_count} pages ({len(planning_text)} chars)"
    )
    hierarchy_plan = await asyncio.to_thread(
        _plan_hierarchy_sync,
        client,
        planning_text,
        planning_model,
    )

    candidates = _generate_boundary_candidates(
        text=combined_text,
        hierarchy_plan=hierarchy_plan,
    )
    print(f"  [llm_segment_v2] generated {len(candidates)} raw boundary candidates")

    verified_boundaries = await asyncio.to_thread(
        _verify_candidates_sync,
        client,
        combined_text,
        candidates,
        model,
        candidate_batch_size,
        candidate_context_chars,
        min_boundary_confidence,
    )
    print(f"  [llm_segment_v2] verified {len(verified_boundaries)} boundary candidates")

    assembled = _assemble_segments_from_boundaries(
        text=combined_text,
        boundaries=verified_boundaries,
        min_segment_chars=min_segment_chars,
    )
    segments_before_dedupe = len(assembled)
    deduped = _dedupe_segments(assembled)
    non_overlap = _enforce_non_overlap(deduped)
    final_segments = _assign_pages(non_overlap, page_spans=page_spans)
    used_fallback = False

    if not final_segments and enable_v1_fallback:
        used_fallback = True
        print("  [llm_segment_v2] no v2 segments; running fallback coarse pass")
        chunks = _chunk_text(
            text=combined_text,
            max_chars=max_chunk_chars,
            overlap_fraction=overlap_fraction,
        )
        fallback_raw: list[dict[str, Any]] = []
        for i, (start, end, chunk_text) in enumerate(chunks, start=1):
            left = combined_text[max(0, start - context_chars):start]
            right = combined_text[end:min(len(combined_text), end + context_chars)]
            try:
                hits = await asyncio.to_thread(
                    _fallback_extract_chunk_sync,
                    client,
                    chunk_text,
                    left,
                    right,
                    hierarchy_plan,
                    model,
                )
            except Exception as exc:
                print(f"  [llm_segment_v2] fallback chunk {i}/{len(chunks)} failed: {exc}")
                continue
            for h in hits:
                fallback_raw.append({
                    "parent": h["parent"],
                    "title": h["title"],
                    "text": h["text"],
                    "start_pos": start + int(h["start_pos"]),
                    "end_pos": start + int(h["end_pos"]),
                    "boundary_source": "fallback_v1",
                    "heading_level": h.get("heading_level", "sub"),
                })

        segments_before_dedupe = len(fallback_raw)
        deduped_fallback = _dedupe_segments(fallback_raw)
        non_overlap_fallback = _enforce_non_overlap(deduped_fallback)
        final_segments = _assign_pages(non_overlap_fallback, page_spans=page_spans)

    print(
        "  [llm_segment_v2] "
        f"{len(final_segments)} segments extracted (fallback={used_fallback})"
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
            "planning_pages_used": planning_count,
            "planning_chars": len(planning_text),
            "candidate_count_raw": len(candidates),
            "candidate_count_verified": len(verified_boundaries),
            "segments_before_dedupe": segments_before_dedupe,
            "segments_after_dedupe": len(final_segments),
            "used_fallback": used_fallback,
        },
    }
