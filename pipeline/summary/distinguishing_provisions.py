from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROUP_PERCENTILE = 0.10
SUMMARY_JSON_SCHEMA_NAME = "distinguishing_provisions_summary"
SUMMARY_JSON_SCHEMA_OBJECT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Concise text summary for the requested analysis task.",
        }
    },
    "required": ["summary"],
    "additionalProperties": False,
}


def _log(message: str) -> None:
    print(f"[distinguishing_provisions] {message}", flush=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cache_dir() -> str:
    return os.environ.get("CACHE_DIR", "").strip()


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _default_llm_output_dir() -> Path:
    root = _project_root()
    cache_dir = _default_cache_dir()
    if cache_dir:
        return (_resolve_path(cache_dir, root) / "04_generosity_llm_output" / "dol_archive").resolve()
    return (root / "outputs" / "04_generosity_llm_output" / "dol_archive").resolve()


def _default_classification_dir() -> Path:
    root = _project_root()
    cache_dir = _default_cache_dir()
    if cache_dir:
        return (_resolve_path(cache_dir, root) / "03_classification_output" / "dol_archive").resolve()
    return (root / "outputs" / "03_classification_output" / "dol_archive").resolve()


def _default_output_dir() -> Path:
    return (_project_root() / "figures").resolve()


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric != numeric:  # NaN
        return None
    return float(numeric)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_document_num(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"document_(\d+)", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _parse_segment_num(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"segment_(\d+)\.json", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _normalize_clause_type_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _normalize_verbose_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "medium", "high"}:
        return text
    return "low"


def _coerce_single_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return ""

    # Split on sentence-ending punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    first = str(parts[0]).strip() if parts else cleaned
    if not first:
        first = cleaned

    # Ensure sentence-ending punctuation.
    if first[-1] not in ".!?":
        first = first + "."
    return first


def _slugify(value: str) -> str:
    text = str(value or "").strip()
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return slug or "clause"


def _extract_clause_type(payload: dict[str, Any]) -> str:
    labels = payload.get("labels", [])
    if isinstance(labels, list):
        for label in labels:
            clause = str(label).strip()
            if clause:
                return clause
    label = str(payload.get("label", "")).strip()
    return label if label else "OTHER"


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    clean = str(text or "").strip()
    limit = max(1, int(max_chars))
    if len(clean) <= limit:
        return clean, False
    suffix = " [truncated]"
    cut = max(1, limit - len(suffix))
    return clean[:cut].rstrip() + suffix, True


def _normalize_detail_scores(raw_detail_scores: Any) -> list[dict[str, Any]]:
    payload = _safe_json_loads(raw_detail_scores)
    if not isinstance(payload, list):
        return []

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip() or f"detail_{idx}"
        score = _safe_float_or_none(item.get("score"))
        reason = str(item.get("reason", "")).strip()
        if score is None and not reason:
            continue
        out.append(
            {
                "name": name,
                "score": score,
                "reason": reason,
            }
        )
    return out


def _format_doc_row(row: dict[str, Any], *, rank: int) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "document_id": str(row.get("document_id", "")).strip(),
        "clause_composite_score": float(row.get("clause_composite_score", 0.0)),
        "segment_count": int(row.get("segment_count", 0)),
    }


def _list_segment_paths(doc_dir: Path) -> list[Path]:
    paths = [p for p in doc_dir.glob("segment_*.json") if p.is_file()]
    return sorted(paths, key=lambda p: (_parse_segment_num(p), p.name))


def _load_clause_score_rows(llm_output_dir: Path) -> list[dict[str, Any]]:
    csv_path = llm_output_dir / "document_clause_composite_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Clause score CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not isinstance(raw, dict):
                continue
            status = str(raw.get("status", "")).strip().lower()
            if status != "ok":
                continue
            document_id = str(raw.get("document_id", "")).strip()
            clause_type = str(raw.get("clause_type", "")).strip()
            score = _safe_float_or_none(raw.get("clause_composite_score"))
            if not document_id or not clause_type or score is None:
                continue
            rows.append(
                {
                    "provider": str(raw.get("provider", "")).strip(),
                    "model": str(raw.get("model", "")).strip(),
                    "document_id": document_id,
                    "clause_type": clause_type,
                    "clause_composite_score": float(score),
                    "segment_count": max(0, _safe_int(raw.get("segment_count", 0))),
                    "detail_scores": _normalize_detail_scores(raw.get("detail_scores_json")),
                    "status": status,
                }
            )
    return rows


def _resolve_canonical_clause_type(
    all_rows: list[dict[str, Any]],
    clause_type: str,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    target_key = _normalize_clause_type_key(clause_type)
    if not target_key:
        raise ValueError("clause_type must be non-empty.")

    available_labels = sorted(
        {str(row.get("clause_type", "")).strip() for row in all_rows if str(row.get("clause_type", "")).strip()},
        key=lambda s: s.lower(),
    )
    matches = [row for row in all_rows if _normalize_clause_type_key(row.get("clause_type", "")) == target_key]
    if not matches:
        sample = ", ".join(available_labels[:20])
        if len(available_labels) > 20:
            sample = f"{sample}, ..."
        raise ValueError(
            f"No rows found for clause_type `{clause_type}`. "
            f"Available clause types ({len(available_labels)}): {sample}"
        )

    label_counts = Counter(str(row.get("clause_type", "")).strip() for row in matches)
    canonical = sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))[0][0]

    matched_rows = sorted(
        matches,
        key=lambda row: (
            -float(row["clause_composite_score"]),
            _parse_document_num(str(row["document_id"])),
            str(row["document_id"]),
        ),
    )
    return canonical, matched_rows, available_labels


def _select_top_bottom_groups(
    clause_rows_desc: list[dict[str, Any]],
    *,
    percentile: float = GROUP_PERCENTILE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not clause_rows_desc:
        return [], [], 0
    size = max(1, int(math.ceil(len(clause_rows_desc) * float(percentile))))
    top_rows = list(clause_rows_desc[:size])
    top_ids = {str(row["document_id"]) for row in top_rows}

    bottom_rows: list[dict[str, Any]] = []
    for row in sorted(
        clause_rows_desc,
        key=lambda item: (
            float(item["clause_composite_score"]),
            _parse_document_num(str(item["document_id"])),
            str(item["document_id"]),
        ),
    ):
        doc_id = str(row["document_id"])
        if doc_id in top_ids:
            continue
        bottom_rows.append(row)
        if len(bottom_rows) >= size:
            break
    return top_rows, bottom_rows, size


def _collect_group_entries(
    *,
    group_rows: list[dict[str, Any]],
    max_segment_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_doc_ids = [str(row["document_id"]) for row in group_rows]
    selected_doc_set = set(selected_doc_ids)

    entries: list[dict[str, Any]] = []
    doc_ids_with_details: set[str] = set()
    raw_chars_total = 0
    excerpt_chars_total = 0
    per_detail_truncation_count = 0

    score_by_doc = {
        str(row["document_id"]): float(row["clause_composite_score"])
        for row in group_rows
    }

    for row in group_rows:
        doc_id = str(row["document_id"])
        detail_scores = row.get("detail_scores", [])
        if not isinstance(detail_scores, list) or not detail_scores:
            continue
        for detail_idx, detail in enumerate(detail_scores, start=1):
            if not isinstance(detail, dict):
                continue
            detail_name = str(detail.get("name", "")).strip() or f"detail_{detail_idx}"
            detail_score = _safe_float_or_none(detail.get("score"))
            detail_reason = str(detail.get("reason", "")).strip()
            detail_text = (
                f"name={detail_name}; "
                f"score={detail_score if detail_score is not None else 'NA'}; "
                f"reason={detail_reason or '[no reason provided]'}"
            )

            doc_ids_with_details.add(doc_id)
            raw_chars_total += len(detail_text)
            excerpt, was_truncated = _truncate_text(detail_text, max_segment_chars)
            excerpt_chars_total += len(excerpt)
            if was_truncated:
                per_detail_truncation_count += 1

            entries.append(
                {
                    "document_id": doc_id,
                    "detail_index": int(detail_idx),
                    "detail_name": detail_name,
                    "detail_score": detail_score,
                    "clause_composite_score": float(score_by_doc.get(doc_id, 0.0)),
                    "excerpt": excerpt,
                    "excerpt_char_count": int(len(excerpt)),
                    "raw_char_count": int(len(detail_text)),
                    "excerpt_was_truncated": bool(was_truncated),
                }
            )

    # Keep deterministic order aligned to ranked docs, then detail index/name.
    doc_order = {doc_id: idx for idx, doc_id in enumerate(selected_doc_ids)}
    entries.sort(
        key=lambda item: (
            doc_order.get(str(item["document_id"]), 10**9),
            int(item.get("detail_index", 10**9)),
            str(item.get("detail_name", "")),
            str(item["document_id"]),
        )
    )

    docs_without_details = [doc_id for doc_id in selected_doc_ids if doc_id not in doc_ids_with_details]
    stats = {
        "documents_selected": int(len(selected_doc_ids)),
        "documents_with_detail_scores": int(len(doc_ids_with_details)),
        "documents_without_detail_scores": docs_without_details,
        "details_before_group_limit": int(len(entries)),
        "detail_text_chars_total_raw": int(raw_chars_total),
        "detail_excerpt_chars_total_after_per_detail_truncation": int(excerpt_chars_total),
        "per_detail_truncation_count": int(per_detail_truncation_count),
        "selected_document_ids": sorted(selected_doc_set, key=_parse_document_num),
    }
    return entries, stats


def _build_corpus_with_group_limit(
    *,
    entries: list[dict[str, Any]],
    max_group_chars: int,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    limit = max(1, int(max_group_chars))
    blocks: list[str] = []
    included_entries: list[dict[str, Any]] = []
    chars_used = 0
    excluded_count = 0
    limit_hit = False

    for entry in entries:
        detail_name = str(entry.get("detail_name", "")).strip()
        detail_score = _safe_float_or_none(entry.get("detail_score"))
        if detail_name:
            detail_score_text = f"{float(detail_score):.3f}" if detail_score is not None else "NA"
            header = (
                f"[document_id={entry['document_id']} "
                f"clause_score={float(entry['clause_composite_score']):.6f} "
                f"detail={detail_name} detail_score={detail_score_text}]"
            )
        else:
            header = (
                f"[document_id={entry['document_id']} "
                f"score={float(entry['clause_composite_score']):.6f}]"
            )
        block = f"{header}\n{entry['excerpt']}"
        block_chars = len(block)

        separator_chars = 2 if blocks else 0
        if chars_used + separator_chars + block_chars > limit:
            excluded_count += 1
            limit_hit = True
            continue

        if separator_chars:
            chars_used += separator_chars
        blocks.append(block)
        included_entries.append(entry)
        chars_used += block_chars

    corpus = "\n\n".join(blocks)
    stats = {
        "group_char_limit": int(limit),
        "group_char_count_used": int(chars_used),
        "group_char_limit_hit": bool(limit_hit),
        "segments_included_in_prompt": int(len(included_entries)),
        "segments_excluded_by_group_char_limit": int(excluded_count),
        "included_excerpt_chars_total": int(sum(int(entry["excerpt_char_count"]) for entry in included_entries)),
    }
    return corpus, stats, included_entries


def _parse_json_loose(text: str) -> Any:
    payload = (text or "").strip()
    if not payload:
        raise ValueError("Empty model response text")

    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload).strip()

    try:
        return json.loads(payload)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", payload, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


def _extract_response_content_text(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                for key in ("text", "content"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
                continue
            for attr in ("text", "content"):
                value = getattr(item, attr, None)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
        return "\n".join(parts).strip()

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()
    return ""


def _extract_response_finish_reason(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    reason = getattr(response.choices[0], "finish_reason", None)
    if reason is None:
        return ""
    return str(reason).strip().lower()


def _next_retry_max_tokens(current_max_tokens: int) -> int:
    current = max(64, int(current_max_tokens))
    # Step up aggressively to recover from truncation while staying bounded.
    if current < 1024:
        return min(8192, current * 2)
    return min(8192, current + max(512, int(current * 0.75)))


def _looks_like_truncated_json(content: str, error: Exception | None, finish_reason: str) -> bool:
    if finish_reason in {"length", "max_tokens"}:
        return True
    if not content:
        return False
    text = content.strip()
    if not text.startswith("{"):
        return False
    err_text = str(error or "").lower()
    if "unterminated string" in err_text:
        return True
    if text.startswith('{"summary"') and not text.endswith("}"):
        return True
    return False


def _validate_summary_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object payload, got {type(payload)}")

    keys = set(payload.keys())
    required = {"summary"}
    if not required.issubset(keys):
        raise ValueError("JSON payload missing required field `summary`")
    extra = sorted(keys - required)
    if extra:
        raise ValueError(f"JSON payload has unexpected fields: {', '.join(extra)}")

    summary = str(payload.get("summary", "")).strip()
    if not summary:
        raise ValueError("JSON payload field `summary` is empty")
    return summary


def _chat_summary_with_retries(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    verbose: str = "low",
    stage_label: str = "OpenRouter summary call",
) -> str:
    last_error: Exception | None = None
    retries = max(1, int(max_retries))
    current_max_tokens = max(64, int(max_tokens))
    verbosity = _normalize_verbose_level(verbose)
    extra_body = {"verbosity": verbosity}
    for attempt_idx in range(retries):
        attempt = attempt_idx + 1
        _log(
            f"{stage_label}: attempt {attempt}/{retries} "
            f"(max_tokens={current_max_tokens}, verbosity={verbosity})"
        )
        response = None
        finish_reason = ""
        try:
            # Prefer strict schema-constrained output; this prevents malformed/truncated JSON objects.
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(temperature),
                max_tokens=int(current_max_tokens),
                extra_body=extra_body,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": SUMMARY_JSON_SCHEMA_NAME,
                        "strict": True,
                        "schema": SUMMARY_JSON_SCHEMA_OBJECT,
                    },
                },
            )
            _log(f"{stage_label}: response received with strict json_schema format")
        except Exception as exc_with_json_schema:
            last_error = exc_with_json_schema
            _log(
                f"{stage_label}: json_schema call failed ({type(exc_with_json_schema).__name__}); "
                "retrying with json_object format"
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=float(temperature),
                    max_tokens=int(current_max_tokens),
                    extra_body=extra_body,
                    response_format={"type": "json_object"},
                )
                _log(f"{stage_label}: response received with json_object format")
            except Exception as exc_with_json_object:
                last_error = exc_with_json_object
                _log(
                    f"{stage_label}: json_object call failed ({type(exc_with_json_object).__name__}); "
                    "retrying without response_format"
                )
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=float(temperature),
                        max_tokens=int(current_max_tokens),
                        extra_body=extra_body,
                    )
                    _log(f"{stage_label}: response received without response_format")
                except Exception as exc_without_response_format:
                    last_error = exc_without_response_format
                    _log(
                        f"{stage_label}: fallback call failed ({type(exc_without_response_format).__name__})"
                    )
                    continue

        finish_reason = _extract_response_finish_reason(response)
        if finish_reason:
            _log(f"{stage_label}: finish_reason={finish_reason}")

        # Some SDK/provider variants expose parsed payload; prefer it when available.
        parsed_payload = None
        if response and getattr(response, "choices", None):
            message = response.choices[0].message
            parsed_payload = getattr(message, "parsed", None)
        if parsed_payload is not None:
            try:
                summary = _validate_summary_payload(parsed_payload)
                _log(f"{stage_label}: success with schema-validated parsed payload")
                return summary
            except Exception as exc:
                last_error = exc
                _log(f"{stage_label}: parsed payload validation failed ({type(exc).__name__}: {exc})")

        content = _extract_response_content_text(response)
        if not content:
            last_error = ValueError("Model returned empty response text")
            _log(f"{stage_label}: empty response content")
            if _looks_like_truncated_json(content, last_error, finish_reason):
                next_tokens = _next_retry_max_tokens(current_max_tokens)
                if next_tokens > current_max_tokens:
                    _log(
                        f"{stage_label}: increasing max_tokens for retry "
                        f"({current_max_tokens} -> {next_tokens})"
                    )
                    current_max_tokens = next_tokens
            continue

        try:
            payload = _parse_json_loose(content)
        except Exception as exc:
            last_error = exc
            _log(
                f"{stage_label}: failed to parse JSON ({type(exc).__name__}: {exc})"
            )
            if _looks_like_truncated_json(content, exc, finish_reason):
                next_tokens = _next_retry_max_tokens(current_max_tokens)
                if next_tokens > current_max_tokens:
                    _log(
                        f"{stage_label}: likely truncated JSON; increasing max_tokens for retry "
                        f"({current_max_tokens} -> {next_tokens})"
                    )
                    current_max_tokens = next_tokens
            continue

        try:
            summary = _validate_summary_payload(payload)
        except Exception as exc:
            last_error = exc
            _log(f"{stage_label}: schema validation failed ({type(exc).__name__}: {exc})")
            if _looks_like_truncated_json(content, exc, finish_reason):
                next_tokens = _next_retry_max_tokens(current_max_tokens)
                if next_tokens > current_max_tokens:
                    _log(
                        f"{stage_label}: increasing max_tokens for retry "
                        f"({current_max_tokens} -> {next_tokens})"
                    )
                    current_max_tokens = next_tokens
            continue

        _log(f"{stage_label}: success with schema-validated summary")
        return summary

    raise RuntimeError(
        f"Unable to get valid JSON summary from OpenRouter after {retries} attempt(s): {last_error}"
    )


def _build_group_summary_prompt(
    *,
    clause_type: str,
    group_name: str,
    group_rows: list[dict[str, Any]],
    corpus_text: str,
    corpus_stats: dict[str, Any],
) -> tuple[str, str]:
    docs_line = ", ".join(
        f"{row['document_id']} ({float(row['clause_composite_score']):.3f})"
        for row in group_rows
    )

    system_prompt = (
        "You analyze collective bargaining agreement provisions. "
        "Summarize concrete clause patterns with focus on worker-favorable vs restrictive terms. "
        "Use only provided evidence and acknowledge uncertainty when data is sparse or truncated."
    )
    user_prompt = (
        f"Clause type: {clause_type}\n"
        f"Group: {group_name}\n"
        f"Documents and clause scores: {docs_line}\n"
        f"Corpus stats: {json.dumps(corpus_stats, ensure_ascii=False)}\n\n"
        "Extracted detail records (from clause scoring output):\n"
        f"{corpus_text}\n\n"
        "Write a concise summary with:\n"
        "1) Core recurring themes,\n"
        "2) Notable concrete terms (timelines, percentages, rights, exclusions),\n"
        "3) Overall pattern for worker favorability.\n\n"
        "Return valid JSON only with this shape:\n"
        '{"summary": "your summary text"}'
    )
    return system_prompt, user_prompt


def _build_compare_prompt(
    *,
    clause_type: str,
    top_summary: str,
    bottom_summary: str,
) -> tuple[str, str]:
    system_prompt = (
        "You compare two summaries of collective bargaining provisions. "
        "Be specific and evidence-grounded. Focus on what is missing in the weaker set."
    )
    user_prompt = (
        f"Clause type: {clause_type}\n\n"
        "Top 10% provision summary:\n"
        f"{top_summary}\n\n"
        "Bottom 10% provision summary:\n"
        f"{bottom_summary}\n\n"
        "Write exactly one concise sentence that summarizes the key difference between the groups. "
        "Explicitly highlight what the bottom group lacks relative to the top group "
        "(missing protections, weaker standards, or absent language). "
        "Do not use bullets, numbering, or multiple sentences.\n\n"
        "Return valid JSON only with this shape:\n"
        '{"summary": "your comparison text"}'
    )
    return system_prompt, user_prompt


def run(
    clause_type: str,
    llm_output_dir: Path,
    classification_dir: Path,
    output_dir: Path,
    model: str = "openai/gpt-5-mini",
    temperature: float = 0.0,
    max_tokens: int = 1400,
    max_segment_chars: int = 1400,
    max_group_chars: int = 60000,
    max_retries: int = 2,
    timeout: float = 120.0,
    verbose: str = "low",
) -> dict[str, Any]:
    verbose_level = _normalize_verbose_level(verbose)
    _log(
        f"Starting run for clause_type=`{clause_type}` "
        f"(model={model}, max_retries={int(max_retries)}, timeout={float(timeout):.1f}s, "
        f"verbose={verbose_level})"
    )
    llm_output_dir = llm_output_dir.expanduser().resolve()
    classification_dir = classification_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Resolved paths: llm_output_dir={llm_output_dir}")
    _log(f"Resolved paths: classification_dir={classification_dir}")
    _log(f"Resolved paths: output_dir={output_dir}")
    _log("Using extracted detail scores from 04_generosity_llm output (classification_dir not read).")

    _log("Loading clause score rows...")
    all_clause_rows = _load_clause_score_rows(llm_output_dir)
    _log(f"Loaded {len(all_clause_rows)} usable clause score rows")
    _log("Resolving canonical clause type and ranking documents...")
    canonical_clause_type, clause_rows_desc, available_clause_types = _resolve_canonical_clause_type(
        all_clause_rows,
        clause_type,
    )
    _log(
        f"Canonical clause type: `{canonical_clause_type}` "
        f"(documents available={len(clause_rows_desc)})"
    )
    top_rows, bottom_rows, group_size_target = _select_top_bottom_groups(clause_rows_desc, percentile=GROUP_PERCENTILE)
    _log(
        f"Selected groups: target_size={group_size_target}, "
        f"top_count={len(top_rows)}, bottom_count={len(bottom_rows)}"
    )

    if not top_rows:
        raise RuntimeError(
            f"No top-group documents selected for clause type `{canonical_clause_type}`."
        )
    if not bottom_rows:
        raise RuntimeError(
            f"No bottom-group documents selected for clause type `{canonical_clause_type}`. "
            "Need at least two distinct documents after non-overlap filtering."
        )

    _log("Collecting top-group extracted details from clause score output...")
    top_entries, top_prelimit_stats = _collect_group_entries(
        group_rows=top_rows,
        max_segment_chars=max_segment_chars,
    )
    _log(
        "Top-group details: "
        f"details={top_prelimit_stats['details_before_group_limit']}, "
        f"docs_with_details={top_prelimit_stats['documents_with_detail_scores']}"
    )
    _log("Collecting bottom-group extracted details from clause score output...")
    bottom_entries, bottom_prelimit_stats = _collect_group_entries(
        group_rows=bottom_rows,
        max_segment_chars=max_segment_chars,
    )
    _log(
        "Bottom-group details: "
        f"details={bottom_prelimit_stats['details_before_group_limit']}, "
        f"docs_with_details={bottom_prelimit_stats['documents_with_detail_scores']}"
    )

    _log("Applying group-level prompt char limits...")
    top_corpus, top_limit_stats, top_entries_included = _build_corpus_with_group_limit(
        entries=top_entries,
        max_group_chars=max_group_chars,
    )
    _log(
        "Top-group prompt corpus: "
        f"included_segments={top_limit_stats['segments_included_in_prompt']}, "
        f"excluded_by_limit={top_limit_stats['segments_excluded_by_group_char_limit']}, "
        f"chars_used={top_limit_stats['group_char_count_used']}"
    )
    bottom_corpus, bottom_limit_stats, bottom_entries_included = _build_corpus_with_group_limit(
        entries=bottom_entries,
        max_group_chars=max_group_chars,
    )
    _log(
        "Bottom-group prompt corpus: "
        f"included_segments={bottom_limit_stats['segments_included_in_prompt']}, "
        f"excluded_by_limit={bottom_limit_stats['segments_excluded_by_group_char_limit']}, "
        f"chars_used={bottom_limit_stats['group_char_count_used']}"
    )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "").strip() or DEFAULT_OPENROUTER_BASE_URL
    _log(f"Initializing OpenRouter client (base_url={base_url})")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(timeout))

    top_system, top_user = _build_group_summary_prompt(
        clause_type=canonical_clause_type,
        group_name="Top 10%",
        group_rows=top_rows,
        corpus_text=top_corpus,
        corpus_stats={**top_prelimit_stats, **top_limit_stats},
    )
    _log("LLM call 1/3: summarizing top 10% provisions...")
    top_summary = _chat_summary_with_retries(
        client=client,
        model=model,
        system_prompt=top_system,
        user_prompt=top_user,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose_level,
        stage_label="LLM call 1/3 (top summary)",
    )
    _log(f"LLM call 1/3 complete (summary_chars={len(top_summary)})")

    bottom_system, bottom_user = _build_group_summary_prompt(
        clause_type=canonical_clause_type,
        group_name="Bottom 10%",
        group_rows=bottom_rows,
        corpus_text=bottom_corpus,
        corpus_stats={**bottom_prelimit_stats, **bottom_limit_stats},
    )
    _log("LLM call 2/3: summarizing bottom 10% provisions...")
    bottom_summary = _chat_summary_with_retries(
        client=client,
        model=model,
        system_prompt=bottom_system,
        user_prompt=bottom_user,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose_level,
        stage_label="LLM call 2/3 (bottom summary)",
    )
    _log(f"LLM call 2/3 complete (summary_chars={len(bottom_summary)})")

    compare_system, compare_user = _build_compare_prompt(
        clause_type=canonical_clause_type,
        top_summary=top_summary,
        bottom_summary=bottom_summary,
    )
    _log("LLM call 3/3: compare-and-contrast (focus on what bottom lacks)...")
    comparison_summary = _chat_summary_with_retries(
        client=client,
        model=model,
        system_prompt=compare_system,
        user_prompt=compare_user,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose_level,
        stage_label="LLM call 3/3 (comparison)",
    )
    comparison_summary = _coerce_single_sentence(comparison_summary)
    _log(f"LLM call 3/3 complete (summary_chars={len(comparison_summary)})")

    top_doc_ids = {str(row["document_id"]) for row in top_rows}
    bottom_doc_ids = {str(row["document_id"]) for row in bottom_rows}
    overlap_docs = sorted(top_doc_ids.intersection(bottom_doc_ids), key=_parse_document_num)

    result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "openrouter",
        "model": str(model).strip() or "openai/gpt-5-mini",
        "verbosity": verbose_level,
        "base_url": base_url,
        "clause_type_input": str(clause_type),
        "clause_type_canonical": canonical_clause_type,
        "llm_output_dir": str(llm_output_dir),
        "classification_dir": str(classification_dir),
        "output_dir": str(output_dir),
        "score_source_csv": str(llm_output_dir / "document_clause_composite_scores.csv"),
        "documents_available_for_clause_type": int(len(clause_rows_desc)),
        "group_percentile": float(GROUP_PERCENTILE),
        "group_size_target": int(group_size_target),
        "top_group_count": int(len(top_rows)),
        "bottom_group_count": int(len(bottom_rows)),
        "top_bottom_overlap_document_ids": overlap_docs,
        "selected_top_documents": [
            _format_doc_row(row, rank=idx)
            for idx, row in enumerate(top_rows, start=1)
        ],
        "selected_bottom_documents": [
            _format_doc_row(row, rank=idx)
            for idx, row in enumerate(bottom_rows, start=1)
        ],
        "top_group_prompt_stats": {
            **top_prelimit_stats,
            **top_limit_stats,
        },
        "bottom_group_prompt_stats": {
            **bottom_prelimit_stats,
            **bottom_limit_stats,
        },
        "top_group_prompt_char_count": int(len(top_user)),
        "bottom_group_prompt_char_count": int(len(bottom_user)),
        "comparison_prompt_char_count": int(len(compare_user)),
        "top_group_segments_included_preview": int(len(top_entries_included)),
        "bottom_group_segments_included_preview": int(len(bottom_entries_included)),
        "top_summary": top_summary,
        "bottom_summary": bottom_summary,
        "comparison_summary": comparison_summary,
        "available_clause_types_count": int(len(available_clause_types)),
    }

    output_path = output_dir / f"distinguishing_provisions_{_slugify(canonical_clause_type)}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_json"] = str(output_path)

    # Keep output path persisted inside the artifact as well.
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Finished. Wrote output JSON to: {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize top and bottom 10% provisions for a clause type using OpenRouter, "
            "then compare what the bottom group lacks."
        )
    )
    parser.add_argument("--clause-type", type=str, required=True)
    parser.add_argument("--llm-output-dir", type=Path, default=_default_llm_output_dir())
    parser.add_argument("--classification-dir", type=Path, default=_default_classification_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--max-segment-chars", type=int, default=1400)
    parser.add_argument("--max-group-chars", type=int, default=60000)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--verbose",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="LLM verbosity hint passed to OpenRouter.",
    )
    args = parser.parse_args()

    summary = run(
        clause_type=args.clause_type,
        llm_output_dir=args.llm_output_dir,
        classification_dir=args.classification_dir,
        output_dir=args.output_dir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_segment_chars=args.max_segment_chars,
        max_group_chars=args.max_group_chars,
        max_retries=args.max_retries,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
