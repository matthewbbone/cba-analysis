from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


from definitions.definitions import WageScale  # noqa: E402


CACHE_DIR = Path("cache")
DEFAULT_OUTPUT_DIR = Path("validation/wage_scale_extractions")
DEFAULT_CLASSIFICATION_DIR = Path("cache/04_classification_output/gpt_5_4_nano")
DEFAULT_CATEGORY = "Compensation"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 16000
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_CONTEXT_WINDOW = 8
DEFAULT_MAX_INPUT_CHARS = 120000
DEFAULT_FULL_TEXT_DIR = Path("scripts")

SYSTEM_PROMPT = """
Extract WageScale JSON from CBA text.

The input may contain plain text and HTML tables. Read wage tables directly and
preserve only wage information explicitly stated in the agreement, including
base rates, wage ranges, effective dates, future rate tables, percentage
increases, time units, and COLA clauses.

Return only valid JSON matching the provided WageScale schema. If no wage scale
is present, return {"base_wages":{},"scheduled_raise":null,"scheduled_rate":null,"cola_clause":false}.
""".strip()

REPAIR_SYSTEM_PROMPT = """
Repair invalid WageScale JSON.

Preserve the wage interpretation in the provided model output, but fix schema
errors so the result validates against the WageScale schema. Return only valid
JSON.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract WageScale data from CBA text using the OpenAI "
            "Responses API."
        )
    )
    parser.add_argument("source", help="Cache source directory, e.g. dol_archive.")
    parser.add_argument(
        "document",
        help="Document filename or stem, e.g. document_116 or 3693ABBYY.",
    )
    parser.add_argument(
        "category",
        nargs="?",
        default=DEFAULT_CATEGORY,
        help=f"Classification category to extract from. Defaults to {DEFAULT_CATEGORY}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to call. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON output. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=DEFAULT_CLASSIFICATION_DIR,
        help=(
            "Root directory for classification outputs. Defaults to "
            f"{DEFAULT_CLASSIFICATION_DIR}."
        ),
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help=(
            "Optional plain-text source to pass to the model instead of classified "
            "OCR chunks. If omitted, scripts/{document}/full.txt is used when it "
            "exists."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens for the extraction response.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["omit", "none", "minimal", "low", "medium", "high", "xhigh"],
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Reasoning effort for GPT-5 models. Use 'omit' to leave the parameter "
            f"unset. Defaults to {DEFAULT_REASONING_EFFORT}."
        ),
    )
    parser.add_argument(
        "--extra-instructions",
        default="",
        help="Optional one-off prompt guidance for this document.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=DEFAULT_MAX_INPUT_CHARS,
        help=(
            "Maximum characters of input text to include. Defaults to "
            f"{DEFAULT_MAX_INPUT_CHARS}."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Number of neighboring chunks to scan for unclassified headers/tables "
            f"around matched category chunks. Defaults to {DEFAULT_CONTEXT_WINDOW}."
        ),
    )
    parser.add_argument(
        "--no-adjacent-context",
        action="store_true",
        help="Use only chunks directly classified as the selected category.",
    )
    return parser.parse_args()


def document_stem_from_arg(document: str) -> str:
    return document[:-4] if document.lower().endswith(".pdf") else document


def validate_source_and_document(source: str, document: str) -> str:
    if Path(source).name != source:
        raise ValueError("source must be a single cache directory name.")
    if Path(document).name != document:
        raise ValueError("document must be a filename or stem, not a path.")
    return document_stem_from_arg(document)


def resolve_classification_path(
    classification_dir: Path,
    source: str,
    document_stem: str,
) -> Path:
    candidates = [
        classification_dir / source / f"{document_stem}_res.json",
        classification_dir / source / f"{document_stem}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    examples = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No classification output found. Tried: {examples}")


def resolve_text_path(document_stem: str, text_file: Path | None) -> Path | None:
    if text_file is not None:
        if not text_file.exists():
            raise FileNotFoundError(f"No text file found: {text_file}")
        if not text_file.is_file():
            raise ValueError(f"text file path is not a file: {text_file}")
        return text_file

    candidate = DEFAULT_FULL_TEXT_DIR / document_stem / "full.txt"
    if candidate.exists():
        return candidate
    return None


def load_classification_rows(classification_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(classification_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected classification output to be a list: {classification_path}")
    return rows


def load_full_text(text_path: Path, max_input_chars: int) -> str:
    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Text file is empty: {text_path}")
    if max_input_chars > 0 and len(text) > max_input_chars:
        return text[:max_input_chars] + "\n\n[full text truncated]"
    return text


def category_matches(row: dict[str, Any], category: str) -> bool:
    return str(row.get("category") or "").strip().casefold() == category.casefold()


def should_include_adjacent_context(row: dict[str, Any]) -> bool:
    return row.get("category") is None


def collect_category_chunks(
    rows: list[dict[str, Any]],
    category: str,
    include_adjacent_context: bool,
    context_window: int,
    max_input_chars: int,
) -> str:
    matching_indexes = [
        index for index, row in enumerate(rows) if category_matches(row, category)
    ]
    if not matching_indexes:
        raise ValueError(f"No chunks classified as {category!r} were found.")

    selected_indexes = set(matching_indexes)
    if include_adjacent_context and context_window > 0:
        for index in matching_indexes:
            start = max(index - context_window, 0)
            end = min(index + context_window + 1, len(rows))
            for nearby_index in range(start, end):
                if should_include_adjacent_context(rows[nearby_index]):
                    selected_indexes.add(nearby_index)

    chunks: list[str] = []
    for index in sorted(selected_indexes):
        row = rows[index]
        content = (row.get("content") or "").strip()
        if not content:
            continue
        chunk_header = (
            f"[chunk {index} | page {row.get('page_num')} | "
            f"type={row.get('type')} | category={row.get('category')}]"
        )
        chunks.append(f"{chunk_header}\n{content}")

    context = "\n\n".join(chunks)
    if max_input_chars > 0 and len(context) > max_input_chars:
        return context[:max_input_chars] + "\n\n[classified chunk context truncated]"
    return context


def build_user_prompt(
    source: str,
    document_stem: str,
    category: str,
    input_path: Path,
    input_kind: str,
    input_context: str,
    extra_instructions: str,
) -> str:
    schema = WageScale.model_json_schema()
    prompt = {
        "task": "Find and extract the collective bargaining agreement wage scale.",
        "source": source,
        "document": document_stem,
        "category_filter": category,
        "input_kind": input_kind,
        "input_path": str(input_path),
        "response_format": "Return only valid JSON matching the WageScale schema.",
        "schema": schema,
        "notes": [
            "The input contains plain text from the CBA, not the PDF itself.",
            "Capture the most complete base wage scale available in the provided input.",
            "Some nearby unclassified headers or tables may be included only to preserve context for the selected category.",
            "Do not create factors merely because labels appear; include occupation, education, or seniority only when they distinguish different wage rates.",
            "Derive factor_values from the wage table or wage schedule being extracted, not from general classification lists, definitions, examples, or separate special-case wage scales.",
            "If factors includes occupation, education, or seniority, factor_values must include exactly the same factor keys with all values used in base_wages and scheduled_rate.",
            "Do not use pay level, grade, scale, date, table number, or legacy category as a factor. If a pay-level table gives rates and another table maps occupation/seniority to pay levels, use the pay-level table only as a lookup to fill the wage rates.",
            "When the chunks contain multiple distinct wage tables for different populations or transition rules, extract the primary wage scale rather than merging unrelated tables into one object.",
            "For base_wages entries, use min and max for stated ranges, mid for stated point wages, and effective_date when stated.",
            "Set time_unit to the unit for the wage rates using a simple singular noun such as hour, week, month, or year. Infer the unit from nearby wording such as hourly rate, weekly salary, monthly salary, or annual salary when the table omits it.",
            "Omit time_unit only when the wage values are not based on a time unit, such as percentages or relative formulas without a stated pay period.",
            "Use scheduled_raise for explicit percentage increases and scheduled_rate for explicit future wage-rate tables.",
            "If a percentage increase applies broadly, represent it once with a descriptive key such as all_workers_YYYY-MM-DD.",
            "Ignore non-wage benefits except where needed to understand a wage schedule.",
        ],
        "input_text": input_context,
    }
    if input_kind == "classified_chunks":
        prompt["notes"][0] = "The input contains classified CBA chunks, not the full PDF."
        prompt["classified_chunks"] = prompt.pop("input_text")
    if extra_instructions.strip():
        prompt["extra_instructions"] = extra_instructions.strip()
    return json.dumps(prompt, indent=2)


def build_repair_prompt(raw_text: str, error: Exception) -> str:
    prompt = {
        "task": "Repair this invalid WageScale JSON so it validates.",
        "response_format": "Return only valid JSON matching the WageScale schema.",
        "schema": WageScale.model_json_schema(),
        "validation_error": str(error),
        "repair_rules": [
            "Do not reinterpret the source contract or add a new extraction.",
            "If a factor appears in factors, factor_values must include that factor.",
            "Every factor value used in base_wages, scheduled_raise, or scheduled_rate keys must appear in factor_values.",
            "Every factor named in keys must be one of occupation, education, or seniority and must also appear in factors.",
            "Do not use pay_level, grade, date, table number, or legacy category as a factor.",
            "If time_unit is present, keep it as a simple singular noun such as hour, week, month, or year.",
        ],
        "invalid_model_output": raw_text,
    }
    return json.dumps(prompt, indent=2)


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


def response_debug_summary(response: Any) -> str:
    output_items: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        content_items: list[dict[str, Any]] = []
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "") or ""
            content_items.append(
                {
                    "type": getattr(content, "type", None),
                    "text_length": len(text),
                    "refusal": getattr(content, "refusal", None),
                }
            )
        output_items.append(
            {
                "type": getattr(item, "type", None),
                "status": getattr(item, "status", None),
                "content": content_items,
            }
        )

    summary = {
        "response_id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "error": to_plain(getattr(response, "error", None)),
        "incomplete_details": to_plain(getattr(response, "incomplete_details", None)),
        "output_text_length": len(getattr(response, "output_text", "") or ""),
        "output": output_items,
        "usage": usage_to_dict(response),
    }
    return json.dumps(summary, indent=2, ensure_ascii=False, default=str)


def usage_to_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if hasattr(usage, key)
    }


def validate_wage_scale_json(raw_text: str) -> WageScale:
    payload = json.loads(raw_text)
    return WageScale.model_validate(payload)


def call_openai_for_wage_scale(
    source: str,
    document_stem: str,
    category: str,
    input_path: Path,
    input_kind: str,
    input_context: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    extra_instructions: str,
) -> tuple[WageScale, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install it in this environment before "
            "running this script."
        ) from exc

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required. Add it to .env or export it before running."
        )

    client = OpenAI()
    request: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_user_prompt(
                            source,
                            document_stem,
                            category,
                            input_path,
                            input_kind,
                            input_context,
                            extra_instructions,
                        ),
                    },
                ],
            }
        ],
        "text": {"format": {"type": "json_object"}},
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort != "omit":
        request["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request)
    raw_text = extract_output_text(response).strip()
    if not raw_text:
        raise ValueError(
            "OpenAI response did not include any output text. "
            "This often means the response was incomplete before the model "
            "emitted visible JSON. Response summary:\n"
            f"{response_debug_summary(response)}"
        )

    try:
        wage_scale = validate_wage_scale_json(raw_text)
    except (json.JSONDecodeError, ValidationError) as exc:
        repair_request: dict[str, Any] = {
            "model": model,
            "instructions": REPAIR_SYSTEM_PROMPT,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_repair_prompt(raw_text, exc),
                        },
                    ],
                }
            ],
            "text": {"format": {"type": "json_object"}},
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort != "omit":
            repair_request["reasoning"] = {"effort": reasoning_effort}

        repair_response = client.responses.create(**repair_request)
        repair_raw_text = extract_output_text(repair_response).strip()
        if not repair_raw_text:
            preview = raw_text[:1000].replace("\n", "\\n")
            raise ValueError(
                "Model output was not valid WageScale JSON, and the repair "
                "response did not include output text. "
                f"Original raw output preview: {preview}\n"
                f"Original response summary:\n{response_debug_summary(response)}\n"
                f"Repair response summary:\n{response_debug_summary(repair_response)}"
            ) from exc

        try:
            wage_scale = validate_wage_scale_json(repair_raw_text)
        except (json.JSONDecodeError, ValidationError) as repair_exc:
            preview = raw_text[:1000].replace("\n", "\\n")
            repair_preview = repair_raw_text[:1000].replace("\n", "\\n")
            raise ValueError(
                "Model output was not valid WageScale JSON after one repair attempt. "
                f"Original raw output preview: {preview}\n"
                f"Repair raw output preview: {repair_preview}\n"
                f"Original response summary:\n{response_debug_summary(response)}\n"
                f"Repair response summary:\n{response_debug_summary(repair_response)}"
            ) from repair_exc

        return wage_scale, repair_response
    return wage_scale, response


def category_slug(category: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", category.lower())
    return re.sub(r"_+", "_", slug).strip("_") or "category"


def output_path_for(output_dir: Path, source: str, document_stem: str, category: str) -> Path:
    filename = f"{document_stem}.json"
    if category.casefold() != DEFAULT_CATEGORY.casefold():
        filename = f"{document_stem}__{category_slug(category)}.json"
    return output_dir / source / filename


def wage_scale_to_dict(wage_scale: WageScale) -> dict[str, Any]:
    payload = wage_scale.model_dump(mode="json", exclude_none=True)
    if not payload["factors"]:
        payload.pop("factors")
    if not payload["factor_values"]:
        payload.pop("factor_values")
    return payload


def write_result(
    output_path: Path,
    source: str,
    document_stem: str,
    category: str,
    input_path: Path,
    input_kind: str,
    input_context: str,
    model: str,
    wage_scale: WageScale,
    response: Any,
) -> None:
    record = {
        "source": source,
        "document": document_stem,
        "category_filter": category,
        "input_kind": input_kind,
        "input_path": str(input_path),
        "input_characters": len(input_context),
        "model": model,
        "response_id": getattr(response, "id", None),
        "usage": usage_to_dict(response),
        "wage_scale": wage_scale_to_dict(wage_scale),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    document_stem = validate_source_and_document(args.source, args.document)
    text_path = resolve_text_path(document_stem, args.text_file)
    if text_path is not None:
        input_path = text_path
        input_kind = "full_text"
        input_context = load_full_text(text_path, args.max_input_chars)
    else:
        input_path = resolve_classification_path(
            args.classification_dir,
            args.source,
            document_stem,
        )
        classification_rows = load_classification_rows(input_path)
        input_kind = "classified_chunks"
        input_context = collect_category_chunks(
            classification_rows,
            args.category,
            include_adjacent_context=not args.no_adjacent_context,
            context_window=args.context_window,
            max_input_chars=args.max_input_chars,
        )
    wage_scale, response = call_openai_for_wage_scale(
        source=args.source,
        document_stem=document_stem,
        category=args.category,
        input_path=input_path,
        input_kind=input_kind,
        input_context=input_context,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        extra_instructions=args.extra_instructions,
    )
    output_path = output_path_for(args.output_dir, args.source, document_stem, args.category)
    write_result(
        output_path=output_path,
        source=args.source,
        document_stem=document_stem,
        category=args.category,
        input_path=input_path,
        input_kind=input_kind,
        input_context=input_context,
        model=args.model,
        wage_scale=wage_scale,
        response=response,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
