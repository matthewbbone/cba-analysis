#!/usr/bin/env python3
"""Extract CBA clauses from OCR text using LangExtract + vLLM.

This runner reads per-page OCR text files produced by the OCR stage:
  ocr_output/document_<id>/page_<####>.txt

It extracts clause labels constrained by the taxonomy in
references/feature_taxonomy_final.md and writes a page-level CSV:
  document_id, document_page, feature_name

The runner supports resumable processing with a JSON cache and optional
sampling that prioritizes partially processed documents.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []

DEFAULT_OCR_DIR = Path("ocr_output")
DEFAULT_TAXONOMY_PATH = Path("references/feature_taxonomy_final.md")
DEFAULT_OUTPUT_CSV = Path("outputs/cba_features.csv")
DEFAULT_OUTPUT_JSONL = Path("outputs/cba_features_annotated.jsonl")
DEFAULT_CACHE_FILE = Path("outputs/extraction_processing_cache.json")
DEFAULT_DEBUG_DIR = Path("outputs/extraction_debug")
DEFAULT_MODEL_ID = "vllm:http://localhost:8000/v1"


@dataclass
class FeatureMeta:
    name: str
    tldr: str
    description: str


def parse_taxonomy(path: Path) -> list[FeatureMeta]:
    """Parse clause names + TLDR from taxonomy markdown."""
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    text = path.read_text(encoding="utf-8")
    rows: list[FeatureMeta] = []
    current_name: str | None = None
    current_tldr = ""
    current_desc = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Skip commented-out headings such as "--### 25. ..."
        if line.startswith("--###"):
            continue
        m = re.match(r"^###\s+\d+\.\s+(.+)$", line)
        if m:
            if current_name:
                rows.append(FeatureMeta(current_name, current_tldr, current_desc))
            current_name = m.group(1).strip()
            current_tldr = ""
            current_desc = ""
            continue
        if current_name and line.startswith("**TLDR**"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_tldr = parts[1].strip()
            continue
        if current_name and line.startswith("**Description**"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_desc = parts[1].strip()
            continue

    if current_name:
        rows.append(FeatureMeta(current_name, current_tldr, current_desc))

    if not any(r.name == "OTHER" for r in rows):
        rows.append(FeatureMeta("OTHER", "Other clause not covered by taxonomy.", ""))

    return rows


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"documents": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"documents": {}}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_page_number(page_file: Path) -> int | None:
    m = re.match(r"page_(\d+)\.txt$", page_file.name)
    if not m:
        return None
    return int(m.group(1))


def list_doc_dirs(ocr_dir: Path) -> list[Path]:
    return sorted(
        p for p in ocr_dir.glob("document_*") if p.is_dir() and re.match(r"^document_\d+$", p.name)
    )


def list_page_files(doc_dir: Path, max_pages: int | None = None) -> list[Path]:
    pages = sorted(
        p for p in doc_dir.glob("page_*.txt")
        if p.is_file() and parse_page_number(p) is not None
    )
    if max_pages is not None:
        pages = [p for p in pages if (parse_page_number(p) or 0) <= max_pages]
    return pages


def normalize_feature_name(raw: str, canonical: dict[str, str], names: set[str]) -> str:
    s = raw.strip()
    if not s:
        return "OTHER"
    k = s.lower()
    if k in canonical:
        return canonical[k]

    # Mild normalization to improve matching from model outputs.
    k2 = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", k)).strip()
    if k2 in canonical:
        return canonical[k2]

    # Try if output has extra wording around the clause name.
    for candidate in names:
        c = candidate.lower()
        if c in k:
            return candidate
    return "OTHER"


def build_prompt_and_examples(lx: Any, features: list[FeatureMeta]) -> tuple[str, list[Any]]:
    """Build LangExtract prompt + few-shot examples."""
    lines = []
    for f in features:
        if f.name == "OTHER":
            continue
        if f.tldr:
            lines.append(f"- {f.name}: {f.tldr}")
        else:
            lines.append(f"- {f.name}")

    prompt = "\n".join([
        "Extract contract clause mentions from the provided CBA page text.",
        "Only return clauses from the allowed list below. If none match, return OTHER.",
        "Use exact source spans from the text for extraction_text.",
        "Set extraction_class to 'clause'.",
        "Put the chosen canonical label in attributes.feature_name.",
        "Allowed clauses:",
        *lines,
        "- OTHER: use only when no listed clause applies.",
    ])

    examples = [
        lx.data.ExampleData(
            text=(
                "The Employer recognizes the Union as the exclusive bargaining "
                "representative for all employees in the bargaining unit."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="clause",
                    extraction_text="recognizes the Union as the exclusive bargaining representative",
                    attributes={"feature_name": "Recognition Clause"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="No employee shall be paid less than the rates listed in Appendix A wage schedule.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="clause",
                    extraction_text="paid less than the rates listed in Appendix A wage schedule",
                    attributes={"feature_name": "Wages Clause"},
                ),
            ],
        ),
    ]
    return prompt, examples


def build_openai_prompt(features: list[FeatureMeta]) -> str:
    lines = []
    for f in features:
        if f.name == "OTHER":
            continue
        if f.tldr:
            lines.append(f"- {f.name}: {f.tldr}")
        else:
            lines.append(f"- {f.name}")

    return "\n".join(
        [
            "You extract CBA clause labels from ONE page of OCR text.",
            "Choose only from the allowed labels below.",
            "If no label applies, return OTHER.",
            "For each detected clause, include exact quoted text from the page in extraction_text.",
            "Allowed labels:",
            *lines,
            "- OTHER: none of the above apply.",
            "Return strict JSON with shape: {\"hits\": [{\"feature_name\": \"...\", \"extraction_text\": \"...\"}]}",
        ]
    )


def get_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    try:
        for item in getattr(response, "output", []):
            contents = getattr(item, "content", None)
            if contents is None and isinstance(item, dict):
                contents = item.get("content", [])
            for content in contents or []:
                if isinstance(content, dict):
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        return text
                    value = content.get("value")
                    if value is not None:
                        return json.dumps(value)
                else:
                    text = getattr(content, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text
                    value = getattr(content, "value", None)
                    if value is not None:
                        return json.dumps(value)
    except Exception:
        pass
    return ""


def get_response_refusal(response: Any) -> str:
    """Extract refusal text if present."""
    try:
        for item in getattr(response, "output", []):
            contents = getattr(item, "content", None)
            if contents is None and isinstance(item, dict):
                contents = item.get("content", [])
            for content in contents or []:
                if isinstance(content, dict):
                    if content.get("type") == "refusal":
                        text = content.get("refusal") or content.get("text") or ""
                        if text:
                            return str(text)
                else:
                    ctype = getattr(content, "type", None)
                    if ctype == "refusal":
                        text = getattr(content, "refusal", None) or getattr(content, "text", None) or ""
                        if text:
                            return str(text)
    except Exception:
        pass
    return ""


def response_dump(response: Any) -> str:
    try:
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(response, ensure_ascii=False)
        except Exception:
            return str(response)


def parse_json_loose(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response text")
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


def extract_features_openai(
    client: Any,
    model: str,
    prompt: str,
    text: str,
    feature_names: list[str],
    canonical: dict[str, str],
    names: set[str],
    max_tokens: int,
    max_retries: int = 2,
    reasoning_effort: str = "low",
) -> tuple[list[str], list[dict[str, Any]]]:
    last_error: Exception | None = None
    last_response_dump = ""
    attempts = max(1, int(max_retries))
    token_budget = max(512, int(max_tokens))
    for _ in range(attempts):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            max_output_tokens=token_budget,
            reasoning={"effort": reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "page_features",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["hits"],
                        "properties": {
                            "hits": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["feature_name", "extraction_text"],
                                    "properties": {
                                        "feature_name": {"type": "string", "enum": feature_names},
                                        "extraction_text": {"type": "string"},
                                    },
                                },
                            }
                        },
                    },
                }
            },
        )
        last_response_dump = response_dump(response)
        incomplete_reason = None
        incomplete = getattr(response, "incomplete_details", None)
        if isinstance(incomplete, dict):
            incomplete_reason = incomplete.get("reason")
        elif incomplete is not None:
            incomplete_reason = getattr(incomplete, "reason", None)

        if incomplete_reason == "max_output_tokens":
            token_budget = min(token_budget * 2, 16384)
            last_error = ValueError("Model output truncated by max_output_tokens")
            continue

        refusal = get_response_refusal(response)
        if refusal:
            # Keep pipeline moving: map refusal to OTHER.
            return ["OTHER"], [{"raw_label": "OTHER", "label": "OTHER", "refusal": refusal}]
        text_payload = get_response_text(response)
        if not text_payload.strip():
            last_error = ValueError("Model returned empty response text")
            continue
        try:
            data = parse_json_loose(text_payload)
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise ValueError(
            f"Unable to parse OpenAI response after {attempts} attempt(s): "
            f"{last_error}. Raw response: {last_response_dump[:1000]}"
        )

    raw = data.get("hits", []) if isinstance(data, dict) else []
    labels = []
    details = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_label = item.get("feature_name", "")
        extraction_text = item.get("extraction_text", "")
        label = normalize_feature_name(str(raw_label), canonical, names)
        labels.append(label)
        details.append(
            {
                "raw_label": str(raw_label),
                "label": label,
                "extraction_text": str(extraction_text),
            }
        )
    dedup = []
    for lab in labels:
        if lab not in dedup:
            dedup.append(lab)
    return dedup, details


def backfill_cache_from_csv(cache: dict[str, Any], output_csv: Path) -> None:
    """Mark pages as processed from existing output CSV rows."""
    if not output_csv.exists():
        return
    try:
        with output_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = (row.get("document_id") or "").strip()
                page_raw = (row.get("document_page") or "").strip()
                if not doc_id or not page_raw:
                    continue
                try:
                    page_num = int(page_raw)
                except Exception:
                    continue
                doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
                done = set(doc_cache.get("processed_pages", []))
                done.add(page_num)
                doc_cache["processed_pages"] = sorted(done)
                doc_cache["last_processed_page"] = max(done)
    except Exception:
        return


def extract_items(result: Any) -> list[Any]:
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


def parse_features_from_result(
    result: Any,
    canonical: dict[str, str],
    names: set[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Map langextract output to canonical taxonomy labels."""
    labels: list[str] = []
    details: list[dict[str, Any]] = []

    for ex in extract_items(result):
        data = ex.model_dump() if hasattr(ex, "model_dump") else {}
        extraction_class = (
            data.get("extraction_class")
            or getattr(ex, "extraction_class", "")
            or ""
        )
        extraction_text = (
            data.get("extraction_text")
            or getattr(ex, "extraction_text", "")
            or ""
        )
        char_interval = data.get("char_interval") or getattr(ex, "char_interval", None)
        start_pos = None
        end_pos = None
        if isinstance(char_interval, dict):
            start_pos = char_interval.get("start_pos")
            end_pos = char_interval.get("end_pos")
        elif char_interval is not None:
            start_pos = getattr(char_interval, "start_pos", None)
            end_pos = getattr(char_interval, "end_pos", None)
        attributes = data.get("attributes") or getattr(ex, "attributes", {}) or {}
        raw_label = (
            attributes.get("feature_name")
            or attributes.get("clause")
            or attributes.get("label")
            or extraction_class
            or extraction_text
        )
        label = normalize_feature_name(str(raw_label), canonical, names)
        labels.append(label)
        details.append(
            {
                "label": label,
                "raw_label": str(raw_label),
                "extraction_class": str(extraction_class),
                "extraction_text": str(extraction_text),
                "attributes": attributes,
                "start_pos": start_pos,
                "end_pos": end_pos,
            }
        )

    # If model returned no extractions, treat page as no-rows (not OTHER).
    dedup = []
    for lab in labels:
        if lab not in dedup:
            dedup.append(lab)
    return dedup, details


def find_span(text: str, extraction_text: str) -> tuple[int | None, int | None]:
    snippet = extraction_text.strip()
    if not snippet:
        return None, None
    idx = text.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    low_text = text.lower()
    low_snippet = snippet.lower()
    idx = low_text.find(low_snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    return None, None


def build_annotated_document_dict(
    document_id: str,
    page_num: int,
    text: str,
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a LangExtract-compatible annotated document dict for JSONL output."""
    page_id = f"{document_id}_page_{page_num:04d}"
    seen: set[tuple[str, str, int | None, int | None]] = set()
    extractions: list[dict[str, Any]] = []

    for d in details:
        label = str(d.get("label", "OTHER"))
        extraction_text = str(d.get("extraction_text", "") or "").strip()
        start_pos = d.get("start_pos")
        end_pos = d.get("end_pos")

        if (start_pos is None or end_pos is None) and extraction_text:
            start_pos, end_pos = find_span(text, extraction_text)
        if not extraction_text:
            extraction_text = label
            start_pos, end_pos = find_span(text, extraction_text)

        key = (label, extraction_text, start_pos, end_pos)
        if key in seen:
            continue
        seen.add(key)

        extraction: dict[str, Any] = {
            "extraction_class": "clause",
            "extraction_text": extraction_text,
            "attributes": {"feature_name": label},
        }
        if start_pos is not None and end_pos is not None:
            extraction["char_interval"] = {"start_pos": int(start_pos), "end_pos": int(end_pos)}
            extraction["alignment_status"] = "match_exact"
        else:
            extraction["char_interval"] = None
            extraction["alignment_status"] = None
        extractions.append(extraction)

    return {
        "document_id": page_id,
        "text": text,
        "extractions": extractions,
    }


def create_model(lx: Any, model_id: str, provider: str | None, provider_kwargs: dict[str, Any]) -> Any:
    """Create a LangExtract model object when factory API is available."""
    if not hasattr(lx, "factory"):
        return None
    if not hasattr(lx.factory, "ModelConfig") or not hasattr(lx.factory, "create_model"):
        return None

    provider_candidates = []
    if provider:
        provider_candidates.append(provider)
    # Common provider string variants across releases.
    provider_candidates.extend(
        [
            "VLLMLanguageModel",
            "langextract_vllm.VLLMLanguageModel",
            None,
        ]
    )

    last_error: Exception | None = None
    seen: set[str | None] = set()
    for candidate in provider_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            config_kwargs: dict[str, Any] = {
                "model_id": model_id,
                "provider_kwargs": provider_kwargs,
            }
            if candidate:
                config_kwargs["provider"] = candidate
            config = lx.factory.ModelConfig(**config_kwargs)
            return lx.factory.create_model(config)
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CBA clauses from OCR text using LangExtract + vLLM."
    )
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR,
                        help="Directory containing document_*/page_####.txt OCR text files")
    parser.add_argument("--taxonomy-path", type=Path, default=DEFAULT_TAXONOMY_PATH,
                        help="Clause taxonomy markdown file")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help="Output CSV path")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL,
                        help="LangExtract-compatible JSONL output path for visualization")
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE,
                        help="JSON cache tracking processed pages")
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR,
                        help="Directory for debug JSON dumps")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete existing outputs/debug/cache before running")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Sample N documents; partially processed docs prioritized")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Only process pages <= N")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help="LangExtract model_id (e.g., vllm:http://localhost:8000/v1)")
    parser.add_argument("--provider", type=str, default="VLLMLanguageModel",
                        help="LangExtract provider class name (optional)")
    parser.add_argument("--backend", type=str, choices=["vllm", "openai"], default="vllm",
                        help="Inference backend for clause extraction")
    parser.add_argument("--openai-model", type=str, default="gpt-5-nano",
                        help="OpenAI model used when --backend openai")
    parser.add_argument("--openai-reasoning-effort", type=str, choices=["low", "medium", "high"], default="low",
                        help="Reasoning effort for OpenAI Responses API")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max output tokens per page")
    parser.add_argument("--openai-max-retries", type=int, default=2,
                        help="Retries for empty/non-JSON OpenAI responses")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Provider timeout in seconds")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="API key for vLLM OpenAI-compatible server when required")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Sleep seconds between page requests")
    args = parser.parse_args()

    if args.clear_cache:
        if args.output_csv.exists():
            args.output_csv.unlink()
        if args.output_jsonl.exists():
            args.output_jsonl.unlink()
        if args.cache_file.exists():
            args.cache_file.unlink()
        if args.debug_dir.exists() and args.debug_dir.is_dir():
            shutil.rmtree(args.debug_dir)
        print(
            "Cleared previous outputs and cache files: "
            f"{args.output_csv}, {args.output_jsonl}, {args.cache_file}, {args.debug_dir}"
        )

    features = parse_taxonomy(args.taxonomy_path)
    feature_names = [f.name for f in features]
    canonical: dict[str, str] = {}
    for name in feature_names:
        canonical[name.lower()] = name
        canonical[re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()] = name
    name_set = set(feature_names)

    lx = None
    model = None
    prompt = ""
    examples: list[Any] = []
    openai_client = None
    if args.backend == "vllm":
        try:
            import langextract as lx  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency: langextract. Install with `uv add langextract`."
            ) from exc

        # Plugin registration for vLLM provider (safe if unavailable).
        try:
            import langextract_vllm  # noqa: F401
        except Exception:
            pass

        prompt, examples = build_prompt_and_examples(lx, features)
        provider_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "api_key": args.api_key,
        }
        model = create_model(
            lx=lx,
            model_id=args.model_id,
            provider=args.provider if args.provider else None,
            provider_kwargs=provider_kwargs,
        )
    else:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency: openai. Install with `uv add openai`."
            ) from exc
        # Required by request: read API key from os.environ["OPENAI_API_KEY"].
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = build_openai_prompt(features)

    cache = load_cache(args.cache_file)
    backfill_cache_from_csv(cache, args.output_csv)

    doc_dirs = list_doc_dirs(args.ocr_dir)
    if not doc_dirs:
        raise RuntimeError(f"No document directories found in {args.ocr_dir} (expected document_*).")

    incomplete: list[Path] = []
    for doc_dir in tqdm(doc_dirs, desc="Checking extraction cache"):
        doc_id = doc_dir.name
        pages = list_page_files(doc_dir, max_pages=args.max_pages)
        if not pages:
            continue
        doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
        done = set(doc_cache.get("processed_pages", []))
        doc_cache["total_pages"] = len(pages)
        if len(done) < len(pages):
            incomplete.append(doc_dir)

    if not incomplete:
        print("All OCR pages appear fully processed per extraction cache.")
        save_cache(args.cache_file, cache)
        return

    if args.sample_size is not None and args.sample_size < len(incomplete):
        if args.sample_size <= 0:
            raise ValueError("--sample-size must be >= 1")
        random.seed(args.seed)
        partial, fresh = [], []
        for doc_dir in incomplete:
            doc_cache = cache.get("documents", {}).get(doc_dir.name, {})
            done = set(doc_cache.get("processed_pages", []))
            total = doc_cache.get("total_pages", 0)
            if done and total and len(done) < total:
                partial.append(doc_dir)
            elif not done:
                fresh.append(doc_dir)
            else:
                partial.append(doc_dir)
        selected: list[Path] = []
        if partial:
            pick = min(len(partial), args.sample_size)
            selected.extend(random.sample(partial, pick))
        remaining = args.sample_size - len(selected)
        if remaining > 0 and fresh:
            selected.extend(random.sample(fresh, min(remaining, len(fresh))))
        incomplete = selected if selected else random.sample(incomplete, args.sample_size)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    save_cache(args.cache_file, cache)

    header = ["document_id", "document_page", "feature_name"]
    total_pages_processed = 0
    total_start = time.time()

    with (
        args.output_csv.open("a", newline="", encoding="utf-8") as out_f,
        args.output_jsonl.open("a", encoding="utf-8") as jsonl_f,
    ):
        writer = csv.DictWriter(out_f, fieldnames=header)
        if out_f.tell() == 0:
            writer.writeheader()

        for doc_dir in tqdm(incomplete, desc="Extracting clauses by document"):
            doc_id = doc_dir.name
            doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
            done = set(doc_cache.get("processed_pages", []))

            page_files = list_page_files(doc_dir, max_pages=args.max_pages)
            # Resume from most recently processed page.
            start_page = max(done) + 1 if done else 1
            page_files = [
                p for p in page_files
                if (parse_page_number(p) or 0) >= start_page and (parse_page_number(p) not in done)
            ]

            for page_file in tqdm(page_files, desc=f"{doc_id} pages", leave=False):
                page_num = parse_page_number(page_file)
                if page_num is None:
                    continue
                text = page_file.read_text(encoding="utf-8", errors="replace").strip()
                if not text:
                    done.add(page_num)
                    doc_cache["processed_pages"] = sorted(done)
                    doc_cache["last_processed_page"] = page_num
                    save_cache(args.cache_file, cache)
                    continue

                try:
                    if args.backend == "vllm":
                        if model is not None:
                            result = lx.extract(
                                model=model,
                                text_or_documents=text,
                                prompt_description=prompt,
                                examples=examples,
                            )
                        else:
                            # Fallback path if factory API is unavailable.
                            result = lx.extract(
                                text_or_documents=text,
                                prompt_description=prompt,
                                examples=examples,
                                model_id=args.model_id,
                            )
                        labels, details = parse_features_from_result(result, canonical, name_set)
                    else:
                        labels, details = extract_features_openai(
                            client=openai_client,
                            model=args.openai_model,
                            prompt=prompt,
                            text=text,
                            feature_names=feature_names,
                            canonical=canonical,
                            names=name_set,
                            max_tokens=args.max_tokens,
                            max_retries=args.openai_max_retries,
                            reasoning_effort=args.openai_reasoning_effort,
                        )
                except Exception as exc:
                    dbg_path = args.debug_dir / f"{doc_id}_page_{page_num:04d}.json"
                    dbg_path.write_text(
                        json.dumps(
                            {
                                "error": str(exc),
                                "document_id": doc_id,
                                "document_page": page_num,
                                "text_preview": text[:1000],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(f"Failed extraction for {doc_id} page {page_num}: {exc}")
                    continue

                for label in labels:
                    writer.writerow(
                        {
                            "document_id": doc_id,
                            "document_page": page_num,
                            "feature_name": label,
                        }
                    )

                annotated_doc = build_annotated_document_dict(
                    document_id=doc_id,
                    page_num=page_num,
                    text=text,
                    details=details,
                )
                jsonl_f.write(json.dumps(annotated_doc, ensure_ascii=False) + "\n")

                # Persist debug artifact for traceability.
                page_debug = args.debug_dir / f"{doc_id}_page_{page_num:04d}.json"
                page_debug.write_text(
                    json.dumps(
                        {
                            "document_id": doc_id,
                            "document_page": page_num,
                            "labels": labels,
                            "details": details,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                total_pages_processed += 1
                done.add(page_num)
                doc_cache["processed_pages"] = sorted(done)
                doc_cache["last_processed_page"] = page_num
                save_cache(args.cache_file, cache)

                if args.sleep:
                    time.sleep(args.sleep)

    elapsed = time.time() - total_start
    if total_pages_processed > 0:
        print(f"Processed pages: {total_pages_processed}")
        print(f"Average seconds per page: {elapsed / total_pages_processed:.2f}")
    else:
        print("No pages processed.")


if __name__ == "__main__":
    main()
