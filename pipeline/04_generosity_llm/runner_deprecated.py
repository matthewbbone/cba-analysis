from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

ALLOWED_FIELD_TYPES = {"number", "string", "boolean", "list[string]", "list[number]"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _resolve_io_path(path_value: Path, cache_base: Path | None) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()
    if cache_base is not None:
        return (cache_base / p).resolve()
    return (_project_root() / p).resolve()


def _default_cache_dir() -> str:
    return os.environ.get("CACHE_DIR", "").strip()


def _default_classification_dir() -> Path:
    cache_dir = _default_cache_dir()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "03_classification_output" / "dol_archive"
    return (_project_root() / "outputs" / "03_classification_output" / "dol_archive").resolve()


def _default_output_dir() -> Path:
    cache_dir = _default_cache_dir()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "04_generosity_llm"
    return (_project_root() / "outputs" / "04_generosity_llm").resolve()


def _normalize_document_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if re.fullmatch(r"document_\d+", value):
        return value
    if re.fullmatch(r"\d+", value):
        return f"document_{int(value)}"
    return value


def _parse_document_num(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"document_(\d+)", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _parse_segment_num_from_json(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"segment_(\d+)\.json", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _slugify(value: str) -> str:
    raw = str(value).strip() or "OTHER"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")
    return slug or "OTHER"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_snake_case(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isnan(out):
            return None
        return out
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def _normalize_field_type(raw_type: Any) -> str:
    text = str(raw_type or "").strip().lower()
    normalized = text.replace(" ", "")
    aliases = {
        "float": "number",
        "int": "number",
        "integer": "number",
        "numeric": "number",
        "str": "string",
        "text": "string",
        "bool": "boolean",
        "array[string]": "list[string]",
        "list[str]": "list[string]",
        "string[]": "list[string]",
        "array[number]": "list[number]",
        "list[float]": "list[number]",
        "list[int]": "list[number]",
        "number[]": "list[number]",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in ALLOWED_FIELD_TYPES:
        return normalized
    return "string"


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _coerce_field_value(value: Any, field_type: str) -> Any:
    ftype = _normalize_field_type(field_type)
    if value is None:
        return None

    if ftype == "number":
        return _safe_float_or_none(value)

    if ftype == "boolean":
        return _coerce_bool(value)

    if ftype == "string":
        text = str(value).strip()
        return text if text else None

    if ftype == "list[string]":
        if isinstance(value, list):
            out = [str(v).strip() for v in value if str(v).strip()]
            return out or None
        text = str(value).strip()
        if not text:
            return None
        return [text]

    if ftype == "list[number]":
        if isinstance(value, list):
            out: list[float] = []
            for item in value:
                numeric = _safe_float_or_none(item)
                if numeric is not None:
                    out.append(float(numeric))
            return out or None
        numeric = _safe_float_or_none(value)
        if numeric is None:
            return None
        return [float(numeric)]

    text = str(value).strip()
    return text if text else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


class GenerosityLlmRunner:
    def __init__(
        self,
        *,
        cache_dir: str | Path | None,
        classification_dir: Path,
        output_dir: Path,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_tokens: int = 2500,
        max_retries: int = 2,
        timeout: float = 120.0,
        top_clause_types: int = 10,
        schema_sample_size: int = 10,
        seed: int = 42,
    ) -> None:
        cache_base = None
        if cache_dir is not None and str(cache_dir).strip():
            cache_base = _resolve_path(str(cache_dir), _project_root())

        self.cache_dir = cache_base
        self.classification_dir = _resolve_io_path(classification_dir, cache_base)
        self.output_dir = _resolve_io_path(output_dir, cache_base)
        self.schemas_dir = self.output_dir / "schemas"
        self.extractions_dir = self.output_dir / "extractions"
        self.rubrics_dir = self.output_dir / "rubrics"
        self.evaluations_dir = self.output_dir / "evaluations"

        self.model = str(model).strip() or "gpt-5-mini"
        self.temperature = float(temperature)
        self.max_tokens = max(256, int(max_tokens))
        self.max_retries = max(1, int(max_retries))
        self.timeout = float(timeout)
        self.top_clause_types = max(1, int(top_clause_types))
        self.schema_sample_size = max(1, int(schema_sample_size))
        self.seed = int(seed)

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                last_error = exc
                continue

            content = response.choices[0].message.content or ""
            if not content.strip():
                last_error = ValueError("Model returned empty response text")
                continue
            try:
                payload = _parse_json_loose(content)
            except Exception as exc:
                last_error = exc
                continue
            if isinstance(payload, dict):
                return payload
            last_error = ValueError(f"Expected JSON object response, got: {type(payload)}")

        raise RuntimeError(f"Unable to get valid JSON from model after {self.max_retries} attempts: {last_error}")

    def _list_document_dirs(self) -> list[Path]:
        if not self.classification_dir.exists() or not self.classification_dir.is_dir():
            return []
        docs = []
        for p in self.classification_dir.iterdir():
            if not p.is_dir():
                continue
            if re.fullmatch(r"document_\d+", p.name) is None:
                continue
            if any(p.glob("segment_*.json")):
                docs.append(p)
        return sorted(docs, key=_parse_document_num)

    @staticmethod
    def _list_segment_json_paths(doc_dir: Path) -> list[Path]:
        paths = [p for p in doc_dir.glob("segment_*.json") if p.is_file()]
        return sorted([p for p in paths if _parse_segment_num_from_json(p) != 10**12], key=_parse_segment_num_from_json)

    @staticmethod
    def _extract_clause_type(payload: dict[str, Any]) -> str:
        labels = payload.get("labels", [])
        if isinstance(labels, list):
            for label in labels:
                text = str(label).strip()
                if text:
                    return text
        label = str(payload.get("label", "")).strip()
        return label if label else "OTHER"

    def _collect_segment_rows(
        self,
        *,
        document_id: str | None,
        sample_size: int | None,
        max_segments: int | None,
        seed: int,
    ) -> list[dict[str, Any]]:
        doc_dirs = self._list_document_dirs()
        if document_id:
            doc_dirs = [d for d in doc_dirs if d.name == document_id]
        if sample_size is not None and sample_size < len(doc_dirs):
            rng = random.Random(int(seed))
            doc_dirs = sorted(rng.sample(doc_dirs, int(sample_size)), key=_parse_document_num)

        rows: list[dict[str, Any]] = []
        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            seg_paths = self._list_segment_json_paths(doc_dir)
            if max_segments is not None:
                seg_paths = seg_paths[: max(0, int(max_segments))]

            for seg_path in seg_paths:
                seg_num = _parse_segment_num_from_json(seg_path)
                if seg_num == 10**12:
                    continue
                payload = _safe_read_json(seg_path)
                if not isinstance(payload, dict):
                    continue
                text = str(payload.get("segment_text", "")).strip()
                if not text:
                    continue
                clause_type = self._extract_clause_type(payload)
                rows.append(
                    {
                        "segment_id": f"{doc_id}::segment_{seg_num}",
                        "document_id": doc_id,
                        "segment_number": int(seg_num),
                        "clause_type": clause_type,
                        "segment_text": text,
                    }
                )

        rows.sort(key=lambda r: (_parse_document_num(r["document_id"]), int(r["segment_number"])))
        return rows

    @staticmethod
    def _top_clause_types(rows: list[dict[str, Any]], top_k: int) -> list[tuple[str, int]]:
        counts: Counter[str] = Counter()
        for row in rows:
            clause_type = str(row.get("clause_type", "")).strip() or "OTHER"
            if clause_type == "OTHER":
                continue
            counts[clause_type] += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
        return ranked[: max(1, int(top_k))]

    def _schema_path(self, clause_type: str) -> Path:
        return self.schemas_dir / f"{_slugify(clause_type)}.schema.json"

    def _extractions_path(self, clause_type: str) -> Path:
        return self.extractions_dir / f"{_slugify(clause_type)}.jsonl"

    def _rubric_path(self, clause_type: str) -> Path:
        return self.rubrics_dir / f"{_slugify(clause_type)}.rubric.json"

    def _distribution_path(self, clause_type: str) -> Path:
        return self.rubrics_dir / f"{_slugify(clause_type)}.distribution.json"

    def _evaluation_path(self, clause_type: str) -> Path:
        return self.evaluations_dir / f"{_slugify(clause_type)}.jsonl"

    def _normalize_schema_payload(
        self,
        *,
        clause_type: str,
        raw_schema: dict[str, Any],
        sample_segment_ids: list[str],
    ) -> dict[str, Any]:
        raw_fields = raw_schema.get("fields", []) if isinstance(raw_schema, dict) else []
        fields: list[dict[str, Any]] = []
        seen = set()

        if isinstance(raw_fields, list):
            for item in raw_fields:
                if isinstance(item, str):
                    field_name = _to_snake_case(item)
                    field_type = "string"
                    description = field_name.replace("_", " ").strip()
                elif isinstance(item, dict):
                    field_name = _to_snake_case(item.get("name", "") or item.get("field", ""))
                    field_type = _normalize_field_type(item.get("type", "string"))
                    description = str(item.get("description", "")).strip() or field_name.replace("_", " ").strip()
                else:
                    continue
                if not field_name or field_name in seen:
                    continue
                seen.add(field_name)
                fields.append(
                    {
                        "name": field_name,
                        "type": field_type,
                        "description": description,
                    }
                )

        if not fields:
            fields = [
                {
                    "name": "key_terms",
                    "type": "list[string]",
                    "description": "Main policy terms and limits explicitly stated in the clause.",
                }
            ]
            seen = {"key_terms"}

        if "notable_provisions" not in seen:
            fields.append(
                {
                    "name": "notable_provisions",
                    "type": "list[string]",
                    "description": "Notable worker-relevant provisions, exceptions, or carve-outs.",
                }
            )

        return {
            "clause_type": clause_type,
            "created_at": _iso_now(),
            "schema_method": "llm_generated_from_sample",
            "sample_segment_ids": sample_segment_ids,
            "fields": fields,
        }

    def _build_clause_schema(self, clause_type: str, clause_rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
        sample_n = min(self.schema_sample_size, len(clause_rows))
        sampled_rows = rng.sample(clause_rows, sample_n) if sample_n < len(clause_rows) else list(clause_rows)

        sample_payload = []
        for row in sampled_rows:
            sample_payload.append(
                {
                    "segment_id": row["segment_id"],
                    "document_id": row["document_id"],
                    "segment_number": row["segment_number"],
                    "segment_excerpt": str(row["segment_text"])[:1800],
                }
            )

        system_prompt = (
            "You design extraction schemas for collective bargaining agreement clauses. "
            "Return strict JSON only. Focus on fields that are commonly present across the sample. "
            "Use concise snake_case field names. Allowed field types: number, string, boolean, "
            "list[string], list[number]. Include notable_provisions as list[string]."
        )
        user_prompt = "\n".join(
            [
                f"Clause type: {clause_type}",
                "Create a schema with 4-12 clause-specific fields commonly found in these segments.",
                "Return JSON object with shape:",
                '{"fields":[{"name":"...","type":"number|string|boolean|list[string]|list[number]","description":"..."}]}',
                "",
                "Sample segments:",
                json.dumps(sample_payload, ensure_ascii=False, indent=2),
            ]
        )

        raw_schema = self._chat_json(system_prompt, user_prompt)
        return self._normalize_schema_payload(
            clause_type=clause_type,
            raw_schema=raw_schema,
            sample_segment_ids=[str(row["segment_id"]) for row in sampled_rows],
        )

    def _create_or_load_schema(
        self,
        *,
        clause_type: str,
        clause_rows: list[dict[str, Any]],
        rng: random.Random,
        force: bool,
    ) -> dict[str, Any]:
        path = self._schema_path(clause_type)
        if path.exists() and not force:
            payload = _safe_read_json(path)
            if isinstance(payload, dict):
                return payload

        schema = self._build_clause_schema(clause_type, clause_rows, rng)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        return schema

    @staticmethod
    def _normalize_extracted_fields(raw_payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        schema_fields = schema.get("fields", []) if isinstance(schema, dict) else []
        payload_fields = raw_payload.get("fields", {}) if isinstance(raw_payload, dict) else {}
        if not isinstance(payload_fields, dict) and isinstance(raw_payload, dict):
            payload_fields = raw_payload
        if not isinstance(payload_fields, dict):
            payload_fields = {}

        out: dict[str, Any] = {}
        for field in schema_fields:
            if not isinstance(field, dict):
                continue
            field_name = str(field.get("name", "")).strip()
            if not field_name:
                continue
            field_type = _normalize_field_type(field.get("type", "string"))
            raw_value = payload_fields.get(field_name)
            out[field_name] = _coerce_field_value(raw_value, field_type)
        return out

    def _extract_segment_fields(
        self,
        *,
        clause_type: str,
        schema: dict[str, Any],
        segment_text: str,
    ) -> dict[str, Any]:
        schema_fields = schema.get("fields", []) if isinstance(schema, dict) else []
        compact_schema = [
            {
                "name": str(field.get("name", "")).strip(),
                "type": _normalize_field_type(field.get("type", "string")),
                "description": str(field.get("description", "")).strip(),
            }
            for field in schema_fields
            if isinstance(field, dict) and str(field.get("name", "")).strip()
        ]

        system_prompt = (
            "You extract structured details from one CBA segment. "
            "Return strict JSON only. Use only the provided field names. "
            "If a value is missing or ambiguous, set it to null."
        )
        user_prompt = "\n".join(
            [
                f"Clause type: {clause_type}",
                "Schema fields:",
                json.dumps(compact_schema, ensure_ascii=False, indent=2),
                "",
                "Return JSON object with shape:",
                '{"fields":{"field_name": value_or_null, "...": value_or_null}}',
                "",
                "Segment text:",
                segment_text[:7000],
            ]
        )
        raw_payload = self._chat_json(system_prompt, user_prompt)
        return self._normalize_extracted_fields(raw_payload, schema)

    def _extract_or_load_clause_rows(
        self,
        *,
        clause_type: str,
        clause_rows: list[dict[str, Any]],
        schema: dict[str, Any],
        force: bool,
        max_segments_per_clause: int | None,
    ) -> list[dict[str, Any]]:
        path = self._extractions_path(clause_type)
        if path.exists() and not force:
            existing = _read_jsonl(path)
            if existing:
                return existing

        work_rows = list(clause_rows)
        if max_segments_per_clause is not None:
            work_rows = work_rows[: max(0, int(max_segments_per_clause))]

        extracted_rows: list[dict[str, Any]] = []
        for row in tqdm(work_rows, desc=f"04_generosity_llm extract {clause_type}", unit="segment"):
            try:
                extracted_fields = self._extract_segment_fields(
                    clause_type=clause_type,
                    schema=schema,
                    segment_text=str(row["segment_text"]),
                )
                status = "ok"
                error_message = ""
            except Exception as exc:
                extracted_fields = {
                    str(field.get("name", "")).strip(): None
                    for field in schema.get("fields", [])
                    if isinstance(field, dict) and str(field.get("name", "")).strip()
                }
                status = "error"
                error_message = str(exc)

            extracted_rows.append(
                {
                    "segment_id": row["segment_id"],
                    "document_id": row["document_id"],
                    "segment_number": row["segment_number"],
                    "clause_type": clause_type,
                    "fields": extracted_fields,
                    "status": status,
                    "error": error_message,
                }
            )

        _write_jsonl(path, extracted_rows)
        return extracted_rows

    @staticmethod
    def _field_distribution(field_type: str, values: list[Any]) -> dict[str, Any]:
        ftype = _normalize_field_type(field_type)
        out: dict[str, Any] = {"field_type": ftype, "non_null_count": len(values)}

        if ftype == "number":
            numeric = [float(v) for v in values if _safe_float_or_none(v) is not None]
            if not numeric:
                out["non_null_count"] = 0
                return out
            arr = np.asarray(numeric, dtype=float)
            out.update(
                {
                    "min": float(np.min(arr)),
                    "p10": float(np.percentile(arr, 10)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p90": float(np.percentile(arr, 90)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                }
            )
            return out

        if ftype == "boolean":
            bool_values = [_coerce_bool(v) for v in values]
            bool_values = [v for v in bool_values if v is not None]
            true_count = int(sum(1 for v in bool_values if v))
            false_count = int(sum(1 for v in bool_values if not v))
            total = true_count + false_count
            out.update(
                {
                    "true_count": true_count,
                    "false_count": false_count,
                    "true_rate": (float(true_count) / float(total)) if total > 0 else None,
                }
            )
            return out

        if ftype == "string":
            text_values = [str(v).strip() for v in values if str(v).strip()]
            counter = Counter(text_values)
            out["top_values"] = [{"value": k, "count": int(v)} for k, v in counter.most_common(15)]
            return out

        if ftype == "list[string]":
            flattened = []
            for item in values:
                if isinstance(item, list):
                    flattened.extend(str(v).strip() for v in item if str(v).strip())
                else:
                    text = str(item).strip()
                    if text:
                        flattened.append(text)
            counter = Counter(flattened)
            out["unique_item_count"] = len(counter)
            out["top_items"] = [{"value": k, "count": int(v)} for k, v in counter.most_common(20)]
            return out

        if ftype == "list[number]":
            flattened: list[float] = []
            for item in values:
                if isinstance(item, list):
                    for entry in item:
                        numeric = _safe_float_or_none(entry)
                        if numeric is not None:
                            flattened.append(float(numeric))
                else:
                    numeric = _safe_float_or_none(item)
                    if numeric is not None:
                        flattened.append(float(numeric))
            if not flattened:
                out["non_null_count"] = 0
                return out
            arr = np.asarray(flattened, dtype=float)
            out.update(
                {
                    "min": float(np.min(arr)),
                    "p10": float(np.percentile(arr, 10)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p90": float(np.percentile(arr, 90)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                }
            )
            return out

        return out

    def _distribution_for_clause(self, schema: dict[str, Any], extracted_rows: list[dict[str, Any]]) -> dict[str, Any]:
        distributions: dict[str, Any] = {}
        fields = schema.get("fields", []) if isinstance(schema, dict) else []

        for field in fields:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name", "")).strip()
            if not name:
                continue
            field_type = _normalize_field_type(field.get("type", "string"))
            values = []
            for row in extracted_rows:
                payload = row.get("fields", {})
                if not isinstance(payload, dict):
                    continue
                value = payload.get(name)
                if value is None:
                    continue
                values.append(value)
            distributions[name] = self._field_distribution(field_type, values)
        return distributions

    @staticmethod
    def _default_rubric_detail(field: dict[str, Any], distribution: dict[str, Any]) -> dict[str, Any]:
        name = str(field.get("name", "")).strip()
        description = str(field.get("description", "")).strip() or name.replace("_", " ")
        field_type = _normalize_field_type(field.get("type", "string"))

        if field_type in {"number", "list[number]"} and distribution.get("non_null_count", 0) > 0:
            anchors = {
                "1": str(round(float(distribution.get("p90", distribution.get("max", 0.0))), 3)),
                "2": str(round(float(distribution.get("p75", distribution.get("p90", 0.0))), 3)),
                "3": str(round(float(distribution.get("p50", distribution.get("p75", 0.0))), 3)),
                "4": str(round(float(distribution.get("p25", distribution.get("p50", 0.0))), 3)),
                "5": str(round(float(distribution.get("p10", distribution.get("p25", 0.0))), 3)),
            }
            guidance = (
                "Use observed numeric values. If a value is missing, assign 3 unless adjacent evidence implies higher/lower generosity."
            )
        elif field_type == "boolean":
            anchors = {
                "1": "Clearly unfavorable/no worker benefit",
                "2": "Mostly absent",
                "3": "Ambiguous or mixed",
                "4": "Mostly present and favorable",
                "5": "Clearly present and strongly worker-favorable",
            }
            guidance = "Map explicit favorable protections toward higher scores. If unclear, assign 3."
        else:
            anchors = {
                "1": "Strongly employer-favorable wording",
                "2": "Some employer-favorable limitations",
                "3": "Typical/neutral wording",
                "4": "Worker-favorable wording",
                "5": "Strongly worker-favorable wording",
            }
            guidance = (
                "Score based on how favorable the observed values are for workers relative to typical values in this clause type."
            )

        return {
            "name": name,
            "description": description,
            "scoring_anchors": anchors,
            "scoring_guidance": guidance,
        }

    def _normalize_rubric_payload(
        self,
        *,
        clause_type: str,
        schema: dict[str, Any],
        distributions: dict[str, Any],
        raw_rubric: dict[str, Any],
    ) -> dict[str, Any]:
        default_by_field: dict[str, dict[str, Any]] = {}
        fields = schema.get("fields", []) if isinstance(schema, dict) else []
        for field in fields:
            if not isinstance(field, dict):
                continue
            fname = str(field.get("name", "")).strip()
            if not fname:
                continue
            default_by_field[fname] = self._default_rubric_detail(field, distributions.get(fname, {}))

        raw_details = raw_rubric.get("details", []) if isinstance(raw_rubric, dict) else []
        provided: dict[str, dict[str, Any]] = {}
        if isinstance(raw_details, list):
            for detail in raw_details:
                if not isinstance(detail, dict):
                    continue
                name = _to_snake_case(detail.get("name", ""))
                if not name:
                    continue
                provided[name] = detail

        details: list[dict[str, Any]] = []
        for field_name, default_detail in default_by_field.items():
            raw_detail = provided.get(field_name, {})
            anchors_raw = raw_detail.get("scoring_anchors", {}) if isinstance(raw_detail, dict) else {}
            if not isinstance(anchors_raw, dict):
                anchors_raw = {}

            anchors = {}
            for score in ["1", "2", "3", "4", "5"]:
                text = str(anchors_raw.get(score, "")).strip()
                if not text:
                    text = str(default_detail["scoring_anchors"].get(score, "")).strip()
                anchors[score] = text

            detail = {
                "name": field_name,
                "description": (
                    str(raw_detail.get("description", "")).strip()
                    if isinstance(raw_detail, dict)
                    else ""
                )
                or str(default_detail["description"]),
                "scoring_anchors": anchors,
                "scoring_guidance": (
                    str(raw_detail.get("scoring_guidance", "")).strip()
                    if isinstance(raw_detail, dict)
                    else ""
                )
                or str(default_detail["scoring_guidance"]),
            }
            details.append(detail)

        return {
            "clause_type": clause_type,
            "created_at": _iso_now(),
            "rubric_method": "llm_calibrated_with_distribution_fallback",
            "details": details,
        }

    def _build_rubric_with_llm(
        self,
        *,
        clause_type: str,
        schema: dict[str, Any],
        distributions: dict[str, Any],
    ) -> dict[str, Any]:
        system_prompt = (
            "You are creating a calibrated 1-5 generosity rubric for one clause type. "
            "Return strict JSON only. Use observed distribution values. "
            "Set score 3 near the median/typical value and 1/5 near observed extremes. "
            "Include guidance for ambiguous or missing values."
        )
        user_prompt = "\n".join(
            [
                f"Clause type: {clause_type}",
                "Schema:",
                json.dumps(schema, ensure_ascii=False, indent=2),
                "",
                "Observed distributions:",
                json.dumps(distributions, ensure_ascii=False, indent=2),
                "",
                "Return JSON object with shape:",
                '{'
                '"details":['
                '{"name":"field_name","description":"...","scoring_anchors":{"1":"...","2":"...","3":"...","4":"...","5":"..."},"scoring_guidance":"..."}'
                "]"
                "}",
            ]
        )
        raw = self._chat_json(system_prompt, user_prompt)
        return self._normalize_rubric_payload(
            clause_type=clause_type,
            schema=schema,
            distributions=distributions,
            raw_rubric=raw,
        )

    def _create_or_load_rubric(
        self,
        *,
        clause_type: str,
        schema: dict[str, Any],
        distributions: dict[str, Any],
        force: bool,
    ) -> dict[str, Any]:
        rubric_path = self._rubric_path(clause_type)
        if rubric_path.exists() and not force:
            payload = _safe_read_json(rubric_path)
            if isinstance(payload, dict):
                return payload

        try:
            rubric = self._build_rubric_with_llm(
                clause_type=clause_type,
                schema=schema,
                distributions=distributions,
            )
        except Exception:
            rubric = self._normalize_rubric_payload(
                clause_type=clause_type,
                schema=schema,
                distributions=distributions,
                raw_rubric={},
            )

        rubric_path.parent.mkdir(parents=True, exist_ok=True)
        rubric_path.write_text(json.dumps(rubric, ensure_ascii=False, indent=2), encoding="utf-8")

        dist_path = self._distribution_path(clause_type)
        dist_path.parent.mkdir(parents=True, exist_ok=True)
        dist_path.write_text(json.dumps(distributions, ensure_ascii=False, indent=2), encoding="utf-8")
        return rubric

    @staticmethod
    def _aggregate_document_clause_payloads(
        *,
        schema: dict[str, Any],
        extracted_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in extracted_rows:
            doc_id = str(row.get("document_id", "")).strip()
            if not doc_id:
                continue
            grouped[doc_id].append(row)

        out: dict[str, dict[str, Any]] = {}
        schema_fields = schema.get("fields", []) if isinstance(schema, dict) else []
        for doc_id, rows in grouped.items():
            field_summary: dict[str, Any] = {}
            for field in schema_fields:
                if not isinstance(field, dict):
                    continue
                field_name = str(field.get("name", "")).strip()
                field_type = _normalize_field_type(field.get("type", "string"))
                if not field_name:
                    continue
                values = []
                for row in rows:
                    payload = row.get("fields", {})
                    if not isinstance(payload, dict):
                        continue
                    value = payload.get(field_name)
                    if value is None:
                        continue
                    values.append(value)
                field_summary[field_name] = GenerosityLlmRunner._field_distribution(field_type, values)
            out[doc_id] = {
                "document_id": doc_id,
                "segment_count": len(rows),
                "field_summaries": field_summary,
            }
        return out

    def _normalize_detail_scores(
        self,
        *,
        rubric: dict[str, Any],
        raw_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rubric_details = rubric.get("details", []) if isinstance(rubric, dict) else []
        by_name: dict[str, dict[str, Any]] = {}
        raw_details = raw_payload.get("detail_scores", []) if isinstance(raw_payload, dict) else []
        if isinstance(raw_details, list):
            for item in raw_details:
                if not isinstance(item, dict):
                    continue
                name = _to_snake_case(item.get("name", ""))
                if not name:
                    continue
                by_name[name] = item

        out: list[dict[str, Any]] = []
        for detail in rubric_details:
            if not isinstance(detail, dict):
                continue
            name = str(detail.get("name", "")).strip()
            if not name:
                continue
            payload = by_name.get(name, {})
            score = _safe_int(payload.get("score", 3), 3)
            score = min(5, max(1, score))
            out.append(
                {
                    "name": name,
                    "score": score,
                    "reason": str(payload.get("reason", "")).strip()
                    if isinstance(payload, dict)
                    else "",
                }
            )
        return out

    def _evaluate_document_clause(
        self,
        *,
        clause_type: str,
        document_id: str,
        rubric: dict[str, Any],
        aggregated_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        system_prompt = (
            "You score clause-level generosity for one document using a provided rubric. "
            "Return strict JSON only. For each rubric detail, assign integer score 1-5 and brief reason."
        )
        user_prompt = "\n".join(
            [
                f"Clause type: {clause_type}",
                f"Document: {document_id}",
                "Rubric:",
                json.dumps(rubric, ensure_ascii=False, indent=2),
                "",
                "Aggregated extracted detail summaries for this document/clauses:",
                json.dumps(aggregated_payload, ensure_ascii=False, indent=2),
                "",
                "Return JSON object with shape:",
                '{"detail_scores":[{"name":"field_name","score":1,"reason":"..."}]}',
            ]
        )
        raw = self._chat_json(system_prompt, user_prompt)
        return self._normalize_detail_scores(rubric=rubric, raw_payload=raw)

    def _evaluate_or_load_clause(
        self,
        *,
        clause_type: str,
        rubric: dict[str, Any],
        schema: dict[str, Any],
        extracted_rows: list[dict[str, Any]],
        force: bool,
    ) -> list[dict[str, Any]]:
        path = self._evaluation_path(clause_type)
        if path.exists() and not force:
            existing = _read_jsonl(path)
            if existing:
                return existing

        doc_payloads = self._aggregate_document_clause_payloads(schema=schema, extracted_rows=extracted_rows)
        eval_rows: list[dict[str, Any]] = []
        for doc_id in tqdm(sorted(doc_payloads.keys(), key=_parse_document_num), desc=f"04_generosity_llm eval {clause_type}", unit="doc"):
            payload = doc_payloads[doc_id]
            try:
                detail_scores = self._evaluate_document_clause(
                    clause_type=clause_type,
                    document_id=doc_id,
                    rubric=rubric,
                    aggregated_payload=payload,
                )
                status = "ok"
                error_message = ""
            except Exception as exc:
                detail_scores = [
                    {
                        "name": str(detail.get("name", "")).strip(),
                        "score": 3,
                        "reason": "Defaulted due to evaluation error.",
                    }
                    for detail in rubric.get("details", [])
                    if isinstance(detail, dict) and str(detail.get("name", "")).strip()
                ]
                status = "error"
                error_message = str(exc)

            scores = [int(row["score"]) for row in detail_scores if isinstance(row.get("score"), int)]
            clause_composite_score = float(np.mean(scores)) if scores else None
            eval_rows.append(
                {
                    "document_id": doc_id,
                    "clause_type": clause_type,
                    "segment_count": _safe_int(payload.get("segment_count", 0)),
                    "detail_scores": detail_scores,
                    "clause_composite_score": clause_composite_score,
                    "status": status,
                    "error": error_message,
                }
            )

        _write_jsonl(path, eval_rows)
        return eval_rows

    def _write_scoring_outputs(self, clause_eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        clause_csv = self.output_dir / "document_clause_composite_scores.csv"
        doc_csv = self.output_dir / "document_composite_scores.csv"

        with clause_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "document_id",
                    "clause_type",
                    "segment_count",
                    "clause_composite_score",
                    "detail_scores_json",
                    "status",
                    "error",
                ],
            )
            writer.writeheader()
            for row in sorted(
                clause_eval_rows,
                key=lambda r: (_parse_document_num(str(r.get("document_id", ""))), str(r.get("clause_type", ""))),
            ):
                writer.writerow(
                    {
                        "document_id": row.get("document_id", ""),
                        "clause_type": row.get("clause_type", ""),
                        "segment_count": _safe_int(row.get("segment_count", 0)),
                        "clause_composite_score": row.get("clause_composite_score"),
                        "detail_scores_json": json.dumps(row.get("detail_scores", []), ensure_ascii=False),
                        "status": row.get("status", ""),
                        "error": row.get("error", ""),
                    }
                )

        by_doc: dict[str, list[float]] = defaultdict(list)
        for row in clause_eval_rows:
            doc_id = str(row.get("document_id", "")).strip()
            score = _safe_float_or_none(row.get("clause_composite_score"))
            if not doc_id or score is None:
                continue
            by_doc[doc_id].append(float(score))

        doc_rows: list[dict[str, Any]] = []
        for doc_id in sorted(by_doc.keys(), key=_parse_document_num):
            clause_scores = by_doc[doc_id]
            doc_rows.append(
                {
                    "document_id": doc_id,
                    "document_composite_score": float(np.mean(clause_scores)) if clause_scores else None,
                    "clause_count_scored": len(clause_scores),
                }
            )

        with doc_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["document_id", "document_composite_score", "clause_count_scored"],
            )
            writer.writeheader()
            writer.writerows(doc_rows)

        return {
            "clause_scores_csv": str(clause_csv),
            "document_scores_csv": str(doc_csv),
            "documents_scored": len(doc_rows),
        }

    def run(
        self,
        *,
        document_id: str | None,
        sample_size: int | None,
        seed: int,
        max_segments: int | None,
        max_segments_per_clause: int | None,
        force: bool,
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.extractions_dir.mkdir(parents=True, exist_ok=True)
        self.rubrics_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

        segment_rows = self._collect_segment_rows(
            document_id=document_id,
            sample_size=sample_size,
            max_segments=max_segments,
            seed=seed,
        )
        top_clause_pairs = self._top_clause_types(segment_rows, self.top_clause_types)
        top_clause_types = [name for name, _ in top_clause_pairs]
        top_clause_counts = {name: int(count) for name, count in top_clause_pairs}

        if not segment_rows or not top_clause_types:
            summary = {
                "model": self.model,
                "classification_dir": str(self.classification_dir),
                "output_dir": str(self.output_dir),
                "segments_available": len(segment_rows),
                "top_clause_types_selected": [],
                "top_clause_type_counts": {},
                "schemas_written": 0,
                "segment_extractions_written": 0,
                "clause_evaluations_written": 0,
                "documents_scored": 0,
                "outputs": {},
                "created_at": _iso_now(),
            }
            (self.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        rng = random.Random(int(seed))
        all_clause_eval_rows: list[dict[str, Any]] = []
        schema_index: dict[str, Any] = {}
        clause_extraction_counts: dict[str, int] = {}
        clause_eval_counts: dict[str, int] = {}

        for clause_type in top_clause_types:
            clause_rows = [row for row in segment_rows if str(row.get("clause_type", "")).strip() == clause_type]
            if not clause_rows:
                continue

            schema = self._create_or_load_schema(
                clause_type=clause_type,
                clause_rows=clause_rows,
                rng=rng,
                force=force,
            )
            schema_index[clause_type] = {
                "schema_path": str(self._schema_path(clause_type)),
                "field_count": len(schema.get("fields", [])) if isinstance(schema.get("fields", []), list) else 0,
            }

            extracted_rows = self._extract_or_load_clause_rows(
                clause_type=clause_type,
                clause_rows=clause_rows,
                schema=schema,
                force=force,
                max_segments_per_clause=max_segments_per_clause,
            )
            clause_extraction_counts[clause_type] = len(extracted_rows)

            distributions = self._distribution_for_clause(schema, extracted_rows)
            rubric = self._create_or_load_rubric(
                clause_type=clause_type,
                schema=schema,
                distributions=distributions,
                force=force,
            )

            eval_rows = self._evaluate_or_load_clause(
                clause_type=clause_type,
                rubric=rubric,
                schema=schema,
                extracted_rows=extracted_rows,
                force=force,
            )
            clause_eval_counts[clause_type] = len(eval_rows)
            all_clause_eval_rows.extend(eval_rows)

        schema_index_path = self.schemas_dir / "schema_index.json"
        schema_index_path.write_text(json.dumps(schema_index, ensure_ascii=False, indent=2), encoding="utf-8")

        scoring_output_summary = self._write_scoring_outputs(all_clause_eval_rows)

        summary = {
            "model": self.model,
            "classification_dir": str(self.classification_dir),
            "output_dir": str(self.output_dir),
            "segments_available": len(segment_rows),
            "top_clause_types_selected": top_clause_types,
            "top_clause_type_counts": top_clause_counts,
            "top_clause_type_count_target": self.top_clause_types,
            "schema_sample_size": self.schema_sample_size,
            "schemas_written": len(schema_index),
            "segment_extractions_written": int(sum(clause_extraction_counts.values())),
            "clause_evaluations_written": int(sum(clause_eval_counts.values())),
            "documents_scored": int(scoring_output_summary.get("documents_scored", 0)),
            "clause_extraction_counts": clause_extraction_counts,
            "clause_evaluation_counts": clause_eval_counts,
            "outputs": {
                "schemas_dir": str(self.schemas_dir),
                "schema_index_json": str(schema_index_path),
                "extractions_dir": str(self.extractions_dir),
                "rubrics_dir": str(self.rubrics_dir),
                "evaluations_dir": str(self.evaluations_dir),
                **scoring_output_summary,
            },
            "created_at": _iso_now(),
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rubric-based generosity scoring pipeline: "
            "schema generation -> extraction -> rubric calibration -> document/clause scoring."
        )
    )
    parser.add_argument("--cache-dir", type=str, default=_default_cache_dir())
    parser.add_argument("--classification-dir", type=Path, default=_default_classification_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--top-clause-types", type=int, default=10)
    parser.add_argument("--schema-sample-size", type=int, default=10)
    parser.add_argument("--document-id", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--max-segments-per-clause", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runner = GenerosityLlmRunner(
        cache_dir=args.cache_dir,
        classification_dir=args.classification_dir,
        output_dir=args.output_dir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        timeout=args.timeout,
        top_clause_types=args.top_clause_types,
        schema_sample_size=args.schema_sample_size,
        seed=args.seed,
    )
    summary = runner.run(
        document_id=_normalize_document_id(args.document_id),
        sample_size=args.sample_size,
        seed=args.seed,
        max_segments=args.max_segments,
        max_segments_per_clause=args.max_segments_per_clause,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
