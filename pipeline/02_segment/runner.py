"""Segment OCR full text into header-aware CBA sections.

This stage reads OCR document folders, asks a managed local vLLM model to infer
the document hierarchy, re-annotates section header lines with markdown heading
levels, and writes both a structured `document_meta.json` and legacy
`segments/segment_*.txt` files for downstream stages.
"""

from __future__ import annotations

import bisect
import concurrent.futures
import ast
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

try:
    from pipeline.utils.vllm_server import VLLMServer
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from pipeline.utils.vllm_server import VLLMServer

load_dotenv()


SCHEMA_VERSION = "section_segmentation_v1"
PLANNING_FIRST_FRACTION = 0.1


@dataclass(frozen=True)
class HeaderEdit:
    line_number: int
    level: int
    header: str


class SegmentationRunner:
    """Infer CBA hierarchy, annotate markdown headers, and materialize sections."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        input_dir: Path,
        output_dir: Path,
        seed: int = 42,
        planning_random_chunks: int = 4,
        planning_random_chunk_chars: int = 4000,
        annotate_chunk_chars: int = 12000,
        annotate_overlap_chars: int = 1500,
        annotate_workers: int = 4,
        max_retries: int = 3,
    ) -> None:
        self.client = client
        self.model = model
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.seed = seed
        self.planning_random_chunks = planning_random_chunks
        self.planning_random_chunk_chars = planning_random_chunk_chars
        self.annotate_chunk_chars = annotate_chunk_chars
        self.annotate_overlap_chars = annotate_overlap_chars
        self.annotate_workers = max(1, annotate_workers)
        self.max_retries = max(1, max_retries)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def _log(cls, message: str) -> None:
        tqdm.write(f"[{cls._timestamp()}] {message}")

    @staticmethod
    def _document_sort_key(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"document_(\d+)", path.name)
        if not match:
            return (10**12, path.name)
        return (int(match.group(1)), path.name)

    @staticmethod
    def _line_starts(lines: list[str]) -> list[int]:
        starts: list[int] = []
        offset = 0
        for line in lines:
            starts.append(offset)
            offset += len(line)
        return starts

    @staticmethod
    def _split_line_ending(line: str) -> tuple[str, str]:
        if line.endswith("\r\n"):
            return line[:-2], "\r\n"
        if line.endswith("\n") or line.endswith("\r"):
            return line[:-1], line[-1]
        return line, ""

    @staticmethod
    def _clean_header(value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"^#{1,6}\s*", "", text).strip()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _strip_code_fence(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @classmethod
    def _parse_json_object(cls, raw: str) -> dict[str, Any]:
        text = cls._strip_code_fence(raw)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                snippet = text
            else:
                snippet = text[start : end + 1]

            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                payload = ast.literal_eval(snippet)

        if not isinstance(payload, dict):
            raise ValueError("Model response was valid JSON but not an object")
        return payload

    @staticmethod
    def _hierarchy_response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "hierarchy_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "organization_description": {"type": "string"},
                        "hierarchy": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "level": {"type": "integer"},
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                    "examples": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "level",
                                    "label",
                                    "description",
                                    "examples",
                                ],
                            },
                        },
                    },
                    "required": [
                        "organization_description",
                        "hierarchy",
                    ],
                },
            },
        }

    @staticmethod
    def _header_annotation_response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "header_annotation_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "headers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "line_number": {"type": "integer"},
                                    "level": {"type": "integer"},
                                    "header": {"type": "string"},
                                },
                                "required": [
                                    "line_number",
                                    "level",
                                    "header",
                                ],
                            },
                        },
                    },
                    "required": ["headers"],
                },
            },
        }

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        context: str,
        allow_failure: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        last_error: Exception | None = None
        schema_disabled = False
        for attempt in range(self.max_retries):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if attempt > 0:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed as a JSON "
                            f"object because: {last_error}. Return valid JSON only."
                        ),
                    }
                )

            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                }
                if response_format is not None and not schema_disabled:
                    request["response_format"] = response_format

                response = self.client.chat.completions.create(
                    **request
                )
                raw = response.choices[0].message.content or ""
                return self._parse_json_object(raw)
            except Exception as exc:
                last_error = exc
                if response_format is not None and not schema_disabled:
                    schema_disabled = True
                    self._log(
                        f"{context}: structured output rejected or failed; "
                        "retrying without json schema"
                    )
                self._log(
                    f"{context}: attempt {attempt + 1}/{self.max_retries} failed: {exc}"
                )

        if allow_failure:
            self._log(
                f"{context}: giving up after {self.max_retries} failed attempts; "
                "using fallback behavior"
            )
            return None
        raise RuntimeError(f"Failed to get parseable JSON from model: {last_error}")

    def _resolve_document_dirs(self) -> list[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        direct_docs = [
            p
            for p in self.input_dir.iterdir()
            if p.is_dir() and re.fullmatch(r"document_\d+", p.name)
        ]
        if direct_docs:
            return sorted(direct_docs, key=self._document_sort_key)

        model_dirs: list[Path] = []
        for child in self.input_dir.iterdir():
            if not child.is_dir():
                continue
            has_docs = any(
                p.is_dir() and re.fullmatch(r"document_\d+", p.name)
                for p in child.iterdir()
            )
            if has_docs:
                model_dirs.append(child)

        if len(model_dirs) == 1:
            return sorted(
                [
                    p
                    for p in model_dirs[0].iterdir()
                    if p.is_dir() and re.fullmatch(r"document_\d+", p.name)
                ],
                key=self._document_sort_key,
            )

        if len(model_dirs) > 1:
            choices = ", ".join(str(p) for p in sorted(model_dirs))
            raise RuntimeError(
                "Multiple OCR model output directories contain document folders. "
                f"Set INPUT_DIRECTORY to a more specific path in main(). Candidates: {choices}"
            )

        return []

    def _planning_sample(self, document_id: str, full_text: str) -> str:
        first_len = int(len(full_text) * PLANNING_FIRST_FRACTION)
        first_len = max(1, first_len) if full_text else 0
        chunks = [
            (
                "FIRST_10_PERCENT",
                0,
                first_len,
                full_text[:first_len],
            )
        ]

        remaining_start = first_len
        remaining_len = max(0, len(full_text) - remaining_start)
        if remaining_len > 0 and self.planning_random_chunks > 0:
            rng = random.Random(f"{self.seed}:{document_id}")
            chunk_len = min(self.planning_random_chunk_chars, remaining_len)
            max_start = max(remaining_start, len(full_text) - chunk_len)
            starts = [
                rng.randint(remaining_start, max_start)
                for _ in range(self.planning_random_chunks)
            ]
            for idx, start in enumerate(sorted(starts), start=1):
                end = min(len(full_text), start + chunk_len)
                chunks.append(
                    (
                        f"RANDOM_CHUNK_{idx}",
                        start,
                        end,
                        full_text[start:end],
                    )
                )

        rendered = []
        for label, start, end, text in chunks:
            rendered.append(f"\n[{label} span={start}:{end}]\n{text}")
        return "\n".join(rendered)

    def _infer_hierarchy(self, document_id: str, full_text: str) -> dict[str, Any]:
        system_prompt = "\n".join(
            [
                "You are an expert in collective bargaining agreement structure.",
                "Infer the document's organizational hierarchy from excerpts.",
                "Describe the hierarchy in natural language and identify the header",
                "patterns that delimit parts, such as Article 1, Section 1.2, or 1.2.4.",
                "Return one JSON object only. Do not include markdown fences.",
                "Use this shape:",
                '{"organization_description": "...", "hierarchy": [',
                '{"level": 1, "markdown": "#", "label": "...", "description": "...", "examples": ["..."]}',
                "]}",
                "Use levels 1, 2, and 3 only; map deeper structures to level 3.",
            ]
        )
        user_prompt = "\n".join(
            [
                f"Document id: {document_id}",
                "Text excerpts:",
                self._planning_sample(document_id, full_text),
            ]
        )
        payload = self._chat_json(
            system_prompt,
            user_prompt,
            max_tokens=8192,
            context=f"{document_id}: hierarchy planning",
            allow_failure=True,
            response_format=self._hierarchy_response_format(),
        )
        if payload is None:
            self._log(
                f"{document_id}: hierarchy planning fallback to default hierarchy"
            )
            return self._normalize_hierarchy({})
        return self._normalize_hierarchy(payload)

    def _normalize_hierarchy(self, payload: dict[str, Any]) -> dict[str, Any]:
        description = str(payload.get("organization_description") or "").strip()
        raw_hierarchy = payload.get("hierarchy")
        hierarchy: list[dict[str, Any]] = []

        if isinstance(raw_hierarchy, list):
            for item in raw_hierarchy:
                if not isinstance(item, dict):
                    continue
                try:
                    level = max(1, min(3, int(item.get("level", 1))))
                except Exception:
                    level = 1
                examples = item.get("examples")
                if not isinstance(examples, list):
                    examples = []
                hierarchy.append(
                    {
                        "level": level,
                        "markdown": "#" * level,
                        "label": str(item.get("label") or f"Level {level}").strip(),
                        "description": str(item.get("description") or "").strip(),
                        "examples": [str(e).strip() for e in examples if str(e).strip()],
                    }
                )

        if not hierarchy:
            hierarchy = [
                {
                    "level": 1,
                    "markdown": "#",
                    "label": "Article",
                    "description": "Top-level agreement sections, often labeled as articles.",
                    "examples": ["Article 1"],
                },
                {
                    "level": 2,
                    "markdown": "##",
                    "label": "Section",
                    "description": "Subsections within a top-level section.",
                    "examples": ["Section 1.1"],
                },
                {
                    "level": 3,
                    "markdown": "###",
                    "label": "Subsection",
                    "description": "Lower-level numbered or lettered parts.",
                    "examples": ["1.1(a)"],
                },
            ]
        if not description:
            labels = ", ".join(h["label"] for h in hierarchy)
            description = f"The document appears to be organized into: {labels}."

        return {
            "organization_description": description,
            "hierarchy": hierarchy,
        }

    def _chunk_line_ranges(self, lines: list[str]) -> list[tuple[int, int]]:
        if not lines:
            return []
        line_starts = self._line_starts(lines)
        ranges: list[tuple[int, int]] = []
        start_idx = 0

        while start_idx < len(lines):
            start_char = line_starts[start_idx]
            end_idx = start_idx + 1
            while (
                end_idx < len(lines)
                and line_starts[end_idx] - start_char < self.annotate_chunk_chars
            ):
                end_idx += 1
            ranges.append((start_idx, end_idx))

            if end_idx >= len(lines):
                break

            end_char = line_starts[end_idx]
            next_char = max(start_char + 1, end_char - self.annotate_overlap_chars)
            next_idx = bisect.bisect_left(line_starts, next_char)
            if next_idx <= start_idx:
                next_idx = start_idx + 1
            start_idx = next_idx

        return ranges

    def _annotate_chunk(
        self,
        document_id: str,
        chunk_idx: int,
        chunk_count: int,
        hierarchy: dict[str, Any],
        lines: list[str],
        start_idx: int,
        end_idx: int,
    ) -> list[HeaderEdit]:
        rendered_lines = []
        for idx in range(start_idx, end_idx):
            body, _ending = self._split_line_ending(lines[idx])
            rendered_lines.append(f"{idx + 1}: {body}")

        system_prompt = "\n".join(
            [
                "You annotate existing OCR text lines with markdown header levels.",
                "Use the supplied hierarchy to identify only true section headers.",
                "Do not rewrite body text. Do not invent missing headers.",
                "Do not mark table-of-contents entries as real headers.",
                "Return one JSON object only, with this shape:",
                '{"headers": [{"line_number": 12, "level": 1, "header": "Article 1 Recognition"}]}',
                "Use levels 1, 2, and 3 only; map deeper levels to 3.",
                'If no lines are true headers, return {"headers": []}.',
            ]
        )
        user_prompt = "\n".join(
            [
                "Hierarchy:",
                json.dumps(hierarchy, ensure_ascii=False),
                "",
                "Numbered text lines:",
                "\n".join(rendered_lines),
            ]
        )
        payload = self._chat_json(
            system_prompt,
            user_prompt,
            max_tokens=4096,
            context=f"{document_id}: chunk {chunk_idx}/{chunk_count}",
            allow_failure=True,
            response_format=self._header_annotation_response_format(),
        )
        if payload is None:
            return []
        raw_headers = payload.get("headers", [])
        if not isinstance(raw_headers, list):
            return []

        edits: list[HeaderEdit] = []
        for item in raw_headers:
            if not isinstance(item, dict):
                continue
            try:
                line_number = int(item.get("line_number"))
                level = max(1, min(3, int(item.get("level", 1))))
            except Exception:
                continue
            if not (start_idx + 1 <= line_number <= end_idx):
                continue

            line_idx = line_number - 1
            current_body, _ending = self._split_line_ending(lines[line_idx])
            if not current_body.strip():
                continue

            header = self._clean_header(str(item.get("header") or current_body))
            if not header:
                continue
            edits.append(HeaderEdit(line_number=line_number, level=level, header=header))

        return edits

    def _annotate_headers(
        self,
        document_id: str,
        hierarchy: dict[str, Any],
        full_text: str,
    ) -> str:
        lines = full_text.splitlines(keepends=True)
        if not lines and full_text:
            lines = [full_text]

        line_starts = self._line_starts(lines)
        chunks = self._chunk_line_ranges(lines)
        chunk_payloads: list[tuple[int, int, int, int]] = []
        for start_idx, end_idx in chunks:
            start_char = line_starts[start_idx]
            end_char = (
                line_starts[end_idx]
                if end_idx < len(line_starts)
                else len(full_text)
            )
            chunk_payloads.append((start_idx, end_idx, start_char, end_char))

        if not chunk_payloads:
            return full_text

        self._log(
            f"{document_id}: annotating {len(chunk_payloads)} chunks with "
            f"{self.annotate_workers} worker(s)"
        )

        chunk_results: list[list[HeaderEdit] | None] = [None] * len(chunk_payloads)
        chunk_started_at: list[float | None] = [None] * len(chunk_payloads)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.annotate_workers
        ) as executor:
            future_to_chunk: dict[concurrent.futures.Future[list[HeaderEdit]], int] = {}
            for chunk_idx, (start_idx, end_idx, start_char, end_char) in enumerate(
                chunk_payloads,
                start=1,
            ):
                self._log(
                    f"{document_id}: chunk {chunk_idx}/{len(chunk_payloads)} started "
                    f"(lines {start_idx + 1}-{end_idx}, chars {start_char}:{end_char})"
                )
                future = executor.submit(
                    self._annotate_chunk,
                    document_id,
                    chunk_idx,
                    len(chunk_payloads),
                    hierarchy,
                    lines,
                    start_idx,
                    end_idx,
                )
                future_to_chunk[future] = chunk_idx - 1
                chunk_started_at[chunk_idx - 1] = time.perf_counter()

            with tqdm(total=len(chunk_payloads), desc="Annotating chunks", unit="chunk") as progress:
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    start_idx, end_idx, start_char, end_char = chunk_payloads[chunk_idx]
                    try:
                        edits = future.result()
                    except Exception:
                        self._log(
                            f"{document_id}: chunk {chunk_idx + 1}/{len(chunk_payloads)} failed "
                            f"(lines {start_idx + 1}-{end_idx}, chars {start_char}:{end_char})"
                        )
                        raise
                    started_at = chunk_started_at[chunk_idx] or time.perf_counter()
                    duration = time.perf_counter() - started_at
                    chunk_results[chunk_idx] = edits
                    self._log(
                        f"{document_id}: chunk {chunk_idx + 1}/{len(chunk_payloads)} finished "
                        f"in {duration:.2f}s with {len(edits)} header edit(s)"
                    )
                    progress.update(1)

        header_edits: dict[int, HeaderEdit] = {}
        for edits in chunk_results:
            if not edits:
                continue
            for edit in edits:
                # Overlap can produce duplicates; keep the first deterministic edit.
                header_edits.setdefault(edit.line_number, edit)

        annotated_lines = list(lines)
        for line_number, edit in sorted(header_edits.items()):
            line_idx = line_number - 1
            _body, ending = self._split_line_ending(annotated_lines[line_idx])
            annotated_lines[line_idx] = f"{'#' * edit.level} {edit.header}{ending}"

        return "".join(annotated_lines)

    def _extract_sections(self, annotated_text: str) -> list[dict[str, Any]]:
        lines = annotated_text.splitlines(keepends=True)
        if not lines and annotated_text:
            lines = [annotated_text]
        line_starts = self._line_starts(lines)

        headers: list[dict[str, Any]] = []
        for idx, line in enumerate(lines):
            body, _ending = self._split_line_ending(line)
            match = re.match(r"^(#{1,3})\s+(.+?)\s*$", body)
            if not match:
                continue
            headers.append(
                {
                    "line_idx": idx,
                    "start": line_starts[idx],
                    "body_start": line_starts[idx] + len(line),
                    "level": len(match.group(1)),
                    "header": self._clean_header(match.group(2)),
                }
            )

        if not headers:
            return [
                {
                    "number": 1,
                    "header": "FULL_DOCUMENT",
                    "level": 1,
                    "parent_headers": [],
                    "span": [0, len(annotated_text)],
                    "text": annotated_text.strip(),
                    "segment_text": f"# FULL_DOCUMENT\n\n{annotated_text.strip()}",
                }
            ]

        sections: list[dict[str, Any]] = []
        if headers[0]["start"] > 0:
            front_matter = annotated_text[: headers[0]["start"]].strip()
            if front_matter:
                sections.append(
                    {
                        "number": len(sections) + 1,
                        "header": "FRONT_MATTER",
                        "level": 1,
                        "parent_headers": [],
                        "span": [0, headers[0]["start"]],
                        "text": front_matter,
                        "segment_text": f"# FRONT_MATTER\n\n{front_matter}",
                    }
                )

        stack: list[dict[str, Any]] = []
        for idx, header in enumerate(headers):
            end = headers[idx + 1]["start"] if idx + 1 < len(headers) else len(annotated_text)
            level = int(header["level"])
            stack = [item for item in stack if int(item["level"]) < level]
            parent_headers = [str(item["header"]) for item in stack]
            segment_text = annotated_text[header["start"] : end].strip()
            sections.append(
                {
                    "number": len(sections) + 1,
                    "header": str(header["header"]),
                    "level": level,
                    "parent_headers": parent_headers,
                    "span": [int(header["start"]), int(end)],
                    "text": annotated_text[header["body_start"] : end].strip(),
                    "segment_text": segment_text,
                }
            )
            stack.append({"level": level, "header": str(header["header"])})

        return sections

    def _save_document(
        self,
        doc_dir: Path,
        source_full_text_path: Path,
        annotated_text: str,
        hierarchy_payload: dict[str, Any],
        sections: list[dict[str, Any]],
    ) -> None:
        output_doc_dir = self.output_dir / doc_dir.name
        output_doc_dir.mkdir(parents=True, exist_ok=True)

        segments_dir = output_doc_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in segments_dir.glob("segment_*.txt"):
            stale_path.unlink()

        serializable_sections = []
        for section in sections:
            section_payload = dict(section)
            segment_text = str(section_payload.pop("segment_text"))
            serializable_sections.append(section_payload)
            segment_path = segments_dir / f"segment_{section_payload['number']}.txt"
            segment_path.write_text(segment_text.strip() + "\n", encoding="utf-8")

        document_meta = {
            "document": {
                "schema_version": SCHEMA_VERSION,
                "document_id": doc_dir.name,
                "source_full_text_path": str(source_full_text_path),
                "model": self.model,
                "organization_description": hierarchy_payload["organization_description"],
                "hierarchy": hierarchy_payload["hierarchy"],
                "chunking": {
                    "planning_first_fraction": PLANNING_FIRST_FRACTION,
                    "planning_random_chunks": self.planning_random_chunks,
                    "annotation_chunk_chars": self.annotate_chunk_chars,
                    "annotation_overlap_chars": self.annotate_overlap_chars,
                },
                "section_count": len(serializable_sections),
            },
            "sections": serializable_sections,
        }

        (output_doc_dir / "full_text.txt").write_text(annotated_text, encoding="utf-8")
        (output_doc_dir / "document_meta.json").write_text(
            json.dumps(document_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def process_document(self, doc_dir: Path, force: bool = False) -> bool:
        output_meta = self.output_dir / doc_dir.name / "document_meta.json"
        if output_meta.exists() and not force:
            print(f"Skipping {doc_dir.name}; output exists. Use --force to rebuild.")
            return False

        full_text_path = doc_dir / "full_text.txt"
        if not full_text_path.exists():
            print(f"Skipping {doc_dir.name}; missing full_text.txt")
            return False

        full_text = full_text_path.read_text(encoding="utf-8", errors="replace")
        hierarchy_payload = self._infer_hierarchy(doc_dir.name, full_text)
        annotated_text = self._annotate_headers(doc_dir.name, hierarchy_payload, full_text)
        sections = self._extract_sections(annotated_text)
        self._save_document(
            doc_dir=doc_dir,
            source_full_text_path=full_text_path,
            annotated_text=annotated_text,
            hierarchy_payload=hierarchy_payload,
            sections=sections,
        )
        print(f"Wrote {len(sections)} sections for {doc_dir.name}")
        return True

    def run(
        self,
        sample_size: int | None = None,
        document_id: str | None = None,
        force: bool = False,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        doc_dirs = self._resolve_document_dirs()

        if document_id is not None:
            doc_dirs = [p for p in doc_dirs if p.name == document_id]

        if sample_size is not None and sample_size < len(doc_dirs):
            rng = random.Random(self.seed)
            doc_dirs = rng.sample(doc_dirs, sample_size)
            doc_dirs = sorted(doc_dirs, key=self._document_sort_key)

        print(f"Processing {len(doc_dirs)} OCR documents from {self.input_dir}")
        processed = 0
        for doc_dir in tqdm(doc_dirs, desc="Segmenting documents", unit="doc"):
            if self.process_document(doc_dir, force=force):
                processed += 1
        print(f"Processed {processed} documents into {self.output_dir}")


def main() -> None:
    PROVIDER = "openrouter"  # "vllm" or "openrouter"
    MODEL = "qwen/qwen3.5-27b"
    # vllm examples:
    #  Qwen/Qwen3.5-27B
    # openrouter:
    # qwen/qwen3.5-27b
    DOL_GROUP = "dol_archive"

    CACHE_DIR = Path(os.environ.get("CACHE_DIR"))
    INPUT_DIRECTORY = CACHE_DIR / "01_ocr_output" / DOL_GROUP / "Qwen_Qwen3_5_27B"
    OUTPUT_DIRECTORY = CACHE_DIR / "02_segmentation_output" / DOL_GROUP

    VLLM_PORT = 8123
    VLLM_MAX_MODEL_LEN = 32768
    NUM_GPUS = 1

    SAMPLE_SIZE = None
    DOCUMENT_ID = None
    SEED = 42
    FORCE = False

    PLANNING_RANDOM_CHUNKS = 4
    PLANNING_RANDOM_CHUNK_CHARS = 2000
    ANNOTATE_CHUNK_CHARS = 1500
    ANNOTATE_OVERLAP_CHARS = 250
    ANNOTATE_WORKERS = 10

    os.environ.setdefault("LOG_DIR", str(CACHE_DIR / "logs"))

    server: VLLMServer | None = None

    try:
        if PROVIDER == "vllm":
            server = VLLMServer(
                model_name=MODEL,
                port=VLLM_PORT,
                max_model_len=VLLM_MAX_MODEL_LEN,
                num_gpus=NUM_GPUS,
            )
            server.start()
            client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{VLLM_PORT}/v1",
                timeout=120,
            )
        elif PROVIDER == "openrouter":
            client = OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
                timeout=120,
            )
        else:
            raise ValueError(f"Unsupported PROVIDER: {PROVIDER}")

        runner = SegmentationRunner(
            client=client,
            model=MODEL,
            input_dir=INPUT_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            seed=SEED,
            planning_random_chunks=PLANNING_RANDOM_CHUNKS,
            planning_random_chunk_chars=PLANNING_RANDOM_CHUNK_CHARS,
            annotate_chunk_chars=ANNOTATE_CHUNK_CHARS,
            annotate_overlap_chars=ANNOTATE_OVERLAP_CHARS,
            annotate_workers=ANNOTATE_WORKERS,
        )
        runner.run(
            sample_size=SAMPLE_SIZE,
            document_id=DOCUMENT_ID,
            force=FORCE,
        )
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()
