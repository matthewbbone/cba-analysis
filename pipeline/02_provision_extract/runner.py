from functools import partial
import json
import random

from openai import OpenAI
import os
import asyncio
from pathlib import Path
import sys
from typing import Any
from tqdm import tqdm

try:
    from pipeline.utils.vllm_server import VLLMServer
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from pipeline.utils.vllm_server import VLLMServer

class ExtractionRunner: 
    _MARKDOWN_ESCAPABLE_CHARACTERS = frozenset("\\`*_{}[]()#+-.!|$")
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        provider: str,
        parties: dict[str, str],
    ):
        
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=240
        )
        self.parties = parties

    def _normalize_party(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "unknown"
        normalized = text.lower()
        aliases = {
            "worker": "worker",
            "workers": "worker",
            "firm": "firm",
            "firms": "firm",
            "union": "union",
            "unions": "union",
            "manager": "manager",
            "managers": "manager",
        }
        if normalized in aliases:
            return aliases[normalized]
        for party in self.parties:
            party_normalized = party.strip().lower()
            if normalized == party_normalized:
                return aliases.get(party_normalized, party_normalized)
        return "unknown"

    @staticmethod
    def _find_unique_match_start(text: str, needle: str) -> int | None:
        start_positions: list[int] = []
        search_start = 0
        while True:
            match_index = text.find(needle, search_start)
            if match_index == -1:
                break
            start_positions.append(match_index)
            search_start = match_index + 1
            if len(start_positions) > 1:
                return None

        if len(start_positions) != 1:
            return None
        return start_positions[0]

    @classmethod
    def _normalize_markdown_escaped_text(cls, text: str) -> tuple[str, list[int]]:
        normalized_chars: list[str] = []
        original_boundaries = [0]
        index = 0

        while index < len(text):
            current_char = text[index]
            if (
                current_char == "\\"
                and index + 1 < len(text)
                and text[index + 1] in cls._MARKDOWN_ESCAPABLE_CHARACTERS
            ):
                normalized_chars.append(text[index + 1])
                index += 2
                original_boundaries.append(index)
                continue

            normalized_chars.append(current_char)
            index += 1
            original_boundaries.append(index)

        return "".join(normalized_chars), original_boundaries

    @classmethod
    def _resolve_unique_span_offsets(
        cls,
        section_text: str,
        span: Any,
    ) -> tuple[int | None, int | None, str]:
        text = str(section_text or "")
        span_text = str(span or "")
        if not text or not span_text:
            return None, None, "unresolved"

        start_pos = cls._find_unique_match_start(text, span_text)
        normalized_text, original_boundaries = cls._normalize_markdown_escaped_text(text)
        normalized_span, _ = cls._normalize_markdown_escaped_text(span_text)
        if normalized_text != text or normalized_span != span_text:
            normalized_start = cls._find_unique_match_start(normalized_text, normalized_span)
            if normalized_start is not None:
                end_index = normalized_start + len(normalized_span)
                normalized_bounds = (
                    original_boundaries[normalized_start],
                    original_boundaries[end_index],
                )
                exact_bounds = (
                    None
                    if start_pos is None
                    else (start_pos, start_pos + len(span_text))
                )
                if normalized_bounds != exact_bounds:
                    return *normalized_bounds, "markdown_escaped"

        if start_pos is not None:
            return start_pos, start_pos + len(span_text), "exact"
        return None, None, "unresolved"

    def _ground_provisions(
        self,
        section_text: str,
        provisions: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(provisions, list):
            return []

        grounded_provisions: list[dict[str, Any]] = []
        for provision in provisions:
            if not isinstance(provision, dict):
                continue

            grounded_provision = dict(provision)
            span_start, span_end, grounding_status = self._resolve_unique_span_offsets(
                section_text=section_text,
                span=grounded_provision.get("span"),
            )
            grounded_provision["span_start"] = span_start
            grounded_provision["span_end"] = span_end
            grounded_provision["grounding_status"] = grounding_status
            grounded_provisions.append(grounded_provision)
        return grounded_provisions

    def _backfill_grounded_spans_in_output_payload(self, output_payload: dict[str, Any]) -> tuple[int, int]:
        sections = output_payload.get("sections", [])
        if not isinstance(sections, list):
            output_payload["sections"] = []
            self._update_output_metadata(output_payload)
            return 0, 0

        updated_sections: list[Any] = []
        grounded_sections = 0
        grounded_provisions = 0

        for section in sections:
            if not isinstance(section, dict):
                updated_sections.append(section)
                continue

            updated_section = dict(section)
            updated_section["provisions"] = self._ground_provisions(
                section_text=updated_section.get("text", ""),
                provisions=updated_section.get("provisions", []),
            )
            grounded_sections += 1
            grounded_provisions += len(updated_section["provisions"])
            updated_sections.append(updated_section)

        output_payload["sections"] = updated_sections
        self._update_output_metadata(output_payload)
        return grounded_sections, grounded_provisions

    def _build_actor_beneficiary_counts(self, sections: list[dict[str, Any]]) -> dict[str, int]:
        counts = {
            f"{self._normalize_party(actor)} {self._normalize_party(beneficiary)}": 0
            for actor in self.parties
            for beneficiary in self.parties
        }
        for party in self.parties:
            counts[f"{self._normalize_party(party)} unknown"] = 0
            counts[f"unknown {self._normalize_party(party)}"] = 0
        counts["unknown unknown"] = 0

        for section in sections:
            if not isinstance(section, dict):
                continue
            provisions = section.get("provisions", [])
            if not isinstance(provisions, list):
                continue
            for provision in provisions:
                if not isinstance(provision, dict):
                    continue
                actor = self._normalize_party(provision.get("subject", provision.get("actor")))
                beneficiary = self._normalize_party(provision.get("beneficiary"))
                combo_key = f"{actor} {beneficiary}"
                counts[combo_key] = counts.get(combo_key, 0) + 1
        return counts

    def _update_output_metadata(self, output_payload: dict[str, Any]) -> None:
        sections = output_payload.get("sections", [])
        if not isinstance(sections, list):
            sections = []

        document_meta = output_payload.get("document_meta_data", {})
        if not isinstance(document_meta, dict):
            document_meta = {}
            output_payload["document_meta_data"] = document_meta

        combo_counts = self._build_actor_beneficiary_counts(sections)
        actor_counts = {
            self._normalize_party(actor).capitalize(): 0
            for actor in self.parties
        }
        actor_counts["unknown"] = 0
        beneficiary_counts = {
            self._normalize_party(beneficiary).capitalize(): 0
            for beneficiary in self.parties
        }
        beneficiary_counts["unknown"] = 0
        for combo_key, count in combo_counts.items():
            actor_key, beneficiary_key = combo_key.split(" ", 1)
            actor_label = actor_key.capitalize() if actor_key != "unknown" else "unknown"
            actor_counts[actor_label] = actor_counts.get(actor_label, 0) + count
            beneficiary_label = beneficiary_key.capitalize() if beneficiary_key != "unknown" else "unknown"
            beneficiary_counts[beneficiary_label] = beneficiary_counts.get(beneficiary_label, 0) + count

        document_meta.pop("actor_provision_type_counts", None)
        document_meta["actor_beneficiary_counts"] = combo_counts
        document_meta["actor_counts"] = actor_counts
        document_meta["beneficiary_counts"] = beneficiary_counts
        document_meta["total_provisions"] = sum(combo_counts.values())
    
    async def process_section(
        self, 
        section: str,
    ) -> tuple[list[dict[str, Any]], float | None]:
        
        system_prompt = " ".join([
            "You are a legal assistant tasked with extracting provisions",
            "and identifying who benefits and who enacts the provision.\n",
            "The legal parties you should identify are as follows:",
            *self.parties.values(),
            "\nReturn your response in a JSON format with the following schema:\n",
            "{provisions: [{'subject': the party that enacts the provision, 'beneficiary': the party that benefits from the provision, 'conditions': the conditions for the provision, 'value': the substantive obligation, prohibition, permission, or right itself, 'span': a minimal verbatim contiguous substring copied from the section text that grounds the provision}]}\n",
            "If there are no provisions with clear beneficiaries and subject in the text, return {provisions: []}.",
            "If there are no conditions stated in the provision, set conditions to 'None'.",
            "The span must be copied exactly from the section text and must be a single contiguous substring.",
            "If you cannot provide an exact verbatim span for a provision, omit that provision instead of paraphrasing.",
        ])
        
        prompt = " ".join([
            "Extract and categorize the provisions in the following text:",
            section
        ])
        
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "extracted_provisions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "provisions"
                    ],
                    "properties": {
                        "provisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "subject": {
                                        "type": "string",
                                        "enum": list(self.parties.keys()),
                                    },
                                    "beneficiary": {
                                        "type": "string",
                                        "enum": list(self.parties.keys()),
                                    },
                                    "conditions": {
                                        "type": "string",
                                    },
                                    "value": {
                                        "type": "string",
                                    },
                                    "span": {
                                        "type": "string",
                                    }
                                },
                                "required": ["subject", "beneficiary", "conditions", "value", "span"],
                            },
                        },
                    },
                },
            }
        }
        
        query = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "response_format": schema,
            "max_tokens": 16384,
        }
        
        # The OpenAI-compatible client is synchronous, so run it in a thread to
        # keep the async worker pool responsive.
        response = await asyncio.to_thread(
            partial(self.client.chat.completions.create, **query)
        )
        raw = response.choices[0].message.content or ""
        cost = getattr(getattr(response, "usage", None), "cost", None)
        payload = json.loads(raw) if raw.strip() else {}
        if payload is None:
            return [], cost
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected response payload: {payload!r}")
        return self._ground_provisions(section, payload.get("provisions", [])), cost

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"documents": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"documents": {}}

    @staticmethod
    def _save_cache(path: Path, cache: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _backfill_cache(output_dir: Path, cache: dict[str, Any]) -> None:
        docs = cache.setdefault("documents", {})
        for doc_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()) if output_dir.exists() else []:
            doc_id = doc_dir.name
            output_path = doc_dir / "provisions.json"
            if not output_path.exists():
                continue
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            processed_sections = sorted(
                section["section_index"]
                for section in payload.get("sections", [])
                if isinstance(section, dict) and "section_index" in section
            )
            if processed_sections:
                doc_cache = docs.setdefault(doc_id, {})
                doc_cache["processed_sections"] = processed_sections
                doc_cache["last_processed_section"] = max(processed_sections)

    @staticmethod
    def _load_output_payload(
        output_path: Path,
        doc_id: str,
        source_payload: dict[str, Any],
        source_path: Path,
    ) -> dict[str, Any]:
        if output_path.exists():
            try:
                return json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "document_meta_data": {
                **source_payload.get("document_meta_data", {}),
                "document_id": doc_id,
                "source_sections_path": str(source_path),
            },
            "sections": [],
        }

    async def process_all(
        self,
        input_dir: Path,
        output_dir: Path,
        cache_file: Path,
        sample_size: int | None,
        document_ids: str | None,
        n_workers: int,
        backfill_grounding_only: bool = False,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache = self._load_cache(cache_file)
        self._backfill_cache(output_dir, cache)

        doc_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir()) if input_dir.exists() else []

        if document_ids:
            target_docs = set(document_ids.split(","))
            doc_dirs = [p for p in doc_dirs if p.name in target_docs]

        if sample_size is not None and sample_size > 0:
            doc_dirs = random.sample(doc_dirs, min(sample_size, len(doc_dirs)))

        if len(doc_dirs) == 0:
            print("No documents to process.")
            return
        else:
            print(f"Processing {len(doc_dirs)} documents...")

        if backfill_grounding_only:
            print("Backfilling grounded spans from existing provision outputs without model calls...")
            docs_updated = 0
            sections_updated = 0
            provisions_updated = 0

            for doc_dir in tqdm(doc_dirs, desc="Backfilling spans", unit="doc"):
                doc_id = doc_dir.name
                output_path = output_dir / doc_id / "provisions.json"
                if not output_path.exists():
                    print(f"Will skip {doc_id}: missing provisions.json")
                    continue

                try:
                    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"Will skip {doc_id}: could not read provisions.json ({exc})")
                    continue

                grounded_section_count, grounded_provision_count = (
                    self._backfill_grounded_spans_in_output_payload(output_payload)
                )
                output_path.write_text(
                    json.dumps(output_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                processed_sections = sorted(
                    section["section_index"]
                    for section in output_payload.get("sections", [])
                    if isinstance(section, dict) and "section_index" in section
                )
                doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
                doc_cache["total_sections"] = len(output_payload.get("sections", []))
                doc_cache["processed_sections"] = processed_sections
                if processed_sections:
                    doc_cache["last_processed_section"] = max(processed_sections)
                else:
                    doc_cache.pop("last_processed_section", None)

                docs_updated += 1
                sections_updated += grounded_section_count
                provisions_updated += grounded_provision_count

            self._save_cache(cache_file, cache)
            print(
                "Backfill complete: "
                f"{docs_updated} docs, {sections_updated} sections, {provisions_updated} provisions re-grounded."
            )
            return

        section_jobs: list[tuple[str, Path, dict[str, Any], int, dict[str, Any]]] = []
        total_sections_processed = 0
        failed_sections = 0
        total_cost = 0.0
        cost_sections = 0
        queue: asyncio.Queue[tuple[str, Path, dict[str, Any], int, dict[str, Any]] | None] = asyncio.Queue(
            maxsize=n_workers * 2
        )
        cache_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()

        for doc_dir in tqdm(doc_dirs, desc="Queueing sections", unit="doc"):
            doc_id = doc_dir.name
            sections_path = doc_dir / "sections.json"
            if not sections_path.exists():
                print(f"Will skip {doc_id}: missing sections.json")
                continue

            source_payload = json.loads(sections_path.read_text(encoding="utf-8"))
            sections = source_payload.get("sections", [])
            doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
            processed_sections = set(doc_cache.get("processed_sections", []))
            doc_cache["total_sections"] = len(sections)

            for idx, section in enumerate(sections):
                if idx in processed_sections:
                    continue
                section_jobs.append((doc_id, sections_path, source_payload, idx, section))

        self._save_cache(cache_file, cache)
        print(f"Total uncached sections to process: {len(section_jobs)}")
        progress_bar = tqdm(total=len(section_jobs), desc="Processing sections", unit="section")

        async def enqueue_jobs() -> None:
            for job in section_jobs:
                await queue.put(job)

        async def worker() -> None:
            nonlocal total_sections_processed, failed_sections, total_cost, cost_sections
            while True:
                try:
                    job = await queue.get()
                except asyncio.CancelledError:
                    break
                if job is None:
                    queue.task_done()
                    return

                doc_id, sections_path, source_payload, section_index, section = job
                try:
                    provisions, cost = await self.process_section(section.get("text", ""))
                    if "openrouter.ai" in self.base_url and cost is not None:
                        total_cost += float(cost)
                        cost_sections += 1

                    async with cache_lock:
                        doc_output_dir = output_dir / doc_id
                        doc_output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = doc_output_dir / "provisions.json"
                        output_payload = self._load_output_payload(
                            output_path=output_path,
                            doc_id=doc_id,
                            source_payload=source_payload,
                            source_path=sections_path,
                        )

                        output_sections = [
                            s
                            for s in output_payload.get("sections", [])
                            if s.get("section_index") != section_index
                        ]
                        output_sections.append(
                            {
                                "section_index": section_index,
                                "header": section.get("header"),
                                "text": section.get("text"),
                                "provisions": provisions,
                            }
                        )
                        output_payload["sections"] = sorted(
                            output_sections,
                            key=lambda s: s["section_index"],
                        )
                        self._update_output_metadata(output_payload)
                        output_path.write_text(
                            json.dumps(output_payload, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )

                        doc_cache = cache["documents"][doc_id]
                        processed_sections = set(doc_cache.get("processed_sections", []))
                        processed_sections.add(section_index)
                        doc_cache["processed_sections"] = sorted(processed_sections)
                        doc_cache["last_processed_section"] = max(processed_sections)
                        self._save_cache(cache_file, cache)
                except Exception as e:
                    print(f"Failed to process {doc_id} section {section_index}: {e}")
                    failed_sections += 1
                finally:
                    async with progress_lock:
                        total_sections_processed += 1
                        progress_bar.update(1)
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
        try:
            await enqueue_jobs()
        finally:
            for _ in workers:
                await queue.put(None)
            await queue.join()
            await asyncio.gather(*workers, return_exceptions=True)
            progress_bar.close()

        if "openrouter.ai" in self.base_url and cost_sections:
            print(f"Average OpenRouter cost per section: {total_cost / cost_sections:.6f} credits")

async def main():
    
    UNION = "A union is an organized association of workers formed to protect and further their rights and interests."
    WORKER = "A worker is an individual performing labor for the firm, who may be a union member or not."
    FIRM = "The firm is the employer or company that the workers are employed by."
    MANAGER = "A manager is an individual who has authority over workers but is not the owner of the firm."
    parties = {"Worker": WORKER, "Firm": FIRM, "Union": UNION, "Manager": MANAGER}
    
    PROVIDER = "vllm" # vllm or "openrouter"
    MODEL_NAME = "Qwen/Qwen3.5-27B"
    # vllm: Qwen/Qwen3.5-27B-FP8
    # openrouter: qwen/qwen3.5-27b
    
    input_model_name = "Qwen/Qwen3.5-27B".replace("/", "-").replace("-", "_").replace(".", "_")
    model_name = MODEL_NAME.replace("/", "-").replace("-", "_").replace(".", "_")
    DOL_GROUP = "cornell_retail_educ" # "dol_archive" "cornell_dol" "cornell_retail_educ"
    
    CACHE_DIR = Path(os.environ.get("CACHE_DIR"))
    INPUT_DIRECTORY =  CACHE_DIR / "01_ocr_output" / DOL_GROUP / input_model_name
    OUTPUT_DIRECTORY = CACHE_DIR / "02_provision_extract" / DOL_GROUP / model_name
    CACHE_FILE = OUTPUT_DIRECTORY / "cache.json"
    
    SAMPLE_SIZE = None
    # DOCUMENT_IDS = "document_2690,document_4027,document_1778,document_3522,document_549"
    # DOCUMENT_IDS = "4208Abbyy,8102ABBYY,7929ABBYY,7423ABBYY,6018ABBYY,3693ABBYY,6513ABBYY,8433ABBYY,K830843_09_07,K800033_12ABBYY,K820213_12_02combined,K800147ABBYY,K811323_06_07combined,K830313_08_07combined"
    DOCUMENT_IDS = "6178_008b185f003_07,6178_008b186f003_01,6178_008b184f002_01,6178_001b022f001_02,6178_008b175f010_03,6178_008b178f008_02"
    N_WORKERS = 10
    N_GPUS = 1
    BACKFILL_GROUNDED_SPANS_ONLY = True
    
    server = None

    if BACKFILL_GROUNDED_SPANS_ONLY:
        runner = ExtractionRunner(
            base_url="http://localhost/backfill-only",
            api_key=None,
            model_name=MODEL_NAME,
            provider=PROVIDER,
            parties=parties,
        )
    elif PROVIDER == "vllm":
        # Start a local OpenAI-compatible API server backed by the selected vLLM model.
        server = VLLMServer(model_name=MODEL_NAME, num_gpus=N_GPUS, language_only=True)
        await asyncio.to_thread(server.start)
        runner = ExtractionRunner(
            base_url=f"http://localhost:{server.port}/v1",
            api_key=None,
            model_name=MODEL_NAME,
            provider=PROVIDER,
            parties=parties,
        )
    else:
        # OpenRouter exposes an OpenAI-compatible chat completions API, so the same
        # Extraction request code can target hosted models without changing worker logic.
        runner = ExtractionRunner(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model_name=MODEL_NAME,
            provider=PROVIDER,
            parties=parties,
        )
        
    try:
        await runner.process_all(
            input_dir=INPUT_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            cache_file=CACHE_FILE,
            sample_size=SAMPLE_SIZE,
            document_ids=DOCUMENT_IDS,
            n_workers=N_WORKERS,
            backfill_grounding_only=BACKFILL_GROUNDED_SPANS_ONLY,
        )
    finally:
        if server is not None:
            server.close()

if __name__ == "__main__":
    asyncio.run(main())
