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
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        provider: str,
        actors: dict[str, str],
        provisions: dict[str, str],
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
        self.actors = actors
        self.provisions = provisions

    def _normalize_actor(self, value: Any) -> str:
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
        for actor in self.actors:
            actor_normalized = actor.strip().lower()
            if normalized == actor_normalized:
                return aliases.get(actor_normalized, actor_normalized)
        return "unknown"

    def _normalize_provision_type(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "unknown"
        normalized = text.lower()
        for provision_type in self.provisions:
            if normalized == provision_type.lower():
                return normalized
        return "unknown"

    def _build_actor_provision_type_counts(self, sections: list[dict[str, Any]]) -> dict[str, int]:
        counts = {
            f"{self._normalize_actor(actor)} {self._normalize_provision_type(provision_type)}": 0
            for actor in self.actors
            for provision_type in self.provisions
        }
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
                actor = self._normalize_actor(provision.get("actor"))
                provision_type = self._normalize_provision_type(provision.get("provision_type"))
                combo_key = f"{actor.lower()} {provision_type.lower()}"
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

        combo_counts = self._build_actor_provision_type_counts(sections)
        actor_counts = {
            self._normalize_actor(actor).capitalize(): 0
            for actor in self.actors
        }
        actor_counts["unknown"] = 0
        for combo_key, count in combo_counts.items():
            actor_key = combo_key.split(" ", 1)[0]
            actor_label = actor_key.capitalize() if actor_key != "unknown" else "unknown"
            actor_counts[actor_label] = actor_counts.get(actor_label, 0) + count

        document_meta["actor_provision_type_counts"] = combo_counts
        document_meta["actor_counts"] = actor_counts
        document_meta["total_provisions"] = sum(combo_counts.values())
    
    async def process_section(
        self, 
        section: str,
    ) -> tuple[str, float | None]:
        
        system_prompt = " ".join([
            "You are a legal assistant tasked with extracting and categorizing",
            "provision types and which actors they refer to.\n",
            "The actors you should identify and extract are as follows:",
            *self.actors.values(),
            "\nThe provision types you should identify and extract are as follows:",
            *self.provisions.values(),
            "\nReturn your response in a JSON format with the following schema:\n",
            "{provisions: [{'actor': the party involved in the provision, 'provision_type': one of the provision types listed above, 'text': the text of the provision from the contract}]}",
            "If there are no provisions in the text, return {provisions: []}. Only extract provisions that are explicitly stated",
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
                                    "actor": {
                                        "type": "string",
                                        "enum": list(self.actors.keys()),
                                    },
                                    "provision_type": {
                                        "type": "string",
                                        "enum": list(self.provisions.keys()),
                                    },
                                    "text": {"type": "string"},
                                },
                                "required": ["actor", "provision_type", "text"],
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
        return payload.get("provisions", []), cost

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
    actors = {"Worker": WORKER, "Firm": FIRM, "Union": UNION, "Manager": MANAGER}

    OBLIGATION = "A requirement that the subject performs an action or set of actions."
    PROHIBITION = "A requirement that the subject not perform an action or set of actions."
    PERMISSION = "Gives the subject liberty to do an action and others may not stop them from doing so."
    RIGHT = "Gives the subject an entitlement to beneficial actions by other entities."
    provisions = {"Obligation": OBLIGATION, "Prohibition": PROHIBITION, "Permission": PERMISSION, "Right": RIGHT}
    
    PROVIDER = "vllm" # vllm or "openrouter"
    MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
    # vllm: Qwen/Qwen3.5-27B-FP8
    # openrouter: qwen/qwen3.5-27b
    
    input_model_name = "Qwen/Qwen3.5-27B-FP8".replace("/", "-").replace("-", "_").replace(".", "_")
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
    
    if PROVIDER == "vllm":
        # Start a local OpenAI-compatible API server backed by the selected vLLM model.
        server = VLLMServer(model_name=MODEL_NAME, num_gpus=N_GPUS, language_only=True)
        await asyncio.to_thread(server.start)
        runner = ExtractionRunner(
            base_url=f"http://localhost:{server.port}/v1",
            api_key=None,
            model_name=MODEL_NAME,
            provider=PROVIDER,
            actors=actors,
            provisions=provisions,
        )
    else:
        # OpenRouter exposes an OpenAI-compatible chat completions API, so the same
        # Extraction request code can target hosted models without changing worker logic.
        runner = ExtractionRunner(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model_name=MODEL_NAME,
            provider=PROVIDER,
            actors=actors,
            provisions=provisions,
        )
        
    try:
        await runner.process_all(
            input_dir=INPUT_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            cache_file=CACHE_FILE,
            sample_size=SAMPLE_SIZE,
            document_ids=DOCUMENT_IDS,
            n_workers=N_WORKERS,
        )
    finally:
        if PROVIDER == "vllm":
            server.close()

if __name__ == "__main__":
    asyncio.run(main())
