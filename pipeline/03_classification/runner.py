import asyncio
import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

load_dotenv()


@dataclass
class FeatureMeta:
    name: str
    tldr: str
    description: str


def parse_taxonomy(path: Path) -> list[FeatureMeta]:
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    text = path.read_text(encoding="utf-8")
    rows: list[FeatureMeta] = []
    current_name: str | None = None
    current_tldr = ""
    current_desc = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
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


def build_prompt() -> str:
    return "\n".join(
        [
            "You classify ONE segment of collective bargaining agreement text.",
            "You will be given top candidate labels from an embedding search step.",
            "Choose exactly one final label from the provided candidate labels, or choose OTHER.",
            'Return strict JSON only with shape: {"selected_label": "...", "reason": "..."}',
        ]
    )


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


def normalize_feature_name(raw: str, canonical: dict[str, str], names: set[str]) -> str:
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


class ClauseExtractionRunner:
    def __init__(
        self,
        cache_dir: str | Path,
        input_dir: Path,
        output_dir: Path,
        taxonomy_path: Path,
        model: str = "openai/gpt-5-mini",
        embedding_model: str = "all-mpnet-base-v2",
        candidate_k: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 2,
        timeout: float = 120.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.input_dir = self.cache_dir / input_dir
        self.output_dir = self.cache_dir / output_dir

        if taxonomy_path.is_absolute():
            self.taxonomy_path = taxonomy_path
        else:
            repo_root = Path(__file__).resolve().parents[2]
            self.taxonomy_path = repo_root / taxonomy_path

        self.model = model
        self.embedding_model_name = embedding_model
        self.candidate_k = max(1, int(candidate_k))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache_file = self.output_dir.parent / "03_classification_cache.json"

        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        self.features = parse_taxonomy(self.taxonomy_path)
        self.prompt = build_prompt()

        feature_names = [f.name for f in self.features]
        self.canonical: dict[str, str] = {}
        for name in feature_names:
            self.canonical[name.lower()] = name
            self.canonical[re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()] = name
        self.feature_names_set = set(feature_names)

        self.retrieval_features = [f for f in self.features if f.name != "OTHER"]
        self.embedding_lock = Lock()
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.feature_embeddings = None
        if self.retrieval_features:
            self.feature_embeddings = self.embedding_model.encode(
                [self._feature_to_embedding_text(f) for f in self.retrieval_features],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    @staticmethod
    def _parse_segment_number(path: Path) -> int | None:
        m = re.match(r"segment_(\d+)\.txt$", path.name)
        if not m:
            return None
        return int(m.group(1))

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
        path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _feature_to_embedding_text(feature: FeatureMeta) -> str:
        parts = [feature.name]
        if feature.tldr:
            parts.append(feature.tldr)
        if feature.description:
            parts.append(feature.description)
        return "\n".join(parts)

    @staticmethod
    def _format_candidates_for_prompt(candidates: list[dict[str, Any]]) -> str:
        return json.dumps(candidates, ensure_ascii=False, indent=2)

    def _retrieve_top_candidates(self, text: str) -> list[dict[str, Any]]:
        if not self.retrieval_features or self.feature_embeddings is None:
            return []

        with self.embedding_lock:
            segment_embedding = self.embedding_model.encode(
                [text],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores = util.cos_sim(segment_embedding, self.feature_embeddings)[0]

        k = min(self.candidate_k, len(self.retrieval_features))
        if k <= 0:
            return []

        top = scores.topk(k=k)
        top_indices = top.indices.tolist()
        top_scores = top.values.tolist()

        candidates: list[dict[str, Any]] = []
        for idx, score in zip(top_indices, top_scores):
            feature = self.retrieval_features[int(idx)]
            candidates.append(
                {
                    "feature_name": feature.name,
                    "similarity": float(score),
                    "tldr": feature.tldr,
                    "description": feature.description,
                }
            )
        return candidates

    def _classify_segment(self, text: str) -> tuple[list[str], list[dict[str, Any]]]:
        candidates = self._retrieve_top_candidates(text)
        candidate_names = [c["feature_name"] for c in candidates]

        if not candidate_names:
            return [
                "OTHER"
            ], [
                {
                    "raw_label": "OTHER",
                    "label": "OTHER",
                    "reason": "No candidate labels available.",
                    "top_candidates": [],
                }
            ]

        user_payload = "\n".join(
            [
                "Segment text:",
                text,
                "",
                "Top candidate labels from embedding retrieval:",
                self._format_candidates_for_prompt(candidates),
                "",
                f"Allowed final labels: {', '.join(candidate_names)}, OTHER",
            ]
        )

        attempts = max(1, int(self.max_retries))
        last_error: Exception | None = None
        payload: dict[str, Any] | None = None

        for _ in range(attempts):
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_payload},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            try:
                response = self.client.chat.completions.create(
                    **kwargs,
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
                maybe = parse_json_loose(content)
                if isinstance(maybe, dict):
                    payload = maybe
                    break
            except Exception as exc:
                last_error = exc
                continue

        if payload is None:
            raise ValueError(
                f"Unable to parse OpenRouter response after {attempts} attempt(s): {last_error}"
            )

        raw_label = str(payload.get("selected_label", "OTHER"))
        reason = str(payload.get("reason", "")).strip()
        label = normalize_feature_name(raw_label, self.canonical, self.feature_names_set)
        if label not in set(candidate_names) and label != "OTHER":
            label = "OTHER"

        details = [
            {
                "raw_label": raw_label,
                "label": label,
                "reason": reason,
                "top_candidates": candidates,
            }
        ]

        return [label], details

    def _list_document_dirs(self) -> list[Path]:
        if not self.input_dir.exists():
            return []
        return sorted(
            [
                p for p in self.input_dir.glob("document_*")
                if p.is_dir() and re.match(r"^document_\d+$", p.name)
            ],
            key=lambda p: int(p.name.split("_")[1]),
        )

    def _list_segment_files(self, doc_dir: Path) -> list[Path]:
        return sorted(
            [
                p for p in doc_dir.glob("segments/segment_*.txt")
                if p.is_file() and self._parse_segment_number(p) is not None
            ],
            key=lambda p: self._parse_segment_number(p) or 10**12,
        )

    async def run(
        self,
        sample_size: int | None = None,
        document_id: str | None = None,
        force: bool = False,
        seed: int = 42,
        workers: int = 25,
    ):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cache = self._load_cache(self.cache_file)

        doc_dirs = self._list_document_dirs()
        if document_id is not None:
            doc_dirs = [d for d in doc_dirs if d.name == document_id]

        if sample_size is not None and sample_size < len(doc_dirs):
            random.seed(seed)
            doc_dirs = random.sample(doc_dirs, sample_size)

        print(f"Processing {len(doc_dirs)} segmented documents from {self.input_dir}")

        jobs: list[tuple[str, int, Path, Path]] = []
        done_by_doc: dict[str, set[int]] = {}

        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            segment_files = self._list_segment_files(doc_dir)
            if not segment_files:
                continue

            doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
            done = set(doc_cache.get("processed_segments", []))
            done_by_doc[doc_id] = done
            doc_cache["total_segments"] = len(segment_files)

            out_doc_dir = self.output_dir / doc_id
            out_doc_dir.mkdir(parents=True, exist_ok=True)

            for segment_file in segment_files:
                segment_number = self._parse_segment_number(segment_file)
                if segment_number is None:
                    continue

                if (not force) and (segment_number in done):
                    continue

                output_path = out_doc_dir / f"segment_{segment_number}.json"
                if output_path.exists() and not force:
                    done.add(segment_number)
                    doc_cache["processed_segments"] = sorted(done)
                    doc_cache["last_processed_segment"] = segment_number
                    continue

                jobs.append((doc_id, segment_number, segment_file, output_path))

        self._save_cache(self.cache_file, cache)

        total_jobs = len(jobs)
        if total_jobs == 0:
            print("No unprocessed segments to classify.")
            return

        worker_count = max(1, int(workers))
        print(f"Queued {total_jobs} segments for async classification with {worker_count} workers")

        queue: asyncio.Queue[tuple[str, int, Path, Path] | None] = asyncio.Queue(maxsize=worker_count * 4)
        cache_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()
        progress_bar = tqdm(total=total_jobs, desc="Clause classification segments", unit="segment")

        async def _enqueue_jobs():
            for job in jobs:
                await queue.put(job)
            for _ in range(worker_count):
                await queue.put(None)

        async def _worker():
            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    return

                doc_id, segment_number, segment_file, output_path = job
                try:
                    segment_text = segment_file.read_text(encoding="utf-8", errors="replace").strip()
                    if not segment_text:
                        payload = {
                            "document_id": doc_id,
                            "segment_number": segment_number,
                            "segment_text": "",
                            "labels": [],
                            "extractions": [],
                            "top_candidates": [],
                            "model": self.model,
                            "provider": "openrouter",
                            "embedding_model": self.embedding_model_name,
                            "candidate_k": self.candidate_k,
                        }
                    else:
                        try:
                            labels, details = await asyncio.to_thread(self._classify_segment, segment_text)
                        except Exception as exc:
                            print(f"Failed classification for {doc_id} segment {segment_number}: {exc}")
                            continue
                        payload = {
                            "document_id": doc_id,
                            "segment_number": segment_number,
                            "segment_text": segment_text,
                            "labels": labels,
                            "extractions": details,
                            "top_candidates": details[0]["top_candidates"] if details else [],
                            "model": self.model,
                            "provider": "openrouter",
                            "embedding_model": self.embedding_model_name,
                            "candidate_k": self.candidate_k,
                        }

                    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                    async with cache_lock:
                        done = done_by_doc.setdefault(doc_id, set())
                        done.add(segment_number)
                        doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
                        doc_cache["processed_segments"] = sorted(done)
                        doc_cache["last_processed_segment"] = segment_number
                        self._save_cache(self.cache_file, cache)
                finally:
                    async with progress_lock:
                        progress_bar.update(1)
                    queue.task_done()

        await asyncio.gather(_enqueue_jobs(), *[asyncio.create_task(_worker()) for _ in range(worker_count)])
        await queue.join()
        progress_bar.close()


def main():
    parser = argparse.ArgumentParser(description="Classify segmented CBA text into clause labels with embedding retrieval + OpenRouter.")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("CACHE_DIR", ""))
    parser.add_argument("--input-dir", type=Path, default=Path("02_segmentation_output") / "dol_archive")
    parser.add_argument("--output-dir", type=Path, default=Path("03_classification_output") / "dol_archive")
    parser.add_argument("--taxonomy-path", type=Path, default=Path("references") / "feature_taxonomy_final.md")
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--embedding-model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--candidate-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--document-id", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    if not args.cache_dir:
        raise RuntimeError("CACHE_DIR is not set and --cache-dir was not provided.")

    runner = ClauseExtractionRunner(
        cache_dir=args.cache_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        taxonomy_path=args.taxonomy_path,
        model=args.model,
        embedding_model=args.embedding_model,
        candidate_k=args.candidate_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    asyncio.run(
        runner.run(
        sample_size=args.sample_size,
        document_id=args.document_id,
        force=args.force,
        seed=args.seed,
        workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
