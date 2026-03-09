import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from dataclasses import dataclass 
import random
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import json
import re
from tqdm import tqdm
import sys
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Literal

try:
    import pipeline.utils.utils as utils
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from pipeline.utils import utils

load_dotenv()

@dataclass
class Chunk:
    document_id: str
    number: str
    span: tuple[int, int, int]

@dataclass
class Page(Chunk):
    pass
    
@dataclass
class Segment(Chunk):
    header: str | None = None
    parent_header: str | None = None
    
@dataclass
class Document:
    document_id: str
    pages: dict[int, Page]
    segments: dict[int, Segment]
    full_text: str
    plan: dict | None = None

class SegmentationRunner:
    
    def __init__(
        self,
        cache_dir: str,
        input_dir: Path,
        output_dir: Path,
        planning_model: str,
        planning_perc: float,
        boundary_model: str,
        boundary_padding: int,
        provider: str = Literal["openai", "openrouter"]
    ): 
        
        self.cache_dir: Path = Path(cache_dir)
        self.input_dir: Path = self.cache_dir / input_dir
        self.output_dir: Path = self.cache_dir / output_dir
        self.documents: dict[str, Document] = {}
        
        self.planning_model: str = planning_model
        self.planning_perc: float = planning_perc
        self.provider: str = provider
        
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        
        self.max_token_param: str = "max_completion_tokens" if provider == "openai" else "max_tokens"
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url, 
            timeout=120
        )
        
        self.boundary_model: str = boundary_model
        self.boundary_padding = boundary_padding
        
    def _process_pages(self, path: Path):
        
        separator = "\n\n"
        offset = 0
        texts = []
        
        page_paths = list(path.glob("page_*.txt"))
        page_paths.sort(key=lambda p: int(p.stem.split("_")[1]))
        
        for page in page_paths:
            
            with open(page, "r") as f:
                text = f.read()
                
            texts.append(text)
                
            page_span = (offset, offset + len(text), len(text))
            offset += len(text) + len(separator)
            
            page_obj = Page(
                document_id=path.name,
                number=int(page.stem.split("_")[1]),
                span=page_span,
            )
            
            self.documents[path.name].pages[page_obj.number] = page_obj
            
        self.documents[path.name].full_text = separator.join(texts)
        
    def _plan_segmentation(self, path: Path):
        
        if os.path.exists(self.output_dir / path.name / "document_meta.json"):
            self.documents[path.name].plan = json.load(open(self.output_dir / path.name / "document_meta.json")).get("plan")
            return 
        
        char_perc = int(len(self.documents[path.name].full_text) * self.planning_perc)
        text = self.documents[path.name].full_text[:char_perc]
        system_prompt = "\n".join([
            "You are an expert in understanding formal document structures.",
            "You've been asked to identify the organization of a collective bargaining agreement.",
            "Infer the highest level segmentation structure of the document",
            "For instance, it may be structured into 'Articles' with 'Sections' and 'Subsections.",
            "You should identify 'Articles' as the highest level.",
            "Then, provide examples of the headers that introduce these segments.",
            "Lastly, provide regex rules that could be used to identify these headers in the text.",
            "The goal is high recall of potential segment boundaries", 
            "so the regex rules should be permissive and capture some false positives.",
            "Return strict JSON with the following fields:",
            '{',
            '  "segment_type": "...",',
            '  "segment_header_examples": ["..."],',
            '  "segment_header_rules": ["..."]',
            "}",
        ])
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "segmentation_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "segment_type": {"type": "string"},
                        "segment_header_examples": {
                            "type": "array",
                            "items": {"type": "string"}, 
                        },
                        "segment_header_rules": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": [
                        "segment_type",
                        "segment_header_examples",
                        "segment_header_rules"
                    ],
                },
            },
        }
        
        response = self.client.chat.completions.create(
            model=self.planning_model,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format=schema,
            **{self.max_token_param: 4800},
            **({"temperature": 0.0} if self.provider != "openai" else {}),
        )
        
        payload = json.loads(response.choices[0].message.content)
        self.documents[path.name].plan = payload
        
    def _get_boundary_candidates(self, path: Path):
        
        text = self.documents[path.name].full_text
        candidates = []
        for rule in self.documents[path.name].plan["segment_header_rules"]:
            try:
                candidates.extend(re.finditer(rule, text, flags=re.MULTILINE))
            except re.error:
                continue
        
        # order and deduplicate candidates
        candidates = sorted(candidates, key=lambda c: c.start())
        unique_candidates = []
        last_end = -1
        for candidate in candidates:
            if candidate.start() >= last_end:
                unique_candidates.append(candidate)
                last_end = candidate.end()
        candidates = unique_candidates
        
        candidate_texts = []
        for candidate in candidates:
        
            pretext = text[candidate.start() - self.boundary_padding:candidate.start()]
            posttext = text[candidate.end():candidate.end() + self.boundary_padding]
        
            candidate_texts.append(
                "".join([
                    pretext,
                    "<BOUNDARY/>",
                    text[candidate.start():candidate.end()],
                    posttext
                ])
            )
            
        return candidates, candidate_texts
            
    def _evaluate_candidates(self, path: Path, candidates: list[re.Match], candidate_texts: list[str]):
        
        if os.path.exists(self.output_dir / path.name / "boundary_evaluations.json"):
            with open(self.output_dir / path.name / "boundary_evaluations.json", "r") as f:
                evaluations = json.load(f)
            return evaluations
        
        plan = self.documents[path.name].plan
        
        system_prompt = "\n".join([
            "You are an expert in understanding formal document structures.",
            "You've been asked to segment a collective bargaining agreement",
            f"by reviewing candidate boundaries between {plan['segment_type']} of the document.",
            f"Decide whether the boundary marked by <BOUNDARY/> is above the {plan['segment_type']} header",
            f"Some examples of {plan['segment_type']} headers are: {', '.join(plan['segment_header_examples'])}",
            f"You should aim for high precision in identifying true {plan['segment_type']} boundaries,",
            f"so only mark a boundary as valid if you are confident it indicates the start of a new {plan['segment_type']}.",
            "If the boundary is in a list of sections like a table of contents, it is not a true boundary.",
            "Return a JSON object with the following fields:",
            '{',
            '  "is_new_segment": true/false,',
            "}",
            ''
        ])
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "boundary_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "is_new_segment": {"type": "boolean"},
                    },
                    "required": ["is_new_segment"],
                }
            }
        }
        
        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        async def _evaluate_one(idx: int, candidate_text: str):
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.boundary_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": candidate_text
                    }
                ],
                response_format=schema,
                **{self.max_token_param: 4800},
                 **({"temperature": 0.0} if self.provider != "openai" else {}),
            )
            payload = json.loads(response.choices[0].message.content)
            return idx, payload

        async def _evaluate_all():
            payloads = [None] * len(candidate_texts)
            tasks = [
                asyncio.create_task(_evaluate_one(i, candidate_text))
                for i, candidate_text in enumerate(candidate_texts)
            ]
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating candidates"):
                idx, payload = await task
                payloads[idx] = payload
            return payloads

        evaluations = asyncio.run(_evaluate_all())
            
        os.makedirs(self.output_dir / path.name, exist_ok=True)
        with open(self.output_dir / path.name / "boundary_evaluations.json", "w") as f:
            json.dump(evaluations, f, indent=4)
            
        return evaluations
    
    def _create_segments(self, path: Path, candidates: list[re.Match], evaluations: list[dict]):
        
        plan = self.documents[path.name].plan
        text = self.documents[path.name].full_text
        segments = []

        spans: list[tuple[int, int]] = []
        if not candidates:
            spans.append((0, len(text)))
        else:
            valid_boundary_starts = [
                candidate.start()
                for i, candidate in enumerate(candidates)
                if i < len(evaluations) and evaluations[i].get("is_new_segment", False)
            ]

            if not valid_boundary_starts:
                spans.append((0, len(text)))
            else:
                span_starts = [0] + valid_boundary_starts
                for i, start in enumerate(span_starts):
                    end = span_starts[i + 1] if i + 1 < len(span_starts) else len(text)
                    spans.append((start, end))

        for start, end in spans:
            if end <= start:
                continue
            segment_span = (start, end, end - start)
            segment_obj = Segment(
                document_id=path.name,
                number=len(segments) + 1,
                span=segment_span,
            )
            segments.append(segment_obj)
            
        self.documents[path.name].segments = {s.number: s for s in segments}
        
    def _save_documents(self, path: Path):
        
        doc_output_dir = self.output_dir / path.name
        os.makedirs(doc_output_dir, exist_ok=True)
        
        with open(doc_output_dir / "document_meta.json", "w") as f:
            json.dump({
                "document_id": self.documents[path.name].document_id,
                "pages": {
                    p.number: {
                        "span": p.span
                    }
                    for p in self.documents[path.name].pages.values()
                },
                "segments": {
                    s.number: {
                        "span": s.span
                    }
                    for s in self.documents[path.name].segments.values()
                },
                "plan": self.documents[path.name].plan,
            }, f, indent=4)
            
        with open(doc_output_dir / "full_text.txt", "w") as f:
            f.write(self.documents[path.name].full_text)
            
        segment_path = doc_output_dir / "segments"
        os.makedirs(segment_path, exist_ok=True)
        for segment in self.documents[path.name].segments.values():
            segment_text = self.documents[path.name].full_text[segment.span[0]:segment.span[1]]
            with open(doc_output_dir / "segments" / f"segment_{segment.number}.txt", "w") as f:
                f.write(segment_text)
        
    def _process_document(self, path: Path):
        
        self.documents[path.name] = Document(
            document_id=path.name,
            pages={},
            segments={},
            full_text="",
            plan=None
        )
        
        self._process_pages(path)
        self._plan_segmentation(path)
        candidates, candidate_texts = self._get_boundary_candidates(path)
        print(f"Assessing {len(candidates)} boundary candidates for document {path.name}")
        evaluations = self._evaluate_candidates(path, candidates, candidate_texts)
        self._create_segments(path, candidates, evaluations)
        self._save_documents(path)
        
    def run(
        self,
        sample_size: int | None = None,
        document_id: str | None = None,
        cached_only: bool = False,
    ):
        
        paths = sorted(self.input_dir.glob("*/"), key=lambda p: p.name)

        if document_id is not None:
            paths = [p for p in paths if p.name == document_id]

        if cached_only:
            paths = [
                p for p in paths
                if (self.output_dir / p.name / "document_meta.json").exists()
                or (self.output_dir / p.name / "boundary_evaluations.json").exists()
            ]

        if sample_size is not None:
            random.shuffle(paths)
            paths = paths[:sample_size]
        
        print(
            f"Processing {len(paths)} documents from {self.input_dir} "
            f"(cached_only={cached_only})"
        )
            
        for path in paths:
            print(f"Processing document: {path.name}")
            self._process_document(path)
        
        
        
def main():
    
    cache_dir = os.environ.get("CACHE_DIR")
    input_dir = Path("01_ocr_output") / "dol_archive"
    output_dir = Path("02_segmentation_output") / "dol_archive"
    runner = SegmentationRunner(
        cache_dir=cache_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        planning_model='gpt-5.2',
        planning_perc=.1,
        boundary_model='gpt-5-mini',
        boundary_padding=300,
        provider="openai"
    )
    runner.run(
        # sample_size=75,
        # document_id="document_790"
        cached_only=True
    )
    
if __name__ == "__main__":
    main()
        
