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
        planning_chars: int,
        boundary_model: str,
        boundary_padding: int,
    ): 
        
        self.cache_dir: Path = Path(cache_dir)
        self.input_dir: Path = self.cache_dir / input_dir
        self.output_dir: Path = self.cache_dir / output_dir
        self.documents: dict[str, Document] = {}
        
        self.planning_model: str = planning_model
        self.planning_chars: int = planning_chars
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1", 
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
        
        text = self.documents[path.name].full_text[:self.planning_chars]
        system_prompt = "\n".join([
            "You are an expert in understanding formal document structures.",
            "You've been asked to identify the hierarchical organization of a collective bargaining agreement.",
            "Infer the highest two-levels of the document's structure.",
            "For instance, it may be structured into 'Articles' with 'Sections' and 'Subsections.",
            "You should identify 'Articles' as the first level and 'Sections' as the second level",
            "without referring to the 'Subsections' as they are not part of the two-level structure.",
            "Return strict JSON with the following fields:",
            '{',
            '  "top_level_type": "...",',
            '  "second_level_type": "...",',
            '  "example_top_level_header": ["..."],'
            '  "example_second_level_header": ["..."]',
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
                        "top_level_type": {"type": "string"},
                        "second_level_type": {"type": "string"},
                        "example_top_level_header": {
                            "type": "array",
                            "items": {"type": "string"}, 
                        },
                        "example_second_level_header": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": [
                        "top_level_type",
                        "second_level_type",
                        "example_top_level_header",
                        "example_second_level_header",
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
            max_tokens=1200,
            temperature=0.0,
        )
        
        payload = utils.parse_json_response(response.choices[0].message.content)
        print(payload)
        self.documents[path.name].plan = payload
        
    def _get_boundary_candidates(self, path: Path):
        
        text = self.documents[path.name].full_text
        candidates = list(re.finditer(rf"\n\n", text))
        
        candidate_texts = []
        for candidate in candidates:
            
            breakpoint = text[candidate.start():candidate.end()]
            pretext = text[candidate.start() - self.boundary_padding:candidate.start()]
            posttext = text[candidate.end():candidate.end() + self.boundary_padding]
            
            candidate_texts.append(
                "".join([
                    pretext,
                    "<BOUNDARY>",
                    breakpoint,
                    "</BOUNDARY>",
                    posttext
                ])
            )
            
        return candidates, candidate_texts
            
    def _evaluate_candidates(self, path: Path, candidates: list[re.Match], candidate_texts: list[str]):
        
        if os.path.exists(self.output_dir / path.name / "boundary_evaluations.json"):
            with open(self.output_dir / path.name / "boundary_evaluations.json", "r") as f:
                payloads = json.load(f)
            return [c for i, c in enumerate(candidates) if payloads[i].get("is_boundary") is True], payloads
        
        plan = self.documents[path.name].plan
        
        system_prompt = "\n".join([
            "You are an expert in understanding formal document structures.",
            "You've been asked to segment a collective bargaining agreement",
            f"by reviewing candidate boundaries between {plan['second_level_type']} of text.",
            "Identify whether the boundary marked by <BOUNDARY> and </BOUNDARY>",
            f"is a valid boundary between two {plan['second_level_type']} of the document.",
            f"Some examples of {plan['second_level_type']} headers are: {', '.join(plan['example_second_level_header'])}",  
            "Return a JSON object with the following fields:",
            '{',
            '  "explanation": "...",',
            '  "is_boundary": true/false,'
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
                        "explanation": {"type": "string"},
                        "is_boundary": {"type": "boolean"},
                    },
                    "required": ["explanation", "is_boundary"]
                }
            }
        }
        
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
                max_tokens=1200,
                temperature=0.0,
            )
            payload = utils.parse_json_response(response.choices[0].message.content)
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

        payloads = asyncio.run(_evaluate_all())
            
        os.makedirs(self.output_dir / path.name, exist_ok=True)
        with open(self.output_dir / path.name / "boundary_evaluations.json", "w") as f:
            json.dump(payloads, f, indent=4)
            
        return [c for i, c in enumerate(candidates) if payloads[i].get("is_boundary") is True]
    
    def _create_segments(self, path: Path, valid_candidates: list[re.Match]):
        
        text = self.documents[path.name].full_text
        segments = []
        
        for i, candidate in enumerate(valid_candidates):
            start = candidate.end()
            end = valid_candidates[i + 1].start() if i + 1 < len(valid_candidates) else len(text)
            segment_span = (start, end, end - start)
            segment_obj = Segment(
                document_id=path.name,
                number=i + 1,
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
        )
        
        self._process_pages(path)
        self._plan_segmentation(path)
        candidates, candidate_texts = self._get_boundary_candidates(path)
        print(f"Assessing {len(candidates)} boundary candidates for document {path.name}")
        valid_candidates = self._evaluate_candidates(path, candidates, candidate_texts)
        self._create_segments(path, valid_candidates)
        self._save_documents(path)
        
    def run(
        self,
        sample_size: int | None = None,
        document_id: str | None = None,
    ):
        
        paths = list(self.input_dir.glob("*/"))
        if document_id is not None:
            paths = [p for p in paths if p.name == document_id]
        elif sample_size is not None:
            random.shuffle(paths)
            paths = paths[:sample_size]
        else:
            paths = self.input_dir.glob("*/")
        
        print(f"Processing {len(paths)} documents from {self.input_dir}")
            
        for path in paths:
            print(f"Processing document: {path.name}")
            self._process_document(path)
        
        
        
def main():
    
    cache_dir = os.environ.get("CACHE_DIR")
    input_dir = "01_ocr_output"
    output_dir = "02_segmentation_output"
    runner = SegmentationRunner(
        cache_dir=cache_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        planning_model='openai/gpt-5.2',
        planning_chars=80_000,
        boundary_model='openai/gpt-5-mini',
        boundary_padding=500,
    )
    runner.run(
        document_id="282ABBYY"
    )
    
if __name__ == "__main__":
    main()
        
