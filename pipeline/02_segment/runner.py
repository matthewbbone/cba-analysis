from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass 
import random

load_dotenv()

@dataclass
class Chunk:
    number: str
    span: tuple[int, int, int]

@dataclass
class Page(Chunk):
    pass
    
@dataclass
class Segment(Chunk):
    pass
    
@dataclass
class Document:
    document_id: str
    pages: dict[int, Page]
    segments: dict[int, Segment]
    full_text: str

class SegmentationRunner:
    
    def __init__(
        self,
        doc_dir: Path,
    ): 
        
        self.doc_dir: Path = doc_dir
        self.documents: dict[str, Document] = {}
        
    def _process_pages(self, path: Path):
        
        offset = 0
        for page in path.glob("page_*.txt"):
            
            with open(page, "r") as f:
                text = f.read()
                
            page_span = (offset, offset + len(text), len(text))
            offset += len(text) + 1  # +1 for newline
            
            page_obj = Page(
                document_id=path.name,
                page_number=int(page.stem.split("_")[1]),
                page_span=page_span,
            )
        
    def _process_document(self, path: Path):
        
        self._process_pages(path)
        
    def run(
        self,
        sample_size: int | None = None,
    ):
        
        if sample_size is not None:
            paths = list(self.doc_dir.glob("document_*/"))
            random.shuffle(paths)
            paths = paths[:sample_size]
        else:
            paths = self.doc_dir.glob("document_*/")
        
        print(f"Processing {len(paths)} documents from {self.doc_dir}")
            
        for path in paths:
            print(f"Processing document: {path.name}")
            self._process_document(path)
        
        
        
def main():
    
    
    
    
        
        
