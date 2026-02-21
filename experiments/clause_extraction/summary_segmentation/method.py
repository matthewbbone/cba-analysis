"""Summary segmentation method for clause extraction.

Adapts the PIC (Pseudo-Instruction for Chunking) approach:
1. Summarize the page text using an LLM
2. Compute sentence embeddings for summary and page sentences
3. Group sentences into segments based on semantic similarity shifts
4. Classify each segment against the clause taxonomy
"""

import json
import os
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

DEFAULT_VERSION = "google/gemini-3-flash-preview"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# Cache the embedding model across calls
_embed_model = None


def _get_embed_model(model_name: str = DEFAULT_EMBED_MODEL) -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?])\s+', text)
    # Also split on newlines that look like section breaks
    sentences = []
    for chunk in raw:
        parts = re.split(r'\n\s*\n', chunk)
        for part in parts:
            s = part.strip()
            if s:
                sentences.append(s)
    return sentences


def _summarize(text: str, client: OpenAI, model: str) -> str:
    """Generate a summary of the page text using the LLM."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal document summarizer. "
                    "Summarize the key topics and clauses discussed in the following page from a "
                    "Collective Bargaining Agreement. Be concise but capture all distinct topics."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def _segment_by_similarity(
    sentences: list[str],
    summary_embedding: np.ndarray,
    sentence_embeddings: np.ndarray,
    threshold: float = 0.15,
) -> list[list[int]]:
    """Group consecutive sentences into segments based on similarity to summary.

    Uses shifts in similarity to detect topic boundaries. Sentences with similar
    similarity scores to the summary are grouped together.
    """
    if len(sentences) == 0:
        return []
    if len(sentences) == 1:
        return [[0]]

    # Compute cosine similarity between each sentence and the summary
    similarities = np.dot(sentence_embeddings, summary_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(summary_embedding) + 1e-8
    )

    # Detect boundaries where similarity changes significantly
    segments = []
    current_segment = [0]

    for i in range(1, len(sentences)):
        diff = abs(float(similarities[i]) - float(similarities[i - 1]))
        if diff > threshold:
            segments.append(current_segment)
            current_segment = [i]
        else:
            current_segment.append(i)

    if current_segment:
        segments.append(current_segment)

    return segments


def _classify_segments(
    segments: list[list[int]],
    sentences: list[str],
    taxonomy: list[dict],
    client: OpenAI,
    model: str,
) -> list[dict]:
    """Classify each segment against the taxonomy using the LLM."""
    feature_names = [t["name"] for t in taxonomy]
    canonical = {}
    for name in feature_names:
        canonical[name.lower()] = name
        canonical[re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()] = name
    names = set(feature_names)

    labels_list = "\n".join(
        f"- {t['name']}: {t.get('tldr', '')}" for t in taxonomy if t["name"] != "OTHER"
    )
    labels_list += "\n- OTHER: none of the above apply."

    # Build a single batch prompt for all segments
    segment_texts = []
    for seg_indices in segments:
        seg_text = " ".join(sentences[i] for i in seg_indices)
        segment_texts.append(seg_text)

    numbered = "\n".join(
        f"[Segment {i+1}]: {seg}" for i, seg in enumerate(segment_texts)
    )

    prompt = "\n".join([
        "Classify each text segment below into one of the allowed CBA clause labels.",
        "Use ONLY the labels listed. If none fit, use OTHER.",
        "",
        "Allowed labels:",
        labels_list,
        "",
        "Segments:",
        numbered,
        "",
        "Return strict JSON:",
        '{"classifications": [{"segment": 1, "clause_label": "<label>"}]}',
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a legal clause classifier."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    output_text = response.choices[0].message.content or ""
    if not output_text.strip():
        return [{"clause_label": "OTHER", "extraction_text": seg, "start_pos": None, "end_pos": None}
                for seg in segment_texts]

    try:
        data = json.loads(output_text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", output_text, flags=re.DOTALL)
        if m:
            data = json.loads(m.group(1))
        else:
            return [{"clause_label": "OTHER", "extraction_text": seg, "start_pos": None, "end_pos": None}
                    for seg in segment_texts]

    classifications = data.get("classifications", [])
    label_map = {}
    for c in classifications:
        if isinstance(c, dict):
            seg_num = c.get("segment", 0)
            raw_label = c.get("clause_label", "OTHER")
            label_map[seg_num] = _normalize_label(str(raw_label), canonical, names)

    results = []
    for i, seg_text in enumerate(segment_texts):
        label = label_map.get(i + 1, "OTHER")
        results.append({
            "clause_label": label,
            "extraction_text": seg_text,
            "start_pos": None,
            "end_pos": None,
        })

    return results


def _normalize_label(raw: str, canonical: dict[str, str], names: set[str]) -> str:
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


def _find_span(text: str, extraction_text: str) -> tuple[int | None, int | None]:
    snippet = extraction_text.strip()
    if not snippet:
        return None, None
    idx = text.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    idx = text.lower().find(snippet.lower())
    if idx >= 0:
        return idx, idx + len(snippet)
    return None, None


async def extract_page(
    text: str,
    taxonomy: list[dict],
    version: str = DEFAULT_VERSION,
) -> list[dict]:
    """Extract clause mentions from a single page using summary-guided segmentation.

    Args:
        text: The OCR text content of the page.
        taxonomy: List of dicts with 'name' and 'tldr' keys.
        version: OpenRouter model identifier.

    Returns:
        List of extraction dicts with clause_label, extraction_text, start_pos, end_pos.
    """
    if not text.strip():
        return []

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=120)

    # Step 1: Split into sentences
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Step 2: Summarize the page
    summary = _summarize(text, client, version)
    if not summary.strip():
        # Fallback: treat entire page as one segment
        return [{
            "clause_label": "OTHER",
            "extraction_text": text,
            "start_pos": 0,
            "end_pos": len(text),
        }]

    # Step 3: Compute embeddings
    embed_model = _get_embed_model()
    all_texts = [summary] + sentences
    embeddings = embed_model.encode(all_texts, normalize_embeddings=True)
    summary_embedding = embeddings[0]
    sentence_embeddings = embeddings[1:]

    # Step 4: Segment by similarity shifts
    segments = _segment_by_similarity(sentences, summary_embedding, sentence_embeddings)

    # Step 5: Classify each segment
    extractions = _classify_segments(segments, sentences, taxonomy, client, version)

    # Step 6: Find spans in original text
    for ext in extractions:
        start, end = _find_span(text, ext["extraction_text"])
        ext["start_pos"] = start
        ext["end_pos"] = end

    return extractions


async def extract_document(
    doc_dir: str,
    pages: list[int] | None = None,
    version: str = DEFAULT_VERSION,
    taxonomy: list[dict] | None = None,
) -> dict[int, list[dict]]:
    """Extract clauses from all pages in a document directory.

    Args:
        doc_dir: Path to ocr_output/document_* directory.
        pages: List of 1-indexed page numbers. If None, processes all pages.
        version: OpenRouter model identifier.
        taxonomy: Clause taxonomy list. If None, loads from default path.

    Returns:
        Dict mapping page numbers to lists of extraction dicts.
    """
    from pathlib import Path

    doc_path = Path(doc_dir)
    if taxonomy is None:
        from experiments.clause_extraction.langextract.method import _load_default_taxonomy
        taxonomy = _load_default_taxonomy()

    if pages is None:
        page_files = sorted(doc_path.glob("page_*.txt"))
        pages = []
        for pf in page_files:
            m = re.match(r"page_(\d+)\.txt$", pf.name)
            if m:
                pages.append(int(m.group(1)))

    from tqdm import tqdm

    results = {}
    for page_num in tqdm(pages, desc=f"    summary_seg pages", leave=False):
        page_file = doc_path / f"page_{page_num:04d}.txt"
        if not page_file.exists():
            results[page_num] = []
            continue
        text = page_file.read_text(encoding="utf-8", errors="replace").strip()
        try:
            extractions = await extract_page(text, taxonomy, version)
            results[page_num] = extractions
        except Exception as e:
            print(f"  summary_segmentation page {page_num} failed: {e}")
            results[page_num] = []

    return results
