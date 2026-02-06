#!/usr/bin/env python3
"""Extract CBA clause features from PDF documents using vision-based LLM analysis.

Reads a taxonomy of collective bargaining agreement (CBA) clause types from a
Markdown reference file, renders each page of input PDF documents as images,
sends them to an OpenAI vision model to identify which clause types appear on
each page, and writes the results to CSV. Supports incremental processing via
a JSON cache so interrupted runs can be resumed without re-processing pages.

Usage:
    python scripts/extract_cba_features.py [--input-dir cbas] [--model gpt-5-nano] ...

Outputs:
    - cba_features.csv: One row per (document, page, feature) triple.
    - processing_cache.json: Tracks which pages have already been processed.
"""
import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
import random
from dotenv import load_dotenv
load_dotenv()

try:
    import fitz  # PyMuPDF
except Exception:
    try:
        import pymupdf as fitz  # fallback for environments without fitz shim
    except Exception as exc:
        print("Missing dependency: PyMuPDF (fitz). Install with: pip install pymupdf", file=sys.stderr)
        raise
try:
    from tqdm import tqdm
except Exception as exc:
    print("Missing dependency: tqdm. Install with: pip install tqdm", file=sys.stderr)
    raise

try:
    from openai import OpenAI
except Exception as exc:
    print("Missing dependency: openai. Install with: pip install openai", file=sys.stderr)
    raise

# ----- Default paths (relative to repo root) -----
TAXONOMY_PATH = Path("references/feature_taxonomy_final.md")
DEFAULT_INPUT_DIR = Path("cbas")
DEFAULT_OUT_FEATURES = Path("outputs/cba_features.csv")
DEFAULT_OUT_DETAILS = Path("outputs/cba_feature_details.csv")
DEFAULT_CACHE_FILE = Path("outputs/processing_cache.json")

# --------------------------------------------------

def snake(s: str) -> str:
    """Convert an arbitrary string to snake_case (lowercase, non-alphanumeric replaced with underscores)."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def parse_taxonomy(path: Path):
    """Parse the Markdown taxonomy file into a dict of feature definitions.

    Expects headings of the form ``### <number>. <Feature Name>`` followed by
    optional ``**TLDR**: ...`` and ``**Description**: ...`` lines, plus any
    bullet-point details.

    Returns:
        dict mapping feature name -> {"description": str, "tldr": str, "details": list[str]}
    """
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy not found: {path}")
    text = path.read_text(encoding="utf-8")
    features = {}
    current = None
    current_desc = None
    for line in text.splitlines():
        line = line.strip()
        # Match section headings like "### 1. Parties to Agreement and Preamble"
        m = re.match(r"^###\s+\d+\.\s+(.+)$", line)
        if m:
            current = m.group(1).strip()
            current_desc = None
            features[current] = {"description": "", "tldr": "", "details": []}
            continue
        if current and line.startswith("**TLDR**"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                features[current]["tldr"] = parts[1].strip()
            continue
        if current and line.startswith("**Description**"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_desc = parts[1].strip()
                features[current]["description"] = current_desc
            continue
        # Collect any sub-bullet details under the current feature
        if current and line.startswith("-"):
            detail = line.lstrip("- ").strip()
            if detail:
                features[current]["details"].append(detail)
    return features


def build_feature_columns(features):
    """Build a flat list of column names from features and their detail sub-items.

    Each column is formatted as ``<feature_name>__<snake_case_detail>``.
    """
    columns = []
    for feat, details in features.items():
        for d in details:
            col = f"{feat.lower()}__{snake(d)}"
            columns.append(col)
    return columns


def render_pdf_to_images(pdf_path: Path, dpi: int = 200, max_pages: int | None = None):
    """Render each page of a PDF to a JPEG image using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default 200). Higher values produce
             larger images but better OCR/vision quality.
        max_pages: If set, only render the first N pages.

    Returns:
        List of (page_number, jpeg_bytes) tuples (1-indexed page numbers).
    """
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    if max_pages is not None:
        page_count = min(page_count, max_pages)

    images = []
    zoom = dpi / 72  # PyMuPDF default is 72 DPI; scale accordingly
    mat = fitz.Matrix(zoom, zoom)

    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpeg")
        images.append((i + 1, img_bytes))
    return images


def image_to_data_url(img_bytes: bytes) -> str:
    """Encode raw JPEG bytes as a base64 data URL for the OpenAI vision API."""
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_json_loose(text: str):
    """Parse JSON from text, falling back to regex extraction if strict parsing fails.

    Some model responses wrap JSON in markdown fences or extra text; this
    function handles that by searching for the first ``{...}`` or ``[...]``
    block when ``json.loads`` fails on the raw text.
    """
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def get_response_text(response) -> str:
    """Extract the text content from an OpenAI Responses API response object.

    Handles multiple response shapes: the convenience ``output_text`` attribute,
    nested ``output[].content[].text``, and structured ``output[].content[].value``.
    Returns an empty string if no text can be extracted.
    """
    if getattr(response, "output_text", None):
        return response.output_text
    try:
        for item in response.output:
            for content in item.content:
                if getattr(content, "text", None):
                    return content.text
                if getattr(content, "value", None):
                    return json.dumps(content.value)
    except Exception:
        pass
    return ""

def load_cache(path: Path):
    """Load the JSON processing cache from disk, returning an empty structure if missing or corrupt."""
    if not path.exists():
        return {"documents": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"documents": {}}


def save_cache(path: Path, cache):
    """Persist the processing cache to disk as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_response(response) -> str:
    """Serialize an API response to a JSON string for debug logging.

    Tries ``model_dump()`` (Pydantic), then plain ``json.dumps``, then ``str()``
    as a last resort.
    """
    try:
        return json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(response, indent=2, ensure_ascii=False)
        except Exception:
            return str(response)


def build_prompt(features):
    """Construct the system prompt sent to the vision model for each page.

    Lists every feature name (with its TLDR if available) so the model knows
    the allowed classification labels, and instructs it to return structured JSON.
    """
    feature_list = ", ".join(features.keys())
    feature_lines = []
    for feat, meta in features.items():
        tldr = meta.get("tldr", "").strip()
        if tldr:
            feature_lines.append(f"{feat} — {tldr}")
        else:
            feature_lines.append(feat)
    feature_context = "\n".join(feature_lines)
    return f"""
You are given ONE page image from a U.S. collective bargaining agreement.
Identify which features from the allowed list are mentioned on this page.

**HALLUCINATING OR MAKING UP FEATURES WILL LEAD TO SEVERE CONSEQUENCES**
**IF YOU ARE UNSURE WHETHER A FEATURE IS PRESENT, IT IS BETTER TO EXCLUDE IT THAN TO HALLUCINATE.**

Feature descriptions:
{feature_context}

Return JSON with this shape:
{{
  "features": []
}}
""".strip()


def merge_page_features(page_features):
    """De-duplicate a list of feature names while preserving order."""
    dedup = []
    for f in page_features:
        if f not in dedup:
            dedup.append(f)
    return dedup


def main():
    """Entry point: parse args, load taxonomy, iterate over PDFs, and classify pages.

    The pipeline per document:
      1. Render PDF pages to JPEG images.
      2. Skip pages already recorded in the processing cache.
      3. Send each remaining page image to the vision model with the feature prompt.
      4. Parse the model's JSON response and append identified features to the CSV.
      5. Update the cache after each page so progress is never lost.
    """
    parser = argparse.ArgumentParser(
        description="Extract CBA clause features from PDF documents using a vision LLM."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing document_*.pdf files")
    parser.add_argument("--output-features", type=Path, default=DEFAULT_OUT_FEATURES,
                        help="Path for the output features CSV")
    parser.add_argument("--output-details", type=Path, default=DEFAULT_OUT_DETAILS,
                        help="Path for the output details CSV (unused, reserved)")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="If set, write raw model responses here on parse failures")
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE,
                        help="JSON file tracking which pages have been processed")
    parser.add_argument("--model", type=str, default="gpt-5-nano",
                        help="OpenAI model to use for vision classification")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Only process the first N pages of each PDF")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Randomly sample N documents instead of processing all")
    parser.add_argument("--dpi", type=int, default=200,
                        help="DPI for PDF-to-image rendering")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between API calls (rate limiting)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

    # Load the clause taxonomy and add a catch-all "OTHER" category
    features = parse_taxonomy(TAXONOMY_PATH)
    if "OTHER" not in features:
        features["OTHER"] = {"description": "Other feature not in the taxonomy.", "details": []}
    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    args.output_details.parent.mkdir(parents=True, exist_ok=True)
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache_file)

    client = OpenAI()
    prompt = build_prompt(features)

    feature_header = ["document_id", "document_page", "feature_name"]

    # Backfill cache from existing CSV so we don't re-process pages from prior runs
    if args.output_features.exists():
        try:
            with args.output_features.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doc_id = (row.get("document_id") or "").strip()
                    page = (row.get("document_page") or "").strip()
                    if not doc_id or not page:
                        continue
                    try:
                        page_num = int(page)
                    except Exception:
                        continue
                    doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
                    pages = set(doc_cache.get("processed_pages", []))
                    pages.add(page_num)
                    doc_cache["processed_pages"] = sorted(pages)
        except Exception:
            pass
    save_cache(args.cache_file, cache)

    # Open CSV in append mode; write header only if file is new/empty
    with args.output_features.open("a", newline="", encoding="utf-8") as f_feat:
        feat_writer = csv.DictWriter(f_feat, fieldnames=feature_header)
        if f_feat.tell() == 0:
            feat_writer.writeheader()

        pdf_files = sorted(args.input_dir.glob("document_*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {args.input_dir}", file=sys.stderr)
            return
        # Filter out documents where every page is already in the cache
        filtered = []
        for p in pdf_files:
            doc_id = p.stem
            doc_cache = cache.get("documents", {}).get(doc_id, {})
            processed_pages = set(doc_cache.get("processed_pages", []))
            try:
                doc = fitz.open(p)
                page_count = doc.page_count
            except Exception:
                page_count = None
            if page_count is None:
                filtered.append(p)
            else:
                if len(processed_pages) < page_count:
                    filtered.append(p)
        pdf_files = filtered
        if not pdf_files:
            print("All documents appear fully processed per cache.", file=sys.stderr)
            return
        # When sampling, prioritize partially-processed docs to finish them first
        if args.sample_size is not None:
            if args.sample_size <= 0:
                print("--sample-size must be >= 1", file=sys.stderr)
                return
            if args.sample_size < len(pdf_files):
                partially_processed = []
                unprocessed = []
                for p in pdf_files:
                    doc_id = p.stem
                    doc_cache = cache.get("documents", {}).get(doc_id, {})
                    processed_pages = set(doc_cache.get("processed_pages", []))
                    total_pages = doc_cache.get("total_pages")
                    if processed_pages and total_pages and len(processed_pages) < total_pages:
                        partially_processed.append(p)
                    elif not processed_pages:
                        unprocessed.append(p)
                    else:
                        partially_processed.append(p)

                chosen = []
                if partially_processed:
                    pick = min(len(partially_processed), args.sample_size)
                    chosen.extend(random.sample(partially_processed, pick))
                remaining = args.sample_size - len(chosen)
                if remaining > 0 and unprocessed:
                    chosen.extend(random.sample(unprocessed, min(remaining, len(unprocessed))))
                pdf_files = chosen if chosen else random.sample(pdf_files, args.sample_size)

        for pdf_path in pdf_files:
            document_id = pdf_path.stem
            document_title = pdf_path.stem
            print(f"Processing {pdf_path.name} ...", file=sys.stderr)

            images = render_pdf_to_images(pdf_path, dpi=args.dpi, max_pages=args.max_pages)
            doc_cache = cache.setdefault("documents", {}).setdefault(document_id, {})
            processed_pages = set(doc_cache.get("processed_pages", []))
            # Resume from the page after the last one we finished
            start_page = max(processed_pages) + 1 if processed_pages else 1
            images_to_process = [(pnum, b) for (pnum, b) in images if pnum >= start_page and pnum not in processed_pages]
            doc_cache["total_pages"] = len(images)
            if not images_to_process:
                continue

            for page_number, img_bytes in tqdm(images_to_process, desc=f"{pdf_path.name} pages"):
                data_url = image_to_data_url(img_bytes)

                # Send the page image to the vision model with structured JSON output
                response = client.responses.create(
                    model=args.model,
                    input=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_image", "image_url": data_url},
                            ],
                        },
                    ],
                    max_output_tokens=4000,
                    reasoning={"effort": "low"},
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "page_features",
                            "schema": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["features"],
                                "properties": {
                                    "features": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": list(features.keys()),
                                        },
                                    },
                                },
                            },
                        }
                    },
                )

                # Parse the model's response into a feature list
                text = get_response_text(response)
                try:
                    data = parse_json_loose(text)
                except Exception as exc:
                    # On parse failure, save raw output for debugging if requested
                    if args.debug_dir is not None:
                        raw_path = args.debug_dir / f"{pdf_path.stem}_page_{page_number}_raw.txt"
                        raw_path.write_text(text or "", encoding="utf-8")
                        resp_path = args.debug_dir / f"{pdf_path.stem}_page_{page_number}_response.json"
                        resp_path.write_text(dump_response(response), encoding="utf-8")
                    print(f"Failed to parse JSON for {pdf_path.name} page {page_number}", file=sys.stderr)
                    continue

                # Write each identified feature as a row in the CSV
                page_features = data.get("features", []) if isinstance(data, dict) else []
                page_features = merge_page_features(page_features)
                for feat in page_features:
                    feat_writer.writerow(
                        {
                            "document_id": document_id,
                            "document_page": page_number,
                            "feature_name": feat,
                        }
                    )
                # Update and persist cache after every page for crash resilience
                processed_pages.add(page_number)
                doc_cache["processed_pages"] = sorted(processed_pages)
                doc_cache["last_processed_page"] = page_number
                save_cache(args.cache_file, cache)

                if args.sleep:
                    time.sleep(args.sleep)

            # no per-document merge needed for the simplified page-level output


if __name__ == "__main__":
    main()
