#!/usr/bin/env python3
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

TAXONOMY_PATH = Path("references/feature_taxonomy_final.md")
DEFAULT_INPUT_DIR = Path("cbas")
DEFAULT_OUT_FEATURES = Path("outputs/cba_features.csv")
DEFAULT_OUT_DETAILS = Path("outputs/cba_feature_details.csv")
DEFAULT_CACHE_FILE = Path("outputs/processing_cache.json")

def snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def parse_taxonomy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy not found: {path}")
    text = path.read_text(encoding="utf-8")
    features = {}
    current = None
    current_desc = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^###\s+\d+\.\s+([A-Z0-9_]+)", line)
        if m:
            current = m.group(1)
            current_desc = None
            features[current] = {"description": "", "details": []}
            continue
        if current and line.startswith("**Description**"):
            # format: **Description**: ...
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_desc = parts[1].strip()
                features[current]["description"] = current_desc
            continue
        if current and line.startswith("-"):
            detail = line.lstrip("- ").strip()
            if detail:
                features[current]["details"].append(detail)
    return features


def build_feature_columns(features):
    columns = []
    for feat, details in features.items():
        for d in details:
            col = f"{feat.lower()}__{snake(d)}"
            columns.append(col)
    return columns


def render_pdf_to_images(pdf_path: Path, dpi: int = 200, max_pages: int | None = None):
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    if max_pages is not None:
        page_count = min(page_count, max_pages)

    images = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpeg")
        images.append((i + 1, img_bytes))
    return images


def image_to_data_url(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_json_loose(text: str):
    try:
        return json.loads(text)
    except Exception:
        # attempt to extract first JSON object/array
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def get_response_text(response) -> str:
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
    if not path.exists():
        return {"documents": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"documents": {}}


def save_cache(path: Path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_response(response) -> str:
    try:
        return json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(response, indent=2, ensure_ascii=False)
        except Exception:
            return str(response)


def build_prompt(features):
    feature_list = ", ".join(features.keys())
    feature_lines = []
    for feat, meta in features.items():
        desc = meta.get("description", "").strip()
        if desc:
            feature_lines.append(f"{feat} — {desc}")
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
    dedup = []
    for f in page_features:
        if f not in dedup:
            dedup.append(f)
    return dedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-features", type=Path, default=DEFAULT_OUT_FEATURES)
    parser.add_argument("--output-details", type=Path, default=DEFAULT_OUT_DETAILS)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

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
    # Build cache from existing CSV if present
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

    with args.output_features.open("a", newline="", encoding="utf-8") as f_feat:
        feat_writer = csv.DictWriter(f_feat, fieldnames=feature_header)
        if f_feat.tell() == 0:
            feat_writer.writeheader()

        pdf_files = sorted(args.input_dir.glob("document_*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {args.input_dir}", file=sys.stderr)
            return
        # filter out documents fully processed per cache
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
            start_page = max(processed_pages) + 1 if processed_pages else 1
            images_to_process = [(pnum, b) for (pnum, b) in images if pnum >= start_page and pnum not in processed_pages]
            doc_cache["total_pages"] = len(images)
            if not images_to_process:
                continue

            for page_number, img_bytes in tqdm(images_to_process, desc=f"{pdf_path.name} pages"):
                data_url = image_to_data_url(img_bytes)

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

                text = get_response_text(response)
                try:
                    data = parse_json_loose(text)
                except Exception as exc:
                    if args.debug_dir is not None:
                        raw_path = args.debug_dir / f"{pdf_path.stem}_page_{page_number}_raw.txt"
                        raw_path.write_text(text or "", encoding="utf-8")
                        resp_path = args.debug_dir / f"{pdf_path.stem}_page_{page_number}_response.json"
                        resp_path.write_text(dump_response(response), encoding="utf-8")
                    print(f"Failed to parse JSON for {pdf_path.name} page {page_number}", file=sys.stderr)
                    continue

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
                processed_pages.add(page_number)
                doc_cache["processed_pages"] = sorted(processed_pages)
                doc_cache["last_processed_page"] = page_number
                save_cache(args.cache_file, cache)

                if args.sleep:
                    time.sleep(args.sleep)

            # no per-document merge needed for the simplified page-level output


if __name__ == "__main__":
    main()
