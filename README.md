# CBA Analysis Toolkit

This repository now uses a two-stage clause extraction pipeline:

1. OCR each `document_*.pdf` page into text files.
2. Run clause extraction on OCR text using taxonomy-constrained labels.
3. Review outputs in Streamlit (OCR highlights + stats + metadata heatmaps).

## Pipeline Overview

### Stage 1: OCR (`clause_extraction/ocr/runner.py`)
- Input: PDF files named `document_*.pdf` (typically in `dol_archive/`).
- Output: per-page OCR text files in `ocr_output/document_<id>/page_####.txt`.
- Cache: `outputs/ocr_processing_cache.json`.
- Model serving: OpenAI-compatible endpoint (default `http://localhost:1234/v1`).
- Behavior: resumes from the most recently processed page.
- Behavior: sampling prioritizes partially processed documents.
- Behavior: optional page cap via `--max-pages`.

Run:

```bash
uv run python clause_extraction/ocr/runner.py --input-dir dol_archive --sample-size 5
```

### Stage 2: Clause Extraction (`clause_extraction/extraction/runner.py`)
- Input: OCR text from `ocr_output/`.
- Taxonomy: `references/feature_taxonomy_final.md` (uses clause headings + TLDR context, plus `OTHER`).
- Backend: `vllm` (default) via LangExtract + vLLM-compatible endpoint.
- Backend: `openai` via OpenAI Responses API (requires `OPENAI_API_KEY`).
- Output CSV: `outputs/cba_features.csv` with exact columns `document_id, document_page, feature_name`.
- `outputs/cba_features_annotated.jsonl` for highlighted review spans.
- `outputs/extraction_processing_cache.json` for resumable extraction.
- `outputs/extraction_debug/*.json` page-level debug artifacts.

Run with vLLM backend:

```bash
uv run python clause_extraction/extraction/runner.py --backend vllm --model-id vllm:http://localhost:8000/v1
```

Run with OpenAI backend:

```bash
uv run python clause_extraction/extraction/runner.py --backend openai --openai-model gpt-5-nano
```

Useful options:
- `--sample-size N` sample documents (partial docs prioritized first).
- `--max-pages N` cap page number processed per document.
- `--clear-cache` remove prior outputs/cache/debug and start fresh.
- `--sleep S` add delay between page requests.

## Review UI (`review_ui/app.py`)

Run:

```bash
uv run streamlit run review_ui/app.py
```

Views:
- `OCR Viewer`: PDF page + OCR text with highlighted extracted clauses.
- `Stats`: page-level and document-level frequency bar charts.
- `Heatmap`: partitioned clause prevalence by metadata dimensions (`naics`, `type`, `statefips1`, `union`, `expire_year`) with company lists per partition.

Defaults:
- PDF source: `processed_cbas/`
- OCR source: `ocr_output/`
- Annotations: `outputs/cba_features_annotated.jsonl`
- Metadata: `dol_archive/CBAList_with_statefips.dta`

Auth:
- Set a password in `.streamlit/secrets.toml`:

```toml
REVIEW_UI_PASSWORD = "your-password-here"
```

## Setup

Install dependencies with `uv`:

```bash
uv pip install -r requirements.txt
uv sync
```

Create `.env` in repo root when using OpenAI backend:

```env
OPENAI_API_KEY=your_key_here
```

## Notes

- The legacy one-step script `scripts/extract_cba_features.py` is still present, but the primary workflow is the two-stage OCR + LangExtract pipeline above.
- Document names must follow `document_<int>.pdf` for cache/sampling/metadata linking.
