# CBA Analysis

This repository is organized around two collaborator-facing areas:

- `pipeline/`
  The OCR, segmentation, classification, generosity, and summary-generation pipeline.
- `review_ui/`
  Streamlit dashboards for inspecting pipeline inputs and outputs.

Everything else should be treated as supporting data/artifacts or internal development material.

## Repo Layout

### Collaborator-facing code

- `pipeline/`
  Production pipeline stages and summary scripts.
- `review_ui/`
  Review dashboards, with `app3.py` as the preferred entrypoint.

### Supporting directories

- `dol_archive/`
  Source metadata and archived input files used by the pipeline.
- `references/`
  Supporting reference material, including `feature_taxonomy_final.md` used by classification.
- `outputs/`
  Generated CSV/JSON/JSONL artifacts and caches.
- `figures/`
  Generated charts and visual outputs.

### Internal / development material

- `development/`
  Research code, experiments, older utilities, and other non-collaborator workflows.

## Main Pipeline

The maintained workflow is:

`PDFs -> OCR text -> segments -> clause labels -> generosity scores -> summary figures`

### 1. OCR

Entry point: `pipeline/01_ocr/runner.py`

- Input: `document_*.pdf`
- Output: one folder per document with `page_####.txt` files and assembled full text
- Providers:
  - local vLLM
  - Google AI Studio through the OpenAI-compatible Gemini endpoint

### 2. Segmentation

Entry point: `pipeline/02_segment/runner.py`

- Input: OCR output folders
- Output: `document_meta.json`, `full_text.txt`, and `segments/segment_*.txt`

### 3. Clause classification

Entry point: `pipeline/03_classification/runner.py`

- Input: segmented text
- Output: one JSON file per segment with clause labels
- Taxonomy source: `references/feature_taxonomy_final.md`

### 4. Generosity scoring

- `pipeline/04_generosity_gab/runner.py`
  Gabriel-style ranking/comparison scoring
- `pipeline/04_generosity_ash/runner.py`
  Rule-based baseline scoring
- `pipeline/04_generosity_llm/runner.py`
  Rubric-based LLM scoring

### 5. Summaries and figures

- `pipeline/summary/*.py`
  Figure and summary builders for clause prevalence, validation, distributions, and time-series outputs

## Environment Setup

### Python

- Preferred Python: `3.12+`
- Preferred dependency manager: `uv`

Install dependencies:

```bash
uv sync
```

### Environment variables

Create a `.env` file or export variables directly:

```bash
export CACHE_DIR=/absolute/path/to/run-cache
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export GOOGLE_API_KEY=...
```

What they are used for:

- `CACHE_DIR`
  Base directory for most pipeline inputs and outputs.
- `OPENAI_API_KEY`
  Used by segmentation and some utility paths inside the pipeline.
- `OPENROUTER_API_KEY`
  Used by classification and `04_generosity_llm`.
- `GOOGLE_API_KEY`
  Used only when OCR runs with `--provider google`.

## Recommended Data Layout

The maintained pipeline works best if `CACHE_DIR` contains:

```text
$CACHE_DIR/
  dol_archive/
    document_*.pdf
  01_ocr_output/
    dol_archive/
  02_segmentation_output/
    dol_archive/
  03_classification_output/
    dol_archive/
  04_generosity_gab_output/
    dol_archive/
  04_generosity_ash_output/
    dol_archive/
  04_generosity_llm_output/
    dol_archive/
```

## How To Run

### OCR

The OCR stage now defaults to `01_ocr_output`:

```bash
uv run python pipeline/01_ocr/runner.py \
  --provider google \
  --input-dir "$CACHE_DIR/dol_archive" \
  --output-dir "$CACHE_DIR/01_ocr_output/dol_archive" \
  --cache-file "$CACHE_DIR/01_ocr_output/01_ocr_cache.json" \
  --model gemini-2.5-flash
```

### Segmentation

```bash
uv run python pipeline/02_segment/runner.py
```

Default behavior:

- cached reruns are enabled by default
- use `--no-cached-only` to process all matching OCR documents, including fresh runs without cache artifacts

Examples:

```bash
uv run python pipeline/02_segment/runner.py --document-id document_790
uv run python pipeline/02_segment/runner.py --no-cached-only
```

### Clause classification

```bash
uv run python pipeline/03_classification/runner.py \
  --cache-dir "$CACHE_DIR" \
  --input-dir 02_segmentation_output/dol_archive \
  --output-dir 03_classification_output/dol_archive \
  --taxonomy-path references/feature_taxonomy_final.md \
  --model openai/gpt-5-mini
```

### Gabriel generosity

```bash
uv run python pipeline/04_generosity_gab/runner.py \
  --cache-dir "$CACHE_DIR" \
  --input-dir 02_segmentation_output/dol_archive \
  --classification-dir 03_classification_output/dol_archive \
  --output-dir 04_generosity_gab_output/dol_archive
```

### ASH baseline generosity

```bash
uv run python pipeline/04_generosity_ash/runner.py \
  --cache-dir "$CACHE_DIR" \
  --input-dir 02_segmentation_output/dol_archive \
  --classification-dir 03_classification_output/dol_archive \
  --output-dir 04_generosity_ash_output/dol_archive
```

### Rubric-based LLM generosity

```bash
uv run python pipeline/04_generosity_llm/runner.py \
  --cache-dir "$CACHE_DIR" \
  --classification-dir 03_classification_output/dol_archive \
  --output-dir 04_generosity_llm_output/dol_archive \
  --model openai/gpt-5-mini
```

### Review UI

Preferred dashboard:

```bash
uv run streamlit run review_ui/app3.py
```

Notes:

- `review_ui/app3.py` is the main collaborator dashboard.
- `review_ui/app.py` exists, but should be treated as a legacy/internal review UI rather than the default collaborator surface.

### Summary scripts

Examples:

```bash
uv run python pipeline/summary/clause_distribution.py --classification-dir "$CACHE_DIR/03_classification_output/dol_archive"
uv run python pipeline/summary/validation.py --llm-output-dir "$CACHE_DIR/04_generosity_llm_output/dol_archive" --gab-output-dir "$CACHE_DIR/04_generosity_gab_output/dol_archive"
uv run python pipeline/summary/time_series_search.py --topic ai --ocr-dir "$CACHE_DIR/01_ocr_output/dol_archive"
```

Most summary scripts write charts into repo-local `figures/`, even when their inputs live under `CACHE_DIR`.

## Expected Outputs

- OCR:
  `$CACHE_DIR/01_ocr_output/dol_archive/document_*/page_####.txt`
- segmentation:
  `$CACHE_DIR/02_segmentation_output/dol_archive/document_*/segments/segment_*.txt`
- classification:
  `$CACHE_DIR/03_classification_output/dol_archive/document_*/segment_*.json`
- generosity:
  `$CACHE_DIR/04_generosity_{gab,ash,llm}_output/dol_archive/...`
- figures:
  repo-local `figures/`
- generated tables and caches:
  repo-local `outputs/`

## Known Caveats

- `pipeline/02_segment/runner.py` defaults to cached reruns; pass `--no-cached-only` for a fresh full pass.
- `CACHE_DIR` is effectively required for most collaborator workflows.
- `review_ui/app3.py` is the only review UI that should be treated as the primary collaborator entrypoint.
- `development/` contains moved experimental and internal material and is not required for normal collaborator onboarding.
