# CBA Analysis

This repository turns archival collective bargaining agreements (CBAs) into structured analysis artifacts. The maintained code is organized as a CLI-first pipeline for OCR, segmentation, clause classification, and generosity scoring, plus Streamlit dashboards for reviewing outputs and a set of summary scripts that generate figures.

The main production path is:

`PDFs -> OCR text -> document segments -> clause labels -> generosity scores -> summary figures`

## What The Repo Contains

### Maintained runnable code

- `pipeline/`
  Main OCR, segmentation, classification, generosity, and summary scripts.
- `review_ui/`
  Streamlit dashboards for inspecting OCR, segmentation, classification, and generosity outputs.
- `clause_extraction/`
  Separate extraction-oriented runners for OCR and clause extraction workflows.
- `generosity/`
  Method-development utilities and comparison scripts for generosity scoring.
- `scripts/`
  Standalone utilities such as direct PDF clause-feature extraction.

### Experiments and references

- `experiments/`
  Evaluation harnesses, reports, and method-specific outputs for OCR, segmentation, sentence parsing, and clause extraction experiments.
- `references/`
  Research references and the clause taxonomy used by the classifier.

### Generated artifacts and source data

- `outputs/`
  Checked-in caches and generated CSV/JSONL artifacts.
- `figures/`
  Generated charts, HTML visualizations, and summary JSON files.
- `dol_archive/`
  Source metadata and example archive files.

## Pipeline Overview

### 1. OCR

Entry point: `pipeline/01_ocr/runner.py`

- Input: `document_*.pdf`
- Output: one directory per document containing `page_####.txt` files plus assembled full text
- Providers:
  - local vLLM
  - Google AI Studio via the OpenAI-compatible Gemini endpoint

### 2. Segmentation

Entry point: `pipeline/02_segment/runner.py`

- Input: OCR output directories
- Output: `document_meta.json`, `full_text.txt`, and `segments/segment_*.txt`
- Method: use an LLM to infer the document structure, score candidate headers, then write top-level segments

### 3. Clause classification

Entry point: `pipeline/03_classification/runner.py`

- Input: segmented text files
- Output: one JSON file per segment with clause labels
- Method: embedding retrieval against `references/feature_taxonomy_final.md` plus an OpenRouter LLM decision step

### 4. Generosity scoring

- `pipeline/04_generosity_gab/runner.py`
  Pairwise/ranking-style Gabriel score aggregation
- `pipeline/04_generosity_ash/runner.py`
  Rule-based ASH baseline from sentence/auth parsing
- `pipeline/04_generosity_llm/runner.py`
  Rubric-based LLM scoring with schema generation, extraction, and evaluation

### 5. Review and summaries

- `review_ui/app3.py`
  Preferred review dashboard
- `pipeline/summary/*.py`
  Figure and table builders for clause prevalence, validation, time series, and document distributions

## Environment Setup

### Python and dependency manager

- Preferred Python: `3.12+` from `pyproject.toml`
- Preferred package manager: `uv`

Bootstrap the environment:

```bash
uv sync
```

If you want to use the project virtualenv directly:

```bash
source .venv/bin/activate
```

### Environment variables

Create a `.env` file or export the variables in your shell:

```bash
export CACHE_DIR=/absolute/path/to/run-cache
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export GOOGLE_API_KEY=...
```

What each variable is used for:

- `CACHE_DIR`
  Base directory for most pipeline inputs and outputs. Many scripts assume it is set.
- `OPENAI_API_KEY`
  Needed by the segmentation stage and some utility scripts.
- `OPENROUTER_API_KEY`
  Needed by classification, `04_generosity_llm`, and some summary/experiment scripts.
- `GOOGLE_API_KEY`
  Needed only if OCR is run with `--provider google`.

### Recommended cache layout

The maintained pipeline works most predictably if `CACHE_DIR` contains:

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

## How To Run The Main Workflow

### OCR

Use explicit output paths so downstream stages see the expected `01_ocr_output` directory name:

```bash
uv run python pipeline/01_ocr/runner.py \
  --provider google \
  --input-dir "$CACHE_DIR/dol_archive" \
  --output-dir "$CACHE_DIR/01_ocr_output/dol_archive" \
  --cache-file "$CACHE_DIR/01_ocr_output/01_ocr_cache.json" \
  --model gemini-2.5-flash
```

If you want local vLLM instead, switch `--provider vllm` and configure the model-related flags.

### Segmentation

```bash
uv run python pipeline/02_segment/runner.py
```

This script currently assumes:

- OCR input is at `$CACHE_DIR/01_ocr_output/dol_archive`
- outputs should be written to `$CACHE_DIR/02_segmentation_output/dol_archive`
- the checked-in `main()` runs with `cached_only=True`, which is useful for reruns but not ideal for a fresh first pass

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

## Review UI

Preferred dashboard:

```bash
uv run streamlit run review_ui/app3.py
```

Notes:

- `review_ui/app3.py` is the broadest dashboard and the best starting point for collaborators.
- `review_ui/app.py` is an older experiment-oriented UI.
- both UIs read heavily from `CACHE_DIR`, plus checked-in `outputs/` and `figures/` artifacts

## Summary And Figure Scripts

Examples:

```bash
uv run python pipeline/summary/clause_distribution.py --classification-dir "$CACHE_DIR/03_classification_output/dol_archive"
uv run python pipeline/summary/validation.py --llm-output-dir "$CACHE_DIR/04_generosity_llm_output/dol_archive" --gab-output-dir "$CACHE_DIR/04_generosity_gab_output/dol_archive"
uv run python pipeline/summary/time_series_search.py --topic ai --ocr-dir "$CACHE_DIR/01_ocr_output/dol_archive"
```

Most summary scripts write final charts into repo-local `figures/`, even when their inputs come from `CACHE_DIR`.

## Other Maintained Utilities

- `scripts/extract_cba_features.py`
  Direct PDF-to-feature extraction utility using an OpenAI vision model.
- `clause_extraction/ocr/runner.py`
  Separate OCR workflow for clause-extraction experiments.
- `clause_extraction/extraction/runner.py`
  Extraction-focused pipeline outside the main `pipeline/01-04_*` path.
- `generosity/compare_scores.py`
  Compare generosity outputs from different methods.

## Outputs You Should Expect

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
- checked-in analysis tables and caches:
  repo-local `outputs/`

## Experiments

`experiments/` contains benchmarking and method-comparison code for OCR, segmentation, sentence parsing, and clause extraction. These folders are useful for research context and replication, but they are not the main collaborator onboarding path and are only summarized here.

## Known Caveats

- The OCR runner defaults to `01_ocr_output_qwen3_5`, but downstream pipeline code often expects `01_ocr_output`. Use explicit OCR output flags if you want the main pipeline stages to line up.
- `pipeline/02_segment/runner.py` is currently configured for cached reruns in `main()`, not a polished first-run CLI.
- `CACHE_DIR` is effectively required for most of the maintained pipeline, even when some scripts also have repo-local fallbacks.
- `review_ui/app3.py` is the most useful dashboard; the other UI files are older variants with narrower or experiment-specific scope.
- `main.py` is a placeholder and is not part of the real execution flow.
- `pyproject.toml` does not appear to declare every third-party package used by all auxiliary scripts and dashboards. If a command fails after `uv sync`, inspect that script's imports and install the missing package.
- `.devcontainer/devcontainer.json` currently points to Python `3.11`, while `pyproject.toml` declares `>=3.12`.
