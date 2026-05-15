# CBA Analysis Pipeline

This repository processes collective bargaining agreement PDFs into structured
provision categories, category summaries, and pairwise generosity rankings.

The current workflow is model-scoped: LLM-derived outputs are written under a
model slug such as `gpt_5_4_nano`, derived from `gpt-5.4-nano`.

## Pipeline Overview

### 01 OCR

Script:

```bash
uv run pipeline/01_ocr/runner.py
```

Input:

```text
cache/{SOURCE}/*.pdf
```

Output:

```text
cache/01_ocr_output/{SOURCE}/{document_id}/{document_id}_res.json
cache/01_ocr_output/{SOURCE}/{document_id}/{document_id}_res.md
```

This step uses `PaddleOCRVL` with `mlx-vlm-server`. It preserves text, headers,
and tables. OCR labels such as footers, footnotes, page numbers, images, and
aside text are ignored.

### 02 Segment

Script:

```bash
uv run pipeline/02_segment/runner.py
```

Input:

```text
cache/01_ocr_output/{SOURCE}/*/*.json
```

Output:

```text
cache/02_segment_output/{SOURCE}/{document_id}/{document_id}_res.json
cache/02_segment_output/{SOURCE}/{document_id}/{document_id}_res.md
```

This step normalizes OCR blocks into section chunks with:

```text
page_num
type
content
```

Tables are retained as `type: "table"`. Headers are merged when OCR splits a
two-line header into adjacent header blocks.

### 03 Provision Filtering

Script:

```bash
uv run pipeline/03_provisions/runner.py
```

Input:

```text
cache/02_segment_output/{SOURCE}/*/*.json
```

Output:

```text
cache/03_provisions_output/{model_slug}/{SOURCE}/{document_id}_res.json
```

This is a chunk-level filtering step. It does not extract spans. Every input
chunk is preserved and annotated with:

```text
is_provision
subject
beneficiary
provision_filter_usage
provision_filter_error
```

`subject` and `beneficiary` are one of `Worker`, `Firm`, `Union`, `Manager`, or
`null`. If an LLM call fails for a chunk, the chunk is retained with
`is_provision: false` and `provision_filter_error`.

### 04 Classification

Script:

```bash
uv run pipeline/04_classification/runner.py
```

Input:

```text
cache/03_provisions_output/{model_slug}/{SOURCE}/*.json
```

Output:

```text
cache/04_classification_output/{model_slug}/{SOURCE}/{document_id}_res.json
```

This step classifies only chunks where `is_provision` is true. It classifies the
full chunk `content` against categories in:

```text
references/provision_taxonomy.json
```

Each chunk receives:

```text
category
classification_usage
```

Non-provision chunks are preserved with:

```text
category: null
classification_usage: null
```

### 05 Summarize

Script:

```bash
uv run pipeline/05_summarize/runner.py
```

Input:

```text
cache/04_classification_output/{model_slug}/{SOURCE}/*.json
```

Output:

```text
cache/05_summarize_output/{model_slug}/{SOURCE}/{document_id}_res.json
```

This step groups classified chunk text by category and summarizes each category
within a document. The saved JSON contains:

```text
sections
category_summaries
```

Each `category_summaries` entry includes:

```text
category
provision_count
summary
summarization_usage
provisions_input_chars
summary_output_chars
compression_ratio
chars_reduced
```

Categories with no provisions are included with an empty summary and no usage
metadata.

### 06 Generosity Comparisons

Script:

```bash
uv run pipeline/06_generosity/runner.py
```

Input:

```text
cache/05_summarize_output/{model_slug}/*/*.json
```

Output:

```text
cache/06_generosity_output/{model_slug}/bradley_terry_results.json
```

This step samples random document pairs without duplicate unordered pairs. For
each pair and taxonomy category, it compares category summaries in both
directions and stores:

```text
score = 1   contract_1 is more generous
score = -1  contract_2 is more generous
score = 0   tie or inconsistent directional comparison
```

The runner preserves raw comparison records, including both directional LLM
calls and usage/cost metadata.

## ELO-Style Ranking Calculator

Script:

```bash
uv run pipeline/06_generosity/elo_calculator.py --method davidson
uv run pipeline/06_generosity/elo_calculator.py --method bradley-terry
```

The calculator reads saved comparisons from `06_generosity/runner.py` and fits
rankings without making LLM calls.

Methods:

- `bradley-terry`: original Bradley-Terry fit; ties are skipped.
- `davidson`: Davidson tied-comparison model; ties are included.

Default input:

```text
cache/06_generosity_output/{model_slug}/bradley_terry_results.json
```

Default outputs:

```text
cache/06_generosity_output/{model_slug}/bradley-terry_rankings.json
cache/06_generosity_output/{model_slug}/davidson_rankings.json
```

Optional CLI arguments:

```bash
--method bradley-terry|davidson
--model-name gpt-5.4-nano
--input path/to/comparisons.json
--output path/to/rankings.json
```

## LLM Utilities

Shared LLM request and accounting code lives in:

```text
pipeline/utils/llm.py
```

It provides:

- `model_slug(model_name)`
- `LLMClient`
- `LLMClientPool`
- JSON-schema chat requests
- usage and cost extraction
- OpenAI/OpenRouter-style routing

Current configured pricing models:

```text
gpt-5.5
gpt-5.4
gpt-5.4-mini
gpt-5.4-nano
```

Pipeline runners currently use:

```text
MODEL_NAME = "gpt-5.4-nano"
```

## Common Review Scripts

### Cost Summary

```bash
uv run review/openai_cost_summary.py
```

Summarizes token usage and estimated cost for `gpt_5_4_nano` outputs across:

```text
03_provisions
04_classification
05_summarize
06_generosity
```

### Summarization Review

```bash
uv run review/summarization_review.py {source} {document_id} {category}
```

Prints the grouped provision text used in the summarization prompt and the saved
summary for one document/category.

Example:

```bash
uv run review/summarization_review.py cornell_retail_educ 6178_008b186f003_01 Healthcare
```

The script resolves both `{document_id}.json` and `{document_id}_res.json`.

### Summarization Reduction Summary

```bash
uv run review/summarization_reduction_summary.py
```

Reports how much category summarization compresses the grouped chunk text.

### Classification Category Coverage

```bash
uv run review/classification_category_coverage.py
```

Reports category coverage across classified chunks:

- average within-document percent of CBA text by category
- pooled text percent by category
- number and percent of CBAs containing each category

### Generosity Scatter

```bash
uv run review/generosity_scatter.py --method davidson
uv run review/generosity_scatter.py --method bradley-terry
```

Fits rankings via `pipeline/06_generosity/elo_calculator.py`, z-normalizes
category log-strength scores within category, computes document composites as
the mean of category z-scores, and writes:

```text
figures/generosity_scatter.png
```

It also prints the correlation between per-document category-score variance and
document composite score.

### Healthcare Pilot Correlation

```bash
uv run review/healthcare_pilot_correlation.py --method davidson
```

Compares three healthcare generosity score sources:

- `Claude Pilot`: `references/pilot_healthcare_scores.csv`
- `Codex Pilot`: `references/doc_scalar_elo_rank.csv`
- `ELO Generosity`: fitted Healthcare ranking from saved pairwise comparisons

The figure is written to:

```text
figures/healthcare_pilot_correlation.png
```

The ELO score is z-normalized before plotting and correlation calculations.

### Category Header Wordclouds

```bash
uv run review/category_header_wordclouds.py
```

Builds category-specific wordclouds from section titles associated with
classified text chunks. Outputs are written under:

```text
figures/
```

## Configuration Notes

Most pipeline runner scripts currently set these values inside `main()`:

```python
SOURCE = "cornell_dol"
MODEL_NAME = "gpt-5.4-nano"
N_WORKERS = ...
```

Change `SOURCE` to process a different source directory. Change `MODEL_NAME` to
write/read a different model-scoped cache directory.

Common sources used in this project:

```text
cornell_dol
cornell_retail_educ
dol_archive
```

## Current Caveats

- `03_provisions` reviews every segmented chunk, including tables.
- `04_classification` only classifies chunks marked `is_provision`.
- The old span-level `extracted_provisions` schema is no longer supported.
- Pairwise generosity comparisons depend on sampled document pairs and can vary
  with `N` and `RANDOM_SEED` in `pipeline/06_generosity/runner.py`.
- PaddleOCR may miss or misclassify some article headers, especially where
  multi-line headers are close together.
