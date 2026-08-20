# CBA Analysis Pipeline

This repository turns scanned collective bargaining agreement (CBA) PDFs into
structured, queryable data: OCR text, grounded provision extractions, and a
classification of each provision by taxonomy subtype and beneficiary (worker /
employer / unclear). Analysis utilities then link that data to CBA metadata and
render figures.

**Everything runs on locally-served, open-weight models via
[vLLM](https://github.com/vllm-project/vllm)** — there is no dependency on a
hosted LLM API. Each stage launches its own vLLM server subprocess and talks to
it over the OpenAI-compatible client, purely as a local wire protocol.

## Pipeline stages

Data flows through three cache-backed stages, each keyed by
`{source}/{model_slug}/{document_id}/...`. `{source}` is a subdirectory of PDFs
(e.g. `dol_archive`, `cornell_dol`, `cornell_retail_educ`); `{model_slug}` is a
filesystem-safe rendering of the HuggingFace model ID that produced the stage's
output (`path_safe_model_name()` in `pipeline/utils/paths.py`, `/` → `_`).

### Stage 1 — OCR (`pipeline/stg_01_ocr/`)

Turns each source PDF into page-level markdown/text. Two families of runner:

- **General** (`general/runner.py`) — any HuggingFace vision-language model
  served through vLLM, one full-page transcription call per page. Default
  `AIDC-AI/Ovis2.6-30B-A3B`, with a special-cased profile for
  `ATH-MaaS/OvisOCR2` (prompt, token limits, and repeat-cleanup tuned for it).
- **Specialized** (`specialized/{glmocr,miner,paddleocr}.py`) — one runner per
  specific checkpoint (`zai-org/GLM-OCR`, `opendatalab/MinerU2.5-Pro-2605-1.2B`,
  `PaddlePaddle/PaddleOCR-VL-1.6`), each supporting `--mode single|layout`
  (MinerU also supports `native`, using its own layout tokens instead of the
  shared detector). `layout` mode pre-segments the page with a shared
  PP-DocLayoutV3 detector (`layout.py`) and OCRs each region separately before
  reassembling reading order.

```bash
uv run python pipeline/stg_01_ocr/general/runner.py \
  --source dol_archive --model-name ATH-MaaS/OvisOCR2 \
  --device 0 --port 8123 --concurrency 16

uv run python -m pipeline.stg_01_ocr.specialized.miner --mode layout --source dol_archive
```

Key flags: `--source`, `--document-id`/`--document-ids` (repeatable),
`--sample`/`--seed`, `--model-name`/`--hf-model`, `--port`, `--num-gpus`,
`--device`, `--max-model-len`, `--gpu-memory-utilization`, `--dpi` (default
200), `--concurrency`, `--force`, `--repetition-retries` (default 2).

Output: `{cache}/stg_01_ocr/{source}/{model_slug}/{document_id}/page_N.txt`
(raw), `page_N.md` (normalized markdown), `full.txt` (whole document,
`--- Page N ---` separated). The shared layout cache lives at
`{cache}/stg_01_ocr/{source}/_layout/{layout_model_slug}/{document_id}/layout.json`.

### Stage 2 — Extract (`pipeline/stg_02_extract/`)

Extracts grounded provision spans from each `full.txt` using
[`langextract`](https://github.com/matthewbbone/langextract) (a fork, pinned in
`pyproject.toml` via `feat/chunk-offset`), configured to call the local vLLM
OpenAI-compatible endpoint rather than a hosted API. `langextract` chunks the
document text and returns extraction spans reconciled against the source
(`grounding_status`, `span_reliable`).

```bash
uv run python pipeline/stg_02_extract/runner.py \
  --provision technology --source dol_archive \
  --model-name google/gemma-4-31B-it --ocr-model-name ATH-MaaS/OvisOCR2
```

`--provision NAME` (required) loads `pipeline/provisions/NAME.yaml` — see
[Provision taxonomy config](#provision-taxonomy-config-pipelineprovisions) below.
Other key flags: `--ocr-model-name` (selects which stage-1 output to read,
default `ATH-MaaS/OvisOCR2`), `--model-name` (default `google/gemma-4-31B-it`),
`--source`, `--document-id`/`--document-ids`, `--sample`/`--seed`, `--force`,
`--concurrency` (documents in parallel).

Output JSONL, one record per extraction, at
`{cache}/stg_02_extract/{source}/{model_slug}/{document_id}/{provision_type}.jsonl`:

```
source, document_id, ocr_model_name, model_name, extraction_class,
extraction_text, attributes: {context}, span_start, span_end,
span_reliable, grounding_status
```

### Stage 3 — Classify (`pipeline/stg_03_classify/`)

Classifies each stage-2 extraction along two dimensions: **beneficiary**
(`worker` / `employer` / `unclear` — which party receives a substantive
benefit) and **subtype**, walked as a taxonomy cascade with guided-decoding
JSON-schema calls: one call settles beneficiary + the top-level subtype, and
one further call per level narrows into the previous level's children. A
reserved `"other"` label is injected as a choice at every level (never declared
in the YAML) and is terminal — choosing it stops the cascade, leaving deeper
`subtype_N` fields `None`.

```bash
uv run python pipeline/stg_03_classify/runner.py \
  --provision technology --taxonomy-depth 2 --source dol_archive \
  --model-name Qwen/Qwen3.8-27B-FP8 --extract-model-name google/gemma-4-31B-it
```

`--taxonomy-depth` (default 1) sets how many cascade levels to run; it cannot
exceed the provision's declared taxonomy depth. `--extract-model-name` selects
which stage-2 output to read. A provision with no `subtype_taxonomy` (see
`wage_table.yaml` below) cannot be classified and this stage will refuse to run
against it.

Output JSONL at
`{cache}/stg_03_classify/{source}/{model_slug}/{document_id}/{provision_type}.jsonl`:

```
source, document_id, extract_model_name, model_name, extraction_class,
extraction_text, span_start, span_end, beneficiary, taxonomy_depth,
subtype_1, subtype_2, ..., subtype_<taxonomy_depth>
```

### Orchestration

`parallel_runs.bash` runs stage 1 across three sources simultaneously, one GPU
and CPU set each, all via `general/runner.py` with `ATH-MaaS/OvisOCR2`:

```bash
bash parallel_runs.bash   # cornell_dol on GPU 0, cornell_retail_educ on GPU 1, dol_archive on GPU 2
```

## Provision taxonomy config (`pipeline/provisions/`)

Each provision type is declared in one YAML file, loaded and validated by
`pipeline/stg_02_extract/structure_provision.py` (`load_provision`). A config
carries both the stage-2 extraction prompt and, optionally, the stage-3
classification taxonomy (`subtype_taxonomy`): a tree of `{label: {description,
subtypes}}`, where labels must be unique snake_case across the whole tree, each
level needs at least two labels, and `"other"` is reserved (declaring it raises
a validation error — stage 3 injects it automatically).

Two provision types currently exist:

- **`technology.yaml`** — new-technology/automation clauses. Level-1 subtypes:
  `preemptive_rights`, `implementation`, `workforce_management`, each with its
  own 6–7 level-2 children (e.g. `preemptive_rights` → `notification_right`,
  `negotiation_right`, `participation_right`, `technology_restriction`, ...).
- **`wage_table.yaml`** — wage-table extraction only; **declares no
  `subtype_taxonomy`**, so it can be extracted (stage 2) but not classified
  (stage 3 refuses to run against it).

## Analysis utilities (`pipeline/utils/`)

### `link_classifications.py`

Reads every classified `{document_id}/{provision_type}.jsonl` under a stage-3
directory and joins it to the CBA metadata list
(`meta_data/CBAList_with_statefips.dta`, keyed by
`document_id == "document_" + cbafile`). Writes two CSVs to `figures/`.

```bash
uv run python pipeline/utils/link_classifications.py \
  --provision-type technology --source dol_archive --level 2
```

The **document table** (`{provision}_{source}_l{level}_documents.csv`) is one
row per classified document, including zero-provision documents, with:

- `has_<subtype>` / `n_<subtype>` — whether/how many provisions of each
  subtype the document carries (a document counts once per subtype however
  many such provisions it holds).
- `n_beneficiary_<label>` / `pct_beneficiary_<label>` for
  `label in (employer, worker, unclear)` — that document's own count/share of
  provisions naming each beneficiary. `pct_beneficiary_*` is `NaN` (not zero)
  for a document with no provisions, since the share is undefined there.
- CBA metadata: `employername`, `union`, `expire_year`, `expire_period` (5-year
  bucket of contract expiration), `naics`/`sector_label`, `wrkrs`, `ownership`.

The **provision table** (`{provision}_{source}_l{level}_provisions.csv`) is one
row per classified provision, joined to the same metadata (`--include-text`
keeps the quoted extraction text).

Key flags: `--source`, `--model-name`, `--provision-type`, `--level`
(taxonomy depth to analyse), `--cache-dir` (default from `CACHE_DIR`),
`--metadata`, `--output-dir` (default `figures/`), `--include-text`, `--dry-run`.

### `plot_classifications.py`

Reads the document CSV `link_classifications.py` wrote and renders figures —
must be run with the same `--provision-type`/`--source`/`--level` the CSV was
built with.

```bash
uv run python pipeline/utils/plot_classifications.py \
  --provision-type technology --source dol_archive --level 2 --no-industry-adjusted
```

For each of two statistics — **subtype prevalence** (% of CBAs carrying a
subtype; a CBA can carry several, so shares overlap) and **beneficiary share**
(each CBA's own % of provisions naming a beneficiary, averaged across CBAs; a
provision names exactly one beneficiary, so shares within a group sum to
~100%) — it writes three figures: overall, by contract-expiration cohort, and
by NAICS industry.

```
{prefix}_subtype_share.png              {prefix}_beneficiary_share.png
{prefix}_share_by_period.png            {prefix}_beneficiary_share_by_period.png
{prefix}_share_by_sector.png            {prefix}_beneficiary_share_by_sector.png
```
(`{prefix} = {provision_type}_{source}_l{level}`)

The two by-cohort figures draw each series **twice**: a solid line for the raw
cohort average, and (unless `--no-industry-adjusted`) a **dotted
industry-composition-adjusted index** — the statistic computed separately
within each NAICS stratum, then averaged across strata with equal weights,
using only strata present in every plotted cohort. This isolates a genuine
within-industry trend from a shift in the corpus's industry mix over time.
CBAs with no NAICS code (roughly half the corpus) are kept as their own
stratum rather than dropped, since that missingness is itself strongly
time-trended; both industry figures also show them as their own "No NAICS
code" row (`--hide-no-naics` to omit it).

Key flags: `--input-dir`/`--output-dir`, `--source`, `--level`,
`--provision-type`, `--min-group-docs` (default 10, drops sparse cohorts),
`--min-sector-docs` (default 5, pools sparse industries into "Other sectors"),
`--min-cbas` (default 0, drops sparse subtype series), `--min-cell-docs`
(default 1, per-cohort inclusion threshold for the adjusted index),
`--no-industry-adjusted`, `--hide-no-naics`, `--dpi`, `--dry-run`.

### `move_extractions.py`

Copies stage-2 (and matching stage-3) outputs for one source from the external
`CACHE_DIR` into the repo-local `cache/`, preserving layout. Never copies
stage-1 OCR text.

```bash
uv run python pipeline/utils/move_extractions.py dol_archive
uv run python pipeline/utils/move_extractions.py dol_archive --document-id document_123 --dry-run
```

### `rank_ocr_models.py`

Fits a regularized Bradley-Terry-style ranking from the reviewer app's pairwise
OCR judgments (`cache/reviewer/ocr_comparisons.jsonl`), treating `both_good`/
`both_bad` verdicts as fractional ties.

```bash
uv run python pipeline/utils/rank_ocr_models.py
uv run python pipeline/utils/rank_ocr_models.py --top-ocr-models
```

Writes `cache/reviewer/ocr_model_rankings.json` and prints a ranking table.

### Shared, non-CLI modules

- `paths.py` — `default_cache_dir()`, `path_safe_model_name()`.
- `gpu.py` — CUDA device-string parsing/validation shared by every runner CLI.
- `vllm_server.py` — the `VLLMServer` class every stage runner uses to launch
  and manage its local vLLM subprocess; not invoked directly.

## The reviewer app (`reviewer/`)

A small TypeScript + Vite single-page app for manually reviewing pipeline
output against source PDFs — reads directly from the repo-local `cache/`
(no copying).

```bash
cd reviewer
npm install
npm run dev   # http://localhost:5178
```

Two modes: **extraction review** (source PDF alongside stage-1 OCR text with
stage-2 extraction spans highlighted by grounding status) and **OCR
comparison** (side-by-side output from two of four OCR models on a sampled
page, with a `left_better`/`right_better`/`both_good`/`both_bad` judgment
appended to `cache/reviewer/ocr_comparisons.jsonl` for `rank_ocr_models.py`).

## Data layout

- **Cache root**: `CACHE_DIR` in `.env` (default `cache` if unset) resolves via
  `default_cache_dir()` in `pipeline/utils/paths.py`. In this deployment it
  points to an external volume outside the repo, where the full pipeline
  output actually lives; the repo-local `cache/` is a smaller working copy
  (source PDFs, `move_extractions.py` output, and `cache/reviewer/`) that the
  reviewer app and test fixtures use directly.
- **`meta_data/CBAList_with_statefips.dta`** — the DOL CBA metadata list
  consumed by `link_classifications.py`, keyed by `cbafile`. Its `naics`/
  `wrkrs`/`ownership` columns needed a left-shift repair for rows with missing
  values (`repair_trailing_columns`), already handled in `load_metadata`.
- **`figures/`** — output of `link_classifications.py` (CSVs) and
  `plot_classifications.py` (PNGs); gitignored.
- **`logs/`** — created on demand by `parallel_runs.bash` (`logs/parallel/`)
  and `VLLMServer`; gitignored.

## Testing

```bash
uv run pytest -q                              # whole suite
uv run pytest tests/test_stg_03_classify.py   # one file
```

Tests are `unittest.TestCase`-style classes run under pytest, building fake
cache trees in temporary directories and mocking out `VLLMServer` — no real
GPU, vLLM server, or network call runs in the suite. `pyproject.toml` sets
`pythonpath = ["."]` so tests import `pipeline.*` directly.

Known drift: `tests/test_stg_02_extract.py` has 5 failing tests asserting a
stale `--default-chat-template-kwargs` value that hasn't been updated to match
a recent `preserve_thinking` addition in `pipeline/stg_02_extract/runner.py`.

## Environment

`pyproject.toml` pins `requires-python = ">=3.13"` and `vllm>=0.26.0`,
`langextract` (from the `matthewbbone/langextract@feat/chunk-offset` fork),
`matplotlib`, `pandas`, `pymupdf`, `pyyaml`, `python-dotenv`. Managed with
[`uv`](https://docs.astral.sh/uv/); `uv run <script>` picks up the project
environment automatically.

A repo-root `.env` (gitignored) configures, per-deployment: `CACHE_DIR`,
`LOG_DIR`, CUDA/GPU visibility, and HF/API tokens. The `openai` package is used
only as a client against each stage's local vLLM server — no hosted API is
called anywhere in the pipeline.

## Legacy / inactive

- `main.py` is an explicit placeholder, not part of the pipeline.
- `review/` is currently empty (distinct from `reviewer/`, the active web app).
- `references/provision_taxonomy.json` is a flat, coarser category list from an
  earlier iteration of this project; it is not read by any current code — the
  live taxonomy lives in `pipeline/provisions/*.yaml`.
