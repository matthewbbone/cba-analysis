# Collective Bargaining Agreement Analysis

This repository converts scanned collective bargaining agreements (CBAs) into
analysis-ready data. It uses local vision-language and language models to:

1. transcribe each PDF;
2. locate and quote provisions of interest;
3. classify those provisions by economic content and beneficiary; and
4. join the results to contract metadata and produce descriptive figures.

The current application studies how CBAs govern technological change. The same
pipeline can support other provisions by adding a YAML specification.

All models run locally through [vLLM](https://github.com/vllm-project/vllm).
The pipeline does not send contract text to a hosted LLM API.

## What is the empirical object?

The pipeline produces two analysis tables:

| Table | Unit of observation | Typical use |
| --- | --- | --- |
| Document table | One CBA, including CBAs with no extracted provision | Estimate provision prevalence across contracts |
| Provision table | One extracted provision | Study provision content, subtype, and beneficiary |

For the technology taxonomy, each provision receives:

- a hierarchical subtype, such as `preemptive_rights` or `implementation`,
  with a more detailed subtype at level 2; and
- a beneficiary: `worker`, `employer`, or `unclear`, interpreted as the party
  receiving the substantive benefit from the provision. Stage 2 assigns this
  alongside the extraction itself; stage 3 carries it through unchanged.

The main document-level measures are:

- **Subtype prevalence:** the percentage of CBAs containing at least one
  provision of a given subtype. Categories can overlap because one CBA can
  contain several provision types.
- **Beneficiary share:** within each CBA, the percentage of its extracted
  provisions assigned to each beneficiary, averaged across CBAs. These shares
  sum to approximately 100% within a group.

A CBA with no extracted provisions has zero subtype indicators but undefined
beneficiary shares (`NaN`). This distinction matters when constructing samples
or regression outcomes.

## Pipeline at a glance

```text
scanned CBA PDFs
      |
      v
Stage 1: OCR                 page text + complete document text
      |
      v
Stage 2: extraction          grounded quotations of relevant provisions,
                             each with a beneficiary
      |
      v
Stage 3: classification      hierarchical subtype labels
      |
      v
metadata link                document- and provision-level CSVs
      |
      v
plots                        overall, cohort, and industry summaries
```

Intermediate results are cached by source, model, and document. A different
model therefore creates a distinct set of artifacts rather than overwriting an
earlier run.

## Quick start

### 1. Install the environment

The project requires Python 3.13+, [`uv`](https://docs.astral.sh/uv/), a
CUDA-capable machine, and enough GPU memory for the selected checkpoints.

```bash
uv sync
```

Optionally create a repo-root `.env` file. The most important setting is:

```dotenv
CACHE_DIR=/path/to/pipeline/cache
```

If it is omitted, the pipeline uses the repo-local `cache/` directory. Model
weights are downloaded from Hugging Face on first use, so any required access
token must also be available in the environment.

### 2. Run one contract through the pipeline

Place PDFs under `cache/<source>/`. The PDF stem is its `document_id`; for
example, `cache/dol_archive/document_10.pdf` has ID `document_10`.

```bash
# OCR
uv run python pipeline/stg_01_ocr/general/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --model-name ATH-MaaS/OvisOCR2 \
  --device 0 --port 8123

# Extract technology provisions
uv run python pipeline/stg_02_extract/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --provision technology \
  --ocr-model-name ATH-MaaS/OvisOCR2 \
  --model-name google/gemma-4-31B-it \
  --device 0 --port 8123

# Classify at the most detailed available taxonomy level
uv run python pipeline/stg_03_classify/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --provision technology \
  --taxonomy-depth 2 \
  --extract-model-name google/gemma-4-31B-it \
  --model-name Qwen/Qwen3.8-27B-FP8 \
  --device 0 --port 8123
```

Each command starts and stops its own local vLLM server. Use a different port
for each simultaneous process.

### 3. Build analysis files and figures

The metadata join reads `meta_data/harmonized_cba_metadata.csv`, which covers
all three archives. `--source` names the cache folder and selects the archive
to join against (`dol_archive` -> `DoL`, `cornell_dol` -> `Cornell_DoL`,
`cornell_retail_educ` -> `Cornell_RetailEd`); within an archive a document
matches the row whose `filename` stem is its document ID.

```bash
uv run python pipeline/utils/link_classifications.py \
  --clause-type technology \
  --source dol_archive \
  --level 2

uv run python pipeline/utils/plot_classifications.py \
  --clause-type technology \
  --source dol_archive \
  --level 2
```

This writes CSVs and PNGs to `figures/`. Pass `--dry-run` to either utility to
inspect its intended inputs and outputs before writing files.

## Outputs and variables

All paths below are relative to `CACHE_DIR`. Model names are made filesystem
safe by replacing `/` with `_`.

### Stage 1: OCR

```text
stg_01_ocr/<source>/<ocr_model>/<document_id>/
  page_1.txt        raw model response
  page_1.md         normalized page transcription
  ...
  full.txt          pages combined with page markers
```

The general runner works with Hugging Face vision-language models served by
vLLM. Specialized runners are also available for GLM-OCR, MinerU, and
PaddleOCR-VL under `pipeline/stg_01_ocr/specialized/`. Their `layout` modes
first detect page regions and then transcribe them in reading order.

Useful controls include `--sample`, `--seed`, repeatable `--document-id`,
`--document-ids`, `--dpi`, `--concurrency`, and `--force`. Run a script with
`--help` for its complete interface.

### Stage 2: grounded extraction

```text
stg_02_extract/<source>/<extract_model>/<document_id>/<provision>.jsonl
```

Each JSONL row is a quoted provision with fields including:

| Field | Meaning |
| --- | --- |
| `extraction_text` | Text identified as a relevant provision |
| `context` | Model-produced contextual attribute |
| `span_start`, `span_end` | Character offsets in the OCR document |
| `span_reliable` | Whether the offsets can be treated as reliable |
| `grounding_status` | Result of reconciling the quotation to source text |
| `document_id`, `source` | Contract identifiers |
| `ocr_model_name`, `model_name` | Provenance for the OCR and extraction models |

Extraction uses `langextract` to chunk long contracts and reconcile returned
text with the OCR source. Researchers auditing results should use the span and
grounding fields rather than treating every quotation as equally reliable.

### Stage 3: classification

```text
stg_03_classify/<source>/<classify_model>/<document_id>/<provision>.jsonl
```

Each extraction retains its quoted text and offsets and gains `beneficiary`,
`subtype_1`, `subtype_2`, and model-provenance fields. Classification proceeds
down the taxonomy one level at a time. The pipeline adds a terminal `other`
category at every level; if selected, deeper subtype fields remain null.

`--taxonomy-depth` cannot exceed the depth declared by the provision. A
provision without a taxonomy can be extracted but not classified.

### Linked analysis data

`link_classifications.py` creates:

```text
figures/<provision>_<source>_l<level>_documents.csv
figures/<provision>_<source>_l<level>_provisions.csv
```

The document table includes:

- `has_<subtype>` and `n_<subtype>`;
- `n_beneficiary_<label>` and `pct_beneficiary_<label>`;
- employer, union, state, and NAICS code with its sector label;
- effective and expiration year, each with a five-year cohort, plus
  `contract_year`/`contract_period` -- the effective year where there is one,
  falling back to the expiration year. Only the DOL rows carry an expiration,
  so `contract_period` is the cohort to group on across archives;
- employment (`n_workers`) and ownership metadata; and
- the `mistral_*` per-contract extractions (NAICS, occupation count, mean and
  median wage, step-up), which the DOL rows alone carry.

The provision table contains one row per classified provision joined to the
same contract metadata. Add `--include-text` if the analysis file should retain
the extracted quotation.

## Figures and interpretation

For both subtype prevalence and beneficiary share, the plotting utility
produces an overall figure and breakdowns by expiration cohort and industry.

By default, cohort plots contain:

- a solid line showing the raw cohort mean; and
- a dotted industry-composition-adjusted index.

The adjusted series computes the outcome within each NAICS stratum and then
averages strata with equal weights, restricting attention to strata observed in
every plotted cohort. A sector absent from one cohort is folded into `Other
sectors` rather than dropped, so its CBAs stay in the index at a coarser grain
and the index survives a thin end cohort; every figure's caption names the
strata used and the sectors folded. Missing NAICS values form their own stratum,
never folded into the remainder, because missingness is time-patterned in this
corpus. Use `--no-industry-adjusted` to omit the index, `--min-sector-docs` to
change how readily a sector is named, or `--hide-no-naics` to omit the
missing-NAICS row from industry plots.

Since the harmonized metadata resolves a NAICS code for many contracts the old
DOL list left blank, the strata are finer than before and the no-NAICS stratum
is much smaller. That makes the thinnest cohort the binding constraint on the
index: raise `--min-group-docs` to drop a sparse end cohort and recover more
named strata.

Important interpretation limits:

- These outputs are descriptive. Industry adjustment does not identify a
  causal effect of time, technology, or contract institutions.
- Contract expiration is not necessarily the date a clause was negotiated or
  first introduced.
- OCR, extraction, and classification errors are generated measurements and
  may be correlated with document age, scan quality, industry, or contract
  complexity.
- Document-level prevalence weights every CBA equally. It is not employment
  weighted unless the researcher explicitly uses `wrkrs`.
- Results depend on the provision definition, taxonomy, model checkpoints, and
  decoding settings. Preserve these as part of the empirical specification.

Useful plotting thresholds include `--min-group-docs` (default 10),
`--min-sector-docs` (default 5), `--min-cbas`, and `--min-cell-docs`. These can
change the plotted sample and should be reported with results.

## Defining an outcome taxonomy

Provision definitions live in `pipeline/provisions/*.yaml` and contain:

1. `clause_type`, the snake_case name of the clause class, which must match the
   filename stem;
2. `clause_description`, a short description of what the class covers; and
3. optionally, a hierarchical `subtype_taxonomy` used by stage 3.

The extraction prompt itself is not in the YAML. It is built around
`clause_type` and `clause_description` by `_prompt_description` in
`pipeline/stg_02_extract/structure_provision.py`, so every provision asks for
the same attributes (`context` and `beneficiary`) in the same wording.

The included specifications are:

- `technology.yaml`: technology and automation clauses, with two taxonomy
  levels. The first level is `preemptive_rights`, `implementation`, or
  `workforce_management`.
- `wage_table.yaml`: wage-table extraction only; it has no classification
  taxonomy.

Taxonomy labels must be unique snake_case names. Each nonterminal level must
have at least two choices. Do not declare `other`, which is reserved and added
by the classifier.

When adding a new economic concept, treat the YAML as part of the measurement
design: define inclusions and exclusions clearly, keep categories mutually
interpretable, and validate the resulting labels on a human-coded sample.

## Human review and OCR comparison

The `reviewer/` directory contains a Vite app that displays source PDFs beside
OCR and extraction output:

```bash
cd reviewer
npm install
npm run dev
```

Open `http://localhost:5178`. Extraction review highlights spans by grounding
status. OCR comparison records pairwise judgments in
`cache/reviewer/ocr_comparisons.jsonl`.

Rank reviewed OCR models with:

```bash
uv run python pipeline/utils/rank_ocr_models.py
uv run python pipeline/utils/rank_ocr_models.py --top-ocr-models
```

The ranking is a regularized Bradley-Terry-style summary; `both_good` and
`both_bad` judgments enter as fractional ties.

## Reproducibility and validation

Before using generated variables in empirical work:

1. sample contracts across years, industries, and scan-quality strata;
2. audit OCR against the PDF;
3. audit extracted text using grounding status and source offsets;
4. compare classifications with blinded human labels; and
5. report precision, recall, and disagreement by economically relevant strata.

The cache path records the model identity, but a full replication archive
should also retain the Git commit, `uv.lock`, provision YAML, CLI arguments,
random seed, and model revision.

Run the test suite with:

```bash
uv run pytest -q
```

Tests use temporary cache trees and mocked model servers, so they require no
GPU or network access.

## Repository map

```text
pipeline/stg_01_ocr/          PDF transcription
pipeline/stg_02_extract/      grounded provision extraction
pipeline/stg_03_classify/     hierarchical classification
pipeline/provisions/          measurement definitions and taxonomies
pipeline/utils/               metadata linking, plots, paths, model ranking
reviewer/                     browser-based human review tool
meta_data/                    CBA metadata used by the analysis join
cache/                        local inputs, cached artifacts, review data
figures/                      generated analysis tables and plots (gitignored)
tests/                        unit and integration tests with mocked inference
```

`parallel_runs.bash` launches OCR for the three configured sources across
three GPUs. `pipeline/utils/move_extractions.py` copies selected stage-2 and
stage-3 outputs from an external `CACHE_DIR` into the repo-local cache for
review; it never copies OCR text. `main.py` and
`references/provision_taxonomy.json` are legacy placeholders and are not used
by the active pipeline.
