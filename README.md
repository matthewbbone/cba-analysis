# Collective Bargaining Agreement Analysis

This repository converts scanned collective bargaining agreements (CBAs) into
analysis-ready data. It uses local vision-language and language models to:

1. transcribe each PDF;
2. locate and quote provisions of interest;
3. enrich them with context summaries and beneficiary labels;
4. classify them by economic content; and
5. join the results to contract metadata and produce descriptive figures.

The current application studies how CBAs govern technological change. The same
pipeline can support other provisions by adding a YAML specification.

Models run locally through [vLLM](https://github.com/vllm-project/vllm) by
default. The general OCR runner and Stages 2–4 can instead use OpenRouter with
an explicit `--endpoint openrouter` override. OpenRouter mode sends rendered
page images during OCR and contract text during extraction, enrichment, and
classification to OpenRouter and the hosted model provider.

## What is the empirical object?

The pipeline produces two analysis tables:

| Table | Unit of observation | Typical use |
| --- | --- | --- |
| Document table | One CBA, including CBAs with no extracted provision | Estimate provision prevalence across contracts |
| Provision table | One extracted provision | Study provision content, subtype, and beneficiary |

For the technology taxonomy, each provision receives:

- a hierarchical subtype, such as `preemptive_rights` or `implementation`,
  with a more detailed subtype at level 2; and
- a beneficiary: `workers`, `employer`, or `unclear`, interpreted as the party
  receiving the substantive benefit from the provision. Stage 3 assigns this
  during enrichment; stage 4 carries it through unchanged.

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
Stage 2: extraction          grounded, source-exact quotations
      |
      v
Stage 3: enrichment          context summaries + beneficiary labels
      |
      v
Stage 4: classification      hierarchical subtype labels
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

To use the optional hosted endpoint, add an OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=your-key-here
```

The presence of this key does not change the default endpoint; every hosted run
must explicitly pass `--endpoint openrouter`.

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

# Enrich with surrounding context and beneficiary
uv run python pipeline/stg_03_enrich/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --provision technology \
  --extract-model-name google/gemma-4-31B-it \
  --model-name Qwen/Qwen3.8-27B-FP8 \
  --device 0 --port 8123

# Classify at the most detailed available taxonomy level
uv run python pipeline/stg_04_classify/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --provision technology \
  --taxonomy-depth 2 \
  --enrich-model-name Qwen/Qwen3.8-27B-FP8 \
  --model-name Qwen/Qwen3.8-27B-FP8 \
  --device 0 --port 8123
```

By default, each command starts and stops its own local vLLM server. Use a
different port for each simultaneous local process. Models that cannot be served
on the shared defaults carry their own serve profile, so
`--model-name Qwen/Qwen3.8-Flash-Next-FP8` shards over 4 GPUs at a 65536-token
context and 0.90 GPU memory utilisation unless `--num-gpus`, `--max-model-len`,
or `--gpu-memory-utilization` says otherwise.

To run the same stages through OpenRouter, select a compatible OpenRouter model
slug explicitly. Stage 1 requires image input, while Stages 2–4 require
structured JSON output. For example, `google/gemini-3.7-flash` supports both:

```bash
# OCR through OpenRouter
uv run python pipeline/stg_01_ocr/general/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --endpoint openrouter \
  --model-name google/gemini-3.7-flash

# Extract technology provisions through OpenRouter
uv run python pipeline/stg_02_extract/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --provision technology \
  --ocr-model-name google/gemini-3.7-flash \
  --endpoint openrouter \
  --model-name google/gemini-3.7-flash

# Enrich those provisions through OpenRouter
uv run python pipeline/stg_03_enrich/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --provision technology \
  --extract-model-name google/gemini-3.7-flash \
  --endpoint openrouter \
  --model-name google/gemini-3.7-flash

# Classify those provisions through OpenRouter
uv run python pipeline/stg_04_classify/runner.py \
  --source dol_archive \
  --document-id document_10 \
  --provision technology \
  --taxonomy-depth 2 \
  --enrich-model-name google/gemini-3.7-flash \
  --endpoint openrouter \
  --model-name google/gemini-3.7-flash
```

Local-only options such as `--port`, `--device`, `--num-gpus`,
`--max-model-len`, and vLLM reasoning-parser settings are ignored in
OpenRouter mode. Provider or model capability errors are returned directly by
OpenRouter; the pipeline does not rewrite or pre-validate model slugs.

Stages 2–4 apply Qwen's recommended thinking-mode sampling defaults to
`qwen/qwen3.8-flash` and the local `Qwen/Qwen3.8-Flash-Next` / `-FP8`
checkpoints: temperature 1.0, top-p 0.95, top-k 20, and presence penalty 0.0.
Local requests also specify min-p 0.0 and repetition penalty 1.0. OpenRouter
requests omit these two neutral settings because Alibaba does not advertise
support for them. Thinking is explicitly enabled.

Flash requests through OpenRouter are restricted to `alibaba`, with fallbacks
disabled and parameter support required. If Alibaba is unavailable or cannot
support the request, it fails instead of routing to another provider. These
defaults do not change other models. Sources:
[Qwen generation guidance](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
and [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).
Use `--force` to regenerate existing outputs with the new settings.

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

The general runner works with vision-language models served locally by vLLM or
remotely through OpenRouter. Specialized runners are also available for
GLM-OCR, MinerU, and PaddleOCR-VL under `pipeline/stg_01_ocr/specialized/`.
Their `layout` modes
first detect page regions and then transcribe them in reading order. These
specialized runners remain vLLM-only because they use model-specific local
serving controls.

Useful controls include `--sample`, `--seed`, repeatable `--document-id`,
`--document-ids`, `--dpi`, `--concurrency`, and `--force`. Run a script with
`--help` for its complete interface.

### Stage 2: grounded extraction

Use `--cuad_test` to restrict extraction to the 102 contracts listed in
`references/cuad/test.json`. This implies `--source cuad`, uses the existing
document-title overrides, and intersects any `--document-id` selection before
applying `--sample`. Only contracts with cached OCR `full.txt` inputs are run.

```bash
uv run python -m pipeline.stg_02_extract.runner \
  --cuad_test --provision joint_ip_ownership \
  --model-name openai/gpt-4.1 --endpoint openrouter
```

```text
stg_02_extract/<source>/<extract_model>/<document_id>/<provision>.jsonl
```

Each JSONL row is a quoted provision with fields including:

| Field | Meaning |
| --- | --- |
| `extraction_text` | Exact OCR text at the final character span |
| `generated_extraction_text` | Text returned by LangExtract before grounding |
| `span_start`, `span_end` | Character offsets in the OCR document |
| `span_reliable` | Whether the offsets can be treated as reliable |
| `grounding_status` | LangExtract's grounding/alignment status |
| `document_id`, `source` | Contract identifiers |
| `ocr_model_name`, `model_name` | Provenance for the OCR and extraction models |

Extraction uses `langextract` to chunk long contracts and reconcile returned
text with the OCR source. Researchers auditing results should use the span and
grounding fields rather than treating every quotation as equally reliable.

Stage 2 asks LangExtract only for provision text. Researchers auditing results
can compare the generated and source-exact fields alongside the grounding data.

### Alternative Stage 2: sentence-ID extraction

`runner2` asks the model to select sentence ranges and reconstructs extracted
passages directly from the original OCR text:

```bash
UV_PROJECT_ENVIRONMENT=/VData/scro4406/cba_analysis/.venv \
uv run python -m pipeline.stg_02_extract.runner2 \
  --source negotiating_tech --provision technology \
  --ocr-model-name ATH-MaaS/OvisOCR2 --model-name Qwen/Qwen3.8-27B \
  --device 0 --port 8124
```

It shares the original runner's document selection, `--cuad_test`, sampling,
GPU, progress, and endpoint flags. Use `--endpoint openrouter` with
`OPENROUTER_API_KEY` for hosted models. `--max-tokens` defaults to 5,000 per
response; `--max-model-len` follows the original runner's model defaults.
Runner2 disables thinking on every request (`enable_thinking=false` for vLLM,
`reasoning.enabled=false` for OpenRouter), overriding model profiles and shared
server request defaults. This applies to all five orchestrator models. Restart
any running invocation to apply changes; use `--force` or a fresh `--output-root`
when switching from existing thinking-enabled results because their cache
fingerprints differ.
The runner owns one server per invocation; programmatic callers can borrow
one using `run(args, server=server)` without transferring lifecycle ownership.

Chonkie 1.7.0 prepares sentences once per document through a validated local
adapter. Headings and table rows remain whole; list markers stay attached to
their first sentence. Document-wide IDs appear as `[S1]`, `[S2]`, etc. in the
prompt and as `S1`, `S2`, etc. in responses. Recursive Markdown chunk proposals
use `N_EXTRACTION_PASSES` and `MAX_CHAR_BUFFER`, snapped to these fixed units.
Each chunk includes its enclosing heading and neighboring units as selectable
context. Character budgets are soft: oversized units remain whole, and endpoint
context errors fail the document rather than truncating its text. Passes run
sequentially, with chunk requests bounded by `LANGEXTRACT_MAX_WORKERS` and
`LANGEXTRACT_BATCH_LENGTH`.

Overlapping selections merge; adjacent selections remain separate. Quotations
retain source punctuation, Markdown, and intervening whitespace. These are
extracted passages, not inferred clause boundaries. The compatibility field
`generated_extraction_text` contains the reconstructed quotation: the model
actually generates IDs. `grounding_status: sentence_ids` and reliable character
offsets identify this method. Recall and retained-text targets still require
empirical evaluation.

Outputs use a separate root (overridable with `--output-root`):

```text
CACHE_DIR/stg_02_extract_runner2/<source>/<safe_model>/<document_id>/<provision>.jsonl
CACHE_DIR/stg_02_extract_runner2/<source>/<safe_model>/<document_id>/<provision>.manifest.json
```

The manifest records the source hash, sentence offsets, prompt/configuration
fingerprint, request settings, responses and usage, and selection provenance.
It is the completion marker for atomic publication. Only matching completed
outputs with a valid file hash are reused; mismatches require `--force` or a
different output root. Failed and incomplete documents are retried. An empty
JSONL with a completed manifest is a valid abstention. Invalid ranges receive
one corrective retry; request errors, missing content, and truncated responses
remain failures. Other documents continue, and any document failure gives a
nonzero exit status.

### Stage 3: enrichment

```text
stg_03_enrich/<source>/<enrich_model>/<document_id>/<provision>.jsonl
```

Each grounded extraction gains a top-level `context` summary and `beneficiary`.
The enrichment prompt uses a sentence-bounded window recovered from the Stage 1
OCR text and sized by `ENRICH_MAX_CHAR_BUFFER` in the provision config.

### Stage 4: classification

```text
stg_04_classify/<source>/<classify_model>/<document_id>/<provision>.jsonl
```

Each extraction retains its quoted text, offsets, grounding metadata, context,
and beneficiary, and gains `subtype_1`, `subtype_2`, and separate extraction,
enrichment, and classification model-provenance fields. Classification proceeds
down the taxonomy one level at a time. The pipeline adds a terminal `other`
category at every level; if selected, deeper subtype fields remain null.

`--taxonomy-depth` cannot exceed the depth declared by the provision. A
provision without a taxonomy can be extracted and enriched but not classified.

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

## Sequential CUAD sweep

Run both extraction methods for all 41 clause types and all 102 CUAD test
contracts, with one vLLM server per model:

```bash
# Validate all test inputs and print the 410-job schedule without inference.
uv run python -m pipeline.stg_02_extract.orchestrator --dry-run

# Resume the full sweep on physical GPU 0.
uv run python -m pipeline.stg_02_extract.orchestrator --device cuda:0

# Run the five technology-provision proxies (50 jobs across both methods).
uv run python -m pipeline.stg_02_extract.orchestrator --target min_5 --device cuda:0

# Run that subset for one model (10 jobs across both methods).
uv run python -m pipeline.stg_02_extract.orchestrator --target min_5 --model-name thomsonreuters/Thomson-1.0-Small --device cuda:0

# Run all three methods for one model, loading it once (15 clause jobs).
UV_PROJECT_ENVIRONMENT=/VData/scro4406/cba_analysis/.venv \
uv run python -m pipeline.stg_02_extract.orchestrator \
  --method all --target min_5 --model-name Qwen/Qwen3.8-27B-FP8 --device cuda:0

# Run all three methods through OpenRouter for an explicitly selected model.
UV_PROJECT_ENVIRONMENT=/VData/scro4406/cba_analysis/.venv \
uv run python -m pipeline.stg_02_extract.orchestrator \
  --endpoint openrouter --method all --target min_5 --model-name openai/gpt-4.1

# Run sentence-ID extraction on all 102 test documents for each min_5 clause.
UV_PROJECT_ENVIRONMENT=/VData/scro4406/cba_analysis/.venv \
uv run python -m pipeline.stg_02_extract.orchestrator \
  --method runner2 --target min_5 --device cuda:0

# Rerun only runner.py (Harness) for these models across all 41 clause types.
uv run python -m pipeline.stg_02_extract.orchestrator \
  --method runner --force --device 0 \
  --model-name Qwen/Qwen3.8-27B \
  --model-name Qwen/Qwen3.8-27B-FP8 \
  --model-name RedHatAI/gemma-4-31B-it-FP8-dynamic \
  --model-name google/gemma-3-12b-it \
  --model-name thomsonreuters/Thomson-1.0-Small
```

`--target min_5` selects `change_of_control`, `audit_rights`,
`post_termination_services`, `rofr_rofo_rofn`, and `anti_assignment`.
It retains all 102 test documents and, by default, all five models. Use
`--model-name MODEL` to select one preset or custom model; repeat the option to
select several. Without this option, the five preset models run.
The endpoint defaults to `vllm`, including when models are explicitly selected.
Use `--endpoint openrouter` to send requests to OpenRouter; this requires at
least one explicit `--model-name` and `OPENROUTER_API_KEY` in the environment or
project `.env`. Repeat `--model-name` to run several hosted models sequentially.
OpenRouter mode works with every `--method` selection, starts no local model
server, and skips Hugging Face generation-config loading. GPU/server flags are
ignored; each extraction script retains its OpenRouter request settings.
`--dry-run` requires no API key and makes no API requests.
Add `--dry-run` to preview the schedule. The default `--target all` selects all
41 clause types.

`--method runner` runs only the standard extraction runner (reported as Harness),
leaving ContractEval outputs untouched. `--method runner2` runs only sentence-ID
extraction (reported as Runner2), using the same shared server and test inputs.
With `--target min_5`, it schedules 25 clause jobs across the five preset models,
or five jobs with a single `--model-name`. `--method contracteval` runs only
ContractEval; the default `--method both` runs ContractEval and Harness.
`--method all` runs ContractEval, Harness, and Runner2 in that order for each
clause, sharing one server per model. It schedules 75 jobs for `min_5` across
the five preset models, or 15 for one model (615 for all 41 clauses and models).
`--force` reruns
existing outputs only for the selected methods. Each selection shares one
vLLM server per model across all selected clauses.

The default model order is `Qwen/Qwen3.8-27B`,
`RedHatAI/gemma-4-31B-it-FP8-dynamic`, `thomsonreuters/Thomson-1.0-Small`,
`Qwen/Qwen3.8-27B-FP8`, and `google/gemma-3-12b-it`. Identifiers are passed
through exactly; explicit `--model-name` selections run in the supplied order.
For each model, clauses run alphabetically, with ContractEval then Harness when
both are selected, followed by Runner2 with `--method all`, before the server
is closed and the next model loads. Document
concurrency is one; Harness retains its configured internal workers and passes.

The shared server uses neutral vLLM generation defaults. Harness receives the
model's generation defaults and its existing thinking-template settings per
request; ContractEval keeps temperature 0, a 5,000-token output budget, and
model-default thinking. Neither method changes the shared server's defaults.

Defaults are GPU 0, one GPU, port 8123, context length 131072, and the standard
OCR input/model directories. Override `--input-root`, `--ocr-model-name`,
`--port`, `--max-model-len`, `--gpu-memory-utilization`, or `--max-num-seqs`
as needed. The orchestrator never truncates ContractEval inputs, automatically
shrinks context, or adds GPUs to fit a model.

`--output-root` is the stage-02 root: existing Harness and ContractEval layouts
are preserved beneath it. For `--method runner2`, the default root is instead
`CACHE_DIR/stg_02_extract_runner2`, including its orchestration report; an explicit
`--output-root` overrides this location. Runner2 preserves its manifest validation,
5,000-token response budget, and disabled thinking on the shared server.
With `--method all`, default runs use each method's existing cache location:
ContractEval and Harness under `CACHE_DIR/stg_02_extract`, and Runner2 under
`CACHE_DIR/stg_02_extract_runner2`. When an explicit `--output-root ROOT` is
provided with `--method all`, Runner2 uses `ROOT/runner2` while the other methods
and the orchestration report use their existing layouts under `ROOT`.
Compatible cached results are reused, failed or
missing extractions are retried, and cache mismatches remain errors. Use
`--force` to regenerate results and `--no-progress` to suppress progress bars.
Missing or unreadable test inputs stop preflight before any server starts.

Progress and errors are saved atomically after every clause job to
`<output-root>/cuad/orchestration/latest.json`. Clause failures allow subsequent
jobs to continue; startup failures or a dead server skip the remaining jobs for
that model. Servers are never restarted within a sweep. Failed ContractEval
requests still score as empty extractions; infrastructure failures are recorded
separately. The command exits nonzero if anything fails, and Ctrl+C or SIGTERM
closes the current server and stops the sweep. A server cleanup failure stops
the sweep to avoid loading another model before GPU release is confirmed.
Generate comparison tables separately after extraction completes.

## ContractEval CUAD baseline

Run one clause and model with the full-document method from
[ContractEval](https://arxiv.org/html/2508.03080v1):

```bash
uv run python -m pipeline.stg_02_extract.contracteval \
  --provision governing_law --model-name openai/gpt-4.1 --endpoint openrouter

uv run python -m pipeline.stg_02_extract.contracteval \
  --provision governing_law --model-name Qwen/Qwen3-8B --endpoint vllm
```

OpenRouter uses `OPENROUTER_API_KEY` from the environment or project `.env`.
The vLLM command starts and closes its server; GPU selection and serving options
follow the extraction runner. The default context window is 131,072 tokens;
adjust `--max-model-len`, `--num-gpus`, `--device`, and
`--gpu-memory-utilization` for the model and available hardware. Context overflow
is reported as a failure; inputs are never truncated or chunked.

The script selects the 102 contracts in `references/cuad/test.json` from the
runner's stage-1 `cuad/<ocr-model>/<document>/full.txt` inputs. It uses the local
CUAD YAML description inside the original CUAD question wrapper and the
reference system prompt and fenced Context/Question template. Each document and
clause receives one plain-text request. Original CUAD answer strings remain the
labels, so OCR differences affect exact containment scores. The local test file
contains 4,182 question pairs across all 41 categories; the paper reports 4,128.

Generation defaults are temperature 0 and a 5,000-token output budget. The budget
is adapted from the reference open-model implementation and shared across both
endpoints; the proprietary reference used a larger budget. `top_p` is 0.9 for
OpenRouter, as in the proprietary reference, and 1.0 for local greedy decoding.
Use `--max-tokens`, `--temperature` (or `--temperature default` to omit it), and
`--thinking default|on|off` to change these settings. Thinking defaults to the
model's behavior. Gemma 4 and Qwen 3 reasoning parsers are selected automatically;
use `--reasoning-parser` for other local reasoning models. Endpoint-separated
reasoning is stored separately and final-answer content is scored. A provider
may reject or ignore unsupported generation controls; requested settings and
returned model metadata are retained for inspection.

Results are written atomically to:

```text
CACHE_DIR/stg_02_extract/cuad/contracteval/<safe_model>/<document_id>/<clause_type>.json
CACHE_DIR/stg_02_extract/cuad/contracteval/<safe_model>/metrics/<clause_type>.json
```

Each result retains the raw answer, labels, scores, usage, finish reason, and
generation settings. Matching cached results are reused; changed inputs, prompts,
labels, or settings require `--force` or a separate `--output-root`. That option
replaces the ContractEval root (model/document directories are appended).
Use `--document-id`, `--sample N --seed S`, and `--concurrency` as needed.

Evaluation follows the [reference evaluator](https://github.com/olivialiu121/ContractEval/blob/main/Evaluation.py):
all gold strings must be contained in the answer to count a positive example as
TP. It reports TP/TN/FP/FN, precision, recall, F1, F2, positive-case token-set
Jaccard, and false abstentions divided by the selected positive-example count.
It preserves upstream edge stripping, case-insensitive substring detection of
“no related clause”, and Jaccard punctuation removal and literal-space splitting.
Undefined precision/recall/F scores are zero; positive-only metrics are JSON
`null` when there are no positive examples. Truncated answers are flagged and
scored. Failed requests and missing answers count as empty extractions: FN on
positive examples, TN on negative examples, zero positive-case Jaccard, and a
false abstention on positive examples. Errors remain reported separately.
Failure records retain their status and error (without inventing answer text),
are included in the comparison table, and are retried on subsequent inference
runs. Offline evaluation also counts missing cached outputs as empty extractions,
including failures from older runs that did not save per-document records.
The comparison table also reads failures and missing-output IDs from saved
aggregate metrics when no per-document record exists; actual output files take
precedence. Its coverage section shows the number of failed or missing cases.
Documents without OCR inputs remain listed separately as missing inputs.

Rescore saved answers and compare an existing runner model on the common set of
documents without starting inference:

```bash
uv run python -m pipeline.stg_02_extract.contracteval \
  --provision governing_law --model-name openai/gpt-4.1 \
  --evaluate-only --compare-runner-model RedHatAI/gemma-4-31B-it-FP8-dynamic
```

Offline rescoring checks input/prompt/label identity but uses saved generation
metadata, so inference flags do not need repeating. Comparison joins the runner's
stored `extraction_text` values in file order. An existing empty JSONL file is an
abstention; missing files are excluded and listed separately. Rows naming a
different OCR model are excluded with an error (empty files contain no provenance
to verify). `--runner-output-root` overrides the stage-2 root containing
`cuad/<model>/...`. Metrics include document IDs, missing coverage, failures, and
truncation counts. The command exits nonzero for missing selected inputs, missing
offline outputs, request failures, or invalid comparison data. Incomplete runner
coverage is reported without failing the run.

### Model comparison table

Generate a standalone HTML table with grouped **ContractEval** and **Harness**
columns (Precision, Recall, F1, true-positive Jaccard, and Corpus retained), one row per eligible
model, plus a final **N Types** column:

```bash
uv run python -m pipeline.stg_02_extract.comparison_table
```

Use the same five-clause subset as the orchestrator with:

```bash
uv run python -m pipeline.stg_02_extract.comparison_table --target min_5
```

The default `--target all` includes all eligible types. With `--target min_5`,
**N Types** counts eligible types only within that subset (at most five).
The complete paired test-coverage requirement still applies. An optional
`--clause-type` must belong to the selected target.

The default output is `figures/cuad/comparison_table.html`, with a companion
`comparison_table.spans.json` containing grounded offsets, gold spans, matched
pairs, unmatched passages, and per-case TP/FP/FN counts. The script reads
`CACHE_DIR/stg_02_extract/cuad` and uses `answer_start` plus annotated answer
lengths from `references/cuad/CUADv1.json`, ignoring saved scores and labels.
The HTML contains only the main comparison table. Each model-methodology pair
has an Examples dropdown with one TP, FP, TN, and FN drawn from its evaluated
cases; unavailable outcome types are identified in the dropdown.

Both methods use greedy one-to-one span matching requiring full containment:
**100% of a gold span's characters must be included in one extraction**.
Missing any part of the gold span disqualifies the match.
Additional surrounding text does not disqualify a match, though
it still reduces the separately reported Jaccard similarity. Each prediction
and gold span participates in at most one match; separate extractions are not
combined to meet the threshold. Extra grounded predictions, including duplicates,
are false positives; ungrounded passages are excluded from extraction metrics
and retained only in diagnostics. Unmatched gold spans are false
negatives. Precision and Recall use pooled span counts across each model's
included types: `TP / (TP + FP)` and `TP / (TP + FN)`. Undefined Precision
and Recall are reported as zero.

**Corpus retained** is the percentage of characters in the evaluated CUAD
documents covered by extracted spans. Each document is counted once, and
overlapping or duplicate spans across included clause types are merged before
counting extracted characters. All grounded extractions count, including false
positives; ungrounded text is excluded. The JSON report includes
`extracted_characters`, `corpus_characters`, and the fraction `corpus_retained`.
An empty corpus produces an undefined fraction, displayed as an em dash.

Harness uses recorded character offsets, checked against the original CUAD
context. ContractEval's free-text output is grounded without consulting gold
labels. Alignment first tries whole paragraphs, then lines and sentences,
allowing whitespace differences and common Markdown/quotation wrappers.
Adjacent fragments within a paragraph merge only across whitespace in the
source, with no unmatched output between them; separate paragraphs/list items
stay separate. Altered or unlocatable passages remain in the ungrounded diagnostics.
Repeated text maps to the next unused occurrence in source order, then the first
unused occurrence; if all occurrences are used, the duplicate prediction is
retained. Ambiguous matches and unmatched passages are recorded in the diagnostics.
This deterministic alignment cannot infer which occurrence a model intended.

Only model–clause pairs covering all 102 contracts in **both ContractEval and
Harness for that same model** are included. Coverage requires the actual test
document IDs; extra training outputs cannot substitute for missing test
documents. Recorded failures count as covered and score as empty extractions.
Incomplete or unpaired types are excluded from both methods' aggregates.
Models with no eligible types are omitted. Evaluation uses only test
documents, even when a complete Harness run also contains training outputs.
By default, all eligible clause types are pooled. **N Types** shows how many
types contribute to a model's aggregate. Both methods within a row use identical
cases; different models can include different types. The companion JSON records
each row's included clause names, case counts, and coverage. The highest unrounded score for each
metric across **both** methodologies is bold, including exact ties.

**Jaccard (TP)** averages the Jaccard values of qualifying matched span pairs
across all included documents and types. Its denominator is the number of true
positives; neither false negatives nor false positives are included. It is
undefined (JSON `null`, shown as a dash) when there are no true positives.
`--jaccard-cases tp` remains accepted; positive-case averaging is no longer supported.
Use `--clause-type NAME` to restrict to one type, or `--model-name MODEL ...` to select
models, `--output PATH` to change the HTML destination, and `--input-root PATH`
or `--cuad-json PATH` to override inputs. No model calls occur. The separate `contracteval.py`
script retains its paper-reference document-level metrics.

## Defining an outcome taxonomy

Provision definitions live in `pipeline/provisions/*.yaml` and contain:

`--provision NAME` (`stg_02_extract/runner.py`) loads `pipeline/provisions/NAME.yaml` by default. When `--source` names a source with its own `pipeline/provisions/<source>/` subdirectory, it loads `pipeline/provisions/<source>/NAME.yaml` instead, so a source can bundle a clause taxonomy that has no bearing on any other source.

1. `clause_type`, the snake_case name of the clause class, which must match the
   filename stem;
2. `clause_description`, a short description of what the class covers; and
3. `ENRICH_MAX_CHAR_BUFFER`, the approximate character budget for Stage 3's
   sentence-bounded context window; and
4. optionally, a hierarchical `subtype_taxonomy` used by Stage 4.

The extraction prompt itself is not in the YAML. It is built around
`clause_type` and `clause_description` by `_prompt_description` in
`pipeline/stg_02_extract/structure_provision.py`; it asks only for matching text.

The included specifications are:

- `technology.yaml`: technology and automation clauses, with two taxonomy
  levels. The first level is `preemptive_rights`, `implementation`, or
  `workforce_management`.
- `wage_table.yaml`: wage-table extraction only; it has no classification
  taxonomy.
- `cuad/*.yaml`: one flat config per CUAD clause category (41 total, e.g.
  `cuad/governing_law.yaml`, `cuad/non_compete.yaml`), sourced from the CUAD
  dataset's own category descriptions. Selected automatically with `--source
  cuad --provision <category_name>`. None declare a `subtype_taxonomy`.

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
pipeline/stg_03_enrich/       context and beneficiary enrichment
pipeline/stg_04_classify/     hierarchical taxonomy classification
pipeline/provisions/          measurement definitions and taxonomies
pipeline/utils/               metadata linking, plots, paths, model ranking
reviewer/                     browser-based human review tool
meta_data/                    CBA metadata used by the analysis join
cache/                        local inputs, cached artifacts, review data
figures/                      generated analysis tables and plots (gitignored)
tests/                        unit and integration tests with mocked inference
```

`parallel_runs.bash` launches OCR for the three configured sources across
three GPUs. `pipeline/utils/move_extractions.py` copies selected Stage 2,
Stage 3 enrichment, and Stage 4 classification outputs from an external
`CACHE_DIR` into the repo-local cache for review; `--no-enrich` and
`--no-classify` control the downstream copies independently, and OCR text is
never copied. `main.py` and
`references/provision_taxonomy.json` are legacy placeholders and are not used
by the active pipeline.
