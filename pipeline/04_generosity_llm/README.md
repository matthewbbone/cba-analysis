# 04_generosity_llm

Rubric-based generosity scoring for CBA clauses using LLM-generated schemas, extracted structured data, calibrated rubrics, and document-level evaluation.

## Purpose

This stage assigns worker-favorability generosity scores on a 1-5 scale by:

1. Building clause-specific extraction schemas from sampled segments.
2. Extracting structured details from all segments in selected clause types.
3. Calibrating clause rubrics from observed extraction distributions.
4. Scoring each document and clause, then computing document composites.

All LLM calls are routed through OpenRouter.

## Input / Output

- Input classification segments (default):
  - `$CACHE_DIR/03_classification_output/dol_archive/document_*/segment_*.json`
  - or `outputs/03_classification_output/dol_archive/document_*/segment_*.json` when running without `CACHE_DIR`
- Output directory (default):
  - `$CACHE_DIR/04_generosity_llm_output/dol_archive`
  - or `outputs/04_generosity_llm_output/dol_archive` when running without `CACHE_DIR`

Artifacts written:

- `schemas/*.schema.json`
- `schemas/schema_index.json`
- `extractions/*.jsonl`
- `rubrics/*.rubric.json`
- `rubrics/*.distribution.json`
- `evaluations/*.jsonl`
- `document_clause_composite_scores.csv`
- `document_composite_scores.csv`
- `summary.json`

## Provider and Model

- Provider: `openrouter`
- Client: `openai.OpenAI` with OpenRouter `base_url`
- Required env var: `OPENROUTER_API_KEY`
- Optional env var: `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- Default model: `openai/gpt-5-mini`

If `OPENROUTER_API_KEY` is missing, the runner fails fast at startup.

## Clause Selection

The runner first loads segment rows with:

- `document_id`
- `segment_number`
- `segment_id`
- `clause_type` (from `labels[0]`, then fallback `label`, else `OTHER`)
- `segment_text`

Procedural clauses are excluded before selection/scoring:

- `Recognition Clause`
- `Recognition`
- `Parties to Agreement and Preamble`
- `Parties to Agreement`
- `Preamble`
- `Bargaining Unit`
- empty label

Top clause types are selected by **document prevalence** (number of distinct documents containing the clause), not raw segment frequency. Segment count is used only as a tie-breaker.

## Stage 1: Schema Creation

For each selected clause type:

1. Randomly sample up to `schema_sample_size` segments (default 10, seeded).
2. Ask the LLM for a clause-specific extraction schema with common fields.
3. Normalize field names/types.

Allowed field types:

- `number`
- `string`
- `boolean`
- `list[string]`
- `list[number]`

`notable_provisions: list[string]` is always appended if not present.

## Stage 2: Structured Extraction

For each selected clause type and each relevant segment:

1. Prompt with schema + segment text.
2. Return strict JSON values for schema fields.
3. Coerce values to schema type; ambiguous/missing values become `null`.
4. Store row status (`ok`/`error`) and error text.

Extraction runs in parallel using `asyncio` with bounded concurrency (`--max-concurrency`).

## Stage 3: Calibrated Rubric

For each clause type:

1. Compute empirical field distributions from extractions.
2. Include only rubric details with support in at least 2 non-null segment extractions.
3. Ask the LLM to produce 1-5 scoring anchors using observed variation:
   - score `3` near median
   - score `1` and `5` near observed extremes
4. Backfill/normalize incomplete rubric details.

Scoring direction policy is enforced:

- Score `5` = more worker-favorable.
- Score `1` = less worker-favorable / more worker-burdensome.
- Higher worker wages, benefits, and rights should increase scores.
- Higher worker costs, obligations, or restrictions should decrease scores.

For numeric anchors, direction checks are applied and anchors are flipped if needed to maintain this policy.

## Stage 4: Evaluation and Composite Scores

For each `(document_id, clause_type)`:

1. Aggregate extracted details across all segments in that document+clause.
2. Ask the LLM to score each rubric detail from 1-5 with a short reason.
3. Missing detail scores default to `3`.

Composite calculations:

- Clause composite score:
  - Mean of detail scores, excluding `notable_provisions`.
- Document composite score:
  - Mean of clause composite scores across scored clause types.

Evaluation also runs with `asyncio` parallelism.

## Caching and Re-run Behavior

Existing artifacts are reused unless `--force` is set:

- Schema files
- Extraction JSONL
- Rubric/distribution files
- Evaluation JSONL

`summary.json` includes provider/model metadata, selected clause types, counts, exclusions, and output paths.

## CLI

Run:

```bash
uv run python pipeline/04_generosity_llm/runner.py \
  --cache-dir "$CACHE_DIR" \
  --classification-dir 03_classification_output/dol_archive \
  --output-dir 04_generosity_llm_output/dol_archive \
  --model openai/gpt-5-mini
```

Key options:

- `--cache-dir`
- `--classification-dir`
- `--output-dir`
- `--model`
- `--temperature`
- `--max-tokens`
- `--max-retries`
- `--timeout`
- `--top-clause-types` (default 10)
- `--schema-sample-size` (default 10)
- `--document-id`
- `--sample-size`
- `--seed`
- `--max-segments`
- `--max-segments-per-clause`
- `--max-concurrency` (default 8)
- `--force`

## Notes

- This method is designed for relative generosity comparison across clauses/documents in the selected run scope.
- Top clause selection is computed after any run filters (`--document-id`, `--sample-size`, etc.).
- The collaborator-facing repo surface is `pipeline/` plus `review_ui/`; this README documents only the `04_generosity_llm` stage within that pipeline.
