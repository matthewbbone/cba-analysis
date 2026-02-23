# Langextract-Style Method (OpenRouter)

## Approach
Implements a lightweight, langextract-style extractor without the `langextract` package. It uses OpenRouter chat completions with strict JSON output and long-document chunking.

## How It Works
1. **Parse Taxonomy:** Load clause names + TLDRs from `references/feature_taxonomy_final.md`.
2. **Build Label Prompt:** Constrain extraction to allowed taxonomy labels.
3. **Combine Pages:** Concatenate selected pages into one text stream and track page char offsets.
4. **Pass 1 (Coarse):** Split long text with separator-aware chunking and overlap, then extract.
5. **Pass 2 (Focused):** Build candidate windows around pass-1 spans and re-extract on smaller chunks.
6. **Normalize + Locate:** Normalize clause labels and align extraction text to source spans.
7. **Deduplicate + Map:** Deduplicate overlap hits, then map global spans back to page-local offsets.

## Tunable Parameters
- `extraction_passes` (default `2`): set to `1` for single-pass behavior
- `max_char_buffer`: pass-1 chunk size
- `overlap_fraction`: overlap for both passes
- `context_chars`: left/right context window for each chunk
- `refinement_margin_chars`: extra context around pass-1 hits for pass 2

## Backend
Uses the OpenAI-compatible OpenRouter endpoint (`https://openrouter.ai/api/v1`) via the `openai` Python SDK.

## Dependencies
- `openai` Python package
- `OPENROUTER_API_KEY` env var
