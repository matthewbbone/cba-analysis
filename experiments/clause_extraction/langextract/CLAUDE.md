# Langextract-Style Method (OpenRouter)

## Approach
Implements a lightweight, langextract-style extractor without the `langextract` package. It uses OpenRouter chat completions with strict JSON output and long-document chunking.

## How It Works
1. **Parse Taxonomy:** Load clause names + TLDRs from `references/feature_taxonomy_final.md`.
2. **Build Label Prompt:** Constrain extraction to allowed taxonomy labels.
3. **Combine Pages:** Concatenate selected pages into one text stream and track page char offsets.
4. **Chunk + Overlap:** Split long text with separator-aware chunking and overlap.
5. **Chunk Extraction:** Call OpenRouter on each chunk with bounded left/right context.
6. **Normalize + Locate:** Normalize clause labels and align extraction text to source spans.
7. **Deduplicate + Map:** Deduplicate overlap hits, then map global spans back to page-local offsets.

## Backend
Uses the OpenAI-compatible OpenRouter endpoint (`https://openrouter.ai/api/v1`) via the `openai` Python SDK.

## Dependencies
- `openai` Python package
- `OPENROUTER_API_KEY` env var
