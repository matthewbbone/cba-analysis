# LangExtract Method

## Approach
Uses Google's [langextract](https://github.com/google/langextract) library for structured information extraction with few-shot examples. Adapted from the existing `clause_extraction/extraction/runner.py`.

## How It Works
1. **Parse Taxonomy:** Load the 44-clause taxonomy from `references/feature_taxonomy_final.md`.
2. **Build Prompt:** Create a `prompt_description` listing all allowed clause labels with TLDRs.
3. **Few-Shot Examples:** Provide `lx.data.ExampleData` / `lx.data.Extraction` examples to guide the model.
4. **Extract:** For each page, call `lx.extract()` which handles chunking, structured output, and source grounding.
5. **Normalize:** Map raw model outputs to canonical clause names using fuzzy matching.
6. **Locate Spans:** Use langextract's `char_interval` for span positions, with fallback text search.

## Backend
Uses langextract with Gemini 2.5 Flash by default. The library handles the model interaction, structured output parsing, and source text alignment.

## Dependencies
- `langextract` — Google's structured extraction library
- `GOOGLE_API_KEY` env var (for Gemini models)
