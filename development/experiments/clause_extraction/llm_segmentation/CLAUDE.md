# LLM Segmentation Method

## Approach
Direct clause segmentation by feeding OCR text to an LLM via OpenRouter and asking it to identify clause boundaries and labels in a single pass.

## How It Works
1. **Parse Taxonomy:** Load the 44-clause taxonomy from `references/feature_taxonomy_final.md`.
2. **Build Prompt:** Create a system prompt that lists all clause types and instructs the model to:
   - Read the page text
   - Identify all clause boundaries
   - For each clause, return the label and the exact quoted text span
3. **Call LLM:** Send the page text to an LLM via OpenRouter API with JSON output format.
4. **Parse Response:** Extract clause labels and text spans from the JSON response.
5. **Locate Spans:** Find exact character positions of each extraction_text in the source page.

## Key Difference from LangExtract
This method does not use any structured extraction framework — it relies purely on the LLM's ability to segment and classify in a single prompt. This tests whether a simpler approach can match the structured extraction approach.

## Dependencies
- `httpx` or `openai` — for OpenRouter API calls
