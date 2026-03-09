# Method: llm_segment

## Description
Two-stage LLM segmentation for CBA structure discovery and extraction.

1. Planning pass:
- Read first 10-25 pages to infer hierarchy style and header patterns.
- Produce parsing guidance for the extraction pass.

2. Extraction pass:
- Process full document in overlapping chunks.
- For each chunk, extract subsection segments with:
  - `parent`: section/article header
  - `title`: subsection header/title
  - `text`: subsection body text
  - character spans in chunk/global text for dedupe and overlap metrics

## Notes
- This is a baseline implementation intended for rapid iteration.
- Uses OpenRouter via `openai` client:
  - `OPENROUTER_API_KEY` required
  - `OPENROUTER_BASE_URL` optional (defaults to `https://openrouter.ai/api/v1`)
- Current experiment variants target:
  - `openai/gpt-5-mini`
  - `google/gemini-2.5-flash`
  - `anthropic/claude-haiku-4.5`
