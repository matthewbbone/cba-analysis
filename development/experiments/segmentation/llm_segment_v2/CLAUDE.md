# Method: llm_segment_v2

## Description
Boundary-first segmentation for CBA hierarchy extraction.

1. Planning pass:
- Read the first 10-25 pages.
- Infer top/subsection hierarchy conventions and heading patterns.

2. Candidate generation:
- Build deterministic heading/boundary candidates from regex and line heuristics.

3. Verification pass:
- Use LLM in candidate batches to verify boundaries and normalize level/title.

4. Deterministic span assembly:
- Build subsection spans from verified boundaries.
- Preserve required schema fields and attach optional debug metadata.

## Output compatibility
- Required segment fields remain compatible with `llm_segment`:
  - `parent`, `title`, `text`, `start_pos`, `end_pos`, `start_page`, `end_page`
- Optional debug fields:
  - `boundary_source`
  - `boundary_confidence`
  - `heading_text`
  - `heading_level`

## Notes
- Uses OpenRouter via `openai` client.
- `OPENROUTER_API_KEY` required.
- `OPENROUTER_BASE_URL` optional (defaults to `https://openrouter.ai/api/v1`).

