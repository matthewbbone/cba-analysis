# CBA Extraction Reviewer

A lightweight TypeScript (Vite) UI with two review modes:

- **Extraction review** compares langextract wage-table extractions to the
  original source PDFs and grounded OCR spans.
- **OCR comparison** samples a random PDF page from `cornell_dol`,
  `cornell_retail_educ`, or `dol_archive` that has text from at least two OCR
  models and asks whether OCR A is better, OCR B is better, both are good and
  tied, or both are bad.

- **Left pane** — the original PDF (native browser viewer).
- **Right pane** — the langextract-style reviewer: the OCR `full.txt` with every
  extraction span highlighted inline, color-coded by grounding status, plus a
  detail card comparing the model's extracted text against the grounded source
  span it was aligned to.

## Data it reads

It reads directly from the repo `cache/` (nothing is copied):

| What            | Path                                                                  |
| --------------- | --------------------------------------------------------------------- |
| Source PDFs     | `cache/<source>/<document_id>.pdf`                                    |
| OCR text        | `cache/stg_01_ocr/<source>/<model>/<document_id>/full.txt`           |
| Extractions     | `cache/stg_02_extract/<source>/<model>/<document_id>/*.jsonl`        |
| OCR judgments   | `cache/reviewer/ocr_comparisons.jsonl`                             |

`<source>` is one of `cornell_dol`, `cornell_retail_educ`, `dol_archive`.
The document list is driven by every `*.jsonl` found under `stg_02_extract`.

Spans are stored against the **think-stripped** OCR text (matching
`pipeline/stg_02_extract/runner.py`), so the server strips `<think>…</think>`
blocks before serving the text to keep highlights aligned.

## Run

```bash
cd reviewer
npm install
npm run dev
```

Opens at http://localhost:5178.

## Controls

- Pick a document from the top dropdown.
- `◀` / `▶` (or `←` / `→`) step through extractions; the highlight scrolls into
  view and the detail card updates.
- Click any highlight in the text to jump to that extraction.
- Drag the divider between panes to resize.
- In **OCR comparison**, use the four result buttons (or keys `1`–`4`). Each
  result is appended to `cache/reviewer/ocr_comparisons.jsonl`, and the next
  weighted-random, not-yet-reviewed model pair is loaded automatically.
- Comparisons are blinded in the UI as **OCR A** and **OCR B**; model identifiers
  are retained only in the saved judgment record for later analysis.
- Sources are randomized before pages are sampled, giving each collection an
  equal opportunity to appear even when their page counts differ substantially.
- Within a source, model matchups containing models with fewer saved comparisons
  receive more weight. Every eligible matchup retains a positive probability,
  and the page is chosen randomly only after the model matchup is selected.
- Model pairs are limited to Qwen 3.6 27B, Ovis 2.6 30B, and Gemma 4 31B so
  reviews focus on identifying the best of the current leading OCR models.
- The **Minimum text difference** filter uses whitespace-normalized Levenshtein
  distance divided by the longer OCR length. The default 5% cutoff therefore
  behaves consistently for both short and long pages.

## Rank OCR models

From the repository root, fit a regularized Bradley-Terry-style ranking from
the saved judgments:

```bash
python pipeline/utils/rank_ocr_models.py
```

The script prints a ranking and writes
`cache/reviewer/ocr_model_rankings.json`. Decisive reviews are ordinary
pairwise wins. A good or bad tie both pulls the two model strengths together
and moves both models above or below a shared neutral-quality baseline, so the
two kinds of ties do not collapse to the same outcome. Use
`--tie-quality-weight` to adjust the balance between those two signals.

## Legend

Highlight color = langextract `alignment_status` (grounding):

- **Exact match** — span matches the extracted text exactly.
- **Partial** — grounded span is a subset of the extracted text.
- **Span >** — grounded span is larger than the extracted text.
- **Fuzzy / No grounding** — approximate or missing alignment (worth reviewing).
