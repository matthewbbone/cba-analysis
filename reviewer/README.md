# CBA Extraction Reviewer

A lightweight TypeScript (Vite) UI with two pair-wise review modes:

- **Extraction comparison** samples a random document that two *extraction*
  models both processed for the selected provision type, and asks whether
  Extraction A is better, Extraction B is better, both are good and tied, or
  both are bad.
- **OCR comparison** samples a random PDF page from `cornell_dol`,
  `cornell_retail_educ`, or `dol_archive` that has text from at least two OCR
  models and asks the same four-way question about the transcriptions.

In **Extraction comparison**:

- **Left pane** — the original PDF (native browser viewer).
- **Right pane** — one shared OCR `full.txt` with *both* models' spans
  highlighted inline, color-coded by which model produced them, above a detail
  card that puts the two models side by side: extracted text, the extraction's
  `attributes.context`, its grounding status, and its character offsets.

Both models' spans are overlaid on a single copy of `full.txt` because the pair
is only sampled when both models recorded the same `ocr_model_name`, so their
offsets index the same text.

## Data it reads

It reads directly from the repo `cache/` (nothing is copied):

| What            | Path                                                                  |
| --------------- | --------------------------------------------------------------------- |
| Source PDFs            | `cache/<source>/<document_id>.pdf`                                        |
| OCR text               | `cache/stg_01_ocr/<source>/<ocr_model>/<document_id>/full.txt`            |
| OCR pages              | `cache/stg_01_ocr/<source>/<ocr_model>/<document_id>/page_N.txt`          |
| Extractions            | `cache/stg_02_extract/<source>/<model>/<document_id>/<provision>.jsonl`   |
| OCR judgments          | `cache/reviewer/ocr_comparisons.jsonl`                                    |
| Extraction judgments   | `cache/reviewer/extraction_comparisons.jsonl`                             |

`<source>` is one of `cornell_dol`, `cornell_retail_educ`, `dol_archive`. The
provision dropdown is driven by every distinct `<provision>.jsonl` basename
found under `stg_02_extract`.

Extraction spans are offsets into the **raw** `full.txt`, matching
`pipeline/stg_02_extract/runner.py`, which reads the file verbatim. `<think>…
</think>` blocks are already removed upstream in `stg_01_ocr`, so `full.txt` is
served unmodified. (The per-page OCR text used by the OCR comparison tab is
still think-stripped defensively.)

Set `EXTRACTION_COMPARISON_FILE` or `OCR_COMPARISON_FILE` to write judgments
somewhere other than `cache/reviewer/`.

## Run

```bash
cd reviewer
npm install
npm run dev
```

Opens at http://localhost:5178.

## Controls

- Pick a provision type from the top dropdown; changing it samples a new pair.
- `◀` / `▶` (or `←` / `→`) step through *both* models' extractions merged into
  document order; the highlight scrolls into view and the detail card updates.
- Click any highlight in the text to jump to that extraction.
- Drag the divider between panes to resize.
- In **Extraction comparison**, use the four result buttons (or keys `1`–`4`).
  Each result is appended to `cache/reviewer/extraction_comparisons.jsonl` and
  the next weighted-random, not-yet-reviewed model pair is loaded automatically.
  Comparisons are blinded as **Extraction A** and **Extraction B**.
- A pair is only offered when both models recorded the same `ocr_model_name`,
  and at least one of them extracted something. A model that found nothing while
  the other found several is a legitimate — and informative — comparison.
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
- Model pairs are limited to Qwen 3.6 27B, Ovis 2.6 30B, OvisOCR2, and Gemma 4
  31B so reviews focus on identifying the best of the current leading OCR
  models. A model becomes eligible page-by-page as its `page_N.txt` outputs are
  added to the stage-01 cache.
- The **Minimum text difference** filter uses whitespace-normalized Levenshtein
  distance divided by the longer OCR length. The default 5% cutoff therefore
  behaves consistently for both short and long pages.

## Rank models

From the repository root, fit a regularized Bradley-Terry-style ranking from
the saved judgments:

```bash
python pipeline/utils/rank_ocr_models.py
python pipeline/utils/rank_ocr_models.py --top-ocr-models
python pipeline/utils/rank_ocr_models.py --stage extraction
```

`--stage` selects which review set to fit. Both stages save the same
`left_model`/`right_model`/`choice` record shape, so they share one model:

| `--stage` | Reads | Writes |
| --- | --- | --- |
| `ocr` (default) | `cache/reviewer/ocr_comparisons.jsonl` | `cache/reviewer/ocr_model_rankings.json` |
| `extraction` | `cache/reviewer/extraction_comparisons.jsonl` | `cache/reviewer/extraction_model_rankings.json` |

`--input` and `--output` override the selected stage's paths.

The script prints a ranking and writes the JSON above. Decisive reviews are
ordinary pairwise wins. A good or bad tie both pulls the two model strengths
together and moves both models above or below a shared neutral-quality
baseline, so the two kinds of ties do not collapse to the same outcome. Use
`--tie-quality-weight` to adjust the balance between those two signals.
`--top-ocr-models` refits the statistics using only head-to-head reviews among
Qwen 3.6 27B, Ovis 2.6 30B, OvisOCR2, and Gemma 4 31B; comparisons involving
other models are excluded entirely. Every model in the cohort must have at
least one saved comparison. That cohort is defined for `--stage ocr` only, so
the flag is rejected for `--stage extraction` — narrow that stage with
`--input` instead. The former `--top-3-general-vlms` and `--general-vlms-only`
spellings remain accepted as compatibility aliases for this current
four-model cohort.

## Legend

In **Extraction comparison**, highlight color = which model claimed the span:

- **Extraction A only** (blue) — only the left model extracted this text.
- **Both models** (purple) — the two models' spans overlap here.
- **Extraction B only** (orange) — only the right model extracted this text.

Overlapping spans are segmented on every boundary, so where one model extracted
a longer passage than the other, the shared part renders purple and the excess
renders in that model's own color.

The detail card additionally badges each extraction with its langextract
`alignment_status` (grounding):

- **Exact match** — span matches the extracted text exactly.
- **Partial** — grounded span is a subset of the extracted text.
- **Span >** — grounded span is larger than the extracted text.
- **Fuzzy / No grounding** — approximate or missing alignment (worth reviewing).

A **Span unreliable** badge means `span_reliable` is false: the pipeline could
not re-anchor the full extracted text in the source, so the highlight may cover
only a prefix of what the model actually returned.
