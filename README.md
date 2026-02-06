# CBA Analysis Toolkit

This repo contains two main components:

1. **Extraction script** (`scripts/extract_cba_features.py`) that labels which contract features appear on each PDF page.
2. **Review UI** (`review_ui/app.py`) for validating extracted features and exploring summary statistics.

## 1) Extraction Script

**Purpose**
- Reads PDFs named `document_*.pdf` from a folder.
- Uses `gpt-5-nano` to detect which CBA provisions appear on each page.
- Writes a consolidated CSV at `outputs/cba_features.csv` with one row per **document-page-feature**.

**Outputs**
- `outputs/cba_features.csv`: columns `document_id`, `document_page`, `feature_name`
- `outputs/processing_cache.json`: tracks processed pages for resumable runs

**Key behaviors**
- **Sampling**: `--sample-size N` processes a random sample.
- **Resume**: skips pages already processed and continues from the last page.
- **Partial priority**: when sampling, prioritizes docs with partially processed pages.
- **Filters**: only reads files matching `document_*.pdf`.

**Run**
```bash
uv run scripts/extract_cba_features.py --input-dir dol_archive --sample-size 5
```

**Common options**
- `--input-dir` directory of PDFs (default `cbas/`)
- `--sample-size` process only N documents
- `--max-pages` limit pages per document
- `--debug-dir` save raw responses for failed JSON parses

## 2) Review UI

**Purpose**
- Review extracted labels page-by-page.
- Add or remove detected features.
- Save reviewer feedback to `outputs/review.csv`.
- View provision frequency stats and heatmaps by metadata.

**Features**
- **Review**: PDF page on the left, detected features on the right.
- **Checklists**: confirm or remove each detected feature.
- **Add missing**: multi-select missing provisions.
- **Save feedback**: writes `outputs/review.csv`.
- **Stats**: page-level and document-level bar charts.
- **Heatmap**: distributions by `naics`, `type`, `statefips1`, `union`, `expire_year` using `dol_archive/CBAList_with_statefips.dta`.

**Run**
```bash
uv run streamlit run review_ui/app.py
```

**Password protection**
Create `review_ui/.streamlit/secrets.toml`:
```toml
REVIEW_UI_PASSWORD = "your-password-here"
```

## Requirements

```bash
uv pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the repo root with your OpenAI API key:

```env
OPENAI_API_KEY=your_key_here
```

## Notes
- The UI defaults to documents in `dol_archive/` and only lists documents found in `outputs/cba_features.csv`.
- NAICS is mapped to 2-digit sector names; state FIPS are mapped to state names.
