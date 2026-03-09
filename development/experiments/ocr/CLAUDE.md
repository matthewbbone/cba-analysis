# OCR Experiment

## Goal
Compare OCR tools on document accuracy for CBA (Collective Bargaining Agreement) PDFs.

## Metrics
- **Character Error Rate (CER):** Proportion of characters that differ between OCR output and ground truth.
- **Word Error Rate (WER):** Proportion of words that differ between OCR output and ground truth.

## Methods

### 1. olmocr
- **Description:** OlmoOCR 2 (7B) running locally via LM Studio. Uses the `olmocr` Python package to build page queries with rendered PDF images and structured prompts.
- **Implementation:** Adapted from `clause_extraction/ocr/runner.py`.

### 2. vision_model
- **Description:** Pass PDF pages rendered as PNG directly to a vision-capable LLM via OpenRouter API. The model extracts text from the image with a simple prompt.
- **Implementation:** Renders PDF pages to PNG, sends to OpenRouter with a text extraction prompt.

### 3. pdftotext
- **Description:** Use `pdftotext` from poppler-utils to extract embedded text directly from PDF files. No OCR involved — relies on the PDF's embedded text layer.
- **Implementation:** Calls `pdftotext` CLI tool via subprocess.

## Folder Structure
```
ocr/
├── CLAUDE.md              # This file
├── results.csv            # Aggregated experiment results
├── run_experiment.py      # Runner script for all methods
├── analyze_results.py     # Analysis and reporting script
├── olmocr/
│   └── method.py          # OlmoOCR 2 implementation
├── vision_model/
│   └── method.py          # Vision model via OpenRouter
└── pdftotext/
    └── method.py          # poppler-utils pdftotext
```
