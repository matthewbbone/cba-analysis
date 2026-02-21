# OlmoOCR 2 Method

## Description
Uses OlmoOCR 2 (7B parameter model) running locally via LM Studio. Renders PDF pages to images and sends them to the model with structured prompts from the `olmocr` Python package.

## Dependencies
- `olmocr` package (installed via `olmocr[bench]`)
- `openai` Python client
- Local LM Studio server running at `http://localhost:1234/v1`

## Usage
```bash
python method.py <pdf_path>
```

## Notes
- Requires LM Studio to be running with the `allenai_olmOCR-2-7B-1025-GGUF` model loaded.
- Adapted from `clause_extraction/ocr/runner.py`.
