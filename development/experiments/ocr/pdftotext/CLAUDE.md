# pdftotext Method

## Description
Uses `pdftotext` from poppler-utils to extract embedded text directly from PDF files. No OCR involved — relies on the PDF's existing text layer.

## Dependencies
- `pdftotext` CLI tool (from poppler-utils, installed via Homebrew)

## Usage
```bash
python method.py <pdf_path>
```

## Notes
- Uses `-layout` flag to preserve spatial layout.
- Will produce empty/poor results for scanned PDFs without a text layer.
