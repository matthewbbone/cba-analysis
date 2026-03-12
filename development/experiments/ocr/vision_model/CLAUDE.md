# Vision Model Method

## Description
Renders PDF pages as PNG images and sends them to Google Gemini 2.5 Pro via OpenRouter API for text extraction.

## Dependencies
- `openai` Python client (for OpenRouter API)
- `pypdfium2` (for PDF-to-PNG rendering)
- `python-dotenv` (for loading API keys)
- `OPENROUTER_API_KEY` in `.env` file

## Usage
```bash
python method.py <pdf_path>
```

## Notes
- Uses `google/gemini-3-flash-preview` model via OpenRouter.
- Renders pages at 200 DPI.
