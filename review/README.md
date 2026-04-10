# OCR And Provisions Review App

Run the app from the repo root:

```bash
python -m pip install -r review/requirements.txt
streamlit run review/app.py
```

The app reads `CACHE_DIR` from the repo root `.env` file and includes:
- a page-by-page comparison of two OCR models against the original PDF
- the extracted provisions view from `02_provision_extract`
