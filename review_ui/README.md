# Review UI

Simple Streamlit dashboard to review PDF pages and list features.

## Run

```bash
streamlit run review_ui/app.py
```

## Notes
- Expects documents named `document_*.pdf` in `../cbas` by default.
- Use the sidebar to pick document and page.
- Requires `streamlit[pdf]` (installs `streamlit-pdf`) for `st.pdf`.
