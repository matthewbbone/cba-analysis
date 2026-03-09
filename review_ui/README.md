# Review UI

The collaborator-facing review dashboard is `review_ui/app3.py`. It is the broadest UI in the repo and can inspect segmentation, classification, and generosity outputs.

## Run

```bash
uv run streamlit run review_ui/app3.py
```

## Notes

- The UI expects the main pipeline outputs under `CACHE_DIR`.
- `review_ui/app.py` is an older, narrower dashboard and should be treated as legacy/internal.
- The dashboard also reads checked-in `outputs/` and `figures/` artifacts where relevant.
- `streamlit[pdf]` is required for PDF preview support.
