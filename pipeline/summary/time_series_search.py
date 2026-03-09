"""Count topic mentions over time from OCR output text.

The script scans document-level OCR text for keyword families such as AI or
automation, joins publication years from DOL metadata, and writes time-series
tables plus figure artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()


TOPIC_CHOICES = ("automation", "ai")

KEYWORD_PATTERNS_BY_TOPIC: dict[str, list[tuple[str, str]]] = {
    "automation": [
        ("new_technology", r"\bnew\s+technolog(?:y|ies)\b|\bnew\s+technolgoy\b"),
        ("automation", r"\bautomation\b|\bautomated\b|\bautomating\b"),
        ("computerized", r"\bcomputeri[sz]ed\b|\bcomputeri[sz]ation\b"),
        ("mechanization", r"\bmechani[sz]ation\b"),
    ],
    "ai": [
        ("artificial_intelligence", r"\bartificial\s+intelligence\b"),
        ("ai", r"\ba\.?i\.?\b"),
        ("machine_learning", r"\bmachine\s+learning\b"),
        ("generative_ai", r"\bgenerative\s+(?:artificial\s+intelligence|ai)\b|\bgen(?:erative)?\s*ai\b"),
        ("large_language_model", r"\blarge\s+language\s+models?\b"),
        ("llm", r"\bllms?\b"),
        ("neural_network", r"\bneural\s+networks?\b"),
        ("deep_learning", r"\bdeep\s+learning\b"),
    ],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cache_dir() -> str:
    return os.environ.get("CACHE_DIR", "").strip()


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _default_ocr_dir() -> Path:
    """Resolve the OCR directory from `CACHE_DIR`; this script requires it."""
    root = _project_root()
    cache_dir = _default_cache_dir()
    if not cache_dir:
        raise RuntimeError("CACHE_DIR is not set. Set CACHE_DIR or pass --ocr-dir explicitly.")
    return (_resolve_path(cache_dir, root) / "01_ocr_output" / "dol_archive").resolve()


def _default_metadata_path() -> Path:
    return (_project_root() / "dol_archive" / "CBAList_fixed.dta").resolve()


def _default_output_dir() -> Path:
    cache_dir = _default_cache_dir()
    root = _project_root()
    if cache_dir:
        return _resolve_path(cache_dir, root) / "keyword_output" / "topic_mentions"
    return (root / "outputs" / "keyword_output" / "topic_mentions").resolve()


def _default_figure_dir() -> Path:
    return (_project_root() / "figures").resolve()


def _document_id_from_cbafile(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    m = re.search(r"(\d+)", text)
    if not m:
        return None
    return f"document_{int(m.group(1))}"


def _parse_year(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None

    try:
        import pandas as pd

        parsed = pd.to_datetime(value, errors="coerce")
        if parsed is not None and not pd.isna(parsed):
            year = int(parsed.year)
            return year if 1900 <= year <= 2100 else None
    except Exception:
        pass

    # Fallback: Stata daily date integer (days since 1960-01-01)
    if re.fullmatch(r"-?\d+", text):
        n = int(text)
        if 0 <= n <= 80000:
            try:
                import pandas as pd

                base = pd.Timestamp("1960-01-01")
                year = int((base + pd.to_timedelta(n, unit="D")).year)
                return year if 1900 <= year <= 2100 else None
            except Exception:
                return None
        if 1900 <= n <= 2100:
            return n
    return None


def _load_document_text(doc_dir: Path) -> str:
    full_text_path = doc_dir / "full_text.txt"
    if not full_text_path.exists() or not full_text_path.is_file():
        return ""
    return full_text_path.read_text(encoding="utf-8", errors="replace")


def _compile_alias_regexes(topic: str) -> list[tuple[str, re.Pattern[str]]]:
    patterns = KEYWORD_PATTERNS_BY_TOPIC.get(topic, [])
    return [
        (alias_name, re.compile(pattern, flags=re.IGNORECASE))
        for alias_name, pattern in patterns
    ]


def _count_mentions(text: str, alias_regexes: list[tuple[str, re.Pattern[str]]]) -> tuple[dict[str, int], int]:
    alias_counts: dict[str, int] = {}
    for alias_name, regex in alias_regexes:
        alias_counts[alias_name] = len(list(regex.finditer(text)))

    if not alias_regexes:
        return alias_counts, 0

    combined_pattern = "|".join(f"(?:{regex.pattern})" for _, regex in alias_regexes)
    combined_regex = re.compile(combined_pattern, flags=re.IGNORECASE)
    total_mentions = len(list(combined_regex.finditer(text)))
    return alias_counts, total_mentions


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (phat + (z * z) / (2.0 * float(n))) / denom
    margin = (
        z
        * math.sqrt(
            (phat * (1.0 - phat) / float(n))
            + ((z * z) / (4.0 * float(n) * float(n)))
        )
        / denom
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def _scale_marker_size(value: float, vmin: float, vmax: float, *, min_size: float = 6.0, max_size: float = 18.0) -> float:
    if vmax <= vmin:
        return (min_size + max_size) / 2.0
    ratio = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
    ratio = max(0.0, min(1.0, ratio))
    return min_size + (max_size - min_size) * ratio


def _marker_size_series(values: list[float], *, min_size: float = 6.0, max_size: float = 18.0) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    return [_scale_marker_size(v, vmin, vmax, min_size=min_size, max_size=max_size) for v in values]


def run(
    *,
    topic: str,
    ocr_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    figure_dir: Path,
    start_year: int | None = None,
    end_year: int | None = None,
    min_cbas_per_year: int = 10,
) -> dict[str, Any]:
    """Create mention-by-year outputs for a given keyword topic family."""
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("The `pandas` package is required.") from exc

    topic_key = str(topic).strip().lower()
    if topic_key not in TOPIC_CHOICES:
        raise ValueError(f"Unsupported topic `{topic}`. Use one of: {', '.join(TOPIC_CHOICES)}")
    min_cbas_per_year = max(1, int(min_cbas_per_year))

    metadata_path = metadata_path.expanduser().resolve()
    ocr_dir = ocr_dir.expanduser().resolve()
    output_dir = (output_dir.expanduser().resolve() / topic_key)
    figure_dir = figure_dir.expanduser().resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR directory not found: {ocr_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_stata(metadata_path, convert_categoricals=False)
    if "cbafile" not in metadata.columns:
        raise RuntimeError("Expected column `cbafile` in CBAList_fixed.dta.")
    if "expirationdate" not in metadata.columns:
        raise RuntimeError("Expected column `expirationdate` in CBAList_fixed.dta.")

    metadata = metadata.copy()
    metadata["document_id"] = metadata["cbafile"].apply(_document_id_from_cbafile)
    metadata["year"] = metadata["expirationdate"].apply(_parse_year)
    metadata = metadata[metadata["document_id"].notna()].copy()
    metadata = metadata.drop_duplicates(subset=["document_id"], keep="first").reset_index(drop=True)

    if start_year is not None:
        metadata = metadata[metadata["year"].notna() & (metadata["year"] >= int(start_year))]
    if end_year is not None:
        metadata = metadata[metadata["year"].notna() & (metadata["year"] <= int(end_year))]
    metadata = metadata.reset_index(drop=True)

    alias_regexes = _compile_alias_regexes(topic_key)
    alias_names = [name for name, _ in alias_regexes]
    topic_mentions_total_col = f"{topic_key}_mentions_total"
    has_topic_mention_col = f"has_{topic_key}_mention"
    documents_with_topic_mentions_col = f"documents_with_{topic_key}_mentions"
    share_documents_with_topic_mentions_col = f"share_documents_with_{topic_key}_mentions"
    pct_documents_with_topic_mentions_col = f"pct_documents_with_{topic_key}_mentions"
    ci95_lower_share_col = f"ci95_lower_share_documents_with_{topic_key}_mentions"
    ci95_upper_share_col = f"ci95_upper_share_documents_with_{topic_key}_mentions"
    ci95_lower_pct_col = f"ci95_lower_pct_documents_with_{topic_key}_mentions"
    ci95_upper_pct_col = f"ci95_upper_pct_documents_with_{topic_key}_mentions"
    topic_mentions_per_100_documents_col = f"{topic_key}_mentions_per_100_documents"
    topic_label = "AI terms" if topic_key == "ai" else "automation terms"

    doc_rows: list[dict[str, Any]] = []
    for _, row in metadata.iterrows():
        document_id = str(row["document_id"]).strip()
        if not document_id:
            continue
        doc_dir = ocr_dir / document_id
        text = _load_document_text(doc_dir) if doc_dir.exists() else ""

        alias_counts, total_mentions = _count_mentions(text, alias_regexes)
        payload: dict[str, Any] = {
            "document_id": document_id,
            "year": int(row["year"]) if row.get("year") is not None and str(row.get("year")) != "nan" else None,
            "employername": str(row.get("employername", "")).strip(),
            "expirationdate": str(row.get("expirationdate", "")).strip(),
            "ocr_text_found": bool(text),
            "text_char_count": int(len(text)),
            topic_mentions_total_col: int(total_mentions),
            has_topic_mention_col: bool(total_mentions > 0),
        }
        for alias_name in alias_names:
            payload[f"mentions_{alias_name}"] = int(alias_counts.get(alias_name, 0))
        doc_rows.append(payload)

    doc_df = pd.DataFrame(doc_rows)
    if doc_df.empty:
        summary = {
            "ocr_dir": str(ocr_dir),
            "metadata_path": str(metadata_path),
            "output_dir": str(output_dir),
            "figure_dir": str(figure_dir),
            "documents_considered": 0,
            "documents_with_ocr_text": 0,
            "documents_with_topic_mentions": 0,
            "total_topic_mentions": 0,
            "topic": topic_key,
            "min_cbas_per_year": int(min_cbas_per_year),
            "keyword_patterns": {name: pattern for name, pattern in KEYWORD_PATTERNS_BY_TOPIC[topic_key]},
            "outputs": {},
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    doc_df["year"] = pd.to_numeric(doc_df["year"], errors="coerce")
    doc_df = doc_df.sort_values(by=["year", "document_id"], na_position="last").reset_index(drop=True)

    yearly = (
        doc_df.dropna(subset=["year"])
        .assign(year=lambda d: d["year"].astype(int))
        .groupby("year", as_index=False)
        .agg(
            documents_total=("document_id", "count"),
            documents_with_ocr_text=("ocr_text_found", "sum"),
            **{
                documents_with_topic_mentions_col: (has_topic_mention_col, "sum"),
                topic_mentions_total_col: (topic_mentions_total_col, "sum"),
            },
            **{f"mentions_{name}": (f"mentions_{name}", "sum") for name in alias_names},
        )
    )
    yearly[share_documents_with_topic_mentions_col] = (
        yearly[documents_with_topic_mentions_col] / yearly["documents_total"]
    ).astype(float)
    yearly[pct_documents_with_topic_mentions_col] = (
        yearly[share_documents_with_topic_mentions_col] * 100.0
    ).astype(float)
    yearly[topic_mentions_per_100_documents_col] = (
        yearly[topic_mentions_total_col] * 100.0 / yearly["documents_total"]
    ).astype(float)
    yearly = yearly[yearly["documents_total"] >= min_cbas_per_year].copy()

    if not yearly.empty:
        ci_bounds = yearly.apply(
            lambda row: _wilson_interval(
                int(row[documents_with_topic_mentions_col]),
                int(row["documents_total"]),
            ),
            axis=1,
        )
        yearly[ci95_lower_share_col] = [float(bounds[0]) for bounds in ci_bounds]
        yearly[ci95_upper_share_col] = [float(bounds[1]) for bounds in ci_bounds]
    else:
        yearly[ci95_lower_share_col] = []
        yearly[ci95_upper_share_col] = []

    yearly[ci95_lower_pct_col] = yearly[ci95_lower_share_col] * 100.0
    yearly[ci95_upper_pct_col] = yearly[ci95_upper_share_col] * 100.0
    yearly = yearly.sort_values("year").reset_index(drop=True)

    doc_csv = output_dir / f"{topic_key}_mentions_by_document.csv"
    yearly_csv = output_dir / f"{topic_key}_mentions_by_year.csv"
    doc_df.to_csv(doc_csv, index=False)
    yearly.to_csv(yearly_csv, index=False)

    chart_html_path = None
    chart_png_path = None
    png_path = figure_dir / f"{topic_key}_mentions_time_series.png"
    if not yearly.empty:
        try:
            import plotly.graph_objects as go

            n_values = [float(v) for v in yearly["documents_total"].tolist()]
            marker_color = "#f97316"
            size_legend_values = [25.0, 100.0, 250.0]
            size_scale_min = min([*n_values, *size_legend_values]) if n_values else 25.0
            size_scale_max = max([*n_values, *size_legend_values]) if n_values else 250.0
            marker_sizes = [
                _scale_marker_size(v, size_scale_min, size_scale_max, min_size=6.0, max_size=18.0)
                for v in n_values
            ]
            size_legend_labels = [f"n={int(v)}" for v in size_legend_values]
            size_legend_sizes = [
                _scale_marker_size(v, size_scale_min, size_scale_max, min_size=6.0, max_size=18.0)
                for v in size_legend_values
            ]

            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=yearly["year"],
                        y=yearly[ci95_upper_pct_col],
                        mode="lines",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=yearly["year"],
                        y=yearly[ci95_lower_pct_col],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(37, 99, 235, 0.18)",
                        name="95% CI",
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=yearly["year"],
                        y=yearly[pct_documents_with_topic_mentions_col],
                        mode="lines",
                        name=f"% CBAs",
                        line={"width": 2},
                    ),
                    go.Scatter(
                        x=yearly["year"],
                        y=yearly[pct_documents_with_topic_mentions_col],
                        mode="markers",
                        name="Yearly estimate",
                        marker={"size": marker_sizes, "opacity": 0.9, "color": marker_color},
                        customdata=yearly["documents_total"],
                        hovertemplate="Year=%{x}<br>Percent=%{y:.2f}%<br>CBAs=%{customdata}<extra></extra>",
                    )
                ]
            )
            for label, size in zip(size_legend_labels, size_legend_sizes):
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=label,
                        marker={"size": size, "opacity": 0.75, "color": marker_color},
                        hoverinfo="skip",
                    )
                )
            fig.update_layout(
                title=f"Documents Mentioning {topic_label.title()} Over Time",
                xaxis_title="Expiration Year",
                yaxis_title="%",
                template="plotly_white",
                legend_title_text="Series / Dot Size",
            )

            html_path = figure_dir / f"{topic_key}_mentions_time_series.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            chart_html_path = str(html_path)

            # Prefer plotly static export; fallback to matplotlib below if kaleido is unavailable.
            try:
                fig.write_image(str(png_path), width=500, height=400, scale=2)
                chart_png_path = str(png_path)
            except Exception:
                chart_png_path = None
        except Exception:
            chart_html_path = None
            chart_png_path = None

        if chart_png_path is None:
            try:
                mpl_config_dir = (figure_dir / ".mplconfig").resolve()
                mpl_config_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
                import matplotlib.pyplot as plt

                n_values = [float(v) for v in yearly["documents_total"].tolist()]
                marker_color = "#f97316"
                size_legend_values = [25.0, 100.0, 250.0]
                size_scale_min = min([*n_values, *size_legend_values]) if n_values else 25.0
                size_scale_max = max([*n_values, *size_legend_values]) if n_values else 250.0
                marker_sizes = [
                    _scale_marker_size(v, size_scale_min, size_scale_max, min_size=6.0, max_size=18.0)
                    for v in n_values
                ]
                marker_sizes_area = [s * s for s in marker_sizes]
                size_legend_sizes = [
                    _scale_marker_size(v, size_scale_min, size_scale_max, min_size=6.0, max_size=18.0)
                    for v in size_legend_values
                ]

                fig2, ax = plt.subplots(figsize=(5, 4))
                line_handle = ax.plot(
                    yearly["year"].tolist(),
                    yearly[pct_documents_with_topic_mentions_col].tolist(),
                    linewidth=2.0,
                    label=f"% CBAs",
                )[0]
                ci_handle = ax.fill_between(
                    yearly["year"].tolist(),
                    yearly[ci95_lower_pct_col].tolist(),
                    yearly[ci95_upper_pct_col].tolist(),
                    alpha=0.2,
                    label="95% CI",
                )
                ax.scatter(
                    yearly["year"].tolist(),
                    yearly[pct_documents_with_topic_mentions_col].tolist(),
                    s=marker_sizes_area,
                    alpha=0.9,
                    color=marker_color,
                )
                ax.set_title(f"Documents Mentioning {topic_label.title()} Over Time")
                ax.set_xlabel("Expiration Year")
                ax.set_ylabel("%")
                ax.grid(True, alpha=0.3)

                legend_main = ax.legend(handles=[line_handle, ci_handle], loc="upper left")
                ax.add_artist(legend_main)
                size_handles = [
                    ax.scatter([], [], s=(size * size), alpha=0.75, color=marker_color, label=f"n={int(round(n_val))}")
                    for n_val, size in zip(size_legend_values, size_legend_sizes)
                ]
                ax.legend(handles=size_handles, title="Dot size", loc="upper right")
                fig2.tight_layout()
                fig2.savefig(png_path, dpi=200)
                plt.close(fig2)
                chart_png_path = str(png_path)
            except Exception:
                chart_png_path = None

    summary = {
        "ocr_dir": str(ocr_dir),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "documents_considered": int(len(doc_df)),
        "documents_with_ocr_text": int(doc_df["ocr_text_found"].sum()),
        "documents_with_topic_mentions": int(doc_df[has_topic_mention_col].sum()),
        "total_topic_mentions": int(doc_df[topic_mentions_total_col].sum()),
        "topic": topic_key,
        "min_cbas_per_year": int(min_cbas_per_year),
        "confidence_interval": {
            "method": "wilson",
            "level": 0.95,
            "assumption": "Bernoulli indicator per CBA for whether at least one topic keyword is mentioned",
        },
        "years_covered": (
            {
                "min": int(yearly["year"].min()),
                "max": int(yearly["year"].max()),
            }
            if not yearly.empty
            else None
        ),
        "keyword_patterns": {name: pattern for name, pattern in KEYWORD_PATTERNS_BY_TOPIC[topic_key]},
        "outputs": {
            "by_document_csv": str(doc_csv),
            "by_year_csv": str(yearly_csv),
            "plotly_time_series_html": chart_html_path,
            "plotly_time_series_png": chart_png_path,
            "time_series_png": chart_png_path,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """CLI entrypoint for OCR keyword time-series summaries."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a time series of topic keywords (`automation` or `ai`) from OCR text using "
            "document metadata in CBAList_fixed.dta."
        )
    )
    parser.add_argument("--topic", type=str, choices=list(TOPIC_CHOICES), default="automation")
    parser.add_argument("--ocr-dir", type=Path, default=_default_ocr_dir())
    parser.add_argument("--metadata-path", type=Path, default=_default_metadata_path())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--figure-dir", type=Path, default=_default_figure_dir())
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--min-cbas-per-year",
        type=int,
        default=10,
        help="Only include years with at least this many CBAs in yearly outputs and charts.",
    )
    args = parser.parse_args()

    summary = run(
        topic=args.topic,
        ocr_dir=args.ocr_dir,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        min_cbas_per_year=args.min_cbas_per_year,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
