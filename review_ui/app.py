"""Legacy Streamlit dashboard for reviewing OCR, annotations, and comparisons.

Collaborator onboarding should use `review_ui/app3.py`. This older UI remains
available for internal review flows and experiment comparisons.
"""

import io
import itertools
import json
import os
from pathlib import Path
import datetime
import html
import re

import streamlit as st
import csv
from collections import Counter, defaultdict
import altair as alt
import pandas as pd
from pypdf import PdfReader, PdfWriter
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR_ENV = os.environ.get("CACHE_DIR", "").strip()
DEFAULT_CACHE_DIR = Path(CACHE_DIR_ENV).expanduser() if CACHE_DIR_ENV else APP_DIR.parent
DEFAULT_COLLECTION = "dol_archive"

DEFAULT_DOC_DIR = DEFAULT_CACHE_DIR / DEFAULT_COLLECTION
DEFAULT_FEATURES_OUTPUT = APP_DIR.parent / "outputs" / "cba_features_annotated.jsonl"
DEFAULT_META_DTA = APP_DIR.parent / "dol_archive" / "CBAList_with_statefips.dta"
DEFAULT_OCR_DIR = DEFAULT_CACHE_DIR / "01_ocr_output" / DEFAULT_COLLECTION
DEFAULT_OCR_EXPERIMENT_DIR = APP_DIR.parent / "development" / "experiments" / "ocr"
DEFAULT_CLAUSE_EXPERIMENT_DIR = APP_DIR.parent / "development" / "experiments" / "clause_extraction"
DEFAULT_PROVISION_EXPERIMENT_DIR = APP_DIR.parent / "development" / "experiments" / "provision_identification"
DEFAULT_SEGMENTATION_EXPERIMENT_DIR = APP_DIR.parent / "development" / "experiments" / "segmentation"
DEFAULT_PDF_DIR = DEFAULT_CACHE_DIR / DEFAULT_COLLECTION
DEFAULT_ANNOTATED_JSONL = APP_DIR.parent / "outputs" / "cba_features_annotated.jsonl"


@st.cache_data(show_spinner=False)
def list_documents(doc_dir: Path):
    return sorted(doc_dir.glob("document_*.pdf"))


@st.cache_data(show_spinner=False)
def load_pdf_page(pdf_path: Path, page_number: int) -> bytes:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number])
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()

@st.cache_data(show_spinner=False)
def load_feature_rows(path: Path):
    """Load extracted features from CSV or annotated JSONL output."""
    if not path.exists():
        return []

    rows = []
    seen = set()
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                raw_doc_id = str(obj.get("document_id", "")).strip()
                if not raw_doc_id:
                    continue

                m = re.match(r"^(document_\d+)_page_(\d+)$", raw_doc_id)
                if m:
                    doc_id = m.group(1)
                    page_num = int(m.group(2))
                else:
                    doc_id = raw_doc_id
                    try:
                        page_num = int(obj.get("document_page"))
                    except Exception:
                        continue

                features = set()
                for ex in obj.get("extractions", []) if isinstance(obj.get("extractions", []), list) else []:
                    if not isinstance(ex, dict):
                        continue
                    feature_name = ""
                    attrs = ex.get("attributes")
                    if isinstance(attrs, dict):
                        feature_name = str(
                            attrs.get("feature_name")
                            or attrs.get("clause")
                            or attrs.get("label")
                            or ""
                        ).strip()
                    if not feature_name:
                        continue
                    features.add(feature_name)

                for feat in sorted(features):
                    key = (doc_id, page_num, feat)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "document_id": doc_id,
                            "document_page": page_num,
                            "feature_name": feat,
                        }
                    )
        return rows

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = str(row.get("document_id", "")).strip()
            feat = str(row.get("feature_name", "")).strip()
            try:
                page = int(row.get("document_page"))
            except Exception:
                continue
            if not doc_id or not feat:
                continue
            key = (doc_id, page, feat)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "document_id": doc_id,
                    "document_page": page,
                    "feature_name": feat,
                }
            )
    return rows


@st.cache_data(show_spinner=False)
def load_features(path: Path):
    features = {}
    for row in load_feature_rows(path):
        key = (row["document_id"], row["document_page"])
        features.setdefault(key, set()).add(row["feature_name"])
    return features


@st.cache_data(show_spinner=False)
def load_metadata(dta_path: Path):
    if not dta_path.exists():
        return pd.DataFrame()
    df = pd.read_stata(dta_path, convert_categoricals=False)
    if "cbafile" in df.columns:
        df = df.copy()
        df["document_id"] = df["cbafile"].apply(lambda x: f"document_{int(x)}" if pd.notna(x) else None)
    # Map statefips1 to state names when available
    if "statefips1" in df.columns:
        fips_to_state = {
            "01": "Alabama",
            "02": "Alaska",
            "04": "Arizona",
            "05": "Arkansas",
            "06": "California",
            "08": "Colorado",
            "09": "Connecticut",
            "10": "Delaware",
            "11": "District of Columbia",
            "12": "Florida",
            "13": "Georgia",
            "15": "Hawaii",
            "16": "Idaho",
            "17": "Illinois",
            "18": "Indiana",
            "19": "Iowa",
            "20": "Kansas",
            "21": "Kentucky",
            "22": "Louisiana",
            "23": "Maine",
            "24": "Maryland",
            "25": "Massachusetts",
            "26": "Michigan",
            "27": "Minnesota",
            "28": "Mississippi",
            "29": "Missouri",
            "30": "Montana",
            "31": "Nebraska",
            "32": "Nevada",
            "33": "New Hampshire",
            "34": "New Jersey",
            "35": "New Mexico",
            "36": "New York",
            "37": "North Carolina",
            "38": "North Dakota",
            "39": "Ohio",
            "40": "Oklahoma",
            "41": "Oregon",
            "42": "Pennsylvania",
            "44": "Rhode Island",
            "45": "South Carolina",
            "46": "South Dakota",
            "47": "Tennessee",
            "48": "Texas",
            "49": "Utah",
            "50": "Vermont",
            "51": "Virginia",
            "53": "Washington",
            "54": "West Virginia",
            "55": "Wisconsin",
            "56": "Wyoming",
        }
        def to_state_name(val):
            if pd.isna(val):
                return val
            s = str(val).strip()
            if s == "":
                return s
            if s.isdigit() and len(s) <= 2:
                s = s.zfill(2)
            return fips_to_state.get(s, s)
        df["statefips1"] = df["statefips1"].apply(to_state_name)
    # Map NAICS to 2-digit sector code
    if "naics" in df.columns:
        naics2_names = {
            "11": "Agriculture, Forestry, Fishing and Hunting",
            "21": "Mining, Quarrying, and Oil and Gas Extraction",
            "22": "Utilities",
            "23": "Construction",
            "31": "Manufacturing",
            "32": "Manufacturing",
            "33": "Manufacturing",
            "42": "Wholesale Trade",
            "44": "Retail Trade",
            "45": "Retail Trade",
            "48": "Transportation and Warehousing",
            "49": "Transportation and Warehousing",
            "51": "Information",
            "52": "Finance and Insurance",
            "53": "Real Estate and Rental and Leasing",
            "54": "Professional, Scientific, and Technical Services",
            "55": "Management of Companies and Enterprises",
            "56": "Administrative and Support and Waste Management and Remediation Services",
            "61": "Educational Services",
            "62": "Health Care and Social Assistance",
            "71": "Arts, Entertainment, and Recreation",
            "72": "Accommodation and Food Services",
            "81": "Other Services (except Public Administration)",
            "92": "Public Administration",
        }
        def to_naics2(val):
            if pd.isna(val):
                return val
            s = str(val).strip()
            if s == "":
                return s
            # keep only digits
            s = "".join(ch for ch in s if ch.isdigit())
            if len(s) < 2:
                return s
            return s[:2]
        df["naics"] = df["naics"].apply(to_naics2)
        df["naics"] = df["naics"].apply(lambda x: naics2_names.get(x, x))
    return df


def _safe_label_value(value, default: str) -> str:
    if pd.isna(value):
        return default
    s = str(value).strip()
    return s if s else default


def build_doc_display_map(meta_df: pd.DataFrame) -> dict[str, str]:
    """Build document_id -> 'Employer v Union' display labels from metadata."""
    if meta_df is None or meta_df.empty or "document_id" not in meta_df.columns:
        return {}

    union_col = "union" if "union" in meta_df.columns else None
    employer_col = "employername" if "employername" in meta_df.columns else None
    if employer_col is None and "employer" in meta_df.columns:
        employer_col = "employer"

    labels = {}
    for _, row in meta_df.iterrows():
        doc_id = row.get("document_id")
        if pd.isna(doc_id):
            continue
        doc_id = str(doc_id).strip()
        if not doc_id:
            continue
        employer_name = _safe_label_value(row[employer_col], "Employer") if employer_col else "Employer"
        union_name = _safe_label_value(row[union_col], "Union") if union_col else "Union"
        labels[doc_id] = f"{employer_name} v {union_name}"
    return labels


def first_n_chars(text: str, n: int = 30) -> str:
    s = str(text or "")
    s = re.sub(r"\s+Clause\b", "", s, flags=re.IGNORECASE)
    return s[:n]


def load_all_feature_names(taxonomy_path: Path):
    if not taxonomy_path.exists():
        return []
    text = taxonomy_path.read_text(encoding="utf-8")
    names = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("###"):
            parts = line.split(".", 1)
            if len(parts) == 2:
                name = parts[1].strip()
                if name and name not in names:
                    names.append(name)
    if "OTHER" not in names:
        names.append("OTHER")
    return names

@st.cache_data(show_spinner=False)
def load_annotated_pages(jsonl_path: Path):
    """Load LangExtract-style JSONL and index by (document_id, page_number)."""
    pages = {}
    if not jsonl_path.exists():
        return pages
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            raw_doc_id = str(obj.get("document_id", "")).strip()
            if not raw_doc_id:
                continue
            m = re.match(r"^(document_\d+)_page_(\d+)$", raw_doc_id)
            if not m:
                continue
            doc_id = m.group(1)
            page_num = int(m.group(2))
            pages[(doc_id, page_num)] = obj
    return pages


def render_clause_highlights(text: str, extractions: list[dict], height: int = 620):
    """Render highlighted text spans for clause annotations."""
    if not text:
        st.info("No text for this page.")
        return

    spans = []
    for ex in extractions or []:
        ci = ex.get("char_interval") if isinstance(ex, dict) else None
        if not isinstance(ci, dict):
            continue
        start = ci.get("start_pos")
        end = ci.get("end_pos")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        label = ""
        attrs = ex.get("attributes") if isinstance(ex, dict) else None
        if isinstance(attrs, dict):
            label = str(attrs.get("feature_name", "")).strip()
        if not label:
            label = str(ex.get("extraction_class", "Clause"))
        spans.append((start, end, label))

    spans.sort(key=lambda x: (x[0], x[1]))
    # Keep non-overlapping spans only to avoid broken markup.
    filtered = []
    last_end = -1
    for start, end, label in spans:
        if start < last_end:
            continue
        filtered.append((start, end, label))
        last_end = end

    palette = [
        "#fde68a", "#bfdbfe", "#fecaca", "#c7d2fe", "#a7f3d0",
        "#fbcfe8", "#fdba74", "#ddd6fe", "#86efac", "#fcd34d",
    ]
    label_to_color = {}
    for _, _, label in filtered:
        if label not in label_to_color:
            label_to_color[label] = palette[len(label_to_color) % len(palette)]

    chunks = []
    cursor = 0
    for start, end, label in filtered:
        if cursor < start:
            chunks.append(html.escape(text[cursor:start]))
        color = label_to_color[label]
        frag = html.escape(text[start:end])
        title = html.escape(label)
        chunks.append(
            f'<mark style="background:{color}; padding:0.05rem 0.15rem; border-radius:3px;" '
            f'title="{title}">{frag}</mark>'
        )
        cursor = end
    if cursor < len(text):
        chunks.append(html.escape(text[cursor:]))

    legend = "".join(
        f'<span style="display:inline-block; margin:0 0.5rem 0.35rem 0;">'
        f'<span style="background:{color}; padding:0.1rem 0.35rem; border-radius:3px;">&nbsp;</span> '
        f'{html.escape(label)}</span>'
        for label, color in label_to_color.items()
    )

    html_doc = f"""
    <div style="font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;">
      <div style="margin-bottom:0.5rem; font-size:0.9rem;">{legend or "No highlighted spans available."}</div>
      <div style="
          white-space: pre-wrap;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 0.9rem;
          height: {height}px;
          overflow-y: auto;
          line-height: 1.45;
          background: #ffffff;
          color: #111827;
        ">{''.join(chunks)}</div>
    </div>
    """
    components.html(html_doc, height=height + 70, scrolling=True)


def build_clause_to_doc_pages(annotations: dict[tuple[str, int], dict]) -> dict[str, dict[str, list[int]]]:
    """Build clause -> document -> pages index from annotated JSONL records."""
    raw_map: dict[str, dict[str, set[int]]] = {}
    for (doc_id, page_num), record in annotations.items():
        labels = set()
        for ex in record.get("extractions", []) if isinstance(record, dict) else []:
            if not isinstance(ex, dict):
                continue
            attrs = ex.get("attributes")
            if not isinstance(attrs, dict):
                continue
            label = str(attrs.get("feature_name", "")).strip()
            if label:
                labels.add(label)
        for label in labels:
            raw_map.setdefault(label, {}).setdefault(doc_id, set()).add(page_num)

    return {
        clause: {doc: sorted(pages) for doc, pages in doc_map.items()}
        for clause, doc_map in raw_map.items()
    }


@st.cache_data(show_spinner=False)
def list_ocr_documents(ocr_dir: Path):
    """List document dirs that have individual page_*.txt OCR files."""
    return sorted(
        [d for d in ocr_dir.iterdir() if d.is_dir() and list(d.glob("page_*.txt"))],
        key=lambda p: p.name,
    )


@st.cache_data(show_spinner=False)
def list_page_files(doc_dir: Path):
    return sorted(doc_dir.glob("page_*.txt"))


def render_ocr_viewer():
    ocr_dir = DEFAULT_OCR_DIR.expanduser().resolve()
    pdf_dir = DEFAULT_PDF_DIR.expanduser().resolve()
    meta_path = DEFAULT_META_DTA.expanduser().resolve()
    annotated_jsonl = DEFAULT_ANNOTATED_JSONL.expanduser().resolve()

    if not ocr_dir.exists():
        st.error("OCR output folder not found.")
        return

    ocr_docs = list_ocr_documents(ocr_dir)
    if not ocr_docs:
        st.warning("No OCR results found. Run the OCR pipeline first.")
        return

    annotations = load_annotated_pages(annotated_jsonl) if annotated_jsonl.exists() else {}
    clause_map = build_clause_to_doc_pages(annotations)
    clause_options = ["All"] + sorted(clause_map.keys())
    clause_filter = st.sidebar.selectbox(
        "Filter by clause",
        clause_options,
        key="ocr_clause_filter",
        help="Limit available documents/pages to those annotated with a selected clause.",
    )

    meta_df = load_metadata(meta_path)
    display_map = build_doc_display_map(meta_df)

    if clause_filter == "All":
        candidate_docs = ocr_docs
    else:
        doc_ids = set(clause_map.get(clause_filter, {}).keys())
        candidate_docs = [d for d in ocr_docs if d.name in doc_ids]
        if not candidate_docs:
            st.warning("No OCR documents found for the selected clause.")
            return

    selected_doc = st.sidebar.selectbox(
        "Document",
        candidate_docs,
        key="ocr_doc",
        format_func=lambda p: display_map.get(p.name, "Employer v Union"),
    )
    selected = selected_doc.name
    doc_dir = ocr_dir / selected

    page_files = list_page_files(doc_dir)
    num_pages = len(page_files)

    if num_pages == 0:
        st.warning("No pages found for this document.")
        return

    if clause_filter == "All":
        page_idx = (
            st.sidebar.number_input(
                "Page",
                min_value=1,
                max_value=num_pages,
                value=1,
                step=1,
                key="ocr_page",
            )
            - 1
        )
        pages_for_filter = None
    else:
        pages_for_filter = clause_map.get(clause_filter, {}).get(selected, [])
        if not pages_for_filter:
            st.warning("No pages found in this document for the selected clause.")
            return
        page_num = st.sidebar.selectbox("Page", pages_for_filter, key="ocr_page_filtered")
        page_idx = int(page_num) - 1

    # Metadata summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Pages processed:** {len(page_files)}")
    if pages_for_filter is not None:
        st.sidebar.markdown(f"**Pages matching clause:** {len(pages_for_filter)}")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader(f"{selected} — Page {page_idx + 1} of {num_pages}")
        pdf_path = pdf_dir / f"{selected}.pdf"
        if pdf_path.exists():
            try:
                pdf_bytes = load_pdf_page(pdf_path, page_idx)
                st.pdf(pdf_bytes, height=800)
            except Exception as exc:
                st.error(f"Failed to render PDF page: {exc}")
        else:
            st.info(f"Source PDF not found: {pdf_path.name}")

    with col_right:
        st.subheader("OCR Text with Clause Highlights")

        # Get page text from individual page files
        page_text = ""
        if page_idx < len(page_files):
            page_text = page_files[page_idx + 1].read_text(encoding="utf-8")

        annotations = load_annotated_pages(annotated_jsonl) if annotated_jsonl.exists() else {}
        page_num = page_idx + 1
        record = annotations.get((selected, page_num))
        if record:
            page_text = str(record.get("text", page_text))
            render_clause_highlights(page_text, record.get("extractions", []))
            st.caption(f"Showing highlights from {annotated_jsonl.name}")
        else:
            render_clause_highlights(page_text, [])
            if annotated_jsonl.exists():
                st.caption("No annotation row found for this page in annotated JSONL.")
            else:
                st.caption("Annotated JSONL not found; showing plain OCR text.")


def render_ocr_comparison():
    """Compare OCR outputs from experiment methods side-by-side."""
    experiment_dir = DEFAULT_OCR_EXPERIMENT_DIR.expanduser().resolve()
    output_dir = experiment_dir / "output"
    results_file = experiment_dir / "results.csv"
    test_pdfs_dir = experiment_dir / "test_pdfs"

    if not output_dir.exists():
        st.warning("No experiment outputs found. Run the OCR experiment first.")
        return

    # Load results CSV for metrics and agreement scores
    metrics = {}  # (pdf_stem, page, method_a, method_b) -> {cer, wer}
    page_agreement = defaultdict(list)  # (pdf_stem, page) -> list of (cer+wer)/2
    if results_file.exists():
        with results_file.open(newline="") as f:
            for row in csv.DictReader(f):
                pdf_stem_r = Path(row["pdf"]).stem
                page_r = int(row["page"])
                cer = float(row["cer"])
                wer = float(row["wer"])
                key = (pdf_stem_r, page_r, row["method_a"], row["method_b"])
                metrics[key] = {"cer": cer, "wer": wer}
                page_agreement[(pdf_stem_r, page_r)].append((cer + wer) / 2)

    # Compute agreement scores: higher = more agreement (1.0 = perfect)
    agreement_scores = {k: 1 - sum(v) / len(v) for k, v in page_agreement.items()}

    # Discover all (doc, page) pairs across all experiment output
    doc_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not doc_dirs:
        st.warning("No experiment output directories found.")
        return

    # Load all method data per document
    all_doc_methods = {}  # pdf_stem -> {method_name -> {page -> text}}
    all_entries = []  # list of (pdf_stem, page)
    for doc_dir in doc_dirs:
        stem = doc_dir.name
        doc_methods = {}
        for mf in sorted(doc_dir.glob("*.json")):
            data = json.loads(mf.read_text())
            doc_methods[mf.stem] = {int(k): v for k, v in data.items()}
        if not doc_methods:
            continue
        all_doc_methods[stem] = doc_methods
        pages = sorted(set().union(*(m.keys() for m in doc_methods.values())))
        for p in pages:
            all_entries.append((stem, p))

    if not all_entries:
        st.warning("No pages found in experiment output.")
        return

    # Rank all entries by agreement score (least agreeable first)
    all_entries.sort(key=lambda x: agreement_scores.get(x, 1.0))

    def entry_label(entry):
        stem, p = entry
        score = agreement_scores.get((stem, p))
        if score is not None:
            return f"{stem} p.{p} (agreement: {score:.4f})"
        return f"{stem} p.{p}"

    # Sidebar: page selection
    selected_entry = st.sidebar.selectbox(
        "Page",
        all_entries,
        format_func=entry_label,
        key="ocr_cmp_page",
    )
    pdf_stem, page_num = selected_entry

    methods = all_doc_methods[pdf_stem]
    preferred_order = ["pdftotext", "vision_model", "olmocr"]
    method_names = [m for m in preferred_order if m in methods]
    method_names += [m for m in sorted(methods.keys()) if m not in preferred_order]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Methods:** {', '.join(method_names)}")
    st.sidebar.markdown(f"**Total pages:** {len(all_entries)}")
    current_score = agreement_scores.get((pdf_stem, page_num))
    if current_score is not None:
        st.sidebar.markdown(f"**Agreement score:** {current_score:.4f}")

    # --- Pairwise metrics table ---
    st.subheader(f"{pdf_stem} — Page {page_num}")

    pairs = list(itertools.combinations(method_names, 2))
    if pairs and metrics:
        metric_rows = []
        for m1, m2 in pairs:
            # Try both orderings
            key = (pdf_stem, page_num, m1, m2)
            key_rev = (pdf_stem, page_num, m2, m1)
            m = metrics.get(key) or metrics.get(key_rev)
            if m:
                metric_rows.append({
                    "Method A": m1,
                    "Method B": m2,
                    "CER": f"{m['cer']:.4f}",
                    "WER": f"{m['wer']:.4f}",
                })
            else:
                metric_rows.append({
                    "Method A": m1,
                    "Method B": m2,
                    "CER": "—",
                    "WER": "—",
                })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    # --- PDF preview (left) + stacked OCR outputs (right) ---
    pdf_path = test_pdfs_dir / f"{pdf_stem}.pdf"
    show_pdf = pdf_path.exists()

    col_left, col_right = st.columns([1, 1], gap="medium")

    if show_pdf:
        with col_left:
            st.markdown("**PDF**")
            try:
                pdf_bytes = load_pdf_page(pdf_path, page_num - 1)
                st.pdf(pdf_bytes, height=900)
            except Exception as exc:
                st.error(f"Failed to render PDF: {exc}")

    with col_right:
        for method in method_names:
            text = methods[method].get(page_num, "")
            char_count = len(text) if text else 0
            st.markdown(f"**{method}** ({char_count} chars)")
            st.text_area(
                label=method,
                value=text or "(no output)",
                height=300,
                key=f"ocr_cmp_text_{method}_{pdf_stem}_{page_num}",
                label_visibility="collapsed",
            )


def _get_visible_spans(text: str, extractions: list[dict]) -> list[tuple[int, int, str]]:
    """Return non-overlapping spans that will actually be highlighted."""
    spans = []
    for ex in extractions or []:
        start = ex.get("start_pos")
        end = ex.get("end_pos")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        label = ex.get("clause_label", "Unknown")
        spans.append((start, end, label))

    spans.sort(key=lambda x: (x[0], x[1]))
    filtered = []
    last_end = -1
    for start, end, label in spans:
        if start < last_end:
            continue
        filtered.append((start, end, label))
        last_end = end
    return filtered


def _render_experiment_highlights(
    text: str,
    extractions: list[dict],
    label_to_color: dict[str, str],
    height: int = 620,
):
    """Render highlighted spans (no legend — rendered separately)."""
    if not text:
        st.info("No text for this page.")
        return

    filtered = _get_visible_spans(text, extractions)

    for _, _, label in filtered:
        if label not in label_to_color:
            idx = len(label_to_color) % len(_CLAUSE_PALETTE)
            label_to_color[label] = _CLAUSE_PALETTE[idx]

    chunks = []
    cursor = 0
    for start, end, label in filtered:
        if cursor < start:
            chunks.append(html.escape(text[cursor:start]))
        color = label_to_color[label]
        frag = html.escape(text[start:end])
        title = html.escape(label)
        chunks.append(
            f'<mark style="background:{color};'
            f' padding:0.05rem 0.15rem;'
            f' border-radius:3px;"'
            f' title="{title}">{frag}</mark>'
        )
        cursor = end
    if cursor < len(text):
        chunks.append(html.escape(text[cursor:]))

    html_doc = f"""
    <div style="font-family: ui-sans-serif, system-ui, sans-serif;">
      <div style="
          white-space: pre-wrap;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 0.9rem;
          height: {height}px;
          overflow-y: auto;
          line-height: 1.45;
          background: #ffffff;
          color: #111827;
        ">{''.join(chunks)}</div>
    </div>
    """
    components.html(html_doc, height=height + 40, scrolling=True)


_CLAUSE_PALETTE = [
    "#fde68a", "#bfdbfe", "#fecaca", "#c7d2fe", "#a7f3d0",
    "#fbcfe8", "#fdba74", "#ddd6fe", "#86efac", "#fcd34d",
    "#fed7aa", "#e9d5ff", "#99f6e4", "#fca5a5", "#a5b4fc",
]


def render_clause_extraction_comparison():
    """Compare clause extraction methods side-by-side with highlighted spans."""
    experiment_dir = DEFAULT_CLAUSE_EXPERIMENT_DIR.expanduser().resolve()
    output_dir = experiment_dir / "output"
    ocr_dir = DEFAULT_OCR_DIR.expanduser().resolve()

    if not output_dir.exists():
        st.warning("No clause extraction experiment outputs found. Run the experiment first.")
        return

    # Discover documents with output
    doc_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not doc_dirs:
        st.warning("No experiment output directories found.")
        return

    # Load method results per document
    doc_methods: dict[str, dict[str, dict]] = {}  # doc_id -> {method -> {page_str -> extractions}}
    for d in doc_dirs:
        methods = {}
        for mf in sorted(d.glob("*.json")):
            methods[mf.stem] = json.loads(mf.read_text())
        if methods:
            doc_methods[d.name] = methods

    if not doc_methods:
        st.warning("No method outputs found.")
        return

    # Metadata for display labels
    meta_df = load_metadata(DEFAULT_META_DTA.expanduser().resolve())
    display_map = build_doc_display_map(meta_df)

    doc_ids = sorted(doc_methods.keys())
    selected_doc = st.sidebar.selectbox(
        "Document",
        doc_ids,
        format_func=lambda d: display_map.get(d, d),
        key="clause_cmp_doc",
    )

    methods = doc_methods[selected_doc]
    method_names = sorted(methods.keys())

    # Collect all pages across methods
    all_pages = sorted(set(
        int(p) for m in methods.values() for p in m.keys()
    ))

    if not all_pages:
        st.warning("No pages found for this document.")
        return

    page_num = st.sidebar.selectbox("Page", all_pages, key="clause_cmp_page")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Methods:** {', '.join(method_names)}")
    st.sidebar.markdown(f"**Pages:** {len(all_pages)}")

    # Load page text from OCR output
    page_file = ocr_dir / selected_doc / f"page_{page_num:04d}.txt"
    page_text = ""
    if page_file.exists():
        page_text = page_file.read_text(encoding="utf-8", errors="replace")

    st.subheader(f"{display_map.get(selected_doc, selected_doc)} — Page {page_num}")

    # Collect only labels that actually appear as visible highlights
    # across all methods (after overlap filtering + bounds checks)
    visible_labels: set[str] = set()
    method_extractions: dict[str, list[dict]] = {}
    for method in method_names:
        exts = methods[method].get(str(page_num), [])
        method_extractions[method] = exts
        for _, _, label in _get_visible_spans(page_text, exts):
            visible_labels.add(label)

    # Build shared color map from visible labels only
    label_to_color: dict[str, str] = {}
    for label in sorted(visible_labels):
        label_to_color[label] = _CLAUSE_PALETTE[
            len(label_to_color) % len(_CLAUSE_PALETTE)
        ]

    # Shared legend
    if label_to_color:
        legend_html = "".join(
            f'<span style="display:inline-block;'
            f' margin:0 0.5rem 0.35rem 0;">'
            f'<span style="background:{color};'
            f' padding:0.1rem 0.35rem;'
            f' border-radius:3px;">&nbsp;</span> '
            f'{html.escape(label)}</span>'
            for label, color in label_to_color.items()
        )
        components.html(
            f'<div style="font-family: ui-sans-serif, system-ui,'
            f' sans-serif; font-size:0.9rem;">'
            f'{legend_html}</div>',
            height=60,
            scrolling=True,
        )

    # Render side-by-side columns
    cols = st.columns(len(method_names), gap="medium")
    for col, method in zip(cols, method_names):
        with col:
            exts = method_extractions[method]
            st.markdown(f"**{method}** ({len(exts)} extractions)")
            _render_experiment_highlights(
                page_text, exts, label_to_color, height=600,
            )


@st.cache_data(show_spinner=False)
def load_provision_result_rows(results_path: Path):
    if not results_path.exists():
        return []
    rows = []
    with results_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _resolve_output_json_path(raw_path: str, experiment_dir: Path) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            (experiment_dir / p).resolve(),
            (APP_DIR.parent / p).resolve(),
            (Path.cwd() / p).resolve(),
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _canonical_segmentation_version(version: str) -> str:
    v = str(version or "").strip()
    if not v:
        return v
    for marker in ("__model_", "__plan_"):
        if marker in v:
            v = v.split(marker, 1)[0]
    return v


@st.cache_data(show_spinner=False)
def load_page_extractions_json(path: Path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    out = {}
    for k, v in data.items():
        try:
            page = int(k)
        except Exception:
            continue
        out[page] = v if isinstance(v, list) else []
    return out


@st.cache_data(show_spinner=False)
def load_segmentation_output_json(path: Path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    clean_segments = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        clean_segments.append(seg)

    out = dict(data)
    out["segments"] = clean_segments
    return out


def _load_doc_page_text_and_spans(doc_dir: Path):
    page_numbers = []
    text_by_page: dict[int, str] = {}
    span_by_page: dict[int, tuple[int, int]] = {}

    if not doc_dir.exists():
        return {
            "page_numbers": page_numbers,
            "text_by_page": text_by_page,
            "span_by_page": span_by_page,
        }

    offset = 0
    has_non_empty = False
    for pf in sorted(doc_dir.glob("page_*.txt")):
        m = re.match(r"page_(\d+)\.txt$", pf.name)
        if not m:
            continue
        page_num = int(m.group(1))
        page_numbers.append(page_num)

        text = pf.read_text(encoding="utf-8", errors="replace").strip()
        text_by_page[page_num] = text
        if not text:
            continue

        if has_non_empty:
            offset += 2  # separator used by segmentation method: "\n\n"

        start = offset
        end = start + len(text)
        span_by_page[page_num] = (start, end)
        offset = end
        has_non_empty = True

    return {
        "page_numbers": page_numbers,
        "text_by_page": text_by_page,
        "span_by_page": span_by_page,
    }


def _segmentation_segments_for_page(
    segments: list[dict],
    page_num: int,
    page_text: str,
    page_span: tuple[int, int] | None,
) -> list[dict]:
    if not page_text:
        return []

    page_len = len(page_text)
    out = []

    for seg in segments or []:
        if not isinstance(seg, dict):
            continue

        local_start = None
        local_end = None
        start = seg.get("start_pos")
        end = seg.get("end_pos")

        if (
            page_span is not None
            and isinstance(start, int)
            and isinstance(end, int)
            and end > start
        ):
            page_start, page_end = page_span
            if start < page_end and end > page_start:
                local_start = max(0, start - page_start)
                local_end = min(page_len, end - page_start)

        if local_start is None or local_end is None:
            start_page = seg.get("start_page")
            end_page = seg.get("end_page")
            if (
                isinstance(start_page, int)
                and isinstance(end_page, int)
                and start_page <= page_num <= end_page
            ):
                snippet = str(seg.get("text", "")).strip()
                if snippet:
                    idx = page_text.find(snippet)
                    if idx < 0 and len(snippet) > 200:
                        idx = page_text.find(snippet[:200])
                    if idx < 0:
                        idx = page_text.lower().find(snippet.lower())
                    if idx >= 0:
                        local_start = idx
                        local_end = min(page_len, idx + len(snippet))

        if (
            local_start is None
            or local_end is None
            or local_end <= local_start
            or local_start < 0
            or local_end > page_len
        ):
            continue

        out.append({
            "parent": str(seg.get("parent", "")).strip() or "Unknown Parent",
            "title": str(seg.get("title", "")).strip() or "Untitled",
            "text": str(seg.get("text", "")).strip(),
            "local_start": int(local_start),
            "local_end": int(local_end),
        })

    out.sort(key=lambda x: (x["local_start"], x["local_end"]))
    filtered = []
    last_end = -1
    for seg in out:
        if seg["local_start"] < last_end:
            continue
        filtered.append(seg)
        last_end = seg["local_end"]
    return filtered


def _render_segmentation_highlights(text: str, page_segments: list[dict], height: int = 580):
    if not text:
        st.info("No text for this page.")
        return

    palette = [
        "#fde68a", "#bfdbfe", "#fecaca", "#c7d2fe", "#a7f3d0",
        "#fbcfe8", "#fdba74", "#ddd6fe", "#86efac", "#fcd34d",
        "#fed7aa", "#99f6e4",
    ]
    chunks = []
    cursor = 0

    for idx, seg in enumerate(page_segments):
        start = seg["local_start"]
        end = seg["local_end"]
        color = palette[idx % len(palette)]
        if cursor < start:
            chunks.append(html.escape(text[cursor:start]))
        frag = html.escape(text[start:end])
        tooltip = html.escape(f'{seg["parent"]} > {seg["title"]}')
        chunks.append(
            f'<mark style="background:{color};'
            f' padding:0.05rem 0.15rem;'
            f' border-radius:3px;" title="{tooltip}">{frag}</mark>'
        )
        cursor = end

    if cursor < len(text):
        chunks.append(html.escape(text[cursor:]))

    html_doc = f"""
    <div style="font-family: ui-sans-serif, system-ui, sans-serif;">
      <div style="margin-bottom:0.45rem; font-size:0.9rem;">
        Highlighted segments on page: {len(page_segments)}
      </div>
      <div style="
          white-space: pre-wrap;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 0.9rem;
          height: {height}px;
          overflow-y: auto;
          line-height: 1.45;
          background: #ffffff;
          color: #111827;
        ">{''.join(chunks)}</div>
    </div>
    """
    components.html(html_doc, height=height + 65, scrolling=True)


def _render_synced_segmentation_panes(
    text: str,
    panes: list[dict],
    height: int = 520,
    lock_scroll: bool = True,
):
    """Render multiple segmentation panes inside one component with optional scroll lock."""
    if not text:
        st.info("No text for this page.")
        return
    if not panes:
        st.info("No segmentation runs selected.")
        return

    palette = [
        "#fde68a", "#bfdbfe", "#fecaca", "#c7d2fe", "#a7f3d0",
        "#fbcfe8", "#fdba74", "#ddd6fe", "#86efac", "#fcd34d",
        "#fed7aa", "#99f6e4",
    ]
    pane_cards = []
    for pane_idx, pane in enumerate(panes):
        page_segments = pane.get("segments", [])
        chunks = []
        cursor = 0

        for idx, seg in enumerate(page_segments):
            start = seg["local_start"]
            end = seg["local_end"]
            color = palette[idx % len(palette)]
            if cursor < start:
                chunks.append(html.escape(text[cursor:start]))
            frag = html.escape(text[start:end])
            tooltip = html.escape(f'{seg["parent"]} > {seg["title"]}')
            chunks.append(
                f'<mark style="background:{color};'
                f' padding:0.05rem 0.15rem;'
                f' border-radius:3px;" title="{tooltip}">{frag}</mark>'
            )
            cursor = end

        if cursor < len(text):
            chunks.append(html.escape(text[cursor:]))

        run_key = html.escape(str(pane.get("run_key", "Run")))
        pane_cards.append(
            f"""
            <div class="seg-pane-card">
              <div class="seg-pane-title">{run_key}</div>
              <div class="seg-pane-subtitle">{len(page_segments)} segments on this page</div>
              <div class="seg-pane-scroll" data-pane-index="{pane_idx}">
                {''.join(chunks)}
              </div>
            </div>
            """
        )

    cols = min(3, max(1, len(panes)))
    rows = (len(panes) + cols - 1) // cols
    component_height = rows * (height + 90) + 40

    html_doc = f"""
    <div style="font-family: ui-sans-serif, system-ui, sans-serif;">
      <div style="margin-bottom:0.45rem; font-size:0.9rem;">
        {("Pane scroll is locked." if lock_scroll else "Pane scroll is unlocked.")}
      </div>
      <div class="seg-grid">
        {''.join(pane_cards)}
      </div>
    </div>
    <style>
      .seg-grid {{
        display: grid;
        grid-template-columns: repeat({cols}, minmax(0, 1fr));
        gap: 0.75rem;
      }}
      .seg-pane-card {{
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.6rem;
        background: #ffffff;
      }}
      .seg-pane-title {{
        font-weight: 600;
        margin-bottom: 0.15rem;
      }}
      .seg-pane-subtitle {{
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.45rem;
      }}
      .seg-pane-scroll {{
        white-space: pre-wrap;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.85rem;
        height: {height}px;
        overflow-y: auto;
        line-height: 1.45;
        background: #ffffff;
        color: #111827;
      }}
    </style>
    <script>
      (() => {{
        const lockScroll = {"true" if lock_scroll else "false"};
        if (!lockScroll) return;
        const panes = Array.from(document.querySelectorAll(".seg-pane-scroll"));
        if (panes.length < 2) return;

        let syncing = false;
        panes.forEach((pane) => {{
          pane.addEventListener("scroll", () => {{
            if (syncing) return;
            syncing = true;
            const paneMax = Math.max(1, pane.scrollHeight - pane.clientHeight);
            const ratio = pane.scrollTop / paneMax;
            panes.forEach((other) => {{
              if (other === pane) return;
              const otherMax = Math.max(0, other.scrollHeight - other.clientHeight);
              other.scrollTop = ratio * otherMax;
            }});
            syncing = false;
          }}, {{ passive: true }});
        }});
      }})();
    </script>
    """
    components.html(html_doc, height=component_height, scrolling=True)


def _get_provision_spans(text: str, extractions: list[dict]) -> list[tuple[int, int]]:
    spans = []
    for ex in extractions or []:
        if not isinstance(ex, dict):
            continue
        start = ex.get("start_pos")
        end = ex.get("end_pos")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        spans.append((start, end))

    spans.sort(key=lambda x: (x[0], x[1]))
    filtered = []
    last_end = -1
    for start, end in spans:
        if start < last_end:
            continue
        filtered.append((start, end))
        last_end = end
    return filtered


def _chars_for_spans(spans: list[tuple[int, int]]) -> set[int]:
    chars = set()
    for start, end in spans:
        chars.update(range(start, end))
    return chars


def _span_iou(set_a: set[int], set_b: set[int]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _render_provision_highlights(text: str, extractions: list[dict], height: int = 580):
    if not text:
        st.info("No text for this page.")
        return

    spans = _get_provision_spans(text, extractions)
    palette = [
        "#fde68a", "#bfdbfe", "#fecaca", "#c7d2fe", "#a7f3d0",
        "#fbcfe8", "#fdba74", "#ddd6fe", "#86efac", "#fcd34d",
        "#fed7aa", "#99f6e4",
    ]
    chunks = []
    cursor = 0

    for idx, (start, end) in enumerate(spans):
        color = palette[idx % len(palette)]
        if cursor < start:
            chunks.append(html.escape(text[cursor:start]))
        frag = html.escape(text[start:end])
        title = f"Provision {idx + 1}"
        chunks.append(
            f'<mark style="background:{color};'
            f' padding:0.05rem 0.15rem;'
            f' border-radius:3px;" title="{title}">{frag}</mark>'
        )
        cursor = end

    if cursor < len(text):
        chunks.append(html.escape(text[cursor:]))

    html_doc = f"""
    <div style="font-family: ui-sans-serif, system-ui, sans-serif;">
      <div style="margin-bottom:0.45rem; font-size:0.9rem;">
        Highlighted provision spans ({len(spans)}). Colors are assigned per span.
      </div>
      <div style="
          white-space: pre-wrap;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 0.9rem;
          height: {height}px;
          overflow-y: auto;
          line-height: 1.45;
          background: #ffffff;
          color: #111827;
        ">{''.join(chunks)}</div>
    </div>
    """
    components.html(html_doc, height=height + 65, scrolling=True)


def render_provision_identification_comparison():
    """Compare provision-identification variants side-by-side."""
    experiment_dir = DEFAULT_PROVISION_EXPERIMENT_DIR.expanduser().resolve()
    output_dir = experiment_dir / "output"
    results_file = experiment_dir / "results.csv"
    ocr_dir = DEFAULT_OCR_DIR.expanduser().resolve()

    if not output_dir.exists():
        st.warning("No provision-identification outputs found. Run the experiment first.")
        return

    rows = load_provision_result_rows(results_file)
    # doc_id -> version -> {page -> [extractions]}
    doc_versions: dict[str, dict[str, dict[int, list[dict]]]] = {}

    if rows:
        cache_by_path: dict[str, dict[int, list[dict]]] = {}
        for row in rows:
            method = str(row.get("method", "")).strip()
            if method and method != "doublepass":
                continue
            doc_id = str(row.get("document_id", "")).strip()
            version = str(row.get("version", "")).strip()
            raw_output_path = str(row.get("output_json_path", "")).strip()
            if not doc_id or not version or not raw_output_path:
                continue
            resolved = _resolve_output_json_path(raw_output_path, experiment_dir)
            if resolved is None:
                continue
            cache_key = str(resolved)
            if cache_key not in cache_by_path:
                cache_by_path[cache_key] = load_page_extractions_json(resolved)
            doc_versions.setdefault(doc_id, {})[version] = cache_by_path[cache_key]

    # Fallback: discover outputs directly if results.csv is missing/incomplete.
    if not doc_versions:
        for doc_dir in sorted([d for d in output_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
            for mf in sorted(doc_dir.glob("*.json")):
                version = mf.stem
                if version.startswith("doublepass__"):
                    version = version[len("doublepass__"):]
                doc_versions.setdefault(doc_dir.name, {})[version] = load_page_extractions_json(mf)

    if not doc_versions:
        st.warning("No provision-identification outputs discovered.")
        return

    meta_df = load_metadata(DEFAULT_META_DTA.expanduser().resolve())
    display_map = build_doc_display_map(meta_df)

    doc_ids = sorted(doc_versions.keys())
    selected_doc = st.sidebar.selectbox(
        "Document",
        doc_ids,
        format_func=lambda d: display_map.get(d, d),
        key="prov_cmp_doc",
    )

    versions = sorted(doc_versions[selected_doc].keys())
    selected_versions = st.sidebar.multiselect(
        "Variants",
        options=versions,
        default=versions[: min(3, len(versions))],
        key="prov_cmp_versions",
    )
    if not selected_versions:
        st.info("Select at least one variant.")
        return

    all_pages = sorted(set(
        p for version in selected_versions for p in doc_versions[selected_doc][version].keys()
    ))
    if not all_pages:
        st.warning("No pages found for selected variant(s).")
        return

    ocr_doc_dir = ocr_dir / selected_doc
    ocr_page_files = list_page_files(ocr_doc_dir) if ocr_doc_dir.exists() else []
    max_page = len(ocr_page_files) if ocr_page_files else max(all_pages)
    default_page = min(max(all_pages[0], 1), max_page)
    page_num = int(
        st.sidebar.number_input(
            "Page",
            min_value=1,
            max_value=max_page,
            value=default_page,
            step=1,
            key="prov_cmp_page_step",
        )
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Variants selected:** {len(selected_versions)}")
    st.sidebar.markdown(f"**Pages with extractions:** {len(all_pages)}")
    st.sidebar.markdown(f"**Document pages:** {max_page}")

    page_file = ocr_dir / selected_doc / f"page_{page_num:04d}.txt"
    page_text = ""
    if page_file.exists():
        page_text = page_file.read_text(encoding="utf-8", errors="replace")

    st.subheader(f"{display_map.get(selected_doc, selected_doc)} — Page {page_num}")

    summary_rows = []
    variant_char_sets: dict[str, set[int]] = {}
    for version in selected_versions:
        exts = doc_versions[selected_doc][version].get(page_num, [])
        spans = _get_provision_spans(page_text, exts)
        covered_chars = sum(end - start for start, end in spans)
        variant_char_sets[version] = _chars_for_spans(spans)

        runtime = None
        for row in rows:
            if (
                str(row.get("document_id", "")).strip() == selected_doc
                and str(row.get("version", "")).strip() == version
                and str(row.get("page", "")).strip() == str(page_num)
            ):
                try:
                    runtime = float(row.get("runtime_sec"))
                except Exception:
                    runtime = None
                break

        summary_rows.append({
            "variant": version,
            "extractions": len(exts),
            "visible_spans": len(spans),
            "covered_chars": covered_chars,
            "runtime_sec": runtime,
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    pairwise = []
    for a, b in itertools.combinations(selected_versions, 2):
        pairwise.append({
            "variant_a": a,
            "variant_b": b,
            "char_iou": round(_span_iou(variant_char_sets[a], variant_char_sets[b]), 4),
        })
    if pairwise:
        st.caption("Pairwise overlap between highlighted character spans (IoU).")
        st.dataframe(pd.DataFrame(pairwise), use_container_width=True, hide_index=True)

    if len(selected_versions) <= 3:
        cols = st.columns(len(selected_versions), gap="medium")
        for col, version in zip(cols, selected_versions):
            with col:
                exts = doc_versions[selected_doc][version].get(page_num, [])
                st.markdown(f"**{version}**")
                st.caption(f"{len(exts)} extracted spans")
                _render_provision_highlights(page_text, exts, height=560)
    else:
        for version in selected_versions:
            exts = doc_versions[selected_doc][version].get(page_num, [])
            st.markdown(f"**{version}**")
            st.caption(f"{len(exts)} extracted spans")
            _render_provision_highlights(page_text, exts, height=450)


def render_segmentation_review():
    """Review segmentation experiment outputs with page-level highlights."""
    experiment_dir = DEFAULT_SEGMENTATION_EXPERIMENT_DIR.expanduser().resolve()
    output_dir = experiment_dir / "output"
    results_file = experiment_dir / "results.csv"
    ocr_dir = DEFAULT_OCR_DIR.expanduser().resolve()
    pdf_dir = DEFAULT_PDF_DIR.expanduser().resolve()
    supported_methods = {"llm_segment", "llm_segment_v2"}

    if not output_dir.exists():
        st.warning("No segmentation outputs found. Run the segmentation experiment first.")
        return

    rows = load_provision_result_rows(results_file)
    # doc_id -> run_key(method:version) -> output_payload
    doc_runs: dict[str, dict[str, dict]] = {}
    runtime_by_run_key: dict[tuple[str, str], float] = {}

    def parse_run_key(run_key: str) -> tuple[str, str]:
        if ":" in run_key:
            method, version = run_key.split(":", 1)
            return method, version
        return run_key, ""

    if rows:
        cache_by_path: dict[str, dict] = {}
        for row in rows:
            row_type = str(row.get("row_type", "run")).strip()
            if row_type and row_type != "run":
                continue
            method = str(row.get("method", "")).strip()
            if method not in supported_methods:
                continue
            doc_id = str(row.get("document_id", "")).strip()
            version = _canonical_segmentation_version(str(row.get("version", "")).strip())
            raw_output_path = str(row.get("output_json_path", "")).strip()
            if not doc_id or not version or not raw_output_path:
                continue
            resolved = _resolve_output_json_path(raw_output_path, experiment_dir)
            if resolved is None:
                continue
            cache_key = str(resolved)
            if cache_key not in cache_by_path:
                cache_by_path[cache_key] = load_segmentation_output_json(resolved)
            run_key = f"{method}:{version}"
            doc_runs.setdefault(doc_id, {})[run_key] = cache_by_path[cache_key]
            try:
                runtime = float(row.get("runtime_sec"))
                runtime_by_run_key[(doc_id, run_key)] = runtime
            except Exception:
                pass

    # Also discover directly from output folder in case results.csv is partial.
    for doc_dir in sorted([d for d in output_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
        for method, pattern in [
            ("llm_segment", "llm_segment__*.json"),
            ("llm_segment_v2", "llm_segment_v2__*.json"),
        ]:
            for mf in sorted(doc_dir.glob(pattern)):
                payload = load_segmentation_output_json(mf)
                version = _canonical_segmentation_version(str(payload.get("version", "")).strip())
                if not version:
                    prefix = f"{method}__"
                    version = mf.stem
                    if version.startswith(prefix):
                        version = version[len(prefix):]
                    version = _canonical_segmentation_version(version)
                if not version:
                    continue
                run_key = f"{method}:{version}"
                doc_runs.setdefault(doc_dir.name, {}).setdefault(run_key, payload)

    if not doc_runs:
        st.warning("No segmentation outputs discovered.")
        return

    meta_df = load_metadata(DEFAULT_META_DTA.expanduser().resolve())
    display_map = build_doc_display_map(meta_df)

    doc_ids = sorted(doc_runs.keys())
    selected_doc = st.sidebar.selectbox(
        "Document",
        doc_ids,
        format_func=lambda d: display_map.get(d, d),
        key="seg_review_doc",
    )

    run_keys_all = sorted(doc_runs[selected_doc].keys())
    methods = sorted({parse_run_key(rk)[0] for rk in run_keys_all})
    selected_methods = st.sidebar.multiselect(
        "Methods",
        options=methods,
        default=methods,
        key="seg_review_methods",
    )
    if not selected_methods:
        st.info("Select at least one method.")
        return

    versions = sorted({
        parse_run_key(rk)[1]
        for rk in run_keys_all
        if parse_run_key(rk)[0] in selected_methods
    })
    selected_versions = st.sidebar.multiselect(
        "Variants",
        options=versions,
        default=versions,
        key="seg_review_versions",
    )
    if not selected_versions:
        st.info("Select at least one variant.")
        return

    selected_run_keys = []
    for method in selected_methods:
        for version in selected_versions:
            run_key = f"{method}:{version}"
            if run_key in doc_runs[selected_doc]:
                selected_run_keys.append(run_key)
    if not selected_run_keys:
        st.warning("No runs found for selected method/variant filters.")
        return

    ocr_doc_dir = ocr_dir / selected_doc
    page_payload = _load_doc_page_text_and_spans(ocr_doc_dir)
    doc_page_numbers = page_payload["page_numbers"]
    text_by_page = page_payload["text_by_page"]
    span_by_page = page_payload["span_by_page"]

    page_candidates = set(doc_page_numbers)
    for run_key in selected_run_keys:
        for seg in doc_runs[selected_doc][run_key].get("segments", []):
            sp = seg.get("start_page")
            ep = seg.get("end_page")
            if isinstance(sp, int):
                page_candidates.add(sp)
            if isinstance(ep, int):
                page_candidates.add(ep)

    if not page_candidates:
        st.warning("No pages found for selected document/variants.")
        return

    max_page = max(page_candidates)
    default_page = min(page_candidates)
    page_num = int(
        st.sidebar.number_input(
            "Page",
            min_value=1,
            max_value=max_page,
            value=default_page,
            step=1,
            key="seg_review_page_step",
        )
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Methods selected:** {len(selected_methods)}")
    st.sidebar.markdown(f"**Variants selected:** {len(selected_versions)}")
    st.sidebar.markdown(f"**Run columns:** {len(selected_run_keys)}")
    st.sidebar.markdown(f"**Document pages:** {len(doc_page_numbers) if doc_page_numbers else max_page}")
    lock_pane_scroll = st.sidebar.checkbox(
        "Lock pane scroll",
        value=(len(selected_run_keys) > 1),
        key="seg_review_lock_scroll",
        help="Synchronize vertical scroll position between segmentation panes.",
    )

    page_text = text_by_page.get(page_num, "")
    if not page_text:
        page_file = ocr_doc_dir / f"page_{page_num:04d}.txt"
        if page_file.exists():
            page_text = page_file.read_text(encoding="utf-8", errors="replace").strip()
    page_span = span_by_page.get(page_num)

    st.subheader(f"{display_map.get(selected_doc, selected_doc)} — Page {page_num}")

    pdf_path = pdf_dir / f"{selected_doc}.pdf"
    if pdf_path.exists():
        with st.expander("Show PDF page", expanded=False):
            try:
                st.pdf(load_pdf_page(pdf_path, page_num - 1), height=700)
            except Exception as exc:
                st.error(f"Failed to render PDF page: {exc}")

    summary_rows = []
    run_key_char_sets: dict[str, set[int]] = {}
    run_key_page_segments: dict[str, list[dict]] = {}

    for run_key in selected_run_keys:
        payload = doc_runs[selected_doc][run_key]
        segments = payload.get("segments", [])
        page_segments = _segmentation_segments_for_page(
            segments=segments,
            page_num=page_num,
            page_text=page_text,
            page_span=page_span,
        )
        run_key_page_segments[run_key] = page_segments
        run_key_char_sets[run_key] = _chars_for_spans(
            [(s["local_start"], s["local_end"]) for s in page_segments]
        )

        method, version = parse_run_key(run_key)
        runtime = runtime_by_run_key.get((selected_doc, run_key))

        summary_rows.append({
            "run_key": run_key,
            "method": method,
            "variant": version,
            "total_segments": len(segments),
            "segments_on_page": len(page_segments),
            "covered_chars_on_page": sum(
                s["local_end"] - s["local_start"] for s in page_segments
            ),
            "runtime_sec": runtime,
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    pairwise = []
    for a, b in itertools.combinations(selected_run_keys, 2):
        pairwise.append({
            "run_a": a,
            "run_b": b,
            "char_iou_on_page": round(_span_iou(run_key_char_sets[a], run_key_char_sets[b]), 4),
        })
    if pairwise:
        st.caption("Pairwise overlap between highlighted character spans on this page (IoU).")
        st.dataframe(pd.DataFrame(pairwise), use_container_width=True, hide_index=True)

    if lock_pane_scroll and len(selected_run_keys) > 1:
        pane_height = 520 if len(selected_run_keys) <= 3 else 430
        _render_synced_segmentation_panes(
            text=page_text,
            panes=[
                {
                    "run_key": run_key,
                    "segments": run_key_page_segments[run_key],
                }
                for run_key in selected_run_keys
            ],
            height=pane_height,
            lock_scroll=True,
        )

        for run_key in selected_run_keys:
            payload = doc_runs[selected_doc][run_key]
            page_segments = run_key_page_segments[run_key]
            st.markdown(f"**{run_key}**")
            if page_segments:
                table_rows = []
                for seg in page_segments:
                    preview = seg["text"]
                    if len(preview) > 180:
                        preview = preview[:180] + "..."
                    table_rows.append({
                        "parent": seg["parent"],
                        "title": seg["title"],
                        "span": f'{seg["local_start"]}-{seg["local_end"]}',
                        "text": preview,
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=220)
            else:
                st.caption("No segments on this page.")
            with st.expander(f"{run_key} hierarchy plan", expanded=False):
                st.json(payload.get("hierarchy_plan", {}))
        return

    if len(selected_run_keys) <= 3:
        cols = st.columns(len(selected_run_keys), gap="medium")
        for col, run_key in zip(cols, selected_run_keys):
            with col:
                payload = doc_runs[selected_doc][run_key]
                page_segments = run_key_page_segments[run_key]
                st.markdown(f"**{run_key}**")
                st.caption(f"{len(page_segments)} segments on this page")
                _render_segmentation_highlights(page_text, page_segments, height=520)
                if page_segments:
                    table_rows = []
                    for seg in page_segments:
                        preview = seg["text"]
                        if len(preview) > 140:
                            preview = preview[:140] + "..."
                        table_rows.append({
                            "parent": seg["parent"],
                            "title": seg["title"],
                            "span": f'{seg["local_start"]}-{seg["local_end"]}',
                            "text": preview,
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=240)
                else:
                    st.caption("No segments on this page.")
                with st.expander("Hierarchy plan", expanded=False):
                    st.json(payload.get("hierarchy_plan", {}))
    else:
        for run_key in selected_run_keys:
            payload = doc_runs[selected_doc][run_key]
            page_segments = run_key_page_segments[run_key]
            st.markdown(f"**{run_key}**")
            st.caption(f"{len(page_segments)} segments on this page")
            _render_segmentation_highlights(page_text, page_segments, height=430)
            if page_segments:
                table_rows = []
                for seg in page_segments:
                    preview = seg["text"]
                    if len(preview) > 180:
                        preview = preview[:180] + "..."
                    table_rows.append({
                        "parent": seg["parent"],
                        "title": seg["title"],
                        "span": f'{seg["local_start"]}-{seg["local_end"]}',
                        "text": preview,
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=220)
            else:
                st.caption("No segments on this page.")
            with st.expander(f"{run_key} hierarchy plan", expanded=False):
                st.json(payload.get("hierarchy_plan", {}))


def main():
    """Launch the experiment-oriented review dashboard."""
    st.set_page_config(page_title="CBA Review UI", layout="wide")
    st.title("CBA Review UI")

    required_password = st.secrets.get("REVIEW_UI_PASSWORD", None)
    if required_password:
        if "auth_ok" not in st.session_state:
            st.session_state.auth_ok = False
        if not st.session_state.auth_ok:
            st.sidebar.subheader("Login")
            pwd = st.sidebar.text_input("Password", type="password")
            if pwd == required_password:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.warning("Enter password to continue.")
                return

    page_choice = st.sidebar.radio(
        "View",
        [
            "OCR Viewer",
            "OCR Comparison",
            "Clause Extraction",
            "Provision Identification",
            "Segmentation",
            "Stats",
            "Heatmap",
        ],
    )

    if page_choice == "OCR Viewer":
        render_ocr_viewer()
        return

    if page_choice == "OCR Comparison":
        render_ocr_comparison()
        return

    if page_choice == "Clause Extraction":
        render_clause_extraction_comparison()
        return

    if page_choice == "Provision Identification":
        render_provision_identification_comparison()
        return

    if page_choice == "Segmentation":
        render_segmentation_review()
        return

    features_path = DEFAULT_FEATURES_OUTPUT.expanduser().resolve()
    feature_rows = load_feature_rows(features_path)
    if not feature_rows:
        st.warning("No extracted feature rows found.")
        return

    meta_path = DEFAULT_META_DTA.expanduser().resolve()
    meta_df = load_metadata(meta_path)

    if page_choice == "Heatmap":
        st.subheader("Provision Heatmap")
        if meta_df.empty:
            st.info("Metadata DTA not found or empty.")
            return

        level = st.radio("Level", ["Page", "Document"], horizontal=True)
        dim = st.selectbox(
            "Group by",
            ["naics", "type", "statefips1", "union", "expire_year"],
        )

        feat_df = pd.DataFrame(feature_rows)
        if level == "Document":
            feat_df = feat_df.drop_duplicates(["document_id", "feature_name"])

        if dim not in meta_df.columns:
            st.warning(f"Column not found in metadata: {dim}")
            return

        merged = feat_df.merge(meta_df[["document_id", dim, "employername"]], on="document_id", how="left")
        merged = merged.dropna(subset=[dim, "feature_name"])

        counts = (
            merged.groupby([dim, "feature_name"])
            .size()
            .reset_index(name="count")
        )

        if level == "Document":
            denom = (
                merged.drop_duplicates(["document_id", dim])
                .groupby(dim)
                .size()
                .reset_index(name="denom")
            )
        else:
            denom = (
                merged.drop_duplicates(["document_id", "document_page", dim])
                .groupby(dim)
                .size()
                .reset_index(name="denom")
            )

        counts = counts.merge(denom, on=dim, how="left")
        counts["percent"] = (counts["count"] / counts["denom"] * 100.0).round(2)

        if counts.empty:
            st.info("No data to display after join.")
            return

        heat = (
            alt.Chart(counts)
            .mark_rect()
            .encode(
                x=alt.X("feature_name:N", title="Provision"),
                y=alt.Y(f"{dim}:N", title=dim),
                color=alt.Color("percent:Q", title=f"% of {'documents' if level=='Document' else 'pages'}"),
                tooltip=[dim, "feature_name", "count", "percent"],
            )
        )
        st.altair_chart(heat, use_container_width=True)

        st.subheader("Companies by Partition")
        companies = (
            merged.dropna(subset=["employername"])
            .groupby(dim)["employername"]
            .apply(lambda s: sorted(set(s)))
            .reset_index()
        )
        for _, row in companies.iterrows():
            company_list = row["employername"]
            st.write(f"**{row[dim]} ({len(company_list)})**")
            st.write("; ".join(company_list[:50]))
        return

    st.subheader("Provision Counts")
    page_counts = Counter()
    doc_sets = defaultdict(set)
    for row in feature_rows:
        feat = row["feature_name"]
        page_counts[feat] += 1
        doc_sets[feat].add(row["document_id"])

    page_rows = [
        {"feature_name": feat, "count": cnt}
        for feat, cnt in page_counts.most_common()
    ]
    doc_rows = sorted(
        [{"feature_name": feat, "count": len(doc_sets[feat])} for feat in doc_sets.keys()],
        key=lambda r: r["count"],
        reverse=True,
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.subheader("Page-Level Frequency")
        page_df = pd.DataFrame(page_rows)
        page_df["feature_short"] = page_df["feature_name"].apply(lambda s: first_n_chars(s, 30))
        page_chart_height = max(550, len(page_df) * 24)
        try:
            page_chart = (
                alt.Chart(page_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y(
                        "feature_short:N",
                        sort="-x",
                        title="Feature (first 30 chars)",
                        axis=alt.Axis(labelLimit=420),
                    ),
                    tooltip=["feature_name", "count"],
                )
                .properties(height=page_chart_height)
            )
            st.altair_chart(page_chart, use_container_width=True)
        except Exception:
            st.bar_chart(page_df.set_index("feature_short")["count"])
        st.dataframe(page_df, use_container_width=True)
    with col_b:
        st.subheader("Document-Level Frequency")
        doc_df = pd.DataFrame(doc_rows)
        doc_df["feature_short"] = doc_df["feature_name"].apply(lambda s: first_n_chars(s, 30))
        doc_chart_height = max(550, len(doc_df) * 24)
        try:
            doc_chart = (
                alt.Chart(doc_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y(
                        "feature_short:N",
                        sort="-x",
                        title="Feature (first 30 chars)",
                        axis=alt.Axis(labelLimit=420),
                    ),
                    tooltip=["feature_name", "count"],
                )
                .properties(height=doc_chart_height)
            )
            st.altair_chart(doc_chart, use_container_width=True)
        except Exception:
            st.bar_chart(doc_df.set_index("feature_short")["count"])
        st.dataframe(doc_df, use_container_width=True)
    return


if __name__ == "__main__":
    main()
