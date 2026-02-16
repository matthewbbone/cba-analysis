import base64
import io
import json
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

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DOC_DIR = APP_DIR.parent / "processed_cbas"
DEFAULT_FEATURES_OUTPUT = APP_DIR.parent / "outputs" / "cba_features_annotated.jsonl"
DEFAULT_META_DTA = APP_DIR.parent / "dol_archive" / "CBAList_with_statefips.dta"
DEFAULT_OCR_DIR = APP_DIR.parent / "ocr_output"
DEFAULT_PDF_DIR = APP_DIR.parent / "dol_archive"
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
    """Build document_id -> 'Union v Employer' display labels from metadata."""
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
        union_name = _safe_label_value(row[union_col], "Union") if union_col else "Union"
        employer_name = _safe_label_value(row[employer_col], "Employer") if employer_col else "Employer"
        labels[doc_id] = f"{union_name} v {employer_name}"
    return labels


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
    ocr_dir = Path(st.sidebar.text_input("OCR output folder", str(DEFAULT_OCR_DIR))).expanduser().resolve()
    pdf_dir = Path(st.sidebar.text_input("Source PDF folder", str(DEFAULT_PDF_DIR))).expanduser().resolve()
    meta_path = Path(st.sidebar.text_input("Metadata DTA", str(DEFAULT_META_DTA))).expanduser().resolve()
    annotated_jsonl = Path(
        st.sidebar.text_input("Annotated JSONL", str(DEFAULT_ANNOTATED_JSONL))
    ).expanduser().resolve()

    if not ocr_dir.exists():
        st.error(f"OCR output folder not found: {ocr_dir}")
        return

    ocr_docs = list_ocr_documents(ocr_dir)
    if not ocr_docs:
        st.warning("No OCR results found. Run the OCR pipeline first.")
        return

    meta_df = load_metadata(meta_path)
    display_map = build_doc_display_map(meta_df)

    selected_doc = st.sidebar.selectbox(
        "Document",
        ocr_docs,
        key="ocr_doc",
        format_func=lambda p: display_map.get(p.name, "Union v Employer"),
    )
    selected = selected_doc.name
    doc_dir = ocr_dir / selected

    page_files = list_page_files(doc_dir)
    num_pages = len(page_files)

    if num_pages == 0:
        st.warning("No pages found for this document.")
        return

    page_idx = st.sidebar.number_input("Page", min_value=1, max_value=num_pages, value=1, step=1, key="ocr_page") - 1

    # Metadata summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Pages processed:** {len(page_files)}")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader(f"{selected} — Page {page_idx + 1} of {num_pages}")
        pdf_path = pdf_dir / f"{selected}.pdf"
        if pdf_path.exists():
            try:
                pdf_bytes = load_pdf_page(pdf_path, page_idx)
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64}" '
                    f'width="100%" height="800" type="application/pdf"></iframe>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Failed to render PDF page: {exc}")
        else:
            st.info(f"Source PDF not found: {pdf_path.name}")

    with col_right:
        st.subheader("OCR Text with Clause Highlights")

        # Get page text from individual page files
        page_text = ""
        if page_idx < len(page_files):
            page_text = page_files[page_idx].read_text(encoding="utf-8")

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


def main():
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

    page_choice = st.sidebar.radio("View", ["OCR Viewer", "Stats", "Heatmap"])

    if page_choice == "OCR Viewer":
        render_ocr_viewer()
        return

    features_path = Path(
        st.sidebar.text_input("Extraction output (CSV/JSONL)", str(DEFAULT_FEATURES_OUTPUT))
    ).expanduser().resolve()
    feature_rows = load_feature_rows(features_path)
    if not feature_rows:
        st.warning(f"No extracted feature rows found in: {features_path}")
        return

    meta_path = st.sidebar.text_input("Metadata DTA", str(DEFAULT_META_DTA))
    meta_path = Path(meta_path).expanduser().resolve()
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
            st.write(", ".join(company_list[:50]))
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
        try:
            page_chart = (
                alt.Chart(page_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y("feature_name:N", sort="-x", title="Feature"),
                    tooltip=["feature_name", "count"],
                )
            )
            st.altair_chart(page_chart, use_container_width=True)
        except Exception:
            st.bar_chart(page_df.set_index("feature_name")["count"])
        st.dataframe(page_df, use_container_width=True)
    with col_b:
        st.subheader("Document-Level Frequency")
        doc_df = pd.DataFrame(doc_rows)
        try:
            doc_chart = (
                alt.Chart(doc_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y("feature_name:N", sort="-x", title="Feature"),
                    tooltip=["feature_name", "count"],
                )
            )
            st.altair_chart(doc_chart, use_container_width=True)
        except Exception:
            st.bar_chart(doc_df.set_index("feature_name")["count"])
        st.dataframe(doc_df, use_container_width=True)
    return


if __name__ == "__main__":
    main()
