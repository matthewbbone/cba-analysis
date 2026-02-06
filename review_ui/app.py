import io
from pathlib import Path
import datetime

import streamlit as st
import csv
from collections import Counter, defaultdict
import altair as alt
import pandas as pd
from pypdf import PdfReader, PdfWriter

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DOC_DIR = APP_DIR.parent / "processed_cbas"
DEFAULT_FEATURES_CSV = APP_DIR.parent / "outputs" / "cba_features.csv"
DEFAULT_META_DTA = APP_DIR.parent / "dol_archive" / "CBAList_with_statefips.dta"


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
def load_features(csv_path: Path):
    if not csv_path.exists():
        return {}
    features = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("document_id", "")
            page = row.get("document_page", "")
            feat = row.get("feature_name", "")
            if not doc_id or not page or not feat:
                continue
            key = (doc_id, int(page))
            features.setdefault(key, set()).add(feat)
    return features


@st.cache_data(show_spinner=False)
def load_feature_rows(csv_path: Path):
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("document_id", "")
            page = row.get("document_page", "")
            feat = row.get("feature_name", "")
            if not doc_id or not page or not feat:
                continue
            rows.append(
                {
                    "document_id": doc_id,
                    "document_page": int(page),
                    "feature_name": feat,
                }
            )
    return rows


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

    doc_dir = st.sidebar.text_input("Document folder", str(DEFAULT_DOC_DIR))
    doc_dir = Path(doc_dir).expanduser().resolve()
    features_csv = st.sidebar.text_input("Features CSV", str(DEFAULT_FEATURES_CSV))
    features_csv = Path(features_csv).expanduser().resolve()
    page_choice = st.sidebar.radio("View", ["Review", "Stats", "Heatmap"])
    features_map = load_features(features_csv)
    feature_rows = load_feature_rows(features_csv)
    meta_path = st.sidebar.text_input("Metadata DTA", str(DEFAULT_META_DTA))
    meta_path = Path(meta_path).expanduser().resolve()
    meta_df = load_metadata(meta_path)
    taxonomy_path = APP_DIR.parent / "references" / "feature_taxonomy_final.md"
    all_features = load_all_feature_names(taxonomy_path)

    if not doc_dir.exists():
        st.error(f"Folder not found: {doc_dir}")
        return

    docs = list_documents(doc_dir)
    if not docs:
        st.warning("No documents found. Expect files named document_*.pdf")
        return

    feature_doc_ids = {doc_id for (doc_id, _page) in features_map.keys()}
    docs = [p for p in docs if p.stem in feature_doc_ids]
    if not docs:
        st.warning("No documents found that match entries in cba_features.csv.")
        return

    doc_names = [p.name for p in docs]
    selected_doc = st.sidebar.selectbox("Document", doc_names)
    pdf_path = doc_dir / selected_doc

    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
    except Exception as exc:
        st.error(f"Failed to open PDF: {exc}")
        return

    page_index = st.sidebar.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
    page_number = int(page_index) - 1
    feature_key = (pdf_path.stem, page_index)
    detected_features = sorted(features_map.get(feature_key, []))

    if page_choice == "Heatmap":
        st.subheader("Provision Heatmap")
        if not feature_rows:
            st.info("No feature rows found in the CSV.")
            return
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

    if page_choice == "Stats":
        st.subheader("Provision Counts")
        if not feature_rows:
            st.info("No feature rows found in the CSV.")
            return

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

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.subheader(f"{selected_doc} — Page {page_index} of {page_count}")
        try:
            pdf_bytes = load_pdf_page(pdf_path, page_number)
            st.pdf(pdf_bytes, height="stretch")
        except Exception as exc:
            st.error(f"Failed to render page: {exc}")

    with col_right:
        st.subheader("Features")
        st.caption("List the features visible on this page.")
        review_rows = []
        if detected_features:
            st.write("Mark each detected feature as correct or incorrect:")
            for feat in detected_features:
                ok = st.checkbox(feat, value=True, key=f"feat_ok_{pdf_path.stem}_{page_index}_{feat}")
                review_rows.append(
                    {
                        "document_id": pdf_path.stem,
                        "document_page": page_index,
                        "feature_name": feat,
                        "label_action": "kept" if ok else "removed",
                    }
                )
        else:
            st.info("No detected features for this page.")
        unlabeled = [f for f in all_features if f not in detected_features]
        added = st.multiselect(
            "Add missing features",
            options=unlabeled,
            default=[],
            help="Select additional features present on this page but not labeled.",
        )
        for feat in added:
            review_rows.append(
                {
                    "document_id": pdf_path.stem,
                    "document_page": page_index,
                    "feature_name": feat,
                    "label_action": "added",
                }
            )
        notes = st.text_area("Notes", height=200, placeholder="Optional reviewer notes...")
        if notes.strip():
            review_rows.append(
                {
                    "document_id": pdf_path.stem,
                    "document_page": page_index,
                    "feature_name": "",
                    "label_action": "note",
                    "note": notes.strip(),
                }
            )

        if st.button("Save review"):
            out_path = APP_DIR.parent / "outputs" / "review.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = out_path.exists()
            fieldnames = ["timestamp", "document_id", "document_page", "feature_name", "label_action", "note"]
            with out_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                ts = datetime.datetime.now().isoformat(timespec="seconds")
                for row in review_rows:
                    writer.writerow(
                        {
                            "timestamp": ts,
                            "document_id": row.get("document_id", ""),
                            "document_page": row.get("document_page", ""),
                            "feature_name": row.get("feature_name", ""),
                            "label_action": row.get("label_action", ""),
                            "note": row.get("note", ""),
                        }
                    )
            st.success(f"Saved {len(review_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
