"""Primary Streamlit dashboard for inspecting pipeline outputs side by side.

Compared with `review_ui/app.py`, this UI is broader: it can browse
segmentation, classification, and all three generosity outputs, while also
falling back across multiple cache/output directory layouts used in the repo.
"""

import html
import io
import json
import os
import random
import re
import csv
from pathlib import Path
import numpy as np

from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
import streamlit as st
import streamlit.components.v1 as components

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR_ENV = os.environ.get("CACHE_DIR", "").strip()
DEFAULT_CACHE_DIR = Path(CACHE_DIR_ENV).expanduser() if CACHE_DIR_ENV else APP_DIR.parent
DEFAULT_COLLECTION = os.environ.get("SEGMENT_COLLECTION", "dol_archive")
DEFAULT_SEGMENTATION_DIR = DEFAULT_CACHE_DIR / "02_segmentation_output" / DEFAULT_COLLECTION
DEFAULT_OCR_DIR = DEFAULT_CACHE_DIR / "01_ocr_output" / DEFAULT_COLLECTION
DEFAULT_PDF_DIR = DEFAULT_CACHE_DIR / DEFAULT_COLLECTION
DEFAULT_CLUSTER_DIR = DEFAULT_CACHE_DIR / "03_clustering_output" / DEFAULT_COLLECTION
DEFAULT_CLAUSE_DIR = DEFAULT_CACHE_DIR / "03_classification_output" / DEFAULT_COLLECTION
DEFAULT_GENEROSITY_ASH_DIR = DEFAULT_CACHE_DIR / "04_generosity_ash_output" / DEFAULT_COLLECTION
DEFAULT_GENEROSITY_GAB_DIR = DEFAULT_CACHE_DIR / "04_generosity_gab_output" / DEFAULT_COLLECTION
DEFAULT_GENEROSITY_LLM_DIR = DEFAULT_CACHE_DIR / "04_generosity_llm_output" / DEFAULT_COLLECTION
GENEROSITY_RATIO_STATUS_COLORS = {
    "finite": "#3b82f6",
    "positive_infinity": "#16a34a",
    "negative_infinity": "#dc2626",
    "undefined": "#9ca3af",
}


def _resolve_clause_root(cache_dir: Path, collection: str) -> Path:
    """Find the first clause-classification directory that exists for the UI."""
    candidates = [
        cache_dir / "03_classification_output" / collection,
        cache_dir / "03_classification_output",
        cache_dir / "03_clause_extraction_output" / collection,
        cache_dir / "03_clause_extraction_output",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_generosity_ash_root(cache_dir: Path, collection: str) -> Path:
    """Support both cache-root and repo-local layouts for ASH outputs."""
    candidates = [
        cache_dir / "04_generosity_ash_output" / collection,
        cache_dir / "04_generosity_ash_output",
        cache_dir / "outputs" / "04_generosity_ash_output" / collection,
        cache_dir / "outputs" / "04_generosity_ash_output",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_generosity_gab_root(cache_dir: Path, collection: str) -> Path:
    """Support both cache-root and repo-local layouts for GAB outputs."""
    candidates = [
        cache_dir / "04_generosity_gab_output" / collection,
        cache_dir / "04_generosity_gab_output",
        cache_dir / "outputs" / "04_generosity_gab_output" / collection,
        cache_dir / "outputs" / "04_generosity_gab_output",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_generosity_llm_root(cache_dir: Path, collection: str) -> Path:
    """Support both cache-root and repo-local layouts for LLM generosity outputs."""
    candidates = [
        cache_dir / "04_generosity_llm_output" / collection,
        cache_dir / "04_generosity_llm_output",
        cache_dir / "outputs" / "04_generosity_llm_output" / collection,
        cache_dir / "outputs" / "04_generosity_llm_output",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@st.cache_data(show_spinner=False)
def _safe_read_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_pdf_page(pdf_path: Path, page_number: int) -> bytes:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number])
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _doc_sort_key(doc_id: str):
    m = re.fullmatch(r"document_(\d+)", doc_id)
    if not m:
        return (10**9, doc_id)
    return (int(m.group(1)), doc_id)


@st.cache_data(show_spinner=False)
def _list_ocr_doc_ids(root: Path):
    if not root.exists() or not root.is_dir():
        return []
    doc_ids = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if re.fullmatch(r"document_\d+", p.name) is None:
            continue
        if list(p.glob("page_*.txt")):
            doc_ids.append(p.name)
    return sorted(doc_ids, key=_doc_sort_key)


@st.cache_data(show_spinner=False)
def _list_seg_doc_ids(root: Path):
    if not root.exists() or not root.is_dir():
        return []
    doc_ids = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if re.fullmatch(r"document_\d+", p.name) is None:
            continue
        if (p / "document_meta.json").exists():
            doc_ids.append(p.name)
    return sorted(doc_ids, key=_doc_sort_key)


@st.cache_data(show_spinner=False)
def _sorted_page_paths(doc_dir: Path):
    pages = list(doc_dir.glob("page_*.txt"))

    def _page_num(p: Path):
        m = re.match(r"page_(\d+)\.txt$", p.name)
        if m:
            return int(m.group(1))
        return 10**9

    return sorted(pages, key=_page_num)


@st.cache_data(show_spinner=False)
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
    for pf in _sorted_page_paths(doc_dir):
        m = re.match(r"page_(\d+)\.txt$", pf.name)
        if not m:
            continue
        page_num = int(m.group(1))
        page_numbers.append(page_num)

        text = pf.read_text(encoding="utf-8", errors="replace")
        text_by_page[page_num] = text

        start = offset
        end = start + len(text)
        span_by_page[page_num] = (start, end)
        offset = end + 2

    return {
        "page_numbers": page_numbers,
        "text_by_page": text_by_page,
        "span_by_page": span_by_page,
    }


def _normalize_segments(meta_segments):
    out = []
    if not isinstance(meta_segments, dict):
        return out
    for k, v in meta_segments.items():
        if not isinstance(v, dict):
            continue
        span = v.get("span")
        if not (isinstance(span, list) or isinstance(span, tuple)) or len(span) != 3:
            continue
        try:
            number = int(k)
            start = int(span[0])
            end = int(span[1])
            length = int(span[2])
        except Exception:
            continue
        out.append({"number": number, "start": start, "end": end, "length": length})
    return sorted(out, key=lambda x: x["number"])


@st.cache_data(show_spinner=False)
def _load_doc_segments(seg_root: Path, doc_id: str):
    payload = _safe_read_json(seg_root / doc_id / "document_meta.json") or {}
    return _normalize_segments(payload.get("segments") or {})


def _segmentation_segments_for_page(
    segments: list[dict],
    page_text: str,
    page_span: tuple[int, int] | None,
) -> list[dict]:
    if not page_text or page_span is None:
        return []

    page_start, page_end = page_span
    page_len = len(page_text)
    out = []

    for seg in segments or []:
        start = seg.get("start_pos")
        end = seg.get("end_pos")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            continue
        if start >= page_end or end <= page_start:
            continue

        local_start = max(0, start - page_start)
        local_end = min(page_len, end - page_start)
        if local_end <= local_start:
            continue

        out.append(
            {
                "parent": str(seg.get("parent", "")).strip() or "Segment",
                "title": str(seg.get("title", "")).strip() or "Untitled",
                "local_start": int(local_start),
                "local_end": int(local_end),
            }
        )

    out.sort(key=lambda x: (x["local_start"], x["local_end"]))
    filtered = []
    last_end = -1
    for seg in out:
        if seg["local_start"] < last_end:
            continue
        filtered.append(seg)
        last_end = seg["local_end"]
    return filtered


def _render_segmentation_highlights(text: str, page_segments: list[dict], height: int = 800):
    if not text:
        st.info("No OCR text for this page.")
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


@st.cache_data(show_spinner=False)
def _list_clause_doc_ids(root: Path):
    if not root.exists() or not root.is_dir():
        return []
    doc_ids = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if re.fullmatch(r"document_\d+", p.name) is None:
            continue
        if list(p.glob("segment_*.json")):
            doc_ids.append(p.name)
    return sorted(doc_ids, key=_doc_sort_key)


@st.cache_data(show_spinner=False)
def _list_clause_segment_numbers(root: Path, doc_id: str):
    doc_dir = root / doc_id
    if not doc_dir.exists():
        return []
    nums = []
    for p in doc_dir.glob("segment_*.json"):
        m = re.match(r"segment_(\d+)\.json$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


@st.cache_data(show_spinner=False)
def _load_clause_segment_payload(root: Path, doc_id: str, segment_number: int):
    payload = _safe_read_json(root / doc_id / f"segment_{segment_number}.json")
    if isinstance(payload, dict):
        return payload
    return {}


@st.cache_data(show_spinner=False)
def _load_clause_classification_rows(clause_root: Path):
    rows = []
    doc_ids = _list_clause_doc_ids(clause_root)
    for doc_id in doc_ids:
        for segment_number in _list_clause_segment_numbers(clause_root, doc_id):
            payload = _load_clause_segment_payload(clause_root, doc_id, int(segment_number))
            if not isinstance(payload, dict):
                continue

            labels = payload.get("labels", [])
            label = "OTHER"
            if isinstance(labels, list) and labels:
                label = str(labels[0]).strip() or "OTHER"

            extraction = {}
            extractions = payload.get("extractions", [])
            if isinstance(extractions, list) and extractions and isinstance(extractions[0], dict):
                extraction = extractions[0]
                if label == "OTHER":
                    label = str(
                        extraction.get("label")
                        or extraction.get("feature_name")
                        or extraction.get("raw_label")
                        or "OTHER"
                    ).strip() or "OTHER"

            reason = str(extraction.get("reason") or payload.get("reason") or "").strip()
            top_candidates = payload.get("top_candidates")
            if not isinstance(top_candidates, list):
                top_candidates = extraction.get("top_candidates")
            if not isinstance(top_candidates, list):
                top_candidates = []

            normalized_candidates = []
            for candidate in top_candidates:
                if not isinstance(candidate, dict):
                    continue
                feature_name = str(
                    candidate.get("feature_name")
                    or candidate.get("label")
                    or candidate.get("raw_label")
                    or ""
                ).strip()
                if not feature_name:
                    continue
                similarity = candidate.get("similarity")
                try:
                    similarity = float(similarity) if similarity is not None else None
                except Exception:
                    similarity = None
                normalized_candidates.append(
                    {
                        "feature_name": feature_name,
                        "similarity": similarity,
                        "tldr": str(candidate.get("tldr") or "").strip(),
                        "description": str(candidate.get("description") or "").strip(),
                    }
                )

            candidate_k = payload.get("candidate_k")
            try:
                candidate_k = int(candidate_k) if candidate_k is not None else None
            except Exception:
                candidate_k = None

            rows.append(
                {
                    "document_id": doc_id,
                    "segment_number": int(segment_number),
                    "label": label,
                    "reason": reason,
                    "segment_text": str(payload.get("segment_text") or ""),
                    "top_candidates": normalized_candidates,
                    "model": str(payload.get("model") or "").strip(),
                    "provider": str(payload.get("provider") or "").strip(),
                    "embedding_model": str(payload.get("embedding_model") or "").strip(),
                    "candidate_k": candidate_k,
                }
            )
    return rows


@st.cache_data(show_spinner=False)
def _build_clause_heatmap(rows: list[dict]):
    doc_ids = sorted({str(row["document_id"]) for row in rows}, key=_doc_sort_key)
    labels = sorted({str(row["label"]) for row in rows})
    if not doc_ids or not labels:
        return doc_ids, labels, np.zeros((0, 0))

    doc_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    label_index = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(doc_ids), len(labels)), dtype=float)

    for row in rows:
        di = doc_index.get(str(row["document_id"]))
        li = label_index.get(str(row["label"]))
        if di is None or li is None:
            continue
        counts[di, li] += 1.0

    # Order clauses from most common to least common across all documents.
    label_totals = counts.sum(axis=0)
    order = np.argsort(-label_totals)
    counts = counts[:, order]
    labels = [labels[i] for i in order]

    return doc_ids, labels, counts


def _render_clause_heatmap(rows: list[dict]):
    doc_ids, labels, heatmap = _build_clause_heatmap(rows)
    if heatmap.size == 0:
        st.info("No clause classification heatmap data available.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        st.info("matplotlib not available for heatmap rendering.")
        return

    fig_h = min(14.0, max(4.0, len(doc_ids) / 180))
    fig_w = min(18.0, max(7.0, len(labels) / 1.8))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="YlGnBu")

    ax.set_xlabel("Clause Type")
    ax.set_ylabel("Document")
    ax.set_title("Clause Type Counts by Document")

    x_step = max(1, len(labels) // 20)
    x_ticks = np.arange(0, len(labels), x_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([labels[i] for i in x_ticks], rotation=45, ha="right", fontsize=8)

    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Segment count")
    st.pyplot(fig, use_container_width=True, clear_figure=True)


def _render_classification_segment_details(selected_row: dict, key_prefix: str):
    st.markdown(f"**Final clause type:** `{selected_row['label']}`")

    meta = []
    if selected_row["model"]:
        meta.append(f"LLM model: `{selected_row['model']}`")
    if selected_row["embedding_model"]:
        meta.append(f"Embedding model: `{selected_row['embedding_model']}`")
    if selected_row["candidate_k"] is not None:
        meta.append(f"Top-k: `{selected_row['candidate_k']}`")
    if selected_row["provider"]:
        meta.append(f"Provider: `{selected_row['provider']}`")
    if meta:
        st.caption(" | ".join(meta))

    if selected_row["reason"]:
        st.markdown(f"**Decision rationale:** {selected_row['reason']}")
    else:
        st.info("No decision rationale returned for this segment.")

    st.markdown("#### Segment Text")
    st.text_area(
        "segment_text",
        value=selected_row["segment_text"],
        height=360,
        key=f"{key_prefix}_segment_text",
        label_visibility="collapsed",
    )

    candidates = selected_row["top_candidates"] if isinstance(selected_row["top_candidates"], list) else []
    if candidates:
        candidates = sorted(
            candidates,
            key=lambda c: c["similarity"] if isinstance(c.get("similarity"), float) else float("-inf"),
            reverse=True,
        )
    st.markdown("#### Retrieval Candidates")
    if not candidates:
        st.info("No embedding retrieval candidates found in this segment payload.")
    else:
        for idx, candidate in enumerate(candidates, start=1):
            feature_name = str(candidate.get("feature_name") or "UNKNOWN")
            similarity = candidate.get("similarity")
            similarity_text = f"{similarity:.4f}" if isinstance(similarity, float) else "n/a"
            with st.expander(f"{idx}. {feature_name} · similarity={similarity_text}", expanded=(idx == 1)):
                tldr = str(candidate.get("tldr") or "").strip()
                if tldr:
                    st.write(f"**TLDR:** {tldr}")
                description = str(candidate.get("description") or "").strip()
                if description:
                    st.write(description)


def _render_clause_classification_view(clause_root: Path):
    if not clause_root.exists():
        st.error(f"Clause classification output folder not found: {clause_root}")
        return

    rows = _load_clause_classification_rows(clause_root)
    if not rows:
        st.warning("No clause classification segment JSON files found.")
        return

    st.markdown("### Clause Types Across Documents")
    _render_clause_heatmap(rows)

    all_labels = sorted({row["label"] for row in rows})
    st.markdown("### Random Clause Sample by Type")
    selected_label = st.selectbox("Clause type", options=all_labels, key="app3_clause_label_pick")

    label_rows = [row for row in rows if row["label"] == selected_label]
    if not label_rows:
        st.warning("No segments found for the selected clause type.")
        return

    label_doc_count = len({row["document_id"] for row in label_rows})
    st.caption(
        f"{len(label_rows)} segment(s) labeled `{selected_label}` across {label_doc_count} document(s)."
    )

    seed_key = f"app3_clause_random_seed_{selected_label}"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = 0
    if st.button("Randomly select segment", key="app3_clause_randomize"):
        st.session_state[seed_key] += 1

    rng = random.Random(f"{selected_label}:{st.session_state[seed_key]}")
    selected_row = rng.choice(label_rows)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Documents indexed:** {len({row['document_id'] for row in rows})}")
    st.sidebar.markdown(f"**Total segments indexed:** {len(rows)}")
    st.sidebar.markdown(f"**Clause types indexed:** {len(all_labels)}")
    st.sidebar.markdown(f"**Segments for `{selected_label}`:** {len(label_rows)}")

    st.subheader(
        f"Sampled Segment — {selected_row['document_id']} / segment_{selected_row['segment_number']}"
    )
    _render_classification_segment_details(
        selected_row=selected_row,
        key_prefix=f"app3_clause_sample_{selected_label}_{st.session_state[seed_key]}",
    )

    st.markdown("#### All Matching Segments")
    matching_rows = sorted(
        label_rows,
        key=lambda row: (_doc_sort_key(row["document_id"]), row["segment_number"]),
    )
    table_rows = [
        {
            "document": row["document_id"],
            "segment": row["segment_number"],
            "reason": row["reason"],
        }
        for row in matching_rows
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def _load_generosity_summary(generosity_root: Path):
    payload = _safe_read_json(generosity_root / "summary.json")
    if isinstance(payload, dict):
        return payload
    return {}


@st.cache_data(show_spinner=False)
def _list_generosity_doc_ids(generosity_root: Path):
    if not generosity_root.exists() or not generosity_root.is_dir():
        return []
    doc_ids = []
    for path in generosity_root.iterdir():
        if not path.is_dir():
            continue
        if re.fullmatch(r"document_\d+", path.name) is None:
            continue
        if (path / "document_summary.json").exists() or list(path.glob("segment_*.json")):
            doc_ids.append(path.name)
    return sorted(doc_ids, key=_doc_sort_key)


@st.cache_data(show_spinner=False)
def _load_generosity_document_summary(generosity_root: Path, doc_id: str):
    payload = _safe_read_json(generosity_root / doc_id / "document_summary.json")
    if isinstance(payload, dict):
        return payload
    return {}


@st.cache_data(show_spinner=False)
def _list_generosity_segment_numbers(generosity_root: Path, doc_id: str):
    doc_dir = generosity_root / doc_id
    if not doc_dir.exists():
        return []
    nums = []
    for path in doc_dir.glob("segment_*.json"):
        m = re.match(r"segment_(\d+)\.json$", path.name)
        if not m:
            continue
        nums.append(int(m.group(1)))
    return sorted(set(nums))


@st.cache_data(show_spinner=False)
def _load_generosity_segment_payload(generosity_root: Path, doc_id: str, segment_number: int):
    payload = _safe_read_json(generosity_root / doc_id / f"segment_{segment_number}.json")
    if isinstance(payload, dict):
        return payload
    return {}


def _count_rows(counts):
    if not isinstance(counts, dict):
        return []
    rows = []
    for label, raw_count in counts.items():
        try:
            count = int(raw_count)
        except Exception:
            continue
        rows.append({"label": str(label), "count": count})
    rows.sort(key=lambda row: (-row["count"], row["label"]))
    return rows


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _format_ratio_value(ratio, ratio_status: str) -> str:
    status = str(ratio_status or "undefined")
    if status == "positive_infinity":
        return "+inf"
    if status == "negative_infinity":
        return "-inf"
    if status == "undefined":
        return "undefined"
    if isinstance(ratio, (int, float)):
        return f"{float(ratio):.4f}"
    return "n/a"


def _format_composite_score_value(score) -> str:
    if isinstance(score, (int, float)):
        numeric = float(score)
        if not np.isnan(numeric):
            return f"{numeric:.4f}"
    return "n/a"


@st.cache_data(show_spinner=False)
def _load_generosity_clause_type_rankings_file(generosity_root: Path):
    preferred = _safe_read_json(generosity_root / "clause_type_document_ratio_rankings.json")
    if isinstance(preferred, dict):
        return preferred
    legacy = _safe_read_json(generosity_root / "document_clause_type_ratio_rankings.json")
    if isinstance(legacy, dict):
        return legacy
    return {}


def _normalize_clause_type_document_rankings(raw_rankings):
    if not isinstance(raw_rankings, dict):
        return {}

    status_order = {
        "positive_infinity": 3,
        "finite": 2,
        "undefined": 1,
        "negative_infinity": 0,
    }
    out: dict[str, list[dict]] = {}
    for raw_clause_type, raw_rows in raw_rankings.items():
        clause_type = str(raw_clause_type).strip()
        if not clause_type:
            continue
        if not isinstance(raw_rows, list):
            continue

        rows: list[dict] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue
            document_id = str(raw_row.get("document_id", "")).strip()
            if not document_id:
                continue

            ratio = _to_float_or_none(raw_row.get("worker_over_firm_ratio"))
            ratio_status = str(raw_row.get("ratio_status", "undefined")).strip() or "undefined"
            worker_benefit_total = _to_int(raw_row.get("worker_benefit_total", raw_row.get("worker_benefit", 0)))
            firm_benefit_total = _to_int(raw_row.get("firm_benefit_total", raw_row.get("firm_benefit", 0)))

            rows.append(
                {
                    "rank": _to_int(raw_row.get("rank", 0)),
                    "document_id": document_id,
                    "clause_type": clause_type,
                    "segment_count": _to_int(raw_row.get("segment_count", 0)),
                    "worker_rights_total": _to_int(raw_row.get("worker_rights_total", 0)),
                    "worker_permissions_total": _to_int(raw_row.get("worker_permissions_total", 0)),
                    "worker_prohibitions_total": _to_int(raw_row.get("worker_prohibitions_total", 0)),
                    "worker_obligations_total": _to_int(raw_row.get("worker_obligations_total", 0)),
                    "worker_benefit_total": worker_benefit_total,
                    "firm_rights_total": _to_int(raw_row.get("firm_rights_total", 0)),
                    "firm_permissions_total": _to_int(raw_row.get("firm_permissions_total", 0)),
                    "firm_prohibitions_total": _to_int(raw_row.get("firm_prohibitions_total", 0)),
                    "firm_obligations_total": _to_int(raw_row.get("firm_obligations_total", 0)),
                    "firm_benefit_total": firm_benefit_total,
                    "worker_over_firm_ratio": ratio,
                    "ratio_status": ratio_status,
                    "avg_finite_segment_ratio": _to_float_or_none(raw_row.get("avg_finite_segment_ratio")),
                    "finite_segment_ratio_count": _to_int(raw_row.get("finite_segment_ratio_count", 0)),
                    "segment_ratio_status_counts": raw_row.get("segment_ratio_status_counts", {}),
                    "composite_clause_score": _to_float_or_none(raw_row.get("composite_clause_score")),
                    "composite_clause_type_count": _to_int(raw_row.get("composite_clause_type_count", 0)),
                }
            )

        if any(_to_int(row.get("rank", 0)) > 0 for row in rows):
            rows.sort(key=lambda row: (_to_int(row.get("rank", 0)), _doc_sort_key(row.get("document_id", ""))))
        else:
            rows.sort(
                key=lambda row: (
                    status_order.get(row.get("ratio_status", "undefined"), 1),
                    row.get("worker_over_firm_ratio") if row.get("worker_over_firm_ratio") is not None else 0.0,
                    row.get("worker_benefit_total", 0),
                    -row.get("firm_benefit_total", 0),
                    _doc_sort_key(row.get("document_id", "")),
                ),
                reverse=True,
            )

        # Enforce contiguous rank values for display safety.
        for idx, row in enumerate(rows, start=1):
            row["rank"] = idx
        out[clause_type] = rows
    return out


def _resolve_clause_type_document_rankings(summary, generosity_root: Path):
    rankings = {}
    if isinstance(summary, dict):
        summary_rankings = summary.get("clause_type_document_ratio_rankings")
        if isinstance(summary_rankings, dict):
            rankings = summary_rankings
        elif isinstance(summary.get("document_clause_type_ratio_rankings"), dict):
            rankings = summary.get("document_clause_type_ratio_rankings")
    if not rankings:
        rankings = _load_generosity_clause_type_rankings_file(generosity_root)
    return _normalize_clause_type_document_rankings(rankings)


def _normalize_document_composite_clause_scores(raw_scores):
    out: dict[str, dict[str, float | int | None]] = {}
    if not isinstance(raw_scores, dict):
        return out

    for raw_doc_id, raw_payload in raw_scores.items():
        document_id = str(raw_doc_id).strip()
        if not document_id:
            continue

        score = None
        count = 0
        if isinstance(raw_payload, dict):
            score = _to_float_or_none(raw_payload.get("composite_clause_score"))
            count = max(0, _to_int(raw_payload.get("composite_clause_type_count", 0)))
        else:
            score = _to_float_or_none(raw_payload)

        if isinstance(score, (int, float)) and np.isnan(float(score)):
            score = None
        out[document_id] = {
            "composite_clause_score": float(score) if isinstance(score, (int, float)) else None,
            "composite_clause_type_count": count,
        }
    return out


def _resolve_document_composite_clause_scores(summary, clause_type_rankings: dict[str, list[dict]]):
    raw_scores = summary.get("document_composite_clause_scores", {}) if isinstance(summary, dict) else {}
    out = _normalize_document_composite_clause_scores(raw_scores)

    # Backfill from clause-ranking rows when summary-level composite map is missing.
    for ranking_rows in clause_type_rankings.values():
        if not isinstance(ranking_rows, list):
            continue
        for row in ranking_rows:
            if not isinstance(row, dict):
                continue
            document_id = str(row.get("document_id", "")).strip()
            if not document_id:
                continue
            score = _to_float_or_none(row.get("composite_clause_score"))
            if isinstance(score, (int, float)) and np.isnan(float(score)):
                score = None
            count = max(0, _to_int(row.get("composite_clause_type_count", 0)))

            payload = out.setdefault(
                document_id,
                {"composite_clause_score": None, "composite_clause_type_count": 0},
            )
            if isinstance(score, (int, float)):
                payload["composite_clause_score"] = float(score)
            payload["composite_clause_type_count"] = max(
                _to_int(payload.get("composite_clause_type_count", 0)),
                count,
            )
    return out


def _build_composite_document_rows(document_composite_clause_scores: dict[str, dict[str, float | int | None]]):
    rows = []
    for document_id, payload in document_composite_clause_scores.items():
        score = payload.get("composite_clause_score") if isinstance(payload, dict) else None
        if not isinstance(score, (int, float)):
            continue
        numeric_score = float(score)
        if np.isnan(numeric_score):
            continue
        rows.append(
            {
                "document_id": str(document_id),
                "composite_clause_score": numeric_score,
                "composite_clause_type_count": max(
                    0,
                    _to_int(payload.get("composite_clause_type_count", 0)),
                ),
            }
        )

    rows.sort(key=lambda row: (-float(row["composite_clause_score"]), _doc_sort_key(row["document_id"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _cbafile_to_document_id(cbafile_value) -> str:
    if cbafile_value is None:
        return ""
    if isinstance(cbafile_value, (int, np.integer)):
        return f"document_{int(cbafile_value)}"
    if isinstance(cbafile_value, (float, np.floating)):
        if np.isnan(float(cbafile_value)):
            return ""
        return f"document_{int(cbafile_value)}"

    text = str(cbafile_value).strip()
    if not text:
        return ""
    match = re.search(r"(\d+)", text)
    if not match:
        return ""
    try:
        return f"document_{int(match.group(1))}"
    except Exception:
        return ""


def _document_display_name(document_id: str, document_name_lookup: dict[str, str]) -> str:
    doc_id = str(document_id).strip()
    if not doc_id:
        return ""
    firm_name = str(document_name_lookup.get(doc_id, "")).strip()
    return firm_name or doc_id


def _document_axis_labels(ranking_rows: list[dict], document_name_lookup: dict[str, str]) -> list[str]:
    base_labels = []
    counts: dict[str, int] = {}
    for row in ranking_rows:
        doc_id = str(row.get("document_id", "")).strip()
        base = _document_display_name(doc_id, document_name_lookup)
        base_labels.append(base)
        counts[base] = counts.get(base, 0) + 1

    labels = []
    for row, base in zip(ranking_rows, base_labels):
        doc_id = str(row.get("document_id", "")).strip()
        if counts.get(base, 0) > 1 and base != doc_id:
            labels.append(f"{base} ({doc_id})")
        else:
            labels.append(base)
    return labels


@st.cache_data(show_spinner=False)
def _load_document_firm_name_lookup(cba_list_path: Path):
    if not cba_list_path.exists():
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}

    try:
        frame = pd.read_stata(str(cba_list_path), convert_categoricals=False)
    except Exception:
        return {}

    if "cbafile" not in frame.columns or "employername" not in frame.columns:
        return {}

    out: dict[str, str] = {}
    for row in frame[["cbafile", "employername"]].to_dict(orient="records"):
        doc_id = _cbafile_to_document_id(row.get("cbafile"))
        if not doc_id:
            continue
        firm_name = str(row.get("employername", "")).strip()
        if not firm_name:
            continue
        out.setdefault(doc_id, firm_name)
    return out


def _render_clause_type_document_ratio_chart(
    clause_type: str,
    ranking_rows: list[dict],
    *,
    metric_mode: str = "ratio",
    document_name_lookup: dict[str, str] | None = None,
    composite_axis_label: str | None = None,
    composite_title_suffix: str | None = None,
):
    if not ranking_rows:
        st.info(f"No ranking rows available for `{clause_type}`.")
        return
    if metric_mode not in {"ratio", "composite"}:
        metric_mode = "ratio"
    if document_name_lookup is None:
        document_name_lookup = {}

    try:
        import matplotlib.pyplot as plt
    except Exception:
        st.info("matplotlib not available for clause-type generosity chart.")
        return

    plot_values = []
    label_values = []
    bar_colors = []
    was_clipped = False
    x_axis_label = "Worker/Firm generosity ratio"
    title_suffix = "Document Ranking"
    clip_caption = ""

    if metric_mode == "composite":
        finite_abs = [
            abs(float(row.get("composite_clause_score")))
            for row in ranking_rows
            if isinstance(row.get("composite_clause_score"), (int, float))
            and not np.isnan(float(row.get("composite_clause_score")))
        ]
        clip_val = max(1.0, float(np.percentile(finite_abs, 95))) * 1.15 if finite_abs else 1.0

        for row in ranking_rows:
            score = row.get("composite_clause_score")
            if not isinstance(score, (int, float)):
                plot_values.append(0.0)
                label_values.append("n/a")
                bar_colors.append("#9ca3af")
                continue
            numeric_score = float(score)
            if np.isnan(numeric_score):
                plot_values.append(0.0)
                label_values.append("n/a")
                bar_colors.append("#9ca3af")
                continue
            clipped_score = min(clip_val, max(-clip_val, numeric_score))
            if clipped_score != numeric_score:
                was_clipped = True
            plot_values.append(clipped_score)
            label_values.append(f"{numeric_score:.2f}")
            if numeric_score > 0:
                bar_colors.append("#2563eb")
            elif numeric_score < 0:
                bar_colors.append("#dc2626")
            else:
                bar_colors.append("#6b7280")

        x_axis_label = composite_axis_label or "Composite clause score (mean across clause types, NaNs excluded)"
        title_suffix = composite_title_suffix or "Composite Scores"
        clip_caption = f"Composite scores are clipped to +/-{clip_val:.2f} for chart readability."
    else:
        finite_abs = [
            abs(float(row["worker_over_firm_ratio"]))
            for row in ranking_rows
            if row.get("ratio_status") == "finite" and isinstance(row.get("worker_over_firm_ratio"), (int, float))
        ]
        finite_cap = max(1.0, float(np.percentile(finite_abs, 95))) if finite_abs else 1.0
        inf_plot_val = finite_cap * 1.15

        for row in ranking_rows:
            ratio_status = str(row.get("ratio_status", "undefined"))
            ratio = row.get("worker_over_firm_ratio")
            if ratio_status == "positive_infinity":
                plot_values.append(inf_plot_val)
                label_values.append("+inf")
            elif ratio_status == "negative_infinity":
                plot_values.append(-inf_plot_val)
                label_values.append("-inf")
            elif ratio_status == "undefined":
                plot_values.append(0.0)
                label_values.append("undef")
            else:
                numeric_ratio = float(ratio) if isinstance(ratio, (int, float)) else 0.0
                clipped_ratio = min(inf_plot_val, max(-inf_plot_val, numeric_ratio))
                if clipped_ratio != numeric_ratio:
                    was_clipped = True
                plot_values.append(clipped_ratio)
                label_values.append(f"{numeric_ratio:.2f}")
            bar_colors.append(GENEROSITY_RATIO_STATUS_COLORS.get(ratio_status, "#9ca3af"))

        clip_caption = (
            f"Finite ratios are clipped to +/-{inf_plot_val:.2f} for chart readability. "
            "Table below shows exact values."
        )

    y = np.arange(len(ranking_rows))
    fig_h = min(22.0, max(4.0, len(ranking_rows) * 0.42))
    fig, ax = plt.subplots(figsize=(12.8, fig_h))

    ax.barh(y, plot_values, color=bar_colors, edgecolor="white", linewidth=0.6)
    max_abs = max(1.0, max(abs(v) for v in plot_values))
    ax.axvline(0.0, color="#4b5563", linewidth=1.0)
    ax.set_xlim(-max_abs * 1.2, max_abs * 1.2)

    for i, plotted in enumerate(plot_values):
        offset = max_abs * 0.03
        if plotted >= 0:
            ax.text(plotted + offset, i, label_values[i], va="center", ha="left", fontsize=7)
        else:
            ax.text(plotted - offset, i, label_values[i], va="center", ha="right", fontsize=7)

    doc_labels = _document_axis_labels(ranking_rows, document_name_lookup)
    ax.set_yticks(y)
    ax.set_yticklabels(doc_labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Firm")
    ax.set_title(f"Clause Type: {clause_type} ({title_suffix})")
    st.pyplot(fig, use_container_width=True, clear_figure=True)

    if was_clipped and clip_caption:
        st.caption(clip_caption)


@st.cache_data(show_spinner=False)
def _load_generosity_segment_score(generosity_root: Path, doc_id: str, segment_number: int):
    csv_path = generosity_root / "segment_generosity_scores.csv"
    if not csv_path.exists():
        return {}

    doc_key = str(doc_id).strip()
    segment_key = str(int(segment_number))
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("document_id", "")).strip() != doc_key:
                    continue
                if str(row.get("segment_number", "")).strip() != segment_key:
                    continue
                ratio_raw = row.get("worker_over_firm_ratio")
                ratio = _to_float_or_none(ratio_raw)
                return {
                    "clause_type": str(row.get("clause_type", "")).strip() or "OTHER",
                    "worker_rights": _to_int(row.get("worker_rights", 0)),
                    "worker_permissions": _to_int(row.get("worker_permissions", 0)),
                    "worker_prohibitions": _to_int(row.get("worker_prohibitions", 0)),
                    "worker_obligations": _to_int(row.get("worker_obligations", 0)),
                    "worker_benefit": _to_int(row.get("worker_benefit", 0)),
                    "firm_rights": _to_int(row.get("firm_rights", 0)),
                    "firm_permissions": _to_int(row.get("firm_permissions", 0)),
                    "firm_prohibitions": _to_int(row.get("firm_prohibitions", 0)),
                    "firm_obligations": _to_int(row.get("firm_obligations", 0)),
                    "firm_benefit": _to_int(row.get("firm_benefit", 0)),
                    "worker_over_firm_ratio": ratio,
                    "ratio_status": str(row.get("ratio_status", "undefined")).strip() or "undefined",
                }
    except Exception:
        return {}
    return {}


@st.cache_data(show_spinner=False)
def _load_generosity_statement_rows_for_segment(
    generosity_root: Path,
    doc_id: str,
    segment_number: int,
    limit: int = 250,
):
    csv_path = generosity_root / "statement_rows.csv"
    out = {"rows": [], "truncated": False}
    if not csv_path.exists():
        return out

    rows = []
    doc_key = str(doc_id)
    segment_key = str(int(segment_number))

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("document_id", "")).strip() != doc_key:
                    continue
                if str(row.get("segment_number", "")).strip() != segment_key:
                    continue

                rows.append(
                    {
                        "sentence_index": str(row.get("sentence_index", "")).strip(),
                        "statement_num": str(row.get("statement_num", "")).strip(),
                        "subnorm": str(row.get("subnorm", "")).strip(),
                        "sentence_type": str(row.get("sentence_type", "")).strip(),
                        "modal": str(row.get("modal", "")).strip(),
                        "verb": str(row.get("verb", "")).strip(),
                        "obligation": str(row.get("obligation", "")).strip(),
                        "constraint": str(row.get("constraint", "")).strip(),
                        "permission": str(row.get("permission", "")).strip(),
                        "entitlement": str(row.get("entitlement", "")).strip(),
                        "full_sentence": str(row.get("full_sentence", "")).strip(),
                    }
                )

                if len(rows) >= max(1, int(limit)):
                    out["truncated"] = True
                    break
    except Exception:
        return {"rows": [], "truncated": False}

    out["rows"] = rows
    return out


def _render_generosity_ash_view(generosity_root: Path, cba_list_path: Path | None = None):
    if not generosity_root.exists():
        st.error(f"04_generosity_ash output folder not found: {generosity_root}")
        return

    summary = _load_generosity_summary(generosity_root)
    doc_ids = _list_generosity_doc_ids(generosity_root)
    clause_type_rankings = _resolve_clause_type_document_rankings(summary=summary, generosity_root=generosity_root)
    composite_clause_scores = _resolve_document_composite_clause_scores(summary, clause_type_rankings)
    document_name_lookup = (
        _load_document_firm_name_lookup(cba_list_path)
        if isinstance(cba_list_path, Path)
        else {}
    )

    st.markdown("### Run Summary")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", int(summary.get("documents_processed", 0) or 0))
        c2.metric("Segments", int(summary.get("segments_processed", 0) or 0))
        c3.metric("Sentences", int(summary.get("sentences_processed", 0) or 0))
        c4.metric("Statement rows", int(summary.get("statement_rows_written", 0) or 0))

        run_meta = []
        model = str(summary.get("model", "")).strip()
        if model:
            run_meta.append(f"spaCy model: `{model}`")
        if "include_tokens" in summary:
            run_meta.append(f"include_tokens: `{bool(summary.get('include_tokens'))}`")
        if run_meta:
            st.caption(" | ".join(run_meta))

        overall_segment_generosity = summary.get("overall_segment_generosity", {})
        if isinstance(overall_segment_generosity, dict) and overall_segment_generosity:
            g1, g2, g3 = st.columns(3)
            g1.metric("Worker benefit total", int(overall_segment_generosity.get("worker_benefit_total", 0) or 0))
            g2.metric("Firm benefit total", int(overall_segment_generosity.get("firm_benefit_total", 0) or 0))
            g3.metric(
                "Overall generosity ratio",
                _format_ratio_value(
                    overall_segment_generosity.get("worker_over_firm_ratio"),
                    overall_segment_generosity.get("ratio_status"),
                ),
            )

        t1, t2, t3 = st.columns(3)
        with t1:
            st.caption("Sentence Type Counts")
            sentence_rows = _count_rows(summary.get("sentence_type_counts"))
            if sentence_rows:
                st.dataframe(sentence_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No sentence type counts in summary.")
        with t2:
            st.caption("Auth Category Counts")
            auth_rows = _count_rows(summary.get("auth_category_counts"))
            if auth_rows:
                st.dataframe(auth_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No auth category counts in summary.")
        with t3:
            st.caption("Agent Counts")
            agent_rows = _count_rows(summary.get("agent_counts"))
            if agent_rows:
                st.dataframe(agent_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No agent counts in summary.")
    else:
        st.info("summary.json not found or unreadable in this output folder.")

    st.markdown("### Clause-Type Generosity Rankings")
    if clause_type_rankings:
        available_clause_types = sorted(clause_type_rankings.keys(), key=lambda label: (label == "OTHER", label.lower()))
        composite_option = "COMPOSITE_CLAUSE_SCORE"
        selected_clause_type = st.selectbox(
            "03 Classification Clause Type",
            options=[composite_option, *available_clause_types],
            format_func=lambda label: "Composite Score (All Clause Types)" if label == composite_option else label,
            key="app3_generosity_clause_type",
        )
        is_composite_view = selected_clause_type == composite_option
        selected_rows = (
            _build_composite_document_rows(composite_clause_scores)
            if is_composite_view
            else list(clause_type_rankings.get(selected_clause_type, []))
        )

        if selected_rows:
            max_docs = len(selected_rows)
            default_docs = min(25, max_docs)
            docs_to_plot = int(
                st.slider(
                    "Documents to show",
                    min_value=1,
                    max_value=max_docs,
                    value=max(1, default_docs),
                    key="app3_generosity_clause_docs",
                )
            )
            metric_mode = "composite" if is_composite_view else "ratio"
            displayed_rows = selected_rows[:docs_to_plot]
            if is_composite_view:
                st.caption(
                    "Composite clause score from pipeline output: mean clause-type generosity ratio per document "
                    "(NaNs excluded)."
                )
            else:
                st.caption(
                    "Ratio formula: "
                    "(worker rights + permissions - prohibitions - obligations) / "
                    "(firm rights + permissions - prohibitions - obligations)"
                )
            _render_clause_type_document_ratio_chart(
                "Composite Score (All Clause Types)" if is_composite_view else selected_clause_type,
                displayed_rows,
                metric_mode=metric_mode,
                document_name_lookup=document_name_lookup,
            )

            table_rows = []
            if is_composite_view:
                for row in displayed_rows:
                    doc_id = str(row.get("document_id", "")).strip()
                    score = row.get("composite_clause_score")
                    count = _to_int(row.get("composite_clause_type_count", 0))
                    table_rows.append(
                        {
                            "rank": int(row.get("rank", 0)),
                            "firm": _document_display_name(doc_id, document_name_lookup),
                            "composite_clause_score": (
                                round(float(score), 4)
                                if isinstance(score, (int, float)) and not np.isnan(float(score))
                                else None
                            ),
                            "clause_types_used": count,
                        }
                    )
            else:
                for row in displayed_rows:
                    ratio = row.get("worker_over_firm_ratio")
                    doc_id = str(row.get("document_id", "")).strip()
                    composite_payload = composite_clause_scores.get(doc_id, {})
                    composite_score = (
                        composite_payload.get("composite_clause_score")
                        if isinstance(composite_payload, dict)
                        else None
                    )
                    composite_count = (
                        _to_int(composite_payload.get("composite_clause_type_count", 0))
                        if isinstance(composite_payload, dict)
                        else 0
                    )
                    table_rows.append(
                        {
                            "rank": int(row.get("rank", 0)),
                            "firm": _document_display_name(doc_id, document_name_lookup),
                            "segments": int(row.get("segment_count", 0)),
                            "worker_benefit_total": int(row.get("worker_benefit_total", 0)),
                            "firm_benefit_total": int(row.get("firm_benefit_total", 0)),
                            "ratio": round(float(ratio), 4) if isinstance(ratio, (int, float)) else None,
                            "ratio_status": row.get("ratio_status", "undefined"),
                            "composite_clause_score": (
                                round(float(composite_score), 4)
                                if isinstance(composite_score, (int, float)) and not np.isnan(float(composite_score))
                                else None
                            ),
                            "composite_clause_type_count": composite_count,
                        }
                    )
            st.dataframe(table_rows, use_container_width=True, hide_index=True, height=320)
        else:
            st.info("No document rankings available for the selected clause type.")
    else:
        st.info(
            "Clause-type document rankings are not available yet. "
            "Run pipeline/04_generosity_ash/runner.py with classification-linked post-processing outputs."
        )

    if not doc_ids:
        st.warning("No `document_*` output folders found in 04_generosity_ash output.")
        return

    selected_doc = st.sidebar.selectbox(
        "Document",
        options=doc_ids,
        format_func=lambda doc_id: _document_display_name(str(doc_id), document_name_lookup),
        key="app3_generosity_doc",
    )
    selected_doc_label = _document_display_name(selected_doc, document_name_lookup)
    segment_numbers = _list_generosity_segment_numbers(generosity_root, selected_doc)
    doc_summary = _load_generosity_document_summary(generosity_root, selected_doc)
    doc_segment_generosity = doc_summary.get("segment_generosity", {}) if isinstance(doc_summary, dict) else {}
    if not isinstance(doc_segment_generosity, dict):
        doc_segment_generosity = {}

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Documents available:** {len(doc_ids)}")
    st.sidebar.markdown(f"**Segments in `{selected_doc_label}`:** {len(segment_numbers)}")
    doc_composite_payload = composite_clause_scores.get(selected_doc, {})
    doc_composite_clause_score = (
        doc_composite_payload.get("composite_clause_score")
        if isinstance(doc_composite_payload, dict)
        else None
    )
    doc_composite_clause_type_count = (
        _to_int(doc_composite_payload.get("composite_clause_type_count", 0))
        if isinstance(doc_composite_payload, dict)
        else 0
    )
    if doc_summary:
        st.sidebar.markdown(
            f"**Sentences in `{selected_doc_label}`:** {int(doc_summary.get('sentences_processed', 0) or 0)}"
        )
        st.sidebar.markdown(
            f"**Statement rows in `{selected_doc_label}`:** {int(doc_summary.get('statement_rows_written', 0) or 0)}"
        )
        if doc_segment_generosity:
            st.sidebar.markdown(
                f"**Doc generosity ratio:** "
                f"`{_format_ratio_value(doc_segment_generosity.get('worker_over_firm_ratio'), doc_segment_generosity.get('ratio_status'))}`"
            )
    if isinstance(doc_composite_clause_score, (int, float)) and not np.isnan(float(doc_composite_clause_score)):
        st.sidebar.markdown(
            f"**Composite clause score:** `{_format_composite_score_value(doc_composite_clause_score)}`"
        )
        st.sidebar.markdown(f"**Composite clause types used:** {doc_composite_clause_type_count}")

    st.markdown("### Document Summary")
    if doc_summary:
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Segments", int(doc_summary.get("segments_processed", 0) or 0))
        d2.metric("Sentences", int(doc_summary.get("sentences_processed", 0) or 0))
        d3.metric("Statement rows", int(doc_summary.get("statement_rows_written", 0) or 0))
        d4.metric(
            "Doc generosity ratio",
            _format_ratio_value(
                doc_segment_generosity.get("worker_over_firm_ratio"),
                doc_segment_generosity.get("ratio_status"),
            ),
        )
        d5.metric("Composite clause score", _format_composite_score_value(doc_composite_clause_score))
    else:
        st.info("document_summary.json not found for selected document.")

    if not segment_numbers:
        st.warning("No segment JSON files found for selected document.")
        return

    selected_segment = st.selectbox(
        "Segment",
        options=segment_numbers,
        format_func=lambda n: f"segment_{n}",
        key="app3_generosity_segment",
    )
    payload = _load_generosity_segment_payload(generosity_root, selected_doc, int(selected_segment))
    if not payload:
        st.warning("Selected segment payload is missing or unreadable.")
        return

    sentences = payload.get("sentences")
    if not isinstance(sentences, list):
        sentences = []
    sentence_count = int(payload.get("sentence_count", len(sentences)) or 0)
    segment_totals = payload.get("segment_totals", {})
    if not isinstance(segment_totals, dict):
        segment_totals = {}
    segment_generosity = payload.get("segment_generosity", {})
    if not isinstance(segment_generosity, dict) or not segment_generosity:
        segment_generosity = _load_generosity_segment_score(
            generosity_root=generosity_root,
            doc_id=selected_doc,
            segment_number=int(selected_segment),
        )
    if not isinstance(segment_generosity, dict):
        segment_generosity = {}
    segment_clause_type = str(payload.get("clause_type", "")).strip() or str(segment_generosity.get("clause_type", "")).strip() or "OTHER"

    st.markdown("### Segment Inspector")
    st.subheader(f"{selected_doc_label} / segment_{int(selected_segment)}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentences", sentence_count)
    m2.metric("Statement rows", int(segment_totals.get("statement_rows", 0) or 0))
    m3.metric("Clause type (03)", segment_clause_type)
    m4.metric(
        "Segment generosity ratio",
        _format_ratio_value(
            segment_generosity.get("worker_over_firm_ratio"),
            segment_generosity.get("ratio_status"),
        ),
    )
    if segment_generosity:
        with st.expander("Segment Generosity Breakdown", expanded=False):
            st.dataframe(
                [
                    {
                        "actor": "worker",
                        "rights": int(segment_generosity.get("worker_rights", 0) or 0),
                        "permissions": int(segment_generosity.get("worker_permissions", 0) or 0),
                        "prohibitions": int(segment_generosity.get("worker_prohibitions", 0) or 0),
                        "obligations": int(segment_generosity.get("worker_obligations", 0) or 0),
                        "benefit": int(segment_generosity.get("worker_benefit", 0) or 0),
                    },
                    {
                        "actor": "firm",
                        "rights": int(segment_generosity.get("firm_rights", 0) or 0),
                        "permissions": int(segment_generosity.get("firm_permissions", 0) or 0),
                        "prohibitions": int(segment_generosity.get("firm_prohibitions", 0) or 0),
                        "obligations": int(segment_generosity.get("firm_obligations", 0) or 0),
                        "benefit": int(segment_generosity.get("firm_benefit", 0) or 0),
                    },
                ],
                use_container_width=True,
                hide_index=True,
            )

    sentence_rows = []
    sentence_lookup = {}
    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        sentence_index = sentence.get("sentence_index")
        try:
            sentence_index = int(sentence_index)
        except Exception:
            sentence_index = len(sentence_rows) + 1
        classification = sentence.get("classification", {})
        if not isinstance(classification, dict):
            classification = {}

        raw_agents = classification.get("subject_agent_types", [])
        if not isinstance(raw_agents, list):
            raw_agents = []
        agents = [str(agent).strip() for agent in raw_agents if str(agent).strip()]

        auth_flags = classification.get("auth_category_flags", {})
        if not isinstance(auth_flags, dict):
            auth_flags = {}
        active_auth = [
            str(name)
            for name in ["obligation", "constraint", "permission", "entitlement"]
            if bool(auth_flags.get(name, False))
        ]

        sentence_lookup[sentence_index] = sentence
        sentence_rows.append(
            {
                "sentence_index": sentence_index,
                "sentence_type": str(classification.get("sentence_type", "other")),
                "agents": ", ".join(agents) if agents else "other",
                "auth_labels": ", ".join(active_auth) if active_auth else "none",
                "text": str(sentence.get("text", "")).strip(),
            }
        )

    sentence_rows.sort(key=lambda row: row["sentence_index"])
    st.dataframe(sentence_rows, use_container_width=True, hide_index=True, height=320)
    if not sentence_rows:
        st.info("No sentence payloads found in selected segment.")
        return

    selected_sentence_idx = st.selectbox(
        "Sentence index",
        options=[row["sentence_index"] for row in sentence_rows],
        key=f"app3_generosity_sentence_{selected_doc}_{selected_segment}",
    )
    selected_sentence = sentence_lookup.get(int(selected_sentence_idx), {})
    if not isinstance(selected_sentence, dict):
        selected_sentence = {}

    selected_classification = selected_sentence.get("classification", {})
    if not isinstance(selected_classification, dict):
        selected_classification = {}

    st.markdown("#### Selected Sentence")
    st.code(str(selected_sentence.get("text", "")).strip())

    active_flags = []
    auth_flags = selected_classification.get("auth_category_flags", {})
    if isinstance(auth_flags, dict):
        for label in ["obligation", "constraint", "permission", "entitlement"]:
            if bool(auth_flags.get(label, False)):
                active_flags.append(label)

    q1, q2, q3 = st.columns(3)
    q1.metric("Sentence Type", str(selected_classification.get("sentence_type", "other")))
    q2.metric(
        "Agents",
        ", ".join(selected_classification.get("subject_agent_types", []))
        if isinstance(selected_classification.get("subject_agent_types"), list)
        and selected_classification.get("subject_agent_types")
        else "other",
    )
    q3.metric("Auth Labels", ", ".join(active_flags) if active_flags else "none")

    evidence = selected_classification.get("classification_evidence", [])
    if isinstance(evidence, list) and evidence:
        st.caption("Classification evidence")
        st.write("\n".join(f"- {str(item)}" for item in evidence))

    auth_features = selected_classification.get("auth_features", {})
    if isinstance(auth_features, dict) and auth_features:
        with st.expander("Auth Features", expanded=False):
            st.json(auth_features)

    with st.expander("Full Classification Payload", expanded=False):
        st.json(selected_classification)

    tokens = selected_sentence.get("tokens", [])
    if isinstance(tokens, list) and tokens:
        token_rows = []
        for token in tokens:
            if not isinstance(token, dict):
                continue
            token_rows.append(
                {
                    "i": token.get("i"),
                    "text": token.get("text"),
                    "lemma": token.get("lemma"),
                    "pos": token.get("pos"),
                    "dep": token.get("dep"),
                    "head_i": token.get("head_i"),
                    "head_text": token.get("head_text"),
                }
            )
        if token_rows:
            st.markdown("#### Dependency Tokens")
            st.dataframe(token_rows, use_container_width=True, hide_index=True, height=260)

    statement_rows_bundle = _load_generosity_statement_rows_for_segment(
        generosity_root=generosity_root,
        doc_id=selected_doc,
        segment_number=int(selected_segment),
        limit=250,
    )
    statement_rows = statement_rows_bundle.get("rows", []) if isinstance(statement_rows_bundle, dict) else []
    st.markdown("#### statement_rows.csv Preview")
    if statement_rows:
        st.dataframe(statement_rows, use_container_width=True, hide_index=True, height=320)
        if bool(statement_rows_bundle.get("truncated")):
            st.caption("Preview limited to first 250 matching rows.")
    else:
        st.info("No matching rows found in statement_rows.csv for selected segment.")


@st.cache_data(show_spinner=False)
def _load_gab_document_rankings_csv(gab_root: Path):
    csv_path = gab_root / "document_composite_rankings.csv"
    rows = []
    if not csv_path.exists():
        return rows

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                document_id = str(row.get("document_id", "")).strip()
                if not document_id:
                    continue
                composite_score = _to_float_or_none(row.get("composite_score"))
                if isinstance(composite_score, (int, float)) and np.isnan(float(composite_score)):
                    composite_score = None
                mean_segment_generosity_score = _to_float_or_none(row.get("mean_segment_generosity_score"))
                mean_segment_rank = _to_float_or_none(row.get("mean_segment_rank"))
                document_rank_raw = _to_int(row.get("document_rank", 0))
                document_rank = document_rank_raw if document_rank_raw > 0 else None

                rows.append(
                    {
                        "document_id": document_id,
                        "segment_count": _to_int(row.get("segment_count", 0)),
                        "ranked_segment_count": _to_int(row.get("ranked_segment_count", 0)),
                        "mean_segment_generosity_score": mean_segment_generosity_score,
                        "mean_segment_rank": mean_segment_rank,
                        "composite_score": composite_score,
                        "document_rank": document_rank,
                    }
                )
    except Exception:
        return []

    rows.sort(
        key=lambda row: (
            row.get("document_rank") if isinstance(row.get("document_rank"), int) else 10**9,
            -(float(row.get("composite_score")) if isinstance(row.get("composite_score"), (int, float)) else float("-inf")),
            _doc_sort_key(str(row.get("document_id", ""))),
        )
    )
    return rows


@st.cache_data(show_spinner=False)
def _load_gab_clause_type_rankings_csv(gab_root: Path):
    csv_path = gab_root / "clause_type_document_rankings.csv"
    out: dict[str, list[dict]] = {}
    if not csv_path.exists():
        return out

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clause_type = str(row.get("clause_type", "")).strip() or "OTHER"
                document_id = str(row.get("document_id", "")).strip()
                if not document_id:
                    continue

                clause_type_score = _to_float_or_none(row.get("clause_type_score"))
                if isinstance(clause_type_score, (int, float)) and np.isnan(float(clause_type_score)):
                    clause_type_score = None
                clause_rank_raw = _to_int(row.get("clause_type_document_rank", 0))
                clause_rank = clause_rank_raw if clause_rank_raw > 0 else None

                out.setdefault(clause_type, []).append(
                    {
                        "clause_type": clause_type,
                        "document_id": document_id,
                        "segment_count": _to_int(row.get("segment_count", 0)),
                        "ranked_segment_count": _to_int(row.get("ranked_segment_count", 0)),
                        "mean_segment_generosity_score": _to_float_or_none(row.get("mean_segment_generosity_score")),
                        "mean_segment_rank": _to_float_or_none(row.get("mean_segment_rank")),
                        "clause_type_score": clause_type_score,
                        "clause_type_document_rank": clause_rank,
                    }
                )
    except Exception:
        return {}

    for clause_rows in out.values():
        clause_rows.sort(
            key=lambda row: (
                row.get("clause_type_document_rank")
                if isinstance(row.get("clause_type_document_rank"), int)
                else 10**9,
                -(
                    float(row.get("clause_type_score"))
                    if isinstance(row.get("clause_type_score"), (int, float))
                    else float("-inf")
                ),
                _doc_sort_key(str(row.get("document_id", ""))),
            )
        )
    return out


def _resolve_gab_clause_type_rankings(summary, csv_rankings: dict[str, list[dict]]):
    if isinstance(csv_rankings, dict) and csv_rankings:
        return csv_rankings

    out: dict[str, list[dict]] = {}
    raw_rankings = summary.get("clause_type_document_rankings", {}) if isinstance(summary, dict) else {}
    if not isinstance(raw_rankings, dict):
        return out

    for raw_clause_type, raw_rows in raw_rankings.items():
        clause_type = str(raw_clause_type).strip() or "OTHER"
        if not isinstance(raw_rows, list):
            continue
        rows: list[dict] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue
            document_id = str(raw_row.get("document_id", "")).strip()
            if not document_id:
                continue
            clause_type_score = _to_float_or_none(raw_row.get("clause_type_score"))
            if isinstance(clause_type_score, (int, float)) and np.isnan(float(clause_type_score)):
                clause_type_score = None
            clause_rank_raw = _to_int(raw_row.get("clause_type_document_rank", 0))
            rows.append(
                {
                    "clause_type": clause_type,
                    "document_id": document_id,
                    "segment_count": _to_int(raw_row.get("segment_count", 0)),
                    "ranked_segment_count": _to_int(raw_row.get("ranked_segment_count", 0)),
                    "mean_segment_generosity_score": _to_float_or_none(raw_row.get("mean_segment_generosity_score")),
                    "mean_segment_rank": _to_float_or_none(raw_row.get("mean_segment_rank")),
                    "clause_type_score": clause_type_score,
                    "clause_type_document_rank": clause_rank_raw if clause_rank_raw > 0 else None,
                }
            )
        rows.sort(
            key=lambda row: (
                row.get("clause_type_document_rank")
                if isinstance(row.get("clause_type_document_rank"), int)
                else 10**9,
                -(
                    float(row.get("clause_type_score"))
                    if isinstance(row.get("clause_type_score"), (int, float))
                    else float("-inf")
                ),
                _doc_sort_key(str(row.get("document_id", ""))),
            )
        )
        if rows:
            out[clause_type] = rows
    return out


@st.cache_data(show_spinner=False)
def _load_gab_segment_rankings_csv(gab_root: Path):
    csv_path = gab_root / "segment_generosity_rankings.csv"
    rows = []
    if not csv_path.exists():
        return rows

    truthy = {"1", "true", "t", "yes", "y"}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                document_id = str(row.get("document_id", "")).strip()
                segment_number = _to_int(row.get("segment_number", 0))
                if not document_id or segment_number <= 0:
                    continue
                score = _to_float_or_none(row.get("segment_generosity_score"))
                rank = _to_float_or_none(row.get("segment_rank"))
                percentile = _to_float_or_none(row.get("segment_percentile"))
                rows.append(
                    {
                        "segment_id": str(row.get("segment_id", "")).strip(),
                        "document_id": document_id,
                        "segment_number": segment_number,
                        "clause_type": str(row.get("clause_type", "")).strip() or "OTHER",
                        "segment_generosity_score": score,
                        "segment_rank": rank,
                        "segment_percentile": percentile,
                        "text_char_count": _to_int(row.get("text_char_count", 0)),
                        "is_truncated": str(row.get("is_truncated", "")).strip().lower() in truthy,
                        "segment_path": str(row.get("segment_path", "")).strip(),
                    }
                )
    except Exception:
        return []

    rows.sort(key=lambda row: (_doc_sort_key(row["document_id"]), row["segment_number"]))
    return rows


def _resolve_gab_document_rankings(summary, csv_rows: list[dict]):
    if csv_rows:
        return csv_rows

    out = []
    raw = summary.get("document_composite_rankings", {}) if isinstance(summary, dict) else {}
    if isinstance(raw, dict):
        for raw_doc_id, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            document_id = str(payload.get("document_id", raw_doc_id)).strip()
            if not document_id:
                continue
            composite_score = _to_float_or_none(payload.get("composite_score"))
            if isinstance(composite_score, (int, float)) and np.isnan(float(composite_score)):
                composite_score = None
            rank_raw = _to_int(payload.get("document_rank", 0))
            out.append(
                {
                    "document_id": document_id,
                    "segment_count": _to_int(payload.get("segment_count", 0)),
                    "ranked_segment_count": _to_int(payload.get("ranked_segment_count", 0)),
                    "mean_segment_generosity_score": _to_float_or_none(payload.get("mean_segment_generosity_score")),
                    "mean_segment_rank": _to_float_or_none(payload.get("mean_segment_rank")),
                    "composite_score": composite_score,
                    "document_rank": rank_raw if rank_raw > 0 else None,
                }
            )

    out.sort(
        key=lambda row: (
            row.get("document_rank") if isinstance(row.get("document_rank"), int) else 10**9,
            -(float(row.get("composite_score")) if isinstance(row.get("composite_score"), (int, float)) else float("-inf")),
            _doc_sort_key(str(row.get("document_id", ""))),
        )
    )
    return out


def _resolve_gab_doc_ids(
    gab_root: Path,
    document_rank_rows: list[dict],
    segment_rows: list[dict],
):
    doc_ids = set(_list_generosity_doc_ids(gab_root))
    doc_ids.update(str(row.get("document_id", "")).strip() for row in document_rank_rows if str(row.get("document_id", "")).strip())
    doc_ids.update(str(row.get("document_id", "")).strip() for row in segment_rows if str(row.get("document_id", "")).strip())
    return sorted(doc_ids, key=_doc_sort_key)


def _resolve_gab_segment_text_path(segment_row: dict, segmentation_root: Path | None = None) -> Path | None:
    segment_path_raw = str(segment_row.get("segment_path", "")).strip()
    if segment_path_raw:
        direct_path = Path(segment_path_raw).expanduser()
        if direct_path.exists():
            return direct_path

    if not isinstance(segmentation_root, Path):
        return None
    document_id = str(segment_row.get("document_id", "")).strip()
    segment_number = _to_int(segment_row.get("segment_number", 0))
    if not document_id or segment_number <= 0:
        return None

    candidates = [
        segmentation_root / document_id / "segments" / f"segment_{segment_number}.txt",
        segmentation_root / document_id / f"segment_{segment_number}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _gab_clause_type_ranked_segments(segment_rows: list[dict], clause_type: str) -> list[dict]:
    clause_key = str(clause_type).strip() or "OTHER"
    ranked_rows: list[dict] = []
    for row in segment_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("clause_type", "")).strip() != clause_key:
            continue
        raw_rank = row.get("segment_rank")
        if not isinstance(raw_rank, (int, float)):
            continue
        rank = float(raw_rank)
        if np.isnan(rank):
            continue
        ranked_rows.append(row)

    ranked_rows.sort(
        key=lambda row: (
            float(row.get("segment_rank", float("inf"))),
            _doc_sort_key(str(row.get("document_id", "")).strip()),
            _to_int(row.get("segment_number", 0)),
        )
    )
    return ranked_rows


def _select_gab_clause_type_segment_examples(ranked_rows: list[dict]) -> list[tuple[str, int]]:
    count = len(ranked_rows)
    if count <= 0:
        return []
    if count == 1:
        return [("Highest", 0)]
    if count == 2:
        return [("Highest", 0), ("Lowest", 1)]
    middle_idx = (count - 1) // 2
    return [("Highest", 0), ("Middle", middle_idx), ("Lowest", count - 1)]


def _render_gab_clause_type_segment_examples(
    *,
    selected_clause_type: str,
    segment_rows: list[dict],
    document_name_lookup: dict[str, str],
    segmentation_root: Path | None,
):
    ranked_rows = _gab_clause_type_ranked_segments(segment_rows, selected_clause_type)
    examples = _select_gab_clause_type_segment_examples(ranked_rows)
    st.markdown("#### Segment Comparison")
    st.caption(
        "Within the selected clause type: Highest = best segment rank, "
        "Middle = median segment rank, Lowest = worst segment rank."
    )
    if not examples:
        st.info(f"No ranked segments found for clause type `{selected_clause_type}`.")
        return
    if len(examples) < 3:
        st.caption("Fewer than 3 ranked segments are available for this clause type.")

    safe_clause_key = re.sub(r"[^a-zA-Z0-9_]+", "_", str(selected_clause_type)).strip("_") or "OTHER"
    total_ranked = len(ranked_rows)
    panel_rows: list[tuple[str, dict, int, int, str]] = []
    for position, anchor_idx in examples:
        state_key = f"app3_gab_segment_compare_idx_{safe_clause_key}_{position.lower()}"
        current_idx = _to_int(st.session_state.get(state_key, anchor_idx), anchor_idx)
        if current_idx < 0 or current_idx >= total_ranked:
            current_idx = anchor_idx
        st.session_state[state_key] = current_idx

        selected_row = ranked_rows[current_idx]
        panel_rows.append((position, selected_row, current_idx, anchor_idx, state_key))

    summary_rows = []
    for position, row, current_idx, anchor_idx, _state_key in panel_rows:
        document_id = str(row.get("document_id", "")).strip()
        segment_number = _to_int(row.get("segment_number", 0))
        rank = row.get("segment_rank")
        percentile = row.get("segment_percentile")
        score = row.get("segment_generosity_score")
        summary_rows.append(
            {
                "position": position,
                "firm": _document_display_name(document_id, document_name_lookup),
                "document_id": document_id,
                "segment": segment_number,
                "view_rank": current_idx + 1,
                "anchor_rank": anchor_idx + 1,
                "segment_rank": (
                    round(float(rank), 3) if isinstance(rank, (int, float)) and not np.isnan(float(rank)) else None
                ),
                "percentile": (
                    round(float(percentile), 2)
                    if isinstance(percentile, (int, float)) and not np.isnan(float(percentile))
                    else None
                ),
                "segment_score": (
                    round(float(score), 4) if isinstance(score, (int, float)) and not np.isnan(float(score)) else None
                ),
            }
        )
    st.dataframe(summary_rows, use_container_width=True, hide_index=True, height=180)

    for position, row, current_idx, anchor_idx, state_key in panel_rows:
        document_id = str(row.get("document_id", "")).strip()
        segment_number = _to_int(row.get("segment_number", 0))
        firm_label = _document_display_name(document_id, document_name_lookup)
        st.markdown(f"**{position}: {firm_label} / segment_{segment_number}**")

        rank = row.get("segment_rank")
        percentile = row.get("segment_percentile")
        score = row.get("segment_generosity_score")
        info_c1, info_c2, info_c3 = st.columns(3)
        info_c1.metric(
            "Rank",
            f"{float(rank):.3f}" if isinstance(rank, (int, float)) and not np.isnan(float(rank)) else "n/a",
        )
        info_c2.metric(
            "Percentile",
            f"{float(percentile):.2f}" if isinstance(percentile, (int, float)) and not np.isnan(float(percentile)) else "n/a",
        )
        info_c3.metric(
            "Segment score",
            f"{float(score):.4f}" if isinstance(score, (int, float)) and not np.isnan(float(score)) else "n/a",
        )

        nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([3.2, 1.5, 1.5, 1.2])
        with nav_c1:
            st.caption(f"Viewing rank `{current_idx + 1}` of `{total_ranked}`")
        with nav_c2:
            go_higher = st.button(
                "One higher",
                key=f"{state_key}_higher",
                disabled=current_idx <= 0,
            )
        with nav_c3:
            go_lower = st.button(
                "One lower",
                key=f"{state_key}_lower",
                disabled=current_idx >= (total_ranked - 1),
            )
        with nav_c4:
            reset_anchor = st.button("Reset", key=f"{state_key}_reset")

        next_idx = current_idx
        if go_higher and next_idx > 0:
            next_idx -= 1
        if go_lower and next_idx < (total_ranked - 1):
            next_idx += 1
        if reset_anchor:
            next_idx = anchor_idx
        if next_idx != current_idx:
            st.session_state[state_key] = next_idx
            st.rerun()

        segment_text_path = _resolve_gab_segment_text_path(row, segmentation_root)
        if segment_text_path is None:
            st.info("Segment source text file not found.")
            continue

        preview = _load_text_preview(segment_text_path, max_chars=2200)
        st.code(str(preview.get("text", "")))
        if bool(preview.get("truncated")):
            st.caption("Preview truncated to first 2,200 characters.")
        st.caption(f"Source: `{segment_text_path}`")


@st.cache_data(show_spinner=False)
def _load_text_preview(path: Path, max_chars: int = 7000):
    if not path.exists():
        return {"text": "", "char_count": 0, "truncated": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "text": text[:max_chars],
        "char_count": len(text),
        "truncated": len(text) > max_chars,
    }


def _render_generosity_gab_view(
    generosity_root: Path,
    cba_list_path: Path | None = None,
    segmentation_root: Path | None = None,
):
    if not generosity_root.exists():
        st.error(f"04_generosity_gab output folder not found: {generosity_root}")
        return

    summary = _load_generosity_summary(generosity_root)
    document_name_lookup = (
        _load_document_firm_name_lookup(cba_list_path)
        if isinstance(cba_list_path, Path)
        else {}
    )
    document_rank_rows = _resolve_gab_document_rankings(summary, _load_gab_document_rankings_csv(generosity_root))
    clause_type_rankings = _resolve_gab_clause_type_rankings(
        summary,
        _load_gab_clause_type_rankings_csv(generosity_root),
    )
    segment_rows = _load_gab_segment_rankings_csv(generosity_root)
    doc_ids = _resolve_gab_doc_ids(generosity_root, document_rank_rows, segment_rows)
    doc_rank_lookup = {
        str(row.get("document_id", "")).strip(): row
        for row in document_rank_rows
        if str(row.get("document_id", "")).strip()
    }

    st.markdown("### Run Summary")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", int(summary.get("documents_processed", len(doc_ids)) or len(doc_ids)))
        c2.metric("Segments", int(summary.get("segments_processed", len(segment_rows)) or len(segment_rows)))
        c3.metric("Segments ranked", int(summary.get("segments_ranked", 0) or 0))
        c4.metric("Documents ranked", int(summary.get("documents_ranked", len(document_rank_rows)) or len(document_rank_rows)))

        run_meta = []
        model = str(summary.get("model", "")).strip()
        if model:
            run_meta.append(f"model: `{model}`")
        if "n_rounds" in summary:
            run_meta.append(f"n_rounds: `{_to_int(summary.get('n_rounds', 0))}`")
        if "matches_per_round" in summary:
            run_meta.append(f"matches_per_round: `{_to_int(summary.get('matches_per_round', 0))}`")
        if "n_parallels" in summary:
            run_meta.append(f"n_parallels: `{_to_int(summary.get('n_parallels', 0))}`")
        if run_meta:
            st.caption(" | ".join(run_meta))
    else:
        st.info("summary.json not found or unreadable in this output folder.")

    st.markdown("### Document Rankings")
    if document_rank_rows or clause_type_rankings:
        available_clause_types = sorted(
            clause_type_rankings.keys(),
            key=lambda label: (label == "OTHER", str(label).lower()),
        )
        composite_option = "COMPOSITE_SCORE"
        metric_options = ([composite_option] if document_rank_rows else []) + available_clause_types
        selected_metric = st.selectbox(
            "Bar chart metric",
            options=metric_options,
            format_func=lambda label: (
                "Composite score (all clause types)"
                if label == composite_option
                else f"Clause type: {label}"
            ),
            key="app3_gab_chart_metric",
        )
        is_composite_view = selected_metric == composite_option
        ranking_rows = (
            list(document_rank_rows)
            if is_composite_view
            else list(clause_type_rankings.get(str(selected_metric), []))
        )

        if ranking_rows:
            safe_metric = re.sub(r"[^a-zA-Z0-9_]+", "_", str(selected_metric)).strip("_") or "metric"
            max_docs = len(ranking_rows)
            default_docs = min(25, max_docs)
            docs_to_plot = int(
                st.slider(
                    "Documents to show",
                    min_value=1,
                    max_value=max_docs,
                    value=max(1, default_docs),
                    key=f"app3_gab_docs_to_plot_{safe_metric}",
                )
            )
            displayed_rows = ranking_rows[:docs_to_plot]

            if is_composite_view:
                st.caption(
                    "Composite score from pipeline output: mean clause-type score per document "
                    "(clause types with NA are excluded)."
                )
                chart_title = "Composite Score (All Clause Types)"
                chart_rows = [
                    {
                        "document_id": row.get("document_id"),
                        "composite_clause_score": row.get("composite_score"),
                    }
                    for row in displayed_rows
                ]
            else:
                st.caption(
                    f"Clause-type score for `{selected_metric}`: mean segment percentile within this clause type."
                )
                chart_title = str(selected_metric)
                chart_rows = [
                    {
                        "document_id": row.get("document_id"),
                        "composite_clause_score": row.get("clause_type_score"),
                    }
                    for row in displayed_rows
                ]

            _render_clause_type_document_ratio_chart(
                chart_title,
                chart_rows,
                metric_mode="composite",
                document_name_lookup=document_name_lookup,
                composite_axis_label=(
                    "Clause-type generosity score (mean segment percentile within selected clause type)"
                    if not is_composite_view
                    else None
                ),
                composite_title_suffix=("Clause-Type Scores" if not is_composite_view else None),
            )
            if not is_composite_view:
                _render_gab_clause_type_segment_examples(
                    selected_clause_type=str(selected_metric),
                    segment_rows=segment_rows,
                    document_name_lookup=document_name_lookup,
                    segmentation_root=segmentation_root,
                )

            if is_composite_view:
                table_rows = [
                    {
                        "rank": row.get("document_rank"),
                        "firm": _document_display_name(str(row.get("document_id", "")).strip(), document_name_lookup),
                        "composite_score": (
                            round(float(row.get("composite_score")), 4)
                            if isinstance(row.get("composite_score"), (int, float))
                            else None
                        ),
                        "ranked_segments": _to_int(row.get("ranked_segment_count", 0)),
                        "segments_total": _to_int(row.get("segment_count", 0)),
                        "mean_segment_score": (
                            round(float(row.get("mean_segment_generosity_score")), 4)
                            if isinstance(row.get("mean_segment_generosity_score"), (int, float))
                            else None
                        ),
                    }
                    for row in displayed_rows
                ]
            else:
                table_rows = [
                    {
                        "rank": row.get("clause_type_document_rank"),
                        "firm": _document_display_name(str(row.get("document_id", "")).strip(), document_name_lookup),
                        "clause_type_score": (
                            round(float(row.get("clause_type_score")), 4)
                            if isinstance(row.get("clause_type_score"), (int, float))
                            else None
                        ),
                        "ranked_segments": _to_int(row.get("ranked_segment_count", 0)),
                        "segments_total": _to_int(row.get("segment_count", 0)),
                        "mean_segment_score": (
                            round(float(row.get("mean_segment_generosity_score")), 4)
                            if isinstance(row.get("mean_segment_generosity_score"), (int, float))
                            else None
                        ),
                    }
                    for row in displayed_rows
                ]

            st.dataframe(
                table_rows,
                use_container_width=True,
                hide_index=True,
                height=320,
            )
        else:
            st.info("No ranking rows available for the selected GAB metric.")
    else:
        st.info(
            "Document rankings not found yet. "
            "Run pipeline/04_generosity_gab/runner.py to generate GAB outputs."
        )

    if not doc_ids:
        st.warning("No document-level GAB outputs found.")
        return

    selected_doc = st.sidebar.selectbox(
        "Document",
        options=doc_ids,
        format_func=lambda doc_id: _document_display_name(str(doc_id), document_name_lookup),
        key="app3_gab_doc",
    )
    selected_doc_label = _document_display_name(selected_doc, document_name_lookup)
    selected_doc_rank = doc_rank_lookup.get(selected_doc, {})
    if not isinstance(selected_doc_rank, dict):
        selected_doc_rank = {}
    doc_segment_rows = [row for row in segment_rows if str(row.get("document_id", "")).strip() == selected_doc]
    doc_segment_rows.sort(
        key=lambda row: (
            float(row.get("segment_rank")) if isinstance(row.get("segment_rank"), (int, float)) else float("inf"),
            _to_int(row.get("segment_number", 0)),
        )
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Documents available:** {len(doc_ids)}")
    st.sidebar.markdown(f"**Ranked segments in `{selected_doc_label}`:** {len(doc_segment_rows)}")
    composite_score = selected_doc_rank.get("composite_score")
    if isinstance(composite_score, (int, float)):
        st.sidebar.markdown(f"**Composite score:** `{_format_composite_score_value(composite_score)}`")
    doc_rank = selected_doc_rank.get("document_rank")
    if isinstance(doc_rank, int):
        st.sidebar.markdown(f"**Document rank:** `{doc_rank}`")

    if not doc_segment_rows:
        st.info("No ranked segments found for selected document.")
        return

    st.markdown("### Segment Rankings")
    st.dataframe(
        [
            {
                "segment": _to_int(row.get("segment_number", 0)),
                "segment_rank": (
                    round(float(row.get("segment_rank")), 3)
                    if isinstance(row.get("segment_rank"), (int, float))
                    else None
                ),
                "segment_score": (
                    round(float(row.get("segment_generosity_score")), 4)
                    if isinstance(row.get("segment_generosity_score"), (int, float))
                    else None
                ),
                "percentile": (
                    round(float(row.get("segment_percentile")), 2)
                    if isinstance(row.get("segment_percentile"), (int, float))
                    else None
                ),
                "text_chars": _to_int(row.get("text_char_count", 0)),
                "truncated": bool(row.get("is_truncated", False)),
            }
            for row in doc_segment_rows[:500]
        ],
        use_container_width=True,
        hide_index=True,
        height=320,
    )
    if len(doc_segment_rows) > 500:
        st.caption("Showing first 500 ranked segments for this document.")


def _llm_slugify(value: str) -> str:
    raw = str(value).strip() or "OTHER"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")
    return slug or "OTHER"


def _llm_score_or_none(value):
    score = _to_float_or_none(value)
    if isinstance(score, (int, float)) and not np.isnan(float(score)):
        return float(score)
    return None


def _normalize_llm_detail_scores(raw_detail_scores):
    payload = raw_detail_scores
    if isinstance(raw_detail_scores, str):
        text = raw_detail_scores.strip()
        if not text:
            payload = []
        else:
            try:
                payload = json.loads(text)
            except Exception:
                payload = []
    if not isinstance(payload, list):
        return []

    out = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        score = _to_int(item.get("score", 0), 0)
        if not (1 <= score <= 5):
            score = None
        out.append(
            {
                "name": name,
                "score": score,
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def _load_llm_document_composite_scores_csv(generosity_root: Path):
    csv_path = generosity_root / "document_composite_scores.csv"
    rows = []
    if not csv_path.exists():
        return rows

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                document_id = str(row.get("document_id", "")).strip()
                if not document_id:
                    continue
                rows.append(
                    {
                        "provider": str(row.get("provider", "")).strip(),
                        "model": str(row.get("model", "")).strip(),
                        "document_id": document_id,
                        "document_composite_score": _llm_score_or_none(row.get("document_composite_score")),
                        "clause_count_scored": max(0, _to_int(row.get("clause_count_scored", 0))),
                    }
                )
    except Exception:
        return []

    rows.sort(
        key=lambda row: (
            -(
                _llm_score_or_none(row.get("document_composite_score"))
                if _llm_score_or_none(row.get("document_composite_score")) is not None
                else float("-inf")
            ),
            _doc_sort_key(str(row.get("document_id", ""))),
        )
    )
    rank = 0
    for row in rows:
        score = _llm_score_or_none(row.get("document_composite_score"))
        if score is None:
            row["document_rank"] = None
            continue
        rank += 1
        row["document_rank"] = rank
    return rows


@st.cache_data(show_spinner=False)
def _load_llm_document_clause_scores_csv(generosity_root: Path):
    csv_path = generosity_root / "document_clause_composite_scores.csv"
    rows = []
    if not csv_path.exists():
        return rows

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                document_id = str(row.get("document_id", "")).strip()
                clause_type = str(row.get("clause_type", "")).strip() or "OTHER"
                if not document_id:
                    continue
                rows.append(
                    {
                        "provider": str(row.get("provider", "")).strip(),
                        "model": str(row.get("model", "")).strip(),
                        "document_id": document_id,
                        "clause_type": clause_type,
                        "segment_count": max(0, _to_int(row.get("segment_count", 0))),
                        "clause_composite_score": _llm_score_or_none(row.get("clause_composite_score")),
                        "detail_scores": _normalize_llm_detail_scores(row.get("detail_scores_json", "")),
                        "status": str(row.get("status", "")).strip() or "unknown",
                        "error": str(row.get("error", "")).strip(),
                    }
                )
    except Exception:
        return []

    rows.sort(
        key=lambda row: (
            _doc_sort_key(str(row.get("document_id", ""))),
            str(row.get("clause_type", "")).lower(),
        )
    )
    return rows


def _resolve_llm_clause_type_rankings(clause_rows: list[dict]):
    out: dict[str, list[dict]] = {}
    for row in clause_rows:
        if not isinstance(row, dict):
            continue
        clause_type = str(row.get("clause_type", "")).strip() or "OTHER"
        normalized = dict(row)
        normalized["clause_type_score"] = _llm_score_or_none(row.get("clause_composite_score"))
        out.setdefault(clause_type, []).append(normalized)

    for clause_type, rows in out.items():
        rows.sort(
            key=lambda row: (
                -(
                    _llm_score_or_none(row.get("clause_type_score"))
                    if _llm_score_or_none(row.get("clause_type_score")) is not None
                    else float("-inf")
                ),
                _doc_sort_key(str(row.get("document_id", ""))),
            )
        )
        rank = 0
        for row in rows:
            score = _llm_score_or_none(row.get("clause_type_score"))
            if score is None:
                row["clause_type_document_rank"] = None
                continue
            rank += 1
            row["clause_type_document_rank"] = rank
    return out


def _resolve_llm_doc_ids(document_rows: list[dict], clause_rows: list[dict]):
    doc_ids = set()
    for row in document_rows:
        document_id = str(row.get("document_id", "")).strip()
        if document_id:
            doc_ids.add(document_id)
    for row in clause_rows:
        document_id = str(row.get("document_id", "")).strip()
        if document_id:
            doc_ids.add(document_id)
    return sorted(doc_ids, key=_doc_sort_key)


@st.cache_data(show_spinner=False)
def _load_jsonl_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows


@st.cache_data(show_spinner=False)
def _load_llm_clause_schema(generosity_root: Path, clause_type: str):
    payload = _safe_read_json(generosity_root / "schemas" / f"{_llm_slugify(clause_type)}.schema.json")
    if isinstance(payload, dict):
        return payload
    return {}


@st.cache_data(show_spinner=False)
def _load_llm_clause_rubric(generosity_root: Path, clause_type: str):
    payload = _safe_read_json(generosity_root / "rubrics" / f"{_llm_slugify(clause_type)}.rubric.json")
    if isinstance(payload, dict):
        return payload
    return {}


@st.cache_data(show_spinner=False)
def _load_llm_clause_distribution(generosity_root: Path, clause_type: str):
    payload = _safe_read_json(generosity_root / "rubrics" / f"{_llm_slugify(clause_type)}.distribution.json")
    if isinstance(payload, dict):
        return payload
    return {}


@st.cache_data(show_spinner=False)
def _load_llm_clause_extractions(generosity_root: Path, clause_type: str):
    path = generosity_root / "extractions" / f"{_llm_slugify(clause_type)}.jsonl"
    return _load_jsonl_rows(path)


@st.cache_data(show_spinner=False)
def _load_classification_segment_preview(
    classification_root: Path,
    document_id: str,
    segment_number: int,
    max_chars: int = 2200,
):
    segment_path = classification_root / str(document_id).strip() / f"segment_{int(segment_number)}.json"
    payload = _safe_read_json(segment_path)
    if not isinstance(payload, dict):
        return {"text": "", "char_count": 0, "truncated": False, "path": str(segment_path)}

    text = str(payload.get("segment_text", "")).strip()
    return {
        "text": text[:max_chars],
        "char_count": len(text),
        "truncated": len(text) > max_chars,
        "path": str(segment_path),
    }


def _render_generosity_llm_view(
    generosity_root: Path,
    cba_list_path: Path | None = None,
    classification_root: Path | None = None,
):
    if not generosity_root.exists():
        st.error(f"04_generosity_llm output folder not found: {generosity_root}")
        return

    summary = _load_generosity_summary(generosity_root)
    document_name_lookup = (
        _load_document_firm_name_lookup(cba_list_path)
        if isinstance(cba_list_path, Path)
        else {}
    )
    document_rank_rows = _load_llm_document_composite_scores_csv(generosity_root)
    document_clause_rows = _load_llm_document_clause_scores_csv(generosity_root)
    clause_type_rankings = _resolve_llm_clause_type_rankings(document_clause_rows)
    doc_ids = _resolve_llm_doc_ids(document_rank_rows, document_clause_rows)
    doc_rank_lookup = {
        str(row.get("document_id", "")).strip(): row
        for row in document_rank_rows
        if str(row.get("document_id", "")).strip()
    }

    st.markdown("### Run Summary")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents scored", int(summary.get("documents_scored", len(document_rank_rows)) or len(document_rank_rows)))
        c2.metric(
            "Clause evaluations",
            int(summary.get("clause_evaluations_written", len(document_clause_rows)) or len(document_clause_rows)),
        )
        c3.metric("Schemas", int(summary.get("schemas_written", 0) or 0))
        c4.metric("Segment extractions", int(summary.get("segment_extractions_written", 0) or 0))

        run_meta = []
        provider = str(summary.get("provider", "")).strip()
        model = str(summary.get("model", "")).strip()
        if provider:
            run_meta.append(f"provider: `{provider}`")
        if model:
            run_meta.append(f"model: `{model}`")
        base_url = str(summary.get("base_url", "")).strip()
        if base_url:
            run_meta.append(f"base_url: `{base_url}`")
        if run_meta:
            st.caption(" | ".join(run_meta))

        top_clause_types = summary.get("top_clause_types_selected", [])
        if isinstance(top_clause_types, list) and top_clause_types:
            st.caption("Top clause types scored: " + ", ".join(str(item) for item in top_clause_types[:10]))
    else:
        st.info("summary.json not found or unreadable in this output folder.")

    st.markdown("### Document Rankings")
    selected_metric = ""
    is_composite_view = True
    if document_rank_rows or clause_type_rankings:
        available_clause_types = sorted(
            clause_type_rankings.keys(),
            key=lambda label: (label == "OTHER", str(label).lower()),
        )
        composite_option = "DOCUMENT_COMPOSITE_SCORE"
        metric_options = ([composite_option] if document_rank_rows else []) + available_clause_types
        selected_metric = st.selectbox(
            "Bar chart metric",
            options=metric_options,
            format_func=lambda label: (
                "Composite score (all scored clause types)"
                if label == composite_option
                else f"Clause type: {label}"
            ),
            key="app3_llm_chart_metric",
        )
        is_composite_view = selected_metric == composite_option
        ranking_rows = (
            list(document_rank_rows)
            if is_composite_view
            else list(clause_type_rankings.get(str(selected_metric), []))
        )

        if ranking_rows:
            safe_metric = re.sub(r"[^a-zA-Z0-9_]+", "_", str(selected_metric)).strip("_") or "metric"
            max_docs = len(ranking_rows)
            default_docs = min(25, max_docs)
            docs_to_plot = int(
                st.slider(
                    "Documents to show",
                    min_value=1,
                    max_value=max_docs,
                    value=max(1, default_docs),
                    key=f"app3_llm_docs_to_plot_{safe_metric}",
                )
            )
            displayed_rows = ranking_rows[:docs_to_plot]

            if is_composite_view:
                st.caption(
                    "Document composite score from pipeline output: mean clause composite score per document "
                    "(clause types with missing score are excluded)."
                )
                chart_title = "Composite Score (All Clause Types)"
                chart_rows = [
                    {
                        "document_id": row.get("document_id"),
                        "composite_clause_score": row.get("document_composite_score"),
                    }
                    for row in displayed_rows
                ]
            else:
                st.caption(
                    f"Clause composite score for `{selected_metric}` on a 1-5 scale. "
                    "Higher is more worker-favorable."
                )
                chart_title = str(selected_metric)
                chart_rows = [
                    {
                        "document_id": row.get("document_id"),
                        "composite_clause_score": row.get("clause_type_score"),
                    }
                    for row in displayed_rows
                ]

            _render_clause_type_document_ratio_chart(
                chart_title,
                chart_rows,
                metric_mode="composite",
                document_name_lookup=document_name_lookup,
                composite_axis_label="Clause/document generosity score (1-5)",
                composite_title_suffix=("Clause-Type Scores" if not is_composite_view else "Composite Scores"),
            )

            if is_composite_view:
                table_rows = [
                    {
                        "rank": row.get("document_rank"),
                        "firm": _document_display_name(str(row.get("document_id", "")).strip(), document_name_lookup),
                        "composite_score": (
                            round(float(row.get("document_composite_score")), 4)
                            if isinstance(row.get("document_composite_score"), (int, float))
                            else None
                        ),
                        "clause_types_scored": _to_int(row.get("clause_count_scored", 0)),
                    }
                    for row in displayed_rows
                ]
            else:
                table_rows = [
                    {
                        "rank": row.get("clause_type_document_rank"),
                        "firm": _document_display_name(str(row.get("document_id", "")).strip(), document_name_lookup),
                        "clause_type_score": (
                            round(float(row.get("clause_type_score")), 4)
                            if isinstance(row.get("clause_type_score"), (int, float))
                            else None
                        ),
                        "segments_used": _to_int(row.get("segment_count", 0)),
                        "status": str(row.get("status", "")).strip(),
                    }
                    for row in displayed_rows
                ]

            st.dataframe(
                table_rows,
                use_container_width=True,
                hide_index=True,
                height=320,
            )
        else:
            st.info("No ranking rows available for the selected LLM metric.")
    else:
        st.info(
            "LLM scoring outputs not found yet. "
            "Run pipeline/04_generosity_llm/runner.py to generate rubric-based outputs."
        )

    if not doc_ids:
        st.warning("No document-level LLM outputs found.")
        return

    selected_doc = st.sidebar.selectbox(
        "Document",
        options=doc_ids,
        format_func=lambda doc_id: _document_display_name(str(doc_id), document_name_lookup),
        key="app3_llm_doc",
    )
    selected_doc_label = _document_display_name(selected_doc, document_name_lookup)
    selected_doc_rank = doc_rank_lookup.get(selected_doc, {})
    if not isinstance(selected_doc_rank, dict):
        selected_doc_rank = {}
    doc_clause_rows = [row for row in document_clause_rows if str(row.get("document_id", "")).strip() == selected_doc]
    doc_clause_rows.sort(
        key=lambda row: (
            -(
                _llm_score_or_none(row.get("clause_composite_score"))
                if _llm_score_or_none(row.get("clause_composite_score")) is not None
                else float("-inf")
            ),
            str(row.get("clause_type", "")).lower(),
        )
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Documents available:** {len(doc_ids)}")
    st.sidebar.markdown(f"**Clauses scored in `{selected_doc_label}`:** {len(doc_clause_rows)}")
    composite_score = _llm_score_or_none(selected_doc_rank.get("document_composite_score"))
    if composite_score is not None:
        st.sidebar.markdown(f"**Document composite score:** `{_format_composite_score_value(composite_score)}`")
    doc_rank = selected_doc_rank.get("document_rank")
    if isinstance(doc_rank, int):
        st.sidebar.markdown(f"**Document rank:** `{doc_rank}`")
    st.sidebar.markdown(f"**Document ID:** `{selected_doc}`")

    st.markdown("### Document Clause Scores")
    if not doc_clause_rows:
        st.info("No clause scores found for selected document.")
        return

    st.dataframe(
        [
            {
                "clause_type": str(row.get("clause_type", "")).strip() or "OTHER",
                "clause_composite_score": (
                    round(float(row.get("clause_composite_score")), 4)
                    if isinstance(row.get("clause_composite_score"), (int, float))
                    else None
                ),
                "segments_used": _to_int(row.get("segment_count", 0)),
                "status": str(row.get("status", "")).strip(),
            }
            for row in doc_clause_rows
        ],
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    clause_options = []
    seen_clause_options = set()
    for row in doc_clause_rows:
        clause_label = str(row.get("clause_type", "")).strip() or "OTHER"
        if clause_label in seen_clause_options:
            continue
        seen_clause_options.add(clause_label)
        clause_options.append(clause_label)
    default_clause = clause_options[0]
    if not is_composite_view and selected_metric in clause_options:
        default_clause = str(selected_metric)
    default_idx = clause_options.index(default_clause) if default_clause in clause_options else 0
    selected_clause_type = st.selectbox(
        "Clause type inspector",
        options=clause_options,
        index=default_idx,
        key=f"app3_llm_clause_for_doc_{selected_doc}",
    )
    selected_clause_row = next(
        (
            row
            for row in doc_clause_rows
            if str(row.get("clause_type", "")).strip() == str(selected_clause_type).strip()
        ),
        {},
    )

    st.markdown("### Clause Inspector")
    st.subheader(f"{selected_doc_label} / {selected_clause_type}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Clause composite score",
        _format_composite_score_value(selected_clause_row.get("clause_composite_score")),
    )
    c2.metric("Segments used", _to_int(selected_clause_row.get("segment_count", 0)))
    detail_scores = selected_clause_row.get("detail_scores", [])
    detail_score_count = len(detail_scores) if isinstance(detail_scores, list) else 0
    c3.metric("Detail scores", detail_score_count)
    c4.metric("Status", str(selected_clause_row.get("status", "")).strip() or "unknown")
    error_text = str(selected_clause_row.get("error", "")).strip()
    if error_text:
        st.warning(f"Evaluation error: {error_text}")

    st.markdown("#### Detail Scores")
    if isinstance(detail_scores, list) and detail_scores:
        st.dataframe(
            [
                {
                    "detail": str(item.get("name", "")).strip(),
                    "score": _to_int(item.get("score", 0)),
                    "reason": str(item.get("reason", "")).strip(),
                }
                for item in detail_scores
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            ],
            use_container_width=True,
            hide_index=True,
            height=260,
        )
    else:
        st.info("No detail scores available for this clause/document row.")

    schema_payload = _load_llm_clause_schema(generosity_root, selected_clause_type)
    rubric_payload = _load_llm_clause_rubric(generosity_root, selected_clause_type)
    distribution_payload = _load_llm_clause_distribution(generosity_root, selected_clause_type)

    with st.expander("Schema Fields", expanded=False):
        fields = schema_payload.get("fields", []) if isinstance(schema_payload, dict) else []
        if isinstance(fields, list) and fields:
            st.dataframe(
                [
                    {
                        "name": str(field.get("name", "")).strip(),
                        "type": str(field.get("type", "")).strip(),
                        "description": str(field.get("description", "")).strip(),
                    }
                    for field in fields
                    if isinstance(field, dict) and str(field.get("name", "")).strip()
                ],
                use_container_width=True,
                hide_index=True,
                height=260,
            )
        else:
            st.info("Schema fields not found for this clause type.")

    with st.expander("Rubric Anchors", expanded=False):
        details = rubric_payload.get("details", []) if isinstance(rubric_payload, dict) else []
        if isinstance(details, list) and details:
            rubric_rows = []
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                anchors = detail.get("scoring_anchors", {})
                if not isinstance(anchors, dict):
                    anchors = {}
                rubric_rows.append(
                    {
                        "detail": str(detail.get("name", "")).strip(),
                        "1": str(anchors.get("1", "")).strip(),
                        "2": str(anchors.get("2", "")).strip(),
                        "3": str(anchors.get("3", "")).strip(),
                        "4": str(anchors.get("4", "")).strip(),
                        "5": str(anchors.get("5", "")).strip(),
                        "guidance": str(detail.get("scoring_guidance", "")).strip(),
                    }
                )
            st.dataframe(rubric_rows, use_container_width=True, hide_index=True, height=320)
        else:
            st.info("Rubric not found for this clause type.")

    with st.expander("Observed Distributions", expanded=False):
        if isinstance(distribution_payload, dict) and distribution_payload:
            st.json(distribution_payload)
        else:
            st.info("Distribution payload not found for this clause type.")

    clause_extraction_rows = _load_llm_clause_extractions(generosity_root, selected_clause_type)
    doc_clause_extraction_rows = [
        row
        for row in clause_extraction_rows
        if str(row.get("document_id", "")).strip() == selected_doc
    ]
    doc_clause_extraction_rows.sort(
        key=lambda row: (
            _to_int(row.get("segment_number", 0)),
            str(row.get("segment_id", "")).strip(),
        )
    )

    st.markdown("#### Segment Extractions")
    if not doc_clause_extraction_rows:
        st.info("No extraction rows found for this document and clause type.")
        return

    st.dataframe(
        [
            {
                "segment": _to_int(row.get("segment_number", 0)),
                "status": str(row.get("status", "")).strip() or "unknown",
                "non_null_fields": (
                    sum(
                        1
                        for value in row.get("fields", {}).values()
                        if value is not None and value != [] and str(value).strip() != ""
                    )
                    if isinstance(row.get("fields", {}), dict)
                    else 0
                ),
                "error": str(row.get("error", "")).strip(),
            }
            for row in doc_clause_extraction_rows[:500]
        ],
        use_container_width=True,
        hide_index=True,
        height=280,
    )
    if len(doc_clause_extraction_rows) > 500:
        st.caption("Showing first 500 extraction rows for this document/clause.")

    segment_options = []
    seen_segment_options = set()
    for row in doc_clause_extraction_rows:
        segment_number = _to_int(row.get("segment_number", 0))
        if segment_number <= 0 or segment_number in seen_segment_options:
            continue
        seen_segment_options.add(segment_number)
        segment_options.append(segment_number)
    if not segment_options:
        st.info("No valid segment numbers found for extraction rows.")
        return
    selected_segment = st.selectbox(
        "Extracted segment",
        options=segment_options,
        format_func=lambda n: f"segment_{_to_int(n, 0)}",
        key=f"app3_llm_segment_for_doc_clause_{selected_doc}_{_llm_slugify(selected_clause_type)}",
    )
    selected_extraction_row = next(
        (
            row
            for row in doc_clause_extraction_rows
            if _to_int(row.get("segment_number", 0)) == _to_int(selected_segment, 0)
        ),
        doc_clause_extraction_rows[0],
    )

    extraction_fields = selected_extraction_row.get("fields", {})
    if not isinstance(extraction_fields, dict):
        extraction_fields = {}
    st.caption("Extracted fields for selected segment")
    st.json(extraction_fields)

    if isinstance(classification_root, Path) and classification_root.exists():
        segment_number = _to_int(selected_extraction_row.get("segment_number", 0))
        if segment_number > 0:
            segment_preview = _load_classification_segment_preview(
                classification_root=classification_root,
                document_id=selected_doc,
                segment_number=segment_number,
                max_chars=2200,
            )
            segment_text = str(segment_preview.get("text", "")).strip()
            if segment_text:
                st.code(segment_text)
                if bool(segment_preview.get("truncated")):
                    st.caption("Preview truncated to first 2,200 characters.")
            else:
                st.info("Segment source text not found in classification output.")
            st.caption(f"Source: `{segment_preview.get('path', '')}`")


def _numeric_or_none(value):
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if np.isnan(numeric):
        return None
    return numeric


def _pearson_corr(x_vals: list[float], y_vals: list[float]):
    if len(x_vals) != len(y_vals) or len(x_vals) < 2:
        return None
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom <= 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denom)


def _average_tie_ranks(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)

    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank_value = ((i + j) / 2.0) + 1.0
        ranks[order[i : j + 1]] = rank_value
        i = j + 1
    return ranks


def _spearman_corr(x_vals: list[float], y_vals: list[float]):
    if len(x_vals) != len(y_vals) or len(x_vals) < 2:
        return None
    x_ranks = _average_tie_ranks(x_vals)
    y_ranks = _average_tie_ranks(y_vals)
    return _pearson_corr(x_ranks.tolist(), y_ranks.tolist())


def _build_ash_metric_doc_values(
    metric_key: str,
    clause_type_rankings: dict[str, list[dict]],
    composite_clause_scores: dict[str, dict[str, float | int | None]],
):
    out: dict[str, float] = {}
    if metric_key == "COMPOSITE_CLAUSE_SCORE":
        for document_id, payload in composite_clause_scores.items():
            if not isinstance(payload, dict):
                continue
            score = _numeric_or_none(payload.get("composite_clause_score"))
            if score is None:
                continue
            out[str(document_id)] = score
        return out

    for row in clause_type_rankings.get(metric_key, []):
        if not isinstance(row, dict):
            continue
        if str(row.get("ratio_status", "undefined")).strip() != "finite":
            continue
        document_id = str(row.get("document_id", "")).strip()
        if not document_id:
            continue
        ratio = _numeric_or_none(row.get("worker_over_firm_ratio"))
        if ratio is None:
            continue
        out[document_id] = ratio
    return out


def _build_gab_metric_doc_values(document_rank_rows: list[dict], metric_key: str = "composite_score"):
    out: dict[str, float] = {}
    for row in document_rank_rows:
        if not isinstance(row, dict):
            continue
        document_id = str(row.get("document_id", "")).strip()
        if not document_id:
            continue
        value = _numeric_or_none(row.get(metric_key))
        if value is None:
            continue
        out[document_id] = value
    return out


def _build_llm_metric_doc_values(document_rank_rows: list[dict], metric_key: str = "document_composite_score"):
    out: dict[str, float] = {}
    for row in document_rank_rows:
        if not isinstance(row, dict):
            continue
        document_id = str(row.get("document_id", "")).strip()
        if not document_id:
            continue
        value = _numeric_or_none(row.get(metric_key))
        if value is None:
            continue
        out[document_id] = value
    return out


def _compare_metric_label(method: str, metric_key: str) -> str:
    method_key = str(method).strip().upper()
    if method_key == "ASH":
        if metric_key == "COMPOSITE_CLAUSE_SCORE":
            return "ASH composite clause score"
        return f"ASH clause ratio: {metric_key}"
    if method_key == "GAB":
        if metric_key == "composite_score":
            return "GAB composite score"
        return "GAB mean segment generosity score"
    if method_key == "LLM":
        if metric_key == "document_composite_score":
            return "LLM document composite score"
        return f"LLM metric: {metric_key}"
    return f"{method_key} metric: {metric_key}"


def _compare_method_metric_map(
    method: str,
    metric_key: str,
    *,
    ash_clause_type_rankings: dict[str, list[dict]],
    ash_composite_scores: dict[str, dict[str, float | int | None]],
    gab_document_rank_rows: list[dict],
    llm_document_rank_rows: list[dict],
):
    method_key = str(method).strip().upper()
    if method_key == "ASH":
        return _build_ash_metric_doc_values(metric_key, ash_clause_type_rankings, ash_composite_scores)
    if method_key == "GAB":
        return _build_gab_metric_doc_values(gab_document_rank_rows, metric_key=metric_key)
    if method_key == "LLM":
        return _build_llm_metric_doc_values(llm_document_rank_rows, metric_key=metric_key)
    return {}


def _render_generosity_compare_view(
    generosity_ash_root: Path,
    generosity_gab_root: Path,
    generosity_llm_root: Path | None = None,
    cba_list_path: Path | None = None,
):
    ash_exists = generosity_ash_root.exists()
    gab_exists = generosity_gab_root.exists()
    llm_exists = isinstance(generosity_llm_root, Path) and generosity_llm_root.exists()

    if not ash_exists and not gab_exists and not llm_exists:
        st.error("No generosity output folders found for ASH, GAB, or LLM.")
        return

    if not ash_exists:
        st.warning(f"04_generosity_ash output folder not found: {generosity_ash_root}")
    if not gab_exists:
        st.warning(f"04_generosity_gab output folder not found: {generosity_gab_root}")
    if isinstance(generosity_llm_root, Path) and not llm_exists:
        st.warning(f"04_generosity_llm output folder not found: {generosity_llm_root}")

    ash_clause_type_rankings = {}
    ash_composite_scores = {}
    if ash_exists:
        ash_summary = _load_generosity_summary(generosity_ash_root)
        ash_clause_type_rankings = _resolve_clause_type_document_rankings(summary=ash_summary, generosity_root=generosity_ash_root)
        ash_composite_scores = _resolve_document_composite_clause_scores(ash_summary, ash_clause_type_rankings)

    gab_document_rank_rows = []
    if gab_exists:
        gab_summary = _load_generosity_summary(generosity_gab_root)
        gab_document_rank_rows = _resolve_gab_document_rankings(gab_summary, _load_gab_document_rankings_csv(generosity_gab_root))

    llm_document_rank_rows = []
    if llm_exists and isinstance(generosity_llm_root, Path):
        llm_document_rank_rows = _load_llm_document_composite_scores_csv(generosity_llm_root)

    document_name_lookup = (
        _load_document_firm_name_lookup(cba_list_path)
        if isinstance(cba_list_path, Path)
        else {}
    )

    ash_metric_options = ["COMPOSITE_CLAUSE_SCORE"] + sorted(
        ash_clause_type_rankings.keys(),
        key=lambda label: (label == "OTHER", str(label).lower()),
    )
    gab_metric_options = ["composite_score", "mean_segment_generosity_score"]
    llm_metric_options = ["document_composite_score"]

    available_methods = []
    if ash_exists:
        available_methods.append("ASH")
    if gab_exists:
        available_methods.append("GAB")
    if llm_exists:
        available_methods.append("LLM")
    if len(available_methods) < 2:
        st.info("Need at least two available generosity outputs to run a comparison.")
        return

    pair_options = []
    for i in range(len(available_methods)):
        for j in range(i + 1, len(available_methods)):
            pair_options.append((available_methods[i], available_methods[j]))

    selected_pair = st.selectbox(
        "Comparison pair",
        options=pair_options,
        format_func=lambda pair: f"{pair[0]} vs {pair[1]}",
        key="app3_compare_pair",
    )
    method_x, method_y = selected_pair

    method_metric_options = {
        "ASH": ash_metric_options,
        "GAB": gab_metric_options,
        "LLM": llm_metric_options,
    }
    method_metric_format = {
        "ASH": lambda key: (
            "Composite clause score (all clause types)"
            if key == "COMPOSITE_CLAUSE_SCORE"
            else f"Clause ratio: {key}"
        ),
        "GAB": lambda key: (
            "Composite score (document ranking)"
            if key == "composite_score"
            else "Mean segment generosity score"
        ),
        "LLM": lambda key: (
            "Document composite score (rubric-based)"
            if key == "document_composite_score"
            else str(key)
        ),
    }

    c_metric_x, c_metric_y = st.columns(2)
    with c_metric_x:
        metric_x = st.selectbox(
            f"{method_x} metric",
            options=method_metric_options[method_x],
            format_func=method_metric_format[method_x],
            key=f"app3_compare_metric_{method_x}_x",
        )
    with c_metric_y:
        metric_y = st.selectbox(
            f"{method_y} metric",
            options=method_metric_options[method_y],
            format_func=method_metric_format[method_y],
            key=f"app3_compare_metric_{method_y}_y",
        )

    x_map = _compare_method_metric_map(
        method_x,
        metric_x,
        ash_clause_type_rankings=ash_clause_type_rankings,
        ash_composite_scores=ash_composite_scores,
        gab_document_rank_rows=gab_document_rank_rows,
        llm_document_rank_rows=llm_document_rank_rows,
    )
    y_map = _compare_method_metric_map(
        method_y,
        metric_y,
        ash_clause_type_rankings=ash_clause_type_rankings,
        ash_composite_scores=ash_composite_scores,
        gab_document_rank_rows=gab_document_rank_rows,
        llm_document_rank_rows=llm_document_rank_rows,
    )

    overlap_doc_ids = sorted(set(x_map.keys()).intersection(y_map.keys()), key=_doc_sort_key)
    x_vals = [float(x_map[doc_id]) for doc_id in overlap_doc_ids]
    y_vals = [float(y_map[doc_id]) for doc_id in overlap_doc_ids]
    pearson = _pearson_corr(x_vals, y_vals)
    spearman = _spearman_corr(x_vals, y_vals)

    x_axis_label = _compare_metric_label(method_x, metric_x)
    y_axis_label = _compare_metric_label(method_y, metric_y)

    st.markdown("### Correlation Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overlap docs", len(overlap_doc_ids))
    c2.metric(f"{method_x} docs with value", len(x_map))
    c3.metric(f"{method_y} docs with value", len(y_map))
    c4.metric("Spearman", f"{spearman:.4f}" if isinstance(spearman, (int, float)) else "n/a")
    p1, p2 = st.columns(2)
    p1.metric("Pearson", f"{pearson:.4f}" if isinstance(pearson, (int, float)) else "n/a")
    p2.metric("Metric pair", f"{method_x} vs {method_y}")

    if len(overlap_doc_ids) < 2:
        st.info(
            "Need at least two overlapping documents with numeric values to compute correlation."
        )
        return

    st.markdown("### Scatter Plot")
    try:
        import matplotlib.pyplot as plt
    except Exception:
        st.info("matplotlib not available for comparison scatter plot.")
    else:
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        ax.scatter(x_vals, y_vals, s=40, alpha=0.8, color="#2563eb")
        if len(x_vals) >= 2:
            try:
                slope, intercept = np.polyfit(np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float), deg=1)
                x_min = min(x_vals)
                x_max = max(x_vals)
                ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], color="#dc2626", linewidth=1.2)
            except Exception:
                pass
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(f"{method_x} vs {method_y} Correlation")
        st.pyplot(fig, use_container_width=True, clear_figure=True)

    st.markdown("### Overlap Table")
    overlap_rows = []
    x_col = f"{method_x.lower()}_value"
    y_col = f"{method_y.lower()}_value"
    diff_col = f"difference ({method_x} - {method_y})"
    for doc_id in overlap_doc_ids:
        overlap_rows.append(
            {
                "firm": _document_display_name(doc_id, document_name_lookup),
                "document_id": doc_id,
                x_col: round(float(x_map[doc_id]), 4),
                y_col: round(float(y_map[doc_id]), 4),
                diff_col: round(float(x_map[doc_id]) - float(y_map[doc_id]), 4),
            }
        )
    overlap_rows.sort(key=lambda row: row[diff_col], reverse=True)
    st.dataframe(overlap_rows, use_container_width=True, hide_index=True, height=380)


def _render_ocr_segments_view(seg_root: Path, ocr_root: Path, pdf_root: Path):
    if not ocr_root.exists():
        st.error(f"OCR output folder not found: {ocr_root}")
        return
    if not seg_root.exists():
        st.error(f"Segmentation output folder not found: {seg_root}")
        return

    ocr_docs = _list_ocr_doc_ids(ocr_root)
    seg_docs = set(_list_seg_doc_ids(seg_root))
    doc_ids = [doc_id for doc_id in ocr_docs if doc_id in seg_docs]

    if not doc_ids:
        st.warning("No documents found that have both OCR pages and segmentation metadata.")
        return

    selected_doc = st.sidebar.selectbox("Document", doc_ids, key="app3_doc")
    page_bundle = _load_doc_page_text_and_spans(ocr_root / selected_doc)
    page_numbers = page_bundle["page_numbers"]
    if not page_numbers:
        st.warning("No OCR pages found for selected document.")
        return

    if st.session_state.get("app3_page_doc") != selected_doc:
        st.session_state["app3_page_doc"] = selected_doc
        st.session_state["app3_page_input"] = 1

    total_pages = len(page_numbers)
    if "app3_page_input" not in st.session_state:
        st.session_state["app3_page_input"] = 1
    if st.session_state["app3_page_input"] < 1:
        st.session_state["app3_page_input"] = 1
    if st.session_state["app3_page_input"] > total_pages:
        st.session_state["app3_page_input"] = total_pages

    page_position = int(
        st.sidebar.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            step=1,
            key="app3_page_input",
        )
    )
    # Match app.py behavior: OCR file index is one step ahead of selected page index.
    ocr_page_idx = min(page_position, total_pages - 1)
    selected_page = int(page_numbers[ocr_page_idx])
    st.sidebar.caption(f"OCR file page number: {selected_page}")

    segments = _load_doc_segments(seg_root, selected_doc)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**OCR pages:** {total_pages}")
    st.sidebar.markdown(f"**Total segments:** {len(segments)}")

    page_text = page_bundle["text_by_page"].get(selected_page, "")
    page_span = page_bundle["span_by_page"].get(selected_page)
    segments_for_page_input = [
        {
            "start_pos": seg["start"],
            "end_pos": seg["end"],
            "title": f"segment_{seg['number']}",
            "parent": "runner.py output",
        }
        for seg in segments
    ]
    page_segments = _segmentation_segments_for_page(
        segments=segments_for_page_input,
        page_text=page_text,
        page_span=page_span,
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader(f"{selected_doc} — Page {page_position} of {total_pages}")
        pdf_path = pdf_root / f"{selected_doc}.pdf"
        if pdf_path.exists():
            try:
                st.pdf(load_pdf_page(pdf_path, page_position - 1), height=900)
            except Exception as exc:
                st.error(f"Failed to render PDF page: {exc}")
        else:
            st.info(f"Source PDF not found: {pdf_path.name}")

    with col_right:
        st.subheader("OCR Text with Segment Highlights")
        _render_segmentation_highlights(page_text, page_segments, height=800)
        st.caption(
            f"{len(page_segments)} highlighted segment spans on page {page_position} "
            f"({len(segments)} total segments in document)."
        )


@st.cache_data(show_spinner=False)
def _load_cluster_segments(cluster_root: Path):
    rows = []
    if not cluster_root.exists():
        return rows

    def _segment_sort_key(path: Path):
        m = re.match(r"segment_(\d+)\.json$", path.name)
        return int(m.group(1)) if m else 10**9

    doc_dirs = sorted(
        [d for d in cluster_root.iterdir() if d.is_dir() and re.fullmatch(r"document_\d+", d.name)],
        key=lambda d: _doc_sort_key(d.name),
    )
    for doc_dir in doc_dirs:
        for segment_path in sorted(doc_dir.glob("segment_*.json"), key=_segment_sort_key):
            payload = _safe_read_json(segment_path)
            if not isinstance(payload, dict):
                continue
            cluster_id = payload.get("cluster_id")
            try:
                cluster_id = int(cluster_id)
            except Exception:
                continue

            segment_number = payload.get("segment_number")
            if not isinstance(segment_number, int):
                segment_number = _segment_sort_key(segment_path)

            cluster_score = payload.get("cluster_score")
            try:
                cluster_score = float(cluster_score) if cluster_score is not None else None
            except Exception:
                cluster_score = None

            rows.append({
                "document_id": str(payload.get("document_id", doc_dir.name)),
                "segment_number": int(segment_number),
                "cluster_id": cluster_id,
                "cluster_score": cluster_score,
                "segment_text": str(payload.get("segment_text", "")).strip(),
                "topics": payload.get("topics", []),
            })
    return rows


@st.cache_data(show_spinner=False)
def _build_cluster_heatmap(rows: list[dict]):
    doc_ids = sorted({str(row["document_id"]) for row in rows}, key=_doc_sort_key)
    cluster_ids = sorted({int(row["cluster_id"]) for row in rows})
    if not doc_ids or not cluster_ids:
        return doc_ids, cluster_ids, np.zeros((0, 0))

    doc_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    cluster_index = {cluster_id: i for i, cluster_id in enumerate(cluster_ids)}
    counts = np.zeros((len(doc_ids), len(cluster_ids)), dtype=float)

    for row in rows:
        di = doc_index.get(str(row["document_id"]))
        ci = cluster_index.get(int(row["cluster_id"]))
        if di is None or ci is None:
            continue
        counts[di, ci] += 1.0

    # Light Gaussian-like smoothing over neighboring cells.
    kernel = np.array(
        [[1.0, 2.0, 1.0],
         [2.0, 4.0, 2.0],
         [1.0, 2.0, 1.0]],
        dtype=float,
    )
    kernel /= kernel.sum()
    padded = np.pad(counts, ((1, 1), (1, 1)), mode="edge")
    smoothed = np.zeros_like(counts)
    for r in range(counts.shape[0]):
        for c in range(counts.shape[1]):
            smoothed[r, c] = float((padded[r:r + 3, c:c + 3] * kernel).sum())

    return doc_ids, cluster_ids, smoothed


def _render_cluster_heatmap(rows: list[dict]):
    doc_ids, cluster_ids, heatmap = _build_cluster_heatmap(rows)
    if heatmap.size == 0:
        st.info("No cluster heatmap data available.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        st.info("matplotlib not available for heatmap rendering.")
        return

    fig_h = min(14.0, max(4.0, len(doc_ids) / 180))
    fig_w = min(16.0, max(7.0, len(cluster_ids) / 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heatmap, aspect="auto", interpolation="bilinear", cmap="YlOrRd")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Document")
    ax.set_title("Smoothed Segment Count Heatmap")

    x_step = max(1, len(cluster_ids) // 20)
    x_ticks = np.arange(0, len(cluster_ids), x_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(cluster_ids[i]) for i in x_ticks], rotation=45, ha="right", fontsize=8)

    y_step = max(1, len(doc_ids) // 25)
    y_ticks = np.arange(0, len(doc_ids), y_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([doc_ids[i] for i in y_ticks], fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Smoothed segment count")
    st.pyplot(fig, use_container_width=True, clear_figure=True)


def _render_cluster_examples_view(cluster_root: Path):
    if not cluster_root.exists():
        st.error(f"Cluster output folder not found: {cluster_root}")
        return

    rows = _load_cluster_segments(cluster_root)
    if not rows:
        st.warning("No cluster segment JSON files found.")
        return

    st.markdown("### Cluster Heatmap")
    _render_cluster_heatmap(rows)

    st.markdown("### Cluster Viewer")
    cluster_ids = sorted({row["cluster_id"] for row in rows})
    selected_cluster = st.selectbox("Cluster", options=cluster_ids, key="app3_cluster_single")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Clusters available:** {len(cluster_ids)}")
    st.sidebar.markdown(f"**Clustered segments:** {len(rows)}")

    cluster_rows = [row for row in rows if row["cluster_id"] == selected_cluster]
    cluster_rows.sort(
        key=lambda row: row["cluster_score"] if row["cluster_score"] is not None else float("-inf"),
        reverse=True,
    )
    st.caption(f"Cluster {selected_cluster}: {len(cluster_rows)} segments")

    def _primary_topic_for_cluster(row: dict):
        topics = row["topics"] if isinstance(row["topics"], list) else []
        for topic in topics:
            if isinstance(topic, dict) and topic.get("topic_num") == selected_cluster:
                return topic
        if topics and isinstance(topics[0], dict):
            return topics[0]
        return None

    # Word cloud
    word_weights: dict[str, float] = {}
    for row in cluster_rows:
        topic = _primary_topic_for_cluster(row)
        if not isinstance(topic, dict):
            continue
        words = topic.get("topic_words", [])
        scores = topic.get("topic_word_scores", [])
        if not isinstance(words, list):
            continue
        for i, word in enumerate(words):
            if not isinstance(word, str):
                continue
            score = None
            if isinstance(scores, list) and i < len(scores):
                try:
                    score = float(scores[i])
                except Exception:
                    score = None
            if score is None:
                score = 1.0 / (i + 1)
            word_weights[word] = max(word_weights.get(word, 0.0), score)

    st.markdown("### Word Cloud")
    if word_weights:
        palette = ["#0f766e", "#1d4ed8", "#be123c", "#7c2d12", "#1f2937", "#4338ca", "#166534"]
        items = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)[:60]
        min_w = min(weight for _, weight in items)
        max_w = max(weight for _, weight in items)
        denom = max(max_w - min_w, 1e-9)
        spans = []
        for word, weight in items:
            size = 14 + ((weight - min_w) / denom) * 34
            color = palette[abs(hash(word)) % len(palette)]
            spans.append(
                f'<span style="display:inline-block; margin:0.2rem 0.45rem; '
                f'font-size:{size:.1f}px; color:{color}; font-weight:600;">'
                f'{html.escape(word)}</span>'
            )
        cloud_html = (
            '<div style="border:1px solid #e5e7eb; border-radius:8px; padding:0.8rem; '
            'background:#ffffff; line-height:1.15;">'
            + "".join(spans)
            + "</div>"
        )
        components.html(cloud_html, height=260, scrolling=False)
    else:
        st.info("No topic words found for this cluster.")

    st.markdown("### Example Segments")
    seed_key = f"app3_cluster_seed_{selected_cluster}"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = 0
    if st.button("Randomize examples", key=f"app3_randomize_{selected_cluster}"):
        st.session_state[seed_key] += 1

    if not cluster_rows:
        st.info("No segments found for this cluster.")
        return

    if len(cluster_rows) <= 3:
        example_rows = cluster_rows
    else:
        rng = random.Random(f"{selected_cluster}:{st.session_state[seed_key]}")
        example_rows = rng.sample(cluster_rows, 3)

    for row in example_rows:
        title = f'{row["document_id"]} / segment_{row["segment_number"]}'
        if row["cluster_score"] is not None:
            title += f' · score={row["cluster_score"]:.3f}'
        with st.expander(title, expanded=True):
            topic = _primary_topic_for_cluster(row)
            if isinstance(topic, dict):
                words = topic.get("topic_words", [])
                if isinstance(words, list) and words:
                    st.caption(f"Topic words: {', '.join(str(w) for w in words[:12])}")
            st.write(row["segment_text"])


def main():
    """Launch the main review dashboard for pipeline artifacts."""
    st.set_page_config(page_title="OCR + Segments Review", layout="wide")
    st.title("OCR + Segments Review")

    cache_root = DEFAULT_CACHE_DIR.expanduser().resolve()
    seg_root = (cache_root / "02_segmentation_output" / DEFAULT_COLLECTION).resolve()
    ocr_root = (cache_root / "01_ocr_output" / DEFAULT_COLLECTION).resolve()
    pdf_root = (cache_root / DEFAULT_COLLECTION).resolve()
    clause_root = _resolve_clause_root(cache_root, DEFAULT_COLLECTION).resolve()
    generosity_ash_root = _resolve_generosity_ash_root(cache_root, DEFAULT_COLLECTION).resolve()
    generosity_gab_root = _resolve_generosity_gab_root(cache_root, DEFAULT_COLLECTION).resolve()
    generosity_llm_root = _resolve_generosity_llm_root(cache_root, DEFAULT_COLLECTION).resolve()

    with st.sidebar:
        st.header("Inputs")
        view = st.radio(
            "View",
            [
                "OCR + Segments",
                "Clause Classification",
                "Generosity (ASH)",
                "Generosity (GAB)",
                "Generosity (LLM)",
                "Generosity Compare",
            ],
            key="app3_view",
        )

    if view == "OCR + Segments":
        st.caption("Side-by-side PDF preview and OCR text with segmentation highlights.")
        _render_ocr_segments_view(seg_root=seg_root, ocr_root=ocr_root, pdf_root=pdf_root)
    elif view == "Clause Classification":
        st.caption("Review segment-level clause classifications with retrieval candidates and rationale.")
        _render_clause_classification_view(clause_root=clause_root)
    elif view == "Generosity (ASH)":
        st.caption(
            "Review sentence-level auth categorization outputs from pipeline/04_generosity_ash/runner.py."
        )
        _render_generosity_ash_view(
            generosity_root=generosity_ash_root,
            cba_list_path=pdf_root / "CBAList_fixed.dta",
        )
    elif view == "Generosity (GAB)":
        st.caption(
            "Review segment-ranking generosity outputs from pipeline/04_generosity_gab/runner.py."
        )
        _render_generosity_gab_view(
            generosity_root=generosity_gab_root,
            cba_list_path=pdf_root / "CBAList_fixed.dta",
            segmentation_root=seg_root,
        )
    elif view == "Generosity (LLM)":
        st.caption(
            "Review rubric-based generosity outputs from pipeline/04_generosity_llm/runner.py."
        )
        _render_generosity_llm_view(
            generosity_root=generosity_llm_root,
            cba_list_path=pdf_root / "CBAList_fixed.dta",
            classification_root=clause_root,
        )
    else:
        st.caption(
            "Compare ASH, GAB, and LLM document-level generosity metrics and correlations."
        )
        _render_generosity_compare_view(
            generosity_ash_root=generosity_ash_root,
            generosity_gab_root=generosity_gab_root,
            generosity_llm_root=generosity_llm_root,
            cba_list_path=pdf_root / "CBAList_fixed.dta",
        )


if __name__ == "__main__":
    main()
