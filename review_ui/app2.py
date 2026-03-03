import json
import os
import re
import html
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR_ENV = os.environ.get("CACHE_DIR")
DEFAULT_CACHE_DIR = Path(CACHE_DIR_ENV).expanduser() if CACHE_DIR_ENV else APP_DIR.parent
DEFAULT_COLLECTION = os.environ.get("SEGMENT_COLLECTION", "dol_archive")
DEFAULT_SEGMENTATION_DIR = DEFAULT_CACHE_DIR / "02_segmentation_output" / DEFAULT_COLLECTION
DEFAULT_OCR_DIR = DEFAULT_CACHE_DIR / "01_ocr_output" / DEFAULT_COLLECTION


def _safe_read_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_read_text(path: Path):
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _sorted_page_paths(doc_dir: Path):
    pages = list(doc_dir.glob("page_*.txt"))

    def _page_num(p: Path):
        try:
            return int(p.stem.split("_")[1])
        except Exception:
            return 10**9

    return sorted(pages, key=_page_num)


def _is_document_dir(path: Path) -> bool:
    return path.is_dir() and re.fullmatch(r"document_\d+", path.name) is not None


def _doc_sort_key(doc_id: str):
    m = re.fullmatch(r"document_(\d+)", doc_id)
    if not m:
        return (10**9, doc_id)
    return (int(m.group(1)), doc_id)


def _list_doc_ids(root: Path):
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        [p.name for p in root.iterdir() if _is_document_dir(p)],
        key=_doc_sort_key,
    )


def _resolve_doc_root(root: Path, preferred_collection: str | None = None):
    notes = []
    if _is_document_dir(root):
        notes.append(f"Using single document directory: {root}")
        return root.parent, [root.name], notes

    direct_doc_ids = _list_doc_ids(root)
    if direct_doc_ids:
        return root, direct_doc_ids, notes

    if not root.exists() or not root.is_dir():
        return root, [], notes

    collection_candidates: list[tuple[Path, list[str]]] = []
    for subdir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        sub_doc_ids = _list_doc_ids(subdir)
        if sub_doc_ids:
            collection_candidates.append((subdir, sub_doc_ids))

    if not collection_candidates:
        return root, [], notes

    selected: tuple[Path, list[str]] | None = None
    if preferred_collection:
        for candidate in collection_candidates:
            if candidate[0].name == preferred_collection:
                selected = candidate
                break

    if selected is None:
        if len(collection_candidates) == 1:
            selected = collection_candidates[0]
        else:
            selected = collection_candidates[0]
            notes.append(
                f"Multiple collections found under {root}; defaulting to '{selected[0].name}'."
            )

    notes.append(f"Using collection directory: {selected[0]}")
    return selected[0], selected[1], notes


def _load_doc(seg_dir: Path, doc_id: str):
    doc_dir = seg_dir / doc_id
    return {
        "meta": _safe_read_json(doc_dir / "document_meta.json"),
        "full_text": _safe_read_text(doc_dir / "full_text.txt"),
        "boundary_evaluations": _safe_read_json(doc_dir / "boundary_evaluations.json"),
        "segments_dir": doc_dir / "segments",
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


def _segment_issues(segments):
    issues = []
    prev_end = None
    for seg in segments:
        if seg["start"] > seg["end"]:
            issues.append(f"segment_{seg['number']}: start > end")
        if seg["end"] - seg["start"] != seg["length"]:
            issues.append(f"segment_{seg['number']}: span length mismatch")
        if prev_end is not None and seg["start"] < prev_end:
            issues.append(f"segment_{seg['number']}: overlaps previous segment")
        prev_end = seg["end"]
    return issues


def _context(full_text: str, start: int, end: int, window: int):
    left = max(0, start - window)
    right = min(len(full_text), end + window)
    pre = full_text[left:start]
    marker = full_text[start:end]
    post = full_text[end:right]
    return pre, marker, post


def _is_new_segment(eval_payload: dict) -> bool:
    # Prefer the new schema key, but tolerate older cached payloads.
    if not isinstance(eval_payload, dict):
        return False
    if "is_new_segment" in eval_payload:
        return eval_payload.get("is_new_segment") is True
    return eval_payload.get("is_boundary") is True


def _boundary_question(plan: dict) -> str:
    segment_type = str(plan.get("segment_type", "")).strip() if isinstance(plan, dict) else ""
    if not segment_type:
        segment_type = "segment"
    return (
        f"Identify whether the boundary marked by <BOUNDARY/> is a valid boundary "
        f"between two {segment_type} of the document."
    )


def _collect_boundary_candidates(full_text: str, plan: dict):
    candidates = []
    notes = []
    rules = plan.get("segment_header_rules") if isinstance(plan, dict) else None

    if isinstance(rules, list):
        for idx, rule in enumerate(rules, start=1):
            if not isinstance(rule, str) or not rule.strip():
                notes.append(f"Skipped empty/non-string regex rule at index {idx}.")
                continue
            try:
                candidates.extend(re.finditer(rule, full_text, flags=re.MULTILINE))
            except re.error as err:
                notes.append(f"Invalid regex rule #{idx}: {err}.")

    if candidates:
        # Keep one match per (start, end) in sorted order.
        deduped = []
        seen = set()
        for m in sorted(candidates, key=lambda x: (x.start(), x.end())):
            key = (m.start(), m.end())
            if key in seen:
                continue
            deduped.append(m)
            seen.add(key)
        return deduped, notes, "plan rules"

    fallback = list(re.finditer(r"\n\n", full_text))
    if isinstance(rules, list) and rules:
        notes.append(
            "No usable matches from plan regex rules; fell back to double-newline candidates."
        )
    return fallback, notes, "double-newline fallback"


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
                "text": str(seg.get("text", "")).strip(),
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


def main():
    st.set_page_config(page_title="Segmentation Review", layout="wide")
    st.title("Segmentation Review")
    st.caption("Review outputs produced by pipeline/02_segment/runner.py")

    with st.sidebar:
        st.header("Inputs")
        seg_root_input = Path(
            st.text_input("Segmentation output dir", value=str(DEFAULT_SEGMENTATION_DIR))
        )
        ocr_root_input = Path(st.text_input("OCR input dir (optional)", value=str(DEFAULT_OCR_DIR)))

    if not seg_root_input.exists():
        st.error(f"Segmentation directory does not exist: {seg_root_input}")
        return

    seg_root, doc_ids, seg_root_notes = _resolve_doc_root(
        seg_root_input,
        preferred_collection=DEFAULT_COLLECTION,
    )
    ocr_root, _, ocr_root_notes = _resolve_doc_root(
        ocr_root_input,
        preferred_collection=seg_root.name if seg_root != seg_root_input else DEFAULT_COLLECTION,
    )

    if not doc_ids:
        st.warning(f"No document folders found in {seg_root_input}")
        return

    with st.sidebar:
        for note in seg_root_notes:
            st.caption(note)
        for note in ocr_root_notes:
            st.caption(note)
        st.caption(f"Seg root: {seg_root}")
        st.caption(f"OCR root: {ocr_root}")
        doc_id = st.selectbox("Document", options=doc_ids)

    payload = _load_doc(seg_root, doc_id)

    full_text = payload["full_text"] or ""
    meta = payload["meta"] or {}
    boundary_evaluations = payload["boundary_evaluations"]
    plan = meta.get("plan") if isinstance(meta.get("plan"), dict) else {}
    top_level_type = str(plan.get("top_level_type", "")).strip()
    above_top_level_key = f"above_Articles" if top_level_type else None

    if not full_text:
        st.error("Missing or unreadable full_text.txt")
        return

    candidates, candidate_notes, candidate_source = _collect_boundary_candidates(full_text, plan)
    n_candidates = len(candidates)

    if not isinstance(boundary_evaluations, list):
        boundary_evaluations = []

    n_eval = len(boundary_evaluations)
    if n_eval != n_candidates:
        st.warning(
            f"Candidate/evaluation count mismatch: {n_candidates} candidates vs {n_eval} evaluations."
        )

    n_aligned = min(n_candidates, n_eval)
    new_segment_idxs = [
        i
        for i in range(n_aligned)
        if isinstance(boundary_evaluations[i], dict)
        and _is_new_segment(boundary_evaluations[i])
    ]

    segments = _normalize_segments((meta.get("segments") or {}))
    seg_issues = _segment_issues(segments)
    ocr_doc = ocr_root / doc_id
    page_bundle = _load_doc_page_text_and_spans(ocr_doc)
    page_numbers = page_bundle["page_numbers"]

    selected_page = None
    with st.sidebar:
        st.header("Page Navigation")
        if page_numbers:
            default_page = st.session_state.get("segments_page_sidebar", int(page_numbers[0]))
            if default_page < page_numbers[0] or default_page > page_numbers[-1]:
                default_page = int(page_numbers[0])

            page_input = int(
                st.number_input(
                    "Page",
                    min_value=int(page_numbers[0]),
                    max_value=int(page_numbers[-1]),
                    value=int(default_page),
                    step=1,
                )
            )
            selected_page = min(page_numbers, key=lambda p: abs(p - page_input))
            st.session_state["segments_page_sidebar"] = selected_page
            if selected_page != page_input:
                st.caption(f"Snapped to nearest available page: {selected_page}")
            page_pos = page_numbers.index(selected_page) + 1
            st.caption(f"Viewing page {selected_page} ({page_pos}/{len(page_numbers)})")
        else:
            st.caption("No OCR pages detected for this document.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", n_candidates)
    c2.metric("Evaluations", n_eval)
    c3.metric("New segments", len(new_segment_idxs))
    c4.metric("Segments", len(segments))
    st.caption(f"Candidate extraction source: {candidate_source}")
    for note in candidate_notes:
        st.warning(note)

    with st.expander("Plan + Integrity", expanded=False):
        st.subheader("Plan")
        st.json(meta.get("plan", {}))

        st.subheader("Segment integrity checks")
        if seg_issues:
            for issue in seg_issues:
                st.error(issue)
        else:
            st.success("No overlap or span-length issues found in segment metadata.")

    st.subheader("Segment Break Evaluations")
    st.caption(f"Model question: {_boundary_question(plan)}")
    mode = st.radio(
        "Show",
        options=["new_segment", "not_new_segment", "all"],
        horizontal=True,
        index=0,
    )
    context_window = st.slider("Context window (chars)", min_value=100, max_value=2000, value=400, step=100)

    filtered = []
    for i in range(n_aligned):
        p = boundary_evaluations[i]
        if not isinstance(p, dict):
            continue
        is_new_segment = _is_new_segment(p)

        if mode == "new_segment" and not is_new_segment:
            continue
        if mode == "not_new_segment" and is_new_segment:
            continue
        filtered.append(i)

    st.caption(f"Showing {len(filtered)} of {n_aligned} aligned evaluations")

    if not filtered:
        st.info("No boundaries match current filters.")
    else:
        selected_pos = st.number_input(
            "Filtered boundary index",
            min_value=1,
            max_value=len(filtered),
            value=1,
            step=1,
        )
        idx = filtered[selected_pos - 1]
        match = candidates[idx]
        eval_payload = boundary_evaluations[idx]

        pre, marker, post = _context(full_text, match.start(), match.end(), context_window)

        left, right = st.columns([2, 1])
        with left:
            st.code(pre + "<BOUNDARY/>" + marker + post)
        with right:
            st.write(f"Candidate #{idx}")
            st.write(f"start={match.start()}, end={match.end()}")
            st.write(f"is_new_segment={_is_new_segment(eval_payload)}")
            if above_top_level_key:
                st.write(f"{above_top_level_key}={eval_payload.get(above_top_level_key)}")
            st.write("explanation:")
            st.write(eval_payload.get("explanation", ""))

    st.subheader("Segments")

    if not segments:
        st.info("No segments found in document_meta.json")
    elif not page_numbers:
        st.info("No OCR page text files found, so page-level highlighting is unavailable.")
    else:
        page_text = page_bundle["text_by_page"].get(selected_page, "")
        page_span = page_bundle["span_by_page"].get(selected_page)

        segments_for_page_input = [
            {
                "start_pos": seg["start"],
                "end_pos": seg["end"],
                "title": f"segment_{seg['number']}",
                "parent": "runner.py output",
                "text": full_text[seg["start"]:seg["end"]],
            }
            for seg in segments
        ]
        page_segments = _segmentation_segments_for_page(
            segments=segments_for_page_input,
            page_text=page_text,
            page_span=page_span,
        )

        st.caption(f"{len(page_segments)} segments highlighted on page {selected_page}")
        _render_segmentation_highlights(page_text, page_segments, height=520)
        if page_segments:
            for seg in page_segments:
                with st.expander(
                    f"{seg['title']} | [{seg['local_start']}, {seg['local_end']})",
                    expanded=False,
                ):
                    st.text(page_text[seg["local_start"]:seg["local_end"]][:2500])


if __name__ == "__main__":
    main()
