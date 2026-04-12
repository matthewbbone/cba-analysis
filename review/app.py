import html
import json
import os
from pathlib import Path
from typing import Any

import fitz
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
DEFAULT_CACHE_DIR = ROOT_DIR / "cba_cache"
DOL_GROUPS = ["dol_archive", "cornell_dol", "cornell_retail_educ"]

OCR_SYSTEM_PROMPT = " ".join(
    [
        "You are a helpful and precise assistant for transcribing the text",
        "of collective bargaining agreements. You are given a single page of",
        "a PDF document as an image, and your task is to extract the text content",
        "as accurately as possible while preserving the original formatting and structure.",
    ]
)

OCR_USER_PROMPT = " ".join(
    [
        "Transcribe the document image into markdown.",
        "Any visually distinct header text that indicates a new article, preamble, or table of contents should be marked as a header in markdown with '##'",
        "Return the markdown text in the following json format: { 'transcribed_text': '...' }",
    ]
)

PROVISION_SYSTEM_PROMPT_TEMPLATE = " ".join(
    [
        "You are a legal assistant tasked with extracting provisions",
        "and identifying who enacts them, who benefits from them,",
        "any stated conditions, and the substantive value of the provision.",
        "The legal parties you should identify are configured in the runner.",
        "Return your response in a JSON format with the following schema:",
        "{provisions: [{'subject': the party that enacts the provision, 'beneficiary': the party that benefits from the provision, 'conditions': the stated conditions or 'None', 'value': the substantive obligation, prohibition, permission, or right itself, 'span': a minimal verbatim contiguous substring grounding the provision}]}",
        "If there are no provisions with clear subjects and beneficiaries in the text, return {provisions: []}.",
        "If you cannot provide an exact verbatim span for a provision, omit that provision.",
    ]
)

PROVISION_USER_PROMPT = "Extract and categorize the provisions in the following text:"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


@st.cache_data
def get_cache_dir() -> Path:
    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir:
        resolved_cache_dir = Path(cache_dir).resolve()
        if resolved_cache_dir.exists():
            return resolved_cache_dir
    return DEFAULT_CACHE_DIR.resolve()


@st.cache_data
def list_pdfs(group: str) -> list[Path]:
    pdf_dir = get_cache_dir() / group
    return sorted(pdf_dir.glob("*.pdf"))


@st.cache_data
def list_model_cache_files(group: str) -> list[Path]:
    output_dir = get_cache_dir() / "01_ocr_output" / group
    return sorted(output_dir.glob("*/cache.json"))


@st.cache_data
def list_provision_model_cache_files(group: str) -> list[Path]:
    output_dir = get_cache_dir() / "02_provision_extract" / group
    return sorted(output_dir.glob("*/cache.json"))


@st.cache_data
def list_models(group: str) -> list[str]:
    return [cache_file.parent.name for cache_file in list_model_cache_files(group)]


@st.cache_data
def list_provision_models(group: str) -> list[str]:
    return [cache_file.parent.name for cache_file in list_provision_model_cache_files(group)]


@st.cache_data
def list_cached_documents(group: str) -> list[str]:
    document_ids: set[str] = set()
    for cache_file in list_model_cache_files(group):
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        document_ids.update(cache.get("documents", {}).keys())
    return sorted(document_ids)


@st.cache_data
def list_provision_cached_documents(group: str) -> list[str]:
    document_ids: set[str] = set()
    for cache_file in list_provision_model_cache_files(group):
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        document_ids.update(cache.get("documents", {}).keys())
    return sorted(document_ids)


@st.cache_data
def read_binary_file(path: str) -> bytes | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


@st.cache_data
def get_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


@st.cache_data
def render_page(pdf_path: str, page_index: int) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        pixmap = page.get_pixmap(dpi=200)
        return pixmap.tobytes("png")
    finally:
        doc.close()


@st.cache_data
def read_json_file(path: str) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


@st.cache_data
def read_provisions(group: str, model_name: str, document_stem: str) -> dict[str, Any] | None:
    provisions_file = (
        get_cache_dir()
        / "02_provision_extract"
        / group
        / model_name
        / document_stem
        / "provisions.json"
    )
    return read_json_file(str(provisions_file))


@st.cache_data
def read_ocr_page(group: str, model_name: str, document_stem: str, page_index: int) -> str | None:
    page_file = (
        get_cache_dir()
        / "01_ocr_output"
        / group
        / model_name
        / document_stem
        / f"page_{page_index:04d}.txt"
    )
    if not page_file.exists():
        return None
    return page_file.read_text(encoding="utf-8")


def _display_party(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return text


def _display_text(value: Any, default: str = "None") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return text


def _normalize_party_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "worker": "worker",
        "workers": "worker",
        "firm": "firm",
        "firms": "firm",
        "union": "union",
        "unions": "union",
        "manager": "manager",
        "managers": "manager",
    }
    if text in aliases:
        return aliases[text]
    return "unknown"


def _get_provision_subject(provision: dict[str, Any]) -> Any:
    return provision.get("subject", provision.get("actor"))


def _get_provision_beneficiary(provision: dict[str, Any]) -> Any:
    return provision.get("beneficiary")


def _get_provision_conditions(provision: dict[str, Any]) -> Any:
    return provision.get("conditions")


def _get_provision_value(provision: dict[str, Any]) -> Any:
    return provision.get("value", provision.get("text"))


def _get_provision_span(provision: dict[str, Any]) -> str:
    return str(provision.get("span", provision.get("text", "")) or "")


def _get_provision_span_start(provision: dict[str, Any]) -> int | None:
    value = provision.get("span_start")
    return value if isinstance(value, int) else None


def _get_provision_span_end(provision: dict[str, Any]) -> int | None:
    value = provision.get("span_end")
    return value if isinstance(value, int) else None


def _get_provision_grounding_status(provision: dict[str, Any]) -> str:
    status = str(provision.get("grounding_status", "") or "").strip().lower()
    if status in {"exact", "unresolved"}:
        return status
    if (
        isinstance(_get_provision_span_start(provision), int)
        and isinstance(_get_provision_span_end(provision), int)
    ):
        return "exact"
    return "unresolved"


def _is_grounded_provision(provision: dict[str, Any], section_text: str) -> bool:
    start = _get_provision_span_start(provision)
    end = _get_provision_span_end(provision)
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    return 0 <= start < end <= len(section_text)


def _display_party_label_from_key(value: str) -> str:
    normalized = _normalize_party_key(value)
    return normalized.capitalize() if normalized != "unknown" else "Unknown"


_PARTY_HIGHLIGHT_COLORS = {
    "worker": "#D2E3FC",
    "firm": "#FFDDBE",
    "union": "#C8E6C9",
    "manager": "#EADDFF",
    "unknown": "#E8EAED",
}


def _highlight_color_for_provision(provision: dict[str, Any]) -> str:
    beneficiary_key = _normalize_party_key(_get_provision_beneficiary(provision))
    return _PARTY_HIGHLIGHT_COLORS.get(beneficiary_key, _PARTY_HIGHLIGHT_COLORS["unknown"])


def _get_grounded_section_provisions(section: dict[str, Any]) -> list[dict[str, Any]]:
    section_text = str(section.get("text", "") or "")
    provisions = section.get("provisions", [])
    if not isinstance(provisions, list):
        return []
    return [
        provision
        for provision in provisions
        if isinstance(provision, dict) and _is_grounded_provision(provision, section_text)
    ]


def _get_ungrounded_section_provisions(section: dict[str, Any]) -> list[dict[str, Any]]:
    section_text = str(section.get("text", "") or "")
    provisions = section.get("provisions", [])
    if not isinstance(provisions, list):
        return []
    return [
        provision
        for provision in provisions
        if isinstance(provision, dict) and not _is_grounded_provision(provision, section_text)
    ]


def _build_provision_attributes_html(provision: dict[str, Any]) -> str:
    subject = html.escape(_display_party(_get_provision_subject(provision)))
    beneficiary = html.escape(_display_party(_get_provision_beneficiary(provision)))
    conditions = html.escape(_display_text(_get_provision_conditions(provision)))
    value = html.escape(_display_text(_get_provision_value(provision)))

    return "".join(
        [
            "<div><strong>Subject:</strong> ",
            subject,
            "</div>",
            "<div><strong>Beneficiary:</strong> ",
            beneficiary,
            "</div>",
            "<div><strong>Conditions:</strong> ",
            conditions,
            "</div>",
            "<div><strong>Value:</strong> ",
            value,
            "</div>",
        ]
    )


def _build_provision_viewer_html(section_text: str, grounded_provisions: list[dict[str, Any]]) -> str:
    sorted_provisions = sorted(
        grounded_provisions,
        key=lambda provision: (
            _get_provision_span_start(provision) or 0,
            -((_get_provision_span_end(provision) or 0) - (_get_provision_span_start(provision) or 0)),
        ),
    )

    span_lengths: dict[int, int] = {}
    points: list[tuple[int, int, int, dict[str, Any]]] = []
    for index, provision in enumerate(sorted_provisions):
        start = _get_provision_span_start(provision)
        end = _get_provision_span_end(provision)
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            continue
        span_lengths[index] = end - start
        points.append((start, 1, index, provision))
        points.append((end, 0, index, provision))

    def _sort_point(point: tuple[int, int, int, dict[str, Any]]) -> tuple[int, int, int]:
        position, boundary_type, span_index, _ = point
        span_length = span_lengths.get(span_index, 0)
        if boundary_type == 0:
            return position, 0, span_length
        return position, 1, -span_length

    points.sort(key=_sort_point)

    html_parts: list[str] = []
    cursor = 0
    for position, boundary_type, span_index, provision in points:
        if position > cursor:
            html_parts.append(html.escape(section_text[cursor:position]))

        if boundary_type == 1:
            color = _highlight_color_for_provision(provision)
            html_parts.append(
                f'<span class="px-highlight" data-idx="{span_index}" '
                f'style="background-color:{color};">'
            )
        else:
            html_parts.append("</span>")
        cursor = position

    if cursor < len(section_text):
        html_parts.append(html.escape(section_text[cursor:]))

    highlighted_text = "".join(html_parts)

    legend_items = []
    for party_key in ["worker", "firm", "union", "manager", "unknown"]:
        legend_items.append(
            '<span class="px-legend-item" '
            f'style="background-color:{_PARTY_HIGHLIGHT_COLORS[party_key]};">'
            f'{html.escape(_display_party_label_from_key(party_key))}'
            "</span>"
        )

    provision_data = [
        {
            "subject": _display_party(_get_provision_subject(provision)),
            "beneficiary": _display_party(_get_provision_beneficiary(provision)),
            "conditions": _display_text(_get_provision_conditions(provision)),
            "value": _display_text(_get_provision_value(provision)),
            "span": _display_text(_get_provision_span(provision)),
            "spanStart": _get_provision_span_start(provision),
            "spanEnd": _get_provision_span_end(provision),
            "groundingStatus": _get_provision_grounding_status(provision),
            "attributesHtml": _build_provision_attributes_html(provision),
        }
        for provision in sorted_provisions
    ]
    serialized_data = json.dumps(provision_data)

    return f"""
    <style>
      .px-viewer {{
        font-family: Arial, sans-serif;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 10px;
        overflow: hidden;
      }}
      .px-meta {{
        background: #fafafa;
        border-bottom: 1px solid rgba(128,128,128,0.25);
        padding: 10px 12px;
        font-size: 13px;
      }}
      .px-legend {{
        margin-bottom: 10px;
      }}
      .px-legend-item {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 999px;
        margin-right: 6px;
        margin-bottom: 4px;
        color: #111;
        font-size: 12px;
      }}
      .px-text {{
        white-space: pre-wrap;
        font-family: monospace;
        line-height: 1.65;
        padding: 12px;
        max-height: 320px;
        overflow-y: auto;
        background: white;
      }}
      .px-highlight {{
        border-radius: 3px;
        padding: 1px 2px;
        cursor: pointer;
      }}
      .px-highlight-current {{
        outline: 2px solid #d93025;
        outline-offset: 1px;
      }}
      .px-attr-row {{
        margin-bottom: 4px;
      }}
      .px-meta code {{
        font-size: 12px;
      }}
    </style>
    <div class="px-viewer">
      <div class="px-meta">
        <div class="px-legend"><strong>Beneficiary colors:</strong> {" ".join(legend_items)}</div>
        <div id="px-attributes"></div>
      </div>
      <div class="px-text" id="px-text">{highlighted_text}</div>
    </div>
    <script>
      (function() {{
        const provisions = {serialized_data};
        const attributeContainer = document.getElementById("px-attributes");
        const textWindow = document.getElementById("px-text");
        let currentIndex = 0;

        function render() {{
          if (!provisions.length) {{
            attributeContainer.innerHTML = "<div>No grounded provisions available.</div>";
            return;
          }}

          const provision = provisions[currentIndex];
          attributeContainer.innerHTML = provision.attributesHtml;

          const previous = textWindow.querySelector(".px-highlight-current");
          if (previous) {{
            previous.classList.remove("px-highlight-current");
          }}

          const current = textWindow.querySelector('[data-idx="' + currentIndex + '"]');
          if (current) {{
            current.classList.add("px-highlight-current");
            current.scrollIntoView({{ block: "center", behavior: "smooth" }});
          }}
        }}

        function jumpTo(index) {{
          if (!provisions.length) {{
            return;
          }}
          currentIndex = Math.max(0, Math.min(index, provisions.length - 1));
          render();
        }}

        textWindow.querySelectorAll(".px-highlight").forEach(function(element) {{
          element.addEventListener("click", function() {{
            jumpTo(parseInt(element.dataset.idx, 10));
          }});
        }});

        render();
      }})();
    </script>
    """


def _render_provision_metadata_list(
    provisions: list[dict[str, Any]],
) -> None:
    for provision_index, provision in enumerate(provisions, start=1):
        subject = _display_party(_get_provision_subject(provision))
        beneficiary = _display_party(_get_provision_beneficiary(provision))
        conditions = _display_text(_get_provision_conditions(provision))
        value = _display_text(_get_provision_value(provision))

        with st.expander(
            f"{provision_index:02d}. {subject} -> {beneficiary}",
            expanded=provision_index == 1,
        ):
            st.markdown(f"**Subject**: {subject}")
            st.markdown(f"**Beneficiary**: {beneficiary}")
            st.markdown(f"**Conditions**: {conditions}")
            st.markdown(f"**Value**: {value}")


def _get_actor_beneficiary_counts(provision_payload: dict[str, Any]) -> dict[str, int]:
    document_meta = provision_payload.get("document_meta_data", {})
    if isinstance(document_meta, dict):
        raw_counts = document_meta.get("actor_beneficiary_counts")
        if isinstance(raw_counts, dict):
            normalized_counts: dict[str, int] = {}
            for key, value in raw_counts.items():
                if not isinstance(value, int):
                    continue
                parts = str(key).split(" ", 1)
                actor = _normalize_party_key(parts[0] if parts else "")
                beneficiary = _normalize_party_key(parts[1] if len(parts) > 1 else "")
                combo_key = f"{actor} {beneficiary}"
                normalized_counts[combo_key] = normalized_counts.get(combo_key, 0) + int(value)
            return normalized_counts

    counts: dict[str, int] = {}
    sections = provision_payload.get("sections", [])
    if not isinstance(sections, list):
        return counts

    for section in sections:
        if not isinstance(section, dict):
            continue
        provisions = section.get("provisions", [])
        if not isinstance(provisions, list):
            continue
        for provision in provisions:
            if not isinstance(provision, dict):
                continue
            actor = _normalize_party_key(_get_provision_subject(provision))
            beneficiary = _normalize_party_key(_get_provision_beneficiary(provision))
            combo_key = f"{actor} {beneficiary}"
            counts[combo_key] = counts.get(combo_key, 0) + 1
    return counts


def _build_actor_beneficiary_table(provision_payload: dict[str, Any]) -> list[dict[str, Any]]:
    combo_counts = _get_actor_beneficiary_counts(provision_payload)
    actor_rows = [
        ("Worker", "worker"),
        ("Firm", "firm"),
        ("Union", "union"),
        ("Manager", "manager"),
        ("Unknown", "unknown"),
    ]
    return [
        {
            "Actor": label,
            "Worker": combo_counts.get(f"{actor_key} worker", 0),
            "Firm": combo_counts.get(f"{actor_key} firm", 0),
            "Union": combo_counts.get(f"{actor_key} union", 0),
            "Manager": combo_counts.get(f"{actor_key} manager", 0),
            "Unknown": combo_counts.get(f"{actor_key} unknown", 0),
        }
        for label, actor_key in actor_rows
    ]


def _get_actor_counts(provision_payload: dict[str, Any]) -> dict[str, int]:
    combo_counts = _get_actor_beneficiary_counts(provision_payload)
    counts = {"Worker": 0, "Firm": 0, "Union": 0, "Manager": 0, "unknown": 0}
    for combo_key, count in combo_counts.items():
        actor_key = combo_key.split(" ", 1)[0]
        if actor_key == "worker":
            counts["Worker"] += count
        elif actor_key == "firm":
            counts["Firm"] += count
        elif actor_key == "union":
            counts["Union"] += count
        elif actor_key == "manager":
            counts["Manager"] += count
        else:
            counts["unknown"] += count
    return counts


def _format_worker_benefit_proxy_ratio(provision_payload: dict[str, Any]) -> str:
    combo_counts = _get_actor_beneficiary_counts(provision_payload)
    actor_keys = ["worker", "firm", "union", "manager", "unknown"]
    worker_benefits = sum(combo_counts.get(f"{actor_key} worker", 0) for actor_key in actor_keys)
    firm_benefits = sum(combo_counts.get(f"{actor_key} firm", 0) for actor_key in actor_keys)
    if firm_benefits == 0:
        return "inf" if worker_benefits > 0 else "n/a"
    return f"{worker_benefits / firm_benefits:.2f}"


def render_download_button(label: str, path: Path, mime: str, key: str) -> None:
    data = read_binary_file(str(path))
    if data is None:
        return
    st.download_button(
        label,
        data=data,
        file_name=path.name,
        mime=mime,
        key=key,
    )


def render_wrapped_text_box(text: str) -> None:
    st.markdown(
        (
            "<div style='white-space: pre-wrap; word-break: break-word; "
            "padding: 0.75rem 1rem; border: 1px solid rgba(128,128,128,0.35); "
            "border-radius: 0.5rem;'>"
            f"{html.escape(text)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_extracted_provisions(group: str, model_name: str, document_stem: str) -> None:
    provision_payload = read_provisions(group, model_name, document_stem)
    if not provision_payload:
        st.warning("No provisions.json found for this model and document.")
        return

    sections = provision_payload.get("sections", [])
    if not isinstance(sections, list) or not sections:
        st.warning("Provision extraction output is missing section data.")
        return

    sections = [
        section
        for section in sections
        if isinstance(section, dict)
        and isinstance(section.get("provisions"), list)
        and len(section.get("provisions", [])) > 0
    ]
    if not sections:
        st.info("No extracted provisions were found in this document.")
        return

    st.markdown("### Document Provision Counts")
    st.table(_build_actor_beneficiary_table(provision_payload))
    proxy_col, equation_col = st.columns([1, 2.4])
    with proxy_col:
        st.metric("Worker Benefit Proxy", _format_worker_benefit_proxy_ratio(provision_payload))
    with equation_col:
        st.markdown(
            "`worker benefit proxy = worker benefits / firm benefits`"
        )

    selected_section = st.selectbox(
        "Select Section",
        sections,
        index=0,
        format_func=lambda section: (
            f"{int(section.get('section_index', 0)):03d} - "
            f"{str(section.get('header', 'Untitled')).strip() or 'Untitled'} "
            f"({len(section.get('provisions', [])) if isinstance(section.get('provisions', []), list) else 0} provision(s))"
        ),
        key=f"provision-section-{group}-{model_name}-{document_stem}",
    )

    st.markdown("### Section Review")
    section_text = str(selected_section.get("text", "") or "")
    provisions = selected_section.get("provisions", [])
    grounded_provisions = _get_grounded_section_provisions(selected_section)
    ungrounded_provisions = _get_ungrounded_section_provisions(selected_section)
    has_span_keys = any(
        isinstance(provision, dict) and "span" in provision
        for provision in provisions
        if isinstance(provisions, list)
    )

    if grounded_provisions:
        components.html(
            _build_provision_viewer_html(section_text, grounded_provisions),
            height=560,
            scrolling=False,
        )
        if ungrounded_provisions:
            st.warning(
                f"{len(ungrounded_provisions)} provision(s) in this section could not be grounded exactly and are listed below."
            )
            with st.expander("Ungrounded Provision Metadata", expanded=False):
                _render_provision_metadata_list(ungrounded_provisions)
    else:
        if has_span_keys:
            st.info(
                "No exact grounded spans are available for this section. The extracted provisions are listed below for review."
            )
        else:
            st.info(
                "This provision file predates grounded spans. Rerun `02_provision_extract` to enable exact text highlighting."
            )
        render_wrapped_text_box(section_text if section_text else "Empty section")
        if isinstance(provisions, list) and provisions:
            _render_provision_metadata_list(
                [provision for provision in provisions if isinstance(provision, dict)]
            )
        else:
            st.info("No provisions extracted for this section.")

    render_download_button(
        "Download extracted provisions",
        get_cache_dir() / "02_provision_extract" / group / model_name / document_stem / "provisions.json",
        "application/json",
        key=f"provisions-json-{group}-{model_name}-{document_stem}",
    )


def render_page_compare(
    group: str,
    pdf_path: Path,
    page_index: int,
    left_model: str | None,
    right_model: str | None,
) -> None:
    pdf_col, left_col, right_col = st.columns([1.15, 1, 1])

    with pdf_col:
        st.subheader("PDF Page")
        st.image(render_page(str(pdf_path), page_index), use_container_width=True)

    for column, model_name, label in [
        (left_col, left_model, "OCR"),
        (right_col, right_model, "OCR"),
    ]:
        with column:
            st.subheader(model_name if model_name is not None else label)

            if model_name is None:
                st.info("No OCR model is available for this group.")
                continue

            text = read_ocr_page(group, model_name, pdf_path.stem, page_index)
            if text is None:
                st.warning("No OCR page found for this model and page.")
            else:
                st.markdown(text)


st.set_page_config(page_title="OCR Comparison and Provision Extraction Review", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.header("OCR Comparison and Provision Extraction Review")

page_tab, provisions_tab, notes_tab = st.tabs(
    ["OCR Model Compare", "Extracted Provisions", "Additional Notes"]
)

with page_tab:
    st.markdown(
        "\n".join(
            [
                "- Each PDF page is rendered as an image and transcribed independently.",
                "- The model is prompted to preserve formatting in markdown and mark major headers with `##`.",
                "- The text is sectioned deterministically using the markdown headers.",
                "- This view compares page-level OCR output from two selected models against the original PDF image.",
                "- Only the 'dol_archive' group has multiple OCR models available for comparison at this time.",
            ]
        )
    )
    with st.expander("OCR Prompts", expanded=False):
        st.markdown("**System prompt**")
        render_wrapped_text_box(OCR_SYSTEM_PROMPT)
        st.markdown("**User prompt**")
        render_wrapped_text_box(OCR_USER_PROMPT)

    control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns([1, 1.5, 0.8, 1, 1])

    with control_col1:
        compare_group = st.selectbox("CBA Collection", DOL_GROUPS, key="compare-group")

    compare_documents = set(list_cached_documents(compare_group))
    compare_pdfs = [pdf for pdf in list_pdfs(compare_group) if pdf.stem in compare_documents]

    if not compare_pdfs:
        st.warning(f"No OCR review documents found for {compare_group}.")
    else:
        compare_pdf_names = [pdf.name for pdf in compare_pdfs]

        with control_col2:
            selected_compare_pdf_name = st.selectbox("Document", compare_pdf_names, key="compare-document")

        compare_pdf_path = next(pdf for pdf in compare_pdfs if pdf.name == selected_compare_pdf_name)
        compare_page_count = get_page_count(str(compare_pdf_path))

        with control_col3:
            compare_page_number = st.number_input(
                "Page",
                min_value=1,
                max_value=compare_page_count,
                value=1,
                step=1,
                key="compare-page",
            )

        compare_models = list_models(compare_group)
        if compare_models:
            left_default = 0
            right_default = 1 if len(compare_models) > 1 else 0
            with control_col4:
                compare_left_model = st.selectbox(
                    "Left OCR model",
                    compare_models,
                    index=left_default,
                    key="compare-left-model",
                )
            with control_col5:
                compare_right_model = st.selectbox(
                    "Right OCR model",
                    compare_models,
                    index=right_default,
                    key="compare-right-model",
                )
        else:
            compare_left_model = None
            compare_right_model = None

        st.caption(
            f"{compare_group} / {compare_pdf_path.name} / "
            f"page {compare_page_number} of {compare_page_count}"
        )
        render_page_compare(
            compare_group,
            compare_pdf_path,
            compare_page_number - 1,
            compare_left_model,
            compare_right_model,
        )

with provisions_tab:
    st.markdown(
        "\n".join(
            [
                "- Provision extraction is run section by section on OCR-derived markdown.",
                "- The model assigns each extracted provision a subject, beneficiary, conditions, value, and a verbatim grounding span.",
                "- Subject and beneficiary categories include Worker, Firm, Union, and Manager.",
                "- Exact spans are resolved to section-relative character offsets for highlighting and auditability.",
                "- Document-level counts and the worker-benefit proxy are aggregated from actor-by-beneficiary crosstabs.",
            ]
        )
    )
    with st.expander("Provision Extraction Prompts", expanded=False):
        st.markdown("**System prompt**")
        render_wrapped_text_box(PROVISION_SYSTEM_PROMPT_TEMPLATE)
        st.markdown("**User prompt**")
        render_wrapped_text_box(PROVISION_USER_PROMPT)

    control_col1, control_col2, control_col3 = st.columns([1, 1.5, 1])

    with control_col1:
        provision_group = st.selectbox("DOL group", DOL_GROUPS, key="provision-group")

    provision_documents = list_provision_cached_documents(provision_group)
    if not provision_documents:
        st.info(f"No provision-extraction output is available for {provision_group}.")
    else:
        with control_col2:
            provision_document_stem = st.selectbox(
                "Document",
                provision_documents,
                key="provision-document",
            )

        provision_models = [
            model_name
            for model_name in list_provision_models(provision_group)
            if read_provisions(provision_group, model_name, provision_document_stem) is not None
        ]

        if not provision_models:
            st.info("No provision-extraction model output is available for this document.")
        else:
            with control_col3:
                provision_model = st.selectbox(
                    "Provision model",
                    provision_models,
                    key="provision-model",
                )

            render_extracted_provisions(provision_group, provision_model, provision_document_stem)

with notes_tab:
    st.markdown(
        "\n".join([
            "### OCR Extraction Notes",
            "- transcription is easy for most models, the text is accurate but the formatting into sections is inconsistent. Models seem to struggle with title pages that have larger text that may look like 'headers'",
            "- qwen-3.5-27b-fp8 is among the best performing models, is free, and faster than it's non-quantized counterpart, so it is the default OCR model",
            "- OCR models generally fall into two categories, those that produce more structured output with custom formats (e.g. mistral, olmo, paddle) and general visual-language models (e.g. qwen, gemini) that produce high quality markdown. The former can be difficult to work with because they add a lot of structure you may or may not need. General VLMs are also typically more accurate in their transcriptions",
            "- below are the estimated costs and runtime of performing OCR on the entire corpus (including all three collections of CBAs)",
            "",
            "| Model | Cost | Runtime |",
            "| --- | ---: | ---: |",
            "| gemini-3.1-flash-lite | $543 | 44 hours |",
            "| gemini-3.1-pro | $5324 | 118 hours |",
            "| claude-sonnet-4.6 | $7388 | 281 hours |",
            "| qwen-3.5-9B | $0 | 147 hours |",
            "| qwen-3.5-35B-A3B | $0 | 251 hours |",
            "| qwen-3.5-27B | $0 | 283 hours |",
            "| qwen-3.5-27B-FP8 | $0 | 87 hours |",
            "### Provision Extraction Notes",
            "- This extraction now focuses on who enacts a provision, who benefits from it, its conditions, its substantive value, and a grounded evidence span",
            "\nHow LLMs Can Improve on Ash's Baseline?\n",
            "- Ash's approach miss implied actors (e.g. 'Compensation shall be paid weekly' implies a 'firm' obligation')",
            "- LLMs can incorporate context from the entire section to identify conditions on a provision",
            "- Ash's segmentation approach was highly customized for Canadian CBAs, LLMs are more flexible",
            "\nPotential Updates to Provision Extraction Approach\n",
            "- We don't need to use Ash's exact taxonomy. We can tailor to our specific use case of generosity or focus on 'worker' vs 'firm' power",
            "- I used qwen-3.5-27b for provision extraction because it's free for experimentation but these judgements would likely be much better from larger, more intelligent models",
            "- Provisions still need to be categorized into 'concepts' or 'clause types' like healthcare, wages, etc."
        ])
    )
