import html
import json
import os
from pathlib import Path
from typing import Any

import fitz
import streamlit as st
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
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
        "You are a legal assistant tasked with extracting and categorizing",
        "provision types and which actors they refer to.",
        "The actors you should identify and extract are configured in the runner.",
        "The provision types you should identify and extract are configured in the runner.",
        "Return your response in a JSON format with the following schema:",
        "{provisions: [{'actor': the party involved in the provision, 'provision_type': one of the provision types listed above, 'text': the text of the provision from the contract}]}",
        "If there are no provisions in the text, return {provisions: []}. Only extract provisions that are explicitly stated",
    ]
)

PROVISION_USER_PROMPT = "Extract and categorize the provisions in the following text:"

load_dotenv(ENV_PATH)


@st.cache_data
def get_cache_dir() -> Path:
    cache_dir = os.environ.get("CACHE_DIR")
    if not cache_dir:
        raise RuntimeError("CACHE_DIR is not set in .env")
    return Path(cache_dir).resolve()


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


def _display_actor(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return text


def _display_provision_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return text


def _normalize_actor_key(value: Any) -> str:
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


def _normalize_provision_type_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"right", "permission", "obligation", "prohibition"}:
        return text
    return "unknown"


def _get_actor_provision_type_counts(provision_payload: dict[str, Any]) -> dict[str, int]:
    document_meta = provision_payload.get("document_meta_data", {})
    if isinstance(document_meta, dict):
        raw_counts = document_meta.get("actor_provision_type_counts")
        if isinstance(raw_counts, dict):
            normalized_counts: dict[str, int] = {}
            for key, value in raw_counts.items():
                if not isinstance(value, int):
                    continue
                parts = str(key).split(" ", 1)
                actor = _normalize_actor_key(parts[0] if parts else "")
                provision_type = _normalize_provision_type_key(parts[1] if len(parts) > 1 else "")
                combo_key = f"{actor} {provision_type}"
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
            actor = _normalize_actor_key(provision.get("actor"))
            provision_type = _normalize_provision_type_key(provision.get("provision_type"))
            combo_key = f"{actor} {provision_type}"
            counts[combo_key] = counts.get(combo_key, 0) + 1
    return counts


def _build_actor_provision_type_table(provision_payload: dict[str, Any]) -> list[dict[str, Any]]:
    combo_counts = _get_actor_provision_type_counts(provision_payload)
    actor_rows = [
        ("Worker", "worker"),
        ("Firm", "firm"),
        ("Union", "union"),
        ("Manager", "manager"),
    ]
    return [
        {
            "Actor": label,
            "Rights": combo_counts.get(f"{actor_key} right", 0),
            "Permissions": combo_counts.get(f"{actor_key} permission", 0),
            "Obligations": combo_counts.get(f"{actor_key} obligation", 0),
            "Prohibitions": combo_counts.get(f"{actor_key} prohibition", 0),
        }
        for label, actor_key in actor_rows
    ]


def _get_actor_counts(provision_payload: dict[str, Any]) -> dict[str, int]:
    combo_counts = _get_actor_provision_type_counts(provision_payload)
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
    combo_counts = _get_actor_provision_type_counts(provision_payload)
    numerator = (
        combo_counts.get("worker right", 0)
        + combo_counts.get("worker permission", 0)
        + combo_counts.get("firm obligation", 0)
        + combo_counts.get("firm prohibition", 0)
        + combo_counts.get("union right", 0)
        + combo_counts.get("union permission", 0)
        + combo_counts.get("manager obligation", 0)
        + combo_counts.get("manager prohibition", 0)
    )
    denominator = (
        combo_counts.get("worker obligation", 0)
        + combo_counts.get("worker prohibition", 0)
        + combo_counts.get("firm right", 0)
        + combo_counts.get("firm permission", 0)
        + combo_counts.get("union obligation", 0)
        + combo_counts.get("union prohibition", 0)
        + combo_counts.get("manager right", 0)
        + combo_counts.get("manager permission", 0)
    )
    if denominator == 0:
        return "inf" if numerator > 0 else "n/a"
    return f"{numerator / denominator:.2f}"


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
    st.table(_build_actor_provision_type_table(provision_payload))
    proxy_col, equation_col = st.columns([1, 2.4])
    with proxy_col:
        st.metric("Worker Benefit Proxy", _format_worker_benefit_proxy_ratio(provision_payload))
    with equation_col:
        st.markdown(
            "`worker benefit proxy = (worker right + worker permission + firm obligation + firm prohibition + "
            "union right + union permission + manager obligation + manager prohibition) / "
            "(worker obligation + worker prohibition + firm right + firm permission + "
            "union obligation + union prohibition + manager right + manager permission)`"
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

    source_col, extracted_col = st.columns([1.1, 1])

    with source_col:
        source_text = str(selected_section.get("text", "") or "")
        st.markdown(source_text if source_text else "_Empty section_")

    with extracted_col:
        st.markdown("### Extracted Provisions")
        provisions = selected_section.get("provisions", [])
        if not isinstance(provisions, list) or not provisions:
            st.info("No provisions extracted for this section.")
        else:
            for provision_index, provision in enumerate(provisions, start=1):
                if not isinstance(provision, dict):
                    continue

                actor = _display_actor(provision.get("actor"))
                provision_type = _display_provision_type(provision.get("provision_type"))
                text = str(provision.get("text", "") or "")

                with st.expander(
                    f"{provision_index:02d}. {actor} / {provision_type}",
                    expanded=provision_index == 1,
                ):
                    st.markdown(text if text else "_Empty provision text_")

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
                "- The model assigns each extracted provision an actor, a type, and extracts the relevant text.",
                "- Actor categories include Worker, Firm, Union, and Manager. Provision types include Right, Permission, Obligation, and Prohibition.",
                "- Document-level counts and the worker-benefit proxy are aggregated from those structured section outputs.",
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
            "- This follows Ash's provision taxonomy of actors and provision types",
            "\nHow LLMs Can Improve on Ash's Baseline?\n",
            "- Ash's approach miss implied actors (e.g. 'Compensation shall be paid weekly' implies a 'firm' obligation')",
            "- LLMs can incorporate context from the entire section to identify conditions on a provision (not currently implemented)",
            "- Ash's segmentation approach was highly customized for Canadian CBAs, LLMs are more flexible",
            "\nPotential Updates to Provision Extraction Approach\n",
            "- We don't need to use Ash's exact taxonomy. We can tailor to our specific use case of generosity or focus on 'worker' vs 'firm' power"
            "- I used qwen-3.5-27b-fp8 for provision extraction because it's free for experimentation but these judgements would likely be much better from larger, more intelligent models",
            "- Provisions still need to be categorized into 'concepts' or 'clause types' like healthcare, wages, etc."
        ])
    )
