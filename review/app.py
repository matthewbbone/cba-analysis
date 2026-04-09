import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import fitz
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
DOL_GROUPS = ["dol_archive", "cornell_dol", "cornell_retail_educ"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

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
def list_paddle_model_cache_files(group: str) -> list[Path]:
    output_dir = get_cache_dir() / "01_paddleocr_output" / group
    return sorted(output_dir.glob("*/cache.json"))


@st.cache_data
def list_cached_documents(group: str) -> set[str]:
    document_ids = set()
    for cache_file in list_model_cache_files(group):
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        document_ids.update(cache.get("documents", {}).keys())
    return document_ids


@st.cache_data
def list_paddle_cached_documents(group: str) -> set[str]:
    document_ids = set()
    for cache_file in list_paddle_model_cache_files(group):
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for document_id, metadata in cache.get("documents", {}).items():
            if not isinstance(metadata, dict):
                continue
            if metadata.get("completed"):
                document_ids.add(document_id)
    return document_ids


@st.cache_data
def list_models(group: str) -> list[str]:
    return [cache_file.parent.name for cache_file in list_model_cache_files(group)]


@st.cache_data
def list_paddle_models(group: str) -> list[str]:
    return [cache_file.parent.name for cache_file in list_paddle_model_cache_files(group)]


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
def read_text_file(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_text(encoding="utf-8")


@st.cache_data
def read_binary_file(path: str) -> bytes | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


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


@st.cache_data
def read_sections(group: str, model_name: str, document_stem: str) -> dict[str, str] | None:
    sections_file = (
        get_cache_dir()
        / "01_ocr_output"
        / group
        / model_name
        / document_stem
        / "sections.json"
    )
    if not sections_file.exists():
        return None
    try:
        return json.loads(sections_file.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data
def list_segmented_documents(group: str) -> set[str]:
    output_dir = get_cache_dir() / "02_segmentation_output" / group
    if not output_dir.exists():
        return set()
    return {path.name for path in output_dir.glob("document_*") if path.is_dir()}


def get_segmented_document_dir(group: str, document_stem: str) -> Path:
    return get_cache_dir() / "02_segmentation_output" / group / document_stem


@st.cache_data
def read_segmented_document_meta(group: str, document_stem: str) -> dict[str, Any] | None:
    return read_json_file(str(get_segmented_document_dir(group, document_stem) / "document_meta.json"))


@st.cache_data
def read_corrected_segment(group: str, document_stem: str, segment_number: int) -> str | None:
    segment_path = get_segmented_document_dir(group, document_stem) / "segments" / f"segment_{segment_number}.txt"
    return read_text_file(str(segment_path))


def get_paddle_document_dir(group: str, model_name: str, document_stem: str) -> Path:
    return (
        get_cache_dir()
        / "01_paddleocr_output"
        / group
        / model_name
        / document_stem
    )


@st.cache_data
def read_paddle_document_manifest(group: str, model_name: str, document_stem: str) -> dict[str, Any] | None:
    return read_json_file(str(get_paddle_document_dir(group, model_name, document_stem) / "document.json"))


def get_paddle_page_entry(
    group: str,
    model_name: str,
    document_stem: str,
    page_index: int,
) -> dict[str, Any] | None:
    manifest = read_paddle_document_manifest(group, model_name, document_stem)
    if not manifest:
        return None
    for page in manifest.get("pages", []):
        if isinstance(page, dict) and page.get("page_index") == page_index:
            return page
    return None


@st.cache_data
def has_paddle_document(group: str, model_name: str, document_stem: str) -> bool:
    return read_paddle_document_manifest(group, model_name, document_stem) is not None


@st.cache_data
def inline_markdown_assets(base_dir: str, markdown_path: str) -> str | None:
    base_path = Path(base_dir)
    markdown_file = Path(markdown_path)
    if not markdown_file.exists():
        return None

    text = markdown_file.read_text(encoding="utf-8")
    replacements: dict[str, str] = {}
    for asset_path in sorted(p for p in base_path.rglob("*") if p.is_file()):
        if asset_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel_path = asset_path.relative_to(base_path).as_posix()
        mime_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        replacements[rel_path] = f"data:{mime_type};base64,{encoded}"

    for rel_path in sorted(replacements, key=len, reverse=True):
        text = text.replace(rel_path, replacements[rel_path])

    return text


def _normalize_block_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def merge_markdown_with_headers_and_footers(markdown_text: str | None, page_json: dict[str, Any] | None) -> str | None:
    if markdown_text is None:
        return None
    if not page_json:
        return markdown_text

    parsing_res = page_json.get("parsing_res_list", [])
    if not isinstance(parsing_res, list):
        return markdown_text

    headers = [
        _normalize_block_text(block.get("block_content"))
        for block in parsing_res
        if isinstance(block, dict) and block.get("block_label") == "header"
    ]
    footers = [
        _normalize_block_text(block.get("block_content"))
        for block in parsing_res
        if isinstance(block, dict) and block.get("block_label") == "footer"
    ]
    headers = [text for text in headers if text]
    footers = [text for text in footers if text]

    parts: list[str] = []
    if headers:
        parts.append("\n\n".join(f"**Header:** {text}" for text in headers))
    parts.append(markdown_text.strip())
    if footers:
        parts.append("\n\n".join(f"**Footer:** {text}" for text in footers))
    return "\n\n".join(part for part in parts if part)


def render_sections(group: str, model_name: str | None, document_stem: str, label: str) -> None:
    st.subheader(label if model_name is None else f"{label}: {model_name}")

    if model_name is None:
        st.info("OCR output is unavailable for this group.")
        return

    sections = read_sections(group, model_name, document_stem)
    if not sections:
        st.warning("No sections.json found for this model and document.")
        return

    for title, content in sections.items():
        st.markdown(f"### {title}")
        st.markdown(content)


def render_corrected_segments(group: str, document_stem: str) -> None:
    document_meta = read_segmented_document_meta(group, document_stem)
    if not document_meta:
        st.warning("No corrected segmentation output found for this document.")
        return

    document_info = document_meta.get("document", {})
    sections = document_meta.get("sections", [])
    if not isinstance(document_info, dict) or not isinstance(sections, list) or not sections:
        st.warning("Corrected segmentation metadata is missing or invalid.")
        return

    st.subheader("Corrected Segments")
    st.caption(
        f"{document_info.get('section_count', len(sections))} segments from "
        f"`02_segmentation_output/{group}/{document_stem}`"
    )

    selected_section = st.selectbox(
        "Segment",
        sections,
        index=0,
        format_func=lambda section: (
            f"{int(section.get('number', 0)):03d} - "
            f"{str(section.get('header', 'Untitled')).strip() or 'Untitled'}"
        ),
        key=f"corrected-segment-{group}-{document_stem}",
    )

    segment_number = int(selected_section.get("number", 0))
    segment_text = read_corrected_segment(group, document_stem, segment_number)

    meta_col, text_col = st.columns([0.9, 1.7])

    with meta_col:
        st.markdown("### Segment Metadata")
        st.json(
            {
                "number": selected_section.get("number"),
                "header": selected_section.get("header"),
                "level": selected_section.get("level"),
                "parent_headers": selected_section.get("parent_headers", []),
                "span": selected_section.get("span"),
            }
        )

        with st.expander("Document Hierarchy", expanded=False):
            st.json(document_info.get("hierarchy", []))

        with st.expander("Document Metadata", expanded=False):
            st.json(document_info)
            render_download_button(
                "Download document metadata",
                get_segmented_document_dir(group, document_stem) / "document_meta.json",
                "application/json",
                key=f"segment-meta-{group}-{document_stem}",
            )

    with text_col:
        st.markdown("### Segment Text")
        if segment_text is None:
            st.warning("No corrected segment text file found for this segment.")
        else:
            st.markdown(segment_text)
            render_download_button(
                "Download corrected segment",
                get_segmented_document_dir(group, document_stem) / "segments" / f"segment_{segment_number}.txt",
                "text/plain",
                key=f"segment-text-{group}-{document_stem}-{segment_number}",
            )


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


st.set_page_config(page_title="OCR Review", layout="wide")
st.title("OCR Review")

with st.sidebar:
    st.header("Compare")

    group = st.selectbox("DOL group", DOL_GROUPS)
    cached_documents = list_cached_documents(group)
    paddle_cached_documents = list_paddle_cached_documents(group)
    segmented_documents = list_segmented_documents(group)
    available_documents = cached_documents | paddle_cached_documents | segmented_documents
    pdfs = [pdf for pdf in list_pdfs(group) if pdf.stem in available_documents]
    models = list_models(group)

    if not pdfs:
        st.warning(f"No cached OCR documents found for {group}.")
        st.stop()

    pdf_names = [pdf.name for pdf in pdfs]
    selected_pdf_name = st.selectbox("Document", pdf_names)
    pdf_path = next(pdf for pdf in pdfs if pdf.name == selected_pdf_name)

    page_count = get_page_count(str(pdf_path))
    page_number = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
    page_index = page_number - 1

    if models:
        left_default = 0
        right_default = 1 if len(models) > 1 else 0
        left_model = st.selectbox("Left model", models, index=left_default)
        right_model = st.selectbox("Right model", models, index=right_default)
    else:
        st.info("No model-specific OCR cache files found for this group.")
        left_model = None
        right_model = None

    paddle_models = [
        model_name
        for model_name in list_paddle_models(group)
        if has_paddle_document(group, model_name, pdf_path.stem)
    ]
    if paddle_models:
        paddle_model = st.selectbox("Paddle model", paddle_models, index=0)
    else:
        paddle_model = None

st.caption(f"{group} / {pdf_path.name} / page {page_number} of {page_count}")

page_tab, sections_tab, paddle_tab = st.tabs(["Page Compare", "Sections", "Paddle Layout"])

with page_tab:
    pdf_col, left_col, right_col = st.columns([1.15, 1, 1])

    with pdf_col:
        st.subheader("PDF page")
        st.image(render_page(str(pdf_path), page_index), use_container_width=True)

    for column, model_name, label in [
        (left_col, left_model, "Left OCR"),
        (right_col, right_model, "Right OCR"),
    ]:
        with column:
            st.subheader(label if model_name is None else f"{label}: {model_name}")

            if model_name is None:
                st.info("OCR output is unavailable for this group.")
                continue

            text = read_ocr_page(group, model_name, pdf_path.stem, page_index)
            if text is None:
                st.warning("No OCR page found for this model and page.")
            else:
                st.markdown(text)

with sections_tab:
    corrected_tab, legacy_tab = st.tabs(["Corrected Segments", "Legacy OCR Sections"])

    with corrected_tab:
        render_corrected_segments(group, pdf_path.stem)

    with legacy_tab:
        left_col, right_col = st.columns(2)

        with left_col:
            render_sections(group, left_model, pdf_path.stem, "Left Sections")

        with right_col:
            render_sections(group, right_model, pdf_path.stem, "Right Sections")

with paddle_tab:
    if paddle_model is None:
        st.info("No Paddle-native OCR output is available for this document.")
    else:
        document_dir = get_paddle_document_dir(group, paddle_model, pdf_path.stem)
        document_manifest = read_paddle_document_manifest(group, paddle_model, pdf_path.stem)
        page_entry = get_paddle_page_entry(group, paddle_model, pdf_path.stem, page_index)

        if not document_manifest or not page_entry:
            st.warning("No Paddle page manifest found for this document and page.")
        else:
            pdf_col, paddle_col = st.columns([1, 1.2])

            with pdf_col:
                st.subheader("PDF page")
                st.image(render_page(str(pdf_path), page_index), use_container_width=True)

            with paddle_col:
                st.subheader(f"Paddle page: {paddle_model}")
                page_dir = document_dir / page_entry["page_dir"]
                page_json_path = document_dir / page_entry["json_path"]
                page_json = read_json_file(str(page_json_path))
                page_markdown = inline_markdown_assets(
                    str(page_dir),
                    str(document_dir / page_entry["markdown_path"]),
                )
                page_markdown = merge_markdown_with_headers_and_footers(page_markdown, page_json)
                if page_markdown is None:
                    st.warning("No Paddle markdown found for this page.")
                else:
                    st.markdown(page_markdown, unsafe_allow_html=True)

            with st.expander("Document Markdown", expanded=False):
                document_markdown = inline_markdown_assets(
                    str(document_dir),
                    str(document_dir / document_manifest["document_markdown_path"]),
                )
                if document_markdown is None:
                    st.warning("No document markdown found.")
                else:
                    st.markdown(document_markdown, unsafe_allow_html=True)
                    render_download_button(
                        "Download document markdown",
                        document_dir / document_manifest["document_markdown_path"],
                        "text/markdown",
                        key=f"doc-md-{paddle_model}-{pdf_path.stem}",
                    )

            with st.expander("Raw Page JSON", expanded=False):
                if page_json is None:
                    st.warning("No page JSON found.")
                else:
                    st.json(page_json)
                    render_download_button(
                        "Download page JSON",
                        page_json_path,
                        "application/json",
                        key=f"page-json-{paddle_model}-{pdf_path.stem}-{page_index}",
                    )

            with st.expander("Rendered Page Images", expanded=False):
                render_paths = page_entry.get("render_paths", [])
                if not render_paths:
                    st.info("No rendered page images were saved for this page.")
                for render_no, render_path in enumerate(render_paths, start=1):
                    render_file = document_dir / render_path
                    image_bytes = read_binary_file(str(render_file))
                    if image_bytes is None:
                        continue
                    st.markdown(f"**Render {render_no}: {render_file.name}**")
                    st.image(image_bytes, use_container_width=True)

            with st.expander("Tables", expanded=False):
                table_html_paths = page_entry.get("table_html_paths", [])
                table_xlsx_paths = page_entry.get("table_xlsx_paths", [])
                if not table_html_paths and not table_xlsx_paths:
                    st.info("No saved table sidecars were found for this page.")

                for table_no, table_path in enumerate(table_html_paths, start=1):
                    html_file = document_dir / table_path
                    html = read_text_file(str(html_file))
                    if html is None:
                        continue
                    st.markdown(f"**Table HTML {table_no}: {html_file.name}**")
                    components.html(html, height=420, scrolling=True)
                    render_download_button(
                        "Download HTML",
                        html_file,
                        "text/html",
                        key=f"table-html-{paddle_model}-{pdf_path.stem}-{page_index}-{table_no}",
                    )

                for table_no, table_path in enumerate(table_xlsx_paths, start=1):
                    xlsx_file = document_dir / table_path
                    if not xlsx_file.exists():
                        continue
                    render_download_button(
                        f"Download XLSX {table_no}",
                        xlsx_file,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"table-xlsx-{paddle_model}-{pdf_path.stem}-{page_index}-{table_no}",
                    )
