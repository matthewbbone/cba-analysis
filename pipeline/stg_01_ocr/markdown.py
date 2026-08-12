"""Markdown normalisation shared by every stage-01 OCR runner.

The raw model response is deliberately kept elsewhere.  This module produces the
comparison/assembly representation: hidden ``<think>`` blocks are removed and HTML
tables are converted to a single, predictable pipe-table representation.

Only complete HTML ``<table>`` elements are rewritten.  Text which contains no
table (the existing Ovis corpus, in particular) passes through byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
import re
from typing import Iterable, Mapping


THINK_BLOCK_PATTERN = re.compile(
    r"<think\b[^>]*>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)
_TABLE_TAG_PATTERN = re.compile(r"</?table\b[^>]*>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_BR_MARKER = "\x00OCR_BR\x00"
_MAX_SPAN = 1_000


@dataclass(frozen=True)
class MarkdownResult:
    """A normalised page and whether merged table cells were expanded."""

    text: str
    expanded_spans: bool = False

    @property
    def expanded_merged_cells(self) -> bool:
        """Descriptive alias used by document/page state reporting."""

        return self.expanded_spans


@dataclass(frozen=True)
class MarkdownBlock:
    """One recognised layout block, already in page reading order.

    ``metadata`` is intentionally opaque: layout implementations can retain labels,
    boxes, or confidence values without coupling the assembler to their schema.
    """

    text: str
    label: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _Cell:
    text: str
    rowspan: int = 1
    colspan: int = 1


class _TableParser(HTMLParser):
    """Collect the visible cells of one complete, non-nested HTML table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_Cell]] = []
        self._row: list[_Cell] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_rowspan = 1
        self._cell_colspan = 1
        self._table_depth = 0
        self.nested_table = False
        self.had_span_attribute = False

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        raw = next((value for key, value in attrs if key.lower() == name), None)
        try:
            value = int(raw) if raw is not None else 1
        except (TypeError, ValueError):
            return 1
        if value < 1:
            return 1
        return min(value, _MAX_SPAN)

    def _finish_cell(self) -> None:
        if self._cell_parts is None:
            return
        if self._row is None:
            self._row = []
        self._row.append(
            _Cell(
                text=_clean_cell_text("".join(self._cell_parts)),
                rowspan=self._cell_rowspan,
                colspan=self._cell_colspan,
            )
        )
        self._cell_parts = None
        self._cell_rowspan = 1
        self._cell_colspan = 1

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._row is not None:
            self.rows.append(self._row)
        self._row = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            if self._table_depth > 1:
                self.nested_table = True
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            if self._row is not None:
                self._finish_row()
            self._row = []
        elif tag in {"td", "th"}:
            self._finish_cell()
            if self._row is None:
                self._row = []
            self._cell_rowspan = self._span(attrs, "rowspan")
            self._cell_colspan = self._span(attrs, "colspan")
            self.had_span_attribute |= (
                self._cell_rowspan > 1 or self._cell_colspan > 1
            )
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(_BR_MARKER)
        elif tag in {"p", "div", "li"} and self._cell_parts:
            # Preserve a visible boundary without carrying arbitrary HTML into the
            # canonical table representation.
            self._cell_parts.append(_BR_MARKER)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in {"br", "hr", "img", "input", "meta", "link"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "table":
            if self._table_depth == 1:
                self._finish_row()
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth != 1:
            return
        if tag in {"td", "th"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._table_depth == 1 and self._cell_parts is not None:
            self._cell_parts.append(data)


def remove_think_blocks(text: str) -> str:
    """Remove complete model reasoning blocks using the legacy stage-01 rule."""

    return THINK_BLOCK_PATTERN.sub("", text)


def _clean_cell_text(text: str) -> str:
    parts = text.replace("\xa0", " ").split(_BR_MARKER)
    cleaned = [_WHITESPACE_PATTERN.sub(" ", part).strip() for part in parts]
    value = "<br>".join(part for part in cleaned if part)
    return value.replace("|", r"\|")


def _expand_rows(rows: list[list[_Cell]]) -> tuple[list[list[str]], bool]:
    """Expand HTML table spans by repeating values into a rectangular grid."""

    grid: list[dict[int, str]] = []
    # column -> (value, number of later source rows still occupied)
    pending: dict[int, tuple[str, int]] = {}
    expanded = False

    for source_row in rows:
        row: dict[int, str] = {}
        next_pending: dict[int, tuple[str, int]] = {}
        for column, (value, remaining) in pending.items():
            row[column] = value
            expanded = True
            if remaining > 1:
                next_pending[column] = (value, remaining - 1)
        pending = next_pending

        cursor = 0
        for cell in source_row:
            positions: list[int] = []
            while len(positions) < cell.colspan:
                while cursor in row:
                    cursor += 1
                positions.append(cursor)
                cursor += 1

            for column in positions:
                row[column] = cell.text
                if cell.rowspan > 1:
                    pending[column] = (cell.text, cell.rowspan - 1)
            expanded |= cell.colspan > 1

        grid.append(row)

    width = max((max(row, default=-1) + 1 for row in grid), default=0)
    rectangular = [
        [row.get(column, "") for column in range(width)]
        for row in grid
    ]
    return rectangular, expanded


def _table_to_markdown(fragment: str) -> tuple[str, bool] | None:
    parser = _TableParser()
    try:
        parser.feed(fragment)
        parser.close()
    except (ValueError, AssertionError):
        # Model-generated HTML can be malformed.  Keeping the raw fragment is safer
        # than deleting content or guessing at a partially parsed table.
        return None

    rows = [row for row in parser.rows if row]
    if parser.nested_table or not rows:
        return None
    grid, expanded = _expand_rows(rows)
    if not grid or not grid[0]:
        return None

    width = len(grid[0])
    lines = ["| " + " | ".join(grid[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in grid[1:])
    return "\n".join(lines), expanded or parser.had_span_attribute


def _complete_table_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for match in _TABLE_TAG_PATTERN.finditer(text):
        token = match.group(0)
        is_closing = bool(re.match(r"<\s*/", token))
        if is_closing:
            if depth == 0:
                continue
            depth -= 1
            if depth == 0:
                ranges.append((start, match.end()))
        else:
            if depth == 0:
                start = match.start()
            depth += 1
    return ranges


def normalize_html_tables(text: str, *, raw_output: bool = False) -> MarkdownResult:
    """Convert complete HTML tables while preserving all other bytes.

    ``raw_output`` bypasses this conversion entirely.  Think-block removal is a
    separate page-level concern handled by :func:`normalize_page_markdown`.
    """

    if raw_output or "<table" not in text.lower():
        return MarkdownResult(text=text)

    ranges = _complete_table_ranges(text)
    if not ranges:
        return MarkdownResult(text=text)

    output: list[str] = []
    offset = 0
    expanded_spans = False
    for start, end in ranges:
        output.append(text[offset:start])
        fragment = text[start:end]
        converted = _table_to_markdown(fragment)
        if converted is None:
            output.append(fragment)
        else:
            markdown, expanded = converted
            output.append(markdown)
            expanded_spans |= expanded
        offset = end
    output.append(text[offset:])
    return MarkdownResult(text="".join(output), expanded_spans=expanded_spans)


def normalize_page_markdown(text: str, *, raw_output: bool = False) -> MarkdownResult:
    """Create the durable page markdown from a verbatim model response.

    Reasoning blocks are always excluded from comparison artifacts.  With
    ``raw_output=True`` the remaining response is left in its model-native syntax.
    """

    stripped = remove_think_blocks(text)
    return normalize_html_tables(stripped, raw_output=raw_output)


def assemble_markdown_blocks(
    blocks: Iterable[str | MarkdownBlock],
    *,
    raw_output: bool = False,
    separator: str = "\n\n",
) -> MarkdownResult:
    """Assemble recognised layout blocks in input (reading) order.

    Empty blocks are omitted and the page is normalised once after assembly, which
    keeps single-shot and region-based runners on exactly the same code path.
    """

    assembled = assemble_raw_blocks(blocks, separator=separator)
    return normalize_page_markdown(assembled, raw_output=raw_output)


def assemble_raw_blocks(
    blocks: Iterable[str | MarkdownBlock],
    *,
    separator: str = "\n\n",
) -> str:
    """Join raw region responses without think stripping or table conversion."""

    values: list[str] = []
    for block in blocks:
        value = block if isinstance(block, str) else block.text
        value = value.strip()
        if value:
            values.append(value)
    return separator.join(values)


# Concise aliases for callers that do not need the result metadata.
def normalize_markdown(text: str, *, raw_output: bool = False) -> str:
    return normalize_page_markdown(text, raw_output=raw_output).text


normalise_html_tables = normalize_html_tables
normalise_page_markdown = normalize_page_markdown
normalise_markdown = normalize_markdown


__all__ = [
    "MarkdownBlock",
    "MarkdownResult",
    "THINK_BLOCK_PATTERN",
    "assemble_markdown_blocks",
    "assemble_raw_blocks",
    "normalise_html_tables",
    "normalise_markdown",
    "normalise_page_markdown",
    "normalize_html_tables",
    "normalize_markdown",
    "normalize_page_markdown",
    "remove_think_blocks",
]
