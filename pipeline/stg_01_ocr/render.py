"""Render one PDF page to PNG in an isolated Python process.

This module deliberately keeps its module-level imports light.  PyMuPDF is
imported only by the subprocess rendering path and Pillow only when a caller
asks for an image object.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Sequence

from pipeline.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from PIL import Image


PNG_DATA_URL_PREFIX = "data:image/png;base64,"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PageRenderError(RuntimeError):
    """Raised when the isolated renderer cannot return a PNG page."""


@dataclass(frozen=True)
class RenderedPage:
    """A rendered page represented by its base64-encoded PNG bytes."""

    png_base64: str

    @property
    def data_url(self) -> str:
        """Return an OpenAI-compatible data URL without re-encoding the PNG."""

        return f"{PNG_DATA_URL_PREFIX}{self.png_base64}"

    def to_image(self) -> Image.Image:
        """Decode the PNG into a detached Pillow image.

        This is CPU-bound and callers in async code should invoke it with
        ``asyncio.to_thread``.
        """

        from PIL import Image

        png_bytes = base64.b64decode(self.png_base64, validate=True)
        with Image.open(BytesIO(png_bytes)) as image:
            image.load()
            return image.copy()


def _validate_png_base64(encoded: str) -> None:
    try:
        png_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PageRenderError("render subprocess returned invalid PNG data") from exc

    if not png_bytes.startswith(PNG_SIGNATURE):
        raise PageRenderError("render subprocess returned invalid PNG data")


def render_page(pdf_path: Path, page_number: int, dpi: int) -> RenderedPage:
    """Rasterize a one-based PDF page number to a base64-encoded PNG."""

    import pymupdf

    scale = dpi / 72
    matrix = pymupdf.Matrix(scale, scale)
    with pymupdf.open(pdf_path) as document:
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")

    return RenderedPage(base64.b64encode(png_bytes).decode("ascii"))


def render_page_isolated(
    pdf_path: Path,
    page_number: int,
    dpi: int,
) -> RenderedPage:
    """Rasterize a page in a subprocess so native-library crashes are isolated."""

    command = [
        sys.executable,
        "-m",
        "pipeline.stg_01_ocr.render",
        str(pdf_path.expanduser().resolve()),
        str(page_number),
        str(dpi),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        if not detail:
            detail = f"render subprocess exited with code {result.returncode}"
        raise PageRenderError(detail)

    png_base64 = result.stdout.strip()
    _validate_png_base64(png_base64)
    return RenderedPage(png_base64)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one PDF page as base64 PNG.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("page_number", type=int)
    parser.add_argument("dpi", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # Native rendering libraries occasionally print deprecation notices to
    # stdout.  Keep stdout as a strict base64-only wire protocol.
    with redirect_stdout(sys.stderr):
        rendered = render_page(args.pdf_path, args.page_number, args.dpi)
    sys.stdout.write(rendered.png_base64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
