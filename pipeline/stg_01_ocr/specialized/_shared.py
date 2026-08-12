"""CPU-only helpers shared by the specialized stage-01 OCR runners.

The asynchronous runners deliberately call the image preparation functions in
this module through one ``asyncio.to_thread`` call per page.  Keeping the whole
decode/crop/rotate/resize/encode pass synchronous here both bounds event-loop
work and guarantees identical crop geometry for every benchmark arm.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import math
import re
from typing import Literal, Sequence

from PIL import Image

from pipeline.stg_01_ocr import layout
from pipeline.stg_01_ocr.render import PNG_DATA_URL_PREFIX, RenderedPage


RegionKind = Literal["text", "table", "formula", "chart", "image"]

DEFAULT_MAX_CROP_ASPECT_RATIO = 50.0
DEFAULT_MIN_CROP_EDGE = 28
MINER_NATIVE_LAYOUT_SIZE = (1036, 1036)

# Keep this expression byte-for-byte aligned with MinerU's native protocol.
MINER_NATIVE_LAYOUT_RE = (
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|>"
    r"<\|ref_start\|>(\w+?)<\|ref_end\|>"
    r"(?:(<\|rotate_(?:up|right|down|left)\|>))?(.*?)(?=<\|box_start\|>|$)"
)
MINER_NATIVE_LAYOUT_PATTERN = re.compile(MINER_NATIVE_LAYOUT_RE, re.DOTALL)


_TABLE_LABELS = frozenset({"table"})
_FORMULA_LABELS = frozenset(
    {
        "equation",
        "equation_block",
        "formula",
        "formula_number",
        "inline_formula",
    }
)
_CHART_LABELS = frozenset({"chart"})
_IMAGE_LABELS = frozenset({"figure", "image", "image_block", "seal"})

_ROTATION_DEGREES = {
    "up": 0,
    "right": 90,
    "down": 180,
    "left": 270,
}


@dataclass(frozen=True)
class PreparedCrop:
    """One prepared region image, paired with the block that produced it."""

    block: layout.LayoutBlock
    data_url: str
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def kind(self) -> RegionKind:
        return classify_layout_label(self.block.label)


@dataclass(frozen=True)
class PreparedMinerLayoutPage:
    """MinerU's fixed-size layout input plus the original page dimensions."""

    data_url: str
    image_width: int
    image_height: int


def classify_layout_label(label: str) -> RegionKind:
    """Collapse detector- and MinerU-specific labels into prompt families.

    Unknown labels intentionally fall back to text recognition.  This is the
    lossless default for headings, captions, lists, references, and any labels
    introduced by a future detector checkpoint.
    """

    normalised = layout.normalise_layout_label(label).replace(" ", "_")
    if normalised in _TABLE_LABELS:
        return "table"
    if normalised in _FORMULA_LABELS:
        return "formula"
    if normalised in _CHART_LABELS:
        return "chart"
    if normalised in _IMAGE_LABELS:
        return "image"
    return "text"


def _png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"{PNG_DATA_URL_PREFIX}{encoded}"


def _normalise_rotation(rotation: str | None) -> str:
    if rotation is None:
        return "up"
    value = rotation.strip().lower()
    if value.startswith("<|rotate_") and value.endswith("|>"):
        value = value[len("<|rotate_") : -len("|>")]
    elif value.startswith("rotate_"):
        value = value[len("rotate_") :]
    if value not in _ROTATION_DEGREES:
        raise ValueError(f"Unsupported layout rotation: {rotation!r}")
    return value


def _rotate_upright(image: Image.Image, rotation: str | None) -> Image.Image:
    """Rotate an oriented crop to upright using MinerU token semantics."""

    direction = _normalise_rotation(rotation)
    angle = _ROTATION_DEGREES[direction]
    if angle == 0:
        return image
    # Pillow's positive angles are counter-clockwise, which is also how the
    # official MinerU helper turns rightward/downward/leftward content upright.
    transpose = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }[angle]
    return image.transpose(transpose)


def _pad_aspect_ratio(image: Image.Image, max_aspect_ratio: float) -> Image.Image:
    width, height = image.size
    if width / height > max_aspect_ratio:
        padded_size = (width, math.ceil(width / max_aspect_ratio))
    elif height / width > max_aspect_ratio:
        padded_size = (math.ceil(height / max_aspect_ratio), height)
    else:
        return image

    padded = Image.new("RGB", padded_size, "white")
    offset = (
        (padded.width - width) // 2,
        (padded.height - height) // 2,
    )
    padded.paste(image, offset)
    return padded


def _upscale_min_edge(image: Image.Image, min_edge: int) -> Image.Image:
    current_min_edge = min(image.size)
    if current_min_edge >= min_edge:
        return image
    scale = min_edge / current_min_edge
    size = (
        math.ceil(image.width * scale),
        math.ceil(image.height * scale),
    )
    return image.resize(size, Image.Resampling.BICUBIC)


def _crop_box(
    block: layout.LayoutBlock,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if len(block.box) != 4 or not all(math.isfinite(value) for value in block.box):
        raise ValueError(f"Invalid layout box for block {block!r}")

    x1, y1, x2, y2 = block.box
    left = max(0, min(image_width, math.floor(min(x1, x2))))
    top = max(0, min(image_height, math.floor(min(y1, y2))))
    right = max(0, min(image_width, math.ceil(max(x1, x2))))
    bottom = max(0, min(image_height, math.ceil(max(y1, y2))))
    if right <= left or bottom <= top:
        raise ValueError(
            f"Layout box {block.box!r} is empty after clamping to "
            f"{image_width}x{image_height}"
        )
    return (left, top, right, bottom)


def preprocess_layout_crops(
    rendered_page: RenderedPage,
    blocks: Sequence[layout.LayoutBlock],
    *,
    max_aspect_ratio: float = DEFAULT_MAX_CROP_ASPECT_RATIO,
    min_edge: int = DEFAULT_MIN_CROP_EDGE,
) -> tuple[PreparedCrop, ...]:
    """Prepare every layout crop in one synchronous per-page CPU pass.

    Boxes are clamped to page bounds, outward-rounded so fractional detector
    coordinates do not discard edge pixels, rotated upright, white-padded to
    the requested maximum aspect ratio, bicubic-upscaled when necessary, and
    encoded as PNG data URLs.  Input order is preserved.
    """

    if not math.isfinite(max_aspect_ratio) or max_aspect_ratio < 1:
        raise ValueError("max_aspect_ratio must be a finite number >= 1")
    if isinstance(min_edge, bool) or not isinstance(min_edge, int) or min_edge < 1:
        raise ValueError("min_edge must be an integer >= 1")

    page = rendered_page.to_image()
    try:
        return _preprocess_image_crops(
            page,
            blocks,
            max_aspect_ratio=max_aspect_ratio,
            min_edge=min_edge,
        )
    finally:
        page.close()


def _preprocess_image_crops(
    page: Image.Image,
    blocks: Sequence[layout.LayoutBlock],
    *,
    max_aspect_ratio: float,
    min_edge: int,
) -> tuple[PreparedCrop, ...]:
    if page.mode != "RGB":
        rgb_page = page.convert("RGB")
    else:
        rgb_page = page

    try:
        prepared: list[PreparedCrop] = []
        for block in blocks:
            box = _crop_box(
                block,
                image_width=rgb_page.width,
                image_height=rgb_page.height,
            )
            crop = rgb_page.crop(box)
            try:
                transformed = _rotate_upright(crop, block.rotation)
                if transformed is not crop:
                    crop.close()
                    crop = transformed
                transformed = _pad_aspect_ratio(crop, max_aspect_ratio)
                if transformed is not crop:
                    crop.close()
                    crop = transformed
                transformed = _upscale_min_edge(crop, min_edge)
                if transformed is not crop:
                    crop.close()
                    crop = transformed
                prepared.append(
                    PreparedCrop(
                        block=block,
                        data_url=_png_data_url(crop),
                        width=crop.width,
                        height=crop.height,
                    )
                )
            finally:
                crop.close()
        return tuple(prepared)
    finally:
        if rgb_page is not page:
            rgb_page.close()


def preprocess_miner_native_crops(
    rendered_page: RenderedPage,
    layout_output: str,
    *,
    max_aspect_ratio: float = DEFAULT_MAX_CROP_ASPECT_RATIO,
    min_edge: int = DEFAULT_MIN_CROP_EDGE,
) -> tuple[PreparedCrop, ...]:
    """Parse and prepare MinerU native regions with one page-image decode."""

    if not math.isfinite(max_aspect_ratio) or max_aspect_ratio < 1:
        raise ValueError("max_aspect_ratio must be a finite number >= 1")
    if isinstance(min_edge, bool) or not isinstance(min_edge, int) or min_edge < 1:
        raise ValueError("min_edge must be an integer >= 1")

    page = rendered_page.to_image()
    try:
        blocks = parse_miner_native_layout(
            layout_output,
            image_width=page.width,
            image_height=page.height,
        )
        return _preprocess_image_crops(
            page,
            blocks,
            max_aspect_ratio=max_aspect_ratio,
            min_edge=min_edge,
        )
    finally:
        page.close()


def prepare_miner_native_layout_page(
    rendered_page: RenderedPage,
    *,
    size: tuple[int, int] = MINER_NATIVE_LAYOUT_SIZE,
) -> PreparedMinerLayoutPage:
    """Resize a full page to MinerU's fixed layout input, without preserving aspect."""

    width, height = size
    if width < 1 or height < 1:
        raise ValueError("size dimensions must be >= 1")

    page = rendered_page.to_image()
    try:
        original_width, original_height = page.size
        if page.mode != "RGB":
            rgb_page = page.convert("RGB")
        else:
            rgb_page = page
        try:
            resized = rgb_page.resize(size, Image.Resampling.BICUBIC)
            try:
                return PreparedMinerLayoutPage(
                    data_url=_png_data_url(resized),
                    image_width=original_width,
                    image_height=original_height,
                )
            finally:
                resized.close()
        finally:
            if rgb_page is not page:
                rgb_page.close()
    finally:
        page.close()


def parse_miner_native_layout(
    output: str,
    *,
    image_width: int,
    image_height: int,
) -> tuple[layout.LayoutBlock, ...]:
    """Parse MinerU layout tokens into pixel-space ``LayoutBlock`` objects.

    Coordinates outside MinerU's documented 0--1000 range and zero-area boxes
    are ignored.  Reversed endpoints are accepted and ordered before scaling,
    matching MinerU's native utility.  Header/footer/page-number families are
    filtered through the shared layout policy.
    """

    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be >= 1")

    blocks: list[layout.LayoutBlock] = []
    for match in MINER_NATIVE_LAYOUT_PATTERN.finditer(output):
        x1, y1, x2, y2 = (int(value) for value in match.group(1, 2, 3, 4))
        if any(value < 0 or value > 1000 for value in (x1, y1, x2, y2)):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x1 == x2 or y1 == y2:
            continue

        label = match.group(5).lower()
        if layout.should_ignore_layout_label(label):
            continue

        rotate_token = match.group(6)
        rotation = None
        if rotate_token is not None:
            rotation = rotate_token.removeprefix("<|rotate_").removesuffix("|>")

        blocks.append(
            layout.LayoutBlock(
                label=label,
                score=1.0,
                box=(
                    x1 * image_width / 1000,
                    y1 * image_height / 1000,
                    x2 * image_width / 1000,
                    y2 * image_height / 1000,
                ),
                order=len(blocks),
                rotation=rotation,
            )
        )
    return tuple(blocks)


__all__ = [
    "DEFAULT_MAX_CROP_ASPECT_RATIO",
    "DEFAULT_MIN_CROP_EDGE",
    "MINER_NATIVE_LAYOUT_PATTERN",
    "MINER_NATIVE_LAYOUT_RE",
    "MINER_NATIVE_LAYOUT_SIZE",
    "PreparedCrop",
    "PreparedMinerLayoutPage",
    "RegionKind",
    "classify_layout_label",
    "parse_miner_native_layout",
    "prepare_miner_native_layout_page",
    "preprocess_layout_crops",
    "preprocess_miner_native_crops",
]
