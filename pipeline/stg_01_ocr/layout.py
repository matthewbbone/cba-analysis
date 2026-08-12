"""Cached PP-DocLayoutV3 detection for stage-01 OCR runners.

The detector is deliberately run before vLLM starts.  Its JSON cache is
shared by every recognition arm so crop geometry and reading order cannot
silently become another benchmark variable.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from pipeline.stg_01_ocr.render import RenderedPage, render_page_isolated
from pipeline.utils.paths import path_safe_model_name


DEFAULT_LAYOUT_MODEL_NAME = "PaddlePaddle/PP-DocLayoutV3_safetensors"
LAYOUT_CACHE_VERSION = 3
IGNORED_LAYOUT_LABELS = frozenset(
    {
        "number",
        "page_number",
        "page number",
        "footnote",
        "header",
        "header_image",
        "header image",
        "footer",
        "footer_image",
        "footer image",
        "aside_text",
        "aside text",
    }
)


@dataclass(frozen=True)
class LayoutBlock:
    """One page region in detector-provided reading order."""

    label: str
    score: float
    box: tuple[float, float, float, float]
    order: int
    polygon_points: tuple[tuple[float, float], ...] = ()
    rotation: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LayoutBlock":
        if not isinstance(value, dict):
            raise ValueError("layout block must be an object")
        box_value = value.get("box")
        if not isinstance(box_value, (list, tuple)) or len(box_value) != 4:
            raise ValueError("layout block box must contain four coordinates")
        polygon_value = value.get("polygon_points", [])
        if not isinstance(polygon_value, (list, tuple)):
            raise ValueError("layout polygon_points must be a sequence")
        try:
            label = str(value["label"])
        except KeyError as exc:
            raise ValueError("layout block is missing label") from exc
        return cls(
            label=label,
            score=float(value.get("score", 1.0)),
            box=tuple(float(item) for item in box_value),  # type: ignore[arg-type]
            order=int(value.get("order", 0)),
            polygon_points=tuple(
                (float(point[0]), float(point[1]))
                for point in polygon_value
            ),
            rotation=value.get("rotation"),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["box"] = list(self.box)
        value["polygon_points"] = [list(point) for point in self.polygon_points]
        return value


def normalise_layout_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("-", "_").split())


def should_ignore_layout_label(label: str) -> bool:
    normalised = normalise_layout_label(label)
    return normalised in IGNORED_LAYOUT_LABELS or normalised.replace("_", " ") in IGNORED_LAYOUT_LABELS


def layout_cache_path(
    output_root: Path,
    source: str,
    document_id: str,
    model_name: str = DEFAULT_LAYOUT_MODEL_NAME,
) -> Path:
    """Return the arm-independent cache path for one document's layout."""

    return (
        output_root.expanduser()
        / source
        / "_layout"
        / path_safe_model_name(model_name)
        / document_id
        / "layout.json"
    )


def calculate_pdf_sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as pdf_file:
        for chunk in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class PPDocLayoutDetector:
    """Lazy wrapper around the transformers PP-DocLayoutV3 implementation."""

    def __init__(
        self,
        model_name: str = DEFAULT_LAYOUT_MODEL_NAME,
        *,
        threshold: float = 0.3,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, PPDocLayoutV3ForObjectDetection

        self.model_name = model_name
        self.threshold = threshold
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {
            int(key): str(value)
            for key, value in self.model.config.id2label.items()
        }

    def detect(self, image: Any) -> list[LayoutBlock]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor(
            [[image.height, image.width]],
            device=self.device,
        )
        processed = self.processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]

        scores = _plain(processed.get("scores", []))
        labels = _plain(processed.get("labels", []))
        boxes = _plain(processed.get("boxes", []))
        polygons = _plain(processed.get("polygon_points", []))
        orders = _plain(processed.get("order_seq", []))
        if not orders:
            orders = list(range(len(boxes)))

        blocks: list[LayoutBlock] = []
        # The checkpoint's post-processor already returns reading order.  Keep
        # that sequence exactly; ``order_seq`` is metadata, not a new sort key.
        for index, (score, label_id, box) in enumerate(zip(scores, labels, boxes)):
            label = self.id2label.get(int(label_id), str(label_id))
            if should_ignore_layout_label(label):
                continue
            polygon = polygons[index] if index < len(polygons) else []
            order_value = orders[index] if index < len(orders) else index
            if isinstance(order_value, (list, tuple)):
                order_value = order_value[0] if order_value else index
            blocks.append(
                LayoutBlock(
                    label=label,
                    score=float(score),
                    box=tuple(float(coordinate) for coordinate in box),  # type: ignore[arg-type]
                    order=int(order_value),
                    polygon_points=tuple(
                        (float(point[0]), float(point[1])) for point in polygon
                    ),
                )
            )
        return blocks


def write_layout_cache(
    path: Path,
    *,
    model_name: str,
    dpi: int,
    threshold: float,
    pdf_sha256: str,
    pages: dict[int, Sequence[LayoutBlock]],
) -> None:
    payload = {
        "version": LAYOUT_CACHE_VERSION,
        "model_name": model_name,
        "dpi": dpi,
        "threshold": threshold,
        "pdf_sha256": pdf_sha256,
        "pages": {
            str(page_number): [block.to_dict() for block in blocks]
            for page_number, blocks in sorted(pages.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_layout_cache(
    path: Path,
    *,
    expected_model_name: str | None = None,
    expected_dpi: int | None = None,
    expected_threshold: float | None = None,
    expected_pdf_sha256: str | None = None,
    expected_pages: int | None = None,
) -> dict[int, tuple[LayoutBlock, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Layout cache root must be an object in {path}")
    if payload.get("version") != LAYOUT_CACHE_VERSION:
        raise ValueError(f"Unsupported layout cache version in {path}")
    if expected_model_name is not None and payload.get("model_name") != expected_model_name:
        raise ValueError(f"Layout model mismatch in {path}")
    if expected_dpi is not None and payload.get("dpi") != expected_dpi:
        raise ValueError(f"Layout DPI mismatch in {path}")
    if expected_threshold is not None:
        try:
            threshold = float(payload.get("threshold", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid layout threshold in {path}") from exc
        if not math.isclose(threshold, expected_threshold, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Layout threshold mismatch in {path}")
    if (
        expected_pdf_sha256 is not None
        and payload.get("pdf_sha256") != expected_pdf_sha256
    ):
        raise ValueError(f"Layout PDF fingerprint mismatch in {path}")

    pages_payload = payload.get("pages", {})
    if not isinstance(pages_payload, dict):
        raise ValueError(f"Layout pages must be an object in {path}")
    pages: dict[int, tuple[LayoutBlock, ...]] = {}
    try:
        for page_number, blocks in pages_payload.items():
            if not isinstance(blocks, list):
                raise ValueError("layout page blocks must be a list")
            pages[int(page_number)] = tuple(
                LayoutBlock.from_dict(block) for block in blocks
            )
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Malformed layout cache in {path}") from exc
    if expected_pages is not None and set(pages) != set(range(1, expected_pages + 1)):
        raise ValueError(f"Layout page set mismatch in {path}")
    return pages


async def build_layout_cache(
    *,
    pdf_path: Path,
    cache_path: Path,
    page_count: int,
    dpi: int,
    detector: PPDocLayoutDetector,
    pdf_digest: str | None = None,
) -> dict[int, tuple[LayoutBlock, ...]]:
    if pdf_digest is None:
        pdf_digest = await asyncio.to_thread(calculate_pdf_sha256, pdf_path)
    pages: dict[int, tuple[LayoutBlock, ...]] = {}
    for page_number in range(1, page_count + 1):
        rendered: RenderedPage = await asyncio.to_thread(
            render_page_isolated,
            pdf_path,
            page_number,
            dpi,
        )
        image = await asyncio.to_thread(rendered.to_image)
        try:
            blocks = await asyncio.to_thread(detector.detect, image)
        finally:
            image.close()
        pages[page_number] = tuple(blocks)

    await asyncio.to_thread(
        write_layout_cache,
        cache_path,
        model_name=detector.model_name,
        dpi=dpi,
        threshold=detector.threshold,
        pdf_sha256=pdf_digest,
        pages=pages,
    )
    return pages


async def ensure_layout_cache(
    *,
    pdf_path: Path,
    cache_path: Path,
    page_count: int,
    dpi: int,
    model_name: str = DEFAULT_LAYOUT_MODEL_NAME,
    threshold: float = 0.3,
    force: bool = False,
    detector: PPDocLayoutDetector | None = None,
) -> dict[int, tuple[LayoutBlock, ...]]:
    pdf_digest = await asyncio.to_thread(calculate_pdf_sha256, pdf_path)
    if cache_path.exists() and not force:
        try:
            return await asyncio.to_thread(
                read_layout_cache,
                cache_path,
                expected_model_name=model_name,
                expected_dpi=dpi,
                expected_threshold=threshold,
                expected_pdf_sha256=pdf_digest,
                expected_pages=page_count,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    actual_detector = detector or await asyncio.to_thread(
        PPDocLayoutDetector,
        model_name,
        threshold=threshold,
    )
    return await build_layout_cache(
        pdf_path=pdf_path,
        cache_path=cache_path,
        page_count=page_count,
        dpi=dpi,
        detector=actual_detector,
        pdf_digest=pdf_digest,
    )


def flatten_layout_pages(
    caches: Iterable[tuple[str, str, dict[int, tuple[LayoutBlock, ...]]]],
) -> dict[tuple[str, str, int], tuple[LayoutBlock, ...]]:
    return {
        (source, document_id, page_number): blocks
        for source, document_id, pages in caches
        for page_number, blocks in pages.items()
    }
