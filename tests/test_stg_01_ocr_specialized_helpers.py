import base64
from io import BytesIO
import unittest
from unittest.mock import patch

from PIL import Image

from pipeline.stg_01_ocr.layout import LayoutBlock
from pipeline.stg_01_ocr.render import PNG_DATA_URL_PREFIX, PNG_SIGNATURE, RenderedPage
from pipeline.stg_01_ocr.specialized._shared import (
    MINER_NATIVE_LAYOUT_RE,
    classify_layout_label,
    parse_miner_native_layout,
    prepare_miner_native_layout_page,
    preprocess_layout_crops,
    preprocess_miner_native_crops,
)


def rendered_page(image: Image.Image) -> RenderedPage:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return RenderedPage(base64.b64encode(buffer.getvalue()).decode("ascii"))


def decode_data_url(data_url: str) -> Image.Image:
    if not data_url.startswith(PNG_DATA_URL_PREFIX):
        raise AssertionError(f"Not a PNG data URL: {data_url[:40]!r}")
    raw = base64.b64decode(data_url.removeprefix(PNG_DATA_URL_PREFIX), validate=True)
    if not raw.startswith(PNG_SIGNATURE):
        raise AssertionError("Data URL payload does not have a PNG signature")
    with Image.open(BytesIO(raw)) as image:
        image.load()
        return image.copy()


def block(
    *,
    label: str = "text",
    box: tuple[float, float, float, float],
    order: int = 0,
    rotation: str | None = None,
) -> LayoutBlock:
    return LayoutBlock(
        label=label,
        score=1.0,
        box=box,
        order=order,
        rotation=rotation,
    )


def miner_region(
    coordinates: tuple[int, int, int, int],
    label: str,
    rotation: str | None = "up",
    tail: str = "",
) -> str:
    coordinate_text = " ".join(str(value) for value in coordinates)
    rotation_token = "" if rotation is None else f"<|rotate_{rotation}|>"
    return (
        f"<|box_start|>{coordinate_text}<|box_end|>"
        f"<|ref_start|>{label}<|ref_end|>{rotation_token}{tail}"
    )


class CountingRenderedPage:
    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.to_image_calls = 0

    def to_image(self) -> Image.Image:
        self.to_image_calls += 1
        return self.image.copy()


class LayoutLabelTests(unittest.TestCase):
    def test_classifies_prompt_families_and_defaults_unknown_labels_to_text(self) -> None:
        cases = {
            "table": "table",
            " Formula-Number ": "formula",
            "equation_block": "formula",
            "chart": "chart",
            "image": "image",
            "seal": "image",
            "paragraph_title": "text",
            "future_detector_label": "text",
        }

        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(classify_layout_label(label), expected)


class CropPreprocessingTests(unittest.TestCase):
    def test_clamps_and_outward_rounds_fractional_crop_boxes(self) -> None:
        image = Image.new("RGB", (4, 3))
        for y in range(image.height):
            for x in range(image.width):
                image.putpixel((x, y), (x * 40, y * 60, 10))

        crops = preprocess_layout_crops(
            rendered_page(image),
            [block(box=(-4.5, -3.0, 1.2, 1.1))],
            min_edge=1,
        )

        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].size, (2, 2))
        self.assertEqual(crops[0].block.box, (-4.5, -3.0, 1.2, 1.1))
        crop_image = decode_data_url(crops[0].data_url)
        self.assertEqual(crop_image.getpixel((0, 0)), (0, 0, 10))
        self.assertEqual(crop_image.getpixel((1, 1)), (40, 60, 10))

    def test_rotates_all_miner_directions_to_upright(self) -> None:
        image = Image.new("RGB", (2, 1))
        red = (255, 0, 0)
        blue = (0, 0, 255)
        image.putpixel((0, 0), red)
        image.putpixel((1, 0), blue)
        blocks = [
            block(box=(0, 0, 2, 1), order=index, rotation=rotation)
            for index, rotation in enumerate(("up", "right", "down", "left"))
        ]

        crops = preprocess_layout_crops(
            rendered_page(image),
            blocks,
            min_edge=1,
        )
        images = [decode_data_url(crop.data_url) for crop in crops]

        self.assertEqual(images[0].size, (2, 1))
        self.assertEqual([images[0].getpixel((x, 0)) for x in range(2)], [red, blue])
        self.assertEqual(images[1].size, (1, 2))
        self.assertEqual([images[1].getpixel((0, y)) for y in range(2)], [blue, red])
        self.assertEqual(images[2].size, (2, 1))
        self.assertEqual([images[2].getpixel((x, 0)) for x in range(2)], [blue, red])
        self.assertEqual(images[3].size, (1, 2))
        self.assertEqual([images[3].getpixel((0, y)) for y in range(2)], [red, blue])

    def test_accepts_full_rotation_tokens(self) -> None:
        image = Image.new("RGB", (2, 1), "black")

        crops = preprocess_layout_crops(
            rendered_page(image),
            [block(box=(0, 0, 2, 1), rotation="<|rotate_right|>")],
            min_edge=1,
        )

        self.assertEqual(crops[0].size, (1, 2))

    def test_white_pads_extreme_aspect_ratio(self) -> None:
        image = Image.new("RGB", (100, 1), "black")

        crops = preprocess_layout_crops(
            rendered_page(image),
            [block(box=(0, 0, 100, 1))],
            max_aspect_ratio=50,
            min_edge=1,
        )

        self.assertEqual(crops[0].size, (100, 2))
        padded = decode_data_url(crops[0].data_url)
        self.assertEqual(padded.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(padded.getpixel((0, 1)), (255, 255, 255))
        self.assertLessEqual(max(padded.size) / min(padded.size), 50)

    def test_bicubic_upscales_to_minimum_edge_without_downscaling(self) -> None:
        original_resize = Image.Image.resize
        with patch.object(
            Image.Image,
            "resize",
            autospec=True,
            side_effect=original_resize,
        ) as resize:
            small = preprocess_layout_crops(
                rendered_page(Image.new("RGB", (2, 3), "black")),
                [block(box=(0, 0, 2, 3))],
            )
        large = preprocess_layout_crops(
            rendered_page(Image.new("RGB", (30, 40), "black")),
            [block(box=(0, 0, 30, 40))],
        )

        self.assertEqual(small[0].size, (28, 42))
        self.assertEqual(large[0].size, (30, 40))
        self.assertEqual(resize.call_count, 1)
        self.assertIs(resize.call_args.args[2], Image.Resampling.BICUBIC)

    def test_decodes_page_once_for_all_crops(self) -> None:
        rendered = CountingRenderedPage(Image.new("RGB", (10, 10), "black"))

        crops = preprocess_layout_crops(  # type: ignore[arg-type]
            rendered,
            [
                block(box=(0, 0, 5, 5), order=0),
                block(box=(5, 5, 10, 10), order=1),
            ],
            min_edge=1,
        )

        self.assertEqual(len(crops), 2)
        self.assertEqual(rendered.to_image_calls, 1)

    def test_rejects_empty_box_after_clamping(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty after clamping"):
            preprocess_layout_crops(
                rendered_page(Image.new("RGB", (10, 10), "black")),
                [block(box=(12, 0, 15, 5))],
                min_edge=1,
            )


class MinerNativeLayoutTests(unittest.TestCase):
    def test_parser_regex_is_the_prescribed_miner_protocol(self) -> None:
        expected = (
            r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|>"
            r"<\|ref_start\|>(\w+?)<\|ref_end\|>"
            r"(?:(<\|rotate_(?:up|right|down|left)\|>))?(.*?)(?=<\|box_start\|>|$)"
        )

        self.assertEqual(MINER_NATIVE_LAYOUT_RE, expected)

    def test_parser_scales_boxes_filters_ignored_labels_and_preserves_order(self) -> None:
        output = "".join(
            [
                miner_region((0, 0, 1000, 50), "header"),
                miner_region((900, 800, 100, 200), "TEXT", "left", "metadata\n"),
                miner_region((250, 100, 750, 900), "table", "right"),
                miner_region((0, 950, 1000, 1000), "footer"),
                miner_region((20, 20, 20, 30), "text"),
                miner_region((0, 0, 1001, 1000), "text"),
            ]
        )

        blocks = parse_miner_native_layout(
            output,
            image_width=200,
            image_height=100,
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].label, "text")
        self.assertEqual(blocks[0].box, (20.0, 20.0, 180.0, 80.0))
        self.assertEqual(blocks[0].rotation, "left")
        self.assertEqual(blocks[0].order, 0)
        self.assertEqual(blocks[1].label, "table")
        self.assertEqual(blocks[1].box, (50.0, 10.0, 150.0, 90.0))
        self.assertEqual(blocks[1].rotation, "right")
        self.assertEqual(blocks[1].order, 1)

    def test_parser_supports_each_optional_rotation_token(self) -> None:
        output = "".join(
            miner_region((0, index * 100, 100, index * 100 + 50), "text", rotation)
            for index, rotation in enumerate(("up", "right", "down", "left", None))
        )

        blocks = parse_miner_native_layout(
            output,
            image_width=1000,
            image_height=1000,
        )

        self.assertEqual(
            [item.rotation for item in blocks],
            ["up", "right", "down", "left", None],
        )

    def test_native_parse_and_crop_pass_decodes_the_page_once(self) -> None:
        image = Image.new("RGB", (200, 100), "white")
        image.paste("black", (20, 20, 180, 80))
        rendered = CountingRenderedPage(image)
        output = "".join(
            [
                miner_region((0, 0, 1000, 100), "page_number"),
                miner_region((100, 200, 900, 800), "table", "up"),
            ]
        )

        crops = preprocess_miner_native_crops(  # type: ignore[arg-type]
            rendered,
            output,
            min_edge=1,
        )

        self.assertEqual(rendered.to_image_calls, 1)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].block.label, "table")
        self.assertEqual(crops[0].block.box, (20.0, 20.0, 180.0, 80.0))
        self.assertEqual(crops[0].size, (160, 60))
        cropped = decode_data_url(crops[0].data_url)
        self.assertEqual(cropped.getpixel((0, 0)), (0, 0, 0))

    def test_native_layout_page_is_bicubic_fixed_size_and_reports_original_size(self) -> None:
        page = rendered_page(Image.new("RGB", (9, 4), "black"))

        original_resize = Image.Image.resize
        with patch.object(
            Image.Image,
            "resize",
            autospec=True,
            side_effect=original_resize,
        ) as resize:
            prepared = prepare_miner_native_layout_page(page, size=(7, 7))

        self.assertEqual((prepared.image_width, prepared.image_height), (9, 4))
        self.assertEqual(decode_data_url(prepared.data_url).size, (7, 7))
        self.assertEqual(resize.call_count, 1)
        self.assertIs(resize.call_args.args[2], Image.Resampling.BICUBIC)


if __name__ == "__main__":
    unittest.main()
