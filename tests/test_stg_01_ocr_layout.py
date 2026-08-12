from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pipeline.stg_01_ocr import common
from pipeline.stg_01_ocr.layout import (
    LayoutBlock,
    layout_cache_path,
    read_layout_cache,
    should_ignore_layout_label,
    write_layout_cache,
)


class LayoutCacheTests(unittest.TestCase):
    def test_cache_path_is_shared_across_recognition_arms(self) -> None:
        path = layout_cache_path(
            Path("out"),
            "source",
            "document",
            "PaddlePaddle/PP-DocLayoutV3_safetensors",
        )
        self.assertEqual(
            path,
            Path("out/source/_layout/PaddlePaddle_PP-DocLayoutV3_safetensors/document/layout.json"),
        )

    def test_layout_cache_round_trip_preserves_detector_order(self) -> None:
        blocks = (
            LayoutBlock("text", 0.9, (1, 2, 30, 40), 8),
            LayoutBlock(
                "table",
                0.8,
                (2, 4, 60, 80),
                3,
                ((2, 4), (60, 4), (60, 80), (2, 80)),
            ),
        )
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            write_layout_cache(
                path,
                model_name="model",
                dpi=200,
                threshold=0.3,
                pdf_sha256="abc",
                pages={1: blocks},
            )
            loaded = read_layout_cache(
                path,
                expected_model_name="model",
                expected_dpi=200,
                expected_threshold=0.3,
                expected_pdf_sha256="abc",
                expected_pages=1,
            )

        self.assertEqual(loaded[1], blocks)
        self.assertEqual([block.order for block in loaded[1]], [8, 3])

    def test_cache_validation_rejects_wrong_dpi(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            write_layout_cache(
                path,
                model_name="model",
                dpi=200,
                threshold=0.3,
                pdf_sha256="abc",
                pages={1: ()},
            )
            with self.assertRaisesRegex(ValueError, "DPI mismatch"):
                read_layout_cache(path, expected_dpi=300)

    def test_cache_validation_rejects_wrong_threshold(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            write_layout_cache(
                path,
                model_name="model",
                dpi=200,
                threshold=0.3,
                pdf_sha256="abc",
                pages={1: ()},
            )
            with self.assertRaisesRegex(ValueError, "threshold mismatch"):
                read_layout_cache(path, expected_threshold=0.5)

    def test_cache_validation_rejects_replaced_pdf(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            write_layout_cache(
                path,
                model_name="model",
                dpi=200,
                threshold=0.3,
                pdf_sha256="old",
                pages={1: ()},
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                read_layout_cache(path, expected_pdf_sha256="new")

    def test_malformed_parseable_cache_raises_validation_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root must be an object"):
                read_layout_cache(path)

    def test_ignored_labels_cover_vendor_header_footer_names(self) -> None:
        for label in ("number", "Footnote", "header_image", "footer-image", "aside text"):
            with self.subTest(label=label):
                self.assertTrue(should_ignore_layout_label(label))
        self.assertFalse(should_ignore_layout_label("table"))


class LayoutPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_caches_reuse_one_detector_and_honor_device(self) -> None:
        async def immediate(function, *args, **kwargs):
            return function(*args, **kwargs)

        with TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "out"
            documents = [
                common.DocumentJob("source", "one", Path("one.pdf"), output_root / "arm/one"),
                common.DocumentJob("source", "two", Path("two.pdf"), output_root / "arm/two"),
            ]
            states = {
                (document.source, document.document_id): common.DocumentState(
                    document.source,
                    document.document_id,
                    document.output_dir,
                    1,
                )
                for document in documents
            }
            args = SimpleNamespace(
                output_root=output_root,
                layout_model_name="layout/model",
                force_layout=False,
                dpi=200,
                layout_threshold=0.4,
                layout_device="cuda:1",
            )
            for document in documents:
                stale = layout_cache_path(
                    output_root,
                    document.source,
                    document.document_id,
                    args.layout_model_name,
                )
                stale.parent.mkdir(parents=True, exist_ok=True)
                stale.write_text("{}", encoding="utf-8")

            detector = object()
            build = AsyncMock(return_value={1: ()})
            with (
                patch("pipeline.stg_01_ocr.common.asyncio.to_thread", side_effect=immediate),
                patch(
                    "pipeline.stg_01_ocr.common.PPDocLayoutDetector",
                    return_value=detector,
                ) as detector_class,
                patch("pipeline.stg_01_ocr.common.build_layout_cache", build),
                patch(
                    "pipeline.stg_01_ocr.common.calculate_pdf_sha256",
                    return_value="digest",
                ),
                patch("pipeline.stg_01_ocr.common._release_layout_memory"),
            ):
                pages = await common._prepare_shared_layouts(documents, states, args)

        detector_class.assert_called_once_with(
            "layout/model",
            threshold=0.4,
            device="cuda:1",
        )
        self.assertEqual(build.await_count, 2)
        self.assertTrue(all(call.kwargs["detector"] is detector for call in build.await_args_list))
        self.assertEqual(set(pages), {("source", "one", 1), ("source", "two", 1)})


if __name__ == "__main__":
    unittest.main()
