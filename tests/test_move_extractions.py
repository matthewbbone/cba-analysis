from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipeline.utils import move_extractions


class MoveExtractionsTests(unittest.TestCase):
    source = "source_a"

    @staticmethod
    def _write(path: Path, text: str = "data\n") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _build_source_cache(self, src_cache: Path) -> None:
        extraction_root = src_cache / "stg_02_extract" / self.source
        # doc_a deliberately appears under two extraction models; its
        # document ID must only be selected once for downstream stages.
        self._write(extraction_root / "extractor_a" / "doc_a" / "wage_tables.jsonl")
        self._write(extraction_root / "extractor_b" / "doc_a" / "wage_tables.jsonl")
        self._write(extraction_root / "extractor_b" / "doc_b" / "wage_tables.jsonl")

        # Stage 03 is keyed by the classification model, which need not match
        # the extraction model that produced its input.
        classify_root = src_cache / "stg_03_classify" / self.source
        self._write(classify_root / "classifier_a" / "doc_a" / "wage_tables.jsonl")
        self._write(classify_root / "classifier_a" / "doc_b" / "wage_tables.jsonl")
        self._write(classify_root / "classifier_b" / "doc_a" / "wage_tables.jsonl")
        # doc_c has classifications but no stage-02 extraction and must not move.
        self._write(classify_root / "classifier_a" / "doc_c" / "wage_tables.jsonl")

        # OCR text is never copied, whatever it contains.
        ocr_root = src_cache / "stg_01_ocr" / self.source
        self._write(ocr_root / "ocr_one" / "doc_a" / "full.txt")
        self._write(ocr_root / "ocr_one" / "doc_a" / "page_1.txt")
        self._write(ocr_root / "ocr_one" / "doc_b" / "full.txt")

    def _move_all(self, src_cache: Path, dst_cache: Path, *, dry_run: bool) -> None:
        document_dirs = move_extractions.move_source(
            self.source,
            "stg_02_extract",
            src_cache,
            dst_cache,
            dry_run,
        )
        move_extractions.move_classifications(
            self.source,
            "stg_03_classify",
            document_dirs,
            src_cache,
            dst_cache,
            dry_run,
        )

    def test_dry_run_deduplicates_stage_02_documents_and_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_cache = root / "source_cache"
            dst_cache = root / "destination_cache"
            self._build_source_cache(src_cache)
            output = StringIO()

            with redirect_stdout(output):
                self._move_all(src_cache, dst_cache, dry_run=True)

            dry_run_output = output.getvalue()

            self.assertFalse(dst_cache.exists())

        self.assertEqual(
            dry_run_output.count("classifier_a/doc_a/wage_tables.jsonl"), 1
        )
        self.assertEqual(
            dry_run_output.count("classifier_b/doc_a/wage_tables.jsonl"), 1
        )
        self.assertIn("extractor_a/doc_a/wage_tables.jsonl", dry_run_output)
        self.assertNotIn("doc_c", dry_run_output)
        self.assertNotIn("ocr_one", dry_run_output)
        self.assertNotIn("full.txt", dry_run_output)

    def test_moves_classifications_for_each_model_arm_of_extracted_documents(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_cache = root / "source_cache"
            dst_cache = root / "destination_cache"
            self._build_source_cache(src_cache)

            with redirect_stdout(StringIO()):
                self._move_all(src_cache, dst_cache, dry_run=False)

            classify_root = dst_cache / "stg_03_classify" / self.source
            classify_files = {
                path.relative_to(classify_root)
                for path in classify_root.rglob("*")
                if path.is_file()
            }

        self.assertEqual(
            classify_files,
            {
                Path("classifier_a/doc_a/wage_tables.jsonl"),
                Path("classifier_a/doc_b/wage_tables.jsonl"),
                Path("classifier_b/doc_a/wage_tables.jsonl"),
            },
        )

    def test_cli_filters_classifications_by_document_id(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_cache = root / "source_cache"
            dst_cache = root / "destination_cache"
            self._build_source_cache(src_cache)

            with redirect_stdout(StringIO()):
                move_extractions.main(
                    [
                        self.source,
                        "--src-cache",
                        str(src_cache),
                        "--dst-cache",
                        str(dst_cache),
                        "--document-id",
                        "doc_b",
                    ]
                )

            classify_root = dst_cache / "stg_03_classify" / self.source
            classify_files = {
                path.relative_to(classify_root)
                for path in classify_root.rglob("*")
                if path.is_file()
            }

        self.assertEqual(
            classify_files, {Path("classifier_a/doc_b/wage_tables.jsonl")}
        )

    def test_cli_no_classify_skips_stage_03(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_cache = root / "source_cache"
            dst_cache = root / "destination_cache"
            self._build_source_cache(src_cache)

            with redirect_stdout(StringIO()):
                move_extractions.main(
                    [
                        self.source,
                        "--src-cache",
                        str(src_cache),
                        "--dst-cache",
                        str(dst_cache),
                        "--no-classify",
                    ]
                )

            classify_exists = (dst_cache / "stg_03_classify").exists()
            extraction_exists = (
                dst_cache / "stg_02_extract" / self.source
            ).is_dir()

        self.assertFalse(classify_exists)
        self.assertTrue(extraction_exists)

    def test_cli_never_copies_ocr_text(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_cache = root / "source_cache"
            dst_cache = root / "destination_cache"
            self._build_source_cache(src_cache)

            with redirect_stdout(StringIO()):
                move_extractions.main(
                    [
                        self.source,
                        "--src-cache",
                        str(src_cache),
                        "--dst-cache",
                        str(dst_cache),
                    ]
                )

            ocr_exists = (dst_cache / "stg_01_ocr").exists()
            extraction_exists = (
                dst_cache / "stg_02_extract" / self.source
            ).is_dir()

        self.assertFalse(ocr_exists)
        self.assertTrue(extraction_exists)


if __name__ == "__main__":
    unittest.main()
