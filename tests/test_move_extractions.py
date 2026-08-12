from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipeline.stg_02_extract import move_extractions


class MoveExtractionsTests(unittest.TestCase):
    source = "source_a"

    @staticmethod
    def _write(path: Path, text: str = "data\n") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _build_source_cache(self, src_cache: Path) -> None:
        extraction_root = src_cache / "stg_02_extract" / self.source
        # doc_a deliberately appears under two extraction models. Its OCR
        # artifacts must still be selected and copied only once per OCR arm.
        self._write(extraction_root / "extractor_a" / "doc_a" / "wage_tables.jsonl")
        self._write(extraction_root / "extractor_b" / "doc_a" / "wage_tables.jsonl")
        self._write(extraction_root / "extractor_b" / "doc_b" / "wage_tables.jsonl")

        ocr_root = src_cache / "stg_01_ocr" / self.source
        self._write(ocr_root / "ocr_one" / "doc_a" / "full.txt")
        self._write(ocr_root / "ocr_one" / "doc_a" / "page_1.txt")
        self._write(ocr_root / "ocr_one" / "doc_a" / "page_1.md")
        self._write(ocr_root / "ocr_one" / "doc_b" / "full.txt")
        self._write(ocr_root / "ocr_one" / "doc_b" / "page_2.txt")

        # Output variants are separate OCR arms and must be retained exactly
        # as named in the stage-01 tree.
        self._write(ocr_root / "ocr_two__layout" / "doc_a" / "full.txt")
        self._write(ocr_root / "ocr_two__layout" / "doc_a" / "page_1.txt")
        self._write(ocr_root / "ocr_two__layout" / "doc_a" / "page_1.md")

        # These are deliberately not final OCR page artifacts.
        self._write(ocr_root / "ocr_one" / "doc_a" / "page_2.retry")
        self._write(ocr_root / "ocr_one" / "doc_a" / "full.previous.txt")
        self._write(ocr_root / "ocr_one" / "doc_a" / "full.previous.1.txt")
        self._write(ocr_root / "ocr_one" / "doc_a" / "page_x.txt")
        self._write(ocr_root / "ocr_one" / "doc_a" / "notes.json")
        self._write(ocr_root / "ocr_one" / "doc_a" / "nested" / "page_3.txt")

        # doc_c has OCR output but no stage-02 extraction and must not move.
        self._write(ocr_root / "ocr_one" / "doc_c" / "full.txt")
        self._write(ocr_root / "ocr_one" / "doc_c" / "page_1.txt")

        # Shared detector caches are not OCR model-output directories.
        self._write(
            ocr_root
            / "_layout"
            / "PaddlePaddle_PP-DocLayoutV3_safetensors"
            / "doc_a"
            / "layout.json"
        )

    def _move_all(self, src_cache: Path, dst_cache: Path, *, dry_run: bool) -> None:
        document_dirs = move_extractions.move_source(
            self.source,
            "stg_02_extract",
            src_cache,
            dst_cache,
            dry_run,
        )
        move_extractions.move_grounding(
            self.source,
            "stg_01_ocr",
            document_dirs,
            src_cache,
            dst_cache,
            dry_run,
        )

    def test_moves_final_ocr_artifacts_for_each_matching_model_and_variant(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_cache = root / "source_cache"
            dst_cache = root / "destination_cache"
            self._build_source_cache(src_cache)

            with redirect_stdout(StringIO()):
                self._move_all(src_cache, dst_cache, dry_run=False)

            copied_root = dst_cache / "stg_01_ocr" / self.source
            copied_files = {
                path.relative_to(copied_root)
                for path in copied_root.rglob("*")
                if path.is_file()
            }

        self.assertEqual(
            copied_files,
            {
                Path("ocr_one/doc_a/full.txt"),
                Path("ocr_one/doc_a/page_1.txt"),
                Path("ocr_one/doc_a/page_1.md"),
                Path("ocr_one/doc_b/full.txt"),
                Path("ocr_one/doc_b/page_2.txt"),
                Path("ocr_two__layout/doc_a/full.txt"),
                Path("ocr_two__layout/doc_a/page_1.txt"),
                Path("ocr_two__layout/doc_a/page_1.md"),
            },
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

        self.assertEqual(dry_run_output.count("ocr_one/doc_a/full.txt"), 1)
        self.assertEqual(dry_run_output.count("ocr_two__layout/doc_a/full.txt"), 1)
        self.assertIn("ocr_one/doc_a/page_1.txt", dry_run_output)
        self.assertIn("ocr_one/doc_a/page_1.md", dry_run_output)
        self.assertNotIn("doc_c", dry_run_output)
        self.assertNotIn("page_2.retry", dry_run_output)
        self.assertNotIn("full.previous", dry_run_output)
        self.assertNotIn("page_x.txt", dry_run_output)
        self.assertNotIn("notes.json", dry_run_output)
        self.assertNotIn("layout.json", dry_run_output)


if __name__ == "__main__":
    unittest.main()
