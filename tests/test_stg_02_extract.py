import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.stg_02_extract import runner
from pipeline.stg_02_extract.structures import (
    WAGE_TABLE_DIMENSIONS,
    WAGE_TABLE_EXTRACTION_CLASS,
    WAGE_TABLE_TASK,
)


class Stage02StructureTests(unittest.TestCase):
    def test_wage_table_task_has_expected_public_shape(self) -> None:
        task = WAGE_TABLE_TASK.as_json_object()

        self.assertEqual(task["extraction_class"], "wage_table")
        self.assertEqual(task["attributes"], {"dimensions": list(WAGE_TABLE_DIMENSIONS)})
        self.assertIn("prompt", task)
        self.assertNotIn("examples", task)


class Stage02DiscoveryTests(unittest.TestCase):
    def test_discovers_full_texts_and_builds_output_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            input_root = cache_root / "stg_01_ocr"
            output_root = cache_root / "stg_02_extract"
            full_path = (
                input_root
                / "source_a"
                / "ocr_model"
                / "doc_1"
                / "full.txt"
            )
            full_path.parent.mkdir(parents=True)
            full_path.write_text("contract text", encoding="utf-8")
            (
                input_root
                / "source_a"
                / "other_model"
                / "doc_2"
            ).mkdir(parents=True)
            (
                input_root
                / "source_b"
                / "ocr_model"
                / "doc_3"
            ).mkdir(parents=True)

            jobs = runner.discover_full_texts(
                input_root=input_root,
                output_root=output_root,
                ocr_model_name="ocr/model",
                model_name="extract/model",
                source_filter="source_a",
            )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].document_id, "doc_1")
        self.assertEqual(jobs[0].input_path, full_path)
        self.assertEqual(
            jobs[0].output_path,
            output_root / "source_a" / "extract_model" / "doc_1" / "wage_tables.jsonl",
        )


class Stage02ExtractionTests(unittest.TestCase):
    def test_process_job_writes_multiple_grounded_wage_tables(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "full.txt"
            output_path = root / "out" / "wage_tables.jsonl"
            input_path.write_text(
                "visible before\n<think>hidden reasoning</think>\nvisible after",
                encoding="utf-8",
            )
            job = runner.ExtractionJob(
                source="source",
                document_id="doc",
                ocr_model_name="ocr/model",
                model_name="extract/model",
                input_path=input_path,
                output_path=output_path,
            )
            seen_text: list[str] = []

            def extractor(text: str, job: runner.ExtractionJob) -> list[dict[str, object]]:
                seen_text.append(text)
                return [
                    {
                        "source": job.source,
                        "document_id": job.document_id,
                        "ocr_model_name": job.ocr_model_name,
                        "model_name": job.model_name,
                        "extraction_class": "wage_table",
                        "extraction_text": "table one",
                        "attributes": {"dimensions": ["occupation"]},
                        "span_start": 0,
                        "span_end": 9,
                        "grounding_status": "match_exact",
                    },
                    {
                        "source": job.source,
                        "document_id": job.document_id,
                        "ocr_model_name": job.ocr_model_name,
                        "model_name": job.model_name,
                        "extraction_class": "wage_table",
                        "extraction_text": "table two",
                        "attributes": {"dimensions": ["experience", "education"]},
                        "span_start": 10,
                        "span_end": 19,
                        "grounding_status": "match_exact",
                    },
                ]

            result = runner.process_extraction_job(job, extractor, force=False)
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.extraction_count, 2)
        self.assertEqual([row["extraction_text"] for row in rows], ["table one", "table two"])
        self.assertNotIn("<think>", seen_text[0])
        self.assertNotIn("hidden reasoning", seen_text[0])

    def test_process_job_skips_existing_output_without_force(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "full.txt"
            output_path = root / "wage_tables.jsonl"
            input_path.write_text("contract text", encoding="utf-8")
            output_path.write_text("already done\n", encoding="utf-8")
            job = runner.ExtractionJob(
                source="source",
                document_id="doc",
                ocr_model_name="ocr/model",
                model_name="extract/model",
                input_path=input_path,
                output_path=output_path,
            )

            result = runner.process_extraction_job(
                job,
                extractor=lambda text, job: self.fail("extractor should not run"),
                force=False,
            )

        self.assertEqual(result.status, "skipped")

    def test_run_queue_collects_results(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            jobs = [
                runner.ExtractionJob(
                    source="source",
                    document_id=f"doc_{index}",
                    ocr_model_name="ocr/model",
                    model_name="extract/model",
                    input_path=root / f"doc_{index}.txt",
                    output_path=root / f"doc_{index}.jsonl",
                )
                for index in range(2)
            ]

            def processor(job: runner.ExtractionJob) -> runner.ExtractionResult:
                return runner.ExtractionResult(job=job, status="completed", extraction_count=1)

            results = runner.run_extraction_queue(
                jobs,
                concurrency=2,
                processor=processor,
                show_progress=False,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(sum(result.extraction_count for result in results), 2)


class Stage02RecordTests(unittest.TestCase):
    def test_extraction_to_record_keeps_grounded_wage_tables_only(self) -> None:
        job = runner.ExtractionJob(
            source="source",
            document_id="doc",
            ocr_model_name="ocr/model",
            model_name="extract/model",
            input_path=Path("full.txt"),
            output_path=Path("wage_tables.jsonl"),
        )
        extraction = SimpleNamespace(
            extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
            extraction_text="Wage table text",
            attributes={"dimensions": ["education", "unknown", "occupation"]},
            char_interval=SimpleNamespace(start_pos=5, end_pos=20),
            alignment_status=SimpleNamespace(value="match_exact"),
        )

        record = runner.extraction_to_record(extraction, job)

        self.assertEqual(record["extraction_text"], "Wage table text")
        self.assertEqual(record["attributes"], {"dimensions": ["occupation", "education"]})
        self.assertEqual(record["span_start"], 5)
        self.assertEqual(record["span_end"], 20)

    def test_extraction_to_record_uses_unknown_for_missing_alignment_status(self) -> None:
        job = runner.ExtractionJob(
            source="source",
            document_id="doc",
            ocr_model_name="ocr/model",
            model_name="extract/model",
            input_path=Path("full.txt"),
            output_path=Path("wage_tables.jsonl"),
        )
        extraction = SimpleNamespace(
            extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
            extraction_text="Wage table text",
            attributes={"dimensions": ["occupation"]},
            char_interval=SimpleNamespace(start_pos=5, end_pos=20),
            alignment_status=None,
        )

        record = runner.extraction_to_record(extraction, job)

        self.assertEqual(record["grounding_status"], "unknown")

    def test_extraction_to_record_drops_ungrounded_extractions(self) -> None:
        job = runner.ExtractionJob(
            source="source",
            document_id="doc",
            ocr_model_name="ocr/model",
            model_name="extract/model",
            input_path=Path("full.txt"),
            output_path=Path("wage_tables.jsonl"),
        )
        extraction = SimpleNamespace(
            extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
            extraction_text="Wage table text",
            attributes={"dimensions": ["occupation"]},
            char_interval=None,
            alignment_status=None,
        )

        self.assertIsNone(runner.extraction_to_record(extraction, job))


class Stage02CliTests(unittest.TestCase):
    def test_parse_args_exposes_langextract_quality_knobs(self) -> None:
        args = runner.parse_args(
            [
                "--extraction-passes",
                "3",
                "--langextract-max-workers",
                "4",
                "--langextract-batch-length",
                "5",
            ]
        )

        self.assertEqual(args.extraction_passes, 3)
        self.assertEqual(args.langextract_max_workers, 4)
        self.assertEqual(args.langextract_batch_length, 5)

    def test_main_smoke_with_mocked_vllm_and_extractor(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            full_path = (
                cache_root
                / "stg_01_ocr"
                / "source"
                / "ocr_model"
                / "doc"
                / "full.txt"
            )
            full_path.parent.mkdir(parents=True)
            full_path.write_text("contract text", encoding="utf-8")
            output_root = cache_root / "stg_02_extract"

            class FakeServer:
                def __init__(self, *args, **kwargs):
                    self.started = False

                def start(self):
                    self.started = True

                def close(self):
                    pass

            def fake_extractor_factory(*args, **kwargs):
                return lambda text, job: [
                    {
                        "source": job.source,
                        "document_id": job.document_id,
                        "ocr_model_name": job.ocr_model_name,
                        "model_name": job.model_name,
                        "extraction_class": "wage_table",
                        "extraction_text": "table",
                        "attributes": {"dimensions": ["occupation"]},
                        "span_start": 0,
                        "span_end": 5,
                        "grounding_status": "match_exact",
                    }
                ]

            with (
                patch.object(runner, "VLLMServer", FakeServer),
                patch.object(runner, "make_langextract_extractor", fake_extractor_factory),
            ):
                runner.main(
                    [
                        "--input-root",
                        str(cache_root / "stg_01_ocr"),
                        "--output-root",
                        str(output_root),
                        "--ocr-model-name",
                        "ocr/model",
                        "--model-name",
                        "extract/model",
                        "--source",
                        "source",
                        "--document-id",
                        "doc",
                        "--force",
                        "--no-progress",
                    ]
                )

            output_path = output_root / "source" / "extract_model" / "doc" / "wage_tables.jsonl"
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["extraction_text"], "table")


if __name__ == "__main__":
    unittest.main()
