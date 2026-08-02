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

            result = runner.process_extraction_job(
                job, extractor, force=False, validate_enabled=False
            )
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
                        "--no-validate",
                        "--no-verify-llm",
                    ]
                )

            output_path = output_root / "source" / "extract_model" / "doc" / "wage_tables.jsonl"
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["extraction_text"], "table")


class Stage02ValidationTests(unittest.TestCase):
    # Spans drawn from real false positives in the reviewed wage_tables.jsonl.
    INJURY_PAY = (
        "All Non-Work Related Injuries 80% of Regular Pay Rate\n"
        "Years of Company Service 0-19 Years of Service\n"
        "A Pay Rate That Is Not Less Than 85% of Regular Pay Rate\n"
        "20 Years and Over 90% of Regular Pay Rate"
    )
    HEADER_ONLY_FRAGMENT = (
        "MAIL SERVICES\n"
        "| NO. POSITION N STEP 1 STEP 2 STEP 3 STEP 4 STEP 5 STEP 6 STEP 7 STEP 8\n"
        "| J01100 Automated Mail Processor"
    )
    RANK_DIFFERENTIAL = (
        "The following pay differential shall be maintained between all ranks.\n"
        "Fire Fighter FAO 8% above Fire Fighter 4\n"
        "Lieutenant 16% above Fire Fighter 4\n"
        "Captain 16% above Fire Lieutenant"
    )
    TEACHER_GRID_NO_DOLLAR = (
        "1985-1986 Salary Schedule\n"
        "| Step | BA | BA+15 | BA+30 | BA+45 |\n"
        "| 1 | 18105 | 18973 | 19484 | 19996 |\n"
        "| 2 | 18785 | 19775 | 20286 | 20796 |"
    )
    PA_PAY_GRID = (
        "COMMONWEALTH OF PENNSYLVANIA 37 1/2 HOUR STANDARD PAY SCHEDULE\n"
        "| PAY STEP | PAY RANGE 1 | PAY RANGE 2 |\n"
        "| Annual | 26,873 | 30,315 |\n"
        "| Hourly | 13.15 | 14.85 |"
    )
    DOLLAR_WAGE_TABLE = (
        "Wage Schedule\n"
        "| Occupation | Start | After 1 Year |\n"
        "| Laborer | $15.00 | $16.25 |"
    )

    def test_rejection_reason_drops_percentage_pay_policy(self) -> None:
        self.assertEqual(
            runner.validate.rejection_reason(self.INJURY_PAY),
            "percentage_only_pay_policy",
        )
        self.assertEqual(
            runner.validate.rejection_reason(self.RANK_DIFFERENTIAL),
            "percentage_only_pay_policy",
        )

    def test_rejection_reason_drops_header_only_fragment(self) -> None:
        self.assertEqual(
            runner.validate.rejection_reason(self.HEADER_ONLY_FRAGMENT),
            "header_only_fragment",
        )

    def test_rejection_reason_keeps_tables_without_dollar_signs(self) -> None:
        self.assertIsNone(runner.validate.rejection_reason(self.TEACHER_GRID_NO_DOLLAR))
        self.assertIsNone(runner.validate.rejection_reason(self.PA_PAY_GRID))

    def test_rejection_reason_keeps_dollar_wage_table(self) -> None:
        self.assertIsNone(runner.validate.rejection_reason(self.DOLLAR_WAGE_TABLE))

    def test_has_pay_amount_ignores_calendar_years(self) -> None:
        prose_with_year = "This Agreement is effective January 1, 2007 for all members."
        self.assertFalse(runner.validate.has_pay_amount(prose_with_year))

    def _record(self, text: str) -> dict[str, object]:
        return {"extraction_text": text}

    def test_apply_validation_filters_and_counts(self) -> None:
        records = [
            self._record(self.DOLLAR_WAGE_TABLE),
            self._record(self.INJURY_PAY),
            self._record(self.HEADER_ONLY_FRAGMENT),
            self._record(self.TEACHER_GRID_NO_DOLLAR),
        ]

        kept, dropped = runner.apply_validation(
            records, validate_enabled=True, verify_client=None, verify_model_name=None
        )

        self.assertEqual(dropped, 2)
        self.assertEqual(
            [row["extraction_text"] for row in kept],
            [self.DOLLAR_WAGE_TABLE, self.TEACHER_GRID_NO_DOLLAR],
        )

    def test_apply_validation_disabled_keeps_everything(self) -> None:
        records = [self._record(self.INJURY_PAY)]

        kept, dropped = runner.apply_validation(
            records, validate_enabled=False, verify_client=None, verify_model_name=None
        )

        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 1)


class Stage02VerifyLlmTests(unittest.TestCase):
    def _client(self, answer: str):
        message = SimpleNamespace(content=answer)
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        completions = SimpleNamespace(create=lambda **kwargs: response)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def test_verify_keeps_on_yes(self) -> None:
        from pipeline.stg_02_extract import validate

        self.assertTrue(
            validate.verify_is_base_wage_table("table", self._client("YES"), "model")
        )

    def test_verify_drops_on_no(self) -> None:
        from pipeline.stg_02_extract import validate

        self.assertFalse(
            validate.verify_is_base_wage_table(
                "coaching stipends", self._client("NO, this is a stipend."), "model"
            )
        )

    def test_verify_fails_open_on_client_error(self) -> None:
        from pipeline.stg_02_extract import validate

        def boom(**kwargs):
            raise RuntimeError("connection refused")

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=boom))
        )
        self.assertTrue(validate.verify_is_base_wage_table("table", client, "model"))

    def test_apply_validation_runs_verify_after_deterministic(self) -> None:
        # Deterministic keeps this (it has $ amounts) but the LLM says NO.
        records = [{"extraction_text": "Coaching\n| Coach | $2,040 | $2,130 |"}]

        kept, dropped = runner.apply_validation(
            records,
            validate_enabled=True,
            verify_client=self._client("NO"),
            verify_model_name="model",
        )

        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [])


class Stage02SpanReconcileTests(unittest.TestCase):
    def _job(self) -> "runner.ExtractionJob":
        return runner.ExtractionJob(
            source="source",
            document_id="doc",
            ocr_model_name="ocr/model",
            model_name="extract/model",
            input_path=Path("full.txt"),
            output_path=Path("wage_tables.jsonl"),
        )

    def test_reconcile_corrupted_span_from_source(self) -> None:
        table = "COMMONWEALTH OF PENNSYLVANIA PAY SCHEDULE with lots of rows here"
        source = "preamble ... " + table + " ... trailer"
        extraction = SimpleNamespace(
            extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
            extraction_text=table,
            attributes={"dimensions": ["experience"]},
            # grounding anchored only the 28-char title prefix
            char_interval=SimpleNamespace(start_pos=13, end_pos=41),
            alignment_status=SimpleNamespace(value="match_lesser"),
        )

        record = runner.extraction_to_record(extraction, self._job(), source_text=source)

        self.assertEqual(record["span_start"], source.index(table))
        self.assertEqual(record["span_end"], source.index(table) + len(table))
        self.assertTrue(record["span_reliable"])

    def test_reconcile_flags_unreliable_when_not_found(self) -> None:
        extraction = SimpleNamespace(
            extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
            extraction_text="a full wage table with many rows " * 5
            + "that does not appear in the source text",
            attributes={"dimensions": ["experience"]},
            char_interval=SimpleNamespace(start_pos=0, end_pos=5),
            alignment_status=SimpleNamespace(value="match_fuzzy"),
        )

        record = runner.extraction_to_record(
            extraction, self._job(), source_text="unrelated source"
        )

        self.assertFalse(record["span_reliable"])

    def test_exact_match_stays_reliable(self) -> None:
        extraction = SimpleNamespace(
            extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
            extraction_text="Wage table text",
            attributes={"dimensions": ["occupation"]},
            char_interval=SimpleNamespace(start_pos=5, end_pos=20),
            alignment_status=SimpleNamespace(value="match_exact"),
        )

        record = runner.extraction_to_record(extraction, self._job())

        self.assertTrue(record["span_reliable"])
        self.assertEqual(record["span_start"], 5)
        self.assertEqual(record["span_end"], 20)


if __name__ == "__main__":
    unittest.main()
