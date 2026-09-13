import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.stg_02_extract.structure_provision import ProvisionSpec
from pipeline.stg_03_enrich import runner


def _spec() -> ProvisionSpec:
    return ProvisionSpec(
        clause_type="technology",
        prompt_description="extract",
        extraction_passes=1,
        langextract_max_workers=1,
        langextract_batch_length=1,
        max_char_buffer=100,
        enrich_max_char_buffer=20,
    )


def _record(text: str, start: int, end: int) -> dict[str, object]:
    return {
        "source": "source",
        "document_id": "doc",
        "ocr_model_name": "ocr/model",
        "model_name": "extract/model",
        "extraction_class": "technology",
        "extraction_text": text,
        "generated_extraction_text": text,
        "span_start": start,
        "span_end": end,
        "span_reliable": True,
        "grounding_status": "match_exact",
    }


def _job(root: Path, input_path: Path, output_path: Path, buffer: int = 20):
    return runner.EnrichmentJob(
        source="source",
        document_id="doc",
        extract_model_name="extract/model",
        model_name="enrich/model",
        clause_type="technology",
        input_path=input_path,
        output_path=output_path,
        ocr_root=root / "stg_01_ocr",
        enrich_max_char_buffer=buffer,
    )


class Stage03WindowTests(unittest.TestCase):
    def test_real_chonkie_window_uses_source_sentence_boundaries(self) -> None:
        source = "Far sentence. Nearby left. CLAUSE Nearby right. Far end."
        start = source.index("CLAUSE")
        window = runner.sentence_window(source, start, start + len("CLAUSE"), 30)

        self.assertEqual(window, "Nearby left. CLAUSE Nearby right. Far end.")

    def test_window_splits_remaining_budget_around_the_extraction(self) -> None:
        calls = []

        def edge(text, budget, *, take_last):
            calls.append((len(text), budget, take_last))
            if take_last:
                return max(0, len(text) - budget), len(text)
            return 0, min(len(text), budget)

        source = "abcdefghijCLAUSEklmnopqrst"
        with patch.object(runner, "_edge_from_chunks", side_effect=edge):
            window = runner.sentence_window(source, 10, 16, 20)

        self.assertEqual(window, "defghijCLAUSEklmnopq")
        self.assertEqual(calls, [(10, 7, True), (10, 7, False)])

    def test_window_reallocates_capacity_at_a_document_edge(self) -> None:
        calls = []

        def edge(text, budget, *, take_last):
            calls.append((budget, take_last))
            return (max(0, len(text) - budget), len(text)) if take_last else (0, budget)

        source = "xCLAUSE" + "r" * 30
        with patch.object(runner, "_edge_from_chunks", side_effect=edge):
            runner.sentence_window(source, 1, 7, 20)

        self.assertEqual(calls, [(1, True), (13, False)])

    def test_long_extraction_is_never_truncated(self) -> None:
        source = "prefix" + "X" * 30 + "suffix"
        with patch.object(runner, "_edge_from_chunks") as edge:
            window = runner.sentence_window(source, 6, 36, 20)
        self.assertEqual(window, "X" * 30)
        edge.assert_not_called()


class Stage03SchemaTests(unittest.TestCase):
    def test_schema_and_parser_require_context_and_beneficiary(self) -> None:
        schema = runner.enrichment_schema()
        self.assertEqual(schema["required"], ["context", "beneficiary"])
        self.assertEqual(
            schema["properties"]["beneficiary"]["enum"],
            ["workers", "employer", "unclear"],
        )
        self.assertEqual(
            runner.parse_enrichment('{"context":"  Nearby rule. ","beneficiary":"workers"}'),
            {"context": "Nearby rule.", "beneficiary": "workers"},
        )
        self.assertEqual(
            runner.parse_enrichment('{"context":null,"beneficiary":"unclear"}'),
            {"context": None, "beneficiary": "unclear"},
        )
        with self.assertRaises(ValueError):
            runner.parse_enrichment('{"context":null,"beneficiary":"union"}')

    def test_factory_uses_one_strict_structured_call(self) -> None:
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"context":"Nearby rule.","beneficiary":"employer"}'
                    )
                )])

        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=Completions())

        with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=Client)}):
            enricher = runner.make_enricher(_spec(), "model", 8123)
            result = enricher("Clause.", "Before. Clause. After.")

        self.assertEqual(result["beneficiary"], "employer")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["response_format"]["json_schema"]["strict"])
        prompt = calls[0]["messages"][1]["content"]
        self.assertIn("Clause.", prompt)
        self.assertIn("Before. Clause. After.", prompt)


class Stage03ProcessingTests(unittest.TestCase):
    def test_discovers_stage2_under_extract_model_and_keys_output_by_enricher(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = (
                root / "stg_02_extract/source/extract_model/doc/technology.jsonl"
            )
            input_path.parent.mkdir(parents=True)
            input_path.write_text("", encoding="utf-8")

            jobs = runner.discover_extractions(
                input_root=root / "stg_02_extract",
                output_root=root / "stg_03_enrich",
                ocr_root=root / "stg_01_ocr",
                extract_model_name="extract/model",
                model_name="enrich/model",
                clause_type="technology",
                enrich_max_char_buffer=5_000,
            )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].extract_model_name, "extract/model")
        self.assertEqual(
            jobs[0].output_path,
            root / "stg_03_enrich/source/enrich_model/doc/technology.jsonl",
        )

    def test_processes_new_stage2_records_and_preserves_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "aaaONEbbbTWOccc"
            ocr_path = root / "stg_01_ocr/source/ocr_model/doc/full.txt"
            ocr_path.parent.mkdir(parents=True)
            ocr_path.write_text(source, encoding="utf-8")
            input_path = root / "in.jsonl"
            records = [_record("ONE", 3, 6), _record("TWO", 9, 12)]
            input_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            output_path = root / "out.jsonl"
            seen = []

            def enrich(text, window):
                seen.append(text)
                return {"context": f"context {text}", "beneficiary": "workers"}

            with patch.object(
                runner, "sentence_window", side_effect=lambda text, start, end, size: text
            ):
                result = runner.process_enrichment_job(
                    _job(root, input_path, output_path),
                    enrich,
                    force=False,
                    request_concurrency=2,
                )
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(result.enrichment_count, 2)
        self.assertEqual(seen, ["ONE", "TWO"])
        self.assertEqual([row["extraction_text"] for row in rows], ["ONE", "TWO"])
        self.assertEqual(rows[0]["extract_model_name"], "extract/model")
        self.assertEqual(rows[0]["model_name"], "enrich/model")
        self.assertEqual(rows[0]["context"], "context ONE")

    def test_empty_input_writes_empty_output_without_ocr(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "in.jsonl"
            input_path.write_text("", encoding="utf-8")
            output_path = root / "out.jsonl"
            result = runner.process_enrichment_job(
                _job(root, input_path, output_path),
                lambda *_: {},
                force=False,
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(output_path.read_text(), "")

    def test_rejects_stale_stage2_records(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "aaaONEbbb"
            ocr_path = root / "stg_01_ocr/source/ocr_model/doc/full.txt"
            ocr_path.parent.mkdir(parents=True)
            doubted = _record("ONE", 3, 6)
            doubted.pop("generated_extraction_text")
            ocr_path.write_text(source, encoding="utf-8")
            input_path = root / "in.jsonl"
            input_path.write_text(json.dumps(doubted) + "\n")
            with self.assertRaisesRegex(ValueError, "rerun stage 2"):
                runner.process_enrichment_job(
                    _job(root, input_path, root / "out.jsonl"),
                    lambda *_: {},
                    force=False,
                )

    def test_missing_ocr_fails_without_writing_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "in.jsonl"
            input_path.write_text(json.dumps(_record("ONE", 0, 3)) + "\n")
            output_path = root / "out.jsonl"

            with self.assertRaisesRegex(FileNotFoundError, "OCR source not found"):
                runner.process_enrichment_job(
                    _job(root, input_path, output_path), lambda *_: {}, force=False
                )

            self.assertFalse(output_path.exists())

    def test_inconsistent_provenance_fails_before_enrichment_or_write(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "ONE"
            ocr_path = root / "stg_01_ocr/source/ocr_model/doc/full.txt"
            ocr_path.parent.mkdir(parents=True)
            ocr_path.write_text(source, encoding="utf-8")
            bad = _record("ONE", 0, 3)
            bad["model_name"] = "other/model"
            input_path = root / "in.jsonl"
            input_path.write_text(json.dumps(bad) + "\n")
            output_path = root / "out.jsonl"
            called = False

            def enrich(*_):
                nonlocal called
                called = True
                return {}

            with self.assertRaisesRegex(ValueError, "model_name provenance"):
                runner.process_enrichment_job(
                    _job(root, input_path, output_path), enrich, force=False
                )

            self.assertFalse(called)
            self.assertFalse(output_path.exists())

    def test_malformed_enrichment_fails_document_without_partial_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ocr_path = root / "stg_01_ocr/source/ocr_model/doc/full.txt"
            ocr_path.parent.mkdir(parents=True)
            ocr_path.write_text("ONE", encoding="utf-8")
            input_path = root / "in.jsonl"
            input_path.write_text(json.dumps(_record("ONE", 0, 3)) + "\n")
            output_path = root / "out.jsonl"

            with self.assertRaisesRegex(ValueError, "unexpected beneficiary"):
                runner.process_enrichment_job(
                    _job(root, input_path, output_path),
                    lambda *_: {"context": None, "beneficiary": "union"},
                    force=False,
                )

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
