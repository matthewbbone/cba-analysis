import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from pipeline.stg_02_extract import runner, structure_provision
from pipeline.stg_02_extract.structure_provision import ProvisionSpec, load_provision


class Stage02ProvisionTests(unittest.TestCase):
    @staticmethod
    def _definition() -> dict[str, object]:
        return {
            "provision_type": "safety_rule",
            "extraction_prompt": "Extract mandatory workplace safety rules.",
            "N_EXTRACTION_PASSES": 2,
            "LANGEXTRACT_MAX_WORKERS": 3,
            "LANGEXTRACT_BATCH_LENGTH": 4,
            "MAX_CHAR_BUFFER": 1_000,
        }

    def _load_temp(
        self,
        definition: object,
        *,
        name: str = "safety_rule",
    ) -> ProvisionSpec:
        with TemporaryDirectory() as tmp_dir:
            provisions_dir = Path(tmp_dir)
            (provisions_dir / f"{name}.yaml").write_text(
                yaml.safe_dump(definition, sort_keys=False), encoding="utf-8"
            )
            with patch.object(structure_provision, "PROVISIONS_DIR", provisions_dir):
                return load_provision(name)

    def test_loads_bundled_prompt_only_configs(self) -> None:
        # Passes and buffer size are tuned per provision, so assert each config's
        # own values rather than one shared number.
        for name, prompt_fragment, passes, char_buffer in (
            ("wage_table", "base-wage", 5, 10_000),
            ("technology", "new technology", 3, 5_000),
        ):
            with self.subTest(name=name):
                provision = load_provision(name)

                self.assertEqual(provision.provision_type, name)
                self.assertIn(prompt_fragment, provision.prompt_description.lower())
                self.assertIn(
                    f'Use "{name}" as extraction_class',
                    provision.prompt_description,
                )
                self.assertIn("context", provision.prompt_description)
                self.assertIn("null", provision.prompt_description)
                self.assertEqual(provision.extraction_passes, passes)
                self.assertEqual(provision.langextract_max_workers, 10)
                self.assertEqual(provision.langextract_batch_length, 10)
                self.assertEqual(provision.max_char_buffer, char_buffer)

    def test_loads_all_configured_values(self) -> None:
        provision = self._load_temp(self._definition())

        self.assertEqual(provision.provision_type, "safety_rule")
        self.assertTrue(
            provision.prompt_description.startswith(
                "Extract mandatory workplace safety rules."
            )
        )
        self.assertEqual(provision.extraction_passes, 2)
        self.assertEqual(provision.langextract_max_workers, 3)
        self.assertEqual(provision.langextract_batch_length, 4)
        self.assertEqual(provision.max_char_buffer, 1_000)

    def test_rejects_unsafe_name_and_type_filename_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            load_provision("../safety_rule")

        definition = {**self._definition(), "provision_type": "different_rule"}
        with self.assertRaises(ValueError):
            self._load_temp(definition)

    def test_missing_definition_raises_file_not_found(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with patch.object(
                structure_provision, "PROVISIONS_DIR", Path(tmp_dir)
            ):
                with self.assertRaises(FileNotFoundError):
                    load_provision("missing_rule")

    def test_rejects_malformed_yaml(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            provisions_dir = Path(tmp_dir)
            (provisions_dir / "safety_rule.yaml").write_text(
                "provision_type: [unterminated", encoding="utf-8"
            )
            with patch.object(structure_provision, "PROVISIONS_DIR", provisions_dir):
                with self.assertRaisesRegex(ValueError, "invalid YAML"):
                    load_provision("safety_rule")

    def test_rejects_non_mapping_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            self._load_temp(["not", "a", "mapping"])

    def test_rejects_missing_and_unknown_keys(self) -> None:
        definition = self._definition()
        for key in definition:
            with self.subTest(missing=key):
                missing_key = {k: v for k, v in definition.items() if k != key}
                with self.assertRaises(ValueError):
                    self._load_temp(missing_key)

        with self.assertRaises(ValueError):
            self._load_temp({**definition, "output_filename": "custom.jsonl"})

    def test_rejects_blank_or_wrong_typed_text_fields(self) -> None:
        invalid_values = (
            ("blank prompt", "extraction_prompt", "   "),
            ("non-string prompt", "extraction_prompt", ["safety rules"]),
            ("non-string provision type", "provision_type", 1),
        )
        for label, key, value in invalid_values:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self._load_temp({**self._definition(), key: value})

    def test_rejects_bool_zero_and_wrong_typed_integer_values(self) -> None:
        integer_keys = (
            "N_EXTRACTION_PASSES",
            "LANGEXTRACT_MAX_WORKERS",
            "LANGEXTRACT_BATCH_LENGTH",
            "MAX_CHAR_BUFFER",
        )
        for key in integer_keys:
            for value in (True, 0, -1, "1", 1.5):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        self._load_temp({**self._definition(), key: value})


class Stage02DiscoveryTests(unittest.TestCase):
    def test_discovers_full_texts_and_builds_provision_output_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            input_root = cache_root / "stg_01_ocr"
            output_root = cache_root / "stg_02_extract"
            full_path = input_root / "source_a" / "ocr_model" / "doc_1" / "full.txt"
            full_path.parent.mkdir(parents=True)
            full_path.write_text("contract text", encoding="utf-8")
            (input_root / "source_a" / "other_model" / "doc_2").mkdir(
                parents=True
            )
            (input_root / "source_b" / "ocr_model" / "doc_3").mkdir(
                parents=True
            )

            jobs = runner.discover_full_texts(
                input_root=input_root,
                output_root=output_root,
                ocr_model_name="ocr/model",
                model_name="extract/model",
                provision_type="safety_rule",
                source_filter="source_a",
            )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].document_id, "doc_1")
        self.assertEqual(jobs[0].provision_type, "safety_rule")
        self.assertEqual(jobs[0].input_path, full_path)
        self.assertEqual(
            jobs[0].output_path,
            output_root / "source_a" / "extract_model" / "doc_1" / "safety_rule.jsonl",
        )

    def test_discovers_selected_document_ids_within_source(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "input"
            output_root = root / "output"
            for source in ("source_a", "source_b"):
                for document_id in ("doc_1", "doc_2", "doc_3"):
                    full_path = (
                        input_root
                        / source
                        / "ATH-MaaS_OvisOCR2"
                        / document_id
                        / "full.txt"
                    )
                    full_path.parent.mkdir(parents=True)
                    full_path.write_text("contract text", encoding="utf-8")

            jobs = runner.discover_full_texts(
                input_root=input_root,
                output_root=output_root,
                ocr_model_name="ATH-MaaS/OvisOCR2",
                model_name="extract/model",
                provision_type="safety_rule",
                source_filter="source_a",
                document_id_filter=["doc_3", "doc_1"],
            )

        self.assertEqual([job.source for job in jobs], ["source_a", "source_a"])
        self.assertEqual([job.document_id for job in jobs], ["doc_1", "doc_3"])


class Stage02ExtractionTests(unittest.TestCase):
    def test_process_job_writes_multiple_generic_provisions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "full.txt"
            output_path = root / "out" / "safety_rule.jsonl"
            input_path.write_text(
                "visible before\n<think>hidden reasoning</think>\nvisible after",
                encoding="utf-8",
            )
            job = runner.ExtractionJob(
                source="source",
                document_id="doc",
                ocr_model_name="ocr/model",
                model_name="extract/model",
                provision_type="safety_rule",
                input_path=input_path,
                output_path=output_path,
            )
            seen_text: list[str] = []

            def extractor(
                text: str, job: runner.ExtractionJob
            ) -> list[dict[str, object]]:
                seen_text.append(text)
                return [
                    {
                        "source": job.source,
                        "document_id": job.document_id,
                        "ocr_model_name": job.ocr_model_name,
                        "model_name": job.model_name,
                        "extraction_class": job.provision_type,
                        "extraction_text": "rule one",
                        "attributes": {"context": "Applies to operators."},
                        "span_start": 0,
                        "span_end": 8,
                        "grounding_status": "match_exact",
                    },
                    {
                        "source": job.source,
                        "document_id": job.document_id,
                        "ocr_model_name": job.ocr_model_name,
                        "model_name": job.model_name,
                        "extraction_class": job.provision_type,
                        "extraction_text": "rule two",
                        "attributes": {"context": None},
                        "span_start": 9,
                        "span_end": 17,
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
        self.assertEqual(
            [row["extraction_text"] for row in rows],
            ["rule one", "rule two"],
        )
        self.assertEqual(
            [row["attributes"] for row in rows],
            [{"context": "Applies to operators."}, {"context": None}],
        )
        self.assertEqual(
            seen_text,
            ["visible before\n<think>hidden reasoning</think>\nvisible after"],
        )

    def test_process_job_skips_existing_output_without_force(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "full.txt"
            output_path = root / "safety_rule.jsonl"
            input_path.write_text("contract text", encoding="utf-8")
            output_path.write_text("already done\n", encoding="utf-8")
            job = runner.ExtractionJob(
                source="source",
                document_id="doc",
                ocr_model_name="ocr/model",
                model_name="extract/model",
                provision_type="safety_rule",
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
                    provision_type="safety_rule",
                    input_path=root / f"doc_{index}.txt",
                    output_path=root / f"doc_{index}.jsonl",
                )
                for index in range(2)
            ]

            def processor(job: runner.ExtractionJob) -> runner.ExtractionResult:
                return runner.ExtractionResult(
                    job=job, status="completed", extraction_count=1
                )

            results = runner.run_extraction_queue(
                jobs,
                concurrency=2,
                processor=processor,
                show_progress=False,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(sum(result.extraction_count for result in results), 2)


class Stage02RecordTests(unittest.TestCase):
    @staticmethod
    def _job() -> runner.ExtractionJob:
        return runner.ExtractionJob(
            source="source",
            document_id="doc",
            ocr_model_name="ocr/model",
            model_name="extract/model",
            provision_type="safety_rule",
            input_path=Path("full.txt"),
            output_path=Path("safety_rule.jsonl"),
        )

    def test_extraction_to_record_keeps_matching_class_and_trims_context(self) -> None:
        extraction = SimpleNamespace(
            extraction_class="safety_rule",
            extraction_text="Safety rule text",
            attributes={"context": "  Applies to night-shift employees.  "},
            char_interval=SimpleNamespace(start_pos=5, end_pos=21),
            alignment_status=SimpleNamespace(value="match_exact"),
        )

        record = runner.extraction_to_record(extraction, self._job())

        self.assertIsNotNone(record)
        self.assertEqual(record["extraction_class"], "safety_rule")
        self.assertEqual(record["extraction_text"], "Safety rule text")
        self.assertEqual(
            record["attributes"],
            {"context": "Applies to night-shift employees."},
        )
        self.assertEqual(record["span_start"], 5)
        self.assertEqual(record["span_end"], 21)

    def test_extraction_to_record_normalizes_invalid_context_to_null(self) -> None:
        attributes_values = [
            None,
            {},
            {"context": None},
            {"context": "  "},
            {"context": 3},
        ]

        for attributes in attributes_values:
            with self.subTest(attributes=attributes):
                extraction = SimpleNamespace(
                    extraction_class="safety_rule",
                    extraction_text="Safety rule text",
                    attributes=attributes,
                    char_interval=SimpleNamespace(start_pos=5, end_pos=21),
                    alignment_status=None,
                )

                record = runner.extraction_to_record(extraction, self._job())

                self.assertIsNotNone(record)
                self.assertEqual(record["attributes"], {"context": None})
                self.assertEqual(record["grounding_status"], "unknown")

    def test_extraction_to_record_drops_other_classes_and_ungrounded_text(self) -> None:
        other_class = SimpleNamespace(
            extraction_class="wage_table",
            extraction_text="Wage table",
            attributes={"context": None},
            char_interval=SimpleNamespace(start_pos=0, end_pos=10),
            alignment_status=None,
        )
        ungrounded = SimpleNamespace(
            extraction_class="safety_rule",
            extraction_text="Safety rule",
            attributes={"context": None},
            char_interval=None,
            alignment_status=None,
        )

        self.assertIsNone(runner.extraction_to_record(other_class, self._job()))
        self.assertIsNone(runner.extraction_to_record(ungrounded, self._job()))


class Stage02LangExtractTests(unittest.TestCase):
    def test_factory_uses_explicit_schema_and_provision_parameters(self) -> None:
        provision = ProvisionSpec(
            provision_type="safety_rule",
            prompt_description="configured prompt",
            extraction_passes=2,
            langextract_max_workers=3,
            langextract_batch_length=4,
            max_char_buffer=1_000,
        )
        job = runner.ExtractionJob(
            source="source",
            document_id="doc",
            ocr_model_name="ocr/model",
            model_name="extract/model",
            provision_type="safety_rule",
            input_path=Path("full.txt"),
            output_path=Path("safety_rule.jsonl"),
        )

        with patch(
            "langextract.extract",
            return_value=SimpleNamespace(extractions=[]),
        ) as extract:
            extractor = runner.make_langextract_extractor(
                provision=provision,
                model_name="extract/model",
                port=8123,
            )
            records = extractor("contract text", job)

        self.assertEqual(records, [])
        extract_kwargs = extract.call_args.kwargs
        self.assertEqual(extract_kwargs["prompt_description"], "configured prompt")
        self.assertEqual(extract_kwargs["config"].provider_kwargs["max_workers"], 3)
        self.assertEqual(extract_kwargs["extraction_passes"], 2)
        self.assertEqual(extract_kwargs["max_workers"], 3)
        self.assertEqual(extract_kwargs["batch_length"], 4)
        self.assertEqual(extract_kwargs["max_char_buffer"], 1_000)
        self.assertFalse(extract_kwargs["show_progress"])
        self.assertNotIn("examples", extract_kwargs)
        self.assertNotIn("temperature", extract_kwargs)
        self.assertNotIn("prompt_validation_level", extract_kwargs)

        output_schema = extract_kwargs["output_schema"]
        extraction_item = output_schema["properties"]["extractions"]["items"]
        self.assertEqual(
            extraction_item["properties"]["safety_rule"],
            {"type": "string"},
        )
        self.assertEqual(
            extraction_item["required"],
            ["safety_rule", "safety_rule_attributes"],
        )
        attributes_schema = extraction_item["properties"][
            "safety_rule_attributes"
        ]
        self.assertEqual(attributes_schema["required"], ["context"])
        self.assertFalse(attributes_schema["additionalProperties"])
        self.assertEqual(
            attributes_schema["properties"]["context"],
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        )


class Stage02ReasoningServeArgsTests(unittest.TestCase):
    def test_default_model_uses_thinking_and_gemma4_parser(self) -> None:
        self.assertEqual(
            runner.reasoning_serve_args(runner.DEFAULT_MODEL_NAME, None),
            [
                "--default-chat-template-kwargs",
                '{"enable_thinking":true}',
                "--reasoning-parser",
                "gemma4",
            ],
        )

    def test_custom_model_preserves_explicit_parser(self) -> None:
        self.assertEqual(
            runner.reasoning_serve_args("custom/model", "custom_parser"),
            [
                "--default-chat-template-kwargs",
                '{"enable_thinking":true}',
                "--reasoning-parser",
                "custom_parser",
            ],
        )

    def test_qwen3_model_selects_its_reasoning_parser(self) -> None:
        self.assertEqual(
            runner.reasoning_serve_args("Qwen/Qwen3.6-35B-A3B-FP8", None),
            [
                "--default-chat-template-kwargs",
                '{"enable_thinking":true}',
                "--reasoning-parser",
                "qwen3",
            ],
        )

    def test_custom_model_without_parser_still_enables_thinking(self) -> None:
        self.assertEqual(
            runner.reasoning_serve_args("custom/model", None),
            [
                "--default-chat-template-kwargs",
                '{"enable_thinking":true}',
            ],
        )


class Stage02CliTests(unittest.TestCase):
    def test_parse_args_requires_provision(self) -> None:
        with self.assertRaises(SystemExit):
            runner.parse_args([])

        args = runner.parse_args(["--provision", "wage_table"])

        self.assertEqual(args.provision, "wage_table")
        self.assertEqual(args.ocr_model_name, "ATH-MaaS/OvisOCR2")
        self.assertFalse(hasattr(args, "validate"))
        self.assertFalse(hasattr(args, "verify_llm"))

    def test_parse_args_accepts_source_and_multiple_document_ids(self) -> None:
        args = runner.parse_args(
            [
                "--provision",
                "wage_table",
                "--source",
                "cornell_dol",
                "--document-ids",
                "doc_a",
                "doc_b",
                "--document-id",
                "doc_c",
            ]
        )

        self.assertEqual(args.source, "cornell_dol")
        self.assertEqual(args.document_id, ["doc_a", "doc_b", "doc_c"])

    def test_parse_and_validate_device_selection(self) -> None:
        default_args = runner.parse_args(["--provision", "wage_table"])
        selected_args = runner.parse_args(
            ["--provision", "wage_table", "--device", "1,3", "--num-gpus", "2"]
        )

        runner.validate_args(default_args)
        runner.validate_args(selected_args)

        self.assertIsNone(default_args.device)
        self.assertEqual(selected_args.device, "1,3")

        mismatched_args = runner.parse_args(
            ["--provision", "wage_table", "--device", "1,3"]
        )
        with self.assertRaisesRegex(ValueError, "selects 2 GPU"):
            runner.validate_args(mismatched_args)

    def test_main_smoke_with_mocked_vllm_and_extractor(self) -> None:
        server_kwargs: list[dict[str, object]] = []
        extractor_kwargs: list[dict[str, object]] = []
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
                    del args
                    server_kwargs.append(kwargs)

                def start(self):
                    pass

                def close(self):
                    pass

            def fake_extractor_factory(*args, **kwargs):
                del args
                extractor_kwargs.append(kwargs)
                return lambda text, job: [
                    {
                        "source": job.source,
                        "document_id": job.document_id,
                        "ocr_model_name": job.ocr_model_name,
                        "model_name": job.model_name,
                        "extraction_class": job.provision_type,
                        "extraction_text": "table",
                        "attributes": {"context": None},
                        "span_start": 0,
                        "span_end": 5,
                        "grounding_status": "match_exact",
                    }
                ]

            with (
                patch.object(runner, "VLLMServer", FakeServer),
                patch.object(
                    runner, "make_langextract_extractor", fake_extractor_factory
                ),
            ):
                runner.main(
                    [
                        "--provision",
                        "wage_table",
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
                        "--device",
                        "3",
                        "--document-id",
                        "doc",
                        "--force",
                        "--no-progress",
                    ]
                )

            output_path = (
                output_root / "source" / "extract_model" / "doc" / "wage_table.jsonl"
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["extraction_text"], "table")
        self.assertEqual(rows[0]["attributes"], {"context": None})
        self.assertEqual(server_kwargs[0]["device"], "3")
        self.assertEqual(
            server_kwargs[0]["extra_serve_args"],
            [
                "--default-chat-template-kwargs",
                '{"enable_thinking":true}',
            ],
        )
        self.assertEqual(
            extractor_kwargs[0]["provision"].provision_type, "wage_table"
        )


class Stage02SpanReconcileTests(unittest.TestCase):
    @staticmethod
    def _job() -> runner.ExtractionJob:
        return runner.ExtractionJob(
            source="source",
            document_id="doc",
            ocr_model_name="ocr/model",
            model_name="extract/model",
            provision_type="safety_rule",
            input_path=Path("full.txt"),
            output_path=Path("safety_rule.jsonl"),
        )

    def test_reconcile_corrupted_span_from_source(self) -> None:
        rule = "Employees must wear supplied protective equipment in the machine shop."
        source = "preamble ... " + rule + " ... trailer"
        extraction = SimpleNamespace(
            extraction_class="safety_rule",
            extraction_text=rule,
            attributes={"context": "Applies in the machine shop."},
            char_interval=SimpleNamespace(start_pos=13, end_pos=25),
            alignment_status=SimpleNamespace(value="match_lesser"),
        )

        record = runner.extraction_to_record(
            extraction,
            self._job(),
            source_text=source,
        )

        self.assertEqual(record["span_start"], source.index(rule))
        self.assertEqual(record["span_end"], source.index(rule) + len(rule))
        self.assertTrue(record["span_reliable"])

    def test_reconcile_flags_unreliable_when_not_found(self) -> None:
        extraction = SimpleNamespace(
            extraction_class="safety_rule",
            extraction_text="a full safety rule with substantial detail " * 5
            + "that does not appear in the source text",
            attributes={"context": None},
            char_interval=SimpleNamespace(start_pos=0, end_pos=5),
            alignment_status=SimpleNamespace(value="match_fuzzy"),
        )

        record = runner.extraction_to_record(
            extraction, self._job(), source_text="unrelated source"
        )

        self.assertFalse(record["span_reliable"])

    def test_exact_match_stays_reliable(self) -> None:
        extraction = SimpleNamespace(
            extraction_class="safety_rule",
            extraction_text="Safety rule text",
            attributes={"context": None},
            char_interval=SimpleNamespace(start_pos=5, end_pos=21),
            alignment_status=SimpleNamespace(value="match_exact"),
        )

        record = runner.extraction_to_record(extraction, self._job())

        self.assertTrue(record["span_reliable"])
        self.assertEqual(record["span_start"], 5)
        self.assertEqual(record["span_end"], 21)


if __name__ == "__main__":
    unittest.main()
