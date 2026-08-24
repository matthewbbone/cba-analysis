import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from pipeline.stg_02_extract import structure_provision
from pipeline.stg_02_extract.structure_provision import (
    ProvisionSpec,
    SubtypeNode,
    load_provision,
)
from pipeline.stg_03_classify import runner


# Two tiers: "retraining" has children, "reassignment" is a leaf, so a depth-2
# run exercises both descending and bottoming out early.
TAXONOMY: dict[str, object] = {
    "retraining": {
        "description": "Training on new equipment.",
        "subtypes": {
            "tuition_support": {"description": "Employer funds outside courses."},
            "on_the_job": {"description": "Training delivered during work hours."},
        },
    },
    "reassignment": {
        "description": "Transfer or displacement of affected employees.",
    },
}
# Flat label-to-description choices as the model sees them at level 1.
SUBTYPES = {
    "retraining": "Training on new equipment.",
    "reassignment": "Transfer or displacement of affected employees.",
}


def _nodes(taxonomy: dict[str, object]) -> dict[str, SubtypeNode]:
    return {
        label: SubtypeNode(
            label=label,
            description=node["description"],
            children=_nodes(node.get("subtypes", {})),
        )
        for label, node in taxonomy.items()
    }


def _spec(taxonomy: dict[str, object] | None = TAXONOMY) -> ProvisionSpec:
    return ProvisionSpec(
        clause_type="technology",
        prompt_description="Extract technology clauses.",
        extraction_passes=1,
        langextract_max_workers=1,
        langextract_batch_length=1,
        max_char_buffer=100,
        subtype_taxonomy=None if taxonomy is None else _nodes(taxonomy),
    )


def _job(
    input_path: Path, output_path: Path, taxonomy_depth: int = 1
) -> runner.ClassificationJob:
    return runner.ClassificationJob(
        source="source",
        document_id="doc",
        extract_model_name="extract/model",
        model_name="classify/model",
        clause_type="technology",
        input_path=input_path,
        output_path=output_path,
        taxonomy_depth=taxonomy_depth,
    )


def _extraction(
    text: str, context: str | None, beneficiary: str = "worker"
) -> dict[str, object]:
    return {
        "source": "source",
        "document_id": "doc",
        "ocr_model_name": "ocr/model",
        "model_name": "extract/model",
        "extraction_class": "technology",
        "extraction_text": text,
        "attributes": {"context": context, "beneficiary": beneficiary},
        "span_start": 10,
        "span_end": 10 + len(text),
        "span_reliable": True,
        "grounding_status": "match_exact",
    }


class Stage03SubtypeConfigTests(unittest.TestCase):
    @staticmethod
    def _definition(taxonomy: object) -> dict[str, object]:
        return {
            "clause_type": "safety_rule",
            "clause_description": "Mandatory workplace safety rules.",
            "subtype_taxonomy": taxonomy,
            "N_EXTRACTION_PASSES": 2,
            "LANGEXTRACT_MAX_WORKERS": 3,
            "LANGEXTRACT_BATCH_LENGTH": 4,
            "MAX_CHAR_BUFFER": 1_000,
        }

    def _load_temp(self, definition: object) -> ProvisionSpec:
        with TemporaryDirectory() as tmp_dir:
            provisions_dir = Path(tmp_dir)
            (provisions_dir / "safety_rule.yaml").write_text(
                yaml.safe_dump(definition, sort_keys=False), encoding="utf-8"
            )
            with patch.object(structure_provision, "PROVISIONS_DIR", provisions_dir):
                return load_provision("safety_rule")

    def test_bundled_technology_config_declares_a_two_tier_taxonomy(self) -> None:
        provision = load_provision("technology")

        self.assertEqual(
            list(provision.subtype_taxonomy),
            ["preemptive_rights", "implementation", "workforce_management"],
        )
        self.assertEqual(structure_provision.taxonomy_depth(provision.subtype_taxonomy), 2)
        for node in provision.subtype_taxonomy.values():
            self.assertTrue(node.description.strip())
            self.assertTrue(node.children)
            for child in node.children.values():
                self.assertTrue(child.description.strip())
                self.assertFalse(child.children)

    def test_subtypes_are_optional(self) -> None:
        self.assertIsNone(load_provision("wage_table").subtype_taxonomy)

    def test_loads_taxonomy_in_declared_order_with_nesting(self) -> None:
        provision = self._load_temp(self._definition(TAXONOMY))

        self.assertEqual(list(provision.subtype_taxonomy), list(TAXONOMY))
        retraining = provision.subtype_taxonomy["retraining"]
        self.assertEqual(retraining.description, "Training on new equipment.")
        self.assertEqual(
            list(retraining.children), ["tuition_support", "on_the_job"]
        )
        # A node with no "subtypes" block is a leaf, not an error.
        self.assertEqual(provision.subtype_taxonomy["reassignment"].children, {})

    def test_taxonomy_depth_and_labels_at_level(self) -> None:
        taxonomy = self._load_temp(self._definition(TAXONOMY)).subtype_taxonomy

        self.assertEqual(structure_provision.taxonomy_depth(taxonomy), 2)
        self.assertEqual(
            list(structure_provision.labels_at_level(taxonomy, 1)),
            ["retraining", "reassignment"],
        )
        self.assertEqual(
            list(structure_provision.labels_at_level(taxonomy, 2)),
            ["tuition_support", "on_the_job"],
        )

    def test_children_of_walks_the_tree_and_stops_off_it(self) -> None:
        taxonomy = self._load_temp(self._definition(TAXONOMY)).subtype_taxonomy

        self.assertEqual(
            list(structure_provision.children_of(taxonomy, ["retraining"])),
            ["tuition_support", "on_the_job"],
        )
        # A leaf and the injected "other" both terminate a cascade.
        self.assertEqual(structure_provision.children_of(taxonomy, ["reassignment"]), {})
        self.assertEqual(structure_provision.children_of(taxonomy, ["other"]), {})

    def test_rejects_malformed_taxonomy(self) -> None:
        leaf = {"description": "Else."}
        invalid_values = (
            ("not a mapping", ["retraining", "reassignment"]),
            ("too few labels", {"retraining": leaf}),
            ("unsafe label", {"Retraining": leaf, "reassignment": leaf}),
            ("node not a mapping", {"retraining": "Training.", "reassignment": leaf}),
            ("missing description", {"retraining": {}, "reassignment": leaf}),
            ("blank description", {"retraining": {"description": "  "}, "reassignment": leaf}),
            (
                "non-string description",
                {"retraining": {"description": 1}, "reassignment": leaf},
            ),
            # A misspelled key would otherwise silently drop the description.
            (
                "unknown node key",
                {"retraining": {"descriptioin": "Training."}, "reassignment": leaf},
            ),
            # Analysis keys columns by bare label, so a repeat merges two classes.
            (
                "duplicate label across branches",
                {
                    "retraining": {"description": "T.", "subtypes": {
                        "reassignment": leaf, "on_the_job": leaf,
                    }},
                    "reassignment": leaf,
                },
            ),
            # Stage 3 injects "other" itself.
            ("reserved label", {"other": leaf, "reassignment": leaf}),
            (
                "too few children",
                {
                    "retraining": {"description": "T.", "subtypes": {"on_the_job": leaf}},
                    "reassignment": leaf,
                },
            ),
        )
        for label, value in invalid_values:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self._load_temp(self._definition(value))


class Stage03SchemaAndPromptTests(unittest.TestCase):
    def test_schema_exposes_only_the_subtype_enum(self) -> None:
        schema = runner.subtype_schema(list(SUBTYPES))

        self.assertEqual(list(schema["properties"]), ["subtype"])
        self.assertEqual(schema["required"], ["subtype"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["subtype"]["enum"],
            ["retraining", "reassignment"],
        )

    def test_subtype_schema_covers_refinement_passes_too(self) -> None:
        schema = runner.subtype_schema(["tuition_support", "on_the_job", "other"])

        self.assertEqual(list(schema["properties"]), ["subtype"])
        self.assertEqual(schema["required"], ["subtype"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["subtype"]["enum"],
            ["tuition_support", "on_the_job", "other"],
        )

    def test_level_options_appends_other_after_the_declared_labels(self) -> None:
        options = runner.level_options(_spec().subtype_taxonomy)

        self.assertEqual(
            list(options), ["retraining", "reassignment", runner.OTHER_LABEL]
        )
        self.assertEqual(options["retraining"], "Training on new equipment.")
        self.assertTrue(options[runner.OTHER_LABEL].strip())

    def test_refine_prompt_names_the_parent_and_lists_only_its_children(self) -> None:
        taxonomy = _spec().subtype_taxonomy
        children = runner.level_options(taxonomy["retraining"].children)
        prompt = runner.build_refine_prompt(
            clause_type="technology",
            parent=taxonomy["retraining"],
            subtypes=children,
            extraction_text="  The Employer shall fund tuition.  ",
            context="Applies to the maintenance unit.",
        )

        self.assertIn('classified as "retraining": Training on new equipment.', prompt)
        for label in children:
            self.assertIn(f"- {label}: ", prompt)
        self.assertNotIn("- reassignment: ", prompt)
        # A refinement pass settles the subtype only; beneficiary is already fixed.
        self.assertNotIn("beneficiary", prompt)
        self.assertIn("The Employer shall fund tuition.", prompt)
        self.assertIn("Applies to the maintenance unit.", prompt)

    def test_refine_prompt_omits_context_section_when_absent(self) -> None:
        taxonomy = _spec().subtype_taxonomy
        for context in (None, "", "   "):
            with self.subTest(context=context):
                prompt = runner.build_refine_prompt(
                    clause_type="technology",
                    parent=taxonomy["retraining"],
                    subtypes=runner.level_options(taxonomy["retraining"].children),
                    extraction_text="Clause text.",
                    context=context,
                )

                self.assertNotIn("Context from elsewhere", prompt)

    def test_prompt_lists_every_option_and_includes_context(self) -> None:
        prompt = runner.build_user_prompt(
            clause_type="technology",
            subtypes=SUBTYPES,
            extraction_text="  The Employer shall train affected employees.  ",
            context="Applies to the maintenance unit.",
        )

        for label in SUBTYPES:
            self.assertIn(f"- {label}: ", prompt)
        # Beneficiary is settled during extraction, so it is never asked here.
        self.assertNotIn("beneficiary", prompt)
        self.assertIn("technology provision", prompt)
        self.assertIn("The Employer shall train affected employees.", prompt)
        self.assertIn("Applies to the maintenance unit.", prompt)

    def test_prompt_omits_context_section_when_absent(self) -> None:
        for context in (None, "", "   "):
            with self.subTest(context=context):
                prompt = runner.build_user_prompt(
                    clause_type="technology",
                    subtypes=SUBTYPES,
                    extraction_text="Clause text.",
                    context=context,
                )

                self.assertNotIn("Context from elsewhere", prompt)

    def test_extraction_beneficiary_falls_back_for_unusable_values(self) -> None:
        self.assertEqual(
            runner.extraction_beneficiary(_extraction("t", None, "employer")),
            "employer",
        )
        # Records written before stage 2 assigned beneficiaries, and anything
        # outside the shared vocabulary, fall back rather than raising.
        for label, record in (
            ("no attributes", {}),
            ("attributes not a mapping", {"attributes": "worker"}),
            ("absent", {"attributes": {"context": None}}),
            ("out of enum", {"attributes": {"beneficiary": "workers"}}),
            ("wrong type", {"attributes": {"beneficiary": 3}}),
        ):
            with self.subTest(label=label):
                self.assertEqual(runner.extraction_beneficiary(record), "unclear")

    def test_parse_subtype_ignores_beneficiary_and_rejects_bad_values(self) -> None:
        names = ["tuition_support", "on_the_job", "other"]
        self.assertEqual(
            runner.parse_subtype('{"subtype": "on_the_job"}', names), "on_the_job"
        )

        for label, content in (
            ("empty", ""),
            ("none", None),
            ("not an object", "[1, 2]"),
            ("bad subtype", '{"subtype": "retraining"}'),
            ("missing subtype", "{}"),
        ):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    runner.parse_subtype(content, names)


class Stage03DiscoveryTests(unittest.TestCase):
    def test_discovers_extractions_and_builds_classify_output_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            input_root = cache_root / "stg_02_extract"
            output_root = cache_root / "stg_03_classify"
            input_path = (
                input_root / "source_a" / "extract_model" / "doc_1" / "technology.jsonl"
            )
            input_path.parent.mkdir(parents=True)
            input_path.write_text("", encoding="utf-8")
            # Same document, different extraction model: not the requested input.
            (input_root / "source_a" / "other_model" / "doc_1").mkdir(parents=True)
            # Right model, wrong provision type.
            wage_path = (
                input_root / "source_a" / "extract_model" / "doc_2" / "wage_table.jsonl"
            )
            wage_path.parent.mkdir(parents=True)
            wage_path.write_text("", encoding="utf-8")
            (input_root / "source_b" / "extract_model" / "doc_3").mkdir(parents=True)

            jobs = runner.discover_extractions(
                input_root=input_root,
                output_root=output_root,
                extract_model_name="extract/model",
                model_name="classify/model",
                clause_type="technology",
                source_filter="source_a",
            )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].document_id, "doc_1")
        self.assertEqual(jobs[0].input_path, input_path)
        self.assertEqual(
            jobs[0].output_path,
            output_root / "source_a" / "classify_model" / "doc_1" / "technology.jsonl",
        )

    def test_discovers_selected_document_ids_within_source(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "input"
            for source in ("source_a", "source_b"):
                for document_id in ("doc_1", "doc_2", "doc_3"):
                    input_path = (
                        input_root
                        / source
                        / "google_gemma-4-31B-it"
                        / document_id
                        / "technology.jsonl"
                    )
                    input_path.parent.mkdir(parents=True)
                    input_path.write_text("", encoding="utf-8")

            jobs = runner.discover_extractions(
                input_root=input_root,
                output_root=root / "output",
                extract_model_name="google/gemma-4-31B-it",
                model_name="classify/model",
                clause_type="technology",
                source_filter="source_a",
                document_id_filter=["doc_3", "doc_1"],
            )

        self.assertEqual([job.source for job in jobs], ["source_a", "source_a"])
        self.assertEqual([job.document_id for job in jobs], ["doc_1", "doc_3"])

    def test_missing_input_root_yields_no_jobs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            jobs = runner.discover_extractions(
                input_root=Path(tmp_dir) / "absent",
                output_root=Path(tmp_dir) / "output",
                extract_model_name="extract/model",
                model_name="classify/model",
                clause_type="technology",
            )

        self.assertEqual(jobs, [])


class Stage03ClassificationTests(unittest.TestCase):
    def test_process_job_writes_one_record_per_extraction(self) -> None:
        seen: list[tuple[str, str | None]] = []

        def classifier(extraction_text: str, context: str | None) -> dict[str, str]:
            seen.append((extraction_text, context))
            return {"subtype_1": "retraining"}

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "technology.jsonl"
            output_path = root / "out" / "technology.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        _extraction("clause one", "Applies to operators."),
                        _extraction("clause two", None),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            job = _job(input_path, output_path)

            result = runner.process_classification_job(
                job, classifier, force=False, request_concurrency=1
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.classification_count, 2)
        self.assertEqual(
            seen,
            [("clause one", "Applies to operators."), ("clause two", None)],
        )
        self.assertEqual(
            [row["extraction_text"] for row in rows], ["clause one", "clause two"]
        )
        self.assertEqual(
            rows[0],
            {
                "source": "source",
                "document_id": "doc",
                "extract_model_name": "extract/model",
                "model_name": "classify/model",
                "extraction_class": "technology",
                "extraction_text": "clause one",
                "span_start": 10,
                "span_end": 20,
                "beneficiary": "worker",
                "taxonomy_depth": 1,
                "subtype_1": "retraining",
            },
        )

    def test_record_carries_one_column_per_level_including_unreached_ones(self) -> None:
        def classifier(extraction_text: str, context: str | None) -> dict[str, str | None]:
            del context
            # "reassignment" is a leaf, so its cascade stops before level 2.
            if extraction_text == "clause one":
                return {
                    "subtype_1": "retraining",
                    "subtype_2": "tuition_support",
                }
            return {
                "subtype_1": "reassignment",
                "subtype_2": None,
            }

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "technology.jsonl"
            output_path = root / "out" / "technology.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        _extraction("clause one", None),
                        _extraction("clause two", None),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            runner.process_classification_job(
                _job(input_path, output_path, taxonomy_depth=2),
                classifier,
                force=False,
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        # Both rows carry the same keys, whichever depth their branch reached.
        self.assertEqual(sorted(rows[0]), sorted(rows[1]))
        self.assertEqual(
            [row["subtype_1"] for row in rows], ["retraining", "reassignment"]
        )
        self.assertEqual([row["subtype_2"] for row in rows], ["tuition_support", None])
        self.assertEqual([row["taxonomy_depth"] for row in rows], [2, 2])
        self.assertNotIn("subtype", rows[0])

    def test_process_job_preserves_order_under_request_concurrency(self) -> None:
        def classifier(extraction_text: str, context: str | None) -> dict[str, str]:
            del context
            return {"subtype_1": "other"}

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "technology.jsonl"
            output_path = root / "out" / "technology.jsonl"
            texts = [f"clause {index}" for index in range(6)]
            input_path.write_text(
                "\n".join(json.dumps(_extraction(text, None)) for text in texts) + "\n",
                encoding="utf-8",
            )

            runner.process_classification_job(
                _job(input_path, output_path),
                classifier,
                force=False,
                request_concurrency=4,
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["extraction_text"] for row in rows], texts)

    def test_existing_output_is_skipped_unless_forced(self) -> None:
        calls: list[str] = []

        def classifier(extraction_text: str, context: str | None) -> dict[str, str]:
            del context
            calls.append(extraction_text)
            return {"subtype_1": "other"}

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "technology.jsonl"
            output_path = root / "out" / "technology.jsonl"
            input_path.write_text(
                json.dumps(_extraction("clause", None)) + "\n", encoding="utf-8"
            )
            output_path.parent.mkdir(parents=True)
            output_path.write_text("stale\n", encoding="utf-8")
            job = _job(input_path, output_path)

            skipped = runner.process_classification_job(job, classifier, force=False)
            self.assertEqual(skipped.status, "skipped")
            self.assertEqual(calls, [])
            self.assertEqual(output_path.read_text(encoding="utf-8"), "stale\n")

            forced = runner.process_classification_job(job, classifier, force=True)
            rows = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(forced.status, "completed")
        self.assertEqual(calls, ["clause"])
        self.assertEqual(len(rows), 1)

    def test_failed_record_fails_document_without_writing_output(self) -> None:
        def classifier(extraction_text: str, context: str | None) -> dict[str, str]:
            del context
            if extraction_text == "clause two":
                raise ValueError("unexpected subtype value: 'nonsense'")
            return {"subtype_1": "retraining"}

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "technology.jsonl"
            output_path = root / "out" / "technology.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        _extraction("clause one", None),
                        _extraction("clause two", None),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            job = _job(input_path, output_path)

            results = runner.run_classification_queue(
                jobs=[job],
                concurrency=1,
                processor=lambda pending: runner.process_classification_job(
                    pending, classifier, force=False
                ),
            )

            self.assertFalse(output_path.exists())

        self.assertEqual([result.status for result in results], ["failed"])
        self.assertIn("unexpected subtype", results[0].error)


def _fake_openai(responses: list[str], calls: list[dict[str, object]]):
    """An OpenAI stub that replays ``responses`` and records each call."""

    from types import SimpleNamespace

    pending = list(responses)

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if not pending:
                raise AssertionError("the classifier made more calls than expected")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=pending.pop(0)))
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append({"client_kwargs": kwargs})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    return type("Module", (), {"OpenAI": FakeClient})


def _classify(responses: list[str], depth: int = 1, provision=None):
    calls: list[dict[str, object]] = []
    module = _fake_openai(responses, calls)
    with patch.dict("sys.modules", {"openai": module}):
        classifier = runner.make_classifier(
            provision=provision if provision is not None else _spec(),
            model_name="classify/model",
            port=9999,
            depth=depth,
        )
        result = classifier("Clause text.", "Some context.")
    # First entry is the client construction; the rest are completions.
    return result, calls[0], calls[1:]


def _enum(call: dict[str, object]) -> list[str]:
    schema = call["response_format"]["json_schema"]["schema"]
    return schema["properties"]["subtype"]["enum"]


class Stage03ClassifierFactoryTests(unittest.TestCase):
    def test_factory_sends_schema_prompt_and_zero_temperature(self) -> None:
        result, client, completions = _classify(
            ['{"subtype": "reassignment"}']
        )

        self.assertEqual(result, {"subtype_1": "reassignment"})
        self.assertEqual(client["client_kwargs"]["base_url"], "http://localhost:9999/v1")
        self.assertEqual(len(completions), 1)
        call = completions[0]
        self.assertEqual(call["model"], "classify/model")
        self.assertEqual(call["temperature"], 0)
        json_schema = call["response_format"]["json_schema"]
        self.assertEqual(call["response_format"]["type"], "json_schema")
        self.assertTrue(json_schema["strict"])
        self.assertEqual(_enum(call), [*SUBTYPES, runner.OTHER_LABEL])
        self.assertEqual(call["messages"][0]["role"], "system")
        self.assertIn("Some context.", call["messages"][1]["content"])

    def test_second_pass_offers_only_the_chosen_parents_children(self) -> None:
        result, _, completions = _classify(
            [
                '{"subtype": "retraining"}',
                '{"subtype": "tuition_support"}',
            ],
            depth=2,
        )

        self.assertEqual(
            result,
            {"subtype_1": "retraining", "subtype_2": "tuition_support"},
        )
        self.assertEqual(len(completions), 2)
        self.assertEqual(_enum(completions[0]), [*SUBTYPES, runner.OTHER_LABEL])
        # Only "retraining"'s own children, never a sibling branch's.
        self.assertEqual(
            _enum(completions[1]),
            ["tuition_support", "on_the_job", runner.OTHER_LABEL],
        )
        self.assertNotIn("beneficiary", completions[1]["messages"][1]["content"])
        self.assertIn('classified as "retraining"', completions[1]["messages"][1]["content"])

    def test_other_at_the_top_level_is_terminal(self) -> None:
        result, _, completions = _classify(
            ['{"subtype": "other"}'], depth=2
        )

        # No second call is issued at all: "other" names no node to refine into.
        self.assertEqual(len(completions), 1)
        self.assertEqual(
            result,
            {"subtype_1": runner.OTHER_LABEL, "subtype_2": None},
        )

    def test_cascade_stops_at_a_leaf_shallower_than_the_requested_depth(self) -> None:
        result, _, completions = _classify(
            ['{"subtype": "reassignment"}'], depth=2
        )

        self.assertEqual(len(completions), 1)
        self.assertEqual(result["subtype_1"], "reassignment")
        self.assertIsNone(result["subtype_2"])

    def test_depth_one_asks_once_and_reports_only_the_first_level(self) -> None:
        result, _, completions = _classify(
            ['{"subtype": "retraining"}'], depth=1
        )

        self.assertEqual(len(completions), 1)
        self.assertEqual(set(result), {"subtype_1"})

    def test_factory_requires_subtypes(self) -> None:
        with self.assertRaisesRegex(ValueError, "no subtypes"):
            runner.make_classifier(
                provision=_spec(taxonomy=None), model_name="classify/model", port=1
            )


class Stage03CliTests(unittest.TestCase):
    def test_parse_args_requires_provision(self) -> None:
        with self.assertRaises(SystemExit):
            runner.parse_args([])

        args = runner.parse_args(["--provision", "technology"])

        self.assertEqual(args.provision, "technology")
        self.assertEqual(args.extract_model_name, runner.DEFAULT_EXTRACT_MODEL_NAME)
        self.assertEqual(args.model_name, runner.DEFAULT_MODEL_NAME)
        self.assertEqual(args.request_concurrency, 8)

    def test_parse_args_accepts_source_and_multiple_document_ids(self) -> None:
        args = runner.parse_args(
            [
                "--provision",
                "technology",
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

    def test_parse_args_accepts_both_taxonomy_depth_spellings(self) -> None:
        self.assertEqual(runner.parse_args(["--provision", "technology"]).taxonomy_depth, 1)
        for flag in ("--taxonomy-depth", "--taxonomy_depth"):
            with self.subTest(flag=flag):
                args = runner.parse_args(["--provision", "technology", flag, "2"])
                self.assertEqual(args.taxonomy_depth, 2)

    def test_validate_args_rejects_non_positive_taxonomy_depth(self) -> None:
        args = runner.parse_args(["--provision", "technology", "--taxonomy-depth", "0"])
        with self.assertRaisesRegex(ValueError, "--taxonomy-depth"):
            runner.validate_args(args)

    def test_main_rejects_depth_beyond_the_declared_taxonomy(self) -> None:
        # technology declares two levels; asking for three is a config error, not
        # a silent no-op, so it must fail before any server starts.
        with self.assertRaisesRegex(ValueError, "declares 2 level"):
            runner.main(["--provision", "technology", "--taxonomy-depth", "3"])

    def test_validate_args_rejects_bad_concurrency_and_device(self) -> None:
        args = runner.parse_args(
            ["--provision", "technology", "--request-concurrency", "0"]
        )
        with self.assertRaisesRegex(ValueError, "--request-concurrency"):
            runner.validate_args(args)

        mismatched_args = runner.parse_args(
            ["--provision", "technology", "--device", "1,3"]
        )
        with self.assertRaisesRegex(ValueError, "selects 2 GPU"):
            runner.validate_args(mismatched_args)

    def test_main_rejects_provision_without_subtypes(self) -> None:
        with self.assertRaisesRegex(ValueError, "no subtypes"):
            runner.main(["--provision", "wage_table"])

    def test_main_smoke_with_mocked_vllm_and_classifier(self) -> None:
        server_kwargs: list[dict[str, object]] = []
        classifier_kwargs: list[dict[str, object]] = []
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            input_root = cache_root / "stg_02_extract"
            output_root = cache_root / "stg_03_classify"
            input_path = (
                input_root / "source" / "extract_model" / "doc" / "technology.jsonl"
            )
            input_path.parent.mkdir(parents=True)
            input_path.write_text(
                json.dumps(_extraction("clause", "Applies to operators.")) + "\n",
                encoding="utf-8",
            )

            class FakeServer:
                def __init__(self, *args, **kwargs):
                    del args
                    server_kwargs.append(kwargs)

                def start(self):
                    pass

                def close(self):
                    pass

            def fake_classifier_factory(*args, **kwargs):
                del args
                classifier_kwargs.append(kwargs)
                return lambda extraction_text, context: {
                    "beneficiary": "worker",
                    "subtype_1": "preemptive_rights",
                    "subtype_2": "notification_right",
                }

            with (
                patch.object(runner, "VLLMServer", FakeServer),
                patch.object(runner, "make_classifier", fake_classifier_factory),
            ):
                runner.main(
                    [
                        "--provision",
                        "technology",
                        "--input-root",
                        str(input_root),
                        "--output-root",
                        str(output_root),
                        "--extract-model-name",
                        "extract/model",
                        "--model-name",
                        "classify/model",
                        "--source",
                        "source",
                        "--device",
                        "3",
                        "--document-id",
                        "doc",
                        "--taxonomy-depth",
                        "2",
                        "--force",
                        "--no-progress",
                    ]
                )

            output_path = (
                output_root / "source" / "classify_model" / "doc" / "technology.jsonl"
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["beneficiary"], "worker")
        self.assertEqual(rows[0]["subtype_1"], "preemptive_rights")
        self.assertEqual(rows[0]["subtype_2"], "notification_right")
        self.assertEqual(rows[0]["taxonomy_depth"], 2)
        self.assertEqual(rows[0]["extract_model_name"], "extract/model")
        self.assertEqual(rows[0]["model_name"], "classify/model")
        self.assertEqual(server_kwargs[0]["device"], "3")
        self.assertIn(
            "--default-chat-template-kwargs", server_kwargs[0]["extra_serve_args"]
        )
        self.assertEqual(
            classifier_kwargs[0]["provision"].clause_type, "technology"
        )
        self.assertEqual(classifier_kwargs[0]["depth"], 2)

    def test_main_reports_skipped_without_starting_server(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            input_root = cache_root / "stg_02_extract"
            output_root = cache_root / "stg_03_classify"
            input_path = (
                input_root / "source" / "extract_model" / "doc" / "technology.jsonl"
            )
            input_path.parent.mkdir(parents=True)
            input_path.write_text(
                json.dumps(_extraction("clause", None)) + "\n", encoding="utf-8"
            )
            output_path = (
                output_root / "source" / "classify_model" / "doc" / "technology.jsonl"
            )
            output_path.parent.mkdir(parents=True)
            output_path.write_text("existing\n", encoding="utf-8")

            def unexpected_server(*args, **kwargs):
                raise AssertionError("server must not start when nothing is pending")

            with patch.object(runner, "VLLMServer", unexpected_server):
                runner.main(
                    [
                        "--provision",
                        "technology",
                        "--input-root",
                        str(input_root),
                        "--output-root",
                        str(output_root),
                        "--extract-model-name",
                        "extract/model",
                        "--model-name",
                        "classify/model",
                        "--no-progress",
                    ]
                )

            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing\n")


if __name__ == "__main__":
    unittest.main()
