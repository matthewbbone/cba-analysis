import importlib.util
from pathlib import Path
import sys
import types
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "pipeline" / "02_provision_extract" / "runner.py"
)
class DummyOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


OPENAI_STUB = types.ModuleType("openai")
OPENAI_STUB.OpenAI = DummyOpenAI
OPENAI_STUB.AsyncOpenAI = DummyOpenAI
sys.modules.setdefault("openai", OPENAI_STUB)
VLLM_SERVER_STUB = types.ModuleType("pipeline.utils.vllm_server")
VLLM_SERVER_STUB.VLLMServer = DummyOpenAI
sys.modules.setdefault("pipeline.utils.vllm_server", VLLM_SERVER_STUB)
MODULE_SPEC = importlib.util.spec_from_file_location("provision_extract_runner", MODULE_PATH)
RUNNER_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(RUNNER_MODULE)
ExtractionRunner = RUNNER_MODULE.ExtractionRunner

PARTIES = {
    "Worker": "worker",
    "Firm": "firm",
    "Union": "union",
    "Manager": "manager",
}


class ResolveUniqueSpanOffsetsTests(unittest.TestCase):
    def test_matches_unescaped_dollar_against_escaped_section_text(self) -> None:
        section_text = r"The Employer shall pay \$5 per hour."
        expected_start = section_text.index(r"\$5 per hour")
        expected_end = expected_start + len(r"\$5 per hour")

        start, end, status = ExtractionRunner._resolve_unique_span_offsets(
            section_text,
            "$5 per hour",
        )

        self.assertEqual((start, end, status), (expected_start, expected_end, "markdown_escaped"))

    def test_matches_unescaped_markdown_special_characters(self) -> None:
        section_text = r"The parties agree to retain \_seniority\_ and \*wage\* terms."
        expected_start = section_text.index(r"\_seniority\_ and \*wage\*")
        expected_end = expected_start + len(r"\_seniority\_ and \*wage\*")

        start, end, status = ExtractionRunner._resolve_unique_span_offsets(
            section_text,
            "_seniority_ and *wage*",
        )

        self.assertEqual((start, end, status), (expected_start, expected_end, "markdown_escaped"))

    def test_rejects_ambiguous_normalized_matches(self) -> None:
        section_text = r"The Employer pays \$5 today and \$5 tomorrow."

        start, end, status = ExtractionRunner._resolve_unique_span_offsets(section_text, "$5")

        self.assertEqual((start, end, status), (None, None, "unresolved"))

    def test_backfill_grounds_existing_payload_without_model_call(self) -> None:
        runner = ExtractionRunner(
            base_url="http://localhost:8000/v1",
            api_key="test",
            model_name="dummy-model",
            provider="vllm",
            parties=PARTIES,
        )
        output_payload = {
            "document_meta_data": {},
            "sections": [
                {
                    "section_index": 0,
                    "header": "Compensation",
                    "text": r"The Employer shall pay \$5 per hour.",
                    "provisions": [
                        {
                            "subject": "Firm",
                            "beneficiary": "Worker",
                            "conditions": "None",
                            "value": "Pay $5 per hour.",
                            "span": "$5 per hour",
                        }
                    ],
                }
            ],
        }

        grounded_sections, grounded_provisions = runner._backfill_grounded_spans_in_output_payload(
            output_payload
        )
        provision = output_payload["sections"][0]["provisions"][0]

        self.assertEqual((grounded_sections, grounded_provisions), (1, 1))
        self.assertEqual(provision["grounding_status"], "markdown_escaped")
        self.assertEqual(
            (provision["span_start"], provision["span_end"]),
            (
                output_payload["sections"][0]["text"].index(r"\$5 per hour"),
                output_payload["sections"][0]["text"].index(r"\$5 per hour") + len(r"\$5 per hour"),
            ),
        )
        self.assertEqual(
            output_payload["document_meta_data"]["actor_beneficiary_counts"]["firm worker"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
