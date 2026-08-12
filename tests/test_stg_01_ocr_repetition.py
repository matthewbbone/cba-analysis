from argparse import Namespace
import warnings
import unittest

from pipeline.stg_01_ocr.repetition import (
    DegenerationMatch,
    RepetitionError,
    RepetitionPolicy,
    SamplingAttempt,
    apply_repetition_disposition,
    default_policy,
    detect_degeneration,
    trim_degeneration,
)
from pipeline.stg_01_ocr.specialized import miner


class DegenerationDetectionTests(unittest.TestCase):
    def test_detects_repeated_40_character_cycle_at_length_limit(self) -> None:
        unit = "abcdefghijklmnopqrstuvwxyz0123456789!@#$"
        self.assertEqual(len(unit), 40)
        text = "valid transcription\n" + unit * 12

        match = detect_degeneration(text, finish_reason="length")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.end, len(text))
        self.assertEqual(match.start, len("valid transcription\n"))
        self.assertEqual(match.unit, unit)
        self.assertEqual(match.repetitions, 12)

    def test_detects_long_punctuation_run_only_above_strict_threshold(self) -> None:
        self.assertIsNone(detect_degeneration("_" * 1000, "length"))
        match = detect_degeneration("_" * 1001, "length")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.run_length, 1001)

    def test_detects_identical_line_run_with_mixed_line_endings(self) -> None:
        line = "identical wage schedule entry " + "X" * 70
        text = "prefix\n" + line + "\r\n" + line + "\n" + line

        match = detect_degeneration(text, "length")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.kind, "line")
        self.assertEqual(match.repetitions, 3)

    def test_minimum_run_length_boundary_is_inclusive(self) -> None:
        self.assertIsNone(detect_degeneration("a" * 199, "length"))
        self.assertIsNotNone(detect_degeneration("a" * 200, "length"))

    def test_stop_finish_requires_at_least_quarter_of_total_output(self) -> None:
        loop = "ABCD" * 50
        prefix_600 = "".join(f"{number:03d}" for number in range(200))

        self.assertEqual(len(prefix_600 + loop), 800)
        self.assertIsNotNone(detect_degeneration(prefix_600 + loop, "stop"))
        self.assertIsNone(detect_degeneration("X" + prefix_600 + loop, "stop"))

    def test_length_finish_bypasses_share_but_not_minimum_length(self) -> None:
        prefix = "".join(f"unique row {number:04d}\n" for number in range(300))
        loop = "LOOP!" * 40

        self.assertLess(len(loop) / len(prefix + loop), 0.1)
        self.assertIsNotNone(detect_degeneration(prefix + loop, "length"))
        self.assertIsNone(detect_degeneration(prefix + loop, "stop"))

    def test_toc_dot_leaders_are_not_repetition(self) -> None:
        toc = "\n".join(
            f"Article {number} {'.' * 70} {number + 10}"
            for number in range(1, 12)
        )
        self.assertIsNone(detect_degeneration(toc, "stop"))

    def test_markdown_table_rule_run_is_not_repetition(self) -> None:
        rule_rows = "| --- | --- |\n" * 30
        self.assertGreaterEqual(len(rule_rows), 200)
        self.assertLessEqual(len(rule_rows), 1000)
        self.assertIsNone(detect_degeneration(rule_rows, "length"))

    def test_distinct_numeric_table_rows_are_not_repetition(self) -> None:
        table = "| Step | Rate |\n| --- | --- |\n" + "".join(
            f"| {step:02d} | ${20 + step}.00 |\n" for step in range(30)
        )
        self.assertIsNone(detect_degeneration(table, "stop"))

    def test_short_periodic_output_is_not_repetition(self) -> None:
        self.assertIsNone(detect_degeneration("abc" * 60, "length"))

    def test_repeated_interior_with_real_tail_is_not_detected(self) -> None:
        unit = "abcdefghijklmnopqrstuvwxyz0123456789!@#$"
        text = "prefix\n" + unit * 12 + "\nREAL END"
        self.assertIsNone(detect_degeneration(text, "length"))

    def test_trim_keeps_prefix_and_one_cycle(self) -> None:
        unit = "abcdefghijklmnopqrstuvwxyz0123456789!@#$"
        prefix = "valid\n"
        text = prefix + unit * 12
        match = detect_degeneration(text, "length")
        assert match is not None

        self.assertEqual(trim_degeneration(text, match), prefix + unit)


class RepetitionPolicyTests(unittest.TestCase):
    def test_default_retries_change_temperature_and_seed(self) -> None:
        policy = default_policy(
            Namespace(repetition_retries=2, fail_on_repetition=False)
        )

        self.assertEqual(len(policy.attempts), 3)
        self.assertEqual([attempt.temperature for attempt in policy.attempts], [0.0, 0.1, 0.3])
        self.assertEqual([attempt.seed for attempt in policy.attempts], [None, 1, 2])
        self.assertEqual(policy.attempts[0].request_overrides(), {"temperature": 0.0})

    def test_sampling_attempt_merges_extra_body_without_mutation(self) -> None:
        base = {
            "model": "ocr",
            "temperature": 0.0,
            "extra_body": {"top_k": 1, "skip_special_tokens": False},
        }
        attempt = SamplingAttempt(
            temperature=0.2,
            repetition_penalty=1.25,
            seed=2,
            overrides={"top_p": 0.01, "extra_body": {"custom": True}},
        )

        merged = attempt.apply_to(base)

        self.assertEqual(base["extra_body"], {"top_k": 1, "skip_special_tokens": False})
        self.assertEqual(merged["temperature"], 0.2)
        self.assertEqual(merged["seed"], 2)
        self.assertEqual(merged["top_p"], 0.01)
        self.assertEqual(
            merged["extra_body"],
            {
                "top_k": 1,
                "skip_special_tokens": False,
                "custom": True,
                "repetition_penalty": 1.25,
            },
        )

    def test_miner_policy_rejects_length_finished_generations(self) -> None:
        policy = miner.repetition_policy(miner.parse_args([]))

        self.assertTrue(policy.reject_length_finish)
        self.assertEqual(policy.retry_count, 2)
        for retry in policy.attempts[1:]:
            self.assertEqual(retry.overrides["presence_penalty"], 0.0)
            self.assertEqual(retry.overrides["frequency_penalty"], 0.0)

    def test_exhaustion_trims_and_warns_once(self) -> None:
        text = "good\n" + "cycle!" * 40
        match = detect_degeneration(text, "length")
        assert match is not None
        policy = RepetitionPolicy((SamplingAttempt(), SamplingAttempt(temperature=0.2, seed=1)))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = apply_repetition_disposition(text, match, policy)

        self.assertEqual(len(caught), 1)
        self.assertLess(len(result), len(text))
        self.assertTrue(result.startswith("good\n"))

    def test_fail_disposition_raises_without_warning(self) -> None:
        text = "cycle!" * 40
        match = detect_degeneration(text, "length")
        assert match is not None
        policy = RepetitionPolicy((SamplingAttempt(),), fail_on_repetition=True)

        with warnings.catch_warnings(record=True) as caught:
            with self.assertRaises(RepetitionError):
                apply_repetition_disposition(text, match, policy)

        self.assertEqual(caught, [])

    def test_degenerated_match_rejects_non_repeated_metadata(self) -> None:
        with self.assertRaises(ValueError):
            DegenerationMatch(0, 200, "x", 1)


if __name__ == "__main__":
    unittest.main()
