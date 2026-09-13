from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from references.cuad import compare_extractions as ce


class IntervalOverlapTests(unittest.TestCase):
    def test_no_overlap(self) -> None:
        self.assertEqual(ce.interval_overlap((0, 10), (10, 20)), 0)
        self.assertEqual(ce.interval_overlap((0, 10), (20, 30)), 0)

    def test_exact_overlap(self) -> None:
        self.assertEqual(ce.interval_overlap((5, 15), (5, 15)), 10)

    def test_partial_overlap(self) -> None:
        self.assertEqual(ce.interval_overlap((0, 10), (5, 15)), 5)

    def test_containment(self) -> None:
        self.assertEqual(ce.interval_overlap((0, 100), (10, 20)), 10)


class JaccardTests(unittest.TestCase):
    def test_no_overlap_is_zero(self) -> None:
        self.assertEqual(ce.jaccard((0, 10), (10, 20)), 0.0)

    def test_exact_match_is_one(self) -> None:
        self.assertEqual(ce.jaccard((5, 15), (5, 15)), 1.0)

    def test_half_overlap(self) -> None:
        # intersection=5, union=15 -> 1/3
        self.assertAlmostEqual(ce.jaccard((0, 10), (5, 15)), 5 / 15)

    def test_containment_below_one(self) -> None:
        # intersection=10, union=100 -> 0.1
        self.assertAlmostEqual(ce.jaccard((0, 100), (10, 20)), 0.1)


class GreedyMatchTests(unittest.TestCase):
    def test_clean_one_to_one(self) -> None:
        golds = [(0, 10), (20, 30)]
        preds = [(0, 10), (20, 30)]
        tp, fp, fn = ce.greedy_match(golds, preds, ce.interval_overlap, 0)
        self.assertEqual((tp, fp, fn), (2, 0, 0))

    def test_no_overlap_is_all_fp_fn(self) -> None:
        golds = [(0, 10)]
        preds = [(100, 110)]
        tp, fp, fn = ce.greedy_match(golds, preds, ce.interval_overlap, 0)
        self.assertEqual((tp, fp, fn), (0, 1, 1))

    def test_one_prediction_spanning_two_golds_matches_once(self) -> None:
        golds = [(0, 10), (20, 30)]
        preds = [(0, 30)]
        tp, fp, fn = ce.greedy_match(golds, preds, ce.interval_overlap, 0)
        # The single prediction can only claim one gold span; the other gold
        # span is unmatched (a false negative), and there is no extra
        # (unmatched) prediction left over to count as a false positive.
        self.assertEqual((tp, fp, fn), (1, 0, 1))

    def test_two_predictions_on_one_gold_match_once(self) -> None:
        golds = [(0, 10)]
        preds = [(0, 5), (5, 10)]
        tp, fp, fn = ce.greedy_match(golds, preds, ce.interval_overlap, 0)
        # Only one prediction can claim the single gold span; the other is an
        # unmatched false positive.
        self.assertEqual((tp, fp, fn), (1, 1, 0))

    def test_jaccard_threshold_excludes_low_overlap(self) -> None:
        golds = [(0, 100)]
        preds = [(90, 100)]  # overlap=10, union=100 -> jaccard=0.1
        tp, fp, fn = ce.greedy_match(golds, preds, ce.jaccard, 0.5)
        self.assertEqual((tp, fp, fn), (0, 1, 1))
        # But it does count under the "any overlap" criterion.
        tp, fp, fn = ce.greedy_match(golds, preds, ce.interval_overlap, 0)
        self.assertEqual((tp, fp, fn), (1, 0, 0))


class DocumentIdResolverTests(unittest.TestCase):
    def test_strip_only_match(self) -> None:
        gold_documents = {"Some Title": ce.GoldDocument(title="Some Title", spans_by_category={})}
        resolve = ce.build_document_id_resolver(gold_documents)
        self.assertEqual(resolve("Some Title "), "Some Title")

    def test_netgear_override(self) -> None:
        title = (
            "NETGEAR,INC_04_21_2003-EX-10.16-AMENDMENT TO THE DISTRIBUTOR "
            "AGREEMENT BETWEEN INGRAM MICRO AND NETGEAR"
        )
        document_id = title + "-"
        gold_documents = {title: ce.GoldDocument(title=title, spans_by_category={})}
        resolve = ce.build_document_id_resolver(gold_documents)
        self.assertEqual(resolve(document_id), title)

    def test_harpoon_override(self) -> None:
        title = (
            "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_"
            "Development Agreement"
        )
        document_id = title + "_Option Agreement"
        gold_documents = {title: ce.GoldDocument(title=title, spans_by_category={})}
        resolve = ce.build_document_id_resolver(gold_documents)
        self.assertEqual(resolve(document_id), title)

    def test_unknown_document_id_resolves_to_none(self) -> None:
        gold_documents = {"Some Title": ce.GoldDocument(title="Some Title", spans_by_category={})}
        resolve = ce.build_document_id_resolver(gold_documents)
        self.assertIsNone(resolve("Nonexistent"))


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def _qa(title: str, category: str, answers: list[tuple[int, str]]) -> dict[str, object]:
    return {
        "id": f"{title}__{category}",
        "question": f"question about {category}",
        "answers": [{"answer_start": start, "text": text} for start, text in answers],
        "is_impossible": not answers,
    }


class EndToEndTests(unittest.TestCase):
    def test_precision_recall_f1_across_two_documents(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            doc_a_title = "Doc A Agreement"
            doc_b_title = "Doc B Agreement"
            cuad_json = {
                "version": "test",
                "data": [
                    {
                        "title": doc_a_title,
                        "paragraphs": [
                            {
                                "context": "x" * 200,
                                "qas": [
                                    _qa(doc_a_title, "Governing Law", [(0, "x" * 10)]),
                                    _qa(doc_a_title, "Non-Compete", []),
                                ],
                            }
                        ],
                    },
                    {
                        "title": doc_b_title,
                        "paragraphs": [
                            {
                                "context": "y" * 200,
                                "qas": [
                                    _qa(doc_b_title, "Governing Law", [(50, "y" * 10)]),
                                    _qa(doc_b_title, "Non-Compete", []),
                                ],
                            }
                        ],
                    },
                ],
            }
            cuad_json_path = root / "CUADv1.json"
            cuad_json_path.write_text(json.dumps(cuad_json), encoding="utf-8")

            model_dir = root / "stg_02_extract" / "cuad" / "test_model"
            # Doc A: exact-match prediction -> true positive under both criteria.
            _write_jsonl(
                model_dir / doc_a_title / "governing_law.jsonl",
                [{"span_start": 0, "span_end": 10}],
            )
            # Doc B: prediction only loosely overlaps the gold span (jaccard
            # well below 0.5) -> true positive for "any overlap" only, and a
            # spurious extra prediction with no gold match at all.
            _write_jsonl(
                model_dir / doc_b_title / "governing_law.jsonl",
                [
                    {"span_start": 45, "span_end": 55},
                    {"span_start": 150, "span_end": 160},
                ],
            )

            gold_documents = ce.load_gold_documents(cuad_json_path)
            resolve_title = ce.build_document_id_resolver(gold_documents)
            predictions = ce.load_predicted_spans(model_dir, "governing_law")
            result = ce.evaluate_clause_type(
                "governing_law", gold_documents, resolve_title, predictions
            )

            self.assertEqual(result.documents_evaluated, 2)
            self.assertEqual(result.documents_total, 2)
            self.assertEqual(result.gold_span_count, 2)
            self.assertEqual(result.predicted_span_count, 3)

            # any overlap: tp=2 (doc A exact, doc B loose overlap), fp=1
            # (doc B's spurious extra prediction), fn=0.
            self.assertEqual(result.any_overlap.true_positives, 2)
            self.assertEqual(result.any_overlap.false_positives, 1)
            self.assertEqual(result.any_overlap.false_negatives, 0)
            self.assertAlmostEqual(result.any_overlap.precision, 2 / 3)
            self.assertAlmostEqual(result.any_overlap.recall, 1.0)

            # jaccard > 0.5: only doc A's exact match qualifies; doc B's gold
            # span goes unmatched (fn) and both of its predictions are
            # unmatched (fp).
            self.assertEqual(result.jaccard_over_half.true_positives, 1)
            self.assertEqual(result.jaccard_over_half.false_positives, 2)
            self.assertEqual(result.jaccard_over_half.false_negatives, 1)
            self.assertAlmostEqual(result.jaccard_over_half.precision, 1 / 3)
            self.assertAlmostEqual(result.jaccard_over_half.recall, 0.5)

    def test_missing_output_file_is_excluded_not_zero_scored(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            title = "Only Doc"
            cuad_json = {
                "version": "test",
                "data": [
                    {
                        "title": title,
                        "paragraphs": [
                            {
                                "context": "z" * 50,
                                "qas": [_qa(title, "Governing Law", [(0, "z" * 5)])],
                            }
                        ],
                    }
                ],
            }
            cuad_json_path = root / "CUADv1.json"
            cuad_json_path.write_text(json.dumps(cuad_json), encoding="utf-8")

            model_dir = root / "stg_02_extract" / "cuad" / "test_model"
            model_dir.mkdir(parents=True)
            # No governing_law.jsonl written anywhere: stage 2 hasn't run yet.

            gold_documents = ce.load_gold_documents(cuad_json_path)
            resolve_title = ce.build_document_id_resolver(gold_documents)
            predictions = ce.load_predicted_spans(model_dir, "governing_law")
            result = ce.evaluate_clause_type(
                "governing_law", gold_documents, resolve_title, predictions
            )

            self.assertEqual(result.documents_evaluated, 0)
            self.assertEqual(result.documents_total, 1)
            self.assertEqual(result.any_overlap.true_positives, 0)
            self.assertEqual(result.any_overlap.false_negatives, 0)

    def test_empty_output_file_counts_as_zero_predictions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            title = "Only Doc"
            cuad_json = {
                "version": "test",
                "data": [
                    {
                        "title": title,
                        "paragraphs": [
                            {
                                "context": "z" * 50,
                                "qas": [_qa(title, "Governing Law", [(0, "z" * 5)])],
                            }
                        ],
                    }
                ],
            }
            cuad_json_path = root / "CUADv1.json"
            cuad_json_path.write_text(json.dumps(cuad_json), encoding="utf-8")

            model_dir = root / "stg_02_extract" / "cuad" / "test_model"
            _write_jsonl(model_dir / title / "governing_law.jsonl", [])

            gold_documents = ce.load_gold_documents(cuad_json_path)
            resolve_title = ce.build_document_id_resolver(gold_documents)
            predictions = ce.load_predicted_spans(model_dir, "governing_law")
            result = ce.evaluate_clause_type(
                "governing_law", gold_documents, resolve_title, predictions
            )

            self.assertEqual(result.documents_evaluated, 1)
            self.assertEqual(result.gold_span_count, 1)
            self.assertEqual(result.predicted_span_count, 0)
            self.assertEqual(result.any_overlap.false_negatives, 1)


if __name__ == "__main__":
    unittest.main()
