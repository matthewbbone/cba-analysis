from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.utils.rank_ocr_models import Judgment, build_ranking, load_judgments


def _scores(*judgments: Judgment) -> dict[str, float]:
    output = build_ranking(judgments)
    return {row["model"]: row["score"] for row in output["rankings"]}


def test_decisive_results_rank_winner_above_loser() -> None:
    scores = _scores(
        Judgment("model-a", "model-b", "left_better"),
        Judgment("model-a", "model-b", "left_better"),
    )

    assert scores["model-a"] > scores["model-b"]


def test_good_and_bad_ties_move_equal_models_in_opposite_directions() -> None:
    good_scores = _scores(Judgment("model-a", "model-b", "both_good"))
    bad_scores = _scores(Judgment("model-a", "model-b", "both_bad"))

    assert good_scores["model-a"] == pytest.approx(good_scores["model-b"])
    assert bad_scores["model-a"] == pytest.approx(bad_scores["model-b"])
    assert good_scores["model-a"] > 50.0
    assert bad_scores["model-a"] < 50.0


def test_tie_valence_breaks_otherwise_identical_pairwise_records() -> None:
    scores = _scores(
        Judgment("good-a", "good-b", "both_good"),
        Judgment("bad-a", "bad-b", "both_bad"),
    )

    assert scores["good-a"] > scores["bad-a"]
    assert scores["good-b"] > scores["bad-b"]


def test_build_ranking_reports_counts_and_sorted_ranks() -> None:
    output = build_ranking(
        [
            Judgment("a", "b", "left_better"),
            Judgment("a", "b", "both_good"),
            Judgment("b", "c", "both_bad"),
        ]
    )

    assert [row["rank"] for row in output["rankings"]] == [1, 2, 3]
    by_model = {row["model"]: row for row in output["rankings"]}
    assert by_model["a"]["decisive_wins"] == 1
    assert by_model["a"]["both_good"] == 1
    assert by_model["b"]["decisive_losses"] == 1
    assert by_model["b"]["both_bad"] == 1
    assert by_model["c"]["both_bad"] == 1


def test_load_judgments_reads_reviewer_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "comparisons.jsonl"
    records = [
        {"left_model": "a", "right_model": "b", "choice": "right_better"},
        {"left_model": "b", "right_model": "c", "choice": "both_bad"},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert load_judgments(input_path) == [
        Judgment("a", "b", "right_better"),
        Judgment("b", "c", "both_bad"),
    ]


def test_load_judgments_rejects_unknown_choice(tmp_path: Path) -> None:
    input_path = tmp_path / "comparisons.jsonl"
    input_path.write_text(
        json.dumps({"left_model": "a", "right_model": "b", "choice": "skip"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid choice"):
        load_judgments(input_path)
