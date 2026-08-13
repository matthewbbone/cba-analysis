from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.utils.rank_ocr_models import (
    TOP_OCR_MODELS,
    TOP_3_GENERAL_VLM_MODELS,
    Judgment,
    build_ranking,
    judgments_within_models,
    load_judgments,
    main,
    parse_args,
)


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


def test_top_ocr_model_cohort_and_flag_aliases() -> None:
    assert TOP_OCR_MODELS == {
        "AIDC-AI_Ovis2.6-30B-A3B",
        "ATH-MaaS_OvisOCR2",
        "Qwen_Qwen3.6-27B-FP8",
        "google_gemma-4-31B-it",
    }
    assert TOP_3_GENERAL_VLM_MODELS is TOP_OCR_MODELS
    assert parse_args([]).top_ocr_models is False
    assert parse_args([]).top_3_general_vlms is False
    for option in (
        "--top-ocr-models",
        "--top-4-ocr-models",
        "--top-3-general-vlms",
        "--general-vlms-only",
    ):
        args = parse_args([option])
        assert args.top_ocr_models is True
        assert args.top_3_general_vlms is True


def test_judgments_within_models_requires_both_top_ocr_models() -> None:
    model_a, model_b, model_c, model_d = sorted(TOP_OCR_MODELS)
    judgments = [
        Judgment(model_a, model_b, "left_better"),
        Judgment(model_c, model_a, "both_good"),
        Judgment(model_d, model_c, "both_bad"),
        Judgment(model_a, "specialized-ocr", "right_better"),
        Judgment("specialized-a", "specialized-b", "both_bad"),
    ]

    assert judgments_within_models(judgments, TOP_OCR_MODELS) == judgments[:3]


def test_main_can_rank_only_top_ocr_models(tmp_path: Path) -> None:
    model_a, model_b, model_c, model_d = sorted(TOP_OCR_MODELS)
    input_path = tmp_path / "comparisons.jsonl"
    output_path = tmp_path / "rankings.json"
    records = [
        {"left_model": model_a, "right_model": model_b, "choice": "left_better"},
        {"left_model": model_b, "right_model": model_c, "choice": "both_good"},
        {"left_model": model_c, "right_model": model_d, "choice": "right_better"},
        {"left_model": model_d, "right_model": model_a, "choice": "both_bad"},
        {"left_model": model_a, "right_model": "specialized", "choice": "left_better"},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--top-ocr-models",
        ]
    ) == 0

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["model_scope"] == "top_ocr_models"
    assert output["included_models"] == sorted(TOP_OCR_MODELS)
    assert output["source_judgment_count"] == 5
    assert output["judgment_count"] == 4
    assert output["model_count"] == 4
    assert output["choice_counts"] == {
        "both_bad": 1,
        "both_good": 1,
        "left_better": 1,
        "right_better": 1,
    }
    assert {row["model"] for row in output["rankings"]} == TOP_OCR_MODELS
    assert {row["total_comparisons"] for row in output["rankings"]} == {2}


def test_main_reports_when_top_ocr_models_have_no_head_to_head_judgments(
    tmp_path: Path,
) -> None:
    model_a = sorted(TOP_OCR_MODELS)[0]
    input_path = tmp_path / "comparisons.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "left_model": model_a,
                "right_model": "specialized",
                "choice": "left_better",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no head-to-head judgments"):
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "rankings.json"),
                "--top-ocr-models",
            ]
        )


def test_main_reports_when_one_top_ocr_model_has_no_judgments(tmp_path: Path) -> None:
    missing_model = "ATH-MaaS_OvisOCR2"
    model_a, model_b, model_c = sorted(TOP_OCR_MODELS - {missing_model})
    input_path = tmp_path / "comparisons.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "left_model": model_a,
                        "right_model": model_b,
                        "choice": "left_better",
                    }
                ),
                json.dumps(
                    {
                        "left_model": model_b,
                        "right_model": model_c,
                        "choice": "both_good",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=missing_model):
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "rankings.json"),
                "--top-ocr-models",
            ]
        )
