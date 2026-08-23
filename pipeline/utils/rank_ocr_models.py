"""Rank models from the reviewer's pairwise comparison judgments.

``--stage`` selects which review set to fit.  Both stages share the reviewer's
``left_model``/``right_model``/``choice`` record shape, so they use the same
model:

* ``ocr`` reads ``cache/reviewer/ocr_comparisons.jsonl`` and writes
  ``cache/reviewer/ocr_model_rankings.json``; and
* ``extraction`` reads ``cache/reviewer/extraction_comparisons.jsonl`` and
  writes ``cache/reviewer/extraction_model_rankings.json``.

The fitted model is a Bradley-Terry-style logistic model.  Decisive judgments
are ordinary pairwise wins.  A ``both_good`` or ``both_bad`` judgment has two
parts:

* a fractional pairwise result that pulls the two model scores together; and
* comparisons against a fixed neutral-quality baseline that move both models
  up for ``both_good`` and down for ``both_bad``.

``--tie-quality-weight`` controls how a tie judgment is divided between those
two pieces.  Its default of 0.5 gives equal total weight to the relative tie
and its absolute good/bad assessment.  L2 regularization makes the ranking
finite for small or perfectly separated review sets.

Run from anywhere in the repository::

    python pipeline/utils/rank_ocr_models.py
    python pipeline/utils/rank_ocr_models.py --top-ocr-models
    python pipeline/utils/rank_ocr_models.py --stage extraction

The stage defaults to ``ocr``, and ``--input``/``--output`` override that
stage's paths.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import AbstractSet, Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWER_CACHE = PROJECT_ROOT / "cache" / "reviewer"
VALID_CHOICES = {"left_better", "right_better", "both_good", "both_bad"}
TOP_OCR_MODELS = frozenset(
    {
        "AIDC-AI_Ovis2.6-30B-A3B",
        "ATH-MaaS_OvisOCR2",
        "Qwen_Qwen3.6-27B-FP8",
        "google_gemma-4-31B-it",
    }
)
# Backwards-compatible import alias.  The named comparison cohort now contains
# four models and includes the specialized OvisOCR2 checkpoint.
TOP_3_GENERAL_VLM_MODELS = TOP_OCR_MODELS


@dataclass(frozen=True)
class Stage:
    """One reviewer comparison set and where its ranking is written."""

    name: str
    label: str
    input_path: Path
    output_path: Path
    # The named subset selected by ``--top-ocr-models``; stages without a named
    # cohort reject that flag.
    cohort: frozenset[str] | None = None
    cohort_scope: str | None = None


STAGES = {
    "ocr": Stage(
        name="ocr",
        label="OCR",
        input_path=REVIEWER_CACHE / "ocr_comparisons.jsonl",
        output_path=REVIEWER_CACHE / "ocr_model_rankings.json",
        cohort=TOP_OCR_MODELS,
        cohort_scope="top_ocr_models",
    ),
    "extraction": Stage(
        name="extraction",
        label="extraction",
        input_path=REVIEWER_CACHE / "extraction_comparisons.jsonl",
        output_path=REVIEWER_CACHE / "extraction_model_rankings.json",
    ),
}
DEFAULT_STAGE = "ocr"
# Backwards-compatible aliases for the original OCR-only defaults.
DEFAULT_INPUT = STAGES[DEFAULT_STAGE].input_path
DEFAULT_OUTPUT = STAGES[DEFAULT_STAGE].output_path


@dataclass(frozen=True)
class Judgment:
    """A validated pairwise comparison used by the ranking model."""

    left_model: str
    right_model: str
    choice: str


@dataclass(frozen=True)
class LogisticObservation:
    """One weighted logistic outcome with a sparse score contrast."""

    coefficients: tuple[tuple[int, float], ...]
    outcome: float
    weight: float


def load_judgments(path: Path, label: str = "comparison") -> list[Judgment]:
    """Load and validate reviewer JSONL records."""

    judgments: list[Judgment] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            left_model = record.get("left_model")
            right_model = record.get("right_model")
            choice = record.get("choice")
            if not isinstance(left_model, str) or not left_model.strip():
                raise ValueError(f"{path}:{line_number}: missing left_model")
            if not isinstance(right_model, str) or not right_model.strip():
                raise ValueError(f"{path}:{line_number}: missing right_model")
            if left_model == right_model:
                raise ValueError(f"{path}:{line_number}: a model cannot be compared to itself")
            if choice not in VALID_CHOICES:
                valid = ", ".join(sorted(VALID_CHOICES))
                raise ValueError(
                    f"{path}:{line_number}: invalid choice {choice!r}; expected one of {valid}"
                )
            judgments.append(Judgment(left_model, right_model, choice))

    if not judgments:
        raise ValueError(f"{path}: no {label} comparison judgments found")
    return judgments


def judgments_within_models(
    judgments: Iterable[Judgment],
    models: AbstractSet[str],
) -> list[Judgment]:
    """Keep head-to-head judgments whose two participants are in ``models``."""

    return [
        judgment
        for judgment in judgments
        if judgment.left_model in models and judgment.right_model in models
    ]


def _make_observations(
    judgments: Sequence[Judgment],
    model_index: dict[str, int],
    tie_quality_weight: float,
) -> list[LogisticObservation]:
    observations: list[LogisticObservation] = []
    relative_tie_weight = 1.0 - tie_quality_weight
    per_model_quality_weight = tie_quality_weight / 2.0

    for judgment in judgments:
        left = model_index[judgment.left_model]
        right = model_index[judgment.right_model]
        if judgment.choice == "left_better":
            observations.append(LogisticObservation(((left, 1.0), (right, -1.0)), 1.0, 1.0))
        elif judgment.choice == "right_better":
            observations.append(LogisticObservation(((right, 1.0), (left, -1.0)), 1.0, 1.0))
        else:
            # A fractional Bradley-Terry outcome represents equal performance.
            if relative_tie_weight:
                observations.append(
                    LogisticObservation(
                        ((left, 1.0), (right, -1.0)),
                        0.5,
                        relative_tie_weight,
                    )
                )

            # A one-coefficient observation is a comparison with a fixed
            # baseline whose log-strength is zero.
            quality_outcome = 1.0 if judgment.choice == "both_good" else 0.0
            if per_model_quality_weight:
                observations.append(
                    LogisticObservation(((left, 1.0),), quality_outcome, per_model_quality_weight)
                )
                observations.append(
                    LogisticObservation(((right, 1.0),), quality_outcome, per_model_quality_weight)
                )

    return observations


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system using Gaussian elimination with pivoting."""

    size = len(vector)
    augmented = [row.copy() + [value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("ranking Hessian is singular; increase --l2")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for cell in range(column, size + 1):
            augmented[column][cell] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for cell in range(column, size + 1):
                augmented[row][cell] -= factor * augmented[column][cell]
    return [augmented[row][size] for row in range(size)]


def _gradient_and_hessian(
    scores: Sequence[float],
    observations: Sequence[LogisticObservation],
    l2: float,
) -> tuple[list[float], list[list[float]]]:
    size = len(scores)
    gradient = [l2 * score for score in scores]
    hessian = [[0.0] * size for _ in range(size)]
    for index in range(size):
        hessian[index][index] = l2

    for observation in observations:
        linear_predictor = sum(
            coefficient * scores[index]
            for index, coefficient in observation.coefficients
        )
        probability = _sigmoid(linear_predictor)
        residual = observation.weight * (probability - observation.outcome)
        curvature = observation.weight * probability * (1.0 - probability)
        for row, row_coefficient in observation.coefficients:
            gradient[row] += residual * row_coefficient
            for column, column_coefficient in observation.coefficients:
                hessian[row][column] += curvature * row_coefficient * column_coefficient
    return gradient, hessian


def fit_scores(
    judgments: Sequence[Judgment],
    *,
    tie_quality_weight: float = 0.5,
    l2: float = 0.5,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> tuple[list[str], list[float], list[float], int]:
    """Fit regularized Bradley-Terry-style log-strengths.

    Returns model names, fitted log-strengths, approximate standard errors, and
    the number of Newton iterations used.
    """

    if not 0.0 <= tie_quality_weight <= 1.0:
        raise ValueError("tie_quality_weight must be between 0 and 1")
    if l2 <= 0.0:
        raise ValueError("l2 must be greater than zero")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be greater than zero")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero")

    models = sorted(
        {model for judgment in judgments for model in (judgment.left_model, judgment.right_model)}
    )
    if not models:
        raise ValueError("at least one judgment is required")
    model_index = {model: index for index, model in enumerate(models)}
    observations = _make_observations(judgments, model_index, tie_quality_weight)
    scores = [0.0] * len(models)

    for iteration in range(1, max_iterations + 1):
        gradient, hessian = _gradient_and_hessian(scores, observations, l2)
        step = _solve_linear_system(hessian, gradient)
        scores = [score - delta for score, delta in zip(scores, step, strict=True)]
        if max(abs(delta) for delta in step) < tolerance:
            break
    else:
        raise RuntimeError(f"ranking model did not converge after {max_iterations} iterations")

    _, final_hessian = _gradient_and_hessian(scores, observations, l2)
    standard_errors = []
    for index in range(len(models)):
        unit_vector = [0.0] * len(models)
        unit_vector[index] = 1.0
        inverse_column = _solve_linear_system(final_hessian, unit_vector)
        standard_errors.append(math.sqrt(max(0.0, inverse_column[index])))

    return models, scores, standard_errors, iteration


def _counts_by_model(judgments: Iterable[Judgment]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "decisive_wins": 0,
            "decisive_losses": 0,
            "both_good": 0,
            "both_bad": 0,
            "total_comparisons": 0,
        }
    )
    for judgment in judgments:
        left = counts[judgment.left_model]
        right = counts[judgment.right_model]
        left["total_comparisons"] += 1
        right["total_comparisons"] += 1
        if judgment.choice == "left_better":
            left["decisive_wins"] += 1
            right["decisive_losses"] += 1
        elif judgment.choice == "right_better":
            right["decisive_wins"] += 1
            left["decisive_losses"] += 1
        else:
            left[judgment.choice] += 1
            right[judgment.choice] += 1
    return counts


def build_ranking(
    judgments: Sequence[Judgment],
    *,
    tie_quality_weight: float = 0.5,
    l2: float = 0.5,
) -> dict[str, Any]:
    """Build a serializable ranking and fitting metadata."""

    models, scores, standard_errors, iterations = fit_scores(
        judgments,
        tie_quality_weight=tie_quality_weight,
        l2=l2,
    )
    counts = _counts_by_model(judgments)
    rows = []
    for model, log_strength, standard_error in zip(
        models, scores, standard_errors, strict=True
    ):
        # This is the fitted chance of beating the neutral-quality baseline.
        quality_score = 100.0 * _sigmoid(log_strength)
        lower = 100.0 * _sigmoid(log_strength - 1.96 * standard_error)
        upper = 100.0 * _sigmoid(log_strength + 1.96 * standard_error)
        rows.append(
            {
                "model": model,
                "score": quality_score,
                "score_ci_95": [lower, upper],
                "log_strength": log_strength,
                "strength": math.exp(log_strength),
                **counts[model],
            }
        )

    rows.sort(key=lambda row: (-row["score"], row["model"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    choice_counts = {choice: 0 for choice in sorted(VALID_CHOICES)}
    for judgment in judgments:
        choice_counts[judgment.choice] += 1

    return {
        "method": "regularized_bradley_terry_with_valued_ties",
        "generated_at": datetime.now(UTC).isoformat(),
        "judgment_count": len(judgments),
        "model_count": len(models),
        "choice_counts": choice_counts,
        "tie_quality_weight": tie_quality_weight,
        "l2": l2,
        "iterations": iterations,
        "score_definition": (
            "Estimated percentage chance of beating a fixed neutral-quality baseline; "
            "higher is better and 50 is neutral."
        ),
        "rankings": rows,
    }


def print_ranking(output: dict[str, Any]) -> None:
    rows = output["rankings"]
    stage_name = output.get("stage")
    if stage_name:
        stage = STAGES.get(stage_name)
        label = stage.label if stage else stage_name
        print(f"{label} model ranking from {output['judgment_count']} judgments\n")
    model_width = max(len("model"), *(len(row["model"]) for row in rows))
    header = (
        f"{'rank':>4}  {'score':>7}  {'W':>3}  {'L':>3}  "
        f"{'good':>4}  {'bad':>3}  {'n':>3}  {'model':<{model_width}}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['rank']:>4}  {row['score']:>7.2f}  "
            f"{row['decisive_wins']:>3}  {row['decisive_losses']:>3}  "
            f"{row['both_good']:>4}  {row['both_bad']:>3}  "
            f"{row['total_comparisons']:>3}  {row['model']:<{model_width}}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGES),
        default=DEFAULT_STAGE,
        help=f"which reviewer comparison set to rank (default: {DEFAULT_STAGE})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="reviewer JSONL input (default: the selected stage's comparisons file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ranking JSON output (default: the selected stage's rankings file)",
    )
    parser.add_argument(
        "--tie-quality-weight",
        type=float,
        default=0.5,
        help="fraction of each tie assigned to its good/bad signal, from 0 to 1 (default: 0.5)",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.5,
        help="L2 regularization strength for sparse comparisons (default: 0.5)",
    )
    parser.add_argument(
        "--top-ocr-models",
        "--top-4-ocr-models",
        "--top-3-general-vlms",
        "--general-vlms-only",
        dest="top_ocr_models",
        action="store_true",
        help=(
            "fit and report only head-to-head comparisons among Qwen 3.6 27B, "
            "Ovis 2.6 30B, OvisOCR2, and Gemma 4 31B (--stage ocr only)"
        ),
    )
    args = parser.parse_args(argv)
    # Preserve the previous parsed Namespace attribute for callers that used it
    # directly; all CLI aliases select the current four-model cohort.
    args.top_3_general_vlms = args.top_ocr_models
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stage = STAGES[args.stage]
    if args.top_ocr_models and stage.cohort is None:
        raise ValueError(
            f"--top-ocr-models is not available for --stage {stage.name}; "
            "restrict the cohort with --input instead"
        )
    input_path = (args.input or stage.input_path).expanduser().resolve()
    output_path = (args.output or stage.output_path).expanduser().resolve()
    judgments = load_judgments(input_path, stage.label)
    source_judgment_count = len(judgments)
    if args.top_ocr_models:
        cohort = stage.cohort
        judgments = judgments_within_models(judgments, cohort)
        if not judgments:
            models = ", ".join(sorted(cohort))
            raise ValueError(
                "no head-to-head judgments found among the top OCR models: "
                f"{models}"
            )
        observed_models = {
            model
            for judgment in judgments
            for model in (judgment.left_model, judgment.right_model)
        }
        missing_models = cohort - observed_models
        if missing_models:
            raise ValueError(
                "no head-to-head judgments found for top OCR model(s): "
                f"{', '.join(sorted(missing_models))}"
            )
    output = build_ranking(
        judgments,
        tie_quality_weight=args.tie_quality_weight,
        l2=args.l2,
    )
    output["stage"] = stage.name
    output["source_path"] = str(input_path)
    if args.top_ocr_models:
        output["model_scope"] = stage.cohort_scope
        output["included_models"] = sorted(stage.cohort)
        output["source_judgment_count"] = source_judgment_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print_ranking(output)
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
