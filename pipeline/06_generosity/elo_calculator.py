import argparse
import json
import math
from pathlib import Path
import sys

from scipy.optimize import minimize

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline.utils.llm import model_slug


DEFAULT_MODEL_NAME = "gpt-5.4-nano"


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def document_ids_from_results(results: dict) -> list[str]:
    return [document["document_id"] for document in results.get("documents", [])]


def observations_by_category(results: dict) -> dict[str, list[dict]]:
    categories = results.get("categories", [])
    observations = {category: [] for category in categories}

    for comparison in results.get("comparisons", []):
        category = comparison.get("category")
        if category not in observations:
            continue

        score = comparison.get("score")
        contract_1 = comparison.get("contract_1_id")
        contract_2 = comparison.get("contract_2_id")
        pair_index = comparison.get("pair_index")
        if not contract_1 or not contract_2:
            continue

        if score == 1:
            observations[category].append(
                {
                    "outcome": "win",
                    "winner": contract_1,
                    "loser": contract_2,
                    "contract_1_id": contract_1,
                    "contract_2_id": contract_2,
                    "pair_index": pair_index,
                }
            )
        elif score == -1:
            observations[category].append(
                {
                    "outcome": "win",
                    "winner": contract_2,
                    "loser": contract_1,
                    "contract_1_id": contract_1,
                    "contract_2_id": contract_2,
                    "pair_index": pair_index,
                }
            )
        elif score == 0:
            observations[category].append(
                {
                    "outcome": "tie",
                    "contract_1_id": contract_1,
                    "contract_2_id": contract_2,
                    "pair_index": pair_index,
                }
            )

    return observations


def fit_bradley_terry(
    document_ids: list[str],
    observations: list[dict],
    max_iter: int = 1000,
    tol: float = 1e-10,
    epsilon: float = 1e-9,
) -> dict:
    decisive_observations = [
        obs for obs in observations
        if obs.get("outcome") == "win"
    ]
    strengths = {doc_id: 1.0 for doc_id in document_ids}
    wins = {doc_id: 0 for doc_id in document_ids}
    losses = {doc_id: 0 for doc_id in document_ids}
    ties = {doc_id: 0 for doc_id in document_ids}

    for obs in observations:
        if obs.get("outcome") == "tie":
            ties[obs["contract_1_id"]] += 1
            ties[obs["contract_2_id"]] += 1
            continue
        winner = obs["winner"]
        loser = obs["loser"]
        wins[winner] += 1
        losses[loser] += 1

    if not decisive_observations:
        return {
            doc_id: {
                "strength": None,
                "log_strength": None,
                "wins": wins[doc_id],
                "losses": losses[doc_id],
                "ties": ties[doc_id],
                "decisive_comparisons": wins[doc_id] + losses[doc_id],
                "total_comparisons": wins[doc_id] + losses[doc_id] + ties[doc_id],
            }
            for doc_id in document_ids
        }

    for _ in range(max_iter):
        next_strengths = {}
        for doc_id in document_ids:
            denominator = 0.0
            for obs in decisive_observations:
                winner = obs["winner"]
                loser = obs["loser"]
                if doc_id not in (winner, loser):
                    continue
                opponent = loser if doc_id == winner else winner
                denominator += 1.0 / (
                    strengths[doc_id] + strengths[opponent] + epsilon
                )
            next_strengths[doc_id] = (
                wins[doc_id] + epsilon
            ) / (denominator + epsilon)

        mean_strength = sum(next_strengths.values()) / len(next_strengths)
        next_strengths = {
            doc_id: strength / mean_strength
            for doc_id, strength in next_strengths.items()
        }

        max_delta = max(
            abs(next_strengths[doc_id] - strengths[doc_id])
            for doc_id in document_ids
        )
        strengths = next_strengths
        if max_delta < tol:
            break

    return {
        doc_id: {
            "strength": strengths[doc_id],
            "log_strength": math.log(strengths[doc_id]),
            "wins": wins[doc_id],
            "losses": losses[doc_id],
            "ties": ties[doc_id],
            "decisive_comparisons": wins[doc_id] + losses[doc_id],
            "total_comparisons": wins[doc_id] + losses[doc_id] + ties[doc_id],
        }
        for doc_id in document_ids
    }


def fit_davidson(
    document_ids: list[str],
    observations: list[dict],
    max_iter: int = 2000,
    epsilon: float = 1e-12,
) -> dict:
    doc_index = {doc_id: index for index, doc_id in enumerate(document_ids)}
    wins = {doc_id: 0 for doc_id in document_ids}
    losses = {doc_id: 0 for doc_id in document_ids}
    ties = {doc_id: 0 for doc_id in document_ids}

    usable_observations = []
    for obs in observations:
        if obs.get("outcome") == "tie":
            doc_1 = obs["contract_1_id"]
            doc_2 = obs["contract_2_id"]
            ties[doc_1] += 1
            ties[doc_2] += 1
            usable_observations.append(("tie", doc_1, doc_2))
        elif obs.get("outcome") == "win":
            winner = obs["winner"]
            loser = obs["loser"]
            wins[winner] += 1
            losses[loser] += 1
            usable_observations.append(("win", winner, loser))

    if not usable_observations:
        return {
            doc_id: {
                "strength": None,
                "log_strength": None,
                "wins": wins[doc_id],
                "losses": losses[doc_id],
                "ties": ties[doc_id],
                "decisive_comparisons": wins[doc_id] + losses[doc_id],
                "total_comparisons": wins[doc_id] + losses[doc_id] + ties[doc_id],
            }
            for doc_id in document_ids
        }

    def negative_log_likelihood(params):
        alphas = params[:-1]
        alphas = alphas - alphas.mean()
        log_tie_param = params[-1]
        nll = 0.0

        for outcome, doc_1, doc_2 in usable_observations:
            alpha_1 = alphas[doc_index[doc_1]]
            alpha_2 = alphas[doc_index[doc_2]]
            log_tie_component = log_tie_param + 0.5 * (alpha_1 + alpha_2)
            max_log_component = max(alpha_1, alpha_2, log_tie_component)
            log_denominator = max_log_component + math.log(
                math.exp(alpha_1 - max_log_component)
                + math.exp(alpha_2 - max_log_component)
                + math.exp(log_tie_component - max_log_component)
                + epsilon
            )

            if outcome == "tie":
                log_probability = log_tie_component - log_denominator
            else:
                log_probability = alpha_1 - log_denominator
            nll -= log_probability

        return nll

    initial_params = [0.0 for _ in document_ids] + [0.0]
    result = minimize(
        negative_log_likelihood,
        initial_params,
        method="BFGS",
        options={"maxiter": max_iter},
    )

    alphas = result.x[:-1]
    alphas = alphas - alphas.mean()
    tie_param = math.exp(result.x[-1])
    strengths = [math.exp(alpha) for alpha in alphas]
    mean_strength = sum(strengths) / len(strengths)
    strengths = [strength / mean_strength for strength in strengths]
    centered_log_strengths = [math.log(strength) for strength in strengths]

    return {
        doc_id: {
            "strength": strengths[doc_index[doc_id]],
            "log_strength": centered_log_strengths[doc_index[doc_id]],
            "wins": wins[doc_id],
            "losses": losses[doc_id],
            "ties": ties[doc_id],
            "decisive_comparisons": wins[doc_id] + losses[doc_id],
            "total_comparisons": wins[doc_id] + losses[doc_id] + ties[doc_id],
            "tie_parameter": tie_param,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
        }
        for doc_id in document_ids
    }


def fit_rankings(document_ids, observations, method):
    if method == "bradley-terry":
        return fit_bradley_terry(document_ids, observations)
    if method == "davidson":
        return fit_davidson(document_ids, observations)
    raise ValueError(f"Unknown method: {method}")


def build_output(results: dict, method: str) -> dict:
    document_ids = document_ids_from_results(results)
    observations = observations_by_category(results)
    rankings_by_category = {
        category: fit_rankings(document_ids, category_observations, method)
        for category, category_observations in observations.items()
    }

    return {
        "method": method,
        "source_model": results.get("model"),
        "agent": results.get("agent"),
        "n_requested_pairs": results.get("n_requested_pairs"),
        "random_seed": results.get("random_seed"),
        "categories": results.get("categories", []),
        "documents": results.get("documents", []),
        "rankings_by_category": rankings_by_category,
        "comparison_counts_by_category": {
            category: {
                "wins_or_losses": sum(
                    1 for obs in category_observations
                    if obs.get("outcome") == "win"
                ),
                "ties": sum(
                    1 for obs in category_observations
                    if obs.get("outcome") == "tie"
                ),
                "total": len(category_observations),
            }
            for category, category_observations in observations.items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit Bradley-Terry or Davidson ELO-style rankings from saved generosity comparisons."
    )
    parser.add_argument(
        "--method",
        choices=["bradley-terry", "davidson"],
        default="davidson",
        help="Ranking model to fit. Bradley-Terry skips ties; Davidson uses ties.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model cache name used to locate default input/output paths.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to runner output JSON with saved comparisons.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write calculated rankings JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_model = model_slug(args.model_name)
    input_path = args.input or (
        Path("cache/06_generosity_output")
        / cache_model
        / "bradley_terry_results.json"
    )
    output_path = args.output or (
        Path("cache/06_generosity_output")
        / cache_model
        / f"{args.method}_rankings.json"
    )

    results = load_results(input_path)
    output = build_output(results, args.method)
    output["source_path"] = input_path.as_posix()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Fitted {args.method} rankings from {input_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
