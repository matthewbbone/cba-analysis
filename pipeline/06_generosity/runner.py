
import asyncio
from itertools import combinations
import json
import math
import random
from pathlib import Path
import sys
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline.utils.llm import LLMClientPool, model_slug
load_dotenv()


def get_category_summary(document: dict, category: str) -> dict:
    for summary in document.get("category_summaries", []):
        if summary.get("category") == category:
            return summary
    return {
        "category": category,
        "provision_count": 0,
        "summary": "",
        "summarization_usage": None,
    }


def process_pairwise_summaries(
    agent: str,
    category: str,
    summary_1: str,
    summary_2: str,
    llm_client: LLMClientPool,
) -> dict:
    if not summary_1.strip() and not summary_2.strip():
        return {
            "more_generous_set": "Equal",
            "usage": None,
        }

    system_prompt = " ".join([
        "You are a legal expert tasked with comparing two provision summaries",
        "from different contracts. Your goal is to identify which set of provisions",
        "is more generous to the specified party. Generosity is determined by which" 
        "set of provisions provides more favorable terms to the specified party in the specified category.",
    ])
    
    prompt = " ".join([
        f"Type of Provisions: {category}",
        f"Set 1 summary:\n{summary_1 or '(No provisions summarized.)'}\n",
        f"Set 2 summary:\n{summary_2 or '(No provisions summarized.)'}\n",
        f"Please review the summaries and determine which set is more generous to the {agent}.",
    ])
    
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "provision_comparison",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "more_generous_set": {
                        "type": "string",
                        "enum": ["Set 1", "Set 2", "Equal"],
                    }
                },
                "required": ["more_generous_set"],
            },
        }
    }
    
    payload, usage = llm_client.call_json(system_prompt, prompt, schema)
    return {
        "more_generous_set": payload.get("more_generous_set"),
        "usage": usage,
    }


def compare_contracts(contract_1_path: str, contract_2_path: str, agent: str, category: str, llm_client: LLMClientPool) -> str:
    
    with open(contract_1_path, "r", encoding="utf-8") as f:
        contract_1 = json.load(f)
    
    with open(contract_2_path, "r", encoding="utf-8") as f:
        contract_2 = json.load(f)
    
    category_summary_1 = get_category_summary(contract_1, category)
    category_summary_2 = get_category_summary(contract_2, category)
    summary_1 = category_summary_1.get("summary", "")
    summary_2 = category_summary_2.get("summary", "")
    
    compare1 = process_pairwise_summaries(
        agent,
        category,
        summary_1,
        summary_2,
        llm_client,
    )
    compare2 = process_pairwise_summaries(
        agent,
        category,
        summary_2,
        summary_1,
        llm_client,
    )
    
    compare1_result = compare1["more_generous_set"]
    compare2_result = compare2["more_generous_set"]

    if compare1_result == "Set 1" and compare2_result == "Set 2":
        score = 1
    elif compare1_result == "Set 2" and compare2_result == "Set 1":
        score = -1
    else:
        score = 0

    return {
        "contract_1_path": contract_1_path,
        "contract_2_path": contract_2_path,
        "agent": agent,
        "category": category,
        "score": score,
        "comparison_objects": {
            "contract_1": {
                "summary": summary_1,
                "provision_count": category_summary_1.get("provision_count", 0),
            },
            "contract_2": {
                "summary": summary_2,
                "provision_count": category_summary_2.get("provision_count", 0),
            },
        },
        "comparisons": {
            "contract_1_vs_contract_2": compare1,
            "contract_2_vs_contract_1": compare2,
        },
    }


def discover_documents(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*/*.json"))


def sample_document_pairs(documents: list[Path], n: int, seed: int) -> list[tuple[Path, Path]]:
    if len(documents) < 2:
        raise ValueError("At least two classified documents are required for pairwise comparison.")

    all_pairs = list(combinations(documents, 2))
    if n > len(all_pairs):
        raise ValueError(
            f"Requested {n} document pairs, but only {len(all_pairs)} unique unordered pairs are available."
        )

    rng = random.Random(seed)
    return rng.sample(all_pairs, n)


def document_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def fit_bradley_terry(
    document_ids: list[str],
    observations: list[dict],
    max_iter: int = 1000,
    tol: float = 1e-10,
    epsilon: float = 1e-9,
) -> dict:
    strengths = {doc_id: 1.0 for doc_id in document_ids}
    wins = {doc_id: 0 for doc_id in document_ids}
    losses = {doc_id: 0 for doc_id in document_ids}

    for obs in observations:
        winner = obs["winner"]
        loser = obs["loser"]
        wins[winner] += 1
        losses[loser] += 1

    if not observations:
        return {
            doc_id: {
                "strength": None,
                "log_strength": None,
                "wins": wins[doc_id],
                "losses": losses[doc_id],
                "decisive_comparisons": wins[doc_id] + losses[doc_id],
            }
            for doc_id in document_ids
        }

    for _ in range(max_iter):
        next_strengths = {}
        for doc_id in document_ids:
            denominator = 0.0
            for obs in observations:
                winner = obs["winner"]
                loser = obs["loser"]
                if doc_id not in (winner, loser):
                    continue
                opponent = loser if doc_id == winner else winner
                denominator += 1.0 / (strengths[doc_id] + strengths[opponent] + epsilon)
            next_strengths[doc_id] = (wins[doc_id] + epsilon) / (denominator + epsilon)

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
            "decisive_comparisons": wins[doc_id] + losses[doc_id],
        }
        for doc_id in document_ids
    }


def sum_comparison_cost(result: dict) -> float:
    total = 0.0
    for comparison in result["comparisons"].values():
        usage = comparison.get("usage")
        if usage:
            total += usage.get("total_cost_usd", 0.0) or 0.0
    return total


async def process_comparison_jobs(
    pairs: list[tuple[Path, Path]],
    categories: list[str],
    agent: str,
    llm_client: LLMClientPool,
    num_workers: int = 8,
) -> list[dict]:
    queue = asyncio.Queue()
    results = []
    errors = []

    for pair_index, (contract_1, contract_2) in enumerate(pairs):
        for category in categories:
            queue.put_nowait((pair_index, contract_1, contract_2, category))

    async def worker(progress: tqdm):
        while True:
            pair_index, contract_1, contract_2, category = await queue.get()
            try:
                result = await asyncio.to_thread(
                    compare_contracts,
                    contract_1.as_posix(),
                    contract_2.as_posix(),
                    agent,
                    category,
                    llm_client,
                )
                result["pair_index"] = pair_index
                result["contract_1_id"] = document_id(contract_1)
                result["contract_2_id"] = document_id(contract_2)
                results.append(result)
            except Exception as exc:
                errors.append((pair_index, contract_1, contract_2, category, exc))
            else:
                progress.update(1)
            finally:
                queue.task_done()

    with tqdm(total=queue.qsize(), desc="Processing comparisons") as progress:
        tasks = [
            asyncio.create_task(worker(progress))
            for _ in range(min(num_workers, queue.qsize()) or 1)
        ]
        await queue.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if errors:
        first_error = errors[0]
        failed_jobs = ", ".join(
            f"pair={pair_index}, category={category}"
            for pair_index, _, _, category, _ in errors[:10]
        )
        raise RuntimeError(
            f"Failed to process {len(errors)} comparison job(s): {failed_jobs}"
        ) from first_error[-1]

    return sorted(results, key=lambda result: (result["pair_index"], result["category"]))


def main():
    N = 300
    RANDOM_SEED = 123
    AGENT = "Worker"
    MODEL_NAME = "gpt-5.4-nano"
    NUM_WORKERS = 8
    MODEL_CACHE_DIR = model_slug(MODEL_NAME)
    INPUT_DIR = Path("cache/05_summarize_output") / MODEL_CACHE_DIR
    OUTPUT_DIR = Path("cache/06_generosity_output") / MODEL_CACHE_DIR
    
    with open("references/provision_taxonomy.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
        
    CATEGORIES = [c["name"] for c in taxonomy.get("categories", [])]
    documents = discover_documents(INPUT_DIR)
    document_ids = [document_id(doc) for doc in documents]
    pairs = sample_document_pairs(documents, N, RANDOM_SEED)
    llm_client = LLMClientPool(MODEL_NAME, size=NUM_WORKERS)

    observations_by_category = {category: [] for category in CATEGORIES}
    total_cost_usd = 0.0

    comparisons = asyncio.run(
        process_comparison_jobs(
            pairs,
            CATEGORIES,
            AGENT,
            llm_client,
            num_workers=NUM_WORKERS,
        )
    )

    for result in comparisons:
        total_cost_usd += sum_comparison_cost(result)

        if result["score"] == 1:
            observations_by_category[result["category"]].append(
                {
                    "winner": result["contract_1_id"],
                    "loser": result["contract_2_id"],
                    "pair_index": result["pair_index"],
                }
            )
        elif result["score"] == -1:
            observations_by_category[result["category"]].append(
                {
                    "winner": result["contract_2_id"],
                    "loser": result["contract_1_id"],
                    "pair_index": result["pair_index"],
                }
            )

    rankings_by_category = {
        category: fit_bradley_terry(document_ids, observations)
        for category, observations in observations_by_category.items()
    }

    output = {
        "agent": AGENT,
        "model": MODEL_NAME,
        "n_requested_pairs": N,
        "random_seed": RANDOM_SEED,
        "num_workers": NUM_WORKERS,
        "categories": CATEGORIES,
        "documents": [
            {"document_id": document_id(doc), "path": doc.as_posix()}
            for doc in documents
        ],
        "comparisons": comparisons,
        "rankings_by_category": rankings_by_category,
        "total_cost_usd": total_cost_usd,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "bradley_terry_results.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()
