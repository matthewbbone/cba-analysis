import asyncio
from pathlib import Path

from dotenv import load_dotenv
import gabriel
import pandas as pd

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).parent
TEXT_DIR = BASE_DIR / "test_text_health"
RUNS_DIR = BASE_DIR / "runs"
OUTPUT_CSV = BASE_DIR / "health_generosity_results.csv"

RATE_ATTRIBUTES = {
    "overall_generosity": "How generous the healthcare benefits are overall for workers",
    "employer_cost_share": "What percentage of healthcare costs are covered by the employer",
    "coverage_breadth": "How broad is the coverage (e.g. types of services covered)",
    "dependent_coverage": "Does the healthcare plan cover dependents (e.g. children, spouses)?",
    "additional_benefits": "Are there any additional healthcare benefits provided (e.g. wellness programs, mental health support)?",
}

MODEL = "gpt-5-nano"
N_RUNS = 3


def load_texts() -> pd.DataFrame:
    rows = []
    for f in sorted(TEXT_DIR.glob("CBA_*.txt")):
        cba_id = f.stem  # e.g. "CBA_01"
        text = f.read_text(encoding="utf-8")
        rows.append({"cba_id": cba_id, "text": text})
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} CBA text files")
    return df


async def run_rate(df: pd.DataFrame) -> pd.DataFrame:
    rate_dir = str(RUNS_DIR / "rate_health")
    result = await gabriel.rate(
        df,
        column_name="text",
        attributes=RATE_ATTRIBUTES,
        save_dir=rate_dir,
        model=MODEL,
        n_runs=N_RUNS,
        modality="text",
        reset_files=True,
    )
    print(f"Rating complete — {len(result)} rows, attributes: {list(RATE_ATTRIBUTES.keys())}")
    return result


RANK_ATTRIBUTES = {
    "employer_cost_share": "What percentage of healthcare costs are covered by the employer",
    "coverage_breadth": "How broad is the coverage (e.g. types of services covered)",
    "dependent_coverage": "Does the healthcare plan cover dependents (e.g. children, spouses)?",
    "additional_benefits": "Are there any additional healthcare benefits provided (e.g. wellness programs, mental health support)?",
}


async def run_rank_per_attribute(df: pd.DataFrame) -> pd.DataFrame:
    """Run gabriel.rank once per dimension attribute, return merged results."""
    merged = df[["cba_id"]].copy()

    for attr_name, attr_desc in RANK_ATTRIBUTES.items():
        print(f"  Ranking attribute: {attr_name}")
        rank_dir = str(RUNS_DIR / f"rank_health_{attr_name}")
        result = await gabriel.rank(
            df,
            column_name="text",
            attributes={attr_name: attr_desc},
            save_dir=rank_dir,
            model=MODEL,
            n_rounds=5,
            matches_per_round=3,
            modality="text",
            reset_files=True,
            id_column="cba_id",
            n_parallels=50,
        )
        # result has cba_id + attr_name (z-score) columns
        merged = merged.merge(
            result[["cba_id", attr_name]].rename(columns={attr_name: f"rank_z_{attr_name}"}),
            on="cba_id",
            how="left",
        )
        print(f"    Done — {len(result)} rows")

    print(f"All attribute rankings complete")
    return merged


async def main():
    df = load_texts()

    rate_result = await run_rate(df)
    rank_result = await run_rank_per_attribute(df)

    # Build merged summary with column names matching healthcare_generosity_scores.csv
    col_rename = {
        "cba_id": "contract_id",
        "employer_cost_share": "score_employer_cost",
        "coverage_breadth": "score_coverage_breadth",
        "dependent_coverage": "score_dependent",
        "additional_benefits": "score_extras",
        "overall_generosity": "score_overall",
    }
    summary = rate_result[["cba_id"] + list(RATE_ATTRIBUTES.keys())].copy()
    summary = summary.rename(columns=col_rename)

    # Composite rating score (mean of raw 0-100 scores) for reference
    dimension_cols = [
        "score_employer_cost",
        "score_coverage_breadth",
        "score_dependent",
        "score_extras",
    ]
    summary["composite_score"] = summary[dimension_cols].mean(axis=1).round(1)

    # Merge per-attribute pairwise ranking z-scores
    rank_z_rename = {
        "rank_z_employer_cost_share": "rank_z_employer_cost",
        "rank_z_coverage_breadth": "rank_z_coverage_breadth",
        "rank_z_dependent_coverage": "rank_z_dependent",
        "rank_z_additional_benefits": "rank_z_extras",
    }
    rank_result = rank_result.rename(columns={"cba_id": "contract_id", **rank_z_rename})
    summary = summary.merge(rank_result, on="contract_id", how="left")

    # Per-attribute ranks from z-scores (higher z = more generous = rank 1)
    rank_z_cols = list(rank_z_rename.values())
    for z_col in rank_z_cols:
        rank_col = z_col.replace("rank_z_", "rank_")
        summary[rank_col] = summary[z_col].rank(ascending=False, method="average").astype(float)

    # Composite rank: average of per-attribute ranks, then re-rank
    attr_rank_cols = [z.replace("rank_z_", "rank_") for z in rank_z_cols]
    summary["avg_dim_rank"] = summary[attr_rank_cols].mean(axis=1).round(2)
    summary["rank"] = summary["avg_dim_rank"].rank(method="min").astype(int)
    summary = summary.sort_values("rank")

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
