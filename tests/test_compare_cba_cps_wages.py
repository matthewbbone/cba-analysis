import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "validation" / "compare_cba_cps_wages.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("compare_cba_cps_wages", MODULE_PATH)
COMPARE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
sys.modules["compare_cba_cps_wages"] = COMPARE
MODULE_SPEC.loader.exec_module(COMPARE)


def write_wage_scale_extraction(
    root: Path,
    source: str,
    document: str,
    base_wages: dict[str, dict[str, object]],
    time_unit: str | None,
) -> None:
    path = root / source / f"{document}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": source,
                "document": document,
                "wage_scale": {
                    "base_wages": base_wages,
                    "time_unit": time_unit,
                    "cola_clause": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_cba_panel_from_wage_scale_extractions(tmp_path: Path) -> None:
    extraction_dir = tmp_path / "wage_scale_extractions"
    write_wage_scale_extraction(
        extraction_dir,
        "cornell_dol",
        "doc_a",
        {
            "occupation=a": {"mid": "20"},
            "occupation=b": {"min": "10", "max": "30"},
            "occupation=c": {"max": "26"},
        },
        "hour",
    )
    write_wage_scale_extraction(
        extraction_dir,
        "dol_archive",
        "document_5",
        {"all_workers": {"mid": "52000"}},
        "year",
    )
    write_wage_scale_extraction(
        extraction_dir,
        "cornell_dol",
        "doc_without_base_wages",
        {},
        "hour",
    )

    metadata = pd.DataFrame(
        [
            {
                "source": "Cornell_DoL",
                "filename": "doc_a.pdf",
                "state_fips": "06",
                "state_name": "California",
                "naics": "236220",
                "naics_description": "Construction",
                "effective_date": "2020-01-01",
                "expiration_date": "2021-12-31",
                "mistral_contract_year": None,
            },
            {
                "source": "DoL",
                "filename": "document_5.pdf",
                "state_fips": "36",
                "state_name": "New York",
                "naics": "611110",
                "naics_description": "Schools",
                "effective_date": None,
                "expiration_date": None,
                "mistral_contract_year": 2020,
            },
            {
                "source": "Cornell_DoL",
                "filename": "doc_without_base_wages.pdf",
                "state_fips": "06",
                "state_name": "California",
                "naics": "236220",
                "naics_description": "Construction",
                "effective_date": "2020-01-01",
                "expiration_date": "2020-12-31",
                "mistral_contract_year": None,
            },
        ]
    )

    panel = COMPARE.build_cba_panel_from_wage_scale_extractions(
        metadata,
        extraction_dir,
    )

    doc_a = panel[panel["document"] == "doc_a"].sort_values("year")
    document_5 = panel[panel["document"] == "document_5"].iloc[0]

    assert doc_a["year"].tolist() == [2020, 2021]
    assert doc_a["mean_wage_rate_value"].iloc[0] == pytest.approx(22)
    assert doc_a["mean_hourly_wage_unweighted"].iloc[0] == pytest.approx(22)
    assert doc_a["mean_weekly_wage"].iloc[0] == pytest.approx(880)
    assert doc_a["rate_count"].iloc[0] == 3

    assert document_5["year"] == 2020
    assert document_5["mean_wage_rate_value"] == pytest.approx(52000)
    assert document_5["mean_hourly_wage_unweighted"] == pytest.approx(25)
    assert document_5["mean_weekly_wage"] == pytest.approx(1000)
    assert "doc_without_base_wages" not in set(panel["document"])


def test_filter_by_min_cps_unweighted_n() -> None:
    merged = pd.DataFrame(
        {
            "cps_unweighted_n": [9, 10, 11, None],
            "value": ["drop", "keep_10", "keep_11", "drop_null"],
        }
    )

    filtered = COMPARE.filter_by_min_cps_unweighted_n(merged, 10)

    assert filtered["value"].tolist() == ["keep_10", "keep_11"]


def test_merge_state_year_panels_and_median_absolute_error_summary() -> None:
    cps = pd.DataFrame(
        [
            {
                "year": 2020,
                "statefip": 6,
                "state_name": "California",
                "ind1950": 246,
                "industry_name": "Construction",
                "mean_weekly_earnings_nominal": 1000,
                "union_covered_weighted_workers": 1,
                "union_covered_unweighted_n": 1,
                "nonunion_mean_weekly_earnings_nominal": 600,
                "nonunion_weighted_workers": 2,
                "nonunion_unweighted_n": 2,
            },
            {
                "year": 2020,
                "statefip": 6,
                "state_name": "California",
                "ind1950": 336,
                "industry_name": "Manufacturing",
                "mean_weekly_earnings_nominal": 500,
                "union_covered_weighted_workers": 3,
                "union_covered_unweighted_n": 2,
            },
            {
                "year": 2020,
                "statefip": 36,
                "state_name": "New York",
                "ind1950": 246,
                "industry_name": "Construction",
                "mean_weekly_earnings_nominal": 800,
                "union_covered_weighted_workers": 4,
                "union_covered_unweighted_n": 3,
            },
        ]
    )
    cba = pd.DataFrame(
        [
            {
                "year": 2020,
                "state_fips": "06",
                "state_name": "California",
                "naics": "236220",
                "mean_hourly_wage_unweighted": 10,
                "contracts_with_wage_count": 1,
                "rate_count": 2,
            },
            {
                "year": 2020,
                "state_fips": "06",
                "state_name": "California",
                "naics": "236220",
                "mean_hourly_wage_unweighted": 50,
                "contracts_with_wage_count": 1,
                "rate_count": 2,
            },
            {
                "year": 2020,
                "state_fips": "06",
                "state_name": "California",
                "naics": "336412",
                "mean_hourly_wage_unweighted": 90,
                "contracts_with_wage_count": 1,
                "rate_count": 3,
            },
            {
                "year": 2020,
                "state_fips": "36",
                "state_name": "New York",
                "naics": "44511",
                "mean_hourly_wage_unweighted": 2600,
                "contracts_with_wage_count": 1,
                "rate_count": 1,
            },
        ]
    )

    minimum_wages = pd.DataFrame(
        [
            {
                "year": 2020,
                "statefip": 6,
                "minimum_wage_hourly": 15,
            },
            {
                "year": 2020,
                "statefip": 36,
                "minimum_wage_hourly": 10,
            },
        ]
    )

    merged = COMPARE.merge_state_year_panels(cps, cba, minimum_wages=minimum_wages)
    summary = COMPARE.summarize_median_absolute_error(merged)

    assert "cba_nonunion_percent_difference_40h" not in merged.columns

    california = merged[merged["statefip"] == 6].iloc[0]
    new_york = merged[merged["statefip"] == 36].iloc[0]

    assert california["cps_mean_weekly_earnings_nominal"] == pytest.approx(625)
    assert california["cba_mean_hourly_wage_unweighted"] == pytest.approx(60)
    assert california["cba_mean_weekly_wage_40h"] == pytest.approx(2400)
    assert california["cba_industry_cells"] == 2
    assert california["cba_contracts_with_wage"] == 3
    assert california["cba_rate_count"] == 7
    assert california["absolute_error_weekly_40h"] == pytest.approx(1775)
    assert california["cps_mean_hourly_wage_40h"] == pytest.approx(15.625)
    assert california["absolute_percentage_error_weekly_40h"] == pytest.approx(284)
    assert california["absolute_percentage_error_hourly_40h"] == pytest.approx(284)
    assert california["minimum_wage_hourly"] == pytest.approx(15)
    assert california["minimum_wage_weekly_40h"] == pytest.approx(600)
    assert california["cba_minimum_wage_hourly_difference"] == pytest.approx(45)
    assert california["cba_minimum_wage_weekly_difference_40h"] == pytest.approx(1800)
    assert california["cba_minimum_wage_percent_difference"] == pytest.approx(300)
    assert california["plausible_cba_hourly"] == pytest.approx(True)

    assert new_york["absolute_error_weekly_40h"] == pytest.approx(103200)
    assert new_york["absolute_percentage_error_weekly_40h"] == pytest.approx(12900)
    assert new_york["minimum_wage_hourly"] == pytest.approx(10)
    assert new_york["cba_minimum_wage_percent_difference"] == pytest.approx(25900)
    assert new_york["plausible_cba_hourly"] == pytest.approx(False)

    raw = summary[summary["comparison"] == "raw_state_year_40h"].iloc[0]
    plausible = summary[
        summary["comparison"] == "plausible_state_year_40h_hourly_1_100"
    ].iloc[0]

    assert raw["merged_rows"] == 2
    assert raw["median_absolute_error_weekly_40h"] == pytest.approx(
        (1775 + 103200) / 2
    )
    assert raw["median_absolute_percentage_error_weekly_40h"] == pytest.approx(
        (284 + 12900) / 2
    )
    assert raw["median_absolute_percentage_error_hourly_40h"] == pytest.approx(
        (284 + 12900) / 2
    )
    assert raw["pearson_correlation"] == pytest.approx(1)
    assert raw["spearman_correlation"] == pytest.approx(1)
    assert plausible["merged_rows"] == 1
    assert plausible["median_absolute_error_weekly_40h"] == pytest.approx(1775)
    assert plausible["median_absolute_percentage_error_weekly_40h"] == pytest.approx(
        284
    )
    assert pd.isna(plausible["pearson_correlation"])
    assert pd.isna(plausible["spearman_correlation"])


def test_crosswalked_industry_merge_and_median_absolute_error_summary() -> None:
    cps = pd.DataFrame(
        [
            {
                "year": 2020,
                "statefip": 6,
                "state_name": "California",
                "ind1950": 246,
                "industry_name": "Construction",
                "mean_weekly_earnings_nominal": 1000,
                "union_covered_weighted_workers": 1,
                "union_covered_unweighted_n": 1,
                "nonunion_mean_weekly_earnings_nominal": 600,
                "nonunion_weighted_workers": 2,
                "nonunion_unweighted_n": 2,
            },
            {
                "year": 2020,
                "statefip": 6,
                "state_name": "California",
                "ind1950": 377,
                "industry_name": "Aircraft and parts",
                "mean_weekly_earnings_nominal": 1200,
                "union_covered_weighted_workers": 1,
                "union_covered_unweighted_n": 1,
                "nonunion_mean_weekly_earnings_nominal": 1000,
                "nonunion_weighted_workers": 1,
                "nonunion_unweighted_n": 1,
            },
            {
                "year": 2020,
                "statefip": 36,
                "state_name": "New York",
                "ind1950": 636,
                "industry_name": "Food stores",
                "mean_weekly_earnings_nominal": 800,
                "union_covered_weighted_workers": 1,
                "union_covered_unweighted_n": 1,
                "nonunion_mean_weekly_earnings_nominal": None,
                "nonunion_weighted_workers": 0,
                "nonunion_unweighted_n": 0,
            },
        ]
    )
    cba = pd.DataFrame(
        [
            {
                "year": 2020,
                "state_fips": "06",
                "state_name": "California",
                "naics": "236220",
                "mean_hourly_wage_unweighted": 20,
                "contracts_with_wage_count": 1,
                "rate_count": 2,
            },
            {
                "year": 2020,
                "state_fips": "06",
                "state_name": "California",
                "naics": "336412",
                "mean_hourly_wage_unweighted": 30,
                "contracts_with_wage_count": 1,
                "rate_count": 3,
            },
            {
                "year": 2020,
                "state_fips": "36",
                "state_name": "New York",
                "naics": "44511",
                "mean_hourly_wage_unweighted": 2600,
                "contracts_with_wage_count": 1,
                "rate_count": 1,
            },
        ]
    )

    merged = COMPARE.merge_state_year_industry_panels(cps, cba)
    summary = COMPARE.summarize_median_absolute_error(
        merged,
        raw_label="raw_state_year_industry_40h",
        plausible_label_prefix="plausible_state_year_industry_40h",
    )

    construction = merged[merged["industry_group"] == "23_construction"].iloc[0]
    manufacturing = merged[merged["industry_group"] == "31_33_manufacturing"].iloc[0]
    retail = merged[merged["industry_group"] == "44_45_retail"].iloc[0]

    assert construction["industry_group_name"] == "Construction"
    assert construction["absolute_error_weekly_40h"] == pytest.approx(200)
    assert construction["absolute_percentage_error_weekly_40h"] == pytest.approx(20)
    assert construction["cps_nonunion_mean_weekly_earnings_nominal"] == pytest.approx(
        600
    )
    assert construction["cps_nonunion_mean_hourly_wage_40h"] == pytest.approx(15)
    assert construction["cba_nonunion_weekly_difference_40h"] == pytest.approx(200)
    assert construction["cba_nonunion_hourly_difference_40h"] == pytest.approx(5)
    assert construction["cba_nonunion_percent_difference_40h"] == pytest.approx(
        100 * (800 - 600) / 600
    )
    assert manufacturing["absolute_error_weekly_40h"] == pytest.approx(0)
    assert manufacturing["absolute_percentage_error_weekly_40h"] == pytest.approx(0)
    assert manufacturing["cba_nonunion_percent_difference_40h"] == pytest.approx(20)
    assert retail["absolute_error_weekly_40h"] == pytest.approx(103200)
    assert retail["absolute_percentage_error_weekly_40h"] == pytest.approx(12900)
    assert pd.isna(retail["cba_nonunion_percent_difference_40h"])
    assert retail["plausible_cba_hourly"] == pytest.approx(False)

    raw = summary[summary["comparison"] == "raw_state_year_industry_40h"].iloc[0]
    plausible = summary[
        summary["comparison"] == "plausible_state_year_industry_40h_hourly_1_100"
    ].iloc[0]

    assert raw["merged_rows"] == 3
    assert raw["median_absolute_error_weekly_40h"] == pytest.approx(200)
    assert raw["median_absolute_percentage_error_weekly_40h"] == pytest.approx(20)
    assert raw["pearson_correlation"] == pytest.approx(
        pd.Series([800, 1200, 104000]).corr(pd.Series([1000, 1200, 800]))
    )
    assert raw["spearman_correlation"] == pytest.approx(
        pd.Series([800, 1200, 104000]).corr(
            pd.Series([1000, 1200, 800]),
            method="spearman",
        )
    )
    assert plausible["merged_rows"] == 2
    assert plausible["median_absolute_error_weekly_40h"] == pytest.approx(100)
    assert plausible["median_absolute_percentage_error_weekly_40h"] == pytest.approx(
        10
    )
    assert plausible["pearson_correlation"] == pytest.approx(1)
    assert plausible["spearman_correlation"] == pytest.approx(1)


def test_compare_cba_cps_labels_targeted_and_main_pipeline_sources(
    tmp_path: Path,
    capsys,
) -> None:
    cps_panel = tmp_path / "cps.csv"
    cba_panel = tmp_path / "cba_extracts.csv"
    wage_scale_dir = tmp_path / "wage_scale_extractions"
    main_pipeline_doclevel = tmp_path / "main_pipeline_doclevel.csv"
    minimum_wage_panel = tmp_path / "annual_minimum_wage.csv"
    output = tmp_path / "state_year.csv"
    summary_output = tmp_path / "state_year_summary.csv"
    industry_output = tmp_path / "state_year_industry.csv"
    industry_summary_output = tmp_path / "state_year_industry_summary.csv"

    pd.DataFrame(
        [
            {
                "year": 2020,
                "statefip": 6,
                "state_name": "California",
                "ind1950": 246,
                "industry_name": "Construction",
                "mean_weekly_earnings_nominal": 1000,
                "union_covered_weighted_workers": 1,
                "union_covered_unweighted_n": 1,
            }
        ]
    ).to_csv(cps_panel, index=False)
    pd.DataFrame(
        [
            {
                "source": "DoL",
                "filename": "document_5.pdf",
                "state_fips": "06",
                "state_name": "California",
                "naics": "236220",
                "naics_description": "Construction",
                "effective_date": "2020-01-01",
                "expiration_date": "2020-12-31",
                "mistral_contract_year": None,
            }
        ]
    ).to_csv(cba_panel, index=False)
    write_wage_scale_extraction(
        wage_scale_dir,
        "dol_archive",
        "document_5",
        {"all_workers": {"mid": "20"}},
        "hour",
    )
    pd.DataFrame(
        [
            {
                "document_id": 5,
                "employer": "Main Pipeline Employer",
                "state": "CA",
                "naics": "236220",
                "naics_description": None,
                "harmonized_sector": "Construction",
                "contract_year": 2020,
                "n_occupations": 1,
                "entry_wage_usd_hr": 30,
                "mean_wage_usd_hr": 30,
                "median_wage_usd_hr": 30,
                "top_wage_usd_hr": 30,
                "data_quality_flag": None,
            }
        ]
    ).to_csv(main_pipeline_doclevel, index=False)
    pd.DataFrame(
        [
            {
                "observation_date": "2020-01-01",
                "STTMINWGCA": 15,
            }
        ]
    ).to_csv(minimum_wage_panel, index=False)

    result = COMPARE.compare_cba_cps(
        cps_panel=cps_panel,
        cba_panel=cba_panel,
        wage_scale_extractions_dir=wage_scale_dir,
        main_pipeline_doclevel=main_pipeline_doclevel,
        minimum_wage_panel=minimum_wage_panel,
        output=output,
        summary_output=summary_output,
        industry_output=industry_output,
        industry_summary_output=industry_summary_output,
    )

    assert set(result.merged["extraction_source"]) == {
        "targeted extraction",
        "main pipeline",
    }
    assert set(result.industry_merged["extraction_source"]) == {
        "targeted extraction",
        "main pipeline",
    }
    assert set(result.summary["extraction_source"]) == {
        "targeted extraction",
        "main pipeline",
    }
    assert output.exists()
    output_columns = set(pd.read_csv(output).columns)
    summary_columns = set(pd.read_csv(summary_output).columns)
    assert "extraction_source" in output_columns
    assert "absolute_percentage_error_hourly_40h" in output_columns
    assert "cba_minimum_wage_percent_difference" in output_columns
    assert "median_absolute_percentage_error_hourly_40h" in summary_columns

    minimum_wage_summary = COMPARE.format_premium_summary_table(
        "State-year median CBA premium over state minimum wage",
        COMPARE.summarize_state_year_minimum_wage_premiums(result.merged),
        "Median premium $/hr",
        "Median premium %",
    )
    nonunion_summary = COMPARE.format_premium_summary_table(
        "State-year-industry median CBA premium over CPS non-union wages",
        COMPARE.summarize_state_year_industry_nonunion_premiums(
            result.industry_merged
        ),
        "Median premium $/hr",
        "Median premium %",
    )
    assert "state minimum wage" in minimum_wage_summary
    assert "CPS non-union wages" in nonunion_summary
    assert "Median premium $/hr" in minimum_wage_summary
    assert "Median premium %" in nonunion_summary

    args = SimpleNamespace(
        output=output,
        summary_output=summary_output,
        industry_output=industry_output,
        industry_summary_output=industry_summary_output,
        hours_per_week=40,
        min_plausible_hourly=1,
        max_plausible_hourly=100,
        min_cps_unweighted_n=None,
    )
    COMPARE.print_run_summary(args, result)
    captured = capsys.readouterr()
    assert "State-year median CBA premium over state minimum wage" in captured.out
    assert (
        "State-year-industry median CBA premium over CPS non-union wages"
        in captured.out
    )
    assert "Median vs union $/hr" not in captured.out
