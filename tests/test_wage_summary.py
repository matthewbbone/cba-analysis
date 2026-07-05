import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "validation" / "wage_summary.py"
MODULE_SPEC = importlib.util.spec_from_file_location("wage_summary", MODULE_PATH)
WAGE_SUMMARY = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
sys.modules["wage_summary"] = WAGE_SUMMARY
MODULE_SPEC.loader.exec_module(WAGE_SUMMARY)


def test_summary_path_normalization() -> None:
    path = Path(
        "cache/05_summarize_output/gpt_5_4_nano/cornell_dol/6018ABBYY_res.json"
    )

    doc = WAGE_SUMMARY.summary_document_from_path(path)

    assert doc.source == "Cornell_DoL"
    assert doc.filename == "6018ABBYY.pdf"
    assert doc.document_key == "Cornell_DoL|6018ABBYY.pdf"
    assert doc.cache_stem == "6018ABBYY"


def test_unknown_summary_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown summary source"):
        WAGE_SUMMARY.summary_document_from_path(
            Path("cache/05_summarize_output/gpt_5_4_nano/other/doc_res.json")
        )


def test_extract_compensation_summary() -> None:
    payload = {
        "category_summaries": [
            {"category": "Benefits", "provision_count": 2, "summary": "Benefits."},
            {
                "category": "Compensation",
                "provision_count": "3",
                "summary": "Journeymen are paid $25.00 per hour.",
            },
        ]
    }

    compensation = WAGE_SUMMARY.extract_compensation_summary(payload)

    assert compensation == {
        "found": True,
        "summary": "Journeymen are paid $25.00 per hour.",
        "provision_count": 3,
    }


def test_join_summaries_to_metadata_handles_duplicates_and_unmatched() -> None:
    docs = [
        WAGE_SUMMARY.SummaryDocument(
            path=Path("root/dol_archive/document_10_res.json"),
            source="DoL",
            filename="document_10.pdf",
            document_key="DoL|document_10.pdf",
            cache_stem="document_10",
        ),
        WAGE_SUMMARY.SummaryDocument(
            path=Path("root/dol_archive/document_99_res.json"),
            source="DoL",
            filename="document_99.pdf",
            document_key="DoL|document_99.pdf",
            cache_stem="document_99",
        ),
    ]
    metadata = pd.DataFrame(
        [
            {
                "metadata_row_id": 0,
                "source": "DoL",
                "filename": "document_10.pdf",
                "document_key": "DoL|document_10.pdf",
                "cba_id": "DoL_10",
                "state_fips": "06",
                "naics": "336412",
            },
            {
                "metadata_row_id": 1,
                "source": "DoL",
                "filename": "document_10.pdf",
                "document_key": "DoL|document_10.pdf",
                "cba_id": "DoL_10",
                "state_fips": "34",
                "naics": "92119",
            },
        ]
    )
    for column in WAGE_SUMMARY.METADATA_COLUMNS_TO_KEEP:
        if column not in metadata.columns:
            metadata[column] = pd.NA

    matched, unmatched = WAGE_SUMMARY.join_summaries_to_metadata(docs, metadata)

    assert len(matched) == 2
    assert set(matched["state_fips"]) == {"06", "34"}
    assert unmatched["filename"].tolist() == ["document_99.pdf"]


def test_active_years_prefers_full_date_range_and_falls_back_on_incomplete_dates() -> None:
    assert WAGE_SUMMARY.active_years_for_contract(
        "2020-05-01", "2022-04-30", None
    ) == [2020, 2021, 2022]
    assert WAGE_SUMMARY.active_years_for_contract("2020-05-01", None, "2021") == [
        2021
    ]
    assert WAGE_SUMMARY.active_years_for_contract("2020-05-01", "2019-04-30", None) == [
        2020
    ]
    assert WAGE_SUMMARY.active_years_for_contract(None, None, None) == []


def test_panel_rate_values_exclude_premiums_and_differentials() -> None:
    extraction = {
        "mean_hourly_wage": 999,
        "rates": [
            {
                "rate_type": "base_wage",
                "hourly_equivalent": 20,
            },
            {
                "rate_type": "minimum_wage",
                "hourly_equivalent": 18,
            },
            {
                "rate_type": "differential",
                "hourly_equivalent": 0.75,
            },
            {
                "rate_type": "premium",
                "hourly_equivalent": 40,
            },
        ],
    }

    summary = WAGE_SUMMARY.summarize_panel_rates(extraction)

    assert summary["has_panel_wage"] is True
    assert summary["panel_rate_count"] == 2
    assert summary["panel_min_hourly_wage"] == 18
    assert summary["panel_mean_hourly_wage"] == 19
    assert summary["panel_max_hourly_wage"] == 20


def test_build_contract_and_rate_rows_from_cached_extractions() -> None:
    matched = pd.DataFrame(
        [
            {
                "metadata_row_id": 7,
                "document_key": "DoL|document_1.pdf",
                "source": "DoL",
                "filename": "document_1.pdf",
                "summary_path": "cache/document_1_res.json",
                "cba_id": "DoL_1",
                "state_name": "California",
                "state_fips": "6",
                "naics": "336412.0",
                "naics_description": "Aircraft manufacturing",
                "effective_date": "2020-01-01",
                "expiration_date": "2021-12-31",
                "mistral_contract_year": pd.NA,
                "n_workers": "10",
            }
        ]
    )
    for column in WAGE_SUMMARY.METADATA_COLUMNS_TO_KEEP:
        if column not in matched.columns:
            matched[column] = pd.NA
    cached = {
        "DoL|document_1.pdf": {
            "model": "gpt-5.4-nano",
            "compensation_provision_count": 2,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "extraction": {
                "has_wage_info": True,
                "currency": "USD",
                "wage_basis": "hourly",
                "summary": "Base rates are stated.",
                "min_hourly_wage": 18,
                "mean_hourly_wage": 19,
                "max_hourly_wage": 20,
                "confidence": "high",
                "notes": "",
                "rates": [
                    {
                        "amount": 20,
                        "basis": "hourly",
                        "hourly_equivalent": 20,
                        "rate_type": "base_wage",
                        "classification": "Journeyman",
                        "effective_date": None,
                        "description": "$20 hourly base",
                        "evidence": "$20 per hour",
                    },
                    {
                        "amount": 0.75,
                        "basis": "hourly",
                        "hourly_equivalent": 0.75,
                        "rate_type": "differential",
                        "classification": None,
                        "effective_date": None,
                        "description": "shift differential",
                        "evidence": "$0.75 per hour shift differential",
                    },
                ],
            },
        }
    }

    extracts = WAGE_SUMMARY.build_contract_extract_rows(matched, cached)
    rates = WAGE_SUMMARY.build_rate_rows(matched, cached)

    assert extracts.iloc[0]["state_fips"] == "06"
    assert extracts.iloc[0]["naics"] == "336412"
    assert extracts.iloc[0]["panel_mean_hourly_wage"] == 20
    assert extracts.iloc[0]["panel_rate_count"] == 1

    assert rates["included_in_panel_wage"].tolist() == [True, False]
    assert rates["rate_type"].tolist() == ["base_wage", "differential"]


def test_year_state_industry_panel_aggregates_unweighted_and_worker_weighted() -> None:
    extracts = pd.DataFrame(
        [
            {
                "document_key": "a",
                "state_fips": "06",
                "state_name": "California",
                "naics": "336412",
                "naics_description": "Aircraft manufacturing",
                "effective_date": "2020-01-01",
                "expiration_date": "2021-12-31",
                "mistral_contract_year": pd.NA,
                "n_workers": 10,
                "panel_mean_hourly_wage": 20,
                "panel_min_hourly_wage": 15,
                "panel_max_hourly_wage": 25,
                "panel_rate_count": 2,
                "has_panel_wage": True,
            },
            {
                "document_key": "b",
                "state_fips": "06",
                "state_name": "California",
                "naics": "336412",
                "naics_description": "Aircraft manufacturing",
                "effective_date": "2020-01-01",
                "expiration_date": "2020-12-31",
                "mistral_contract_year": pd.NA,
                "n_workers": 30,
                "panel_mean_hourly_wage": 30,
                "panel_min_hourly_wage": 28,
                "panel_max_hourly_wage": 35,
                "panel_rate_count": 1,
                "has_panel_wage": True,
            },
            {
                "document_key": "c",
                "state_fips": "06",
                "state_name": "California",
                "naics": "336412",
                "naics_description": "Aircraft manufacturing",
                "effective_date": None,
                "expiration_date": None,
                "mistral_contract_year": 2020,
                "n_workers": 5,
                "panel_mean_hourly_wage": None,
                "panel_min_hourly_wage": None,
                "panel_max_hourly_wage": None,
                "panel_rate_count": 0,
                "has_panel_wage": False,
            },
            {
                "document_key": "d",
                "state_fips": None,
                "state_name": None,
                "naics": "336412",
                "naics_description": "Aircraft manufacturing",
                "effective_date": "2020-01-01",
                "expiration_date": "2020-12-31",
                "mistral_contract_year": pd.NA,
                "n_workers": 100,
                "panel_mean_hourly_wage": 99,
                "panel_min_hourly_wage": 99,
                "panel_max_hourly_wage": 99,
                "panel_rate_count": 1,
                "has_panel_wage": True,
            },
        ]
    )

    panel = WAGE_SUMMARY.build_year_state_industry_panel(extracts)

    ca_2020 = panel[(panel["year"] == 2020) & (panel["state_fips"] == "06")].iloc[0]
    ca_2021 = panel[(panel["year"] == 2021) & (panel["state_fips"] == "06")].iloc[0]

    assert ca_2020["contract_count"] == 3
    assert ca_2020["contracts_with_wage_count"] == 2
    assert ca_2020["rate_count"] == 3
    assert ca_2020["mean_hourly_wage_unweighted"] == pytest.approx(25)
    assert ca_2020["median_hourly_wage_unweighted"] == pytest.approx(25)
    assert ca_2020["min_hourly_wage"] == pytest.approx(15)
    assert ca_2020["max_hourly_wage"] == pytest.approx(35)
    assert ca_2020["mean_hourly_wage_worker_weighted"] == pytest.approx(27.5)
    assert ca_2020["worker_weighted_contract_count"] == 2
    assert ca_2020["total_represented_workers"] == pytest.approx(45)
    assert ca_2020["represented_workers_with_wage"] == pytest.approx(40)

    assert ca_2021["contract_count"] == 1
    assert ca_2021["mean_hourly_wage_unweighted"] == pytest.approx(20)
