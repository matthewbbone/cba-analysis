import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "validation" / "cps_wages.py"
MODULE_SPEC = importlib.util.spec_from_file_location("cps_wages", MODULE_PATH)
CPS_WAGES = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
sys.modules["cps_wages"] = CPS_WAGES
MODULE_SPEC.loader.exec_module(CPS_WAGES)

MIN_UNION_ASEC_YEAR = CPS_WAGES.MIN_UNION_ASEC_YEAR
aggregate_union_wages = CPS_WAGES.aggregate_union_wages
build_industry_time_series = CPS_WAGES.build_industry_time_series
build_state_time_series = CPS_WAGES.build_state_time_series
extract_value_labels = CPS_WAGES.extract_value_labels
read_sample_ids = CPS_WAGES.read_sample_ids
run_figures_only = CPS_WAGES.run_figures_only
select_usable_samples = CPS_WAGES.select_usable_samples
submit_only = CPS_WAGES.submit_only
wait_for_extract_ready = CPS_WAGES.wait_for_extract_ready


class FakeVariable:
    def __init__(self, codes: dict[str, int]) -> None:
        self.codes = codes


class FakeDdi:
    def get_variable_info(self, name: str) -> FakeVariable:
        assert name == "STATEFIP"
        return FakeVariable({"California": 6, "New York": 36})


def test_read_sample_ids_and_skip_pre_1990_samples(tmp_path: Path) -> None:
    samples_file = tmp_path / "asec_id_list.txt"
    samples_file.write_text(
        "\n".join(
            [
                "# comment",
                "cps1962_03s",
                "",
                "cps1989_03s",
                "cps1990_03s",
                "cps2025_03s",
            ]
        )
    )

    sample_ids = read_sample_ids(samples_file)
    selection = select_usable_samples(sample_ids)

    assert selection.selected == ["cps1990_03s", "cps2025_03s"]
    assert selection.skipped == ["cps1962_03s", "cps1989_03s"]
    assert MIN_UNION_ASEC_YEAR == 1990


def test_read_sample_ids_rejects_unexpected_sample_id(tmp_path: Path) -> None:
    samples_file = tmp_path / "asec_id_list.txt"
    samples_file.write_text("cps2025_02s\n")

    with pytest.raises(ValueError, match="expected format"):
        read_sample_ids(samples_file)


def test_extract_value_labels_handles_ipumspy_codebook_shape() -> None:
    assert extract_value_labels(FakeDdi(), "STATEFIP") == {
        6: "California",
        36: "New York",
    }


def test_aggregate_union_wages_filters_and_computes_weighted_means() -> None:
    df = pd.DataFrame(
        [
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 2,
                "EARNWEEK": 1000,
                "EARNWT": 2,
            },
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 3,
                "EARNWEEK": 500,
                "EARNWT": 1,
            },
            {
                "YEAR": 1991,
                "STATEFIP": 36,
                "IND1950": 246,
                "UNION": 2,
                "EARNWEEK": 200,
                "EARNWT": 4,
            },
            {
                "YEAR": 1989,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 2,
                "EARNWEEK": 1000,
                "EARNWT": 2,
            },
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 1,
                "EARNWEEK": 1000,
                "EARNWT": 2,
            },
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 1,
                "EARNWEEK": 800,
                "EARNWT": 1,
            },
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 2,
                "EARNWEEK": 1000,
                "EARNWT": 0,
            },
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 2,
                "EARNWEEK": 9999.99,
                "EARNWT": 2,
            },
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 2,
                "EARNWEEK": 0,
                "EARNWT": 2,
            },
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 246,
                "UNION": 1,
                "EARNWEEK": 9999.99,
                "EARNWT": 2,
            },
            {
                "YEAR": 1990,
                "STATEFIP": 6,
                "IND1950": 0,
                "UNION": 2,
                "EARNWEEK": 1000,
                "EARNWT": 2,
            },
        ]
    )

    panel = aggregate_union_wages(
        df,
        state_labels={6: "California", 36: "New York"},
        industry_labels={246: "Construction"},
    )

    assert len(panel) == 2

    california = panel[panel["statefip"] == 6].iloc[0]
    assert california["year"] == 1990
    assert california["state_name"] == "California"
    assert california["industry_name"] == "Construction"
    assert california["mean_weekly_earnings_nominal"] == pytest.approx(2500 / 3)
    assert california["union_covered_weighted_workers"] == 3
    assert california["union_covered_unweighted_n"] == 2
    assert california["nonunion_mean_weekly_earnings_nominal"] == pytest.approx(
        2800 / 3
    )
    assert california["nonunion_weighted_workers"] == 3
    assert california["nonunion_unweighted_n"] == 2

    new_york = panel[panel["statefip"] == 36].iloc[0]
    assert new_york["year"] == 1991
    assert new_york["state_name"] == "New York"
    assert new_york["industry_name"] == "Construction"
    assert new_york["mean_weekly_earnings_nominal"] == 200
    assert new_york["union_covered_weighted_workers"] == 4
    assert new_york["union_covered_unweighted_n"] == 1
    assert pd.isna(new_york["nonunion_mean_weekly_earnings_nominal"])
    assert new_york["nonunion_weighted_workers"] == 0
    assert new_york["nonunion_unweighted_n"] == 0


def test_panel_time_series_uses_worker_weighted_means() -> None:
    panel = pd.DataFrame(
        [
            {
                "year": 1990,
                "statefip": 6,
                "state_name": "California",
                "ind1950": 246,
                "industry_name": "Construction",
                "mean_weekly_earnings_nominal": 100,
                "union_covered_weighted_workers": 2,
                "union_covered_unweighted_n": 1,
            },
            {
                "year": 1990,
                "statefip": 36,
                "state_name": "New York",
                "ind1950": 246,
                "industry_name": "Construction",
                "mean_weekly_earnings_nominal": 200,
                "union_covered_weighted_workers": 3,
                "union_covered_unweighted_n": 2,
            },
            {
                "year": 1990,
                "statefip": 6,
                "state_name": "California",
                "ind1950": 336,
                "industry_name": "Blast furnaces",
                "mean_weekly_earnings_nominal": 400,
                "union_covered_weighted_workers": 4,
                "union_covered_unweighted_n": 3,
            },
        ]
    )

    industry_time_series = build_industry_time_series(panel)
    state_time_series = build_state_time_series(panel)

    construction = industry_time_series[
        industry_time_series["ind1950"] == 246
    ].iloc[0]
    california = state_time_series[state_time_series["statefip"] == 6].iloc[0]

    assert construction["mean_weekly_earnings_nominal"] == pytest.approx(160)
    assert construction["union_covered_weighted_workers"] == 5
    assert construction["union_covered_unweighted_n"] == 3

    assert california["mean_weekly_earnings_nominal"] == pytest.approx(300)
    assert california["union_covered_weighted_workers"] == 6
    assert california["union_covered_unweighted_n"] == 4


def test_run_figures_only_reads_timeseries_inputs(tmp_path: Path, monkeypatch) -> None:
    industry_file = tmp_path / "industry.csv"
    state_file = tmp_path / "state.csv"
    industry_plot = tmp_path / "industry.png"
    state_plot = tmp_path / "state.png"

    pd.DataFrame(
        [
            {
                "year": 1990,
                "ind1950": 246,
                "industry_name": "Construction",
                "mean_weekly_earnings_nominal": 100,
                "union_covered_weighted_workers": 2,
                "union_covered_unweighted_n": 1,
            }
        ]
    ).to_csv(industry_file, index=False)
    pd.DataFrame(
        [
            {
                "year": 1990,
                "statefip": 6,
                "state_name": "California",
                "mean_weekly_earnings_nominal": 200,
                "union_covered_weighted_workers": 3,
                "union_covered_unweighted_n": 2,
            }
        ]
    ).to_csv(state_file, index=False)

    calls = []

    def fake_plot(time_series, entity_code_column, entity_label_column, output, title, top_n):
        calls.append(
            (
                len(time_series),
                entity_code_column,
                entity_label_column,
                output,
                title,
                top_n,
            )
        )

    monkeypatch.setattr(CPS_WAGES, "plot_time_series", fake_plot)

    industry_time_series, state_time_series = run_figures_only(
        industry_timeseries_file=industry_file,
        state_timeseries_file=state_file,
        industry_plot_output=industry_plot,
        state_plot_output=state_plot,
        plot_top_n=4,
    )

    assert len(industry_time_series) == 1
    assert len(state_time_series) == 1
    assert calls == [
        (
            1,
            "ind1950",
            "industry_name",
            industry_plot,
            "Union-covered weekly earnings by industry",
            4,
        ),
        (
            1,
            "statefip",
            "state_name",
            state_plot,
            "Union-covered weekly earnings by state",
            4,
        ),
    ]


def test_wait_for_extract_ready_polls_until_completed(monkeypatch) -> None:
    statuses = ["queued", "started", "completed"]
    sleeps = []

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        def extract_status(self, extract_id: int, collection: str) -> str:
            assert extract_id == 123
            assert collection == "cps"
            return statuses.pop(0)

    monkeypatch.setattr(
        CPS_WAGES,
        "import_ipumspy",
        lambda: (FakeClient, object, object),
    )
    monkeypatch.setattr(CPS_WAGES, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(CPS_WAGES.time, "sleep", lambda seconds: sleeps.append(seconds))

    status = wait_for_extract_ready(123, poll_interval_seconds=5, timeout_seconds=60)

    assert status == "completed"
    assert sleeps == [5, 5]


def test_wait_for_extract_ready_fails_on_failed_status(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        def extract_status(self, extract_id: int, collection: str) -> str:
            assert extract_id == 123
            assert collection == "cps"
            return "failed"

    monkeypatch.setattr(
        CPS_WAGES,
        "import_ipumspy",
        lambda: (FakeClient, object, object),
    )
    monkeypatch.setattr(CPS_WAGES, "get_api_key", lambda: "test-key")

    with pytest.raises(RuntimeError, match="ended with status failed"):
        wait_for_extract_ready(123, poll_interval_seconds=5, timeout_seconds=60)


def test_submit_only_submits_selected_samples(tmp_path: Path, monkeypatch) -> None:
    samples_file = tmp_path / "asec_id_list.txt"
    samples_file.write_text("cps1989_03s\ncps1990_03s\n")
    submitted_samples = []

    class FakeExtract:
        extract_id = 456

    class FakeMicrodataExtract:
        def __init__(self, collection, samples, variables, description) -> None:
            assert collection == "cps"
            submitted_samples.extend(samples)

        def add_data_quality_flags(self, variables) -> None:
            assert variables == ["UNION", "EARNWEEK"]

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        def submit_extract(self, extract) -> FakeExtract:
            return FakeExtract()

    monkeypatch.setattr(
        CPS_WAGES,
        "import_ipumspy",
        lambda: (FakeClient, FakeMicrodataExtract, object),
    )
    monkeypatch.setattr(CPS_WAGES, "get_api_key", lambda: "test-key")

    extract_id = submit_only(samples_file)

    assert extract_id == 456
    assert submitted_samples == ["cps1990_03s"]
