"""Tests for the taxonomy-level handling in the classification figures."""

from collections.abc import Sequence
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from pipeline.utils import link_classifications, plot_classifications


PROVISION_TYPE = "technology"
SOURCE = "dol_archive"


def _documents(subtypes: list[str]) -> pd.DataFrame:
    """A minimal document table carrying one level's ``has_``/``n_`` columns."""

    frame = pd.DataFrame(
        {
            "document_id": ["document_1", "document_2"],
            "meta_matched": [True, True],
            "expire_period": ["2005-2009", "2010-2014"],
            "sector_label": ["Manufacturing", "Retail"],
            "n_provisions": [1, 0],
        }
    )
    for index, subtype in enumerate(subtypes):
        frame[f"n_{subtype}"] = [1 if index == 0 else 0, 0]
        frame[f"has_{subtype}"] = frame[f"n_{subtype}"] > 0
    return frame


class SubtypeStyleTests(unittest.TestCase):
    def test_colour_and_marker_pairs_stay_unique_past_the_palette_length(self) -> None:
        # Level 2 draws more series than there are hues, so the pair must vary
        # even once the hues wrap -- otherwise two subtypes render identically.
        subtypes = [f"subtype_{index}" for index in range(20)]

        colors, markers, _ = plot_classifications.subtype_style(
            subtypes, PROVISION_TYPE
        )
        pairs = [(colors[subtype], markers[subtype]) for subtype in subtypes]

        self.assertEqual(len(set(pairs)), len(subtypes))

    def test_absence_series_keeps_its_reserved_colour(self) -> None:
        colors, markers, labels = plot_classifications.subtype_style(
            ["retraining", "reassignment"], PROVISION_TYPE
        )

        self.assertEqual(
            colors[plot_classifications.NONE_KEY], plot_classifications.NONE_COLOR
        )
        self.assertNotIn(
            plot_classifications.NONE_COLOR,
            [colors["retraining"], colors["reassignment"]],
        )
        self.assertIn(PROVISION_TYPE, labels[plot_classifications.NONE_KEY])
        self.assertEqual(
            markers[plot_classifications.NONE_KEY], plot_classifications.NONE_MARKER
        )


class MinCbasTests(unittest.TestCase):
    @staticmethod
    def _documents(counts: dict[str, int], total: int = 10) -> pd.DataFrame:
        """A table where each subtype is carried by exactly ``counts[subtype]`` CBAs."""

        frame = pd.DataFrame(
            {
                "document_id": [f"document_{index}" for index in range(total)],
                "meta_matched": [True] * total,
                "expire_period": ["2005-2009"] * total,
                "sector_label": ["Manufacturing"] * total,
                "n_provisions": [1] * total,
            }
        )
        for subtype, count in counts.items():
            flags = [index < count for index in range(total)]
            frame[f"has_{subtype}"] = flags
            frame[f"n_{subtype}"] = [int(flag) for flag in flags]
        return frame

    def test_splits_on_the_document_count_not_the_provision_count(self) -> None:
        documents = self._documents({"kept": 4, "edge": 3, "dropped": 2})
        # One document holding many provisions of a subtype still counts once.
        documents.loc[0, "n_kept"] = 9

        kept, dropped = plot_classifications.frequent_subtypes(
            documents, ["kept", "edge", "dropped"], min_cbas=3
        )

        self.assertEqual(kept, ["kept", "edge"])
        self.assertEqual(dropped, [("dropped", 2)])

    def test_absent_column_counts_as_zero(self) -> None:
        documents = self._documents({"kept": 4})

        kept, dropped = plot_classifications.frequent_subtypes(
            documents, ["kept", "never_used"], min_cbas=1
        )

        self.assertEqual(kept, ["kept"])
        self.assertEqual(dropped, [("never_used", 0)])

    def test_kept_subtypes_hold_their_declared_order(self) -> None:
        documents = self._documents({"a": 5, "b": 1, "c": 5})

        kept, _ = plot_classifications.frequent_subtypes(
            documents, ["a", "b", "c"], min_cbas=5
        )

        self.assertEqual(kept, ["a", "c"])


class MinCbasCliTests(unittest.TestCase):
    def _argv(self, tmp_dir: Path, *extra: str) -> list[str]:
        return [
            "--source",
            SOURCE,
            "--clause-type",
            PROVISION_TYPE,
            "--level",
            "2",
            "--input-dir",
            str(tmp_dir),
            "--output-dir",
            str(tmp_dir / "fig"),
            *extra,
        ]

    @staticmethod
    def _write(tmp_dir: Path) -> None:
        prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 2)
        frame = MinCbasTests._documents(
            {"workforce_training": 8, "notification_right": 2}
        )
        frame.to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)

    def test_industry_adjustment_flags_parse(self) -> None:
        default = plot_classifications.parse_args([])
        self.assertTrue(default.industry_adjusted)
        self.assertEqual(
            default.min_cell_docs, plot_classifications.DEFAULT_MIN_CELL_DOCS
        )

        self.assertTrue(default.show_unknown)

        opted_out = plot_classifications.parse_args(
            ["--no-industry-adjusted", "--min-cell-docs", "3", "--hide-no-naics"]
        )
        self.assertFalse(opted_out.industry_adjusted)
        self.assertEqual(opted_out.min_cell_docs, 3)
        self.assertFalse(opted_out.show_unknown)

    def test_both_flag_spellings_are_accepted(self) -> None:
        self.assertEqual(
            plot_classifications.parse_args([]).min_cbas,
            plot_classifications.DEFAULT_MIN_CBAS,
        )
        for flag in ("--min-cbas", "--min_cbas"):
            with self.subTest(flag=flag):
                self.assertEqual(
                    plot_classifications.parse_args([flag, "7"]).min_cbas, 7
                )

    def test_sparse_subtypes_are_hidden_and_reported(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir)

            output = StringIO()
            with redirect_stdout(output):
                status = plot_classifications.main(
                    self._argv(tmp_dir, "--min-cbas", "5")
                )
            summary = output.getvalue()

        self.assertEqual(status, 0)
        # Hidden series are named with their counts, never dropped silently.
        self.assertIn("hiding", summary)
        self.assertIn("notification_right (2)", summary)
        self.assertNotIn("workforce_training (", summary)

    def test_threshold_of_zero_hides_nothing(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir)

            output = StringIO()
            with redirect_stdout(output):
                plot_classifications.main(self._argv(tmp_dir, "--min-cbas", "0"))

        self.assertNotIn("hiding", output.getvalue())

    def test_style_is_assigned_before_filtering_so_colours_stay_stable(self) -> None:
        # Styling the survivors instead would slide each one into the gap the
        # hidden series left, so the same subtype would change colour whenever
        # --min-cbas changed.  Prove that gap exists, then that main avoids it.
        full = ["first", "second", "third"]
        by_full, _, _ = plot_classifications.subtype_style(full, PROVISION_TYPE)
        by_survivors, _, _ = plot_classifications.subtype_style(
            ["first", "third"], PROVISION_TYPE
        )
        self.assertNotEqual(by_full["third"], by_survivors["third"])

        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir)
            styled: list[Sequence[str]] = []
            original = plot_classifications.subtype_style

            def spy(subtypes, clause_type):
                styled.append(list(subtypes))
                return original(subtypes, clause_type)

            with patch.object(plot_classifications, "subtype_style", spy):
                with redirect_stdout(StringIO()):
                    plot_classifications.main(self._argv(tmp_dir, "--min-cbas", "5"))

        # The sparse subtype is still present when the palette is handed out.
        self.assertEqual(len(styled), 1)
        self.assertIn("notification_right", styled[0])
        self.assertIn("workforce_training", styled[0])

    def test_filtering_everything_out_fails_loudly(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir)

            with self.assertRaisesRegex(SystemExit, "lower --min-cbas"):
                with redirect_stdout(StringIO()):
                    plot_classifications.main(self._argv(tmp_dir, "--min-cbas", "99"))

    def test_negative_threshold_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir)

            with self.assertRaisesRegex(SystemExit, "--min-cbas"):
                with redirect_stdout(StringIO()):
                    plot_classifications.main(self._argv(tmp_dir, "--min-cbas", "-1"))


def _beneficiary_documents() -> pd.DataFrame:
    """Four CBAs: two with a clean 75/25 employer/worker split, one all-worker,
    and one with no provisions at all (its share is undefined, not zero)."""

    return pd.DataFrame(
        {
            "document_id": ["document_1", "document_2", "document_3", "document_4"],
            "meta_matched": [True, True, True, True],
            "expire_period": ["2005-2009", "2005-2009", "2010-2014", "2010-2014"],
            "sector_label": ["Manufacturing", "Manufacturing", "Retail trade", "Retail trade"],
            "n_provisions": [4, 4, 2, 0],
            "pct_beneficiary_employer": [75.0, 75.0, 0.0, None],
            "pct_beneficiary_workers": [25.0, 25.0, 100.0, None],
            "pct_beneficiary_unclear": [0.0, 0.0, 0.0, None],
        }
    )


class BeneficiaryStyleTests(unittest.TestCase):
    def test_colors_markers_and_labels_are_assigned_per_beneficiary(self) -> None:
        colors, markers, labels = plot_classifications.beneficiary_style(
            ("employer", "workers", "unclear")
        )

        self.assertEqual(len(set(colors.values())), 3)
        self.assertEqual(len(set(markers.values())), 3)
        self.assertEqual(labels["employer"], "Employer")
        # No absence series here -- every provision names exactly one party.
        self.assertNotIn(plot_classifications.NONE_KEY, colors)

    def test_overridden_beneficiaries_keep_fixed_colors_regardless_of_order(
        self,
    ) -> None:
        forward, _, _ = plot_classifications.beneficiary_style(
            ("workers", "employer", "unclear")
        )
        reversed_order, _, _ = plot_classifications.beneficiary_style(
            ("employer", "unclear", "workers")
        )

        for colors in (forward, reversed_order):
            for beneficiary in ("workers", "employer", "unclear"):
                self.assertEqual(
                    colors[beneficiary],
                    plot_classifications.BENEFICIARY_COLOR_OVERRIDES[beneficiary],
                )
        self.assertEqual(forward, reversed_order)


class BeneficiaryMeanTests(unittest.TestCase):
    def test_overall_mean_excludes_cbas_with_no_provisions(self) -> None:
        documents = _beneficiary_documents()

        means, n = plot_classifications.beneficiary_overall_means(
            documents, ("employer", "workers", "unclear")
        )

        # document_4 has no provisions and must not pull the mean toward zero.
        self.assertEqual(n, 3)
        self.assertAlmostEqual(means["employer"], (75.0 + 75.0 + 0.0) / 3)
        self.assertAlmostEqual(means["workers"], (25.0 + 25.0 + 100.0) / 3)

    def test_group_mean_is_the_mean_of_shares_not_a_pooled_share(self) -> None:
        documents = _beneficiary_documents()

        means, totals = plot_classifications.beneficiary_group_means(
            documents, "expire_period", ("employer", "workers", "unclear")
        )

        self.assertEqual(totals["2005-2009"], 2)
        self.assertAlmostEqual(means.loc["2005-2009", "employer"], 75.0)
        self.assertEqual(totals["2010-2014"], 1)
        self.assertAlmostEqual(means.loc["2010-2014", "employer"], 0.0)
        self.assertAlmostEqual(means.loc["2010-2014", "workers"], 100.0)


class BeneficiaryFigureTests(unittest.TestCase):
    def test_by_period_excludes_the_provision_less_cba_from_its_cohort_count(self) -> None:
        documents = _beneficiary_documents()
        colors, markers, labels = plot_classifications.beneficiary_style(
            ("employer", "workers", "unclear")
        )

        figure = plot_classifications.plot_beneficiary_by_period(
            documents, ("employer", "workers", "unclear"), colors, markers, labels,
            min_docs=1, clause_type=PROVISION_TYPE,
        )

        ax = figure.axes[0]
        labels_seen = [tick.get_text() for tick in ax.get_xticklabels()]
        self.assertIn("2010-2014\n1 CBAs", labels_seen)

    def test_no_naics_row_is_shown_by_default_and_hidden_on_request(self) -> None:
        documents = _beneficiary_documents()
        documents.loc[2:, "sector_label"] = plot_classifications.UNKNOWN_SECTOR
        colors, _, labels = plot_classifications.beneficiary_style(
            ("employer", "workers", "unclear")
        )

        def rows(show_unknown):
            with redirect_stdout(StringIO()):
                figure = plot_classifications.plot_beneficiary_by_sector(
                    documents, ("employer", "workers", "unclear"), colors, labels,
                    min_docs=1, clause_type=PROVISION_TYPE,
                    show_unknown=show_unknown,
                )
            return [t.get_text() for t in figure.axes[0].get_yticklabels()]

        shown = rows(True)
        self.assertTrue(
            any(plot_classifications.NO_NAICS_LABEL in row for row in shown)
        )
        # Pinned to the bottom rather than ordered among the real sectors.
        self.assertIn(plot_classifications.NO_NAICS_LABEL, shown[-1])

        hidden = rows(False)
        self.assertFalse(
            any(plot_classifications.NO_NAICS_LABEL in row for row in hidden)
        )

    def test_by_sector_pools_sparse_sectors(self) -> None:
        documents = _beneficiary_documents()
        colors, markers, labels = plot_classifications.beneficiary_style(
            ("employer", "workers", "unclear")
        )

        figure = plot_classifications.plot_beneficiary_by_sector(
            documents, ("employer", "workers", "unclear"), colors, labels,
            min_docs=10, clause_type=PROVISION_TYPE,
        )

        ax = figure.axes[0]
        labels_seen = [tick.get_text() for tick in ax.get_yticklabels()]
        self.assertEqual(len(labels_seen), 1)
        self.assertIn(plot_classifications.OTHER_SECTOR_LABEL, labels_seen[0])


def _unbalanced_documents() -> pd.DataFrame:
    """Two cohorts whose industry *mix* flips, with each industry held constant.

    Manufacturing always scores 20, Construction always 80, so an honest
    composition-adjusted index is a flat 50 in both cohorts.  The raw average
    swings 68 -> 32 purely because the mix inverts.  A fixture with a stable mix
    would pass even if the code just returned the raw mean, which is the bug
    worth guarding against here.
    """

    rows = []
    for cohort, mfg, con in (("2000-2004", 1, 4), ("2005-2009", 4, 1)):
        rows += [("Manufacturing", cohort, 20.0)] * mfg
        rows += [("Construction", cohort, 80.0)] * con
    frame = pd.DataFrame(
        rows, columns=["sector_label", "expire_period", "pct_beneficiary_employer"]
    )
    frame["document_id"] = [f"document_{i}" for i in range(len(frame))]
    frame["n_provisions"] = 1
    frame["pct_beneficiary_workers"] = 100.0 - frame["pct_beneficiary_employer"]
    frame["pct_beneficiary_unclear"] = 0.0
    frame["sector_group"] = frame["sector_label"]
    return frame


class CompositionAdjustedTests(unittest.TestCase):
    SERIES = ["employer", "workers", "unclear"]

    def _adjust(self, documents, **kwargs):
        return plot_classifications.composition_adjusted(
            documents, "expire_period", self.SERIES,
            plot_classifications._beneficiary_means_for, **kwargs
        )

    def test_adjustment_removes_a_mix_shift_the_raw_average_reports(self) -> None:
        documents = _unbalanced_documents()

        raw = documents.groupby("expire_period")["pct_beneficiary_employer"].mean()
        adjusted, counts, used, dropped = self._adjust(documents)

        # The raw average moves with the mix; the adjusted index must not.
        self.assertAlmostEqual(raw["2000-2004"], 68.0)
        self.assertAlmostEqual(raw["2005-2009"], 32.0)
        self.assertAlmostEqual(adjusted.loc["2000-2004", "employer"], 50.0)
        self.assertAlmostEqual(adjusted.loc["2005-2009", "employer"], 50.0)
        self.assertEqual(sorted(used), ["Construction", "Manufacturing"])
        self.assertEqual(dropped, [])
        self.assertEqual(counts["2000-2004"], 5)

    def test_weights_are_honoured(self) -> None:
        adjusted, _, _, _ = self._adjust(
            _unbalanced_documents(),
            weights={"Manufacturing": 3.0, "Construction": 1.0},
        )

        # (3*20 + 1*80) / 4 = 35, not the unweighted 50.
        self.assertAlmostEqual(adjusted.loc["2000-2004", "employer"], 35.0)

    def test_stratum_missing_from_a_cohort_is_dropped_and_reported(self) -> None:
        documents = _unbalanced_documents()
        extra = documents.iloc[[0]].copy()
        extra[["sector_label", "sector_group"]] = "Retail trade"
        extra["document_id"] = "document_one_cohort_only"
        documents = pd.concat([documents, extra], ignore_index=True)

        _, _, used, dropped = self._adjust(documents)

        self.assertNotIn("Retail trade", used)
        # The reason names the cohort it is missing from, so nothing hides.
        self.assertIn("2005-2009", dict(dropped)["Retail trade"])

    def test_min_cell_docs_excludes_a_thin_stratum(self) -> None:
        _, _, used, dropped = self._adjust(_unbalanced_documents(), min_cell_docs=2)

        # Each industry has a single-CBA cell in one cohort, so neither clears 2.
        self.assertEqual(used, [])
        self.assertEqual(len(dropped), 2)

    def test_single_surviving_stratum_yields_no_index(self) -> None:
        documents = _unbalanced_documents()
        adjusted, _, used, _ = self._adjust(
            documents[documents["sector_group"] == "Construction"]
        )

        # An index over one stratum is just the raw line relabelled.
        self.assertEqual(used, ["Construction"])
        self.assertTrue(adjusted["employer"].isna().all())


class UnknownStratumTests(unittest.TestCase):
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame({
            "document_id": [f"document_{i}" for i in range(6)],
            "sector_label": ["Construction"] * 3
            + [plot_classifications.UNKNOWN_SECTOR] * 3,
            "n_provisions": [1] * 6,
        })

    def test_unknown_is_retained_as_its_own_stratum_when_asked(self) -> None:
        pooled, _ = plot_classifications._grouped_by_sector(
            self._frame(), min_docs=1, keep_unknown=True
        )

        groups = set(pooled["sector_group"])
        self.assertIn(plot_classifications.UNKNOWN_SECTOR, groups)
        # Never folded into the remainder bucket, even below the threshold.
        self.assertNotIn(plot_classifications.OTHER_SECTOR_LABEL, groups)

    def test_helper_still_drops_unknown_unless_asked(self) -> None:
        # The helper's own default stays exclusive; callers that want the bucket
        # opt in, so no existing caller changed behaviour when it was added.
        pooled, _ = plot_classifications._grouped_by_sector(self._frame(), min_docs=1)

        self.assertNotIn(
            plot_classifications.UNKNOWN_SECTOR, set(pooled["sector_group"])
        )
        self.assertEqual(len(pooled), 3)

    def test_unknown_sorts_below_the_real_sectors_however_large(self) -> None:
        # The bucket outnumbers every real sector here, so size ordering alone
        # would float it to the top and read as the biggest industry.
        totals = pd.Series({
            plot_classifications.UNKNOWN_SECTOR: 99,
            "Construction": 40,
            plot_classifications.OTHER_SECTOR_LABEL: 14,
            "Manufacturing": 26,
        })

        order = plot_classifications._sector_order(totals)

        self.assertEqual(order[0], "Construction")
        self.assertEqual(order[-2:], [
            plot_classifications.OTHER_SECTOR_LABEL,
            plot_classifications.UNKNOWN_SECTOR,
        ])

    def test_tick_label_renames_the_bucket_away_from_industry_language(self) -> None:
        tick = plot_classifications._sector_tick(
            plot_classifications.UNKNOWN_SECTOR, 99
        )

        self.assertIn(plot_classifications.NO_NAICS_LABEL, tick)
        self.assertIn("99 CBAs", tick)
        self.assertNotIn(plot_classifications.UNKNOWN_SECTOR, tick)


class AdjustedLineOnFiguresTests(unittest.TestCase):
    @staticmethod
    def _dotted(figure) -> list:
        return [
            line
            for line in figure.axes[0].get_lines()
            if line.get_linestyle() == plot_classifications.ADJUSTED_LINESTYLE
        ]

    @staticmethod
    def _documents() -> pd.DataFrame:
        documents = _unbalanced_documents()
        documents["has_implementation"] = True
        documents["n_implementation"] = 1
        return documents

    def _beneficiary_figure(self, **kwargs):
        colors, markers, labels = plot_classifications.beneficiary_style(
            ("employer", "workers", "unclear")
        )
        with redirect_stdout(StringIO()):
            return plot_classifications.plot_beneficiary_by_period(
                self._documents(), ("employer", "workers", "unclear"),
                colors, markers, labels, min_docs=1,
                clause_type=PROVISION_TYPE, **kwargs
            )

    def test_one_dotted_line_per_series_plus_two_style_legend_entries(self) -> None:
        figure = self._beneficiary_figure(industry_adjusted=True, min_sector_docs=1)

        self.assertEqual(len(self._dotted(figure)), 3)
        # Each series appears once, plus the two line-style proxies.
        texts = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
        self.assertIn(plot_classifications.ADJUSTED_LABEL, texts)
        self.assertIn(plot_classifications.OVERALL_LABEL, texts)
        self.assertEqual(len(texts), 5)

    def test_opting_out_draws_no_dotted_line_and_no_style_legend(self) -> None:
        figure = self._beneficiary_figure(industry_adjusted=False)

        self.assertEqual(self._dotted(figure), [])
        texts = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
        self.assertNotIn(plot_classifications.ADJUSTED_LABEL, texts)

    def test_subtype_period_figure_draws_the_dotted_index_too(self) -> None:
        subtypes = ["implementation"]
        colors, markers, labels = plot_classifications.subtype_style(
            subtypes, PROVISION_TYPE
        )

        with redirect_stdout(StringIO()):
            figure = plot_classifications.plot_by_period(
                self._documents(), subtypes, colors, markers, labels,
                min_docs=1, clause_type=PROVISION_TYPE,
                industry_adjusted=True, min_sector_docs=1,
            )

        # One per subtype plus the absence series.
        self.assertEqual(len(self._dotted(figure)), 2)


class PoolNonSpanningTests(unittest.TestCase):
    """The index needs strata present in every cohort; folding beats discarding."""

    COHORTS = ["2000-2004", "2005-2009"]

    @staticmethod
    def _frame(rows) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=["sector_group", "expire_period"])
        frame["document_id"] = [f"document_{i}" for i in range(len(frame))]
        return frame

    def _pool(self, rows, **kwargs):
        return plot_classifications._pool_non_spanning(
            self._frame(rows), "expire_period", self.COHORTS, 1, **kwargs
        )

    def test_a_stratum_missing_a_cohort_joins_the_remainder(self) -> None:
        frame, folded = self._pool(
            [("Manufacturing", c) for c in self.COHORTS]
            + [("Retail trade", "2000-2004")]
        )

        self.assertEqual(folded, ["Retail trade"])
        groups = set(frame["sector_group"])
        self.assertEqual(
            groups, {"Manufacturing", plot_classifications.OTHER_SECTOR_LABEL}
        )
        # Folded, not dropped: the row is still in the sample.
        self.assertEqual(len(frame), 3)

    def test_a_stratum_spanning_every_cohort_is_untouched(self) -> None:
        rows = [(sector, c) for sector in ("Manufacturing", "Retail trade")
                for c in self.COHORTS]
        frame, folded = self._pool(rows)

        self.assertEqual(folded, [])
        self.assertEqual(set(frame["sector_group"]), {"Manufacturing", "Retail trade"})

    def test_the_no_naics_bucket_is_never_folded(self) -> None:
        """Pooling it in would make the remainder mean "no industry" as well."""

        frame, folded = self._pool(
            [("Manufacturing", c) for c in self.COHORTS]
            + [(plot_classifications.UNKNOWN_SECTOR, "2000-2004")]
        )

        self.assertEqual(folded, [])
        self.assertIn(plot_classifications.UNKNOWN_SECTOR, set(frame["sector_group"]))

    def test_folding_keeps_the_index_estimable_where_it_would_collapse(self) -> None:
        """One spanning sector plus thin ones: without folding there is no index."""

        rows = (
            [("Manufacturing", c) for c in self.COHORTS]
            + [("Retail trade", "2000-2004"), ("Construction", "2005-2009")]
        )
        series = ["employer", "workers", "unclear"]

        def adjust(frame):
            frame = frame.copy()
            frame["n_provisions"] = 1
            frame["pct_beneficiary_employer"] = 50.0
            frame["pct_beneficiary_workers"] = 50.0
            frame["pct_beneficiary_unclear"] = 0.0
            return plot_classifications.composition_adjusted(
                frame, "expire_period", series,
                plot_classifications._beneficiary_means_for,
            )

        _, _, used_raw, _ = adjust(self._frame(rows))
        self.assertEqual(used_raw, ["Manufacturing"])

        folded_frame, folded = self._pool(rows)
        _, _, used, _ = adjust(folded_frame)
        self.assertEqual(sorted(folded), ["Construction", "Retail trade"])
        self.assertEqual(
            sorted(used),
            sorted(["Manufacturing", plot_classifications.OTHER_SECTOR_LABEL]),
        )


class BeneficiaryMainCliTests(unittest.TestCase):
    def _argv(self, tmp_dir: Path) -> list[str]:
        return [
            "--source",
            SOURCE,
            "--clause-type",
            PROVISION_TYPE,
            "--level",
            "1",
            "--input-dir",
            str(tmp_dir),
            "--output-dir",
            str(tmp_dir / "fig"),
        ]

    def test_beneficiary_figures_are_written_when_columns_are_present(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 1)
            frame = _documents(["implementation", "preemptive_rights"])
            for beneficiary in link_classifications.BENEFICIARY_LABELS:
                frame[f"pct_beneficiary_{beneficiary}"] = 50.0 if beneficiary != "unclear" else 0.0
            frame.to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)

            with redirect_stdout(StringIO()):
                status = plot_classifications.main(self._argv(tmp_dir))

            self.assertEqual(status, 0)
            self.assertTrue((tmp_dir / "fig" / f"{prefix}_beneficiary_share.png").is_file())

    def test_missing_beneficiary_columns_warn_instead_of_failing(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 1)
            # An older document table, built before beneficiary columns existed.
            _documents(["implementation", "preemptive_rights"]).to_csv(
                tmp_dir / f"{prefix}_documents.csv", index=False
            )

            output = StringIO()
            with redirect_stdout(output):
                status = plot_classifications.main(self._argv(tmp_dir))

            self.assertEqual(status, 0)
            self.assertIn("pct_beneficiary_", output.getvalue())
            self.assertFalse((tmp_dir / "fig" / f"{prefix}_beneficiary_share.png").is_file())


class LevelSelectionTests(unittest.TestCase):
    def _argv(
        self, tmp_dir: Path, level: int, extra: Sequence[str] = ()
    ) -> list[str]:
        return [
            "--source",
            SOURCE,
            "--clause-type",
            PROVISION_TYPE,
            "--level",
            str(level),
            "--input-dir",
            str(tmp_dir),
            "--output-dir",
            str(tmp_dir / "fig"),
            *extra,
        ]

    @staticmethod
    def _write(tmp_dir: Path, level: int, subtypes: list[str]) -> None:
        prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, level)
        _documents(subtypes).to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)

    def test_each_level_reads_its_own_table(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir, 1, ["implementation", "preemptive_rights"])
            # workforce_training is a child of implementation; notification_right
            # is a child of preemptive_rights -- see pipeline/provisions/technology.yaml.
            self._write(tmp_dir, 2, ["workforce_training", "notification_right"])

            for level in (1, 2):
                with self.subTest(level=level):
                    output = StringIO()
                    with redirect_stdout(output):
                        status = plot_classifications.main(self._argv(tmp_dir, level))

                    self.assertEqual(status, 0)
                    self.assertIn(f"_l{level}_documents.csv", output.getvalue())
                    stem = f"{PROVISION_TYPE}_{SOURCE}_l{level}"
                    if level == 1:
                        expected = f"{stem}_subtype_share.png"
                    else:
                        expected = f"{stem}_implementation_subtype_share.png"
                    self.assertTrue((tmp_dir / "fig" / expected).is_file())

    def test_level_2_splits_into_one_figure_set_per_level1_category(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            # Unlike ``_write``, give both subtypes a nonzero count -- ``_documents``
            # only ever populates its first subtype, which would starve
            # notification_right and leave preemptive_rights empty too.
            frame = pd.DataFrame(
                {
                    "document_id": ["document_1", "document_2", "document_3"],
                    "meta_matched": [True, True, True],
                    "expire_period": ["2005-2009", "2010-2014", "2010-2014"],
                    "sector_label": ["Manufacturing", "Retail", "Retail"],
                    "n_provisions": [1, 1, 0],
                    "n_workforce_training": [1, 0, 0],
                    "has_workforce_training": [True, False, False],
                    "n_notification_right": [0, 1, 0],
                    "has_notification_right": [False, True, False],
                }
            )
            prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 2)
            frame.to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)

            output = StringIO()
            with redirect_stdout(output):
                status = plot_classifications.main(
                    self._argv(
                        tmp_dir,
                        2,
                        extra=["--min-cbas", "1", "--min-group-docs", "1"],
                    )
                )

            self.assertEqual(status, 0)
            stem = f"{PROVISION_TYPE}_{SOURCE}_l2"
            for group in ("implementation", "preemptive_rights"):
                self.assertTrue(
                    (tmp_dir / "fig" / f"{stem}_{group}_subtype_share.png").is_file()
                )
                self.assertTrue(
                    (tmp_dir / "fig" / f"{stem}_{group}_share_by_period.png").is_file()
                )
                self.assertTrue(
                    (tmp_dir / "fig" / f"{stem}_{group}_share_by_sector.png").is_file()
                )
            # workforce_management has no member in this fixture and no CBA
            # carries it, so its figures are skipped rather than drawn empty.
            self.assertFalse(
                (
                    tmp_dir
                    / "fig"
                    / f"{stem}_workforce_management_subtype_share.png"
                ).is_file()
            )
            self.assertIn(
                "no level-2 subtype under 'workforce_management'", output.getvalue()
            )

    def test_missing_table_names_the_level_to_rebuild(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir, 1, ["implementation", "preemptive_rights"])

            with self.assertRaisesRegex(SystemExit, "--level 2"):
                with redirect_stdout(StringIO()):
                    plot_classifications.main(self._argv(tmp_dir, 2))

    def test_table_built_at_another_level_is_rejected_not_plotted_as_zero(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            # Level-1 labels written under the level-2 name.  "other" is present
            # at every level, so it must not be taken as proof of a match.
            prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 2)
            _documents(["implementation", "other"]).to_csv(
                tmp_dir / f"{prefix}_documents.csv", index=False
            )

            with self.assertRaisesRegex(SystemExit, "no level-2 subtype columns"):
                with redirect_stdout(StringIO()):
                    plot_classifications.main(self._argv(tmp_dir, 2))

    def test_level_beyond_the_taxonomy_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir, 1, ["implementation"])

            with self.assertRaisesRegex(SystemExit, "declares 2 level"):
                with redirect_stdout(StringIO()):
                    plot_classifications.main(self._argv(tmp_dir, 3))


class IncludeNoneTests(unittest.TestCase):
    """The absence series is optional -- a per-category figure drops it."""

    @staticmethod
    def _documents() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "document_id": ["document_1", "document_2", "document_3"],
                "sector_group": ["A", "A", "B"],
                "n_provisions": [1, 0, 1],
                "has_notification_right": [True, False, False],
                "has_other_in_preemptive_rights": [False, False, True],
            }
        )

    def test_overall_shares_omits_none_key_when_asked(self) -> None:
        documents = self._documents()

        shares, counts = plot_classifications.overall_shares(
            documents,
            ["notification_right", "other_in_preemptive_rights"],
            include_none=False,
        )

        self.assertNotIn(plot_classifications.NONE_KEY, shares.index)
        self.assertAlmostEqual(shares["other_in_preemptive_rights"], 100 / 3)
        self.assertEqual(counts["other_in_preemptive_rights"], 1)

    def test_overall_shares_keeps_none_key_by_default(self) -> None:
        documents = self._documents()

        shares, counts = plot_classifications.overall_shares(
            documents, ["notification_right"]
        )

        self.assertIn(plot_classifications.NONE_KEY, shares.index)
        self.assertEqual(counts[plot_classifications.NONE_KEY], 1)

    def test_series_order_respects_include_none(self) -> None:
        self.assertNotIn(
            plot_classifications.NONE_KEY,
            plot_classifications.series_order(["a"], include_none=False),
        )
        self.assertIn(
            plot_classifications.NONE_KEY,
            plot_classifications.series_order(["a"]),
        )

    def test_document_shares_omits_none_key_when_asked(self) -> None:
        documents = self._documents()

        shares, totals = plot_classifications.document_shares(
            documents,
            "sector_group",
            ["notification_right", "other_in_preemptive_rights"],
            include_none=False,
        )

        self.assertNotIn(plot_classifications.NONE_KEY, shares.columns)
        self.assertEqual(shares.loc["B", "other_in_preemptive_rights"], 100.0)


class GroupSeriesKeysTests(unittest.TestCase):
    def test_appends_a_styled_other_bucket_when_the_column_exists(self) -> None:
        documents = pd.DataFrame({"has_other_in_implementation": [True]})
        colors, markers, labels = {}, {}, {}

        keys = plot_classifications.group_series_keys(
            "implementation",
            ["job_displacement"],
            documents,
            colors,
            markers,
            labels,
        )

        self.assertEqual(keys, ["job_displacement", "other_in_implementation"])
        self.assertEqual(
            colors["other_in_implementation"], plot_classifications.NONE_COLOR
        )
        self.assertEqual(
            markers["other_in_implementation"], plot_classifications.NONE_MARKER
        )
        self.assertIn("implementation", labels["other_in_implementation"])

    def test_falls_back_to_the_bare_members_when_the_column_is_missing(self) -> None:
        documents = pd.DataFrame({"document_id": ["document_1"]})
        colors, markers, labels = {}, {}, {}

        with redirect_stdout(StringIO()) as output:
            keys = plot_classifications.group_series_keys(
                "implementation",
                ["job_displacement"],
                documents,
                colors,
                markers,
                labels,
            )

        self.assertEqual(keys, ["job_displacement"])
        self.assertNotIn("other_in_implementation", colors)
        self.assertIn("has_other_in_implementation", output.getvalue())


class Level2OtherBucketFigureTests(unittest.TestCase):
    """End to end: a level-2 category figure shows its own "other" share."""

    def _argv(self, tmp_dir: Path) -> list[str]:
        return [
            "--source",
            SOURCE,
            "--clause-type",
            PROVISION_TYPE,
            "--level",
            "2",
            "--input-dir",
            str(tmp_dir),
            "--output-dir",
            str(tmp_dir / "fig"),
            "--min-cbas",
            "1",
            "--min-group-docs",
            "1",
        ]

    def test_the_group_figure_reports_the_others_own_share_not_the_absence(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            frame = pd.DataFrame(
                {
                    "document_id": ["document_1", "document_2", "document_3"],
                    "meta_matched": [True, True, True],
                    "expire_period": ["2005-2009", "2010-2014", "2010-2014"],
                    "sector_label": ["Manufacturing", "Retail", "Retail"],
                    "n_provisions": [1, 1, 0],
                    "n_job_displacement": [1, 0, 0],
                    "has_job_displacement": [True, False, False],
                    "n_other_in_implementation": [0, 1, 0],
                    "has_other_in_implementation": [False, True, False],
                }
            )
            prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 2)
            frame.to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)

            output = StringIO()
            with redirect_stdout(output):
                status = plot_classifications.main(self._argv(tmp_dir))

            captured = output.getvalue()

            self.assertEqual(status, 0)
            # No fallback warning: the column is present, so the real
            # other-bucket share is drawn rather than the group being reduced
            # to its bare members.
            self.assertNotIn(
                "carries no has_other_in_implementation column", captured
            )
            stem = f"{PROVISION_TYPE}_{SOURCE}_l2_implementation"
            self.assertTrue((tmp_dir / "fig" / f"{stem}_subtype_share.png").is_file())


class FocusCategoryFigureTests(unittest.TestCase):
    """End to end: --focus-category draws the beneficiary-scoped count series."""

    def _argv(self, tmp_dir: Path) -> list[str]:
        return [
            "--source",
            SOURCE,
            "--clause-type",
            PROVISION_TYPE,
            "--level",
            "2",
            "--input-dir",
            str(tmp_dir),
            "--output-dir",
            str(tmp_dir / "fig"),
            "--min-cbas",
            "1",
            "--min-group-docs",
            "1",
            "--focus-category",
            "preemptive_rights",
        ]

    def test_draws_the_figure_from_the_provisions_table(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            documents = pd.DataFrame(
                {
                    "document_id": ["document_1", "document_2"],
                    "meta_matched": [True, True],
                    "expire_period": ["2005-2009", "2010-2014"],
                    "sector_label": ["Manufacturing", "Retail"],
                    "n_provisions": [2, 1],
                    "n_technology_restriction": [1, 0],
                    "has_technology_restriction": [True, False],
                    "n_other_in_preemptive_rights": [1, 1],
                    "has_other_in_preemptive_rights": [True, True],
                }
            )
            provisions = pd.DataFrame(
                {
                    "document_id": ["document_1", "document_1", "document_2"],
                    "subtype_1": [
                        "preemptive_rights",
                        "preemptive_rights",
                        "preemptive_rights",
                    ],
                    "subtype": ["technology_restriction", "other", "other"],
                    "beneficiary": ["employer", "employer", "workers"],
                    "expire_period": ["2005-2009", "2005-2009", "2010-2014"],
                }
            )
            prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 2)
            documents.to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)
            provisions.to_csv(tmp_dir / f"{prefix}_provisions.csv", index=False)

            output = StringIO()
            with redirect_stdout(output):
                status = plot_classifications.main(self._argv(tmp_dir))

            captured = output.getvalue()

            self.assertEqual(status, 0)
            self.assertNotIn("carries no subtype_1 column", captured)
            stem = f"{PROVISION_TYPE}_{SOURCE}_l2_preemptive_rights"
            self.assertTrue(
                (tmp_dir / "fig" / f"{stem}_counts_by_period.png").is_file()
            )

    def test_rejects_a_category_outside_the_taxonomy(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            argv = self._argv(tmp_dir)
            argv[argv.index("preemptive_rights")] = "not_a_real_category"
            with self.assertRaises(SystemExit):
                plot_classifications.main(argv)


class FocusCategoryStyleTests(unittest.TestCase):
    """Unit-level: no combined total, colour is beneficiary, line style is named-vs-other."""

    def _provisions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "document_id": ["d1", "d1", "d2", "d2"],
                "subtype_1": ["preemptive_rights"] * 4,
                "subtype": [
                    "technology_restriction",
                    "other",
                    "technology_restriction",
                    "other",
                ],
                "beneficiary": ["employer", "employer", "workers", "workers"],
                "expire_period": ["2005-2009"] * 4,
            }
        )

    def test_four_lines_no_all_series_solid_named_dotted_other(self) -> None:
        fig = plot_classifications.plot_focus_category_counts_by_period(
            self._provisions(), "preemptive_rights", min_docs=1
        )
        self.assertIsNotNone(fig)
        lines = fig.axes[0].get_lines()
        # One (named, other) pair per beneficiary -- no separate "all" line.
        self.assertEqual(len(lines), 4)

        colors, _, _ = plot_classifications.beneficiary_style(
            plot_classifications.FOCUS_BENEFICIARIES
        )
        styles = {(line.get_color(), line.get_linestyle()) for line in lines}
        for beneficiary in plot_classifications.FOCUS_BENEFICIARIES:
            self.assertIn((colors[beneficiary], "-"), styles)
            self.assertIn(
                (colors[beneficiary], plot_classifications.ADJUSTED_LINESTYLE),
                styles,
            )

    def test_beneficiary_colors_match_the_beneficiary_share_figures(self) -> None:
        fig = plot_classifications.plot_focus_category_counts_by_period(
            self._provisions(), "preemptive_rights", min_docs=1
        )
        colors, _, _ = plot_classifications.beneficiary_style(
            plot_classifications.BENEFICIARY_COLOR_OVERRIDES.keys()
        )
        line_colors = {line.get_color() for line in fig.axes[0].get_lines()}
        self.assertIn(colors["employer"], line_colors)
        self.assertIn(colors["workers"], line_colors)


class Level1BeneficiaryEraHeatmapTests(unittest.TestCase):
    """level1_beneficiary_era_shares and the heatmap built on top of it."""

    def _provisions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "document_id": ["d1", "d1", "d2", "d3", "d4", "d5", "d6", "d7"],
                "beneficiary": [
                    "employer",
                    "employer",
                    "employer",
                    "employer",
                    "workers",
                    "workers",
                    "unclear",
                    "employer",
                ],
                "subtype": [
                    "preemptive_rights",
                    "other",
                    "preemptive_rights",
                    "other",
                    "implementation",
                    "implementation",
                    "preemptive_rights",
                    "preemptive_rights",
                ],
                "expire_year": [
                    2005.0,
                    2005.0,
                    2008.0,
                    2020.0,
                    2010.0,
                    2025.0,
                    2005.0,
                    1990.0,
                ],
            }
        )

    def test_shares_are_document_level_within_each_columns_own_denominator(
        self,
    ) -> None:
        shares, totals, excluded_docs = plot_classifications.level1_beneficiary_era_shares(
            self._provisions(), ["preemptive_rights", "implementation"]
        )

        self.assertEqual(
            list(shares.columns),
            [
                "Employer (00-14)",
                "Employer (15-29)",
                "Workers (00-14)",
                "Workers (15-29)",
            ],
        )
        self.assertEqual(
            list(shares.index), ["preemptive_rights", "implementation", "other"]
        )
        # d1 (two rows) and d2 carry an employer provision in 2000-2014 --
        # denominator 2, both with a preemptive_rights row (100%), only d1
        # with an "other" row (50%).
        self.assertEqual(totals["Employer (00-14)"], 2)
        self.assertEqual(shares.loc["preemptive_rights", "Employer (00-14)"], 100.0)
        self.assertEqual(shares.loc["other", "Employer (00-14)"], 50.0)
        self.assertEqual(shares.loc["implementation", "Employer (00-14)"], 0.0)
        # d3 alone carries an employer provision in 2015-2029.
        self.assertEqual(totals["Employer (15-29)"], 1)
        self.assertEqual(shares.loc["other", "Employer (15-29)"], 100.0)
        self.assertEqual(totals["Workers (00-14)"], 1)
        self.assertEqual(shares.loc["implementation", "Workers (00-14)"], 100.0)
        self.assertEqual(totals["Workers (15-29)"], 1)
        self.assertEqual(shares.loc["implementation", "Workers (15-29)"], 100.0)
        # d6 is out of scope entirely (unclear beneficiary); d7 is an in-scope
        # beneficiary whose only provision falls outside both eras -- only the
        # latter counts as an excluded document.
        self.assertEqual(excluded_docs, 1)

    def test_figure_annotates_every_cell_and_draws_a_colorbar(self) -> None:
        fig = plot_classifications.plot_level1_beneficiary_era_heatmap(
            self._provisions(), ["preemptive_rights", "implementation"], "technology"
        )

        self.assertIsNotNone(fig)
        # 3 rows (2 named + "other") x 4 columns = 12 annotated cells.
        self.assertEqual(len(fig.axes[0].texts), 12)
        # A colorbar adds its own axes alongside the heatmap's.
        self.assertEqual(len(fig.axes), 2)

    def test_missing_expire_year_warns_and_skips(self) -> None:
        provisions = self._provisions().drop(columns=["expire_year"])
        output = StringIO()
        with redirect_stdout(output):
            fig = plot_classifications.plot_level1_beneficiary_era_heatmap(
                provisions, ["preemptive_rights", "implementation"], "technology"
            )

        self.assertIsNone(fig)
        self.assertIn("carries no expire_year column", output.getvalue())


class Level1HeatmapFigureTests(unittest.TestCase):
    """End to end: --level1-heatmap always reads the level-1 provisions table."""

    def _argv(self, tmp_dir: Path) -> list[str]:
        return [
            "--source",
            SOURCE,
            "--clause-type",
            PROVISION_TYPE,
            "--level",
            "1",
            "--input-dir",
            str(tmp_dir),
            "--output-dir",
            str(tmp_dir / "fig"),
            "--min-cbas",
            "1",
            "--min-group-docs",
            "1",
            "--level1-heatmap",
        ]

    def test_writes_the_heatmap_from_the_level1_provisions_table(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            documents = pd.DataFrame(
                {
                    "document_id": ["document_1", "document_2"],
                    "meta_matched": [True, True],
                    "expire_period": ["2005-2009", "2020-2024"],
                    "sector_label": ["Manufacturing", "Retail"],
                    "n_provisions": [1, 1],
                    "n_preemptive_rights": [1, 0],
                    "has_preemptive_rights": [True, False],
                    "n_other": [0, 1],
                    "has_other": [False, True],
                }
            )
            provisions = pd.DataFrame(
                {
                    "document_id": ["document_1", "document_2"],
                    "subtype": ["preemptive_rights", "other"],
                    "beneficiary": ["employer", "workers"],
                    "expire_year": [2005.0, 2020.0],
                }
            )
            prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 1)
            documents.to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)
            provisions.to_csv(tmp_dir / f"{prefix}_provisions.csv", index=False)

            output = StringIO()
            with redirect_stdout(output):
                status = plot_classifications.main(self._argv(tmp_dir))

            self.assertEqual(status, 0)
            self.assertTrue(
                (tmp_dir / "fig" / f"{prefix}_beneficiary_era_heatmap.png").is_file()
            )

    def test_missing_level1_table_warns_instead_of_failing(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            documents = pd.DataFrame(
                {
                    "document_id": ["document_1"],
                    "meta_matched": [True],
                    "expire_period": ["2005-2009"],
                    "sector_label": ["Manufacturing"],
                    "n_provisions": [1],
                    "n_preemptive_rights": [1],
                    "has_preemptive_rights": [True],
                }
            )
            prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, 1)
            documents.to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)
            # No provisions CSV written at all.

            output = StringIO()
            with redirect_stdout(output):
                status = plot_classifications.main(self._argv(tmp_dir))

            self.assertEqual(status, 0)
            self.assertIn("level-1 provisions table not found", output.getvalue())
            self.assertFalse(
                (tmp_dir / "fig" / f"{prefix}_beneficiary_era_heatmap.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
