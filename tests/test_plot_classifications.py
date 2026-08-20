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
            "--provision-type",
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

            def spy(subtypes, provision_type):
                styled.append(list(subtypes))
                return original(subtypes, provision_type)

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
            "pct_beneficiary_worker": [25.0, 25.0, 100.0, None],
            "pct_beneficiary_unclear": [0.0, 0.0, 0.0, None],
        }
    )


class BeneficiaryStyleTests(unittest.TestCase):
    def test_colors_markers_and_labels_are_assigned_per_beneficiary(self) -> None:
        colors, markers, labels = plot_classifications.beneficiary_style(
            ("employer", "worker", "unclear")
        )

        self.assertEqual(len(set(colors.values())), 3)
        self.assertEqual(len(set(markers.values())), 3)
        self.assertEqual(labels["employer"], "Employer")
        # No absence series here -- every provision names exactly one party.
        self.assertNotIn(plot_classifications.NONE_KEY, colors)


class BeneficiaryMeanTests(unittest.TestCase):
    def test_overall_mean_excludes_cbas_with_no_provisions(self) -> None:
        documents = _beneficiary_documents()

        means, n = plot_classifications.beneficiary_overall_means(
            documents, ("employer", "worker", "unclear")
        )

        # document_4 has no provisions and must not pull the mean toward zero.
        self.assertEqual(n, 3)
        self.assertAlmostEqual(means["employer"], (75.0 + 75.0 + 0.0) / 3)
        self.assertAlmostEqual(means["worker"], (25.0 + 25.0 + 100.0) / 3)

    def test_group_mean_is_the_mean_of_shares_not_a_pooled_share(self) -> None:
        documents = _beneficiary_documents()

        means, totals = plot_classifications.beneficiary_group_means(
            documents, "expire_period", ("employer", "worker", "unclear")
        )

        self.assertEqual(totals["2005-2009"], 2)
        self.assertAlmostEqual(means.loc["2005-2009", "employer"], 75.0)
        self.assertEqual(totals["2010-2014"], 1)
        self.assertAlmostEqual(means.loc["2010-2014", "employer"], 0.0)
        self.assertAlmostEqual(means.loc["2010-2014", "worker"], 100.0)


class BeneficiaryFigureTests(unittest.TestCase):
    def test_by_period_excludes_the_provision_less_cba_from_its_cohort_count(self) -> None:
        documents = _beneficiary_documents()
        colors, markers, labels = plot_classifications.beneficiary_style(
            ("employer", "worker", "unclear")
        )

        figure = plot_classifications.plot_beneficiary_by_period(
            documents, ("employer", "worker", "unclear"), colors, markers, labels,
            min_docs=1, provision_type=PROVISION_TYPE,
        )

        ax = figure.axes[0]
        labels_seen = [tick.get_text() for tick in ax.get_xticklabels()]
        self.assertIn("2010-2014\n1 CBAs", labels_seen)

    def test_no_naics_row_is_shown_by_default_and_hidden_on_request(self) -> None:
        documents = _beneficiary_documents()
        documents.loc[2:, "sector_label"] = plot_classifications.UNKNOWN_SECTOR
        colors, _, labels = plot_classifications.beneficiary_style(
            ("employer", "worker", "unclear")
        )

        def rows(show_unknown):
            with redirect_stdout(StringIO()):
                figure = plot_classifications.plot_beneficiary_by_sector(
                    documents, ("employer", "worker", "unclear"), colors, labels,
                    min_docs=1, provision_type=PROVISION_TYPE,
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
            ("employer", "worker", "unclear")
        )

        figure = plot_classifications.plot_beneficiary_by_sector(
            documents, ("employer", "worker", "unclear"), colors, labels,
            min_docs=10, provision_type=PROVISION_TYPE,
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
    frame["pct_beneficiary_worker"] = 100.0 - frame["pct_beneficiary_employer"]
    frame["pct_beneficiary_unclear"] = 0.0
    frame["sector_group"] = frame["sector_label"]
    return frame


class CompositionAdjustedTests(unittest.TestCase):
    SERIES = ["employer", "worker", "unclear"]

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
            ("employer", "worker", "unclear")
        )
        with redirect_stdout(StringIO()):
            return plot_classifications.plot_beneficiary_by_period(
                self._documents(), ("employer", "worker", "unclear"),
                colors, markers, labels, min_docs=1,
                provision_type=PROVISION_TYPE, **kwargs
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
                min_docs=1, provision_type=PROVISION_TYPE,
                industry_adjusted=True, min_sector_docs=1,
            )

        # One per subtype plus the absence series.
        self.assertEqual(len(self._dotted(figure)), 2)


class BeneficiaryMainCliTests(unittest.TestCase):
    def _argv(self, tmp_dir: Path) -> list[str]:
        return [
            "--source",
            SOURCE,
            "--provision-type",
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
    def _argv(self, tmp_dir: Path, level: int) -> list[str]:
        return [
            "--source",
            SOURCE,
            "--provision-type",
            PROVISION_TYPE,
            "--level",
            str(level),
            "--input-dir",
            str(tmp_dir),
            "--output-dir",
            str(tmp_dir / "fig"),
        ]

    @staticmethod
    def _write(tmp_dir: Path, level: int, subtypes: list[str]) -> None:
        prefix = link_classifications.table_prefix(PROVISION_TYPE, SOURCE, level)
        _documents(subtypes).to_csv(tmp_dir / f"{prefix}_documents.csv", index=False)

    def test_each_level_reads_its_own_table(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._write(tmp_dir, 1, ["implementation", "preemptive_rights"])
            self._write(tmp_dir, 2, ["workforce_training", "notification_right"])

            for level in (1, 2):
                with self.subTest(level=level):
                    output = StringIO()
                    with redirect_stdout(output):
                        status = plot_classifications.main(self._argv(tmp_dir, level))

                    self.assertEqual(status, 0)
                    self.assertIn(f"_l{level}_documents.csv", output.getvalue())
                    self.assertTrue(
                        (
                            tmp_dir
                            / "fig"
                            / f"{PROVISION_TYPE}_{SOURCE}_l{level}_subtype_share.png"
                        ).is_file()
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


if __name__ == "__main__":
    unittest.main()
