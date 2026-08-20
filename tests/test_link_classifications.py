from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from pipeline.utils import link_classifications

SOURCE = "dol_archive"
MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
MODEL_DIR = "Qwen_Qwen3.8-27B-FP8"
PROVISION_TYPE = "technology"

METADATA_HEADER = (
    "employername,cbafile,location,union,expirationdate,naics,wrkrs,type,expire_year\n"
)


def _record(
    document_id: str,
    subtype: str,
    span_start: int = 0,
    child: str | None = None,
    beneficiary: str = "worker",
) -> str:
    """A stage-3 record with one column per taxonomy level.

    ``child`` is None when the cascade stopped at level 1, which is what stage 3
    writes for a leaf or for the "other" escape hatch.
    """

    return json.dumps(
        {
            "beneficiary": beneficiary,
            "document_id": document_id,
            "extract_model_name": MODEL_NAME,
            "extraction_class": PROVISION_TYPE,
            "extraction_text": f"clause about {subtype}",
            "model_name": MODEL_NAME,
            "source": SOURCE,
            "span_end": span_start + 10,
            "span_start": span_start,
            "taxonomy_depth": 2,
            "subtype_1": subtype,
            "subtype_2": child,
        }
    )


class RepairTrailingColumnsTests(unittest.TestCase):
    def test_complete_row_is_unchanged(self) -> None:
        self.assertEqual(
            link_classifications.repair_trailing_columns("484230", "608", "PRIVATE"),
            ("484230", "608", "PRIVATE"),
        )

    def test_missing_naics_and_workers_shifts_ownership_left(self) -> None:
        self.assertEqual(
            link_classifications.repair_trailing_columns("PRIVATE", "", ""),
            ("", "", "PRIVATE"),
        )

    def test_missing_workers_shifts_ownership_into_workers(self) -> None:
        self.assertEqual(
            link_classifications.repair_trailing_columns("484230", "PRIVATE", ""),
            ("484230", "", "PRIVATE"),
        )

    def test_lone_six_digit_value_is_a_naics_code(self) -> None:
        self.assertEqual(
            link_classifications.repair_trailing_columns("445110", "PUBLIC", ""),
            ("445110", "", "PUBLIC"),
        )

    def test_lone_short_value_is_a_worker_count(self) -> None:
        # A worker count can look like a two-digit NAICS prefix; only a full
        # six-digit code is treated as an industry code.
        self.assertEqual(
            link_classifications.repair_trailing_columns("48", "PRIVATE", ""),
            ("", "48", "PRIVATE"),
        )

    def test_all_blank_row_does_not_raise(self) -> None:
        self.assertEqual(
            link_classifications.repair_trailing_columns("", "", ""), ("", "", "")
        )


class SectorLabelTests(unittest.TestCase):
    def test_manufacturing_prefixes_share_one_sector(self) -> None:
        labels = {
            link_classifications.sector_label(code)
            for code in ("311111", "321113", "336411")
        }
        self.assertEqual(labels, {"Manufacturing"})

    def test_blank_naics_is_unknown(self) -> None:
        self.assertEqual(
            link_classifications.sector_label(""), link_classifications.UNKNOWN_SECTOR
        )

    def test_unmapped_prefix_is_reported_verbatim(self) -> None:
        self.assertEqual(link_classifications.sector_label("991111"), "NAICS 99")


class LoadMetadataTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _load(self, rows: str) -> pd.DataFrame:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "meta.csv"
            self._write(path, METADATA_HEADER + rows)
            with redirect_stdout(StringIO()):
                return link_classifications.load_metadata(path)

    def test_every_shift_pattern_recovers_ownership(self) -> None:
        frame = self._load(
            "Complete Co,1,IL,IBEW,\"Sep 30, 2026\",484230,608,PRIVATE,2026\n"
            "No Naics Co,2,NV,IBEW,\"Mar 31, 2019\",48,PRIVATE,,2019\n"
            "No Naics No Wrkrs Co,3,PA,IBEW,\"Dec 31, 2022\",PUBLIC,,,2022\n"
        )
        self.assertEqual(list(frame["ownership"]), ["PRIVATE", "PRIVATE", "PUBLIC"])
        self.assertEqual(list(frame["naics"]), ["484230", "", ""])
        self.assertEqual(list(frame["wrkrs"]), ["608", "48", ""])

    def test_document_id_has_no_float_suffix(self) -> None:
        frame = self._load("Seven Co,7,IL,IBEW,\"Sep 30, 2026\",484230,608,PRIVATE,2026\n")
        self.assertEqual(list(frame["document_id"]), ["document_7"])

    def test_out_of_range_expire_year_becomes_unknown(self) -> None:
        frame = self._load(
            "Sentinel Co,1,IL,IBEW,\"Sep 30, 1800\",484230,608,PRIVATE,1800\n"
            "Real Co,2,IL,IBEW,\"Sep 30, 2007\",484230,608,PRIVATE,2007\n"
        )
        self.assertTrue(pd.isna(frame["expire_year"].iloc[0]))
        self.assertEqual(frame["expire_period"].iloc[0], "")
        self.assertEqual(frame["expire_period"].iloc[1], "2005-2009")

    def test_duplicate_cbafile_rows_collapse_with_a_warning(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "meta.csv"
            self._write(
                path,
                METADATA_HEADER
                + "First Co,7,IL,IBEW,\"Sep 30, 2026\",484230,608,PRIVATE,2026\n"
                + "Second Co,7,NY,IBEW,\"Sep 30, 2024\",445110,10,PUBLIC,2024\n",
            )
            output = StringIO()
            with redirect_stdout(output):
                frame = link_classifications.load_metadata(path)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["employername"].iloc[0], "First Co")
        self.assertIn("duplicate metadata document ID", output.getvalue())
        self.assertIn("document_7", output.getvalue())


class BuildTablesTests(unittest.TestCase):
    source = SOURCE

    @staticmethod
    def _write(path: Path, text: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _build_cache(self, cache_dir: Path) -> Path:
        classify_dir = (
            cache_dir / link_classifications.CLASSIFY_STAGE_NAME / self.source / MODEL_DIR
        )
        # doc_1 repeats one subtype: it must still count as a single document.
        # The last row stops at level 1, so it drops out of a --level 2 run.
        # Beneficiaries are 3 employer, 1 worker, so its own share is 75%/25%.
        self._write(
            classify_dir / "document_1" / f"{PROVISION_TYPE}.jsonl",
            "\n".join(
                [
                    _record(
                        "document_1", "implementation", 0, "job_displacement",
                        beneficiary="employer",
                    ),
                    _record(
                        "document_1", "implementation", 20, "job_displacement",
                        beneficiary="employer",
                    ),
                    _record(
                        "document_1", "implementation", 40, "job_displacement",
                        beneficiary="employer",
                    ),
                    _record(
                        "document_1", "preemptive_rights", 60, "notification_right",
                        beneficiary="worker",
                    ),
                ]
            )
            + "\n",
        )
        # doc_2 has no metadata row; doc_3 is a genuine zero.
        self._write(
            classify_dir / "document_9999" / f"{PROVISION_TYPE}.jsonl",
            _record("document_9999", "other") + "\n",
        )
        self._write(classify_dir / "document_3" / f"{PROVISION_TYPE}.jsonl", "")
        return classify_dir

    def _build_metadata(self, path: Path) -> None:
        self._write(
            path,
            METADATA_HEADER
            + "One Co,1,IL,IBEW,\"Sep 30, 2007\",311111,608,PRIVATE,2007\n"
            + "Three Co,3,NY,IBEW,\"Sep 30, 2019\",PUBLIC,,,2019\n",
        )

    def _tables(self, tmp_dir: Path):
        classify_dir = self._build_cache(tmp_dir / "cache")
        metadata_path = tmp_dir / "meta.csv"
        self._build_metadata(metadata_path)
        subtypes = ["preemptive_rights", "implementation", "workforce_management", "other"]
        provisions = link_classifications.load_classifications(
            classify_dir, PROVISION_TYPE
        )
        provisions, _ = link_classifications.select_level(provisions, 1)
        with redirect_stdout(StringIO()):
            metadata = link_classifications.load_metadata(metadata_path)
        documents = link_classifications.build_document_table(
            link_classifications.discover_document_ids(classify_dir),
            provisions,
            metadata,
            subtypes,
        )
        return documents, provisions, subtypes

    def test_repeated_subtype_counts_once_at_document_level(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            documents, _, subtypes = self._tables(Path(tmp_dir))

        row = documents[documents["document_id"] == "document_1"].iloc[0]
        self.assertEqual(row["n_implementation"], 3)
        self.assertEqual(row["n_provisions"], 4)
        self.assertTrue(row["has_implementation"])
        self.assertTrue(row["has_preemptive_rights"])
        # Four provisions, but only two document-subtype pairs.
        pairs = sum(int(row[f"has_{subtype}"]) for subtype in subtypes)
        self.assertEqual(pairs, 2)

    def test_beneficiary_share_is_the_document_own_percentage(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            documents, _, _ = self._tables(Path(tmp_dir))

        row = documents[documents["document_id"] == "document_1"].iloc[0]
        self.assertEqual(row["n_beneficiary_employer"], 3)
        self.assertEqual(row["n_beneficiary_worker"], 1)
        self.assertEqual(row["n_beneficiary_unclear"], 0)
        self.assertAlmostEqual(row["pct_beneficiary_employer"], 75.0)
        self.assertAlmostEqual(row["pct_beneficiary_worker"], 25.0)
        self.assertAlmostEqual(row["pct_beneficiary_unclear"], 0.0)

    def test_beneficiary_share_is_undefined_not_zero_with_no_provisions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            documents, _, _ = self._tables(Path(tmp_dir))

        row = documents[documents["document_id"] == "document_3"].iloc[0]
        self.assertEqual(row["n_provisions"], 0)
        for beneficiary in link_classifications.BENEFICIARY_LABELS:
            self.assertTrue(pd.isna(row[f"pct_beneficiary_{beneficiary}"]))

    def test_empty_classification_file_keeps_a_zero_document_row(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            documents, _, subtypes = self._tables(Path(tmp_dir))

        self.assertEqual(len(documents), 3)
        row = documents[documents["document_id"] == "document_3"].iloc[0]
        self.assertEqual(row["n_provisions"], 0)
        self.assertFalse(any(row[f"has_{subtype}"] for subtype in subtypes))
        self.assertTrue(row["meta_matched"])

    def test_document_without_metadata_is_kept_and_flagged(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            documents, _, _ = self._tables(Path(tmp_dir))

        row = documents[documents["document_id"] == "document_9999"].iloc[0]
        self.assertFalse(row["meta_matched"])
        self.assertEqual(row["n_provisions"], 1)

    def test_metadata_columns_are_joined_onto_documents(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            documents, _, _ = self._tables(Path(tmp_dir))

        row = documents[documents["document_id"] == "document_1"].iloc[0]
        self.assertEqual(row["employername"], "One Co")
        self.assertEqual(row["sector_label"], "Manufacturing")
        self.assertEqual(row["expire_period"], "2005-2009")
        self.assertEqual(row["ownership"], "PRIVATE")

    def test_malformed_json_names_the_file_and_line(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            classify_dir = self._build_cache(Path(tmp_dir) / "cache")
            path = classify_dir / "document_1" / f"{PROVISION_TYPE}.jsonl"
            path.write_text(path.read_text() + "{not json}\n", encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                link_classifications.load_classifications(classify_dir, PROVISION_TYPE)

        self.assertIn(f"{PROVISION_TYPE}.jsonl:5", str(caught.exception))


class MainTests(BuildTablesTests):
    def _argv(self, tmp_dir: Path, *extra: str) -> list[str]:
        return [
            "--source",
            self.source,
            "--model-name",
            MODEL_NAME,
            "--provision-type",
            PROVISION_TYPE,
            "--cache-dir",
            str(tmp_dir / "cache"),
            "--metadata",
            str(tmp_dir / "meta.csv"),
            "--output-dir",
            str(tmp_dir / "out"),
            *extra,
        ]

    def test_writes_both_tables(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._build_cache(tmp_dir / "cache")
            self._build_metadata(tmp_dir / "meta.csv")

            output = StringIO()
            with redirect_stdout(output):
                status = link_classifications.main(self._argv(tmp_dir))

            documents = pd.read_csv(tmp_dir / "out" / "technology_dol_archive_l1_documents.csv")
            provisions = pd.read_csv(
                tmp_dir / "out" / "technology_dol_archive_l1_provisions.csv"
            )
            summary = output.getvalue()

        self.assertEqual(status, 0)
        self.assertEqual(len(documents), 3)
        self.assertEqual(len(provisions), 5)
        self.assertNotIn("extraction_text", provisions.columns)
        self.assertIn("document-subtype pairs:      3", summary)

    def test_level_two_reports_child_labels_and_drops_unlabelled_rows(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._build_cache(tmp_dir / "cache")
            self._build_metadata(tmp_dir / "meta.csv")

            output = StringIO()
            with redirect_stdout(output):
                status = link_classifications.main(
                    self._argv(tmp_dir, "--level", "2")
                )

            documents = pd.read_csv(
                tmp_dir / "out" / "technology_dol_archive_l2_documents.csv"
            )
            provisions = pd.read_csv(
                tmp_dir / "out" / "technology_dol_archive_l2_provisions.csv"
            )
            summary = output.getvalue()

        self.assertEqual(status, 0)
        # Level 1's tables are untouched: the two levels do not share a filename.
        self.assertFalse(
            (tmp_dir / "out" / "technology_dol_archive_l1_documents.csv").exists()
        )
        # The "other" row carries no level-2 label and is dropped, not miscounted.
        self.assertEqual(len(provisions), 4)
        self.assertIn("dropped, no label at level:  1", summary)
        self.assertEqual(
            sorted(provisions["subtype"].unique()),
            ["job_displacement", "notification_right"],
        )
        # Columns key off the level-2 labels now, not the top-level classes.
        self.assertIn("has_job_displacement", documents.columns)
        self.assertNotIn("has_implementation", documents.columns)
        row = documents[documents["document_id"] == "document_1"].iloc[0]
        self.assertEqual(row["n_job_displacement"], 3)
        self.assertEqual(row["n_notification_right"], 1)

    def test_level_beyond_the_taxonomy_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._build_cache(tmp_dir / "cache")
            self._build_metadata(tmp_dir / "meta.csv")

            with self.assertRaisesRegex(SystemExit, "declares 2 level"):
                with redirect_stdout(StringIO()):
                    link_classifications.main(self._argv(tmp_dir, "--level", "3"))

    def test_include_text_adds_the_quoted_provision(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._build_cache(tmp_dir / "cache")
            self._build_metadata(tmp_dir / "meta.csv")

            with redirect_stdout(StringIO()):
                link_classifications.main(self._argv(tmp_dir, "--include-text"))

            provisions = pd.read_csv(
                tmp_dir / "out" / "technology_dol_archive_l1_provisions.csv"
            )

        self.assertIn("extraction_text", provisions.columns)

    def test_dry_run_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            self._build_cache(tmp_dir / "cache")
            self._build_metadata(tmp_dir / "meta.csv")

            with redirect_stdout(StringIO()):
                status = link_classifications.main(self._argv(tmp_dir, "--dry-run"))

            wrote_anything = (tmp_dir / "out").exists()

        self.assertEqual(status, 0)
        self.assertFalse(wrote_anything)


if __name__ == "__main__":
    unittest.main()
