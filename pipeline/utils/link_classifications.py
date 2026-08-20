"""Link stage-03 classifications to the DOL CBA metadata list.

Reads every ``<document_id>/<provision_type>.jsonl`` under a stage-03
classification directory and joins it to ``meta_data/CBAList_with_statefips.dta``
on ``document_id == f"document_{cbafile}"``, writing two CSV tables.

The document table is the primary output: one row per classified document,
including the documents where extraction found nothing.  Its ``has_<subtype>``
columns are the document-level counting unit the EDA figures are built from --
a document counts once per subtype however many provisions of that subtype it
contains.  Its ``pct_beneficiary_<label>`` columns are each document's own
share of provisions naming that beneficiary, the per-CBA percentage a
mean-of-shares figure averages across CBAs.

Run from anywhere in the repository::

    uv run python pipeline/utils/link_classifications.py
    uv run python pipeline/utils/link_classifications.py --provision-type technology
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from pipeline.stg_02_extract.structure_provision import (
    RESERVED_SUBTYPE_LABEL,
    labels_at_level,
    load_provision,
    taxonomy_depth,
)
from pipeline.utils.paths import (
    PROJECT_ROOT,
    default_cache_dir,
    path_safe_model_name,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(PROJECT_ROOT / ".env")

CLASSIFY_STAGE_NAME = "stg_03_classify"
DEFAULT_SOURCE = "dol_archive"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_PROVISION_TYPE = "technology"
DEFAULT_METADATA = PROJECT_ROOT / "meta_data" / "CBAList_with_statefips.dta"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"

# `expire_year` carries sentinels (1800) and stray typos (2373, 9211); anything
# outside this window is treated as unknown rather than plotted.
MIN_EXPIRE_YEAR = 1980
MAX_EXPIRE_YEAR = 2035
PERIOD_YEARS = 5

# The metadata list carries a NAICS code but no industry name.  Two-digit
# prefixes are the NAICS sector, and manufacturing, retail, and transportation
# each span several prefixes.
NAICS_SECTORS: Mapping[str, str] = {
    "11": "Agriculture, forestry, fishing and hunting",
    "21": "Mining, quarrying, oil and gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale trade",
    "44": "Retail trade",
    "45": "Retail trade",
    "48": "Transportation and warehousing",
    "49": "Transportation and warehousing",
    "51": "Information",
    "52": "Finance and insurance",
    "53": "Real estate and rental and leasing",
    "54": "Professional, scientific, and technical services",
    "55": "Management of companies and enterprises",
    "56": "Administrative, support, and waste management",
    "61": "Educational services",
    "62": "Health care and social assistance",
    "71": "Arts, entertainment, and recreation",
    "72": "Accommodation and food services",
    "81": "Other services (except public administration)",
    "92": "Public administration",
}
UNKNOWN_SECTOR = "Unknown"

METADATA_COLUMNS = [
    "cbafile",
    "employername",
    "union",
    "location",
    "expirationdate",
    "expire_year",
    "expire_period",
    "naics",
    "naics_sector",
    "sector_label",
    "wrkrs",
    "ownership",
]
PROVISION_COLUMNS = [
    "document_id",
    "source",
    "provision_type",
    "classify_model",
    "extract_model",
    "subtype",
    "beneficiary",
    "span_start",
    "span_end",
]

# Every provision names exactly one beneficiary, unlike subtypes which a CBA can
# carry several of -- so these three are exhaustive and mutually exclusive, and
# a CBA's own pct_beneficiary_* columns sum to 100 (documents with no provision
# excepted, where the percentage is undefined rather than zero).
BENEFICIARY_LABELS = ("employer", "worker", "unclear")


def discover_document_ids(classify_dir: Path) -> list[str]:
    """Return every classified document ID, including those with no provisions.

    The zero-provision documents are the honest denominator: extraction ran on
    them and found nothing, which is different from never having been run.
    """

    if not classify_dir.is_dir():
        raise SystemExit(f"classification directory not found: {classify_dir}")
    return sorted(
        path.name
        for path in classify_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def load_classifications(classify_dir: Path, provision_type: str) -> pd.DataFrame:
    """Read every classification record under ``classify_dir``."""

    records: list[dict[str, object]] = []
    for document_id in discover_document_ids(classify_dir):
        path = classify_dir / document_id / f"{provision_type}.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return pd.DataFrame.from_records(records)


def repair_trailing_columns(
    naics: str, wrkrs: str, type_: str
) -> tuple[str, str, str]:
    """Undo the left-shift in the metadata list's last three columns.

    Missing values were dropped rather than blanked when the list was built, so
    the present values are packed to the left: a row with no NAICS code reads
    ``naics="PRIVATE"``, and one with no worker count reads ``wrkrs="PRIVATE"``.
    Re-anchor on the trailing PRIVATE/PUBLIC token instead of trusting position.
    """

    values = [value.strip() for value in (naics, wrkrs, type_) if value.strip()]
    ownership = values.pop() if values else ""
    if len(values) == 2:
        return values[0], values[1], ownership
    if len(values) == 1:
        # A lone survivor is ambiguous; only a full six-digit code is a NAICS.
        if re.fullmatch(r"\d{6}", values[0]):
            return values[0], "", ownership
        return "", values[0], ownership
    return "", "", ownership


def sector_label(naics: str) -> str:
    """Return the NAICS sector name for a code, or a readable placeholder."""

    code = str(naics).strip()[:2]
    if not code:
        return UNKNOWN_SECTOR
    return NAICS_SECTORS.get(code, f"NAICS {code}")


def _expire_period(year: float) -> str:
    if pd.isna(year):
        return ""
    start = int(year) // PERIOD_YEARS * PERIOD_YEARS
    return f"{start}-{start + PERIOD_YEARS - 1}"


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the CBA list, repair the shifted columns, and key it by document ID."""

    if not path.is_file():
        raise SystemExit(f"metadata file not found: {path}")
    # CSV support keeps the tests free of a Stata dependency.
    frame = pd.read_csv(path, dtype=str) if path.suffix == ".csv" else pd.read_stata(path)

    # Normalise every non-numeric column to stripped text.  Stata writes missing
    # strings as "", but read_csv yields NaN, and str(NaN) == "nan" would be
    # mistaken for a real value by the shift repair below.
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column]
        frame[column] = (
            values.astype(object).where(values.notna(), "").astype(str).str.strip()
        )

    repaired = frame.apply(
        lambda row: repair_trailing_columns(
            str(row.get("naics", "")),
            str(row.get("wrkrs", "")),
            str(row.get("type", "")),
        ),
        axis=1,
        result_type="expand",
    )
    frame["naics"], frame["wrkrs"], frame["ownership"] = (
        repaired[0],
        repaired[1],
        repaired[2],
    )
    frame = frame.drop(columns=["type"], errors="ignore")

    cbafile = pd.to_numeric(frame["cbafile"], errors="coerce")
    frame["document_id"] = "document_" + cbafile.astype("Int64").astype(str)
    frame = frame[cbafile.notna()]

    year = pd.to_numeric(frame["expire_year"], errors="coerce")
    frame["expire_year"] = year.where(year.between(MIN_EXPIRE_YEAR, MAX_EXPIRE_YEAR))
    frame["expire_period"] = frame["expire_year"].map(_expire_period)
    frame["naics_sector"] = frame["naics"].str[:2]
    frame["sector_label"] = frame["naics"].map(sector_label)

    duplicated = frame["document_id"][frame["document_id"].duplicated()].unique()
    if len(duplicated):
        print(
            f"  [warn] dropping {len(duplicated)} duplicate metadata document ID(s): "
            f"{', '.join(sorted(duplicated))}"
        )
    return frame.drop_duplicates("document_id", keep="first")


def _metadata_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = ["document_id"] + [
        column for column in METADATA_COLUMNS if column in metadata.columns
    ]
    return metadata[columns]


def table_prefix(provision_type: str, source: str, level: int) -> str:
    """Filename stem for one provision type, source and taxonomy level.

    The level is part of the name so tables built at different levels sit side
    by side instead of overwriting each other -- their ``has_<label>`` columns
    key off different labels and are not interchangeable.
    """

    return f"{provision_type}_{source}_l{level}"


def select_level(provisions: pd.DataFrame, level: int) -> tuple[pd.DataFrame, int]:
    """Narrow to one taxonomy level, exposing its label as ``subtype``.

    Stage 3 writes one ``subtype_<level>`` column per level it classified.  A row
    whose cascade stopped above ``level`` -- the branch bottomed out, or the model
    answered "other" higher up -- carries no label here, and is dropped rather
    than counted as though it belonged to some subtype at this level.  The caller
    reports how many were dropped.
    """

    if provisions.empty:
        return provisions, 0

    column = f"subtype_{level}"
    if column not in provisions.columns:
        raise SystemExit(
            f"classification records carry no {column} column; re-run stage 3 "
            f"with --taxonomy-depth {level}"
        )
    frame = provisions.rename(columns={column: "subtype"})
    labelled = frame["subtype"].notna()
    return frame[labelled], int((~labelled).sum())


def build_provision_table(
    provisions: pd.DataFrame,
    metadata: pd.DataFrame,
    include_text: bool,
) -> pd.DataFrame:
    """Return one row per classified provision, with its document's metadata."""

    frame = provisions.rename(
        columns={
            "extraction_class": "provision_type",
            "model_name": "classify_model",
            "extract_model_name": "extract_model",
        }
    )
    columns = list(PROVISION_COLUMNS)
    if include_text:
        columns.append("extraction_text")
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[columns]

    merged = frame.merge(_metadata_columns(metadata), on="document_id", how="left")
    merged.insert(1, "meta_matched", merged["cbafile"].notna())
    return merged


def build_document_table(
    document_ids: Sequence[str],
    provisions: pd.DataFrame,
    metadata: pd.DataFrame,
    subtypes: Sequence[str],
    beneficiaries: Sequence[str] = BENEFICIARY_LABELS,
) -> pd.DataFrame:
    """Return one row per classified document, zero-provision documents included.

    ``n_<subtype>`` counts provisions; ``has_<subtype>`` is the document-level
    unit -- three task-allocation provisions in one document are one document.

    ``n_beneficiary_<label>`` counts provisions naming that beneficiary, and
    ``pct_beneficiary_<label>`` is that count's share of the document's own
    provisions -- the per-CBA percentage a mean-of-shares figure averages
    across CBAs, not a share of the whole corpus. It is left blank (NaN) for a
    document with no provisions, where the percentage is undefined rather than
    zero.
    """

    frame = pd.DataFrame({"document_id": list(document_ids)})
    merged = frame.merge(_metadata_columns(metadata), on="document_id", how="left")
    merged.insert(1, "meta_matched", merged["cbafile"].notna())

    if provisions.empty:
        counts = pd.DataFrame(index=frame["document_id"], columns=list(subtypes))
        counts = counts.fillna(0).astype(int)
    else:
        counts = (
            pd.crosstab(provisions["document_id"], provisions["subtype"])
            .reindex(index=frame["document_id"], columns=list(subtypes))
            .fillna(0)
            .astype(int)
        )
    counts.index.name = "document_id"
    merged["n_provisions"] = counts.sum(axis=1).to_numpy()
    for subtype in subtypes:
        merged[f"n_{subtype}"] = counts[subtype].to_numpy()
    for subtype in subtypes:
        merged[f"has_{subtype}"] = merged[f"n_{subtype}"] > 0

    if provisions.empty or "beneficiary" not in provisions.columns:
        beneficiary_counts = pd.DataFrame(
            index=frame["document_id"], columns=list(beneficiaries)
        ).fillna(0).astype(int)
    else:
        beneficiary_counts = (
            pd.crosstab(provisions["document_id"], provisions["beneficiary"])
            .reindex(index=frame["document_id"], columns=list(beneficiaries))
            .fillna(0)
            .astype(int)
        )
    beneficiary_counts.index.name = "document_id"
    denominator = merged["n_provisions"].astype(float).replace(0, float("nan"))
    for beneficiary in beneficiaries:
        merged[f"n_beneficiary_{beneficiary}"] = beneficiary_counts[beneficiary].to_numpy()
        merged[f"pct_beneficiary_{beneficiary}"] = (
            beneficiary_counts[beneficiary].to_numpy() / denominator * 100
        )
    return merged


def _print_summary(
    documents: pd.DataFrame,
    provisions: pd.DataFrame,
    subtypes: Sequence[str],
    unlabelled: int = 0,
    beneficiaries: Sequence[str] = BENEFICIARY_LABELS,
) -> None:
    with_any = int((documents["n_provisions"] > 0).sum())
    pairs = int(sum(documents[f"has_{subtype}"].sum() for subtype in subtypes))
    print(f"documents classified:        {len(documents)}")
    print(f"documents with a provision:  {with_any}")
    print(f"provisions:                  {len(provisions)}")
    if unlabelled:
        print(f"dropped, no label at level:  {unlabelled}")
    print(f"document-subtype pairs:      {pairs}")
    print(f"documents without metadata:  {int((~documents['meta_matched']).sum())}")
    # Deeper levels carry longer labels, so size the column to the widest one
    # rather than truncating the table's alignment away.
    width = max((len(subtype) for subtype in subtypes), default=0) + 2
    print(
        f"\n{'subtype':<{width}}  docs   % of with-any   % of classified   provisions"
    )
    for subtype in sorted(subtypes, key=lambda s: -int(documents[f"has_{s}"].sum())):
        docs = int(documents[f"has_{subtype}"].sum())
        share_any = 100 * docs / with_any if with_any else 0.0
        share_all = 100 * docs / len(documents) if len(documents) else 0.0
        count = int(documents[f"n_{subtype}"].sum())
        print(
            f"{subtype:<{width}}{docs:>6}{share_any:>15.1f}{share_all:>18.1f}{count:>13}"
        )

    with_provisions = documents[documents["n_provisions"] > 0]
    print(
        f"\nmean within-CBA beneficiary share, {len(with_provisions)} CBAs with a "
        "provision:"
    )
    for beneficiary in beneficiaries:
        column = f"pct_beneficiary_{beneficiary}"
        if column not in with_provisions.columns or with_provisions.empty:
            continue
        print(f"  {beneficiary:<10}{with_provisions[column].mean():>6.1f}%")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"cache source folder (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"classification model (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--provision-type",
        default=DEFAULT_PROVISION_TYPE,
        help=f"provision type to link (default: {DEFAULT_PROVISION_TYPE})",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help=(
            "taxonomy level to analyse: 1 is the top-level class, 2 its "
            "children, and so on (default: 1)"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="cache dir holding the classification stage (default: CACHE_DIR from .env)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"CBA metadata list (default: {DEFAULT_METADATA.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for the CSV tables (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="keep the quoted provision text in the provisions CSV (default: off)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the summary without writing any CSV",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    spec = load_provision(args.provision_type)
    if not spec.subtype_taxonomy:
        raise SystemExit(f"provision {args.provision_type} declares no subtypes")
    if args.level < 1:
        raise SystemExit("--level must be at least 1")
    available = taxonomy_depth(spec.subtype_taxonomy)
    if args.level > available:
        raise SystemExit(
            f"--level {args.level} exceeds the {args.provision_type} taxonomy, "
            f"which declares {available} level(s)"
        )
    # "other" is offered at every level by stage 3, so it is a label that can
    # appear in the records even though the config never declares it.
    subtypes = [
        *labels_at_level(spec.subtype_taxonomy, args.level),
        RESERVED_SUBTYPE_LABEL,
    ]

    cache_dir = args.cache_dir.expanduser() if args.cache_dir else default_cache_dir()
    classify_dir = (
        cache_dir
        / CLASSIFY_STAGE_NAME
        / args.source
        / path_safe_model_name(args.model_name)
    )
    print(f"reading {classify_dir}")

    document_ids = discover_document_ids(classify_dir)
    provisions = load_classifications(classify_dir, args.provision_type)
    provisions, unlabelled = select_level(provisions, args.level)
    metadata = load_metadata(args.metadata.expanduser())

    documents = build_document_table(
        document_ids, provisions, metadata, subtypes, BENEFICIARY_LABELS
    )
    provision_table = build_provision_table(provisions, metadata, args.include_text)

    print()
    _print_summary(documents, provision_table, subtypes, unlabelled, BENEFICIARY_LABELS)

    prefix = table_prefix(args.provision_type, args.source, args.level)
    outputs = {
        args.output_dir / f"{prefix}_documents.csv": documents,
        args.output_dir / f"{prefix}_provisions.csv": provision_table,
    }
    print()
    for path, table in outputs.items():
        if args.dry_run:
            print(f"[dry-run] would write {len(table)} row(s) to {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"Wrote {len(table)} row(s) to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
