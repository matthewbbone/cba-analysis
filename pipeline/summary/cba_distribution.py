from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


STATE_FIPS_TO_ABBR = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
    60: "AS",
    66: "GU",
    69: "MP",
    72: "PR",
    78: "VI",
}
US_STATE_ABBRS = set(STATE_FIPS_TO_ABBR.values())

NAICS_SECTOR_LABELS = {
    11: "Agriculture, Forestry, Fishing and Hunting",
    21: "Mining, Quarrying, and Oil and Gas Extraction",
    22: "Utilities",
    23: "Construction",
    31: "Manufacturing",
    32: "Manufacturing",
    33: "Manufacturing",
    42: "Wholesale Trade",
    44: "Retail Trade",
    45: "Retail Trade",
    48: "Transportation and Warehousing",
    49: "Transportation and Warehousing",
    51: "Information",
    52: "Finance and Insurance",
    53: "Real Estate and Rental and Leasing",
    54: "Professional, Scientific, and Technical Services",
    55: "Management of Companies and Enterprises",
    56: "Administrative and Support and Waste Management and Remediation Services",
    61: "Educational Services",
    62: "Health Care and Social Assistance",
    71: "Arts, Entertainment, and Recreation",
    72: "Accommodation and Food Services",
    81: "Other Services (except Public Administration)",
    92: "Public Administration",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cache_dir() -> str:
    return os.environ.get("CACHE_DIR", "").strip()


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _default_llm_output_dir() -> Path:
    root = _project_root()
    cache_dir = _default_cache_dir()
    if cache_dir:
        return (_resolve_path(cache_dir, root) / "04_generosity_llm_output" / "dol_archive").resolve()
    return (root / "outputs" / "04_generosity_llm_output" / "dol_archive").resolve()


def _default_dol_archive_dir() -> Path:
    return (_project_root() / "dol_archive").resolve()


def _default_output_dir() -> Path:
    return (_project_root() / "figures").resolve()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _parse_document_num(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"document_(\d+)", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _to_document_id_from_cbafile(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    numeric = _safe_float_or_none(raw_value)
    if numeric is None:
        text = str(raw_value).strip()
        if not text:
            return None
        m = re.search(r"(\d+)", text)
        if not m:
            return None
        return f"document_{int(m.group(1))}"
    return f"document_{int(numeric)}"


def _to_state_abbr_from_fips(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    numeric = _safe_float_or_none(raw_value)
    if numeric is None:
        text = str(raw_value).strip()
        if not text:
            return None
        if text.upper() in US_STATE_ABBRS:
            return text.upper()
        return None
    fips = int(numeric)
    if fips <= 0:
        return None
    return STATE_FIPS_TO_ABBR.get(fips, f"FIPS_{fips}")


def _industry_label(naics_value: Any, type_value: Any) -> str:
    naics_text = str(naics_value or "").strip()
    if naics_text and naics_text.lower() != "nan":
        m = re.search(r"\d{2,6}", naics_text)
        if m:
            sector = int(m.group(0)[:2])
            sector_label = NAICS_SECTOR_LABELS.get(sector)
            if sector_label:
                return f"{sector:02d} - {sector_label}"
            return f"{sector:02d} - Other/Unknown NAICS Sector"
        return " ".join(naics_text.split()).upper()

    type_text = str(type_value or "").strip()
    if type_text and type_text.lower() != "nan":
        return " ".join(type_text.split()).upper()
    return "UNKNOWN"


def _extract_states_from_row(row: Any, state_cols: list[str], location_col: str | None) -> list[str]:
    states = set()
    for col in state_cols:
        abbr = _to_state_abbr_from_fips(row.get(col))
        if abbr:
            states.add(abbr)

    if not states and location_col:
        raw_location = str(row.get(location_col, "")).strip().upper()
        if raw_location:
            if raw_location == "VARIES":
                states.add("VARIES")
            else:
                tokens = re.split(r"[^A-Z]+", raw_location)
                for token in tokens:
                    if token in US_STATE_ABBRS:
                        states.add(token)

    if not states:
        states.add("UNKNOWN")
    return sorted(states)


def _load_cba_metadata(dol_archive_dir: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to read CBA metadata .dta files.") from exc

    candidates = [
        dol_archive_dir / "CBAList_with_statefips.dta",
        dol_archive_dir / "CBAList_fixed.dta",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_stata(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        lower_to_col = {str(col).strip().lower(): str(col) for col in df.columns}
        cbafile_col = lower_to_col.get("cbafile")
        naics_col = lower_to_col.get("naics")
        type_col = lower_to_col.get("type")
        location_col = lower_to_col.get("location")
        employer_col = lower_to_col.get("employername")
        if not cbafile_col:
            continue

        state_cols = sorted(
            [
                col
                for col in df.columns
                if re.fullmatch(r"statefips\d+", str(col).strip().lower()) is not None
            ],
            key=lambda c: _safe_int(re.sub(r"[^0-9]", "", str(c)), 0),
        )

        out: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            doc_id = _to_document_id_from_cbafile(row.get(cbafile_col))
            if not doc_id:
                continue
            industry = _industry_label(
                row.get(naics_col) if naics_col else None,
                row.get(type_col) if type_col else None,
            )
            states = _extract_states_from_row(row, state_cols, location_col)
            employer_name = (
                str(row.get(employer_col, "")).strip()
                if employer_col
                else ""
            )
            payload = out.get(doc_id)
            if payload is None:
                out[doc_id] = {
                    "industry": industry,
                    "states": states,
                    "employername": employer_name,
                }
                continue

            merged_states = set(payload.get("states", []))
            merged_states.update(states)
            payload["states"] = sorted(merged_states)
            if str(payload.get("industry", "")).strip() == "UNKNOWN" and industry != "UNKNOWN":
                payload["industry"] = industry
            if not str(payload.get("employername", "")).strip() and employer_name:
                payload["employername"] = employer_name

        if out:
            return out, str(path)

    raise FileNotFoundError(
        f"Could not read usable CBA metadata from .dta files in {dol_archive_dir}"
    )


def _load_processed_document_ids(llm_output_dir: Path) -> list[str]:
    candidates = [
        llm_output_dir / "document_composite_scores.csv",
        llm_output_dir / "document_clause_composite_scores.csv",
    ]
    out = set()
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                doc_id = str(row.get("document_id", "")).strip()
                if doc_id:
                    out.add(doc_id)
    return sorted(out, key=_parse_document_num)


def run(
    *,
    llm_output_dir: Path,
    dol_archive_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    llm_output_dir = llm_output_dir.expanduser().resolve()
    dol_archive_dir = dol_archive_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_doc_ids = _load_processed_document_ids(llm_output_dir)
    if not processed_doc_ids:
        raise RuntimeError(
            f"No processed document IDs found in {llm_output_dir} "
            "(expected document_composite_scores.csv or document_clause_composite_scores.csv)."
        )

    metadata_by_doc, metadata_source = _load_cba_metadata(dol_archive_dir)

    joint_counts: dict[tuple[str, str], int] = defaultdict(int)
    industry_doc_counts: dict[str, int] = defaultdict(int)
    state_doc_counts: dict[str, int] = defaultdict(int)
    missing_metadata_docs = []

    for doc_id in processed_doc_ids:
        payload = metadata_by_doc.get(doc_id)
        if not payload:
            industry = "UNKNOWN"
            states = ["UNKNOWN"]
            missing_metadata_docs.append(doc_id)
        else:
            industry = str(payload.get("industry", "UNKNOWN")).strip() or "UNKNOWN"
            raw_states = payload.get("states", [])
            if isinstance(raw_states, list):
                states = [str(s).strip() for s in raw_states if str(s).strip()]
            else:
                states = []
            if not states:
                states = ["UNKNOWN"]

        industry_doc_counts[industry] += 1
        for state in sorted(set(states)):
            joint_counts[(industry, state)] += 1
            state_doc_counts[state] += 1

    industries = sorted(
        industry_doc_counts.keys(),
        key=lambda ind: (-industry_doc_counts[ind], ind.lower()),
    )
    states = sorted(
        state_doc_counts.keys(),
        key=lambda st: (-state_doc_counts[st], st),
    )

    long_rows = []
    for industry in industries:
        for state in states:
            count = int(joint_counts.get((industry, state), 0))
            if count <= 0:
                continue
            long_rows.append(
                {
                    "industry": industry,
                    "state": state,
                    "cba_count": count,
                }
            )

    wide_rows = []
    for industry in industries:
        row: dict[str, Any] = {"industry": industry}
        joint_total = 0
        for state in states:
            count = int(joint_counts.get((industry, state), 0))
            row[state] = count
            joint_total += count
        row["joint_total"] = joint_total
        row["industry_cba_count"] = int(industry_doc_counts.get(industry, 0))
        wide_rows.append(row)

    long_csv_path = output_dir / "cba_joint_distribution_industry_state_long.csv"
    with long_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["industry", "state", "cba_count"],
        )
        writer.writeheader()
        writer.writerows(long_rows)

    wide_csv_path = output_dir / "cba_joint_distribution_industry_state.csv"
    with wide_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["industry", *states, "joint_total", "industry_cba_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wide_rows)

    summary = {
        "llm_output_dir": str(llm_output_dir),
        "dol_archive_dir": str(dol_archive_dir),
        "metadata_source_dta": metadata_source,
        "processed_cba_count": int(len(processed_doc_ids)),
        "missing_metadata_cba_count": int(len(missing_metadata_docs)),
        "industry_count": int(len(industries)),
        "state_count": int(len(states)),
        "outputs": {
            "joint_distribution_wide_csv": str(wide_csv_path),
            "joint_distribution_long_csv": str(long_csv_path),
        },
    }
    summary_path = output_dir / "cba_joint_distribution_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a joint distribution table for CBAs processed by 04_generosity_llm "
            "across industries and states using dol_archive .dta metadata."
        )
    )
    parser.add_argument("--llm-output-dir", type=Path, default=_default_llm_output_dir())
    parser.add_argument("--dol-archive-dir", type=Path, default=_default_dol_archive_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()

    summary = run(
        llm_output_dir=args.llm_output_dir,
        dol_archive_dir=args.dol_archive_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
