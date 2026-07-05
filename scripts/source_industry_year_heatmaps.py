from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


DEFAULT_METADATA = Path("validation/harmonized_cba_metadata.csv")
DEFAULT_OUTPUT_DIR = Path("figures/source_industry_year_heatmaps")
SOURCES = ["Cornell_DoL", "Cornell_RetailEd", "DoL"]
HEATMAP_BUCKET_LABELS = ["0 CBAs", "1-4 CBAs", "5-30 CBAs", "30+ CBAs"]
HEATMAP_BUCKET_COLORS = ["#f7fbff", "#c6dbef", "#6baed6", "#08519c"]

NAICS_SECTOR_LABELS = {
    11: "Agriculture, forestry, fishing, and hunting",
    21: "Mining",
    22: "Utilities",
    23: "Construction",
    31: "Manufacturing",
    32: "Manufacturing",
    33: "Manufacturing",
    42: "Wholesale trade",
    44: "Retail trade",
    45: "Retail trade",
    48: "Transportation and warehousing",
    49: "Transportation and warehousing",
    51: "Information",
    52: "Finance and insurance",
    53: "Real estate and rental and leasing",
    54: "Professional, scientific, and technical services",
    55: "Management of companies and enterprises",
    56: "Administrative support and waste services",
    61: "Educational services",
    62: "Health care and social assistance",
    71: "Arts, entertainment, and recreation",
    72: "Accommodation and food services",
    81: "Other services",
    92: "Public administration",
}


def derive_year(metadata: pd.DataFrame) -> pd.Series:
    effective_year = pd.to_datetime(
        metadata["effective_date"],
        errors="coerce",
    ).dt.year
    mistral_year = pd.to_numeric(
        metadata["mistral_contract_year"],
        errors="coerce",
    )
    expiration_year = pd.to_datetime(
        metadata["expiration_date"],
        errors="coerce",
    ).dt.year
    return effective_year.fillna(mistral_year).fillna(expiration_year)


def derive_naics_sector(metadata: pd.DataFrame) -> pd.Series:
    naics = pd.to_numeric(metadata["naics"], errors="coerce")
    mistral_naics = pd.to_numeric(metadata["mistral_naics"], errors="coerce")
    best_naics = naics.fillna(mistral_naics)
    sector_code = best_naics.dropna().astype(int).astype(str).str[:2].astype(int)
    sector = pd.Series(index=metadata.index, dtype="object")
    sector.loc[sector_code.index] = sector_code.map(NAICS_SECTOR_LABELS)
    return sector.fillna("Unknown")


def build_coverage(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["source"].isin(SOURCES)].copy()
    metadata["year"] = derive_year(metadata)
    metadata["industry"] = derive_naics_sector(metadata)
    metadata = metadata.dropna(subset=["year"])
    metadata["year"] = metadata["year"].astype(int)

    return (
        metadata.groupby(["source", "year", "industry"], as_index=False)
        .size()
        .rename(columns={"size": "n_cbas"})
        .sort_values(["source", "year", "industry"])
        .reset_index(drop=True)
    )


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(",", "")
    )


def save_heatmap(coverage: pd.DataFrame, source: str, output_dir: Path) -> Path:
    source_coverage = coverage[coverage["source"].eq(source)].copy()
    if source_coverage.empty:
        raise ValueError(f"No coverage rows found for source {source!r}.")

    pivot = source_coverage.pivot_table(
        index="industry",
        columns="year",
        values="n_cbas",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig_width = max(10, 0.32 * len(pivot.columns))
    fig_height = max(5, 0.34 * len(pivot.index))
    bucketed = np.select(
        [
            pivot.to_numpy() == 0,
            pivot.to_numpy() <= 4,
            pivot.to_numpy() <= 30,
            pivot.to_numpy() > 30,
        ],
        [0, 1, 2, 3],
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(
        bucketed,
        aspect="auto",
        cmap=ListedColormap(HEATMAP_BUCKET_COLORS),
        vmin=-0.5,
        vmax=3.5,
    )

    ax.set_title(f"{source}: CBA Coverage by Year and Industry")
    ax.set_xlabel("Year")
    ax.set_ylabel("Industry")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    colorbar = fig.colorbar(image, ax=ax, ticks=range(len(HEATMAP_BUCKET_LABELS)))
    colorbar.ax.set_yticklabels(HEATMAP_BUCKET_LABELS)
    colorbar.set_label("Coverage bucket")
    fig.tight_layout()

    path = output_dir / f"{slugify(source)}_year_industry_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_outputs(metadata_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = build_coverage(metadata_path)

    written: list[Path] = []
    coverage_path = output_dir / "source_year_industry_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    written.append(coverage_path)

    for source in SOURCES:
        written.append(save_heatmap(coverage, source, output_dir))

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create year-by-industry CBA coverage heatmaps for Cornell_DoL, "
            "Cornell_RetailEd, and DoL."
        )
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = write_outputs(args.metadata, args.output_dir)
    print(f"Wrote {len(written)} files to {args.output_dir}:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
