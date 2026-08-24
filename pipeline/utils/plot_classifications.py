"""Plot subtype prevalence and beneficiary share for linked stage-03 classifications.

Reads the document table written by ``link_classifications.py`` and renders six
figures: subtype prevalence overall, across contract-expiration cohorts, and by
NAICS sector, plus the same three cuts for beneficiary share.

The subtype figures report a share of *documents*, not of provisions: for each
group, the percentage of CBAs carrying at least one provision of each subtype,
plus the percentage carrying no provision of the type at all.  A document with
ten task-allocation provisions counts once, and because a CBA can carry several
subtypes the shares in a group do not sum to 100.

The beneficiary figures report a different statistic: each CBA's own percentage
of provisions naming a beneficiary, averaged across CBAs -- the mean of
within-CBA shares, not a pooled share of all provisions. Every provision names
exactly one beneficiary, so these shares sum to ~100 within a group.

The two by-cohort figures draw each series twice.  The solid line is the overall
average over every CBA in the cohort.  The dotted line is a
composition-adjusted index: the same statistic computed separately within each
industry stratum, then averaged across strata with equal weights, so a change in
the mix of industries cannot move it.  Only strata present in every plotted
cohort are used, and CBAs carrying no NAICS code are held as a stratum of their
own rather than dropped -- about half the corpus has no NAICS code and that
missingness is itself strongly time-trended, so dropping it would leave the
recent cohorts resting on a handful of contracts.

Equal weights remove *mix* effects, not *within-stratum* change: if every
stratum trends the same way, the dotted line trends too.  A gap between the two
lines is the part of the raw movement that composition explains.

Run from anywhere in the repository::

    uv run python pipeline/utils/plot_classifications.py
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

try:
    import matplotlib
except ModuleNotFoundError:  # pragma: no cover - exercised only without matplotlib
    raise SystemExit("matplotlib is required: uv add matplotlib")

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.stg_02_extract.structure_provision import (
    RESERVED_SUBTYPE_LABEL,
    labels_at_level,
    load_provision,
    taxonomy_depth,
)
from pipeline.utils.link_classifications import (
    BENEFICIARY_LABELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROVISION_TYPE,
    DEFAULT_SOURCE,
    UNKNOWN_SECTOR,
    table_prefix,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MIN_GROUP_DOCS = 10
DEFAULT_MIN_SECTOR_DOCS = 5
# Show every subtype by default; deeper taxonomy levels are where raising this
# earns its keep, by dropping the long tail of rarely-used labels.
DEFAULT_MIN_CBAS = 0
# One CBA is enough to estimate a cell.  Raising this trades the thin recent
# cohorts for a narrower but better-populated stratum panel.
DEFAULT_MIN_CELL_DOCS = 1
DEFAULT_DPI = 150
OTHER_SECTOR_LABEL = "Other sectors"
# The unknown bucket is shown on the industry figures as its own row -- about
# half the corpus carries no NAICS code, so excluding it would hide more than it
# clarifies.  It is displayed under a name that cannot be mistaken for an
# industry, and pinned below the real sectors.
NO_NAICS_LABEL = "No NAICS code"

# The absence of any provision is its own series on every figure.  It is keyed
# by a name no subtype can collide with and painted neutral grey, so the six
# categorical hues stay reserved for the subtypes themselves.
NONE_KEY = "__none__"
NONE_COLOR = "#6b6a65"

# Categorical hues in fixed order -- slot N is always the same subtype, however
# many series a given figure draws.
SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
# Marker shapes double as identity on the line figure, so the series stay
# distinguishable without relying on colour alone.
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
NONE_MARKER = "o"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"

# Cohorts are contracts expiring in a window, not contracts signed in one.  The
# harmonized metadata does carry an effective date, but only for part of the DOL
# list and none of it before harmonization, so expiration stays the cohort here;
# the document table's `contract_period` is the effective-date alternative.
PERIOD_AXIS_LABEL = "Contract expiration cohort"
PERIOD_CAPTION = (
    "Cohorts are contract expiration dates, so a contract expiring 2008 was "
    "typically signed 2003-2005."
)


def subtype_style(
    subtypes: Sequence[str], clause_type: str
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return stable colors, markers and display labels, shared across figures."""

    # Deeper taxonomy levels draw more series than there are hues, so the marker
    # advances only once the hues wrap.  That keeps the (colour, marker) pair
    # unique -- indexing both by the same number would give series 0 and series 8
    # an identical appearance on the line figure.
    colors = {
        subtype: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, subtype in enumerate(subtypes)
    }
    markers = {
        subtype: SERIES_MARKERS[(index // len(SERIES_COLORS)) % len(SERIES_MARKERS)]
        for index, subtype in enumerate(subtypes)
    }
    labels = {subtype: subtype.replace("_", " ").capitalize() for subtype in subtypes}

    colors[NONE_KEY] = NONE_COLOR
    markers[NONE_KEY] = NONE_MARKER
    labels[NONE_KEY] = f"No {clause_type} provisions"
    return colors, markers, labels


def _plural(count: int, noun: str) -> str:
    """Return ``"1 CBA"`` / ``"3 CBAs"`` -- cohorts of one are common enough."""

    return f"{count} {noun}" + ("" if count == 1 else "s")


def frequent_subtypes(
    documents: pd.DataFrame, subtypes: Sequence[str], min_cbas: int
) -> tuple[list[str], list[tuple[str, int]]]:
    """Split subtypes into those carried by at least ``min_cbas`` CBAs and the rest.

    A CBA counts once per subtype however many such provisions it holds, so this
    is the document-level count the figures already report -- not a provision
    count.  Returns the kept subtypes in their declared order, and the dropped
    ones with their counts so the caller can say what it hid.
    """

    kept: list[str] = []
    dropped: list[tuple[str, int]] = []
    for subtype in subtypes:
        column = f"has_{subtype}"
        count = int(documents[column].astype(bool).sum()) if column in documents else 0
        if count >= min_cbas:
            kept.append(subtype)
        else:
            dropped.append((subtype, count))
    return kept, dropped


def series_order(subtypes: Sequence[str]) -> list[str]:
    """Return the plotting order: subtypes as declared, absence last."""

    return [*subtypes, NONE_KEY]


def _subtype_shares_for(
    frame: pd.DataFrame, subtypes: Sequence[str]
) -> pd.Series:
    """Percentage of the CBAs in ``frame`` carrying each series.

    The single definition of the subtype statistic, so the raw lines and the
    composition-adjusted lines cannot drift apart.  Index is the subtypes plus
    :data:`NONE_KEY`; the values do not sum to 100 because one CBA can carry
    several subtypes.
    """

    total = len(frame)
    shares = {}
    for subtype in subtypes:
        column = f"has_{subtype}"
        carried = int(frame[column].astype(bool).sum()) if column in frame else 0
        shares[subtype] = carried / total * 100 if total else 0.0
    without_any = int((frame["n_provisions"] == 0).sum())
    shares[NONE_KEY] = without_any / total * 100 if total else 0.0
    return pd.Series(shares, dtype=float)


def document_shares(
    documents: pd.DataFrame, group: str, subtypes: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Return per-group percentages of documents carrying each series.

    Rows are groups, columns are the subtypes plus :data:`NONE_KEY`.  A row does
    not sum to 100: one CBA can carry several subtypes, and the absence series
    is the complement of carrying any.
    """

    grouped = documents.groupby(group, dropna=True)
    totals = grouped.size()
    keys = series_order(subtypes)
    shares = pd.DataFrame(
        [_subtype_shares_for(frame, subtypes)[keys] for _, frame in grouped],
        index=totals.index,
        columns=keys,
    )
    return shares, totals


def overall_shares(
    documents: pd.DataFrame, subtypes: Sequence[str]
) -> tuple[pd.Series, pd.Series]:
    """Return the same shares for the whole table, plus the document counts."""

    counts = {}
    for subtype in subtypes:
        column = f"has_{subtype}"
        counts[subtype] = (
            int(documents[column].astype(bool).sum())
            if column in documents.columns
            else 0
        )
    counts[NONE_KEY] = int((documents["n_provisions"] == 0).sum())
    counts = pd.Series(counts, dtype=float)
    return counts / max(len(documents), 1) * 100, counts


def beneficiary_style(
    beneficiaries: Sequence[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Colors, markers and display labels for the beneficiary series.

    Unlike :func:`subtype_style`, there is no absence series here: every
    provision names exactly one beneficiary, so the three labels already
    exhaust the space.
    """

    colors = {
        beneficiary: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, beneficiary in enumerate(beneficiaries)
    }
    markers = {
        beneficiary: SERIES_MARKERS[index % len(SERIES_MARKERS)]
        for index, beneficiary in enumerate(beneficiaries)
    }
    labels = {
        beneficiary: beneficiary.replace("_", " ").capitalize()
        for beneficiary in beneficiaries
    }
    return colors, markers, labels


def _with_a_provision(documents: pd.DataFrame) -> pd.DataFrame:
    """CBAs with at least one provision -- the only ones a beneficiary share is defined for."""

    return documents[documents["n_provisions"] > 0]


def _beneficiary_means_for(
    frame: pd.DataFrame, beneficiaries: Sequence[str]
) -> pd.Series:
    """Mean within-CBA beneficiary share over the CBAs in ``frame``.

    The single definition of the beneficiary statistic, shared by the raw and
    the composition-adjusted paths.  ``frame`` is expected to be already
    restricted to CBAs carrying a provision -- see :func:`_with_a_provision`.
    """

    return pd.Series(
        {
            beneficiary: frame[f"pct_beneficiary_{beneficiary}"].mean()
            for beneficiary in beneficiaries
        },
        dtype=float,
    )


def beneficiary_overall_means(
    documents: pd.DataFrame, beneficiaries: Sequence[str]
) -> tuple[pd.Series, int]:
    """Mean within-CBA beneficiary share across every classified CBA.

    Each CBA contributes its own percentage of provisions naming a beneficiary;
    this is the mean of those percentages, not a pooled share of all provisions
    corpus-wide -- a handful of provision-heavy CBAs cannot dominate the figure.
    """

    active = _with_a_provision(documents)
    return _beneficiary_means_for(active, beneficiaries), len(active)


def beneficiary_group_means(
    documents: pd.DataFrame, group: str, beneficiaries: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Mean within-CBA beneficiary share per group, and the CBA count behind each.

    Rows are groups, columns are the beneficiaries. A row sums to ~100 because
    every provision names exactly one beneficiary, unlike the subtype shares
    which overlap.
    """

    active = _with_a_provision(documents)
    grouped = active.groupby(group, dropna=True)
    totals = grouped.size()
    means = pd.DataFrame(
        [_beneficiary_means_for(frame, beneficiaries)[list(beneficiaries)]
         for _, frame in grouped],
        index=totals.index,
        columns=list(beneficiaries),
    )
    return means, totals


def _caption(fig, ax, text: str) -> None:
    """Place a wrapped caption a fixed gap below the axes and its labels."""

    fig.canvas.draw()
    extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )
    fig.text(
        0.02,
        extent.y0 - 0.16 / fig.get_figheight(),
        textwrap.fill(text, width=110),
        fontsize=8,
        color=TEXT_SECONDARY,
        ha="left",
        va="top",
    )


def _save(fig, path: Path, dpi: int, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would write {path}")
        plt.close(fig)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _tidy(ax, axis: str = "x") -> None:
    """Recede the frame and put a light grid behind the marks."""

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis=axis, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


ADJUSTED_LINESTYLE = ":"
ADJUSTED_LABEL = "Industry-adjusted (equal weights)"
OVERALL_LABEL = "Overall average"


def _style_legend_handles() -> list:
    """Two proxies naming the line styles, so series need only one entry each.

    Near-black rather than grey: the absence series is already grey, and two
    similar greys side by side in the legend read as the same series.
    """

    from matplotlib.lines import Line2D

    return [
        Line2D([], [], color=TEXT_PRIMARY, linewidth=2, label=OVERALL_LABEL),
        Line2D(
            [],
            [],
            color=TEXT_PRIMARY,
            linewidth=2,
            linestyle=ADJUSTED_LINESTYLE,
            label=ADJUSTED_LABEL,
        ),
    ]


def _legend_rows(count: int, ncol: int) -> int:
    return -(-count // ncol)


def _title_pad(series: int, ncol: int) -> float:
    """Points to reserve above the axes for a legend of ``series`` entries.

    Deeper taxonomy levels draw many more series, so the legend grows several
    rows taller and a fixed pad would let it run over the title.
    """

    return 14 + 12 * _legend_rows(series, ncol)


def _legend(ax, ncol: int, extra_handles: Sequence = ()) -> None:
    """Legend above the axes; the caption sits below, so they cannot collide.

    ``extra_handles`` are appended after the series entries.  The period figures
    use them to explain the two line styles once, rather than listing every
    series twice.
    """

    # Matplotlib fills the legend column by column; feeding it the entries in
    # column-major order makes the rendered grid read left to right.
    handles, labels = ax.get_legend_handles_labels()
    handles = [*handles, *extra_handles]
    labels = [*labels, *(handle.get_label() for handle in extra_handles)]
    rows = _legend_rows(len(handles), ncol)
    order = [
        column + row * ncol
        for column in range(ncol)
        for row in range(rows)
        if column + row * ncol < len(handles)
    ]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        borderaxespad=0,
    )


def plot_overall(
    documents: pd.DataFrame,
    subtypes,
    colors,
    labels,
    clause_type: str,
) -> "matplotlib.figure.Figure":
    shares, counts = overall_shares(documents, subtypes)
    shares = shares.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(shares) + 2.0))
    ax.barh(
        [labels[key] for key in shares.index],
        shares.to_numpy(),
        color=[colors[key] for key in shares.index],
        height=0.62,
        zorder=2,
    )
    for position, (key, share) in enumerate(shares.items()):
        ax.text(
            share + 1.0,
            position,
            f"{share:.1f}%  ({int(counts[key])} CBAs)",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.set_xlim(0, min(100, max(shares.max() * 1.32, 5)))
    ax.set_xlabel("% of classified CBAs", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(
        f"Share of CBAs with each {clause_type} provision subtype",
        loc="left",
        color=TEXT_PRIMARY,
    )
    _tidy(ax)
    _caption(
        fig,
        ax,
        f"{len(documents)} classified CBAs. A CBA counts once per subtype, however "
        "many provisions of it the CBA contains, so the shares overlap and do not "
        "sum to 100%.",
    )
    return fig


def _pool_non_spanning(
    frame: pd.DataFrame,
    group: str,
    groups: Sequence[str],
    min_cell_docs: int,
    industry: str = "sector_group",
    protect: Sequence[str] = (UNKNOWN_SECTOR,),
) -> tuple[pd.DataFrame, list[str]]:
    """Fold industries that miss a plotted cohort into the pooled remainder.

    ``composition_adjusted`` can only use strata present in every cohort, and on
    its own it drops the rest -- discarding those CBAs from the index rather
    than counting them anywhere coarser.  Folding them into the remainder keeps
    them in the sample at a coarser grain, which is the same trade the remainder
    bucket already makes for the sectors too small to name, and it is what keeps
    the index estimable when the thinnest cohort holds only a handful of CBAs.

    ``protect`` is never folded: the no-NAICS bucket is not an industry, so
    pooling it into the remainder would make the remainder mean something else.
    Returns the reframed rows and the folded names, so the caller can name them.
    """

    sizes = frame.groupby([industry, group]).size()
    strata = list(dict.fromkeys(frame[industry]))
    folded = sorted(
        stratum
        for stratum in strata
        if stratum not in protect
        and stratum != OTHER_SECTOR_LABEL
        and any(int(sizes.get((stratum, name), 0)) < min_cell_docs for name in groups)
    )
    if not folded:
        return frame, []
    frame = frame.copy()
    frame[industry] = frame[industry].where(
        ~frame[industry].isin(folded), OTHER_SECTOR_LABEL
    )
    return frame, folded


def _adjusted_overlay(
    ax,
    frame: pd.DataFrame,
    series: Sequence[str],
    stat_for,
    kept: Sequence[str],
    colors,
    markers,
    min_sector_docs: int,
    min_cell_docs: int,
    weights: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Draw the dotted composition-adjusted line for each series.

    Returns the adjusted values, so the caller can size the y axis over both
    lines, and a caption fragment naming the strata and the sample behind them.
    Returns ``(None, reason)`` when the index is not estimable, in which case
    nothing is drawn and the figure keeps its raw lines only.
    """

    grouped = _grouped_by_sector(frame, min_sector_docs, keep_unknown=True)
    if grouped is None:
        return None, " No CBA carries a sector, so no industry-adjusted index is shown."
    pooled, _ = grouped
    pooled = pooled[pooled["expire_period"].isin(list(kept))]
    pooled, folded = _pool_non_spanning(
        pooled, "expire_period", list(kept), min_cell_docs
    )

    adjusted, counts, used, dropped = composition_adjusted(
        pooled,
        "expire_period",
        list(series),
        stat_for,
        weights=weights,
        min_cell_docs=min_cell_docs,
    )
    if len(used) < 2:
        print(
            "  [warn] fewer than two industry strata span every cohort; "
            "skipping the industry-adjusted line"
        )
        return None, (
            " Too few industry strata span every cohort for an industry-adjusted "
            "index, so only the overall average is shown."
        )

    adjusted = adjusted.reindex(list(kept))
    counts = counts.reindex(list(kept))
    for key in series:
        ax.plot(
            range(len(kept)),
            adjusted[key].to_numpy(),
            color=colors[key],
            marker=markers[key],
            markersize=5.0,
            markerfacecolor="none",
            linewidth=1.8,
            linestyle=ADJUSTED_LINESTYLE,
            label="_nolegend_",
            zorder=2,
        )

    per_cohort = ", ".join(
        f"{period} {int(counts.get(period, 0))}" for period in kept
    )
    fragment = (
        f" The dotted index averages the statistic computed within each of "
        f"{len(used)} industry strata ({', '.join(sorted(used))}), weighting every "
        f"stratum equally, so a shift in the mix of strata cannot move it. CBAs "
        f"behind it per cohort: {per_cohort}."
    )
    if folded:
        fragment += (
            f" {len(folded)} "
            f"{'industry missing a cohort was' if len(folded) == 1 else 'industries missing a cohort were'}"
            " folded into "
            f"{OTHER_SECTOR_LABEL.lower()} rather than dropped from the index: "
            f"{', '.join(folded)}."
        )
    if dropped:
        fragment += (
            f" Strata excluded for not spanning every cohort: "
            f"{', '.join(sorted(name for name, _ in dropped))}."
        )
    return adjusted, fragment


def plot_by_period(
    documents: pd.DataFrame,
    subtypes,
    colors,
    markers,
    labels,
    min_docs: int,
    clause_type: str,
    industry_adjusted: bool = True,
    min_sector_docs: int = DEFAULT_MIN_SECTOR_DOCS,
    min_cell_docs: int = 1,
) -> "matplotlib.figure.Figure | None":
    known = documents[
        documents["expire_period"].notna() & (documents["expire_period"] != "")
    ]
    if known.empty:
        print("  [warn] no document carries an expiration cohort; skipping figure")
        return None
    shares, totals = document_shares(known, "expire_period", subtypes)
    shares = shares.sort_index()

    kept = [period for period in shares.index if totals.get(period, 0) >= min_docs]
    dropped = [
        f"{period} ({_plural(int(totals.get(period, 0)), 'CBA')})"
        for period in shares.index
        if period not in kept
    ]
    if not kept:
        print("  [warn] no expiration cohort meets --min-group-docs; skipping figure")
        return None
    shares = shares.loc[kept]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    positions = range(len(kept))
    keys = series_order(subtypes)
    for key in keys:
        ax.plot(
            list(positions),
            shares[key].to_numpy(),
            color=colors[key],
            marker=markers[key],
            markersize=5.5,
            linewidth=2,
            label=labels[key],
            zorder=2,
        )
    adjusted_fragment = ""
    highest = shares.to_numpy().max()
    if industry_adjusted:
        adjusted, adjusted_fragment = _adjusted_overlay(
            ax,
            known,
            keys,
            _subtype_shares_for,
            kept,
            colors,
            markers,
            min_sector_docs,
            min_cell_docs,
        )
        if adjusted is not None:
            highest = max(highest, float(adjusted.to_numpy(dtype=float).max()))

    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [f"{period}\n{int(totals.get(period, 0))} CBAs" for period in kept]
    )
    ax.set_ylim(0, min(100, max(highest * 1.15, 10)))
    ax.set_xlabel(PERIOD_AXIS_LABEL, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("% of CBAs in cohort", color=TEXT_SECONDARY, fontsize=9)
    extra = _style_legend_handles() if industry_adjusted else []
    ax.set_title(
        f"Share of CBAs with each {clause_type} subtype, by expiration cohort",
        loc="left",
        color=TEXT_PRIMARY,
        pad=_title_pad(len(keys) + len(extra), ncol=4),
    )
    _tidy(ax, axis="y")
    _legend(ax, ncol=4, extra_handles=extra)
    caption = PERIOD_CAPTION + (
        " Shares are of the CBAs in each cohort and overlap, so they do not sum "
        "to 100%."
    )
    if dropped:
        caption += f" Cohorts below {min_docs} CBAs omitted: {', '.join(dropped)}."
    caption += adjusted_fragment
    _caption(fig, ax, caption)
    return fig


def _grouped_by_sector(
    documents: pd.DataFrame, min_docs: int, keep_unknown: bool = False
) -> tuple[pd.DataFrame, pd.Series] | None:
    """Add a ``sector_group`` column pooling sparse sectors into one remainder.

    Shared by every by-sector figure so a sector's inclusion threshold and its
    "Other sectors" catch-all stay identical whichever statistic is plotted.
    Returns ``None`` if no document survives.

    ``keep_unknown`` retains the CBAs carrying no NAICS code as their own
    stratum rather than dropping them: it is never pooled into the remainder,
    because the composition-adjusted index needs it held fixed alongside the
    real industries -- roughly half the corpus has no NAICS code and that
    missingness is itself time-trended.  The by-sector figures leave it off, so
    an unknown sector is not drawn as though it were an industry.
    """

    doc_counts = documents["sector_label"].value_counts()
    named = [
        sector
        for sector, count in doc_counts.items()
        if sector != UNKNOWN_SECTOR and count >= min_docs
    ]
    if keep_unknown:
        named.append(UNKNOWN_SECTOR)
        known = documents.copy()
    else:
        known = documents[documents["sector_label"] != UNKNOWN_SECTOR].copy()
    if known.empty:
        return None
    known["sector_group"] = known["sector_label"].where(
        known["sector_label"].isin(named), OTHER_SECTOR_LABEL
    )
    return known, doc_counts


def _sector_order(totals: pd.Series) -> list[str]:
    """Largest sectors first, then the pooled remainder, then the unknown bucket.

    Both catch-all rows are pinned below the real sectors however many CBAs they
    hold, so size ordering never implies that "no NAICS code" is an industry.
    """

    pinned = [OTHER_SECTOR_LABEL, UNKNOWN_SECTOR]
    order = [s for s in totals.sort_values(ascending=False).index if s not in pinned]
    return order + [s for s in pinned if s in totals.index]


def _sector_tick(sector: str, count: int) -> str:
    """Row label for one sector: display name over its CBA count."""

    name = NO_NAICS_LABEL if sector == UNKNOWN_SECTOR else sector
    return f"{name}\n{count} CBAs"


def composition_adjusted(
    documents: pd.DataFrame,
    group: str,
    series: Sequence[str],
    stat_for,
    industry: str = "sector_group",
    weights: Mapping[str, float] | None = None,
    min_cell_docs: int = 1,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[tuple[str, str]]]:
    """Weighted mean across industry strata of each stratum's own group statistic.

    Direct standardisation: for each group (an expiration cohort), ``stat_for``
    is evaluated separately within every industry stratum, and those per-stratum
    values are averaged with fixed weights.  Because a stratum contributes the
    same weight however many CBAs it holds, a stratum that grows or shrinks
    across cohorts can no longer move the index -- which is the whole point.

    Only strata present in *every* group are used, so ``documents`` must already
    be narrowed to the groups being plotted; letting the stratum set drift
    between groups would readmit composition through presence and absence.

    ``weights`` maps stratum to weight, defaulting to 1.0 for any stratum it
    omits -- so ``None`` is an unweighted mean.  Returns the adjusted values
    (groups x series), the CBA count behind each group, the strata used, and the
    strata dropped paired with the reason, so the caller can name them rather
    than hiding the panel it chose.
    """

    groups = list(dict.fromkeys(documents[group]))
    cells = documents.groupby([group, industry], dropna=True)
    sizes = cells.size()

    stats: dict[tuple[str, str], pd.Series] = {}
    for key, frame in cells:
        stats[key] = stat_for(frame, series)

    strata = list(dict.fromkeys(documents[industry]))
    used: list[str] = []
    dropped: list[tuple[str, str]] = []
    for stratum in strata:
        thin = [
            str(name)
            for name in groups
            if int(sizes.get((name, stratum), 0)) < min_cell_docs
        ]
        if thin:
            dropped.append((stratum, f"absent or below {min_cell_docs} in {', '.join(thin)}"))
        else:
            used.append(stratum)

    keys = list(series)
    if len(used) < 2:
        empty = pd.DataFrame(index=pd.Index(groups, name=group), columns=keys, dtype=float)
        return empty, pd.Series(0, index=pd.Index(groups, name=group)), used, dropped

    weight = {stratum: float((weights or {}).get(stratum, 1.0)) for stratum in used}
    total_weight = sum(weight.values())

    adjusted = pd.DataFrame(index=pd.Index(groups, name=group), columns=keys, dtype=float)
    counts = pd.Series(0, index=pd.Index(groups, name=group))
    for name in groups:
        blended = pd.Series(0.0, index=keys)
        for stratum in used:
            blended = blended + stats[(name, stratum)][keys] * weight[stratum]
            counts[name] += int(sizes.get((name, stratum), 0))
        adjusted.loc[name] = (blended / total_weight).to_numpy()
    return adjusted, counts, used, dropped


def _sector_caption_tail(min_docs: int, unknown_docs: int, show_unknown: bool) -> str:
    """The pooling and no-NAICS sentences, shared by both industry figures."""

    tail = (
        f" Sectors with fewer than {min_docs} CBAs are pooled into "
        f"{OTHER_SECTOR_LABEL!r}."
    )
    if show_unknown:
        return tail + (
            f" The {NO_NAICS_LABEL!r} row is the {unknown_docs} CBAs whose metadata "
            "carries no NAICS code. It is not an industry, so it is pinned below the "
            "real sectors and never pooled into the remainder."
        )
    return tail + f" {unknown_docs} CBAs carry no NAICS code and are excluded."


def _no_naics_count(documents: pd.DataFrame) -> int:
    """CBAs with no NAICS code, counted on the frame the figure actually plots.

    The by-sector figures use different denominators -- every classified CBA for
    the subtype figure, only provision-carrying ones for the beneficiary figure --
    so each has to count its own, or the caption contradicts the row label.
    """

    return int((documents["sector_label"] == UNKNOWN_SECTOR).sum())


def plot_by_sector(
    documents: pd.DataFrame,
    subtypes,
    colors,
    labels,
    min_docs: int,
    clause_type: str,
    show_unknown: bool = True,
) -> "matplotlib.figure.Figure | None":
    grouped = _grouped_by_sector(documents, min_docs, keep_unknown=show_unknown)
    if grouped is None:
        print("  [warn] no documents carry a NAICS sector; skipping figure")
        return None
    known, doc_counts = grouped

    shares, totals = document_shares(known, "sector_group", subtypes)
    order = _sector_order(totals)
    shares = shares.loc[order]

    keys = series_order(subtypes)
    # One slot per industry, split evenly between the series; the bar is drawn a
    # touch thinner than its slot so neighbours never touch.
    slot = 0.86
    height = slot / len(keys)

    fig, ax = plt.subplots(figsize=(10, slot * len(order) + 2.6))
    for index, key in enumerate(keys):
        offsets = [
            group - slot / 2 + (index + 0.5) * height for group in range(len(order))
        ]
        ax.barh(
            offsets,
            shares[key].to_numpy(),
            height=height * 0.86,
            color=colors[key],
            label=labels[key],
            zorder=2,
        )
    ax.set_yticks(list(range(len(order))))
    ax.set_yticklabels(
        [_sector_tick(sector, int(totals.get(sector, 0))) for sector in order]
    )
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, min(100, max(shares.to_numpy().max() * 1.04, 10)))
    ax.set_xlabel("% of CBAs in industry", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("NAICS sector", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(
        f"Share of CBAs with each {clause_type} subtype, by industry",
        loc="left",
        color=TEXT_PRIMARY,
        pad=_title_pad(len(keys), ncol=4),
    )
    _tidy(ax)
    _legend(ax, ncol=4)

    _caption(
        fig,
        ax,
        "Shares are of the CBAs in each industry and overlap, so they do not sum "
        "to 100%."
        + _sector_caption_tail(
            min_docs, int(doc_counts.get(UNKNOWN_SECTOR, 0)), show_unknown
        ),
    )
    return fig


BENEFICIARY_MEAN_CAPTION = (
    "Each CBA's own % of provisions naming a beneficiary is computed, then "
    "averaged across CBAs -- the mean of within-CBA shares, not a pooled share "
    "of all provisions. Every provision names exactly one beneficiary, so bars "
    "sum to ~100%."
)


def plot_beneficiary_overall(
    documents: pd.DataFrame,
    beneficiaries: Sequence[str],
    colors,
    labels,
    clause_type: str,
) -> "matplotlib.figure.Figure":
    means, n = beneficiary_overall_means(documents, beneficiaries)
    means = means.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(means) + 2.0))
    ax.barh(
        [labels[key] for key in means.index],
        means.to_numpy(),
        color=[colors[key] for key in means.index],
        height=0.5,
        zorder=2,
    )
    for position, (key, share) in enumerate(means.items()):
        ax.text(
            share + 1.0,
            position,
            f"{share:.1f}%",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.set_xlim(0, min(100, max(means.max() * 1.25, 5)))
    ax.set_xlabel(
        "Mean within-CBA share of provisions", color=TEXT_SECONDARY, fontsize=9
    )
    ax.set_title(
        f"Average beneficiary of {clause_type} provisions",
        loc="left",
        color=TEXT_PRIMARY,
    )
    _tidy(ax)
    _caption(
        fig,
        ax,
        f"{n} classified CBAs with at least one {clause_type} provision. "
        + BENEFICIARY_MEAN_CAPTION,
    )
    return fig


def plot_beneficiary_by_period(
    documents: pd.DataFrame,
    beneficiaries: Sequence[str],
    colors,
    markers,
    labels,
    min_docs: int,
    clause_type: str,
    industry_adjusted: bool = True,
    min_sector_docs: int = DEFAULT_MIN_SECTOR_DOCS,
    min_cell_docs: int = 1,
) -> "matplotlib.figure.Figure | None":
    known = documents[
        documents["expire_period"].notna() & (documents["expire_period"] != "")
    ]
    means, totals = beneficiary_group_means(known, "expire_period", beneficiaries)
    if means.empty:
        print(
            "  [warn] no document with a provision carries an expiration cohort; "
            "skipping figure"
        )
        return None
    means = means.sort_index()

    kept = [period for period in means.index if totals.get(period, 0) >= min_docs]
    dropped = [
        f"{period} ({_plural(int(totals.get(period, 0)), 'CBA')})"
        for period in means.index
        if period not in kept
    ]
    if not kept:
        print("  [warn] no expiration cohort meets --min-group-docs; skipping figure")
        return None
    means = means.loc[kept]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    positions = range(len(kept))
    for beneficiary in beneficiaries:
        ax.plot(
            list(positions),
            means[beneficiary].to_numpy(),
            color=colors[beneficiary],
            marker=markers[beneficiary],
            markersize=5.5,
            linewidth=2,
            label=labels[beneficiary],
            zorder=2,
        )
    adjusted_fragment = ""
    highest = means.to_numpy().max()
    if industry_adjusted:
        adjusted, adjusted_fragment = _adjusted_overlay(
            ax,
            _with_a_provision(known),
            beneficiaries,
            _beneficiary_means_for,
            kept,
            colors,
            markers,
            min_sector_docs,
            min_cell_docs,
        )
        if adjusted is not None:
            highest = max(highest, float(adjusted.to_numpy(dtype=float).max()))

    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [f"{period}\n{int(totals.get(period, 0))} CBAs" for period in kept]
    )
    ax.set_ylim(0, min(100, max(highest * 1.15, 10)))
    ax.set_xlabel(PERIOD_AXIS_LABEL, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Mean within-CBA share", color=TEXT_SECONDARY, fontsize=9)
    extra = _style_legend_handles() if industry_adjusted else []
    ax.set_title(
        f"Average beneficiary of {clause_type} provisions, by expiration cohort",
        loc="left",
        color=TEXT_PRIMARY,
        pad=_title_pad(len(beneficiaries) + len(extra), ncol=3),
    )
    _tidy(ax, axis="y")
    _legend(ax, ncol=3, extra_handles=extra)
    caption = PERIOD_CAPTION + " " + BENEFICIARY_MEAN_CAPTION
    caption += " Only CBAs with at least one provision in the cohort are averaged."
    if dropped:
        caption += f" Cohorts below {min_docs} CBAs omitted: {', '.join(dropped)}."
    caption += adjusted_fragment
    _caption(fig, ax, caption)
    return fig


def plot_beneficiary_by_sector(
    documents: pd.DataFrame,
    beneficiaries: Sequence[str],
    colors,
    labels,
    min_docs: int,
    clause_type: str,
    show_unknown: bool = True,
) -> "matplotlib.figure.Figure | None":
    grouped = _grouped_by_sector(documents, min_docs, keep_unknown=show_unknown)
    if grouped is None:
        print("  [warn] no documents carry a NAICS sector; skipping figure")
        return None
    known, _ = grouped

    means, totals = beneficiary_group_means(known, "sector_group", beneficiaries)
    if means.empty:
        print(
            "  [warn] no document with a provision carries a NAICS sector; "
            "skipping figure"
        )
        return None
    order = _sector_order(totals)
    means = means.loc[order]

    slot = 0.86
    height = slot / len(beneficiaries)

    fig, ax = plt.subplots(figsize=(10, slot * len(order) + 2.6))
    for index, beneficiary in enumerate(beneficiaries):
        offsets = [
            group - slot / 2 + (index + 0.5) * height for group in range(len(order))
        ]
        ax.barh(
            offsets,
            means[beneficiary].to_numpy(),
            height=height * 0.86,
            color=colors[beneficiary],
            label=labels[beneficiary],
            zorder=2,
        )
    ax.set_yticks(list(range(len(order))))
    ax.set_yticklabels(
        [_sector_tick(sector, int(totals.get(sector, 0))) for sector in order]
    )
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, min(100, max(means.to_numpy().max() * 1.1, 10)))
    ax.set_xlabel("Mean within-CBA share", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("NAICS sector", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(
        f"Average beneficiary of {clause_type} provisions, by industry",
        loc="left",
        color=TEXT_PRIMARY,
        pad=_title_pad(len(beneficiaries), ncol=3),
    )
    _tidy(ax)
    _legend(ax, ncol=3)

    caption = BENEFICIARY_MEAN_CAPTION
    caption += " Only CBAs with at least one provision in the sector are averaged."
    caption += _sector_caption_tail(
        min_docs, _no_naics_count(_with_a_provision(documents)), show_unknown
    )
    _caption(fig, ax, caption)
    return fig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory holding the linked CSVs (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for the figures (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"cache source folder the CSVs were built from (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help=(
            "taxonomy level to plot; must match the --level "
            "link_classifications.py built the tables with (default: 1)"
        ),
    )
    parser.add_argument(
        "--clause-type",
        default=DEFAULT_PROVISION_TYPE,
        help=f"provision type to plot (default: {DEFAULT_PROVISION_TYPE})",
    )
    parser.add_argument(
        "--min-group-docs",
        type=int,
        default=DEFAULT_MIN_GROUP_DOCS,
        help=(
            "omit expiration cohorts holding fewer CBAs "
            f"(default: {DEFAULT_MIN_GROUP_DOCS})"
        ),
    )
    parser.add_argument(
        "--min-sector-docs",
        type=int,
        default=DEFAULT_MIN_SECTOR_DOCS,
        help=(
            "pool sectors with fewer CBAs into a remainder group "
            f"(default: {DEFAULT_MIN_SECTOR_DOCS})"
        ),
    )
    parser.add_argument(
        "--min-cbas",
        "--min_cbas",
        dest="min_cbas",
        type=int,
        default=DEFAULT_MIN_CBAS,
        metavar="N",
        help=(
            "omit subtypes carried by fewer than N CBAs, counting each CBA once "
            "per subtype. Unlike --min-group-docs and --min-sector-docs, which "
            f"drop sparse groups, this drops sparse series (default: {DEFAULT_MIN_CBAS})"
        ),
    )
    parser.add_argument(
        "--hide-no-naics",
        dest="show_unknown",
        action="store_false",
        help=(
            "omit the 'No NAICS code' row from the two industry figures. It is "
            "shown by default because about half the corpus carries no NAICS code"
        ),
    )
    parser.add_argument(
        "--no-industry-adjusted",
        dest="industry_adjusted",
        action="store_false",
        help=(
            "omit the dotted industry-adjusted index from the two time-series "
            "figures. Worth using at deeper taxonomy levels, where doubling the "
            "line count makes the cohort figure unreadable"
        ),
    )
    parser.add_argument(
        "--min-cell-docs",
        type=int,
        default=DEFAULT_MIN_CELL_DOCS,
        metavar="N",
        help=(
            "minimum CBAs in an industry-by-cohort cell for that industry to join "
            "the adjusted index. An industry must clear it in every plotted cohort, "
            f"or it is excluded entirely (default: {DEFAULT_MIN_CELL_DOCS})"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"figure resolution (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the figures without writing any PNG",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    spec = load_provision(args.clause_type)
    if not spec.subtype_taxonomy:
        raise SystemExit(f"provision {args.clause_type} declares no subtypes")
    if args.level < 1:
        raise SystemExit("--level must be at least 1")
    if args.min_cbas < 0:
        raise SystemExit("--min-cbas must not be negative")
    available = taxonomy_depth(spec.subtype_taxonomy)
    if args.level > available:
        raise SystemExit(
            f"--level {args.level} exceeds the {args.clause_type} taxonomy, "
            f"which declares {available} level(s)"
        )
    # Must match the --level the document table was built with: stage 3 offers
    # "other" at every level, so it joins the config's own labels.
    declared = list(labels_at_level(spec.subtype_taxonomy, args.level))
    subtypes = [*declared, RESERVED_SUBTYPE_LABEL]

    prefix = table_prefix(args.clause_type, args.source, args.level)
    documents_path = args.input_dir / f"{prefix}_documents.csv"
    if not documents_path.is_file():
        raise SystemExit(
            f"document table not found: {documents_path}\n"
            "run pipeline/utils/link_classifications.py first, with "
            f"--level {args.level}"
        )
    documents = pd.read_csv(documents_path, dtype={"expire_period": str})
    # A table built at another level carries none of this level's columns, and
    # the per-series fallback would silently plot every share as zero.  Only the
    # declared labels are evidence: "other" appears at every level, so its column
    # says nothing about which level built the table.
    if not any(f"has_{subtype}" in documents.columns for subtype in declared):
        raise SystemExit(
            f"{documents_path} carries no level-{args.level} subtype columns; "
            f"rebuild it with link_classifications.py --level {args.level}"
        )
    documents["sector_label"] = documents["sector_label"].fillna(UNKNOWN_SECTOR)
    without_any = int((documents["n_provisions"] == 0).sum())
    print(
        f"read {len(documents)} document(s) from {documents_path}; "
        f"{without_any} carry no {args.clause_type} provision"
    )

    # Style the full label set before filtering, so a subtype keeps the same
    # colour and marker whatever --min-cbas is set to and figures stay
    # comparable across runs.
    colors, markers, labels = subtype_style(subtypes, args.clause_type)

    if args.min_cbas > 0:
        subtypes, dropped = frequent_subtypes(documents, subtypes, args.min_cbas)
        if dropped:
            hidden = ", ".join(
                f"{subtype} ({count})" for subtype, count in sorted(
                    dropped, key=lambda item: (-item[1], item[0])
                )
            )
            print(
                f"  hiding {len(dropped)} subtype(s) carried by fewer than "
                f"{args.min_cbas} CBAs: {hidden}"
            )
        if not subtypes:
            raise SystemExit(
                f"no subtype is carried by at least {args.min_cbas} CBAs; "
                "lower --min-cbas"
            )

    figures = {
        f"{prefix}_subtype_share.png": plot_overall(
            documents, subtypes, colors, labels, args.clause_type
        ),
        f"{prefix}_share_by_period.png": plot_by_period(
            documents,
            subtypes,
            colors,
            markers,
            labels,
            args.min_group_docs,
            args.clause_type,
            industry_adjusted=args.industry_adjusted,
            min_sector_docs=args.min_sector_docs,
            min_cell_docs=args.min_cell_docs,
        ),
        f"{prefix}_share_by_sector.png": plot_by_sector(
            documents,
            subtypes,
            colors,
            labels,
            args.min_sector_docs,
            args.clause_type,
            show_unknown=args.show_unknown,
        ),
    }

    if all(f"pct_beneficiary_{b}" in documents.columns for b in BENEFICIARY_LABELS):
        b_colors, b_markers, b_labels = beneficiary_style(BENEFICIARY_LABELS)
        figures[f"{prefix}_beneficiary_share.png"] = plot_beneficiary_overall(
            documents, BENEFICIARY_LABELS, b_colors, b_labels, args.clause_type
        )
        figures[f"{prefix}_beneficiary_share_by_period.png"] = plot_beneficiary_by_period(
            documents,
            BENEFICIARY_LABELS,
            b_colors,
            b_markers,
            b_labels,
            args.min_group_docs,
            args.clause_type,
            industry_adjusted=args.industry_adjusted,
            min_sector_docs=args.min_sector_docs,
            min_cell_docs=args.min_cell_docs,
        )
        figures[f"{prefix}_beneficiary_share_by_sector.png"] = plot_beneficiary_by_sector(
            documents,
            BENEFICIARY_LABELS,
            b_colors,
            b_labels,
            args.min_sector_docs,
            args.clause_type,
            show_unknown=args.show_unknown,
        )
    else:
        print(
            "  [warn] document table carries no pct_beneficiary_* columns; "
            "rebuild it with the updated link_classifications.py to add the "
            "beneficiary figures"
        )

    for name, figure in figures.items():
        if figure is not None:
            _save(figure, args.output_dir / name, args.dpi, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
