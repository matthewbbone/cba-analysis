"""Plot subtype prevalence and beneficiary share for linked stage-04 classifications.

Reads the document table written by ``link_classifications.py`` and renders six
figures: subtype prevalence overall, across contract-expiration cohorts, and by
NAICS sector, plus the same three cuts for beneficiary share.

Below the top taxonomy level, the three subtype figures are drawn once per
level-1 category rather than once for the whole level: a level-2 run over a
taxonomy with three level-1 categories therefore writes 3x as many subtype
figures, each comparing only the subtypes that were offered as alternatives to
one another under one level-1 label. These per-category figures replace the
usual "no provision of the type at all" bucket with an "other" bucket specific
to the category: the share of CBAs holding a provision link_classifications.py
already placed under this level-1 label but that was then classified "other"
among its children, rather than any of the named subtypes -- not the same
population as a CBA carrying no provision of the clause type at all. The
beneficiary figures are unaffected -- beneficiary is independent of subtype.

The subtype figures report a share of *documents*, not of provisions: for each
group, the percentage of CBAs carrying at least one provision of each subtype,
plus (outside the per-category level-2+ figures) the percentage carrying no
provision of the type at all.  A document with ten task-allocation provisions
counts once, and because a CBA can carry several subtypes the shares in a
group do not sum to 100.

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
from matplotlib.colors import LinearSegmentedColormap

from pipeline.stg_02_extract.structure_provision import (
    RESERVED_SUBTYPE_LABEL,
    SubtypeNode,
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

# The absence of any provision is its own series on every level-1 figure.  It is
# keyed by a name no subtype can collide with and painted neutral grey, so the
# six categorical hues stay reserved for the subtypes themselves.  Figures split
# by level-1 category (see ``level1_groups``) drop this series -- a CBA with no
# provision of the level-2 group's parent category is not the same thing as a
# CBA with no provision of the whole clause type -- and draw an
# ``other_in_<level1 label>`` series in its place, painted with the same grey so
# both read as "the catch-all bucket" wherever either appears.
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

# Fixed hues for the beneficiary labels with an intuitive colour, so a
# reader does not have to recheck the legend between beneficiary figures.
# Any other beneficiary label still draws from SERIES_COLORS.
BENEFICIARY_COLOR_OVERRIDES = {
    "workers": SERIES_COLORS[0],  # blue
    "employer": SERIES_COLORS[7],  # red
    "unclear": SERIES_COLORS[5],  # green
}

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


def level1_groups(
    taxonomy: Mapping[str, SubtypeNode], subtypes: Sequence[str]
) -> dict[str, list[str]]:
    """Partition ``subtypes`` by their level-1 ancestor in ``taxonomy``.

    Used to split a deeper level's flat subtype list into one series group per
    top-level classification, so a level-2+ run can render a separate set of
    figures per level-1 category instead of one figure mixing every branch.
    The injected "other" label names no node in the taxonomy and so belongs to
    no single level-1 ancestor -- it is dropped from every group rather than
    assigned to one arbitrarily. Groups are returned in taxonomy declaration
    order, each holding its members in ``subtypes`` order.
    """

    ancestor: dict[str, str] = {}

    def walk(nodes: Mapping[str, SubtypeNode], top: str) -> None:
        for label, node in nodes.items():
            ancestor[label] = top
            walk(node.children, top)

    for top_label, node in taxonomy.items():
        walk({top_label: node}, top_label)

    groups: dict[str, list[str]] = {top_label: [] for top_label in taxonomy}
    for subtype in subtypes:
        top_label = ancestor.get(subtype)
        if top_label is not None:
            groups[top_label].append(subtype)
    return groups


def group_series_keys(
    top_label: str,
    members: Sequence[str],
    documents: pd.DataFrame,
    colors: dict[str, str],
    markers: dict[str, str],
    labels: dict[str, str],
) -> list[str]:
    """Series to draw for one level-1 category's figures.

    ``members`` plus that category's own "other" bucket, in place of the
    level-1 absence series: a CBA with a provision link_classifications.py
    already placed under ``top_label`` but that was then classified "other"
    among its children, not a CBA with no provision of the clause type at all.

    Mutates ``colors``/``markers``/``labels`` in place to add a style entry for
    the synthetic ``other_in_<top_label>`` key, painted the same grey as
    :data:`NONE_KEY` so both read as "the catch-all bucket" wherever either
    appears. Falls back to the bare member list, with a warning, when
    ``documents`` predates the ``has_other_in_<top_label>`` column -- an older
    table built before this bucket existed.
    """

    group_keys = list(members)
    other_column = f"has_other_in_{top_label}"
    if other_column not in documents.columns:
        print(
            f"  [warn] documents table carries no {other_column} column; "
            "rebuild it with the updated link_classifications.py to show the "
            f"'other' bucket for {top_label!r}"
        )
        return group_keys

    other_key = f"other_in_{top_label}"
    colors.setdefault(other_key, NONE_COLOR)
    markers.setdefault(other_key, NONE_MARKER)
    labels.setdefault(other_key, f"Other {top_label.replace('_', ' ')} subtype")
    group_keys.append(other_key)
    return group_keys


def series_order(subtypes: Sequence[str], include_none: bool = True) -> list[str]:
    """Return the plotting order: subtypes as declared, absence last.

    ``include_none`` is off for a figure split by level-1 category: those
    figures carry their own ``other_in_<level1 label>`` catch-all as an
    ordinary member of ``subtypes`` instead, so appending :data:`NONE_KEY` on
    top would draw two different catch-all buckets on the same figure.
    """

    return [*subtypes, NONE_KEY] if include_none else list(subtypes)


def _subtype_shares_for(
    frame: pd.DataFrame, subtypes: Sequence[str], include_none: bool = True
) -> pd.Series:
    """Percentage of the CBAs in ``frame`` carrying each series.

    The single definition of the subtype statistic, so the raw lines and the
    composition-adjusted lines cannot drift apart.  Index is the subtypes plus
    :data:`NONE_KEY` when ``include_none``; the values do not sum to 100
    because one CBA can carry several subtypes.
    """

    total = len(frame)
    shares = {}
    for subtype in subtypes:
        column = f"has_{subtype}"
        carried = int(frame[column].astype(bool).sum()) if column in frame else 0
        shares[subtype] = carried / total * 100 if total else 0.0
    if include_none:
        without_any = int((frame["n_provisions"] == 0).sum())
        shares[NONE_KEY] = without_any / total * 100 if total else 0.0
    return pd.Series(shares, dtype=float)


def document_shares(
    documents: pd.DataFrame,
    group: str,
    subtypes: Sequence[str],
    include_none: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return per-group percentages of documents carrying each series.

    Rows are groups, columns are the subtypes plus :data:`NONE_KEY` when
    ``include_none``. A row does not sum to 100: one CBA can carry several
    subtypes, and the absence series is the complement of carrying any.
    """

    grouped = documents.groupby(group, dropna=True)
    totals = grouped.size()
    keys = series_order(subtypes, include_none=include_none)
    shares = pd.DataFrame(
        [
            _subtype_shares_for(frame, subtypes, include_none=include_none)[keys]
            for _, frame in grouped
        ],
        index=totals.index,
        columns=keys,
    )
    return shares, totals


def overall_shares(
    documents: pd.DataFrame, subtypes: Sequence[str], include_none: bool = True
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
    if include_none:
        counts[NONE_KEY] = int((documents["n_provisions"] == 0).sum())
    counts = pd.Series(counts, dtype=float)
    return counts / max(len(documents), 1) * 100, counts


def beneficiary_style(
    beneficiaries: Sequence[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Colors, markers and display labels for the beneficiary series.

    Unlike :func:`subtype_style`, there is no absence series here: every
    provision names exactly one beneficiary, so the three labels already
    exhaust the space. Labels in BENEFICIARY_COLOR_OVERRIDES always draw
    their fixed hue; any other label falls back to the categorical
    sequence, skipping the reserved hues.
    """

    fallback_colors = [
        color
        for color in SERIES_COLORS
        if color not in BENEFICIARY_COLOR_OVERRIDES.values()
    ]
    next_fallback = iter(fallback_colors)
    colors = {
        beneficiary: BENEFICIARY_COLOR_OVERRIDES.get(beneficiary)
        or next(next_fallback)
        for beneficiary in beneficiaries
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
    include_none: bool = True,
) -> "matplotlib.figure.Figure":
    shares, counts = overall_shares(documents, subtypes, include_none=include_none)
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
    include_none: bool = True,
) -> "matplotlib.figure.Figure | None":
    known = documents[
        documents["expire_period"].notna() & (documents["expire_period"] != "")
    ]
    if known.empty:
        print("  [warn] no document carries an expiration cohort; skipping figure")
        return None
    shares, totals = document_shares(
        known, "expire_period", subtypes, include_none=include_none
    )
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
    keys = series_order(subtypes, include_none=include_none)
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
    include_none: bool = True,
) -> "matplotlib.figure.Figure | None":
    grouped = _grouped_by_sector(documents, min_docs, keep_unknown=show_unknown)
    if grouped is None:
        print("  [warn] no documents carry a NAICS sector; skipping figure")
        return None
    known, doc_counts = grouped

    shares, totals = document_shares(
        known, "sector_group", subtypes, include_none=include_none
    )
    order = _sector_order(totals)
    shares = shares.loc[order]

    keys = series_order(subtypes, include_none=include_none)
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


FOCUS_COUNT_CAPTION = (
    "Counts of individual provisions, not CBAs -- a document with several such "
    "provisions contributes to every applicable series once per provision."
)

FOCUS_BENEFICIARIES = ("employer", "workers")


def _named_other_style_handles() -> list:
    """Two proxies naming solid-vs-dotted, so each beneficiary needs only one legend entry.

    Mirrors :func:`_style_legend_handles`: colour already carries the
    beneficiary, so line style is free to carry named-subtype-vs-"other"
    instead of being repeated per beneficiary.
    """

    from matplotlib.lines import Line2D

    return [
        Line2D([], [], color=TEXT_PRIMARY, linewidth=2, label="Named subtype at level 2"),
        Line2D(
            [],
            [],
            color=TEXT_PRIMARY,
            linewidth=2,
            linestyle=ADJUSTED_LINESTYLE,
            label="Classified 'other' at level 2",
        ),
    ]


def plot_focus_category_counts_by_period(
    provisions: pd.DataFrame,
    level1_label: str,
    min_docs: int,
    beneficiaries: Sequence[str] = FOCUS_BENEFICIARIES,
) -> "matplotlib.figure.Figure | None":
    """Time series of raw provision counts for one level-1 category, by beneficiary.

    Unlike the share figures elsewhere in this module, which report the
    percentage of CBAs carrying a subtype, this counts the provisions
    themselves: one solid line per beneficiary for provisions the category's
    named level-2 subtypes absorbed, and one dotted line, same colour and
    marker, for the "other" bucket its own children left over. Beneficiary
    colours match :func:`beneficiary_style`, so a reader does not have to
    recheck the legend between this figure and the beneficiary-share figures.
    Requires a provisions table built at level 2 or deeper, whose
    ``subtype_1`` column names each provision's level-1 ancestor independently
    of ``subtype``, which by then names the deeper label.
    """

    if "subtype_1" not in provisions.columns:
        print(
            "  [warn] provisions table carries no subtype_1 column; rebuild it "
            "with the updated link_classifications.py at --level 2 or deeper "
            "to show the focus-category figure"
        )
        return None

    scoped = provisions[
        (provisions["subtype_1"] == level1_label)
        & provisions["beneficiary"].isin(beneficiaries)
        & provisions["expire_period"].notna()
        & (provisions["expire_period"] != "")
    ]
    if scoped.empty:
        print(
            f"  [warn] no {'/'.join(beneficiaries)} {level1_label} provision "
            "carries an expiration cohort; skipping focus-category figure"
        )
        return None

    totals = scoped.groupby("expire_period").size()
    periods = sorted(totals.index)
    kept = [period for period in periods if totals.get(period, 0) >= min_docs]
    dropped = [
        f"{period} ({_plural(int(totals.get(period, 0)), 'provision')})"
        for period in periods
        if period not in kept
    ]
    if not kept:
        print(
            "  [warn] no expiration cohort meets --min-group-docs; skipping "
            "focus-category figure"
        )
        return None

    colors, markers, labels = beneficiary_style(beneficiaries)
    category_title = level1_label.replace("_", " ")

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    positions = range(len(kept))
    highest = 0
    for beneficiary in beneficiaries:
        subset = scoped[scoped["beneficiary"] == beneficiary]
        is_other = subset["subtype"] == RESERVED_SUBTYPE_LABEL
        named = subset[~is_other].groupby("expire_period").size()
        other = subset[is_other].groupby("expire_period").size()
        named_values = [int(named.get(period, 0)) for period in kept]
        other_values = [int(other.get(period, 0)) for period in kept]
        highest = max(highest, max(named_values, default=0), max(other_values, default=0))

        ax.plot(
            list(positions),
            named_values,
            color=colors[beneficiary],
            marker=markers[beneficiary],
            markersize=5.5,
            linewidth=2,
            label=labels[beneficiary],
            zorder=2,
        )
        ax.plot(
            list(positions),
            other_values,
            color=colors[beneficiary],
            marker=markers[beneficiary],
            markersize=5.5,
            markerfacecolor="none",
            linewidth=1.8,
            linestyle=ADJUSTED_LINESTYLE,
            label="_nolegend_",
            zorder=2,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [f"{period}\n{int(totals.get(period, 0))} provisions" for period in kept]
    )
    ax.set_ylim(0, max(highest * 1.15, 5))
    ax.set_xlabel(PERIOD_AXIS_LABEL, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Number of provisions", color=TEXT_SECONDARY, fontsize=9)
    extra = _named_other_style_handles()
    ax.set_title(
        f"{category_title.capitalize()} provisions by beneficiary, by expiration cohort",
        loc="left",
        color=TEXT_PRIMARY,
        pad=_title_pad(len(beneficiaries) + len(extra), ncol=2),
    )
    _tidy(ax, axis="y")
    _legend(ax, ncol=2, extra_handles=extra)
    caption = PERIOD_CAPTION + " " + FOCUS_COUNT_CAPTION
    if dropped:
        caption += f" Cohorts below {min_docs} provisions omitted: {', '.join(dropped)}."
    _caption(fig, ax, caption)
    return fig


# Eras coarser than the five-year cohorts elsewhere in this module: a heatmap
# with one column per (beneficiary, cohort) pair would need as many columns as
# there are cohorts times beneficiaries, crowding out the level-1 row labels
# it exists to compare. Bounds are inclusive expiration years.
LEVEL1_ERA_BUCKETS = (("00-14", 2000, 2014), ("15-29", 2015, 2029))

# White to violet: a hue this module does not already use for a beneficiary or
# a subtype, so shading a cell by magnitude never reads as though it were
# naming an identity the row/column labels do not already carry.
HEATMAP_CMAP = LinearSegmentedColormap.from_list("cba_seq_violet", ["#ffffff", "#4a3aa7"])


def level1_beneficiary_era_shares(
    provisions: pd.DataFrame,
    level1_labels: Sequence[str],
    beneficiaries: Sequence[str] = FOCUS_BENEFICIARIES,
    era_buckets: Sequence[tuple[str, int, int]] = LEVEL1_ERA_BUCKETS,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Share of CBAs carrying each level-1 category (rows), by beneficiary-era (columns).

    Each column's own denominator is the CBAs holding at least one provision
    naming that column's beneficiary within that column's era -- not the whole
    corpus -- since a CBA with no such provision could not carry any subtype of
    it either. A CBA counts once per cell however many matching provisions it
    holds, the same document-level counting unit as the rest of this module's
    subtype-share figures. Rows are the taxonomy's level-1 categories in
    declared order, plus the "other" escape hatch stage 4 offers at every
    level, appended last so it reads as the catch-all it is. Requires a
    provisions table built at level 1, whose ``subtype`` column already names
    the level-1 label (or "other") directly.

    Returns the share matrix, each column's denominator (CBA count), and the
    number of CBAs that carry a provision naming an included beneficiary but
    only outside every era -- present nowhere in the matrix, not merely absent
    from one cell.
    """

    rows = [*level1_labels, RESERVED_SUBTYPE_LABEL]
    shares = pd.DataFrame(index=rows, dtype=float)
    totals: dict[str, int] = {}
    covered_docs: set = set()
    in_scope_docs = set(
        provisions.loc[provisions["beneficiary"].isin(beneficiaries), "document_id"]
    )

    for beneficiary in beneficiaries:
        for era_label, start_year, end_year in era_buckets:
            column = f"{beneficiary.capitalize()} ({era_label})"
            mask = (provisions["beneficiary"] == beneficiary) & provisions[
                "expire_year"
            ].between(start_year, end_year)
            scoped = provisions[mask]
            covered_docs |= set(scoped["document_id"])
            denominator = scoped["document_id"].nunique()
            totals[column] = int(denominator)
            if denominator == 0:
                shares[column] = [0.0 for _ in rows]
                continue
            shares[column] = [
                scoped.loc[scoped["subtype"] == row, "document_id"].nunique()
                / denominator
                * 100
                for row in rows
            ]

    excluded_docs = len(in_scope_docs - covered_docs)
    return shares, pd.Series(totals), excluded_docs


def plot_level1_beneficiary_era_heatmap(
    provisions: pd.DataFrame,
    level1_labels: Sequence[str],
    clause_type: str,
    beneficiaries: Sequence[str] = FOCUS_BENEFICIARIES,
    era_buckets: Sequence[tuple[str, int, int]] = LEVEL1_ERA_BUCKETS,
) -> "matplotlib.figure.Figure | None":
    """Heatmap of the share of CBAs: level-1 category x (beneficiary, era).

    Cell shading is a single sequential hue standing for magnitude alone --
    independent of which beneficiary or era a column names -- so the color
    channel never competes with the identity the row and column labels already
    carry. Requires a provisions table built at level 1 (``subtype`` already
    the level-1 label or "other") with the harmonized ``expire_year`` column.
    """

    if "expire_year" not in provisions.columns:
        print(
            "  [warn] provisions table carries no expire_year column; skipping "
            "level-1 beneficiary/era heatmap"
        )
        return None

    shares, totals, excluded_docs = level1_beneficiary_era_shares(
        provisions, level1_labels, beneficiaries, era_buckets
    )
    values = shares.to_numpy(dtype=float)
    if totals.sum() == 0:
        print(
            "  [warn] no CBA falls into a plotted beneficiary/era column; "
            "skipping level-1 beneficiary/era heatmap"
        )
        return None

    row_labels = [row.replace("_", " ").capitalize() for row in shares.index]
    column_labels = [
        f"{column}\n{int(totals[column])} CBAs" for column in shares.columns
    ]
    vmax = max(float(values.max()), 1.0)

    fig, ax = plt.subplots(
        figsize=(1.9 * len(shares.columns) + 2.2, 0.65 * len(shares.index) + 1.8)
    )
    mesh = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(shares.columns)))
    ax.set_xticklabels(column_labels, color=TEXT_SECONDARY, fontsize=9)
    ax.set_yticks(range(len(shares.index)))
    ax.set_yticklabels(row_labels, color=TEXT_SECONDARY, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # imshow's cells touch by default; a thin white seam substitutes for the
    # gridlines every other figure in this module draws behind its marks.
    ax.set_xticks([x - 0.5 for x in range(1, len(shares.columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(shares.index))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for row_index, row in enumerate(shares.index):
        for col_index, column in enumerate(shares.columns):
            value = float(shares.loc[row, column])
            text_color = "white" if value / vmax > 0.5 else TEXT_PRIMARY
            ax.text(
                col_index,
                row_index,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    colorbar.set_label("% of CBAs", color=TEXT_SECONDARY, fontsize=9)

    category_title = clause_type.replace("_", " ").capitalize()
    ax.set_title(
        f"{category_title} provisions by level-1 category, beneficiary and era",
        loc="left",
        color=TEXT_PRIMARY,
        pad=14,
    )
    caption = (
        "Each cell is the % of CBAs in its column carrying at least one "
        "provision of that row's category -- a CBA counts once however many "
        "matching provisions it holds, so a column's cells can sum past 100%. "
        "A column's own denominator (below its label) is the CBAs holding at "
        "least one provision naming that beneficiary within that era, not the "
        "whole corpus. Eras group five-year expiration cohorts into "
        f"{era_buckets[0][1]}-{era_buckets[0][2]} and "
        f"{era_buckets[1][1]}-{era_buckets[1][2]}; a beneficiary outside "
        f"{' or '.join(beneficiaries)} is excluded entirely."
    )
    if excluded_docs:
        caption += (
            f" {_plural(excluded_docs, 'CBA')} carrying a provision naming an "
            "included beneficiary fall outside every era and are excluded from "
            "every column's denominator."
        )
    _caption(fig, ax, caption)
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
        "--focus-category",
        default=None,
        metavar="LEVEL1_LABEL",
        help=(
            "level-1 category to additionally plot as a provision-count time "
            "series, split into its 'other' bucket vs named subtypes with one "
            f"line pair per beneficiary in {', '.join(FOCUS_BENEFICIARIES)}. "
            "Requires the provisions table built at --level 2 or deeper. "
            "Example: preemptive_rights"
        ),
    )
    parser.add_argument(
        "--level1-heatmap",
        action="store_true",
        help=(
            "additionally plot a heatmap of provision counts: level-1 category "
            "(plus 'other') by row, beneficiary crossed with a 2000-2014 / "
            f"2015-2029 era by column, for beneficiaries in {', '.join(FOCUS_BENEFICIARIES)}. "
            "Always reads the level-1 provisions table regardless of --level"
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
    if args.focus_category is not None and args.focus_category not in spec.subtype_taxonomy:
        raise SystemExit(
            f"--focus-category {args.focus_category!r} is not a level-1 label of "
            f"the {args.clause_type} taxonomy; choose one of "
            f"{', '.join(spec.subtype_taxonomy)}"
        )
    # Must match the --level the document table was built with: stage 4 offers
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

    figures: dict[str, "matplotlib.figure.Figure | None"] = {}
    if args.level == 1:
        figures[f"{prefix}_subtype_share.png"] = plot_overall(
            documents, subtypes, colors, labels, args.clause_type
        )
        figures[f"{prefix}_share_by_period.png"] = plot_by_period(
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
        )
        figures[f"{prefix}_share_by_sector.png"] = plot_by_sector(
            documents,
            subtypes,
            colors,
            labels,
            args.min_sector_docs,
            args.clause_type,
            show_unknown=args.show_unknown,
        )
    else:
        # Below the top level, one flat figure per statistic would mix series
        # from every level-1 branch; split into one figure set per level-1
        # category instead, so each figure only compares subtypes that were
        # actually offered as alternatives to one another during classification.
        groups = level1_groups(spec.subtype_taxonomy, subtypes)
        for top_label, members in groups.items():
            if not members:
                print(
                    f"  [warn] no level-{args.level} subtype under {top_label!r} "
                    "survives filtering; skipping its figures"
                )
                continue
            group_keys = group_series_keys(
                top_label, members, documents, colors, markers, labels
            )
            group_clause_type = f"{args.clause_type} / {top_label.replace('_', ' ')}"
            figures[f"{prefix}_{top_label}_subtype_share.png"] = plot_overall(
                documents, group_keys, colors, labels, group_clause_type,
                include_none=False,
            )
            figures[f"{prefix}_{top_label}_share_by_period.png"] = plot_by_period(
                documents,
                group_keys,
                colors,
                markers,
                labels,
                args.min_group_docs,
                group_clause_type,
                industry_adjusted=args.industry_adjusted,
                min_sector_docs=args.min_sector_docs,
                min_cell_docs=args.min_cell_docs,
                include_none=False,
            )
            figures[f"{prefix}_{top_label}_share_by_sector.png"] = plot_by_sector(
                documents,
                group_keys,
                colors,
                labels,
                args.min_sector_docs,
                group_clause_type,
                show_unknown=args.show_unknown,
                include_none=False,
            )

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

    if args.focus_category is not None:
        provisions_path = args.input_dir / f"{prefix}_provisions.csv"
        if not provisions_path.is_file():
            print(
                f"  [warn] provisions table not found: {provisions_path}; "
                "skipping focus-category figure"
            )
        else:
            provisions = pd.read_csv(provisions_path, dtype={"expire_period": str})
            figure_name = f"{prefix}_{args.focus_category}_counts_by_period.png"
            figures[figure_name] = plot_focus_category_counts_by_period(
                provisions,
                args.focus_category,
                args.min_group_docs,
            )

    if args.level1_heatmap:
        level1_prefix = table_prefix(args.clause_type, args.source, 1)
        level1_provisions_path = args.input_dir / f"{level1_prefix}_provisions.csv"
        if not level1_provisions_path.is_file():
            print(
                f"  [warn] level-1 provisions table not found: "
                f"{level1_provisions_path}; run link_classifications.py with "
                "--level 1 first; skipping level-1 heatmap"
            )
        else:
            level1_provisions = pd.read_csv(level1_provisions_path)
            figures[f"{level1_prefix}_beneficiary_era_heatmap.png"] = (
                plot_level1_beneficiary_era_heatmap(
                    level1_provisions, list(spec.subtype_taxonomy), args.clause_type
                )
            )

    for name, figure in figures.items():
        if figure is not None:
            _save(figure, args.output_dir / name, args.dpi, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
