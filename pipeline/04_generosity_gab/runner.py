"""Compute Gabriel-style generosity rankings from classified CBA segments.

This stage reads segmentation outputs plus clause labels, ranks segments within
each clause type, converts ranks to within-clause scores, and aggregates those
scores up to document-level composites. Paths default to `CACHE_DIR` when set
and otherwise fall back to `outputs/` inside the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

SEGMENT_ATTRIBUTE_NAME = "segment_generosity"
SEGMENT_ATTRIBUTE_PROMPT = (
    "How generous this segment is for workers overall. "
    "Higher means stronger worker benefits/rights and fewer worker burdens."
)
MIN_DOCS_PER_CLAUSE_FOR_RANKING = 5
TOP_CLAUSE_TYPES_DEFAULT = 10

# Keep aligned with 04_generosity_llm/runner.py.
EXCLUDED_PROCEDURAL_CLAUSE_TYPES = [
    "Recognition Clause",
    "Recognition",
    "Parties to Agreement and Preamble",
    "Parties to Agreement",
    "Preamble",
    "Bargaining Unit",
    "",
]
EXCLUDED_PROCEDURAL_CLAUSE_TYPE_KEYS = {
    re.sub(r"[^a-z0-9]+", " ", str(label).strip().lower()).strip()
    for label in EXCLUDED_PROCEDURAL_CLAUSE_TYPES
    if str(label).strip()
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _default_cache_dir() -> str:
    return os.environ.get("CACHE_DIR", "").strip()


def _default_input_dir() -> Path:
    """Return the default segmentation directory for the active cache layout."""
    cache_dir = _default_cache_dir()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "02_segmentation_output" / "dol_archive"
    return (_project_root() / "outputs" / "02_segmentation_output" / "dol_archive").resolve()


def _default_classification_dir() -> Path:
    """Return the default classification directory for the active cache layout."""
    cache_dir = _default_cache_dir()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "03_classification_output" / "dol_archive"
    return (_project_root() / "outputs" / "03_classification_output" / "dol_archive").resolve()


def _default_output_dir() -> Path:
    """Return the default output directory for Gabriel scoring artifacts."""
    cache_dir = _default_cache_dir()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "04_generosity_gab_output" / "dol_archive"
    return (_project_root() / "outputs" / "04_generosity_gab_output" / "dol_archive").resolve()


def _normalize_document_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if re.fullmatch(r"document_\d+", value):
        return value
    if re.fullmatch(r"\d+", value):
        return f"document_{int(value)}"
    return value


def _parse_document_num(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"document_(\d+)", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _parse_segment_num(path_or_name: str | Path) -> int:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    m = re.fullmatch(r"segment_(\d+)\.txt", name)
    if not m:
        return 10**12
    return int(m.group(1))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _normalize_clause_type_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _is_excluded_procedural_clause_type(clause_type: Any) -> bool:
    return _normalize_clause_type_key(clause_type) in EXCLUDED_PROCEDURAL_CLAUSE_TYPE_KEYS


class GenerosityGabRunner:
    """Aggregate segment-level pairwise generosity comparisons into document scores."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        input_dir: Path,
        classification_dir: Path,
        output_dir: Path,
        model: str = "gpt-5-nano",
        n_rounds: int = 5,
        matches_per_round: int = 3,
        n_parallels: int = 50,
        top_clause_types: int = TOP_CLAUSE_TYPES_DEFAULT,
        reset_files: bool = False,
        max_chars_per_segment: int = 8000,
    ) -> None:
        if str(cache_dir).strip():
            self.cache_dir = _resolve_path(str(cache_dir), _project_root())
        else:
            self.cache_dir = _project_root()

        self.input_dir = input_dir.resolve() if input_dir.is_absolute() else (self.cache_dir / input_dir).resolve()
        self.classification_dir = (
            classification_dir.resolve()
            if classification_dir.is_absolute()
            else (self.cache_dir / classification_dir).resolve()
        )
        self.output_dir = output_dir.resolve() if output_dir.is_absolute() else (self.cache_dir / output_dir).resolve()

        self.model = str(model).strip() or "gpt-5-nano"
        self.n_rounds = max(1, int(n_rounds))
        self.matches_per_round = max(1, int(matches_per_round))
        self.n_parallels = max(1, int(n_parallels))
        self.top_clause_types = max(1, int(top_clause_types))
        self.reset_files = bool(reset_files)
        self.max_chars_per_segment = max(1, int(max_chars_per_segment))

    def _list_document_dirs(self) -> list[Path]:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            return []
        docs = []
        for p in self.input_dir.iterdir():
            if not p.is_dir():
                continue
            if re.fullmatch(r"document_\d+", p.name) is None:
                continue
            seg_dir = p / "segments"
            if seg_dir.exists() and any(seg_dir.glob("segment_*.txt")):
                docs.append(p)
                continue
            if any(p.glob("segment_*.txt")):
                docs.append(p)
        return sorted(docs, key=_parse_document_num)

    @staticmethod
    def _list_segment_files(doc_dir: Path) -> list[Path]:
        seg_dir = doc_dir / "segments"
        if seg_dir.exists():
            paths = [p for p in seg_dir.glob("segment_*.txt") if p.is_file()]
        else:
            paths = [p for p in doc_dir.glob("segment_*.txt") if p.is_file()]
        return sorted([p for p in paths if _parse_segment_num(p) != 10**12], key=_parse_segment_num)

    @staticmethod
    def _segment_id(document_id: str, segment_number: int) -> str:
        return f"{document_id}::segment_{int(segment_number)}"

    @staticmethod
    def _slugify_clause_type(clause_type: str) -> str:
        raw = str(clause_type).strip() or "OTHER"
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")
        return slug or "OTHER"

    def _load_segment_clause_type(self, document_id: str, segment_number: int) -> str:
        payload_path = self.classification_dir / document_id / f"segment_{segment_number}.json"
        if not payload_path.exists():
            return "OTHER"
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            return "OTHER"
        if not isinstance(payload, dict):
            return "OTHER"

        labels = payload.get("labels", [])
        if isinstance(labels, list):
            for label in labels:
                normalized = str(label).strip()
                if normalized:
                    return normalized

        label = str(payload.get("label", "")).strip()
        return label if label else "OTHER"

    def _collect_segment_rows(
        self,
        *,
        document_id: str | None,
        sample_size: int | None,
        seed: int,
        max_segments: int | None,
    ) -> list[dict[str, Any]]:
        doc_dirs = self._list_document_dirs()
        if document_id:
            doc_dirs = [d for d in doc_dirs if d.name == document_id]
        if sample_size is not None and sample_size < len(doc_dirs):
            rng = random.Random(int(seed))
            doc_dirs = sorted(rng.sample(doc_dirs, int(sample_size)), key=_parse_document_num)

        rows: list[dict[str, Any]] = []
        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            segment_files = self._list_segment_files(doc_dir)
            if max_segments is not None:
                segment_files = segment_files[: max(0, int(max_segments))]

            for segment_file in segment_files:
                segment_number = _parse_segment_num(segment_file)
                if segment_number == 10**12:
                    continue
                raw_text = segment_file.read_text(encoding="utf-8", errors="replace")
                is_truncated = len(raw_text) > self.max_chars_per_segment
                segment_text = raw_text[: self.max_chars_per_segment]
                rows.append(
                    {
                        "segment_id": self._segment_id(doc_id, segment_number),
                        "document_id": doc_id,
                        "segment_number": int(segment_number),
                        "clause_type": self._load_segment_clause_type(doc_id, int(segment_number)),
                        "segment_path": str(segment_file),
                        "segment_text": segment_text,
                        "text_char_count": len(raw_text),
                        "is_truncated": bool(is_truncated),
                    }
                )
        return [
            row
            for row in rows
            if not _is_excluded_procedural_clause_type(row.get("clause_type", ""))
        ]

    @staticmethod
    def _top_clause_types(rows: list[dict[str, Any]], top_k: int) -> list[tuple[str, int]]:
        doc_sets: dict[str, set[str]] = defaultdict(set)
        segment_counts: Counter[str] = Counter()
        for row in rows:
            clause_type = str(row.get("clause_type", "")).strip() or "OTHER"
            if clause_type == "OTHER":
                continue
            if _is_excluded_procedural_clause_type(clause_type):
                continue
            document_id = str(row.get("document_id", "")).strip()
            if not document_id:
                continue
            doc_sets[clause_type].add(document_id)
            segment_counts[clause_type] += 1

        clause_types = sorted(
            doc_sets.keys(),
            key=lambda clause_type: (
                -len(doc_sets[clause_type]),  # primary: prevalence across documents
                -segment_counts[clause_type],  # tie-break: segment frequency
                clause_type.lower(),
            ),
        )
        top = clause_types[: max(1, int(top_k))]
        return [(clause_type, len(doc_sets[clause_type])) for clause_type in top]

    async def run_gabriel(self, clause_type: str, segments_df, *, reset_files: bool):
        try:
            import gabriel
        except Exception as exc:
            raise RuntimeError(
                "The `gabriel` package is required for GAB ranking. "
                "Install it in this environment before running this pipeline."
            ) from exc

        clause_slug = self._slugify_clause_type(clause_type)
        rank_save_dir = self.output_dir / "runs" / "rank_segment_generosity" / clause_slug
        rank_save_dir.mkdir(parents=True, exist_ok=True)

        result = await gabriel.rank(
            segments_df,
            column_name="segment_text",
            attributes={SEGMENT_ATTRIBUTE_NAME: SEGMENT_ATTRIBUTE_PROMPT},
            save_dir=str(rank_save_dir),
            model=self.model,
            n_rounds=self.n_rounds,
            matches_per_round=self.matches_per_round,
            modality="text",
            reset_files=bool(reset_files),
            id_column="segment_id",
            n_parallels=self.n_parallels,
        )
        return result

    @staticmethod
    def _build_segment_rankings(segments_df, gab_rank_df):
        rank_df = gab_rank_df[["segment_id", SEGMENT_ATTRIBUTE_NAME]].copy()
        merged = segments_df.merge(rank_df, on="segment_id", how="left")
        merged = merged.rename(columns={SEGMENT_ATTRIBUTE_NAME: "segment_generosity_score"})

        merged["segment_rank"] = None
        merged["segment_percentile"] = None
        valid_mask = merged["segment_generosity_score"].notna()
        n_ranked = int(valid_mask.sum())
        if n_ranked > 0:
            merged.loc[valid_mask, "segment_rank"] = (
                merged.loc[valid_mask, "segment_generosity_score"]
                .rank(ascending=False, method="average")
                .astype(float)
            )
            if n_ranked == 1:
                merged.loc[valid_mask, "segment_percentile"] = 100.0
            else:
                merged.loc[valid_mask, "segment_percentile"] = (
                    (n_ranked - merged.loc[valid_mask, "segment_rank"]) / (n_ranked - 1) * 100.0
                )

        merged = merged.sort_values(
            by=["segment_rank", "document_id", "segment_number"],
            na_position="last",
        ).reset_index(drop=True)
        return merged

    @staticmethod
    def _build_clause_type_document_rankings(segment_rankings_df):
        by_clause_doc = segment_rankings_df.groupby(["clause_type", "document_id"], sort=True)
        rows: list[dict[str, Any]] = []
        for (clause_type, document_id), group in by_clause_doc:
            ranked = group[group["segment_generosity_score"].notna()].copy()
            rows.append(
                {
                    "clause_type": str(clause_type),
                    "document_id": str(document_id),
                    "segment_count": int(len(group)),
                    "ranked_segment_count": int(len(ranked)),
                    "mean_segment_generosity_score": (
                        float(ranked["segment_generosity_score"].mean()) if len(ranked) > 0 else None
                    ),
                    "mean_segment_rank": float(ranked["segment_rank"].mean()) if len(ranked) > 0 else None,
                    # Clause-type score for cross-document comparison within this clause type.
                    "clause_type_score": float(ranked["segment_percentile"].mean()) if len(ranked) > 0 else None,
                }
            )

        import pandas as pd

        clause_docs = pd.DataFrame(rows)
        if clause_docs.empty:
            return clause_docs

        clause_docs["clause_type_document_rank"] = None
        for clause_type, idx in clause_docs.groupby("clause_type", sort=True).groups.items():
            clause_idx = list(idx)
            valid_mask = clause_docs.loc[clause_idx, "clause_type_score"].notna()
            if not bool(valid_mask.any()):
                continue
            rank_idx = list(clause_docs.loc[clause_idx].loc[valid_mask].index)
            clause_docs.loc[rank_idx, "clause_type_document_rank"] = (
                clause_docs.loc[rank_idx, "clause_type_score"]
                .rank(ascending=False, method="min")
                .astype(int)
            )

        clause_docs = clause_docs.sort_values(
            by=["clause_type", "clause_type_document_rank", "clause_type_score", "document_id"],
            ascending=[True, True, False, True],
            na_position="last",
        ).reset_index(drop=True)
        return clause_docs

    @staticmethod
    def _build_document_rankings(clause_type_document_rankings_df):
        by_doc = clause_type_document_rankings_df.groupby("document_id", sort=True)
        rows: list[dict[str, Any]] = []
        for document_id, group in by_doc:
            valid = group[group["clause_type_score"].notna()].copy()

            composite_score = float(valid["clause_type_score"].mean()) if len(valid) > 0 else None
            clause_type_count_used = int(valid["clause_type"].nunique()) if len(valid) > 0 else 0

            mean_segment_generosity_score = None
            mean_segment_rank = None
            weighted_df = valid[valid["ranked_segment_count"] > 0].copy() if len(valid) > 0 else valid
            if len(weighted_df) > 0:
                weight_sum = float(weighted_df["ranked_segment_count"].sum())
                if weight_sum > 0:
                    mean_segment_generosity_score = float(
                        (
                            weighted_df["mean_segment_generosity_score"]
                            * weighted_df["ranked_segment_count"]
                        ).sum()
                        / weight_sum
                    )
                    mean_segment_rank = float(
                        (
                            weighted_df["mean_segment_rank"]
                            * weighted_df["ranked_segment_count"]
                        ).sum()
                        / weight_sum
                    )

            rows.append(
                {
                    "document_id": str(document_id),
                    "segment_count": int(group["segment_count"].sum()),
                    "ranked_segment_count": int(group["ranked_segment_count"].sum()),
                    "clause_type_count_used": clause_type_count_used,
                    "mean_segment_generosity_score": mean_segment_generosity_score,
                    "mean_segment_rank": mean_segment_rank,
                    # Final document score requested by user:
                    # average of clause-type scores.
                    "composite_score": composite_score,
                }
            )

        import pandas as pd

        docs = pd.DataFrame(rows)
        if docs.empty:
            return docs

        docs["document_rank"] = None
        valid_mask = docs["composite_score"].notna()
        if bool(valid_mask.any()):
            rank_idx = list(docs.loc[valid_mask].index)
            docs.loc[rank_idx, "document_rank"] = (
                docs.loc[rank_idx, "composite_score"]
                .rank(ascending=False, method="min")
                .astype(int)
            )

        docs = docs.sort_values(
            by=["document_rank", "composite_score", "document_id"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)
        return docs

    async def run(
        self,
        *,
        sample_size: int | None,
        seed: int,
        document_id: str | None,
        max_segments: int | None,
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("The `pandas` package is required for this pipeline.") from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        segment_rows = self._collect_segment_rows(
            document_id=document_id,
            sample_size=sample_size,
            seed=seed,
            max_segments=max_segments,
        )

        if not segment_rows:
            summary = {
                "model": self.model,
                "input_dir": str(self.input_dir),
                "classification_dir": str(self.classification_dir),
                "output_dir": str(self.output_dir),
                "documents_processed": 0,
                "segments_processed": 0,
                "segments_ranked": 0,
                "documents_ranked": 0,
                "clause_types_ranked": 0,
                "n_rounds": self.n_rounds,
                "matches_per_round": self.matches_per_round,
                "n_parallels": self.n_parallels,
                "top_clause_type_count_target": int(self.top_clause_types),
                "top_clause_types_selected": [],
                "top_clause_type_document_counts": {},
                "top_clause_type_segment_counts": {},
                "excluded_procedural_clause_types": EXCLUDED_PROCEDURAL_CLAUSE_TYPES,
                "segment_attribute_prompt": SEGMENT_ATTRIBUTE_PROMPT,
                "min_documents_per_clause_for_ranking": MIN_DOCS_PER_CLAUSE_FOR_RANKING,
                "clause_type_score_method": "mean(segment_percentile within clause_type)",
                "document_composite_method": "mean(clause_type_score across selected top clause types)",
            }
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return summary

        segments_df = pd.DataFrame(segment_rows)
        top_clause_pairs = self._top_clause_types(segment_rows, self.top_clause_types)
        top_clause_types = [name for name, _ in top_clause_pairs]
        top_clause_type_document_counts = {name: int(count) for name, count in top_clause_pairs}
        top_clause_type_segment_counts: dict[str, int] = {}
        for row in segment_rows:
            clause_type = str(row.get("clause_type", "")).strip() or "OTHER"
            if clause_type in top_clause_type_document_counts:
                top_clause_type_segment_counts[clause_type] = top_clause_type_segment_counts.get(clause_type, 0) + 1

        if not top_clause_types:
            summary = {
                "model": self.model,
                "input_dir": str(self.input_dir),
                "classification_dir": str(self.classification_dir),
                "output_dir": str(self.output_dir),
                "documents_processed": int(segments_df["document_id"].nunique()),
                "segments_processed": int(len(segments_df)),
                "segments_ranked": 0,
                "documents_ranked": 0,
                "clause_types_ranked": 0,
                "n_rounds": self.n_rounds,
                "matches_per_round": self.matches_per_round,
                "n_parallels": self.n_parallels,
                "top_clause_type_count_target": int(self.top_clause_types),
                "top_clause_types_selected": [],
                "top_clause_type_document_counts": {},
                "top_clause_type_segment_counts": {},
                "excluded_procedural_clause_types": EXCLUDED_PROCEDURAL_CLAUSE_TYPES,
                "segment_attribute_prompt": SEGMENT_ATTRIBUTE_PROMPT,
                "min_documents_per_clause_for_ranking": MIN_DOCS_PER_CLAUSE_FOR_RANKING,
                "clause_type_score_method": "mean(segment_percentile within clause_type)",
                "document_composite_method": "mean(clause_type_score across selected top clause types)",
                "clause_types_skipped_for_low_document_count": {},
                "clause_type_document_rankings": {},
                "document_composite_rankings": {},
                "outputs": {},
            }
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return summary

        selected_segments_df = segments_df[segments_df["clause_type"].isin(top_clause_types)].copy().reset_index(drop=True)
        clause_types_skipped_low_docs = {
            str(clause_type): int(doc_count)
            for clause_type, doc_count in top_clause_type_document_counts.items()
            if int(doc_count) < MIN_DOCS_PER_CLAUSE_FOR_RANKING
        }

        clause_ranked_frames = []
        for clause_type in top_clause_types:
            clause_df = selected_segments_df[selected_segments_df["clause_type"] == clause_type].copy().reset_index(drop=True)
            if clause_df.empty:
                continue
            clause_doc_count = int(top_clause_type_document_counts.get(clause_type, 0))

            if clause_doc_count < MIN_DOCS_PER_CLAUSE_FOR_RANKING:
                skipped = clause_df.copy()
                skipped["segment_generosity_score"] = None
                skipped["segment_rank"] = None
                skipped["segment_percentile"] = None
                skipped["ranking_note"] = "insufficient-documents-for-clause-ranking"
                skipped["clause_type_segment_count"] = int(len(clause_df))
                skipped["clause_type_document_count"] = clause_doc_count
                clause_ranked_frames.append(skipped)
                continue

            if len(clause_df) == 1:
                single = clause_df.copy()
                single["segment_generosity_score"] = 0.0
                single["segment_rank"] = 1.0
                single["segment_percentile"] = 100.0
                single["ranking_note"] = "single-segment-clause"
                single["clause_type_segment_count"] = 1
                single["clause_type_document_count"] = clause_doc_count
                clause_ranked_frames.append(single)
                continue

            gab_rank_df = await self.run_gabriel(clause_type, clause_df, reset_files=self.reset_files)
            if not isinstance(gab_rank_df, pd.DataFrame):
                gab_rank_df = pd.DataFrame(gab_rank_df)

            if "segment_id" not in gab_rank_df.columns or SEGMENT_ATTRIBUTE_NAME not in gab_rank_df.columns:
                raise RuntimeError(
                    f"Unexpected GAB rank output format for clause_type={clause_type!r}. "
                    f"Expected columns: `segment_id`, `{SEGMENT_ATTRIBUTE_NAME}`."
                )

            ranked_clause = self._build_segment_rankings(clause_df, gab_rank_df)
            ranked_clause["ranking_note"] = "gabriel_rank"
            ranked_clause["clause_type_segment_count"] = int(len(clause_df))
            ranked_clause["clause_type_document_count"] = clause_doc_count
            clause_ranked_frames.append(ranked_clause)

        if clause_ranked_frames:
            segment_rankings = pd.concat(clause_ranked_frames, ignore_index=True)
            segment_rankings = segment_rankings.sort_values(
                by=["clause_type", "segment_rank", "document_id", "segment_number"],
                na_position="last",
            ).reset_index(drop=True)
        else:
            segment_rankings = selected_segments_df.copy()
            segment_rankings["segment_generosity_score"] = None
            segment_rankings["segment_rank"] = None
            segment_rankings["segment_percentile"] = None
            segment_rankings["ranking_note"] = "no-ranked-segments"
            segment_rankings["clause_type_segment_count"] = None
            segment_rankings["clause_type_document_count"] = None

        clause_type_document_rankings = self._build_clause_type_document_rankings(segment_rankings)
        document_rankings = self._build_document_rankings(clause_type_document_rankings)

        segment_csv = self.output_dir / "segment_generosity_rankings.csv"
        clause_doc_csv = self.output_dir / "clause_type_document_rankings.csv"
        doc_csv = self.output_dir / "document_composite_rankings.csv"

        segment_rankings[
            [
                "segment_id",
                "document_id",
                "segment_number",
                "clause_type",
                "segment_generosity_score",
                "segment_rank",
                "segment_percentile",
                "clause_type_segment_count",
                "clause_type_document_count",
                "ranking_note",
                "text_char_count",
                "is_truncated",
                "segment_path",
            ]
        ].to_csv(segment_csv, index=False)
        clause_type_document_rankings.to_csv(clause_doc_csv, index=False)
        document_rankings.to_csv(doc_csv, index=False)

        doc_summary_map: dict[str, dict[str, Any]] = {}
        if not document_rankings.empty:
            for row in document_rankings.to_dict(orient="records"):
                doc_id = str(row.get("document_id", "")).strip()
                if not doc_id:
                    continue
                payload = {
                    "document_id": doc_id,
                    "segment_count": _safe_int(row.get("segment_count", 0)),
                    "ranked_segment_count": _safe_int(row.get("ranked_segment_count", 0)),
                    "clause_type_count_used": _safe_int(row.get("clause_type_count_used", 0)),
                    "composite_score": row.get("composite_score"),
                    "document_rank": _safe_int(row.get("document_rank", 0)) or None,
                    "mean_segment_generosity_score": row.get("mean_segment_generosity_score"),
                    "mean_segment_rank": row.get("mean_segment_rank"),
                }
                doc_summary_map[doc_id] = payload

                doc_out_dir = self.output_dir / doc_id
                doc_out_dir.mkdir(parents=True, exist_ok=True)
                (doc_out_dir / "document_summary.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        clause_rankings_by_type: dict[str, list[dict[str, Any]]] = {}
        if not clause_type_document_rankings.empty:
            for clause_type, group in clause_type_document_rankings.groupby("clause_type", sort=True):
                clause_rankings_by_type[str(clause_type)] = group.to_dict(orient="records")

        summary = {
            "model": self.model,
            "input_dir": str(self.input_dir),
            "classification_dir": str(self.classification_dir),
            "output_dir": str(self.output_dir),
            "documents_processed": int(segment_rankings["document_id"].nunique()),
            "segments_processed": int(len(segment_rankings)),
            "segments_ranked": int(segment_rankings["segment_generosity_score"].notna().sum()),
            "documents_ranked": int(document_rankings["composite_score"].notna().sum()) if not document_rankings.empty else 0,
            "clause_types_ranked": int(
                sum(
                    1
                    for clause_type in top_clause_types
                    if int(top_clause_type_document_counts.get(clause_type, 0)) >= MIN_DOCS_PER_CLAUSE_FOR_RANKING
                )
            ),
            "n_rounds": self.n_rounds,
            "matches_per_round": self.matches_per_round,
            "n_parallels": self.n_parallels,
            "top_clause_type_count_target": int(self.top_clause_types),
            "top_clause_types_selected": top_clause_types,
            "top_clause_type_document_counts": top_clause_type_document_counts,
            "top_clause_type_segment_counts": top_clause_type_segment_counts,
            "excluded_procedural_clause_types": EXCLUDED_PROCEDURAL_CLAUSE_TYPES,
            "segment_attribute_prompt": SEGMENT_ATTRIBUTE_PROMPT,
            "min_documents_per_clause_for_ranking": MIN_DOCS_PER_CLAUSE_FOR_RANKING,
            "clause_type_score_method": "mean(segment_percentile within clause_type)",
            "document_composite_method": "mean(clause_type_score across selected top clause types)",
            "clause_types_skipped_for_low_document_count": clause_types_skipped_low_docs,
            "clause_type_document_rankings": clause_rankings_by_type,
            "document_composite_rankings": doc_summary_map,
            "outputs": {
                "segment_rankings_csv": str(segment_csv),
                "clause_type_document_rankings_csv": str(clause_doc_csv),
                "document_rankings_csv": str(doc_csv),
            },
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary


def main() -> None:
    """CLI entrypoint for Gabriel generosity scoring."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank segment-level generosity with GAB within each clause type and aggregate to "
            "document-level composite scores."
        )
    )
    parser.add_argument("--cache-dir", type=str, default=_default_cache_dir())
    parser.add_argument("--input-dir", type=Path, default=_default_input_dir())
    parser.add_argument("--classification-dir", type=Path, default=_default_classification_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--document-id", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--top-clause-types", type=int, default=TOP_CLAUSE_TYPES_DEFAULT)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--matches-per-round", type=int, default=3)
    parser.add_argument("--n-parallels", type=int, default=50)
    parser.add_argument("--max-chars-per-segment", type=int, default=8000)
    parser.set_defaults(reset_files=False)
    parser.add_argument(
        "--reset-files",
        dest="reset_files",
        action="store_true",
        help="Reset Gabriel run files and recompute rankings instead of reusing cached files.",
    )
    parser.add_argument(
        "--no-reset-files",
        dest="reset_files",
        action="store_false",
        help="Reuse cached Gabriel run files (default).",
    )
    args = parser.parse_args()

    runner = GenerosityGabRunner(
        cache_dir=args.cache_dir,
        input_dir=args.input_dir,
        classification_dir=args.classification_dir,
        output_dir=args.output_dir,
        model=args.model,
        n_rounds=args.n_rounds,
        matches_per_round=args.matches_per_round,
        n_parallels=args.n_parallels,
        top_clause_types=args.top_clause_types,
        reset_files=bool(args.reset_files),
        max_chars_per_segment=args.max_chars_per_segment,
    )
    summary = asyncio.run(
        runner.run(
            sample_size=args.sample_size,
            seed=args.seed,
            document_id=_normalize_document_id(args.document_id),
            max_segments=args.max_segments,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
