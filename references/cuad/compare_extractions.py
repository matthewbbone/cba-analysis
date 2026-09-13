"""Score stage-02 CUAD extractions against CUAD's own ground-truth spans.

For each clause type, this reads the SQuAD-style ground truth in
``references/cuad/CUADv1.json`` and the corresponding
``cache/stg_02_extract/cuad/<model>/<document_id>/<clause_type>.jsonl``
extraction files, and reports precision/recall/F1 under two character-span
overlap criteria:

* any overlap: a predicted span counts if it shares at least one character
  with a gold span.
* jaccard > 0.5: a predicted span counts only if its overlap-to-union ratio
  with a gold span exceeds one half.

Matching is greedy 1:1 per document per clause type (each gold span can be
claimed by at most one prediction and vice versa), so a single long
prediction spanning several short gold clauses -- or several predictions
landing on one gold clause -- cannot inflate the true-positive count.

Run from anywhere in the repository::

    python references/cuad/compare_extractions.py \\
        --model-name RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --clause-type governing_law

Omit ``--clause-type`` to evaluate every clause type that has at least one
output file under the model's stage-02 directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.utils.paths import PROJECT_ROOT, default_cache_dir, path_safe_model_name

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False

load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_SOURCE = "cuad"
DEFAULT_CUAD_JSON = PROJECT_ROOT / "references" / "cuad" / "CUADv1.json"
JACCARD_THRESHOLD = 0.5

# Maps our snake_case provision names (pipeline/provisions/cuad/*.yaml) to the
# exact category string CUAD uses in each qas id ("<title>__<Category
# Name>"). These strings differ from both our names and the CUAD README's
# prose in several places (casing, pluralization, punctuation), so this is a
# literal lookup table rather than a normalization function.
CATEGORY_BY_CLAUSE_TYPE: dict[str, str] = {
    "document_name": "Document Name",
    "parties": "Parties",
    "agreement_date": "Agreement Date",
    "effective_date": "Effective Date",
    "expiration_date": "Expiration Date",
    "renewal_term": "Renewal Term",
    "notice_to_terminate_renewal": "Notice Period To Terminate Renewal",
    "governing_law": "Governing Law",
    "most_favored_nation": "Most Favored Nation",
    "non_compete": "Non-Compete",
    "exclusivity": "Exclusivity",
    "no_solicit_of_customers": "No-Solicit Of Customers",
    "competitive_restriction_exception": "Competitive Restriction Exception",
    "no_solicit_of_employees": "No-Solicit Of Employees",
    "non_disparagement": "Non-Disparagement",
    "termination_for_convenience": "Termination For Convenience",
    "rofr_rofo_rofn": "Rofr/Rofo/Rofn",
    "change_of_control": "Change Of Control",
    "anti_assignment": "Anti-Assignment",
    "revenue_profit_sharing": "Revenue/Profit Sharing",
    "price_restriction": "Price Restrictions",
    "minimum_commitment": "Minimum Commitment",
    "volume_restriction": "Volume Restriction",
    "ip_ownership_assignment": "Ip Ownership Assignment",
    "joint_ip_ownership": "Joint Ip Ownership",
    "license_grant": "License Grant",
    "non_transferable_license": "Non-Transferable License",
    "affiliate_ip_license_licensor": "Affiliate License-Licensor",
    "affiliate_ip_license_licensee": "Affiliate License-Licensee",
    "unlimited_all_you_can_eat_license": "Unlimited/All-You-Can-Eat-License",
    "irrevocable_or_perpetual_license": "Irrevocable Or Perpetual License",
    "source_code_escrow": "Source Code Escrow",
    "post_termination_services": "Post-Termination Services",
    "audit_rights": "Audit Rights",
    "uncapped_liability": "Uncapped Liability",
    "cap_on_liability": "Cap On Liability",
    "liquidated_damages": "Liquidated Damages",
    "warranty_duration": "Warranty Duration",
    "insurance": "Insurance",
    "covenant_not_to_sue": "Covenant Not To Sue",
    "third_party_beneficiary": "Third Party Beneficiary",
}

# The two CUAD entries whose "title" does not match any document_id even
# after stripping whitespace (a genuine naming inconsistency in the source
# dataset between its PDF filenames and its SQuAD-format titles). Keyed by
# document_id, valued by the CUAD title it corresponds to.
DOCUMENT_ID_TITLE_OVERRIDES: dict[str, str] = {
    "NETGEAR,INC_04_21_2003-EX-10.16-AMENDMENT TO THE DISTRIBUTOR AGREEMENT "
    "BETWEEN INGRAM MICRO AND NETGEAR-": (
        "NETGEAR,INC_04_21_2003-EX-10.16-AMENDMENT TO THE DISTRIBUTOR "
        "AGREEMENT BETWEEN INGRAM MICRO AND NETGEAR"
    ),
    "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_"
    "Development Agreement_Option Agreement": (
        "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_"
        "Development Agreement"
    ),
}

Span = tuple[int, int]


@dataclass(frozen=True)
class GoldDocument:
    title: str
    spans_by_category: dict[str, list[Span]]


def load_gold_documents(cuad_json_path: Path) -> dict[str, GoldDocument]:
    """Load CUADv1.json into ``title -> GoldDocument``."""

    payload = json.loads(cuad_json_path.read_text(encoding="utf-8"))
    documents: dict[str, GoldDocument] = {}
    for entry in payload["data"]:
        title = entry["title"]
        paragraph = entry["paragraphs"][0]
        prefix = f"{title}__"
        spans_by_category: dict[str, list[Span]] = {}
        for qa in paragraph["qas"]:
            qa_id = qa["id"]
            if not qa_id.startswith(prefix):
                raise ValueError(
                    f"unexpected qas id {qa_id!r} for title {title!r}"
                )
            category = qa_id[len(prefix):]
            spans_by_category[category] = [
                (answer["answer_start"], answer["answer_start"] + len(answer["text"]))
                for answer in qa["answers"]
            ]
        documents[title] = GoldDocument(title=title, spans_by_category=spans_by_category)
    return documents


def build_document_id_resolver(
    gold_documents: dict[str, GoldDocument],
) -> Callable[[str], str | None]:
    """Return a function mapping a stage-02 document_id to its CUAD title."""

    titles_by_stripped = {title.strip(): title for title in gold_documents}

    def resolve(document_id: str) -> str | None:
        override = DOCUMENT_ID_TITLE_OVERRIDES.get(document_id)
        key = (override if override is not None else document_id).strip()
        return titles_by_stripped.get(key)

    return resolve


def gold_spans_for(document: GoldDocument, clause_type: str) -> list[Span]:
    category = CATEGORY_BY_CLAUSE_TYPE[clause_type]
    return document.spans_by_category.get(category, [])


def discover_clause_types(model_dir: Path) -> list[str]:
    clause_types: set[str] = set()
    for document_dir in model_dir.iterdir():
        if not document_dir.is_dir():
            continue
        for jsonl_path in document_dir.glob("*.jsonl"):
            clause_types.add(jsonl_path.stem)
    return sorted(clause_types)


def load_predicted_spans(model_dir: Path, clause_type: str) -> dict[str, list[Span]]:
    """Spans per document_id for one clause type.

    A document_id is present only when its ``<clause_type>.jsonl`` file
    exists -- a missing file means stage 2 has not processed that document
    yet (excluded from evaluation), while an existing, empty file means zero
    predicted spans (included, and counts fully against recall).
    """

    predictions: dict[str, list[Span]] = {}
    for document_dir in sorted(model_dir.iterdir()):
        if not document_dir.is_dir():
            continue
        jsonl_path = document_dir / f"{clause_type}.jsonl"
        if not jsonl_path.exists():
            continue
        spans: list[Span] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            spans.append((record["span_start"], record["span_end"]))
        predictions[document_dir.name] = spans
    return predictions


def interval_overlap(a: Span, b: Span) -> float:
    """Overlapping character count between two ``[start, end)`` intervals."""

    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def jaccard(a: Span, b: Span) -> float:
    overlap = interval_overlap(a, b)
    if overlap <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union > 0 else 0.0


def greedy_match(
    golds: Sequence[Span],
    preds: Sequence[Span],
    score_fn: Callable[[Span, Span], float],
    min_score: float,
) -> tuple[int, int, int]:
    """Greedy 1:1 matching by descending score; returns (tp, fp, fn).

    Each gold span may be claimed by at most one prediction and vice versa,
    so one prediction spanning several gold spans -- or several predictions
    landing on one gold span -- cannot inflate the true-positive count.
    """

    candidates: list[tuple[float, int, int]] = []
    for gold_index, gold in enumerate(golds):
        for pred_index, pred in enumerate(preds):
            score = score_fn(gold, pred)
            if score > min_score:
                candidates.append((score, gold_index, pred_index))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    matched_golds: set[int] = set()
    matched_preds: set[int] = set()
    true_positives = 0
    for _, gold_index, pred_index in candidates:
        if gold_index in matched_golds or pred_index in matched_preds:
            continue
        matched_golds.add(gold_index)
        matched_preds.add(pred_index)
        true_positives += 1

    false_positives = len(preds) - len(matched_preds)
    false_negatives = len(golds) - len(matched_golds)
    return true_positives, false_positives, false_negatives


@dataclass
class Totals:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def add(self, true_positives: int, false_positives: int, false_negatives: int) -> None:
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        denominator = precision + recall
        return 2 * precision * recall / denominator if denominator else 0.0


@dataclass
class ClauseTypeResult:
    clause_type: str
    documents_evaluated: int
    documents_total: int
    gold_span_count: int
    predicted_span_count: int
    any_overlap: Totals = field(default_factory=Totals)
    jaccard_over_half: Totals = field(default_factory=Totals)


def evaluate_clause_type(
    clause_type: str,
    gold_documents: dict[str, GoldDocument],
    resolve_title: Callable[[str], str | None],
    predictions: dict[str, list[Span]],
) -> ClauseTypeResult:
    any_totals = Totals()
    jaccard_totals = Totals()
    gold_span_count = 0
    predicted_span_count = 0
    documents_evaluated = 0

    for document_id, pred_spans in predictions.items():
        title = resolve_title(document_id)
        if title is None:
            continue
        gold = gold_spans_for(gold_documents[title], clause_type)

        documents_evaluated += 1
        gold_span_count += len(gold)
        predicted_span_count += len(pred_spans)
        any_totals.add(*greedy_match(gold, pred_spans, interval_overlap, 0))
        jaccard_totals.add(*greedy_match(gold, pred_spans, jaccard, JACCARD_THRESHOLD))

    return ClauseTypeResult(
        clause_type=clause_type,
        documents_evaluated=documents_evaluated,
        documents_total=len(gold_documents),
        gold_span_count=gold_span_count,
        predicted_span_count=predicted_span_count,
        any_overlap=any_totals,
        jaccard_over_half=jaccard_totals,
    )


def combined_result(results: Sequence[ClauseTypeResult]) -> ClauseTypeResult:
    combined = ClauseTypeResult(
        clause_type="combined",
        documents_evaluated=0,
        documents_total=results[0].documents_total if results else 0,
        gold_span_count=sum(result.gold_span_count for result in results),
        predicted_span_count=sum(result.predicted_span_count for result in results),
    )
    for result in results:
        combined.any_overlap.add(
            result.any_overlap.true_positives,
            result.any_overlap.false_positives,
            result.any_overlap.false_negatives,
        )
        combined.jaccard_over_half.add(
            result.jaccard_over_half.true_positives,
            result.jaccard_over_half.false_positives,
            result.jaccard_over_half.false_negatives,
        )
    return combined


def print_results(results: Sequence[ClauseTypeResult]) -> None:
    rows = list(results)
    if len(rows) > 1:
        rows = [*rows, combined_result(results)]

    name_width = max(len("clause_type"), *(len(row.clause_type) for row in rows))
    header = (
        f"{'clause_type':<{name_width}}  {'docs':>9}  {'gold':>5}  {'pred':>5}  "
        f"{'anyP':>6} {'anyR':>6} {'anyF1':>6}  "
        f"{'jacP':>6} {'jacR':>6} {'jacF1':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        docs = (
            f"{row.documents_evaluated}/{row.documents_total}"
            if row.clause_type != "combined"
            else ""
        )
        print(
            f"{row.clause_type:<{name_width}}  {docs:>9}  {row.gold_span_count:>5}  "
            f"{row.predicted_span_count:>5}  "
            f"{row.any_overlap.precision:>6.3f} {row.any_overlap.recall:>6.3f} "
            f"{row.any_overlap.f1:>6.3f}  "
            f"{row.jaccard_over_half.precision:>6.3f} {row.jaccard_over_half.recall:>6.3f} "
            f"{row.jaccard_over_half.f1:>6.3f}"
        )


def write_csv(path: Path, results: Sequence[ClauseTypeResult]) -> None:
    rows = list(results)
    if len(rows) > 1:
        rows = [*rows, combined_result(results)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "clause_type",
                "documents_evaluated",
                "documents_total",
                "gold_span_count",
                "predicted_span_count",
                "any_overlap_precision",
                "any_overlap_recall",
                "any_overlap_f1",
                "jaccard_gt_0.5_precision",
                "jaccard_gt_0.5_recall",
                "jaccard_gt_0.5_f1",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.clause_type,
                    row.documents_evaluated,
                    row.documents_total,
                    row.gold_span_count,
                    row.predicted_span_count,
                    f"{row.any_overlap.precision:.4f}",
                    f"{row.any_overlap.recall:.4f}",
                    f"{row.any_overlap.f1:.4f}",
                    f"{row.jaccard_over_half.precision:.4f}",
                    f"{row.jaccard_over_half.recall:.4f}",
                    f"{row.jaccard_over_half.f1:.4f}",
                ]
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"stage-02 source directory name (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="extraction model name, as passed to stg_02_extract's --model-name",
    )
    parser.add_argument(
        "--clause-type",
        dest="clause_types",
        action="append",
        metavar="NAME",
        help=(
            "clause type to evaluate; may be repeated. Default: every "
            "clause type with at least one output file under the model's "
            "stage-02 directory"
        ),
    )
    parser.add_argument(
        "--cuad-json",
        type=Path,
        default=DEFAULT_CUAD_JSON,
        help=f"path to CUADv1.json (default: {DEFAULT_CUAD_JSON})",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="stage-02 output root (default: CACHE_DIR/stg_02_extract)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="also write the results table to this CSV path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root or (default_cache_dir() / "stg_02_extract")
    model_dir = input_root / args.source / path_safe_model_name(args.model_name)
    if not model_dir.is_dir():
        raise ValueError(f"no stage-02 output directory found at {model_dir}")

    gold_documents = load_gold_documents(args.cuad_json)
    resolve_title = build_document_id_resolver(gold_documents)

    clause_types = sorted(set(args.clause_types)) if args.clause_types else discover_clause_types(model_dir)
    if not clause_types:
        raise ValueError(f"no clause-type JSONL files found under {model_dir}")
    unknown = sorted(set(clause_types) - set(CATEGORY_BY_CLAUSE_TYPE))
    if unknown:
        raise ValueError(f"unknown CUAD clause type(s): {', '.join(unknown)}")

    results = [
        evaluate_clause_type(
            clause_type,
            gold_documents,
            resolve_title,
            load_predicted_spans(model_dir, clause_type),
        )
        for clause_type in clause_types
    ]

    print_results(results)
    if args.output_csv:
        write_csv(args.output_csv, results)
        print(f"\nWrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
