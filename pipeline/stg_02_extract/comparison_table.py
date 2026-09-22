"""Compare cached ContractEval, Harness, and Runner2 spans against CUAD gold spans.

All methods use one-to-one span matching requiring full gold-span containment.
Aggregate clause types with complete test coverage in the compared methods per model.
Runner2 is included when it has complete clause outputs for that model.
Run with --clause-type joint_ip_ownership to select a single provision.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html import escape
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.stg_02_extract.span_grounding import GroundedAnswer, ground_answer
from pipeline.stg_02_extract.cuad_targets import MIN_5_CLAUSES
from pipeline.utils.paths import PROJECT_ROOT, default_cache_dir, path_safe_model_name
from references.cuad.compare_extractions import (
    CATEGORY_BY_CLAUSE_TYPE,
    DEFAULT_CUAD_JSON,
    build_document_id_resolver,
    greedy_match_pairs,
    jaccard,
    Totals,
)

METHODS = ("ContractEval", "Harness", "Runner2")
REQUIRED_METHODS = ("ContractEval", "Harness")
METRICS = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
           ("jaccard_mean", "Jaccard (TP)"), ("corpus_retained", "Corpus retained"))
Case = tuple[str, str]  # CUAD title, clause type
TEST_JSON = PROJECT_ROOT / "references/cuad/test.json"


def gold_containment(gold: tuple[int, int], prediction: tuple[int, int]) -> float:
    """Score 1 only when one extraction contains the entire nonempty gold span."""
    return float(prediction[0] <= gold[0] < gold[1] <= prediction[1])


@dataclass
class GoldCase:
    context: str
    spans: list[tuple[int, int]]


@dataclass
class Run:
    model_key: str
    method: str
    model_name: str
    predictions: dict[Case, GroundedAnswer] = field(default_factory=dict)
    available_count: int | None = None
    failures: dict[Case, str] = field(default_factory=dict)


def load_gold(path: Path) -> dict[Case, GoldCase]:
    gold = {}
    clauses = {category: clause for clause, category in CATEGORY_BY_CLAUSE_TYPE.items()}
    for entry in json.loads(path.read_text(encoding="utf-8"))["data"]:
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                prefix = title + "__"
                if not qa["id"].startswith(prefix):
                    raise ValueError(f"Invalid CUAD question ID: {qa['id']}")
                clause = clauses.get(qa["id"][len(prefix):])
                if clause is not None:
                    context = paragraph["context"]
                    spans = []
                    for answer in qa["answers"]:
                        start = answer["answer_start"]
                        end = start + len(answer["text"])
                        if not 0 <= start < end <= len(context) or context[start:end] != answer["text"]:
                            raise ValueError(f"Gold span does not match CUAD context: {qa['id']}")
                        spans.append((start, end))
                    if (title, clause) in gold:
                        raise ValueError(f"Multiple contexts for CUAD question: {qa['id']}")
                    gold[(title, clause)] = GoldCase(context, spans)
    return gold


def paired_cases(manifests, test_titles: set[str]) -> dict[str, set[Case]]:
    """Intersect complete clauses across available methods, requiring both baselines."""
    models = {}
    for run, cases in manifests:
        methods = models.setdefault(run.model_key, {})
        if run.method in methods:
            raise ValueError(f"Duplicate run for {run.model_key}/{run.method}")
        methods[run.method] = set(cases)
    eligible = {}
    for model, methods in models.items():
        if not all(method in methods for method in REQUIRED_METHODS):
            continue
        common = set.intersection(*methods.values())
        cases = set()
        for clause in {clause for _, clause in common}:
            required = {(title, clause) for title in test_titles}
            if required and required <= common:
                cases.update(required)
        if cases:
            eligible[model] = cases
    return eligible


def discover_runs(root: Path, gold: dict[Case, GoldCase], clause_type=None, model_names=None,
                  test_titles: set[str] | None = None, *, clause_types: set[str] | None = None,
                  runner2_root: Path | None = None):
    if test_titles is None:
        test_titles = {title for title, _ in gold}
    resolve = build_document_id_resolver({title: None for title, _ in gold})
    selected = {path_safe_model_name(name) for name in model_names} if model_names else None
    if runner2_root is None:
        runner2_root = root.parent.parent / "stg_02_extract_runner2" / root.name
    manifests, unmatched = [], []
    for method in METHODS:
        parent = {"ContractEval": root / "contracteval", "Harness": root,
                  "Runner2": runner2_root}[method]
        extension = "json" if method == "ContractEval" else "jsonl"
        if not parent.exists():
            continue
        for model_dir in sorted(parent.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith(".") or model_dir.name == "contracteval":
                continue
            if selected is not None and model_dir.name not in selected:
                continue
            paths = {}
            for path in sorted(model_dir.glob(f"*/*.{extension}")):
                if path.parent.name == "metrics" or path.stem not in CATEGORY_BY_CLAUSE_TYPE:
                    continue
                if clause_type is not None and path.stem != clause_type:
                    continue
                if clause_types is not None and path.stem not in clause_types:
                    continue
                case = (resolve(path.parent.name), path.stem)
                if case not in gold:
                    unmatched.append(str(path))
                    continue
                if case in paths:
                    raise ValueError(f"Multiple files resolve to the same CUAD document/clause: {path}")
                paths[case] = path
            if method == "ContractEval":
                # Older runs recorded failures only in aggregate metrics.
                for summary_path in sorted((model_dir / "metrics").glob("*.json")):
                    clause = summary_path.stem
                    if clause not in CATEGORY_BY_CLAUSE_TYPE or (clause_type is not None and clause != clause_type):
                        continue
                    if clause_types is not None and clause not in clause_types:
                        continue
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    errors = {name: "missing cached output" for name in summary.get("missing_outputs", [])}
                    errors.update(summary.get("failures", {}))
                    for name, error in errors.items():
                        case = (resolve(name), clause)
                        if case in gold:
                            paths.setdefault(case, {
                                "status": "failed", "answer": None, "error": error,
                                "model_name": summary.get("model_name", model_dir.name),
                            })
            available_count = len(paths)
            if paths and test_titles is not None:
                # Check actual test identities, not the total number of files:
                # training outputs cannot fill gaps in test coverage.
                complete = {}
                for clause in sorted({clause for _, clause in paths}):
                    required = {(title, clause) for title in test_titles}
                    covered = required.intersection(paths)
                    if covered == required:
                        complete.update({case: paths[case] for case in required})
                    else:
                        print(f"Skipping incomplete {method} run {model_dir.name}/{clause}: "
                              f"{len(covered)}/{len(required)} test documents", file=sys.stderr)
                paths = complete
            if paths:
                manifests.append((Run(model_dir.name, method, model_dir.name, available_count=available_count), paths))
    eligible = paired_cases(manifests, test_titles)
    if selected is not None:
        missing = selected - eligible.keys()
        if missing:
            raise ValueError(f"No matching outputs with complete test coverage in the compared methods for models: {', '.join(sorted(missing))}")
    for model in sorted({run.model_key for run, _ in manifests} - eligible.keys()):
        print(f"Skipping {model}: no clause type has complete test coverage in the compared methods", file=sys.stderr)
    # Select pairs before opening outputs. Incomplete or unpaired runs must
    # neither reduce another model's population nor trigger unnecessary reads.
    manifests = [(run, paths) for run, paths in manifests if run.model_key in eligible]
    for run, paths in manifests:
        for case in sorted(eligible[run.model_key]):
            path = paths[case]
            try:
                if run.method == "ContractEval":
                    record = path if isinstance(path, dict) else json.loads(path.read_text(encoding="utf-8"))
                    if record.get("status") == "failed":
                        prediction = GroundedAnswer()
                        run.failures[case] = record.get("error") or "request failed"
                    else:
                        answer = record["answer"]
                        if not isinstance(answer, str) or not answer.strip():
                            raise ValueError("missing final answer")
                        prediction = ground_answer(answer, gold[case].context)
                    records = [record]
                else:
                    raw = path.read_text(encoding="utf-8")
                    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    prediction = GroundedAnswer()
                    for record in records:
                        start, end = record["span_start"], record["span_end"]
                        context = gold[case].context
                        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(context):
                            raise ValueError(f"invalid {run.method} span offsets")
                        if context[start:end] != record["extraction_text"]:
                            raise ValueError(f"{run.method} offsets do not match original CUAD context")
                        prediction.spans.append((start, end))
                for record in records:
                    model_name = record.get("model_name")
                    if model_name:
                        if path_safe_model_name(model_name) != run.model_key:
                            raise ValueError("model_name does not match cache directory")
                        run.model_name = model_name
                run.predictions[case] = prediction
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid extraction file {path}: {exc}") from exc
    return [run for run, _ in manifests], unmatched


def covered_characters(spans: list[tuple[int, int]]) -> int:
    """Count the union of character spans, including overlaps only once."""
    total = end = 0
    for start, stop in sorted(spans):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def evaluate(runs: list[Run], gold: dict[Case, GoldCase], *, test_titles: set[str] | None = None):
    if not runs:
        raise ValueError("No matching ContractEval, Harness, or Runner2 output files found")
    if test_titles is None:
        test_titles = {title for title, _ in gold}
    eligible = paired_cases([(run, run.predictions) for run in runs], test_titles)
    if not eligible:
        raise ValueError("No model/clause pair has complete test coverage in the compared methods")
    rows, coverage, diagnostics = {}, [], []
    for run in runs:
        if run.model_key not in eligible:
            continue
        common = eligible[run.model_key]
        totals = Totals()
        matched_scores = []
        examples = {kind: None for kind in ("TP", "FP", "TN", "FN")}
        ungrounded_count = ambiguous_count = 0
        corpus_contexts = {}
        extracted_spans = {}
        for case in sorted(common):
            predicted, gold_spans = run.predictions[case], gold[case].spans
            context = gold[case].context
            title = case[0]
            if title in corpus_contexts and corpus_contexts[title] != context:
                raise ValueError(f"Inconsistent CUAD contexts for document: {title}")
            corpus_contexts[title] = context
            extracted_spans.setdefault(title, []).extend(predicted.spans)
            pairs = greedy_match_pairs(gold_spans, predicted.spans, gold_containment, 0.0)
            tp = len(pairs)
            fp = len(predicted.spans) - tp
            fn = len(gold_spans) - tp
            totals.add(tp, fp, fn)
            similarities = [jaccard(gold_spans[g], predicted.spans[p]) for g, p, _ in pairs]
            matched_scores.extend(similarities)
            matched_golds = {g for g, _, _ in pairs}
            matched_predictions = {p for _, p, _ in pairs}
            base = {"document_title": case[0], "clause_type": case[1]}
            if pairs and examples["TP"] is None:
                g, p, coverage_score = pairs[0]
                examples["TP"] = {
                    **base, "gold_text": context[slice(*gold_spans[g])],
                    "extraction_text": context[slice(*predicted.spans[p])],
                    "gold_coverage": coverage_score,
                    "jaccard": jaccard(gold_spans[g], predicted.spans[p]),
                }
            if examples["FP"] is None:
                unmatched_prediction = next((p for p in range(len(predicted.spans))
                                             if p not in matched_predictions), None)
                if unmatched_prediction is not None:
                    examples["FP"] = {
                        **base,
                        "extraction_text": context[slice(*predicted.spans[unmatched_prediction])],
                    }
            if not gold_spans and not predicted.spans and examples["TN"] is None:
                examples["TN"] = base
            if examples["FN"] is None:
                unmatched_gold = next((g for g in range(len(gold_spans)) if g not in matched_golds), None)
                if unmatched_gold is not None:
                    examples["FN"] = {
                        **base, "gold_text": context[slice(*gold_spans[unmatched_gold])],
                        "failure": run.failures.get(case),
                    }
            ungrounded_count += len(predicted.ungrounded)
            ambiguous_count += predicted.ambiguous_passages
            diagnostics.append({
                "model_name": run.model_name, "method": run.method,
                "document_title": case[0], "clause_type": case[1],
                "gold_spans": gold_spans, "predicted_spans": predicted.spans,
                "ungrounded_passages": predicted.ungrounded,
                "ambiguous_passages": predicted.ambiguous_passages,
                "matches": [{"gold_index": g, "prediction_index": p, "gold_coverage": score,
                             "jaccard": jaccard(gold_spans[g], predicted.spans[p])}
                            for g, p, score in pairs],
                "TP": tp, "FP": fp, "FN": fn,
                "failure": run.failures.get(case),
            })
        corpus_characters = sum(len(context) for context in corpus_contexts.values())
        extracted_characters = sum(covered_characters(spans) for spans in extracted_spans.values())
        metrics = {
            "TP": totals.true_positives, "FP": totals.false_positives, "FN": totals.false_negatives,
            "precision": totals.precision, "recall": totals.recall, "f1": totals.f1,
            "jaccard_mean": sum(matched_scores) / len(matched_scores) if matched_scores else None,
            "extracted_characters": extracted_characters,
            "corpus_characters": corpus_characters,
            "corpus_retained": extracted_characters / corpus_characters if corpus_characters else None,
            "ungrounded_predictions": ungrounded_count,
            "ambiguous_passages": ambiguous_count,
            "failure_count": sum(case in run.failures for case in common),
            "examples": examples,
        }
        clause_types = sorted({clause for _, clause in common})
        row = rows.setdefault(run.model_key, {
            "model_name": run.model_name, "n_types": len(clause_types), "clause_types": clause_types,
            "case_count": len(common), "positive_count": sum(bool(gold[case].spans) for case in common),
            "gold_span_count": sum(len(gold[case].spans) for case in common),
        })
        if run.model_name != run.model_key:
            row["model_name"] = run.model_name
        row[run.method] = metrics
        available = run.available_count if run.available_count is not None else len(run.predictions)
        coverage.append({"model_name": run.model_name, "method": run.method,
                         "available": available, "evaluated": len(common),
                         "n_types": len(clause_types), "clause_types": clause_types,
                         "excluded": available - len(common),
                         "failures": metrics["failure_count"],
                         "ungrounded": ungrounded_count, "ambiguous": ambiguous_count})
    all_cases = set.union(*eligible.values())
    return {
        "rows": [rows[key] for key in sorted(rows)],
        "methods": [method for method in METHODS if any(method in row for row in rows.values())],
        "coverage": coverage, "diagnostics": diagnostics,
        "cases": sorted(all_cases),
        "positive_count": sum(bool(gold[case].spans) for case in all_cases),
        "gold_span_count": sum(len(gold[case].spans) for case in all_cases),
        "test_document_count": len(test_titles),
        "jaccard_cases": "tp",
        "aggregation": "pooled span counts over each model's complete paired clause types",
        "match_rule": "greedy one-to-one full gold-span containment (100% character coverage)",
        "extraction_rule": "only grounded spans count as extractions; ungrounded passages are diagnostic only",
        "corpus_retained_rule": "union of grounded extracted spans across evaluated clause types per document / "
                                "characters in unique evaluated CUAD documents; ungrounded text excluded",
    }


def render_html(report: dict, gold_path: Path) -> str:
    methods = report.get("methods", [method for method in METHODS
                                     if any(method in row for row in report["rows"])])
    maxima = {
        metric: max((row[method][metric] for row in report["rows"] for method in methods
                     if method in row and row[method][metric] is not None), default=None)
        for metric, _ in METRICS if metric != "corpus_retained"
    }
    body = []

    def render_examples(examples: dict) -> str:
        items = []
        for kind in ("TP", "FP", "TN", "FN"):
            example = examples.get(kind)
            if example is None:
                content = "No example in evaluated cases."
            else:
                parts = [f'<b>{escape(example["clause_type"])}</b>',
                         escape(example["document_title"])]
                if example.get("gold_text") is not None:
                    parts.append(f'<span><i>Gold:</i> {escape(example["gold_text"])}</span>')
                if example.get("extraction_text") is not None:
                    parts.append(f'<span><i>Extraction:</i> {escape(example["extraction_text"])}</span>')
                if example.get("gold_coverage") is not None:
                    parts.append(f'<span>Gold coverage: {example["gold_coverage"]:.3f}; '
                                 f'Jaccard: {example["jaccard"]:.3f}</span>')
                if example.get("failure"):
                    parts.append(f'<span>Request failure: {escape(example["failure"])}</span>')
                content = "".join(f"<span>{part}</span>" if not part.startswith("<span") else part
                                  for part in parts)
            items.append(f'<dt>{kind}</dt><dd>{content}</dd>')
        return '<details><summary>Examples</summary><dl>' + "".join(items) + '</dl></details>'

    for row in report["rows"]:
        cells = [f'<th scope="row">{escape(row["model_name"])}</th>']
        for method in methods:
            for metric, _ in METRICS:
                value = row.get(method, {}).get(metric)
                text = "—" if value is None else f"{value:.3f}"
                if metric == "corpus_retained" and value is not None:
                    text = f"{value:.2%}"
                if value is not None and metric in maxima and value == maxima[metric]:
                    text = f"<strong>{text}</strong>"
                cells.append(f"<td>{text}</td>")
            cells.append(f'<td class="examples">{render_examples(row.get(method, {}).get("examples", {}))}</td>')
        cells.append(f'<td title="{escape(", ".join(row["clause_types"]), quote=True)}">{row["n_types"]}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    subheaders = "".join(
        "".join(f'<th scope="col">{label}</th>' for _, label in METRICS) + '<th scope="col">Examples</th>'
        for _ in methods
    )
    headers = ''.join(f'<th colspan="{len(METRICS) + 1}" scope="colgroup">{escape(method)}</th>'
                      for method in methods)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUAD methodology comparison</title>
<style>
body {{font: 15px/1.5 system-ui, sans-serif; color: #202932; margin: 40px auto; max-width: 1400px; padding: 0 24px;}}
.scroll {{overflow-x: auto;}}
table {{border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin: 24px 0;}}
th, td {{padding: 12px 16px; border-bottom: 1px solid #dce2e6; text-align: right;}}
thead th {{background: #edf2f5; text-align: center;}} tbody th, td:first-child {{text-align: left;}}
strong {{font-weight: 800; color: #102e44;}}
td.examples {{text-align: left; min-width: 110px; vertical-align: top;}}
summary {{cursor: pointer; color: #174a6e;}}
dl {{width: min(560px, 70vw); margin: 10px 0 0; white-space: normal;}}
dt {{font-weight: 800; margin-top: 10px;}} dd {{margin: 2px 0 8px;}}
dd span {{display: block; margin-top: 3px; overflow-wrap: anywhere;}}
</style></head><body>
<div class="scroll"><table aria-label="Model and extraction methodology comparison">
<caption>An extraction is considered correct when it contains the entire gold span (100% character coverage).
Corpus retained is extracted characters / corpus characters, counting each document and overlapping
characters once across evaluated clause types. Ungrounded passages are excluded from all metrics.</caption>
<thead><tr><th rowspan="2" scope="col">Model</th>{headers}<th rowspan="2" scope="col">N Types</th></tr>
<tr>{subheaders}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
</body></html>
'''


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all", "min_5"), default="all",
                        help="Compare all eligible clause types (default), or the orchestrator's five technology-provision proxies.")
    parser.add_argument("--input-root", type=Path, default=default_cache_dir() / "stg_02_extract/cuad")
    parser.add_argument("--runner2-root", type=Path,
                        help="Runner2 CUAD cache directory; defaults to the sibling stg_02_extract_runner2/cuad cache.")
    parser.add_argument("--cuad-json", type=Path, default=DEFAULT_CUAD_JSON)
    parser.add_argument("--test-json", type=Path, default=TEST_JSON,
                        help="CUAD test split defining the required complete document coverage.")
    parser.add_argument("--clause-type", "--provision", choices=sorted(CATEGORY_BY_CLAUSE_TYPE),
                        help="Optionally restrict to one clause type; default aggregates all eligible types.")
    parser.add_argument("--model-name", nargs="+", action="extend")
    parser.add_argument("--jaccard-cases", choices=("tp",), default="tp",
                        help="Jaccard is calculated only over true-positive matched span pairs.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figures/cuad/comparison_table.html")
    args = parser.parse_args(argv)
    clause_types = set(MIN_5_CLAUSES) if args.target == "min_5" else None
    if clause_types is not None and args.clause_type is not None and args.clause_type not in clause_types:
        parser.error(f"--clause-type {args.clause_type} is outside --target min_5")
    gold = load_gold(args.cuad_json.expanduser())
    resolve = build_document_id_resolver({title: None for title, _ in gold})
    test_titles = set()
    for entry in json.loads(args.test_json.expanduser().read_text(encoding="utf-8"))["data"]:
        title = resolve(entry["title"])
        if title is None:
            raise ValueError(f"Test document absent from CUAD ground truth: {entry['title']}")
        test_titles.add(title)
    if not test_titles:
        raise ValueError("CUAD test split is empty")
    runs, unmatched = discover_runs(args.input_root.expanduser(), gold, args.clause_type,
                                    args.model_name, test_titles, clause_types=clause_types,
                                    runner2_root=args.runner2_root.expanduser() if args.runner2_root else None)
    if not runs:
        raise ValueError("No model/clause pair has complete test coverage in the compared methods")
    report = evaluate(runs, gold, test_titles=test_titles)
    report["target"] = args.target
    report["requested_clause_types"] = ([args.clause_type] if args.clause_type else
                                       sorted(clause_types if clause_types is not None else CATEGORY_BY_CLAUSE_TYPE))
    report["test_document_count"] = len(test_titles)
    report["test_split"] = str(args.test_json)
    report["unmatched_files"] = unmatched
    for path in unmatched:
        print(f"No matching CUAD ground truth: {path}", file=sys.stderr)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    report["ground_truth"] = str(args.cuad_json)
    output.write_text(render_html(report, args.cuad_json), encoding="utf-8")
    output.with_suffix(".spans.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({len(report['rows'])} models, {len(test_titles)} test documents per type)")
    for row in report["rows"]:
        print(f"  {row['model_name']}: {row['n_types']} types, {row['case_count']} document–clause cases")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
