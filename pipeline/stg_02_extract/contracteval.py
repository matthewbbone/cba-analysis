"""Full-document ContractEval baseline using stage-1 text and CUAD test labels.

Prompt and scoring source (MIT): https://github.com/olivialiu121/ContractEval
Copyright (c) 2025 Shuang Liu; see references/cuad/CONTRACTEVAL_LICENSE.
The original labels are deliberately not repaired to match OCR text.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.stg_02_extract import runner
from pipeline.stg_02_extract.structure_provision import load_provision, resolve_provisions_dir
from pipeline.utils.paths import PROJECT_ROOT, default_cache_dir, path_safe_model_name
from pipeline.utils.vllm_server import VLLMServer, add_endpoint_argument, openai_client_kwargs, validate_borrowed_server
from references.cuad.compare_extractions import (
    CATEGORY_BY_CLAUSE_TYPE,
    build_document_id_resolver,
)

TEST_JSON = PROJECT_ROOT / "references/cuad/test.json"
SYSTEM_PROMPT = '''You are an assistant with strong legal knowledge, supporting senior lawyers by preparing reference materials.
Given a Context and a Question, extract and return only the sentence(s) from the Context that directly address or relate to the Question. Do not rephrase or summarize in any way—respond with exact sentences from the Context relevant to the Question. If a relevant sentence contains unrelated elements such as page numbers or whitespace, include them exactly as they appear.
If no part of the Context is relevant to the Question, respond with: "No related clause."
'''
PROMPT = "Context:\n```\n{context}\n```\nQuestion:\n```\n{question}\n```\n"
EDGE_CHARS = " \n`"


def build_question(provision) -> str:
    category = CATEGORY_BY_CLAUSE_TYPE[provision.clause_type]
    return (
        f'Highlight the parts (if any) of this contract related to "{category}" '
        f'that should be reviewed by a lawyer. Details: {provision.clause_description}'
    )


def score_answer(answer: str, labels: list[str]) -> dict:
    """Mirror upstream inclusion and Evaluation.py's abstention/token rules."""
    prediction = answer.strip(EDGE_CHARS)
    abstained = "no related clause" in prediction.lower()
    covered = bool(labels) and all(label.strip(EDGE_CHARS) in prediction for label in labels)
    classification = ("TP" if covered else "FN") if labels else ("TN" if abstained else "FP")

    def tokens(text: str) -> set[str]:
        for punctuation in ".,;:":
            text = text.replace(punctuation, "")
        return set(text.lower().replace("/", " ").split(" "))

    jaccard = None
    if labels:
        gold, pred = tokens(" ".join(labels)), tokens(prediction)
        jaccard = len(gold & pred) / len(gold | pred)
    return {
        "classification": classification,
        "positive": bool(labels),
        "abstained": abstained,
        "false_abstention": bool(labels) and abstained,
        "jaccard": jaccard,
    }


def aggregate(scores: dict[str, dict]) -> dict:
    counts = {label: sum(s["classification"] == label for s in scores.values())
              for label in ("TP", "TN", "FP", "FN")}
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    positives = [s for s in scores.values() if s["positive"]]
    return {
        **counts,
        "document_ids": sorted(scores),
        "count": len(scores),
        "positive_count": len(positives),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "f2": 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0.0,
        "jaccard_mean": sum(s["jaccard"] for s in positives) / len(positives) if positives else None,
        "false_abstention_rate": sum(s["false_abstention"] for s in positives) / len(positives) if positives else None,
    }


def score_empty_extraction(labels: list[str]) -> dict:
    """Score a failed/missing answer as zero extractions, without fake text."""
    return {
        "classification": "FN" if labels else "TN",
        "positive": bool(labels), "abstained": True,
        "false_abstention": bool(labels),
        "jaccard": 0.0 if labels else None,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def temperature(value: str) -> float | None:
    if value == "default":
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 2:
        raise argparse.ArgumentTypeError("temperature must be default or a number between 0 and 2")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provision", required=True)
    parser.add_argument("--model-name", required=True)
    add_endpoint_argument(parser)
    parser.add_argument("--input-root", type=Path, default=runner.default_input_root())
    parser.add_argument("--output-root", type=Path,
                        default=default_cache_dir() / "stg_02_extract/cuad/contracteval",
                        help="ContractEval root; model/document directories are appended.")
    parser.add_argument("--ocr-model-name", default=runner.DEFAULT_OCR_MODEL_NAME)
    parser.add_argument("--document-id", "--document-ids", nargs="+", action="extend")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--device")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    parser.add_argument("--temperature", type=temperature, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--compare-runner-model")
    parser.add_argument("--runner-output-root", type=Path, default=runner.default_output_root(),
                        help="Stage-02 root containing cuad/<model>/<document>/*.jsonl.")
    args = parser.parse_args(argv)
    if args.force and args.evaluate_only:
        parser.error("--force cannot be combined with --evaluate-only")
    if args.max_tokens < 1 or args.max_num_seqs < 1 or not 1 <= args.port <= 65535:
        parser.error("invalid token budget, sequence count, or port")
    # Offline evaluation neither needs CUDA selection nor endpoint credentials.
    if not args.evaluate_only:
        runner.validate_args(args)
    elif args.concurrency < 1 or (args.sample is not None and args.sample < 1):
        parser.error("concurrency and sample must be positive")
    return args


def inference_settings(args) -> dict:
    request = {"model": args.model_name, "max_tokens": args.max_tokens}
    if args.temperature is not None:
        request["temperature"] = args.temperature
    # The proprietary reference uses top_p=.9; open models use greedy decoding.
    request["top_p"] = 0.9 if args.endpoint == "openrouter" else 1.0
    extra = {}
    if args.thinking != "default":
        if args.endpoint == "openrouter":
            extra["reasoning"] = {"enabled": args.thinking == "on"}
        else:
            extra["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    if extra:
        request["extra_body"] = extra
    parser = args.reasoning_parser
    if args.endpoint == "vllm" and parser is None:
        parser = next((p for marker, p in runner.REASONING_PARSER_MODEL_MARKERS
                       if marker in args.model_name.casefold()), None)
    return {
        "endpoint": args.endpoint, "request": request,
        "max_model_len": args.max_model_len if args.endpoint == "vllm" else None,
        "reasoning_parser": parser if args.endpoint == "vllm" else None,
        "thinking": args.thinking,
    }


def discover(args, clause_type):
    entries = json.loads(TEST_JSON.read_text(encoding="utf-8"))["data"]
    documents = {entry["title"]: entry["paragraphs"][0] for entry in entries}
    resolve = build_document_id_resolver(documents)
    selected = set(documents)
    if args.document_id:
        unknown = [name for name in args.document_id if resolve(name) is None]
        if unknown:
            raise ValueError(f"Document IDs outside CUAD test split: {unknown}")
        selected = {resolve(name) for name in args.document_id}
    jobs = runner.discover_full_texts(
        args.input_root, args.output_root, args.ocr_model_name, args.model_name,
        clause_type, source_filter="cuad",
    )
    jobs = [replace(job, output_path=args.output_root.expanduser() /
                    path_safe_model_name(args.model_name) / job.document_id / f"{clause_type}.json")
            for job in jobs if resolve(job.document_id) in selected]
    matched = {resolve(job.document_id) for job in jobs}
    missing = sorted(selected - matched)
    if args.sample is not None and args.sample < len(jobs):
        jobs = sorted(random.Random(args.seed).sample(jobs, args.sample), key=lambda j: j.document_id)
    category = CATEGORY_BY_CLAUSE_TYPE[clause_type]
    labels = {}
    for job in jobs:
        title = resolve(job.document_id)
        qa = next(qa for qa in documents[title]["qas"] if qa["id"] == f"{title}__{category}")
        labels[job.document_id] = [answer["text"] for answer in qa["answers"]]
    return jobs, labels, missing


def compare_runner(args, clause_type, records, labels):
    scores, baseline, missing, errors = {}, {}, [], {}
    root = args.runner_output_root.expanduser() / "cuad" / path_safe_model_name(args.compare_runner_model)
    for document_id, record in records.items():
        path = root / document_id / f"{clause_type}.jsonl"
        if not path.exists():
            missing.append(document_id)
            continue
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(row.get("ocr_model_name") != args.ocr_model_name for row in rows):
                raise ValueError("runner OCR model differs from selected input")
            answer = "\n".join(row["extraction_text"] for row in rows) if rows else "No related clause."
            scores[document_id] = score_answer(answer, labels[document_id])
            baseline[document_id] = record["scores"]
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors[document_id] = str(exc)
    return {"model_name": args.compare_runner_model, "contracteval": aggregate(baseline),
            "runner": aggregate(scores), "missing_runner_outputs": sorted(missing), "errors": errors}


def run(args, *, server: VLLMServer | None = None) -> dict:
    """Run a clause and return metrics, without taking ownership of a borrowed server."""
    owned_server = server is None
    if server is not None:
        validate_borrowed_server(server, args)
    provision = load_provision(args.provision, provisions_dir=resolve_provisions_dir("cuad"))
    question = build_question(provision)
    settings = inference_settings(args)
    jobs, labels, missing_inputs = discover(args, provision.clause_type)
    records, prepared, failures = {}, {}, {}
    # Validate the whole cache before spending tokens or starting a server.
    for job in jobs:
        try:
            text = job.input_path.read_text(encoding="utf-8")
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": PROMPT.format(context=text, question=question)}]
            input_hash = digest({"messages": messages, "labels": labels[job.document_id],
                                 "ocr_model_name": args.ocr_model_name, "model_name": args.model_name})
            fingerprint = digest({"input_hash": input_hash, "settings": settings})
            prepared[job.document_id] = (messages, input_hash, fingerprint)
            if job.output_path.exists() and not args.force:
                record = json.loads(job.output_path.read_text(encoding="utf-8"))
                if record.get("input_hash") != input_hash or (
                    not args.evaluate_only and record.get("fingerprint") != fingerprint
                ):
                    raise ValueError(f"Cached input/prompt/settings mismatch: {job.output_path}; use --force or another --output-root")
                if record.get("status") == "failed" or not isinstance(record.get("answer"), str) or not record["answer"].strip():
                    # Retry failures during inference; retain them in offline metrics.
                    if not args.evaluate_only:
                        continue
                    record["status"] = "failed"
                    failures[job.document_id] = record.get("error") or "Model returned no final-answer content"
                    record["scores"] = score_empty_extraction(labels[job.document_id])
                else:
                    record["scores"] = score_answer(record["answer"], labels[job.document_id])
                records[job.document_id] = record
        except OSError as exc:
            failures[job.document_id] = str(exc)
    pending = [job for job in jobs if job.document_id not in records and job.document_id not in failures]
    reused = len(records)
    if pending and not args.evaluate_only:
        from openai import OpenAI

        client = None
        try:
            serve_args = ["--generation-config", "vllm"]
            if settings["reasoning_parser"]:
                serve_args.extend(["--reasoning-parser", settings["reasoning_parser"]])
            if owned_server:
                server = VLLMServer(
                    model_name=args.model_name, endpoint=args.endpoint, port=args.port,
                    max_model_len=args.max_model_len, num_gpus=args.num_gpus, device=args.device,
                    gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
                    extra_serve_args=serve_args,
                )
                server.start()
            client = OpenAI(**openai_client_kwargs(args.endpoint, args.port))

            def process(job):
                messages, input_hash, fingerprint = prepared[job.document_id]
                record = {
                    "source": "cuad", "document_id": job.document_id,
                    "model_name": args.model_name, "ocr_model_name": args.ocr_model_name,
                    "clause_type": provision.clause_type, "question": question,
                    "input_path": str(job.input_path), "input_hash": input_hash,
                    "fingerprint": fingerprint, "labels": labels[job.document_id],
                    "answer": None, "generation": settings, "returned_model": None,
                    "usage": None, "finish_reason": None, "truncated": False,
                    "reasoning": None,
                }
                try:
                    response = client.chat.completions.create(messages=messages, **settings["request"])
                    choice = response.choices[0]
                    answer = choice.message.content
                    record.update({
                        "answer": answer, "returned_model": response.model,
                        "usage": response.usage.model_dump() if response.usage is not None else None,
                        "finish_reason": choice.finish_reason, "truncated": choice.finish_reason == "length",
                        "reasoning": getattr(choice.message, "reasoning", None) or getattr(choice.message, "reasoning_content", None),
                    })
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("Model returned no final-answer content")
                    record.update(status="completed", scores=score_answer(answer, labels[job.document_id]))
                except Exception as exc:
                    record.update(status="failed", error=str(exc),
                                  scores=score_empty_extraction(labels[job.document_id]))
                atomic_json(job.output_path, record)
                records[job.document_id] = record
                return runner.ExtractionResult(job=job, status=record["status"],
                                               extraction_count=int(record["status"] == "completed"),
                                               error=record.get("error"))

            results = runner.run_extraction_queue(pending, args.concurrency, process, not args.no_progress)
            failures.update({r.job.document_id: r.error for r in results if r.status == "failed"})
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if owned_server and server is not None:
                    server.close()
    missing_outputs = [job.document_id for job in pending] if args.evaluate_only else []
    for document_id in failures.keys() | set(missing_outputs):
        records.setdefault(document_id, {
            "status": "failed" if document_id in failures else "missing",
            "answer": None, "scores": score_empty_extraction(labels[document_id]),
            "truncated": False,
        })
    summary = {
        "model_name": args.model_name, "clause_type": provision.clause_type,
        "split": str(TEST_JSON), "metrics": aggregate({key: r["scores"] for key, r in records.items()}),
        "selected_document_ids": [job.document_id for job in jobs],
        "missing_inputs": missing_inputs, "failures": failures,
        "missing_outputs": missing_outputs,
        "failure_scoring": "Failed requests and missing outputs count as empty extractions.",
        "truncation_count": sum(r["truncated"] for r in records.values()),
        "execution": {"reused": reused, "completed": sum(
            job.document_id in records and job.document_id not in failures
            for job in pending
        ) if not args.evaluate_only else 0, "failed": len(failures)},
    }
    if args.compare_runner_model:
        summary["comparison"] = compare_runner(args, provision.clause_type, records, labels)
    path = args.output_root.expanduser() / path_safe_model_name(args.model_name) / "metrics" / f"{provision.clause_type}.json"
    atomic_json(path, summary)
    print(json.dumps(summary, indent=2))
    print(f"Metrics: {path}")
    return summary


def main(argv=None) -> int:
    summary = run(parse_args(argv))
    return int(bool(summary["failures"] or summary["missing_inputs"] or summary["missing_outputs"] or
                    summary.get("comparison", {}).get("errors")))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
