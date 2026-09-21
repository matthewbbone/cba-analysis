"""Run selected CUAD test clauses sequentially, sharing one vLLM server per model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.stg_02_extract import contracteval, runner
from pipeline.stg_02_extract.cuad_targets import MIN_5_CLAUSES
from pipeline.stg_02_extract.structure_provision import load_provision, resolve_provisions_dir
from pipeline.utils.generation import harness_request_defaults
from pipeline.utils.gpu import validate_cuda_device_selection
from pipeline.utils.vllm_server import VLLMServer
from references.cuad.compare_extractions import CATEGORY_BY_CLAUSE_TYPE, build_document_id_resolver, load_gold_documents


MODELS = (
    "Qwen/Qwen3.8-27B",
    "RedHatAI/gemma-4-31B-it-FP8-dynamic",
    "google/gemma-4-31b-it",
    "Qwen/Qwen3.8-27B-FP8",
    "google/gemma-3-12b-it",
)
METHODS = ("ContractEval", "Harness")
TEST_DOCUMENT_COUNT = 102
CLAUSE_COUNT = 41


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all", "min_5"), default="all",
                        help="Clause selection: all 41 types (default), or the five technology-provision proxies.")
    parser.add_argument("--model-name", action="append", metavar="MODEL",
                        help="Run only this model; repeat to select more than one. Defaults to the five preset models.")
    parser.add_argument("--method", choices=("both", "runner", "contracteval"), default="both",
                        help="Extraction method: both (default), runner.py only (Harness), or ContractEval only.")
    parser.add_argument("--input-root", type=Path, default=runner.default_input_root())
    parser.add_argument("--output-root", type=Path, default=runner.default_output_root(),
                        help="Stage-02 root for both methods and the orchestration report.")
    parser.add_argument("--ocr-model-name", default=runner.DEFAULT_OCR_MODEL_NAME)
    parser.add_argument("--device", default="0", help="One physical GPU ID; accepts 0 or cuda:0.")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.model_name and any(not model.strip() for model in args.model_name):
        parser.error("model-name must not be empty")
    try:
        args.device = validate_cuda_device_selection(args.device.removeprefix("cuda:"), 1)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.port <= 65535 or args.max_model_len < 1 or args.max_num_seqs < 1:
        parser.error("port, context length, and sequence count must be valid positive values")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be greater than zero and at most one")
    args.input_root = args.input_root.expanduser()
    args.output_root = args.output_root.expanduser()
    return args


def preflight(args) -> tuple[list[str], list[str]]:
    """Validate the selected provisions and complete test inputs before using a GPU."""
    clauses = sorted(CATEGORY_BY_CLAUSE_TYPE)
    if len(clauses) != CLAUSE_COUNT:
        raise ValueError(f"Expected {CLAUSE_COUNT} CUAD clause types, found {len(clauses)}")
    if args.target == "min_5":
        clauses = sorted(MIN_5_CLAUSES)
    for clause in clauses:
        provision = load_provision(clause, provisions_dir=resolve_provisions_dir("cuad"))
        if provision.clause_type != clause:
            raise ValueError(f"Provision {clause} declares clause type {provision.clause_type}")
    documents = load_gold_documents(contracteval.TEST_JSON)
    if len(documents) != TEST_DOCUMENT_COUNT:
        raise ValueError(f"Expected {TEST_DOCUMENT_COUNT} test contracts, found {len(documents)}")
    resolve = build_document_id_resolver(documents)
    matched, errors = {}, []
    jobs = runner.discover_full_texts(args.input_root, args.output_root, args.ocr_model_name,
                                      MODELS[0], clauses[0], source_filter="cuad")
    for job in jobs:
        title = resolve(job.document_id)
        if title is None:
            continue
        if title in matched:
            errors.append(f"Multiple cached document IDs resolve to {title}")
            continue
        matched[title] = job.document_id
        try:
            job.input_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"Unreadable input {job.input_path}: {exc}")
    errors.extend(f"Missing OCR input: {title}" for title in sorted(documents.keys() - matched.keys()))
    for title, document in documents.items():
        absent = {CATEGORY_BY_CLAUSE_TYPE[clause] for clause in clauses} - document.spans_by_category.keys()
        if absent:
            errors.append(f"Missing CUAD labels for {title}: {', '.join(sorted(absent))}")
    if errors:
        raise ValueError("CUAD preflight failed:\n" + "\n".join(errors))
    return clauses, sorted(matched.values())


def load_generation_defaults(model_name: str) -> dict:
    # Use the installed vLLM loader, including its fallback for models without
    # generation_config.json. This loads configuration, never model weights.
    from vllm.transformers_utils.config import try_get_generation_config

    config = try_get_generation_config(model_name, trust_remote_code=True)
    return harness_request_defaults(config.to_diff_dict() if config is not None else {})


def clause_args(args, model: str, clause: str, method: str, document_ids: list[str]):
    argv = [
        "--provision", clause, "--model-name", model, "--endpoint", "vllm",
        "--input-root", str(args.input_root), "--ocr-model-name", args.ocr_model_name,
        "--device", args.device, "--num-gpus", "1", "--port", str(args.port),
        "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
        "--concurrency", "1", "--document-id", *document_ids,
    ]
    output_root = args.output_root
    module = runner
    if method == "ContractEval":
        module = contracteval
        output_root = output_root / "cuad" / "contracteval"
    else:
        argv.append("--cuad_test")
    argv.extend(["--output-root", str(output_root)])
    if args.gpu_memory_utilization is not None:
        argv.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.force:
        argv.append("--force")
    if args.no_progress:
        argv.append("--no-progress")
    return module.parse_args(argv)


def execute_job(args, task, document_ids, server, request_defaults) -> dict:
    selected = clause_args(args, task["model_name"], task["clause_type"], task["method"], document_ids)
    if task["method"] == "ContractEval":
        summary = contracteval.run(selected, server=server)
        errors = dict(summary["failures"])
        for name in summary["missing_inputs"] + summary["missing_outputs"]:
            errors[name] = "Missing input or output"
        counts = summary["execution"]
        evaluated = summary["metrics"]["count"]
    else:
        selected.harness_request_defaults = request_defaults
        results = runner.run(selected, server=server)
        errors = {r.job.document_id: r.error for r in results if r.status == "failed"}
        for name in set(document_ids) - {r.job.document_id for r in results}:
            errors[name] = "Missing input or result"
        counts = {
            "completed": sum(r.status == "completed" for r in results),
            "reused": sum(r.status == "skipped" for r in results),
            "failed": sum(r.status == "failed" for r in results),
        }
        evaluated = len(results)
    if evaluated != len(document_ids):
        errors["coverage"] = f"Expected {len(document_ids)} documents, got {evaluated}"
    return {"status": "failed" if errors else "completed", "counts": counts, "errors": errors}


def main(argv=None) -> int:
    args = parse_args(argv)
    clauses, document_ids = preflight(args)
    models = args.model_name or list(MODELS)
    methods = METHODS if args.method == "both" else (
        "Harness" if args.method == "runner" else "ContractEval",
    )
    tasks = [{"model_name": model, "clause_type": clause, "method": method,
              "status": "pending", "document_count": len(document_ids)}
             for model in models for clause in clauses for method in methods]
    print(f"{len(models)} models, {len(clauses)} clauses, {len(document_ids)} test documents; "
          f"{len(tasks)} sequential jobs on GPU {args.device}")
    if args.dry_run:
        for index, task in enumerate(tasks, 1):
            print(f"{index:03}: {task['model_name']} / {task['clause_type']} / {task['method']}")
        return 0

    report = {"started_at": now(), "status": "running", "document_ids": document_ids,
              "settings": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()}, "models": [], "jobs": tasks}
    report_path = args.output_root / "cuad" / "orchestration" / "latest.json"

    def save():
        report["updated_at"] = now()
        contracteval.atomic_json(report_path, report)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        save()
        for model in models:
            model_tasks = [task for task in tasks if task["model_name"] == model]
            first_args = clause_args(args, model, clauses[0], "ContractEval", document_ids)
            settings = contracteval.inference_settings(first_args)
            serve_args = ["--generation-config", "vllm"]
            if settings["reasoning_parser"]:
                serve_args.extend(["--reasoning-parser", settings["reasoning_parser"]])
            serving = dict(model_name=model, endpoint="vllm", device=args.device, num_gpus=1,
                           port=args.port, max_model_len=args.max_model_len,
                           gpu_memory_utilization=args.gpu_memory_utilization,
                           max_num_seqs=args.max_num_seqs, extra_serve_args=serve_args)
            model_report = {"model_name": model, "status": "starting", "serving": serving}
            report["models"].append(model_report)
            save()
            server = None
            try:
                request_defaults = None
                if "Harness" in methods:
                    request_defaults = load_generation_defaults(model)
                    model_report["harness_request_defaults"] = request_defaults
                server = VLLMServer(**serving)
                server.start()
                model_report["status"] = "running"
                save()
                for task in model_tasks:
                    if server.server is None or server.server.poll() is not None:
                        raise RuntimeError("vLLM server exited; remaining jobs will not be attempted")
                    task.update(status="running", started_at=now())
                    save()
                    print(f"Running {model} / {task['clause_type']} / {task['method']}")
                    try:
                        task.update(execute_job(args, task, document_ids, server, request_defaults))
                    except Exception as exc:
                        task.update(status="failed", error=str(exc), error_type=type(exc).__name__)
                    task["finished_at"] = now()
                    save()
                model_report["status"] = "failed" if any(t["status"] == "failed" for t in model_tasks) else "completed"
            except Exception as exc:
                model_report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
                for task in model_tasks:
                    if task["status"] == "pending":
                        task.update(status="unavailable", error=str(exc))
                save()
            finally:
                if server is not None:
                    try:
                        server.close()
                    except Exception as exc:
                        model_report.update(status="failed", cleanup_error=str(exc))
                        save()
                        # Do not start another model if GPU release is uncertain.
                        raise RuntimeError(f"Unable to close server for {model}") from exc
                save()
        failed = any(m["status"] == "failed" for m in report["models"])
        report["status"] = "failed" if failed else "completed"
        return int(failed)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        for task in tasks:
            if task["status"] in ("pending", "running"):
                task["status"] = "interrupted"
        for model_report in report["models"]:
            if model_report["status"] in ("starting", "running"):
                model_report["status"] = "interrupted"
        return 130
    except Exception:
        report["status"] = "failed"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        report["finished_at"] = now()
        save()
        print(f"Orchestration report: {report_path}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
