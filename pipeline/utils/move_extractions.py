from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.paths import PROJECT_ROOT, default_cache_dir

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(PROJECT_ROOT / ".env")

STAGE_NAME = "stg_02_extract"
CLASSIFY_STAGE_NAME = "stg_03_classify"
KNOWN_SOURCES = ("dol_archive", "cornell_dol", "cornell_retail_educ")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # copy2 preserves file metadata (mtime, permissions).
    shutil.copy2(str(src), str(dst))


def move_source(
    source: str,
    stage: str,
    src_cache: Path,
    dst_cache: Path,
    dry_run: bool,
    document_ids: Sequence[str] | None = None,
) -> list[Path]:
    src_dir = src_cache / stage / source
    dst_dir = dst_cache / stage / source

    if not src_dir.exists():
        if document_ids is not None:
            print(f"  [warn] extraction source not found: {src_dir}")
            return []
        raise SystemExit(f"source not found: {src_dir}")
    if not src_dir.is_dir():
        raise SystemExit(f"source is not a directory: {src_dir}")

    # Document identity comes from the stage layout, not an extraction
    # filename: <extraction-model>/<document-id>/.  Keeping this at exactly two
    # levels also handles nested extraction artifacts and empty document dirs.
    all_doc_rel_dirs = sorted(
        document_dir.relative_to(src_dir)
        for model_dir in src_dir.iterdir()
        if model_dir.is_dir()
        for document_dir in model_dir.iterdir()
        if document_dir.is_dir()
    )
    selected_ids = None if document_ids is None else set(document_ids)
    doc_rel_dirs = [
        rel_dir
        for rel_dir in all_doc_rel_dirs
        if selected_ids is None or rel_dir.parts[1] in selected_ids
    ]
    if selected_ids is None:
        files = [path for path in src_dir.rglob("*") if path.is_file()]
    else:
        files = sorted(
            path
            for rel_dir in doc_rel_dirs
            for path in (src_dir / rel_dir).rglob("*")
            if path.is_file()
        )
        matched_ids = _extracted_document_ids(doc_rel_dirs)
        for document_id in sorted(selected_ids - set(matched_ids)):
            print(f"  [warn] no extraction found for selected document: {document_id}")

    print(f"copying {len(files)} file(s)")
    print(f"  from {src_dir}")
    print(f"    to {dst_dir}")

    if dry_run:
        for path in files:
            print(f"  [dry-run] {path.relative_to(src_dir)}")
        return doc_rel_dirs

    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    for path in files:
        rel = path.relative_to(src_dir)
        _copy_file(path, dst_dir / rel)

    print(f"done: copied {len(files)} file(s) to {dst_dir}")
    return doc_rel_dirs


def _extracted_document_ids(doc_rel_dirs: Sequence[Path]) -> list[str]:
    """Return the de-duplicated document IDs represented in stage 02."""

    return sorted(
        {
            rel_dir.parts[1]
            for rel_dir in doc_rel_dirs
            if len(rel_dir.parts) >= 2
        }
    )


def _classification_artifacts(document_dir: Path) -> list[Path]:
    """Return every classification output file for one document."""

    if not document_dir.is_dir():
        return []
    return sorted(path for path in document_dir.rglob("*") if path.is_file())


def move_classifications(
    source: str,
    classify_stage: str,
    doc_rel_dirs: Sequence[Path],
    src_cache: Path,
    dst_cache: Path,
    dry_run: bool,
    document_ids: Sequence[str] | None = None,
) -> None:
    """Copy stage-03 classifications for the selected documents.

    Stage 03 is keyed by the *classification* model, which need not match the
    extraction model that produced its input, so documents are matched by ID
    across every classification model arm.
    """

    src_stage_dir = src_cache / classify_stage / source
    dst_stage_dir = dst_cache / classify_stage / source
    selected_document_ids = (
        _extracted_document_ids(doc_rel_dirs)
        if document_ids is None
        else sorted(set(document_ids))
    )

    print(
        "copying classification file(s) for "
        f"{len(selected_document_ids)} selected document(s)"
    )
    print(f"  from {src_stage_dir}")
    print(f"    to {dst_stage_dir}")

    if not src_stage_dir.is_dir():
        print(f"  [warn] classification source not found: {src_stage_dir}")
        return

    model_dirs = sorted(
        path
        for path in src_stage_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    copied = 0
    matched_documents: set[str] = set()
    matched_model_documents = 0
    for model_dir in model_dirs:
        for document_id in selected_document_ids:
            artifacts = _classification_artifacts(model_dir / document_id)
            if not artifacts:
                continue
            matched_documents.add(document_id)
            matched_model_documents += 1
            for artifact in artifacts:
                rel = artifact.relative_to(src_stage_dir)
                if dry_run:
                    print(f"  [dry-run] {rel}")
                else:
                    _copy_file(artifact, dst_stage_dir / rel)
                copied += 1

    for document_id in sorted(set(selected_document_ids) - matched_documents):
        print(f"  [warn] no classifications for selected document: {document_id}")

    if dry_run:
        return

    print(
        f"done: copied {copied} classification file(s) from "
        f"{matched_model_documents} model/document output(s) to {dst_stage_dir}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy stg_02_extract extractions and the matching stg_03_classify "
            "outputs for every classification model, for one source from the "
            "external CACHE_DIR into the working-directory cache/, preserving "
            "the stage/source directory layout.  OCR text from stg_01_ocr is "
            "never copied."
        )
    )
    parser.add_argument(
        "source",
        help=f"source folder to copy (e.g. {', '.join(KNOWN_SOURCES)})",
    )
    parser.add_argument(
        "--document-id",
        "--document-ids",
        dest="document_ids",
        metavar="DOCUMENT_ID",
        nargs="+",
        action="extend",
        help=(
            "copy only these document IDs; accepts one or more values and may "
            "be repeated"
        ),
    )
    parser.add_argument(
        "--stage",
        default=STAGE_NAME,
        help=f"stage folder under the cache (default: {STAGE_NAME})",
    )
    parser.add_argument(
        "--classify-stage",
        default=CLASSIFY_STAGE_NAME,
        help=(
            "stage folder holding classification outputs "
            f"(default: {CLASSIFY_STAGE_NAME})"
        ),
    )
    parser.add_argument(
        "--no-classify",
        dest="no_classify",
        action="store_true",
        help="do not copy matching stg_03_classify outputs",
    )
    parser.add_argument(
        "--src-cache",
        default=None,
        help="source cache dir (default: CACHE_DIR from .env)",
    )
    parser.add_argument(
        "--dst-cache",
        default=None,
        help="destination cache dir (default: <project>/cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be copied without copying anything",
    )
    args = parser.parse_args(argv)

    src_cache = (
        Path(args.src_cache).expanduser().resolve()
        if args.src_cache
        else default_cache_dir()
    )
    dst_cache = (
        Path(args.dst_cache).expanduser().resolve()
        if args.dst_cache
        else PROJECT_ROOT / "cache"
    )

    if src_cache == dst_cache:
        raise SystemExit("source and destination cache are the same directory")

    doc_rel_dirs = move_source(
        args.source,
        args.stage,
        src_cache,
        dst_cache,
        args.dry_run,
        document_ids=args.document_ids,
    )

    if not args.no_classify:
        move_classifications(
            args.source,
            args.classify_stage,
            doc_rel_dirs,
            src_cache,
            dst_cache,
            args.dry_run,
            document_ids=args.document_ids,
        )


if __name__ == "__main__":
    main()
