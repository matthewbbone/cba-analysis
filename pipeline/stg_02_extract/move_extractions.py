from __future__ import annotations

import argparse
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
INPUT_STAGE_NAME = "stg_01_ocr"
GROUNDING_FILENAME = "full.txt"
OCR_MODEL_DIR = "AIDC-AI_Ovis2.6-30B-A3B"
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
) -> list[Path]:
    src_dir = src_cache / stage / source
    dst_dir = dst_cache / stage / source

    if not src_dir.exists():
        raise SystemExit(f"source not found: {src_dir}")
    if not src_dir.is_dir():
        raise SystemExit(f"source is not a directory: {src_dir}")

    files = [path for path in src_dir.rglob("*") if path.is_file()]
    # Document directories are the parents of the extraction files, kept as
    # <model>/<document_id> paths relative to the source dir so the matching
    # grounding full.txt can be located under the input stage.
    doc_rel_dirs = sorted({path.parent.relative_to(src_dir) for path in files})

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


def move_grounding(
    source: str,
    input_stage: str,
    doc_rel_dirs: list[Path],
    src_cache: Path,
    dst_cache: Path,
    dry_run: bool,
) -> None:
    src_stage_dir = src_cache / input_stage / source
    dst_stage_dir = dst_cache / input_stage / source

    print(f"copying grounding {GROUNDING_FILENAME} file(s)")
    print(f"  from {src_stage_dir}")
    print(f"    to {dst_stage_dir}")

    copied = 0
    missing = 0
    for rel_dir in doc_rel_dirs:
        # Extractions may come from a different model than the OCR stage, so
        # groundings are always looked up under the fixed OCR model dir.
        ocr_rel_dir = Path(OCR_MODEL_DIR) / rel_dir.name
        src_grounding = src_stage_dir / ocr_rel_dir / GROUNDING_FILENAME
        if not src_grounding.exists():
            missing += 1
            print(f"  [warn] missing grounding: {src_grounding}")
            continue

        dst_grounding = dst_stage_dir / ocr_rel_dir / GROUNDING_FILENAME
        if dry_run:
            print(f"  [dry-run] {ocr_rel_dir / GROUNDING_FILENAME}")
        else:
            _copy_file(src_grounding, dst_grounding)
        copied += 1

    if dry_run:
        return

    summary = f"done: copied {copied} grounding file(s) to {dst_stage_dir}"
    if missing:
        summary += f" ({missing} missing)"
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy stg_02_extract extractions (and their grounding full.txt "
            "files) for one source from the external CACHE_DIR into the "
            "working-directory cache/, preserving the stage/source directory "
            "layout."
        )
    )
    parser.add_argument(
        "source",
        help=f"source folder to copy (e.g. {', '.join(KNOWN_SOURCES)})",
    )
    parser.add_argument(
        "--stage",
        default=STAGE_NAME,
        help=f"stage folder under the cache (default: {STAGE_NAME})",
    )
    parser.add_argument(
        "--input-stage",
        default=INPUT_STAGE_NAME,
        help=(
            "input stage folder holding the grounding "
            f"{GROUNDING_FILENAME} files (default: {INPUT_STAGE_NAME})"
        ),
    )
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help=f"do not move the matching {GROUNDING_FILENAME} grounding files",
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
    args = parser.parse_args()

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
        args.source, args.stage, src_cache, dst_cache, args.dry_run
    )

    if not args.no_grounding:
        move_grounding(
            args.source,
            args.input_stage,
            doc_rel_dirs,
            src_cache,
            dst_cache,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
