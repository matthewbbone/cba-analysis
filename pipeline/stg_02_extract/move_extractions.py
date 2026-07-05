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
KNOWN_SOURCES = ("dol_archive", "cornell_dol", "cornell_retail_educ")


def move_source(
    source: str,
    stage: str,
    src_cache: Path,
    dst_cache: Path,
    dry_run: bool,
) -> None:
    src_dir = src_cache / stage / source
    dst_dir = dst_cache / stage / source

    if not src_dir.exists():
        raise SystemExit(f"source not found: {src_dir}")
    if not src_dir.is_dir():
        raise SystemExit(f"source is not a directory: {src_dir}")

    files = [path for path in src_dir.rglob("*") if path.is_file()]
    print(f"moving {len(files)} file(s)")
    print(f"  from {src_dir}")
    print(f"    to {dst_dir}")

    if dry_run:
        for path in files:
            print(f"  [dry-run] {path.relative_to(src_dir)}")
        return

    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    for path in files:
        rel = path.relative_to(src_dir)
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # str() avoids cross-filesystem rename failures; shutil.move falls back
        # to copy+delete when src and dst live on different mounts.
        shutil.move(str(path), str(target))

    # Remove now-empty source directory tree.
    shutil.rmtree(src_dir, ignore_errors=True)
    print(f"done: moved {len(files)} file(s) to {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move stg_02_extract extractions for one source from the external "
            "CACHE_DIR into the working-directory cache/, preserving the "
            "stage/source directory layout."
        )
    )
    parser.add_argument(
        "source",
        help=f"source folder to move (e.g. {', '.join(KNOWN_SOURCES)})",
    )
    parser.add_argument(
        "--stage",
        default=STAGE_NAME,
        help=f"stage folder under the cache (default: {STAGE_NAME})",
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
        help="list what would be moved without moving anything",
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

    move_source(args.source, args.stage, src_cache, dst_cache, args.dry_run)


if __name__ == "__main__":
    main()
