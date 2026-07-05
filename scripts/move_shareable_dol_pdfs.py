#!/usr/bin/env python3
"""Move DOL archive PDFs listed in the shareable CBA manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_MANIFEST = Path("data_repository/shareable_100_cba_manifest.csv")
DEFAULT_SOURCE_DIR = Path("data_repository/dol_archive")
DEFAULT_DEST_DIR = Path("cache/dol_archive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move PDFs from data_repository/dol_archive to cache/dol_archive "
            "when the PDF stem matches raw_document_id in the manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--dest-dir", type=Path, default=DEFAULT_DEST_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without moving files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite same-named files already present in the destination.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each moved, skipped, or missing file.",
    )
    return parser.parse_args()


def load_raw_document_ids(manifest_path: Path) -> set[str]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "raw_document_id" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain a raw_document_id column")
        return {
            raw_document_id
            for row in reader
            if (raw_document_id := row.get("raw_document_id", "").strip())
        }


def find_pdfs_by_stem(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    }


def main() -> int:
    args = parse_args()

    raw_document_ids = load_raw_document_ids(args.manifest)
    source_pdfs = find_pdfs_by_stem(args.source_dir)
    dest_pdfs = find_pdfs_by_stem(args.dest_dir) if args.dest_dir.exists() else {}

    args.dest_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    already_present = 0
    missing = 0

    for raw_document_id in sorted(raw_document_ids):
        source_path = source_pdfs.get(raw_document_id)
        dest_path = args.dest_dir / f"{raw_document_id}.pdf"

        if source_path is None:
            if raw_document_id in dest_pdfs:
                already_present += 1
                if args.verbose:
                    print(f"already present: {dest_path}")
            else:
                missing += 1
                if args.verbose:
                    print(f"missing: {raw_document_id}.pdf")
            continue

        if dest_path.exists() and not args.overwrite:
            already_present += 1
            if args.verbose:
                print(f"skip existing: {dest_path}")
            continue

        action = "would move" if args.dry_run else "move"
        if args.verbose or args.dry_run:
            print(f"{action}: {source_path} -> {dest_path}")

        if not args.dry_run:
            shutil.move(str(source_path), str(dest_path))
        moved += 1

    if args.dry_run:
        moved_label = "would_move"
    else:
        moved_label = "moved"
    print(
        f"manifest_ids={len(raw_document_ids)} "
        f"{moved_label}={moved} "
        f"already_present={already_present} "
        f"missing={missing}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
