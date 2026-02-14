#!/usr/bin/env python3
"""
Organize the AAU Zebrafish ReID dataset into a normalized folder layout.

Default layout:
  data/interim/aau-zebrafish-reid/
    annotations.csv
    vid1/*.png
    vid2/*.png
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DATASET_NAME = "aau-zebrafish-reid"
ZIP_NAME = "aau-zebrafish-reid.zip"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize AAU Zebrafish ReID dataset into vid folders."
    )
    parser.add_argument(
        "--source",
        default=f"data/raw/{DATASET_NAME}",
        help="Source dataset directory containing the raw unzip.",
    )
    parser.add_argument(
        "--dest",
        default=f"data/interim/{DATASET_NAME}",
        help="Destination dataset directory.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the zip file in the source directory.",
    )
    parser.add_argument(
        "--clean-empty",
        action="store_true",
        help="Remove empty source folders after organizing.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def move_images(source_dir: Path, dest_dir: Path, copy: bool) -> tuple[int, int]:
    moved = 0
    skipped = 0
    for prefix in ("Vid1", "Vid2"):
        target_dir = dest_dir / prefix.lower()
        ensure_dir(target_dir)
        for src in sorted(source_dir.rglob(f"{prefix}_*.png")):
            dst = target_dir / src.name
            if dst.exists():
                skipped += 1
                continue
            move_or_copy(src, dst, copy=copy)
            moved += 1
    return moved, skipped


def move_annotations(source_dir: Path, dest_dir: Path, copy: bool) -> None:
    src = source_dir / "annotations.csv"
    if not src.exists():
        return
    dst = dest_dir / "annotations.csv"
    if dst.exists():
        return
    move_or_copy(src, dst, copy=copy)


def remove_zip(source_dir: Path) -> None:
    zip_path = source_dir / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()


def remove_empty_dirs(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                continue
    try:
        path.rmdir()
    except OSError:
        pass


def main() -> int:
    args = parse_args()
    root = repo_root()
    source_dir = (root / args.source).resolve()
    dest_dir = (root / args.dest).resolve()

    ensure_dir(dest_dir)

    moved, skipped = move_images(source_dir, dest_dir, copy=args.copy)
    move_annotations(source_dir, dest_dir, copy=args.copy)

    if not args.keep_zip:
        remove_zip(source_dir)

    if args.clean_empty and source_dir.exists():
        remove_empty_dirs(source_dir)

    print(
        f"Organized {DATASET_NAME}: moved {moved} files, skipped {skipped} existing files."
    )
    print(f"Destination: {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
