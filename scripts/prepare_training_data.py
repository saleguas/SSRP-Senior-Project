#!/usr/bin/env python3
"""
Materialize the merged training dataset into a single YOLO root.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.dataset import prepare_yolo_dataset
from src.pipeline.utils import list_image_files, write_json

DEFAULT_MANIFEST = "configs/datasets/domain_general_fish_all.json"
DEFAULT_DEST = "data/training/domain-general-fish-all-yolo"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the merged training YOLO dataset into data/training/."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Training manifest JSON.",
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help="Destination YOLO dataset root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = _resolve(args.manifest)
    dest_root = _resolve(args.dest)

    if dest_root.exists():
        if args.force:
            shutil.rmtree(dest_root, ignore_errors=True)
        else:
            raise FileExistsError(
                f"Destination already exists: {dest_root}. Use --force to rebuild."
            )

    dataset = prepare_yolo_dataset(manifest_path, output_root=dest_root)
    summary = {
        "manifest": str(manifest_path),
        "dataset_root": str(dataset.dataset_root),
        "yolo_root": str(dataset.yolo_root),
        "train_images": len(list_image_files(dataset.images_train)),
        "val_images": len(list_image_files(dataset.images_val)),
        "yaml_path": str(dataset.yaml_path),
    }
    write_json(dest_root / "build_info.json", summary)

    print(f"Prepared training data at {dest_root}")
    print(
        f"train images={summary['train_images']} val images={summary['val_images']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
