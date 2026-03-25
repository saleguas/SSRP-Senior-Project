#!/usr/bin/env python3
"""
Materialize the merged training dataset into a single YOLO root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset_layout import (
    AVAILABLE_TRAINING_MANIFEST,
    AVAILABLE_TRAINING_ROOT,
    build_available_training_manifest,
    ensure_data_layout,
)
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


def _resolve_manifest_source(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    return path


def _validate_manifest_sources(manifest_path: Path) -> None:
    if not manifest_path.exists() or manifest_path.suffix.lower() != ".json":
        return

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    source_specs = payload.get("sources")
    if not isinstance(source_specs, list):
        return

    missing_paths: list[Path] = []
    invalid_roots: list[Path] = []
    for spec in source_specs:
        if not isinstance(spec, dict):
            continue
        path_value = spec.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            continue

        source_root = _resolve_manifest_source(manifest_path, path_value)
        if not source_root.exists():
            missing_paths.append(source_root)
            continue

        if source_root.is_file() and source_root.suffix.lower() == ".json":
            continue
        if (source_root / "data.yaml").exists():
            continue
        if (source_root / "annotations.csv").exists():
            continue
        invalid_roots.append(source_root)

    problems: list[str] = []
    if missing_paths:
        problems.append(
            "Missing source paths:\n"
            + "\n".join(f"  - {path}" for path in missing_paths)
        )
    if invalid_roots:
        problems.append(
            "Source roots exist but are not recognizable datasets:\n"
            + "\n".join(f"  - {path}" for path in invalid_roots)
        )
    if problems:
        raise FileNotFoundError(
            "Cannot rebuild training data from the manifest.\n" + "\n".join(problems)
        )


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
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Build a merged dataset from whichever training datasets are currently ready.",
    )
    parser.add_argument(
        "--manifest-out",
        default="",
        help="Where to write the generated available-only manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ensure_data_layout()

    manifest_default = DEFAULT_MANIFEST
    dest_default = DEFAULT_DEST
    if args.available_only:
        manifest_default = str(AVAILABLE_TRAINING_MANIFEST)
        dest_default = str(AVAILABLE_TRAINING_ROOT)

    manifest_path = _resolve(args.manifest if args.manifest != DEFAULT_MANIFEST else manifest_default)
    dest_root = _resolve(args.dest if args.dest != DEFAULT_DEST else dest_default)

    if args.available_only:
        manifest_output = (
            _resolve(args.manifest_out)
            if args.manifest_out
            else manifest_path
        )
        manifest_path = build_available_training_manifest(manifest_output)

    _validate_manifest_sources(manifest_path)

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
