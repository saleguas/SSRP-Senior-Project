#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_constants import DATASET_SPECS
from src.dataset_layout import (
    AVAILABLE_TRAINING_MANIFEST,
    GENERATIVE_ROOT,
    INTERIM_ROOT,
    PROCESSED_ROOT,
    RAW_ROOT,
    TRAINING_ROOT,
    build_available_training_manifest,
    dataset_paths,
    ensure_data_layout,
    is_training_source_ready,
)


def _dir_state(path: Path | None) -> str:
    if path is None:
        return "-"
    if not path.exists():
        return "missing"
    try:
        next(path.iterdir())
    except StopIteration:
        return "empty"
    return "present"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the dataset storage layout and which training datasets are ready."
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create the expected data folders before printing status.",
    )
    parser.add_argument(
        "--write-available-manifest",
        action="store_true",
        help="Generate data/training/manifests/domain_general_fish_available.json from ready datasets.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.init:
        ensure_data_layout()

    if args.write_available_manifest:
        manifest_path = build_available_training_manifest(AVAILABLE_TRAINING_MANIFEST)
        print(f"Wrote available training manifest: {manifest_path}")

    print("Data roots")
    print(f"  raw:        {RAW_ROOT}")
    print(f"  interim:    {INTERIM_ROOT}")
    print(f"  processed:  {PROCESSED_ROOT}")
    print(f"  training:   {TRAINING_ROOT}")
    print(f"  generative: {GENERATIVE_ROOT}")
    print()
    print("Datasets")
    for spec in DATASET_SPECS:
        paths = dataset_paths(spec.name)
        train_ready = (
            "yes" if is_training_source_ready(paths.training_source) else "-"
        )
        train_source = str(paths.training_source) if paths.training_source else "-"
        print(
            f"  {spec.name} [{spec.role}] "
            f"raw={_dir_state(paths.raw_root)} "
            f"interim={_dir_state(paths.interim_root)} "
            f"processed={_dir_state(paths.processed_root)} "
            f"generative={_dir_state(paths.generative_root)} "
            f"train_ready={train_ready}"
        )
        print(f"    training_source: {train_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
