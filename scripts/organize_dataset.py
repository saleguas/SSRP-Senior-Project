#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_constants import DATASET_SPECS, get_dataset_spec
from src.dataset_organize import run_organize


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dataset-specific organize script for a canonical dataset name."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name or alias.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported datasets and exit.",
    )
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def _print_datasets() -> None:
    for spec in DATASET_SPECS:
        aliases = ", ".join(spec.aliases)
        print(
            f"{spec.name} [{spec.role}]"
            + (f" aliases: {aliases}" if aliases else "")
        )


def main() -> int:
    args, passthrough = _parse_args()
    if args.list:
        _print_datasets()
        return 0
    if not args.dataset:
        raise SystemExit("Dataset name required. Use --list to see options.")

    spec = get_dataset_spec(args.dataset)
    return run_organize(spec.name, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
