#!/usr/bin/env python3
"""
Download the AAU Zebrafish Re-Identification dataset via the Kaggle API.

By default this only downloads the zip. Use --unzip to extract.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "aalborguniversity/aau-zebrafish-reid"
ZIP_NAME = "aau-zebrafish-reid.zip"


def resolve_output_dir(output: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(output)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    return out_dir


def download_dataset(out_dir: Path, force: bool) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise SystemExit(
            "Kaggle API not available. Install with: pip install kaggle"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / ZIP_NAME

    if zip_path.exists() and not force:
        print(f"Zip already exists: {zip_path}")
        return zip_path

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        DATASET_SLUG, path=str(out_dir), force=force, quiet=False
    )

    if not zip_path.exists():
        candidates = list(out_dir.glob("*.zip"))
        if len(candidates) == 1:
            candidates[0].rename(zip_path)
        elif len(candidates) > 1:
            print(
                f"Multiple zip files found in {out_dir}; expected {ZIP_NAME}",
                file=sys.stderr,
            )

    return zip_path


def unzip_dataset(zip_path: Path, out_dir: Path, overwrite: bool) -> None:
    if not zip_path.exists():
        raise SystemExit(f"Zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        if overwrite:
            zf.extractall(out_dir)
            return

        for member in zf.infolist():
            target = out_dir / member.filename
            if target.exists():
                continue
            zf.extract(member, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the AAU Zebrafish ReID dataset from Kaggle."
    )
    parser.add_argument(
        "--output",
        default="data/raw/aau-zebrafish-reid",
        help="Output directory for the downloaded zip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the zip already exists.",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Extract the zip after downloading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files when unzipping.",
    )

    args = parser.parse_args()
    out_dir = resolve_output_dir(args.output)
    zip_path = download_dataset(out_dir, force=args.force)

    if args.unzip:
        unzip_dataset(zip_path, out_dir, overwrite=args.overwrite)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
