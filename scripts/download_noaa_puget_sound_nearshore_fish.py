#!/usr/bin/env python3
"""
Download the NOAA Puget Sound Nearshore Fish 2017-2018 dataset.

By default this downloads both the image archive and the annotations zip into
data/raw/noaa-puget-sound-nearshore-fish/. Downloads are resumable when the
server supports HTTP range requests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

IMAGE_ZIP_URL = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-images.zip"
ANNOTATIONS_ZIP_URL = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-annotations-2023.08.19.zip"
IMAGE_ZIP_NAME = "noaa_estuary_fish-images.zip"
ANNOTATIONS_ZIP_NAME = "noaa_estuary_fish-annotations-2023.08.19.zip"
DEFAULT_OUTPUT = "data/raw/noaa-puget-sound-nearshore-fish"
CHUNK_SIZE = 8 * 1024 * 1024


def resolve_output_dir(output: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(output)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    return out_dir


def remote_size(url: str) -> int | None:
    req = Request(url, method="HEAD")
    try:
        with urlopen(req) as response:
            header = response.headers.get("Content-Length")
            return int(header) if header else None
    except Exception:
        return None


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def download_file(url: str, destination: Path, force: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    total_size = remote_size(url)

    if destination.exists() and total_size is not None and existing_size == total_size and not force:
        print(f"Already complete: {destination} ({format_bytes(total_size)})")
        return destination
    if destination.exists() and total_size is None and not force:
        print(f"Existing file kept (remote size unknown): {destination}")
        return destination
    if force and destination.exists():
        destination.unlink()
        existing_size = 0

    headers = {}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        print(f"Resuming {destination.name} from {format_bytes(existing_size)}")
    else:
        print(f"Downloading {destination.name}")

    try:
        req = Request(url, headers=headers)
        response = urlopen(req)
        status = getattr(response, "status", None)
        if existing_size > 0 and status != 206:
            print(
                f"Server ignored resume request for {destination.name}; restarting download."
            )
            response.close()
            destination.unlink(missing_ok=True)
            existing_size = 0
            mode = "wb"
            response = urlopen(Request(url))

        with response:
            expected = total_size if total_size is not None else None
            downloaded = existing_size
            with destination.open(mode) as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if expected:
                        pct = downloaded / expected * 100
                        print(
                            f"\r{destination.name}: {pct:6.2f}% "
                            f"({format_bytes(downloaded)} / {format_bytes(expected)})",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r{destination.name}: {format_bytes(downloaded)}",
                            end="",
                            flush=True,
                        )
            print()
    except HTTPError as exc:
        raise SystemExit(f"HTTP error downloading {url}: {exc}") from exc
    except URLError as exc:
        raise SystemExit(f"Network error downloading {url}: {exc}") from exc

    final_size = destination.stat().st_size
    if total_size is not None and final_size != total_size:
        raise SystemExit(
            f"Download incomplete for {destination}: "
            f"expected {total_size} bytes, got {final_size} bytes"
        )

    print(f"Saved {destination} ({format_bytes(final_size)})")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the NOAA Puget Sound Nearshore Fish dataset."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart download even if the local file already exists.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip the image archive download.",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Skip the annotations zip download.",
    )
    parser.add_argument(
        "--print-source",
        action="store_true",
        help="Print the archive URLs and exit.",
    )
    args = parser.parse_args()

    if args.print_source:
        print(IMAGE_ZIP_URL)
        print(ANNOTATIONS_ZIP_URL)
        return 0

    out_dir = resolve_output_dir(args.output)
    if not args.skip_images:
        download_file(IMAGE_ZIP_URL, out_dir / IMAGE_ZIP_NAME, force=args.force)
    if not args.skip_annotations:
        download_file(
            ANNOTATIONS_ZIP_URL,
            out_dir / ANNOTATIONS_ZIP_NAME,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
