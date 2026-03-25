from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.dataset_constants import (
    AAU_ZEBRAFISH_REID,
    DEEP_VISION_FISH,
    KAKADU_FISHAI,
    LIAO_LAB_VIDEOS,
    MIT_RIVER_HERRING,
    NOAA_PUGET_SOUND_NEARSHORE_FISH,
)

DATASET_SLUG_AAU = "aalborguniversity/aau-zebrafish-reid"
AAU_ZIP_NAME = "aau-zebrafish-reid.zip"

DEEP_VISION_PAGE_URL = (
    "https://metadata.nmdc.no/metadata-api/landingpage/"
    "01d102345aef4639f063a13ea20cd3f3"
)
DEEP_VISION_ZIP_URL = (
    "https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip"
)
DEEP_VISION_ZIP_NAME = "fishDatasetSimulationAlgorithm.zip"

KAKADU_PAGE_URL = "https://zenodo.org/records/7250921"
KAKADU_ZIP_URL = (
    "https://zenodo.org/records/7250921/files/"
    "202210-KakaduFishAI-TrainingData.zip?download=1"
)
KAKADU_ZIP_NAME = "202210-KakaduFishAI-TrainingData.zip"

MIT_IMAGE_ZIP_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "mit-river-herring/mit_river_herring.zip"
)
MIT_METADATA_ZIP_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "mit-river-herring/mit_sea_grant_river_herring.json.zip"
)

NOAA_IMAGE_ZIP_URL = (
    "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/"
    "noaa_estuary_fish-images.zip"
)
NOAA_ANNOTATIONS_ZIP_URL = (
    "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/"
    "noaa_estuary_fish-annotations-2023.08.19.zip"
)
NOAA_IMAGE_ZIP_NAME = "noaa_estuary_fish-images.zip"
NOAA_ANNOTATIONS_ZIP_NAME = "noaa_estuary_fish-annotations-2023.08.19.zip"

CHUNK_SIZE = 8 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_output_dir(output: str) -> Path:
    out_dir = Path(output).expanduser()
    if not out_dir.is_absolute():
        out_dir = (repo_root() / out_dir).resolve()
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

    if (
        destination.exists()
        and total_size is not None
        and existing_size == total_size
        and not force
    ):
        print(f"Already complete: {destination} ({format_bytes(total_size)})")
        return destination
    if destination.exists() and total_size is None and not force:
        print(f"Existing file kept (remote size unknown): {destination}")
        return destination
    if force and destination.exists():
        destination.unlink()
        existing_size = 0

    headers: dict[str, str] = {}
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


def download_aau(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download the AAU Zebrafish ReID dataset from Kaggle."
    )
    parser.add_argument(
        "--output",
        default=f"data/raw/{AAU_ZEBRAFISH_REID}",
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
    args = parser.parse_args(argv)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise SystemExit(
            "Kaggle API not available. Install with: pip install kaggle"
        ) from exc

    out_dir = resolve_output_dir(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / AAU_ZIP_NAME

    if zip_path.exists() and not args.force:
        print(f"Zip already exists: {zip_path}")
    else:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            DATASET_SLUG_AAU, path=str(out_dir), force=args.force, quiet=False
        )
        if not zip_path.exists():
            candidates = list(out_dir.glob("*.zip"))
            if len(candidates) == 1:
                candidates[0].rename(zip_path)

    if args.unzip:
        with zipfile.ZipFile(zip_path, "r") as archive:
            if args.overwrite:
                archive.extractall(out_dir)
            else:
                for member in archive.infolist():
                    target = out_dir / member.filename
                    if not target.exists():
                        archive.extract(member, out_dir)
    return 0


def download_deep_vision(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download the Deep Vision fish dataset.")
    parser.add_argument(
        "--output",
        default=f"data/raw/{DEEP_VISION_FISH}",
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart download even if the local file already exists.",
    )
    parser.add_argument(
        "--print-source",
        action="store_true",
        help="Print the dataset landing page URL and exit.",
    )
    args = parser.parse_args(argv)
    if args.print_source:
        print(DEEP_VISION_PAGE_URL)
        return 0
    out_dir = resolve_output_dir(args.output)
    download_file(DEEP_VISION_ZIP_URL, out_dir / DEEP_VISION_ZIP_NAME, force=args.force)
    return 0


def download_kakadu(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download the Kakadu FishAI training dataset from Zenodo."
    )
    parser.add_argument(
        "--output",
        default=f"data/raw/{KAKADU_FISHAI}",
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart download even if the local file already exists.",
    )
    parser.add_argument(
        "--print-source",
        action="store_true",
        help="Print the dataset landing page URL and exit.",
    )
    args = parser.parse_args(argv)
    if args.print_source:
        print(KAKADU_PAGE_URL)
        return 0
    out_dir = resolve_output_dir(args.output)
    download_file(KAKADU_ZIP_URL, out_dir / KAKADU_ZIP_NAME, force=args.force)
    return 0


def download_mit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download the MIT Sea Grant River Herring dataset from LILA."
    )
    parser.add_argument(
        "--output",
        default=f"data/raw/{MIT_RIVER_HERRING}",
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart downloads even if local files already exist.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only the metadata zip.",
    )
    args = parser.parse_args(argv)
    out_dir = resolve_output_dir(args.output)
    targets = [(MIT_METADATA_ZIP_URL, "mit_sea_grant_river_herring.json.zip")]
    if not args.metadata_only:
        targets.insert(0, (MIT_IMAGE_ZIP_URL, "mit_river_herring.zip"))
    for url, filename in targets:
        download_file(url, out_dir / filename, force=args.force)
    return 0


def download_noaa(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download the NOAA Puget Sound Nearshore Fish dataset."
    )
    parser.add_argument(
        "--output",
        default=f"data/raw/{NOAA_PUGET_SOUND_NEARSHORE_FISH}",
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
    args = parser.parse_args(argv)
    if args.print_source:
        print(NOAA_IMAGE_ZIP_URL)
        print(NOAA_ANNOTATIONS_ZIP_URL)
        return 0
    out_dir = resolve_output_dir(args.output)
    if not args.skip_images:
        download_file(NOAA_IMAGE_ZIP_URL, out_dir / NOAA_IMAGE_ZIP_NAME, force=args.force)
    if not args.skip_annotations:
        download_file(
            NOAA_ANNOTATIONS_ZIP_URL,
            out_dir / NOAA_ANNOTATIONS_ZIP_NAME,
            force=args.force,
        )
    return 0


def run_download(dataset_name: str, argv: list[str]) -> int:
    if dataset_name == AAU_ZEBRAFISH_REID:
        return download_aau(argv)
    if dataset_name == DEEP_VISION_FISH:
        return download_deep_vision(argv)
    if dataset_name == KAKADU_FISHAI:
        return download_kakadu(argv)
    if dataset_name == MIT_RIVER_HERRING:
        return download_mit(argv)
    if dataset_name == NOAA_PUGET_SOUND_NEARSHORE_FISH:
        return download_noaa(argv)
    if dataset_name == LIAO_LAB_VIDEOS:
        raise SystemExit(
            f"{LIAO_LAB_VIDEOS} does not have a download step. Use organize_dataset.py instead."
        )
    raise SystemExit(f"Unsupported dataset: {dataset_name}")
