#!/usr/bin/env python3
"""
Normalize the MIT Sea Grant River Herring dataset into the repo's interim layout.

Default output layout:
  data/interim/mit-sea-grant-river-herring/
    annotations.csv
    coonamessett_1086961/
      coonamessett_1086961_frame_000000.png
      ...

Images are extracted directly from the downloaded zip into the normalized layout.
The generated annotations.csv matches the project's existing semicolon-delimited
format closely enough for the current loaders and training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

DATASET_NAME = "mit-sea-grant-river-herring"
IMAGE_ZIP_NAME = "mit_river_herring.zip"
METADATA_ZIP_NAME = "mit_sea_grant_river_herring.json.zip"
METADATA_JSON_NAME = "mit_sea_grant_river_herring.json"
ARCHIVE_ROOT = "mit_river_herring"
COPY_CHUNK_SIZE = 8 * 1024 * 1024
CSV_HEADER = [
    "Filename",
    "Object ID",
    "Annotation tag",
    "Upper left corner X",
    "Upper left corner Y",
    "Lower right corner X",
    "Lower right corner Y",
    "Right,Turning,Occlusion,Glitch",
    "Buffer",
    "",
    "",
]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    archive_name: str
    clip_key: str
    clip_name: str
    file_name: str
    dest_path: Path
    width: int
    height: int
    location: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the MIT Sea Grant River Herring dataset."
    )
    parser.add_argument(
        "--source-zip",
        default=f"data/raw/{DATASET_NAME}/{IMAGE_ZIP_NAME}",
        help="Path to the downloaded image zip.",
    )
    parser.add_argument(
        "--metadata-json",
        default=f"data/raw/{DATASET_NAME}/metadata/{METADATA_JSON_NAME}",
        help="Path to the extracted metadata JSON.",
    )
    parser.add_argument(
        "--metadata-zip",
        default=f"data/raw/{DATASET_NAME}/{METADATA_ZIP_NAME}",
        help="Path to the metadata zip (used if --metadata-json is missing).",
    )
    parser.add_argument(
        "--dest",
        default=f"data/interim/{DATASET_NAME}",
        help="Destination dataset directory.",
    )
    parser.add_argument(
        "--location",
        action="append",
        default=[],
        help="Restrict to one or more locations (repeatable): coonamessett, ipswich, santuit.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit organization to the first N clips after filtering.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted images.",
    )
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def _load_metadata(metadata_json: Path, metadata_zip: Path) -> dict:
    if metadata_json.exists():
        with metadata_json.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if metadata_zip.exists():
        with zipfile.ZipFile(metadata_zip) as zf:
            with zf.open(METADATA_JSON_NAME) as handle:
                with io.TextIOWrapper(handle, encoding="utf-8") as reader:
                    return json.load(reader)

    raise FileNotFoundError(
        f"Metadata not found. Checked {metadata_json} and {metadata_zip}."
    )


def _clip_key(file_name: str) -> str:
    parts = file_name.split("/")
    if len(parts) < 2:
        raise ValueError(f"Unexpected image path: {file_name}")
    return "/".join(parts[:2])


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _clip_name(clip_key: str) -> str:
    left, right = clip_key.split("/", 1)
    return f"{_sanitize_name(left)}_{_sanitize_name(right)}"


def _frame_name(clip_name: str, original_file_name: str) -> str:
    stem = Path(original_file_name).stem.lower()
    return f"{clip_name}_{stem}.png"


def _iter_selected_images(
    images: Iterable[dict],
    dest_root: Path,
    locations: List[str],
    max_clips: int | None,
) -> List[ImageRecord]:
    normalized_locations = {value.strip().lower() for value in locations if value.strip()}
    filtered_images = [
        image
        for image in images
        if not normalized_locations
        or str(image.get("location", "")).strip().lower() in normalized_locations
    ]

    ordered_clip_keys = sorted({_clip_key(image["file_name"]) for image in filtered_images})
    if max_clips is not None:
        ordered_clip_keys = ordered_clip_keys[: max(0, max_clips)]
    allowed_clips = set(ordered_clip_keys)

    records: List[ImageRecord] = []
    for image in filtered_images:
        clip_key = _clip_key(image["file_name"])
        if clip_key not in allowed_clips:
            continue
        clip_name = _clip_name(clip_key)
        file_name = _frame_name(clip_name, image["file_name"])
        records.append(
            ImageRecord(
                image_id=str(image["id"]),
                archive_name=f"{ARCHIVE_ROOT}/{image['file_name']}",
                clip_key=clip_key,
                clip_name=clip_name,
                file_name=file_name,
                dest_path=dest_root / clip_name / file_name,
                width=int(image["width"]),
                height=int(image["height"]),
                location=str(image.get("location", "")),
            )
        )

    return records


def _extract_images(
    source_zip: Path,
    image_records: List[ImageRecord],
    force: bool,
) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    missing: List[str] = []

    with zipfile.ZipFile(source_zip) as zf:
        total = len(image_records)
        for index, record in enumerate(image_records, start=1):
            record.dest_path.parent.mkdir(parents=True, exist_ok=True)
            if record.dest_path.exists() and not force:
                skipped += 1
            else:
                try:
                    with zf.open(record.archive_name) as src, record.dest_path.open(
                        "wb"
                    ) as dst:
                        shutil.copyfileobj(src, dst, COPY_CHUNK_SIZE)
                except KeyError:
                    missing.append(record.archive_name)
                    continue
                extracted += 1

            if index % 1000 == 0 or index == total:
                print(
                    f"Processed {index}/{total} images "
                    f"(extracted {extracted}, skipped {skipped})"
                )

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} files in archive. First missing entries: {preview}"
        )

    return extracted, skipped


def _bbox_xyxy(annotation: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = annotation["bbox"]
    x1 = max(0, min(width - 1, int(math.floor(x))))
    y1 = max(0, min(height - 1, int(math.floor(y))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y + h))))
    return x1, y1, x2, y2


def _write_annotations_csv(
    dest_root: Path,
    image_records: List[ImageRecord],
    annotations: Iterable[dict],
    categories: Iterable[dict],
) -> int:
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    image_lookup = {record.image_id: record for record in image_records}
    annotations_by_image: Dict[str, List[dict]] = defaultdict(list)

    for annotation in annotations:
        image_id = str(annotation.get("image_id"))
        if image_id in image_lookup:
            annotations_by_image[image_id].append(annotation)

    annotations_path = dest_root / "annotations.csv"
    row_count = 0
    with annotations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(CSV_HEADER)

        for record in image_records:
            image_annotations = sorted(
                annotations_by_image.get(record.image_id, []),
                key=lambda item: str(item.get("id", "")),
            )
            for object_id, annotation in enumerate(image_annotations, start=1):
                x1, y1, x2, y2 = _bbox_xyxy(annotation, record.width, record.height)
                writer.writerow(
                    [
                        record.file_name,
                        object_id,
                        category_names.get(int(annotation["category_id"]), "fish"),
                        x1,
                        y1,
                        x2,
                        y2,
                        "0,0,0,0",
                        0,
                        "",
                        "",
                    ]
                )
                row_count += 1

    return row_count


def main() -> int:
    args = parse_args()
    source_zip = _resolve_path(args.source_zip)
    metadata_json = _resolve_path(args.metadata_json)
    metadata_zip = _resolve_path(args.metadata_zip)
    dest_root = _resolve_path(args.dest)

    if not source_zip.exists():
        raise FileNotFoundError(f"Image zip not found: {source_zip}")

    coco = _load_metadata(metadata_json, metadata_zip)
    image_records = _iter_selected_images(
        coco.get("images", []),
        dest_root=dest_root,
        locations=args.location,
        max_clips=args.max_clips,
    )
    if not image_records:
        raise SystemExit("No images matched the requested filters.")

    selected_clips = sorted({record.clip_name for record in image_records})
    print(
        f"Selected {len(image_records)} images across {len(selected_clips)} clips "
        f"into {dest_root}"
    )

    extracted, skipped = _extract_images(
        source_zip=source_zip,
        image_records=image_records,
        force=args.force,
    )
    annotation_rows = _write_annotations_csv(
        dest_root=dest_root,
        image_records=image_records,
        annotations=coco.get("annotations", []),
        categories=coco.get("categories", []),
    )

    print(
        f"Organized {DATASET_NAME}: extracted {extracted} images, "
        f"skipped {skipped} existing images, wrote {annotation_rows} annotation rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
