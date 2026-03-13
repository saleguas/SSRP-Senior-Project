#!/usr/bin/env python3
"""
Build a YOLO-ready dataset root for the NOAA Puget Sound Nearshore Fish dataset.

Policy:
- keep `fish` boxes as positives
- keep `empty` and `crab` images as negatives with empty label files
- drop `fish_or_crab` and `unknown` images entirely because their labels are
  ambiguous for a one-class fish detector
- split train/val by `location` to reduce near-duplicate frame leakage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.utils import is_image_file, link_or_copy, write_json

DATASET_NAME = "noaa-puget-sound-nearshore-fish"
DEFAULT_SOURCE = f"data/raw/{DATASET_NAME}"
DEFAULT_DEST = f"data/processed/{DATASET_NAME}-yolo"
IMAGES_ZIP_NAME = "noaa_estuary_fish-images.zip"
ANNOTATIONS_ZIP_NAME = "noaa_estuary_fish-annotations-2023.08.19.zip"
ANNOTATIONS_JSON_NAME = "noaa_estuary_fish-2023.08.19.json"
POSITIVE_CATEGORY = "fish"
NEGATIVE_CATEGORIES = {"empty", "crab"}
EXCLUDED_CATEGORIES = {"fish_or_crab", "unknown"}
VAL_LOCATION_MODULUS = 5


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset root for the NOAA Puget Sound Nearshore Fish dataset."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Raw dataset directory.")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Output YOLO dataset root.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the destination even if it already exists.",
    )
    return parser.parse_args()


def _first_image(images_root: Path) -> Path | None:
    for path in images_root.rglob("*"):
        if is_image_file(path):
            return path
    return None


def _ensure_annotations_extracted(source_dir: Path) -> Path:
    annotations_root = source_dir / "unzipped" / "annotations"
    annotations_path = annotations_root / ANNOTATIONS_JSON_NAME
    if annotations_path.exists():
        return annotations_path

    zip_path = source_dir / ANNOTATIONS_ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"NOAA annotations zip not found: {zip_path}")

    annotations_root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(annotations_root))
    if not annotations_path.exists():
        raise FileNotFoundError(f"NOAA annotation extraction failed: {annotations_path}")
    return annotations_path


def _ensure_images_extracted(source_dir: Path) -> Path:
    images_root = source_dir / "unzipped" / "images"
    if _first_image(images_root) is not None:
        return images_root

    zip_path = source_dir / IMAGES_ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"NOAA images zip not found: {zip_path}")

    images_root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(images_root))
    if _first_image(images_root) is None:
        raise FileNotFoundError(f"NOAA image extraction failed under {images_root}")
    return images_root


def _load_payload(annotations_path: Path) -> dict:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {annotations_path}")
    return payload


def _category_name(annotation: dict, categories: Dict[int, str]) -> str:
    category_id = int(annotation["category_id"])
    return categories.get(category_id, str(category_id))


def _build_basename_index(images_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for path in images_root.rglob("*"):
        if is_image_file(path):
            index[path.name].append(path)
    return index


def _resolve_image_path(images_root: Path, basename_index: Dict[str, List[Path]], file_name: str) -> Path:
    direct = images_root / file_name
    if direct.exists():
        return direct

    candidates = basename_index.get(Path(file_name).name, [])
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(f"Missing NOAA image: {file_name}")
    raise FileNotFoundError(
        f"Multiple NOAA images matched {file_name}; expected one, found {len(candidates)}"
    )


def _select_val_locations(images: Iterable[dict]) -> Tuple[List[str], str]:
    locations = sorted({str(image["location"]) for image in images})
    if not locations:
        return ([], "none")

    selected = [
        location
        for location in locations
        if int(hashlib.md5(location.encode("utf-8")).hexdigest(), 16) % VAL_LOCATION_MODULUS == 0
    ]
    if 0 < len(selected) < len(locations):
        return (sorted(selected), f"md5_mod_{VAL_LOCATION_MODULUS}")

    val_count = max(1, int(round(len(locations) * 0.2)))
    return (locations[-val_count:], "sorted_tail_fallback")


def _fish_boxes(
    annotations: Iterable[dict],
    categories: Dict[int, str],
) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    for annotation in annotations:
        if _category_name(annotation, categories) != POSITIVE_CATEGORY:
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x, y, w, h = (float(value) for value in bbox[:4])
        boxes.append((x, y, x + w, y + h))
    return boxes


def _yolo_lines(
    boxes: Iterable[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> List[str]:
    lines: List[str] = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0.0, min(x1, width - 1))
        y1 = max(0.0, min(y1, height - 1))
        x2 = max(0.0, min(x2, width))
        y2 = max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        xc = (x1 + x2) / 2.0 / width
        yc = (y1 + y2) / 2.0 / height
        w = (x2 - x1) / float(width)
        h = (y2 - y1) / float(height)
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def _write_label(label_path: Path, lines: List[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_output(source_dir: Path, output_root: Path, force: bool) -> Dict[str, object]:
    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"Destination already exists: {output_root}. Use --force to rebuild."
            )
        shutil.rmtree(output_root, ignore_errors=True)

    annotations_path = _ensure_annotations_extracted(source_dir)
    images_root = _ensure_images_extracted(source_dir)
    payload = _load_payload(annotations_path)

    categories = {int(category["id"]): str(category["name"]) for category in payload["categories"]}
    images = {str(image["id"]): image for image in payload["images"]}
    annotations_by_image: Dict[str, List[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[str(annotation["image_id"])].append(annotation)

    basename_index = _build_basename_index(images_root)
    val_locations, split_policy = _select_val_locations(images.values())
    val_location_set = set(val_locations)

    counts = Counter()
    excluded_by_reason = Counter()
    for image_id, image in sorted(images.items(), key=lambda item: str(item[1]["file_name"])):
        annotations = annotations_by_image.get(image_id, [])
        labels = {_category_name(annotation, categories) for annotation in annotations}
        excluded_labels = sorted(label for label in labels if label in EXCLUDED_CATEGORIES)
        if excluded_labels:
            counts["excluded_images"] += 1
            for label in excluded_labels:
                excluded_by_reason[label] += 1
            continue

        unexpected = sorted(
            label
            for label in labels
            if label and label not in NEGATIVE_CATEGORIES and label != POSITIVE_CATEGORY
        )
        if unexpected:
            counts["excluded_images"] += 1
            for label in unexpected:
                excluded_by_reason[label] += 1
            continue

        split_name = "val" if str(image["location"]) in val_location_set else "train"
        source_image = _resolve_image_path(images_root, basename_index, str(image["file_name"]))
        with Image.open(source_image) as handle:
            width, height = handle.size

        fish_boxes = _fish_boxes(annotations, categories)
        label_lines = _yolo_lines(fish_boxes, width, height)

        dest_image = output_root / "images" / split_name / source_image.name
        label_path = output_root / "labels" / split_name / f"{source_image.stem}.txt"
        link_or_copy(source_image, dest_image)
        _write_label(label_path, label_lines)

        counts[f"{split_name}_images"] += 1
        counts[f"{split_name}_boxes"] += len(label_lines)
        if label_lines:
            counts[f"{split_name}_positive_images"] += 1
        else:
            counts[f"{split_name}_negative_images"] += 1

    yaml_path = output_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: fish",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_info = {
        "name": DATASET_NAME,
        "source_root": str(source_dir),
        "images_root": str(images_root),
        "annotations_path": str(annotations_path),
        "positive_category": POSITIVE_CATEGORY,
        "negative_categories": sorted(NEGATIVE_CATEGORIES),
        "excluded_categories": sorted(EXCLUDED_CATEGORIES),
        "split_policy": split_policy,
        "val_location_modulus": VAL_LOCATION_MODULUS,
        "val_locations": val_locations,
        "counts": dict(counts),
        "excluded_by_reason": dict(excluded_by_reason),
    }
    write_json(output_root / "build_info.json", build_info)
    return build_info


def main() -> int:
    args = _parse_args()
    source_dir = _resolve(args.source)
    output_root = _resolve(args.dest)
    build_info = _build_output(source_dir, output_root, force=args.force)
    counts = build_info["counts"]
    print(f"Built {DATASET_NAME} YOLO dataset at {output_root}")
    print(
        f"train images={counts.get('train_images', 0)} val images={counts.get('val_images', 0)} "
        f"train boxes={counts.get('train_boxes', 0)} val boxes={counts.get('val_boxes', 0)} "
        f"excluded images={counts.get('excluded_images', 0)}"
    )
    print(
        f"train negatives={counts.get('train_negative_images', 0)} "
        f"val negatives={counts.get('val_negative_images', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
