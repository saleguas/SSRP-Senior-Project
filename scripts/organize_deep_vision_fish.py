#!/usr/bin/env python3
"""
Build a YOLO-ready dataset root for the Deep Vision fish dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.utils import is_image_file, link_or_copy, write_json

DATASET_NAME = "deep-vision-fish"
ZIP_NAME = "fishDatasetSimulationAlgorithm.zip"
DEFAULT_SOURCE = f"data/raw/{DATASET_NAME}"
DEFAULT_DEST = f"data/processed/{DATASET_NAME}-yolo"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset root for the Deep Vision fish dataset."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Raw dataset directory.")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Output YOLO dataset root.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the destination even if it already exists.",
    )
    return parser.parse_args()


def _ensure_extracted(source_dir: Path) -> Path:
    extracted_root = source_dir / "unzipped" / "fish_dataset"
    if extracted_root.exists():
        return extracted_root

    zip_path = source_dir / ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"Deep Vision zip not found: {zip_path}")

    extracted_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(extracted_root.parent))
    if not extracted_root.exists():
        raise FileNotFoundError(f"Deep Vision extraction failed: {extracted_root}")
    return extracted_root


def _normalize_rel_path(value: str) -> Path:
    return Path(value.strip().lstrip("/\\"))


def _read_rows(csv_path: Path) -> List[Tuple[Path, Tuple[int, int, int, int]]]:
    rows: List[Tuple[Path, Tuple[int, int, int, int]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5:
                continue
            rel_path = _normalize_rel_path(row[0])
            bbox = tuple(int(float(value)) for value in row[1:5])
            rows.append((rel_path, bbox))  # type: ignore[arg-type]
    return rows


def _relative_name(rel_path: Path) -> str:
    parts = [part.replace(" ", "-") for part in rel_path.parts]
    return "__".join(parts)


def _build_basename_index(dataset_root: Path) -> Tuple[Dict[str, List[Path]], List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    all_images: List[Path] = []
    for path in dataset_root.rglob("*"):
        if is_image_file(path):
            index[path.name].append(path)
            all_images.append(path)
    return index, all_images


def _resolve_image_path(
    dataset_root: Path,
    basename_index: Dict[str, List[Path]],
    all_images: List[Path],
    rel_path: Path,
) -> Path:
    source_image = dataset_root / rel_path
    if source_image.exists():
        return source_image

    candidates = basename_index.get(rel_path.name, [])
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1 and len(rel_path.parts) >= 2:
        year, split = rel_path.parts[0], rel_path.parts[1]
        narrowed = [path for path in candidates if year in path.parts and split in path.parts]
        if len(narrowed) == 1:
            return narrowed[0]

    suffix_matches = [path for path in all_images if path.name.endswith(rel_path.name)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    if len(suffix_matches) > 1 and len(rel_path.parts) >= 2:
        year, split = rel_path.parts[0], rel_path.parts[1]
        narrowed = [path for path in suffix_matches if year in path.parts and split in path.parts]
        if len(narrowed) == 1:
            return narrowed[0]

    raise FileNotFoundError(f"Missing Deep Vision image: {source_image}")


def _yolo_lines(boxes: Iterable[Tuple[int, int, int, int]], width: int, height: int) -> List[str]:
    lines: List[str] = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
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


def _prepare_split_maps(dataset_root: Path) -> Tuple[Dict[Path, List[Tuple[int, int, int, int]]], Dict[Path, List[Tuple[int, int, int, int]]]]:
    train_csvs = [
        dataset_root / "2017" / "train" / "source-train2017-annotations.csv",
        dataset_root / "2018" / "train" / "source-train2018-annotations.csv",
        dataset_root / "2017" / "test" / "test_2017_annotations.csv",
        dataset_root / "2018" / "test" / "test_2018_annotations.csv",
    ]
    val_csv = dataset_root / "val_annotations.csv"

    val_rows = _read_rows(val_csv)
    val_paths = {path for path, _ in val_rows}

    train_map: Dict[Path, List[Tuple[int, int, int, int]]] = defaultdict(list)
    val_map: Dict[Path, List[Tuple[int, int, int, int]]] = defaultdict(list)

    for rel_path, bbox in val_rows:
        val_map[rel_path].append(bbox)

    for csv_path in train_csvs:
        for rel_path, bbox in _read_rows(csv_path):
            if rel_path in val_paths:
                continue
            train_map[rel_path].append(bbox)

    return dict(train_map), dict(val_map)


def _build_output(dataset_root: Path, output_root: Path, force: bool) -> Dict[str, int]:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Destination already exists: {output_root}. Use --force to rebuild.")
        shutil.rmtree(output_root, ignore_errors=True)

    train_map, val_map = _prepare_split_maps(dataset_root)
    basename_index, all_images = _build_basename_index(dataset_root)
    counts = {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}

    for split_name, image_map in (("train", train_map), ("val", val_map)):
        images_dir = output_root / "images" / split_name
        labels_dir = output_root / "labels" / split_name
        for rel_path, boxes in sorted(image_map.items()):
            source_image = _resolve_image_path(dataset_root, basename_index, all_images, rel_path)

            with Image.open(source_image) as img:
                width, height = img.size

            linked_name = _relative_name(rel_path)
            dest_image = images_dir / linked_name
            link_or_copy(source_image, dest_image)

            label_path = labels_dir / f"{Path(linked_name).stem}.txt"
            _write_label(label_path, _yolo_lines(boxes, width, height))

            counts[f"{split_name}_images"] += 1
            counts[f"{split_name}_boxes"] += len(boxes)

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
    write_json(
        output_root / "build_info.json",
        {
            "name": DATASET_NAME,
            "source_root": str(dataset_root),
            "counts": counts,
        },
    )
    return counts


def main() -> int:
    args = _parse_args()
    source_dir = _resolve(args.source)
    output_root = _resolve(args.dest)
    dataset_root = _ensure_extracted(source_dir)
    counts = _build_output(dataset_root, output_root, force=args.force)
    print(f"Built {DATASET_NAME} YOLO dataset at {output_root}")
    print(
        f"train images={counts['train_images']} val images={counts['val_images']} "
        f"train boxes={counts['train_boxes']} val boxes={counts['val_boxes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
