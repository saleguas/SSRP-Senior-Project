#!/usr/bin/env python3
"""
Build a YOLO-ready dataset root for the Kakadu FishAI training dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.utils import link_or_copy, write_json

DATASET_NAME = "kakadu-fishai"
ZIP_NAME = "202210-KakaduFishAI-TrainingData.zip"
ANNOTATIONS_NAME = "KakaduFishAI_boundingbox.json"
DEFAULT_SOURCE = f"data/raw/{DATASET_NAME}"
DEFAULT_DEST = f"data/processed/{DATASET_NAME}-yolo"
VAL_MODULUS = 10


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset root for the Kakadu FishAI dataset."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Raw dataset directory.")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Output YOLO dataset root.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the destination even if it already exists.",
    )
    return parser.parse_args()


def _find_7z() -> Path:
    candidates = [
        shutil.which("7z"),
        shutil.which("7z.exe"),
        str(Path("C:/Program Files/7-Zip/7z.exe")),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "7-Zip was not found. Install 7-Zip or extract the Kakadu zip manually."
    )


def _ensure_extracted(source_dir: Path) -> Path:
    extracted_root = source_dir / "unzipped"
    annotations_path = extracted_root / ANNOTATIONS_NAME
    if annotations_path.exists():
        return extracted_root

    zip_path = source_dir / ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"Kakadu zip not found: {zip_path}")

    extracted_root.mkdir(parents=True, exist_ok=True)
    seven_zip = _find_7z()
    subprocess.run(
        [str(seven_zip), "x", "-y", f"-o{extracted_root}", str(zip_path)],
        check=True,
    )
    if not annotations_path.exists():
        raise FileNotFoundError(f"Kakadu extraction failed: {annotations_path}")
    return extracted_root


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


def _split_name(image_id: int) -> str:
    return "val" if image_id % VAL_MODULUS == 0 else "train"


def _build_output(dataset_root: Path, output_root: Path, force: bool) -> Dict[str, int]:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Destination already exists: {output_root}. Use --force to rebuild.")
        shutil.rmtree(output_root, ignore_errors=True)

    payload = json.loads((dataset_root / ANNOTATIONS_NAME).read_text(encoding="utf-8"))
    images = {image["id"]: image for image in payload["images"]}
    annotations_by_image: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        x, y, w, h = annotation["bbox"]
        annotations_by_image[int(annotation["image_id"])].append(
            (int(x), int(y), int(x + w), int(y + h))
        )

    counts = {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}
    for image_id, image in sorted(images.items()):
        split_name = _split_name(int(image_id))
        image_name = str(image["file_name"])
        source_image = dataset_root / image_name
        if not source_image.exists():
            raise FileNotFoundError(f"Missing Kakadu image: {source_image}")

        images_dir = output_root / "images" / split_name
        labels_dir = output_root / "labels" / split_name
        dest_image = images_dir / image_name
        link_or_copy(source_image, dest_image)

        width = int(image["width"])
        height = int(image["height"])
        boxes = annotations_by_image.get(int(image_id), [])
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
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
            "val_modulus": VAL_MODULUS,
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
