from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image

from src.data_registry import load_annotations_csv
from .utils import link_or_copy, list_image_files, repo_root


@dataclass(frozen=True)
class YoloDataset:
    dataset_root: Path
    yolo_root: Path
    yaml_path: Path
    images_train: Path
    images_val: Path
    labels_train: Path
    labels_val: Path
    imgsz: int


def _auto_imgsz(sample_image: Path) -> int:
    with Image.open(sample_image) as img:
        width, height = img.size
    max_dim = max(width, height)
    stride = 32
    size = (max_dim // stride) * stride
    size = max(640, min(1280, size))
    return int(size)


def _list_videos(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def _list_frames(video_dir: Path) -> List[Path]:
    return list_image_files(video_dir)


def _split_frames(videos: List[Path]) -> Tuple[List[Path], List[Path]]:
    if len(videos) >= 5:
        val_video_count = max(1, int(round(len(videos) * 0.2)))
        train_videos = videos[:-val_video_count]
        val_videos = videos[-val_video_count:]
        train_frames = [frame for v in train_videos for frame in _list_frames(v)]
        val_frames = [frame for v in val_videos for frame in _list_frames(v)]
        return train_frames, val_frames

    if len(videos) >= 2:
        train_videos = videos[:-1]
        val_videos = [videos[-1]]
        train_frames = [frame for v in train_videos for frame in _list_frames(v)]
        val_frames = [frame for v in val_videos for frame in _list_frames(v)]
        return train_frames, val_frames

    if not videos:
        return [], []

    frames = _list_frames(videos[0])
    split_index = int(len(frames) * 0.8)
    return frames[:split_index], frames[split_index:]


def _yolo_label_lines(
    annotations: List[Tuple[int, int, int, int]],
    width: int,
    height: int,
) -> List[str]:
    lines: List[str] = []
    for x1, y1, x2, y2 in annotations:
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


def _write_label_file(label_path: Path, lines: List[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _first_image(folder: Path) -> Optional[Path]:
    images = list_image_files(folder)
    return images[0] if images else None


def _yolo_dataset_from_root(
    dataset_root: Path,
    yolo_root: Path,
) -> YoloDataset:
    images_train = yolo_root / "images" / "train"
    images_val = yolo_root / "images" / "val"
    labels_train = yolo_root / "labels" / "train"
    labels_val = yolo_root / "labels" / "val"
    yaml_path = yolo_root / "data.yaml"
    sample_image = _first_image(images_train) or _first_image(images_val)
    imgsz = _auto_imgsz(sample_image) if sample_image else 640
    return YoloDataset(
        dataset_root=dataset_root,
        yolo_root=yolo_root,
        yaml_path=yaml_path,
        images_train=images_train,
        images_val=images_val,
        labels_train=labels_train,
        labels_val=labels_val,
        imgsz=imgsz,
    )


def _existing_yolo_dataset(
    yolo_root: Path,
    dataset_root: Optional[Path] = None,
) -> Optional[YoloDataset]:
    yolo_root = yolo_root.resolve()
    yaml_path = yolo_root / "data.yaml"
    if not yaml_path.exists():
        return None
    return _yolo_dataset_from_root(dataset_root or yolo_root, yolo_root)


def _sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return safe.strip("-") or "dataset"


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    return path


def _combine_split(
    split_name: str,
    sources: List[Tuple[str, YoloDataset]],
    dest_images_dir: Path,
    dest_labels_dir: Path,
) -> None:
    for prefix, dataset in sources:
        source_images_dir = dataset.images_train if split_name == "train" else dataset.images_val
        source_labels_dir = dataset.labels_train if split_name == "train" else dataset.labels_val
        for image_path in list_image_files(source_images_dir):
            linked_name = f"{prefix}__{image_path.name}"
            linked_stem = Path(linked_name).stem
            image_dst = dest_images_dir / linked_name
            label_src = source_labels_dir / f"{image_path.stem}.txt"
            label_dst = dest_labels_dir / f"{linked_stem}.txt"

            link_or_copy(image_path, image_dst)
            if label_src.exists():
                link_or_copy(label_src, label_dst)
            else:
                _write_label_file(label_dst, [])


def _prepare_manifest_yolo_dataset(
    manifest_path: Path,
    output_root: Optional[Path] = None,
) -> YoloDataset:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {manifest_path}")

    source_specs = manifest.get("sources")
    if not isinstance(source_specs, list) or not source_specs:
        raise ValueError(f"Dataset manifest has no sources: {manifest_path}")

    output_name = _sanitize_name(str(manifest.get("name") or manifest_path.stem))
    class_name = str(manifest.get("single_class_name") or "fish")
    if output_root is None:
        output_root = repo_root() / "data" / "processed" / f"{output_name}-yolo"
    output_root = output_root.resolve()
    build_info_path = output_root / "build_info.json"
    expected_build_info: dict[str, Any] = {
        "name": output_name,
        "single_class_name": class_name,
        "sources": source_specs,
    }

    if output_root.exists():
        if build_info_path.exists():
            existing_build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
            if existing_build_info == expected_build_info:
                existing = _existing_yolo_dataset(output_root, dataset_root=manifest_path)
                if existing is not None:
                    return existing
        shutil.rmtree(output_root, ignore_errors=True)

    prepared_sources: List[Tuple[str, YoloDataset]] = []
    for spec in source_specs:
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid source entry in {manifest_path}: {spec!r}")
        path_value = spec.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"Source entry missing path in {manifest_path}: {spec!r}")

        source_root = _resolve_manifest_path(manifest_path, path_value)
        source_name = _sanitize_name(str(spec.get("name") or source_root.name))
        prepared_sources.append((source_name, prepare_yolo_dataset(source_root)))

    images_train = output_root / "images" / "train"
    images_val = output_root / "images" / "val"
    labels_train = output_root / "labels" / "train"
    labels_val = output_root / "labels" / "val"
    yaml_path = output_root / "data.yaml"

    _combine_split("train", prepared_sources, images_train, labels_train)
    _combine_split("val", prepared_sources, images_val, labels_val)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    build_info_path.write_text(
        json.dumps(expected_build_info, indent=2) + "\n",
        encoding="utf-8",
    )

    return _yolo_dataset_from_root(manifest_path, output_root)


def prepare_yolo_dataset(dataset_root: Path, output_root: Optional[Path] = None) -> YoloDataset:
    dataset_root = dataset_root.resolve()
    if dataset_root.is_file() and dataset_root.suffix.lower() == ".json":
        return _prepare_manifest_yolo_dataset(dataset_root, output_root=output_root)

    existing_yolo = _existing_yolo_dataset(dataset_root)
    if existing_yolo is not None:
        return existing_yolo

    annotations_csv = dataset_root / "annotations.csv"
    if not annotations_csv.exists():
        raise FileNotFoundError(f"annotations.csv not found in {dataset_root}")
    if output_root is None:
        output_root = repo_root() / "data" / "processed" / f"{dataset_root.name}-yolo"
    output_root = output_root.resolve()

    images_train = output_root / "images" / "train"
    images_val = output_root / "images" / "val"
    labels_train = output_root / "labels" / "train"
    labels_val = output_root / "labels" / "val"
    yaml_path = output_root / "data.yaml"

    videos = _list_videos(dataset_root)
    train_frames, val_frames = _split_frames(videos)

    sample_image = (train_frames or val_frames)[0] if (train_frames or val_frames) else None
    imgsz = _auto_imgsz(sample_image) if sample_image else 640

    if yaml_path.exists():
        return YoloDataset(
            dataset_root=dataset_root,
            yolo_root=output_root,
            yaml_path=yaml_path,
            images_train=images_train,
            images_val=images_val,
            labels_train=labels_train,
            labels_val=labels_val,
            imgsz=imgsz,
        )

    annotations_map = load_annotations_csv(annotations_csv)

    for split_frames, images_dir, labels_dir in (
        (train_frames, images_train, labels_train),
        (val_frames, images_val, labels_val),
    ):
        for image_path in split_frames:
            image_path = image_path.resolve()
            if not image_path.exists():
                continue

            with Image.open(image_path) as img:
                width, height = img.size

            annotation_list = annotations_map.get(image_path.name, [])
            boxes = [ann.bbox for ann in annotation_list]
            label_lines = _yolo_label_lines(boxes, width, height)

            label_path = labels_dir / (image_path.stem + ".txt")
            _write_label_file(label_path, label_lines)

            dest_image = images_dir / image_path.name
            link_or_copy(image_path, dest_image)

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

    return YoloDataset(
        dataset_root=dataset_root,
        yolo_root=output_root,
        yaml_path=yaml_path,
        images_train=images_train,
        images_val=images_val,
        labels_train=labels_train,
        labels_val=labels_val,
        imgsz=imgsz,
    )
