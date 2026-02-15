from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from src.data_registry import load_annotations_csv
from .utils import link_or_copy, repo_root


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
    return sorted(video_dir.glob("*.png"))


def _split_frames(videos: List[Path]) -> Tuple[List[Path], List[Path]]:
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
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
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


def prepare_yolo_dataset(dataset_root: Path, output_root: Optional[Path] = None) -> YoloDataset:
    dataset_root = dataset_root.resolve()
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
