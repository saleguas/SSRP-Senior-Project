#!/usr/bin/env python3
"""
Dataset registry and lightweight loaders for the project data layout.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Annotation:
    object_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    flags: Tuple[int, int, int, int]
    buffer: int


@dataclass(frozen=True)
class FrameRecord:
    dataset: str
    video: str
    frame: str
    frame_index: Optional[int]
    image_path: Path
    annotations: List[Annotation]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root(base_dir: Optional[Path] = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).resolve()
    return repo_root() / "data"


def _stage_dir(stage: str, base_dir: Optional[Path] = None) -> Path:
    stage_norm = stage.strip().lower()
    return data_root(base_dir) / stage_norm


def list_datasets(stage: str = "interim", base_dir: Optional[Path] = None) -> List[str]:
    root = _stage_dir(stage, base_dir)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_videos(
    dataset: str, stage: str = "interim", base_dir: Optional[Path] = None
) -> List[str]:
    dataset_dir = _stage_dir(stage, base_dir) / dataset
    if not dataset_dir.exists():
        return []
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])


def list_datasets_with_videos(
    stage: str = "interim", base_dir: Optional[Path] = None
) -> Dict[str, List[str]]:
    datasets = list_datasets(stage=stage, base_dir=base_dir)
    return {name: list_videos(name, stage=stage, base_dir=base_dir) for name in datasets}


def _parse_flags(value: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    nums = [int(p) for p in parts] if parts else []
    while len(nums) < 4:
        nums.append(0)
    return tuple(nums[:4])  # type: ignore[return-value]


def _parse_frame_index(filename: str) -> Optional[int]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    return int(match.group(1)) if match else None


@lru_cache(maxsize=8)
def load_annotations_csv(csv_path: Path) -> Dict[str, List[Annotation]]:
    annotations: Dict[str, List[Annotation]] = {}
    if not csv_path.exists():
        return annotations

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, [])
        header = [h.strip() for h in header]
        idx = {name: i for i, name in enumerate(header) if name}

        def get_value(row: List[str], key: str) -> str:
            col = idx.get(key)
            if col is None or col >= len(row):
                return ""
            return row[col].strip()

        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue

            filename = get_value(row, "Filename")
            if not filename:
                continue

            annotation = Annotation(
                object_id=int(get_value(row, "Object ID") or 0),
                label=get_value(row, "Annotation tag"),
                bbox=(
                    int(get_value(row, "Upper left corner X") or 0),
                    int(get_value(row, "Upper left corner Y") or 0),
                    int(get_value(row, "Lower right corner X") or 0),
                    int(get_value(row, "Lower right corner Y") or 0),
                ),
                flags=_parse_flags(get_value(row, "Right,Turning,Occlusion,Glitch")),
                buffer=int(get_value(row, "Buffer") or 0),
            )
            annotations.setdefault(filename, []).append(annotation)

    return annotations


def _resolve_video_dir(dataset_dir: Path, video: str) -> Path:
    video_dir = dataset_dir / video
    if video_dir.exists():
        return video_dir
    for candidate in dataset_dir.iterdir():
        if candidate.is_dir() and candidate.name.lower() == video.lower():
            return candidate
    return video_dir


def _iter_image_files(folder: Path) -> Iterator[Path]:
    for image_path in sorted(folder.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def iter_frames(
    dataset: str,
    video: str,
    stage: str = "interim",
    base_dir: Optional[Path] = None,
    include_empty: bool = True,
) -> Iterator[FrameRecord]:
    dataset_dir = _stage_dir(stage, base_dir) / dataset
    video_dir = _resolve_video_dir(dataset_dir, video)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    annotations_path = dataset_dir / "annotations.csv"
    annotations_map = load_annotations_csv(annotations_path)

    for image_path in _iter_image_files(video_dir):
        frame_name = image_path.name
        ann = annotations_map.get(frame_name, [])
        if ann or include_empty:
            yield FrameRecord(
                dataset=dataset,
                video=video_dir.name,
                frame=frame_name,
                frame_index=_parse_frame_index(frame_name),
                image_path=image_path,
                annotations=ann,
            )


def iter_dataset_frames(
    dataset: str,
    stage: str = "interim",
    base_dir: Optional[Path] = None,
    include_empty: bool = True,
) -> Iterable[FrameRecord]:
    for video in list_videos(dataset, stage=stage, base_dir=base_dir):
        yield from iter_frames(
            dataset=dataset,
            video=video,
            stage=stage,
            base_dir=base_dir,
            include_empty=include_empty,
        )


if __name__ == "__main__":
    datasets = list_datasets_with_videos()
    if not datasets:
        print("No datasets found in data/interim.")
    else:
        for name, videos in datasets.items():
            print(f"{name}: {', '.join(videos)}")
