from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.dataset_constants import (
    AAU_ZEBRAFISH_REID,
    DATASET_SPECS,
    DEEP_VISION_FISH,
    KAKADU_FISHAI,
    LIAO_LAB_VIDEOS,
    MIT_RIVER_HERRING,
    NOAA_PUGET_SOUND_NEARSHORE_FISH,
    THREE_D_ZEF20,
    get_dataset_spec,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DATA_ROOT = repo_root() / "data"
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"
TRAINING_ROOT = DATA_ROOT / "training"
TRAINING_MANIFESTS_ROOT = TRAINING_ROOT / "manifests"
GENERATIVE_ROOT = DATA_ROOT / "generative"

AVAILABLE_TRAINING_MANIFEST = (
    TRAINING_MANIFESTS_ROOT / "domain_general_fish_available.json"
)
AVAILABLE_TRAINING_ROOT = TRAINING_ROOT / "domain-general-fish-available-yolo"


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    role: str
    raw_root: Path
    interim_root: Path | None
    processed_root: Path | None
    generative_root: Path | None
    training_source: Path | None


def dataset_paths(dataset_name: str) -> DatasetPaths:
    spec = get_dataset_spec(dataset_name)
    raw_root = RAW_ROOT / spec.name
    interim_root: Path | None = None
    processed_root: Path | None = None
    generative_root: Path | None = None
    training_source: Path | None = None

    if spec.name in {AAU_ZEBRAFISH_REID, MIT_RIVER_HERRING}:
        interim_root = INTERIM_ROOT / spec.name
        training_source = interim_root
    elif spec.name in {
        DEEP_VISION_FISH,
        KAKADU_FISHAI,
        NOAA_PUGET_SOUND_NEARSHORE_FISH,
        THREE_D_ZEF20,
    }:
        processed_root = PROCESSED_ROOT / f"{spec.name}-yolo"
        training_source = processed_root
    elif spec.name == LIAO_LAB_VIDEOS:
        generative_root = GENERATIVE_ROOT / spec.name

    return DatasetPaths(
        name=spec.name,
        role=spec.role,
        raw_root=raw_root,
        interim_root=interim_root,
        processed_root=processed_root,
        generative_root=generative_root,
        training_source=training_source,
    )


def ensure_data_layout() -> list[Path]:
    roots = [
        RAW_ROOT,
        INTERIM_ROOT,
        PROCESSED_ROOT,
        TRAINING_ROOT,
        TRAINING_MANIFESTS_ROOT,
        GENERATIVE_ROOT,
    ]
    created: list[Path] = []
    for root in roots:
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
            created.append(root)

    for spec in DATASET_SPECS:
        paths = dataset_paths(spec.name)
        if not paths.raw_root.exists():
            paths.raw_root.mkdir(parents=True, exist_ok=True)
            created.append(paths.raw_root)
        if paths.interim_root is not None and not paths.interim_root.exists():
            paths.interim_root.mkdir(parents=True, exist_ok=True)
            created.append(paths.interim_root)
        if paths.generative_root is not None and not paths.generative_root.exists():
            paths.generative_root.mkdir(parents=True, exist_ok=True)
            created.append(paths.generative_root)
    return created


def is_training_source_ready(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    if path.is_file() and path.suffix.lower() == ".json":
        return True
    if (path / "data.yaml").exists():
        return True
    if (path / "annotations.csv").exists():
        return True
    return False


def available_training_sources() -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for spec in DATASET_SPECS:
        if spec.role != "training":
            continue
        paths = dataset_paths(spec.name)
        if is_training_source_ready(paths.training_source):
            sources.append((spec.name, paths.training_source))
    return sources


def _relative_path(from_dir: Path, target: Path) -> str:
    return Path(
        os.path.relpath(Path(target).resolve(), start=Path(from_dir).resolve())
    ).as_posix()


def build_available_training_manifest(
    manifest_path: Path = AVAILABLE_TRAINING_MANIFEST,
    dataset_names: Iterable[str] | None = None,
) -> Path:
    manifest_path = manifest_path.resolve()
    ensure_data_layout()

    allowed = {get_dataset_spec(name).name for name in dataset_names} if dataset_names else None
    sources = [
        (name, path)
        for name, path in available_training_sources()
        if allowed is None or name in allowed
    ]
    if not sources:
        raise FileNotFoundError("No training datasets are ready yet.")

    source_specs = []
    for name, path in sources:
        source_specs.append(
            {
                "name": name,
                "path": _relative_path(manifest_path.parent, path),
            }
        )

    payload = {
        "name": "domain-general-fish-available-training",
        "single_class_name": "fish",
        "sources": source_specs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path
