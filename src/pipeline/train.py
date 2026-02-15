from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict

from ultralytics import YOLO

from .dataset import prepare_yolo_dataset
from .utils import configure_ultralytics, ensure_dir, repo_root, require_cuda, write_json


def _auto_epochs(train_count: int) -> int:
    if train_count < 1000:
        return 50
    if train_count < 3000:
        return 30
    return 20


def _weights_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _ensure_base_weights(models_root: Path, name: str) -> Path:
    target = models_root / name
    if target.exists():
        return target

    repo_file = (models_root.parent / name).resolve()
    if repo_file.exists():
        shutil.move(str(repo_file), str(target))
        return target

    cwd = Path.cwd()
    models_root.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(models_root)
        YOLO(name)
    finally:
        os.chdir(cwd)

    if target.exists():
        return target

    raise FileNotFoundError(f"Unable to download base weights: {name}")


def train_detector(dataset_root: Path, output_weights: Path) -> Dict[str, str]:
    require_cuda()
    dataset = prepare_yolo_dataset(dataset_root)

    models_root = ensure_dir(repo_root() / "models")
    configure_ultralytics(models_root)
    run_dir = models_root / "yolo_fish"

    train_images = list((dataset.images_train).glob("*.png"))
    epochs = _auto_epochs(len(train_images))

    base_weights = _ensure_base_weights(models_root, "yolov8n.pt")
    _ensure_base_weights(models_root, "yolo11n.pt")
    model = YOLO(str(base_weights))
    model.train(
        data=str(dataset.yaml_path),
        imgsz=dataset.imgsz,
        epochs=epochs,
        batch=-1,
        device=0,
        patience=10,
        workers=0,
        save=True,
        save_period=1,
        plots=False,
        project=str(models_root),
        name=run_dir.name,
        exist_ok=True,
        verbose=True,
    )

    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    if not best_pt.exists() or not last_pt.exists():
        raise FileNotFoundError("Training completed but weights were not found.")

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, output_weights)

    output_last = _weights_suffix(output_weights, "last")
    shutil.copy2(last_pt, output_last)

    latest_pt = models_root / "latest.pt"
    shutil.copy2(best_pt, latest_pt)

    metadata = {
        "dataset_root": str(dataset.dataset_root),
        "yolo_root": str(dataset.yolo_root),
        "run_dir": str(run_dir),
        "best": str(best_pt),
        "last": str(last_pt),
        "output_best": str(output_weights),
        "output_last": str(output_last),
    }
    write_json(models_root / "latest.json", metadata)

    return metadata
