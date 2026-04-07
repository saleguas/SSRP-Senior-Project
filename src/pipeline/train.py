from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from .dataset import prepare_yolo_dataset
from .utils import (
    configure_ultralytics,
    ensure_dir,
    list_image_files,
    repo_root,
    require_cuda,
    tee_output,
    write_json,
)

DEFAULT_BASE_MODEL = "yolo11s.pt"


def _auto_epochs(train_count: int) -> int:
    if train_count < 1000:
        return 50
    if train_count < 3000:
        return 30
    return 20


def _auto_workers() -> int:
    logical_cores = os.cpu_count() or 8
    return max(2, min(8, logical_cores // 4))


def _recommended_imgsz(dataset_imgsz: int, train_count: int) -> int:
    if train_count >= 100000:
        return min(dataset_imgsz, 960)
    if train_count >= 25000:
        return min(dataset_imgsz, 1024)
    return dataset_imgsz


def _weights_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _run_name_from_output(path: Path) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in path.stem)
    return safe.strip("-.") or "train"


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


def _resolve_model_reference(models_root: Path, model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    if model_name.lower().endswith(".pt"):
        return str(_ensure_base_weights(models_root, Path(model_name).name))

    return model_name


def describe_detector(model_name: str = DEFAULT_BASE_MODEL, classes: int = 1) -> Dict[str, Any]:
    models_root = ensure_dir(repo_root() / "models")
    configure_ultralytics(models_root)

    model_ref = _resolve_model_reference(models_root, model_name)
    pretrained = YOLO(model_ref)
    model = DetectionModel(cfg=pretrained.model.yaml, nc=classes, verbose=False)

    def format_stage(stage: list[Any]) -> str:
        source, repeats, module, args = stage
        return f"from={source}, repeats={repeats}, module={module}, args={args}"

    return {
        "model": model_ref,
        "classes": classes,
        "stage_count": len(model.model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "stride": [int(value) for value in model.stride.tolist()],
        "depth_multiple": pretrained.model.yaml.get("depth_multiple"),
        "width_multiple": pretrained.model.yaml.get("width_multiple"),
        "backbone": [format_stage(stage) for stage in pretrained.model.yaml.get("backbone", [])],
        "head": [format_stage(stage) for stage in pretrained.model.yaml.get("head", [])],
    }


def train_detector(
    dataset_root: Path,
    output_weights: Path,
    epochs: Optional[int] = None,
    log_path: Optional[Path] = None,
    imgsz: Optional[int] = None,
    workers: Optional[int] = None,
    batch: Optional[int] = None,
    deterministic: bool = False,
    model_name: str = DEFAULT_BASE_MODEL,
) -> Dict[str, str]:
    require_cuda()
    dataset = prepare_yolo_dataset(dataset_root)

    models_root = ensure_dir(repo_root() / "models")
    configure_ultralytics(models_root)
    runs_root = ensure_dir(models_root / "runs")
    run_name = _run_name_from_output(output_weights)
    run_dir = runs_root / run_name

    train_images = list_image_files(dataset.images_train)
    train_epochs = epochs if epochs is not None else _auto_epochs(len(train_images))
    train_imgsz = imgsz if imgsz is not None else _recommended_imgsz(dataset.imgsz, len(train_images))
    train_workers = workers if workers is not None else _auto_workers()
    train_batch = batch if batch is not None else -1

    base_weights = _resolve_model_reference(models_root, model_name)
    model = YOLO(base_weights)
    if log_path is None:
        log_path = output_weights.with_suffix(".train.log")

    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER

    with tee_output(log_path, loggers=(ULTRALYTICS_LOGGER,)):
        print(f"Training dataset: {dataset.dataset_root}")
        print(f"YOLO dataset: {dataset.yolo_root}")
        print(f"Run directory: {run_dir}")
        print(f"Output weights: {output_weights}")
        print(f"Model: {base_weights}")
        print(f"Epochs: {train_epochs}")
        print(f"Image size: {train_imgsz}")
        print(f"Batch: {train_batch}")
        print(f"Workers: {train_workers}")
        print(f"Deterministic: {deterministic}")
        model.train(
            data=str(dataset.yaml_path),
            imgsz=train_imgsz,
            epochs=train_epochs,
            batch=train_batch,
            device=0,
            patience=10,
            workers=train_workers,
            save=True,
            save_period=1,
            plots=False,
            project=str(runs_root),
            name=run_name,
            exist_ok=True,
            verbose=True,
            deterministic=deterministic,
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
        "log_path": str(log_path),
        "epochs": str(train_epochs),
        "imgsz": str(train_imgsz),
        "batch": str(train_batch),
        "workers": str(train_workers),
        "deterministic": str(deterministic),
        "model": str(base_weights),
        "best": str(best_pt),
        "last": str(last_pt),
        "output_best": str(output_weights),
        "output_last": str(output_last),
    }
    write_json(models_root / "latest.json", metadata)

    return metadata
