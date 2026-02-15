from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from .dataset import prepare_yolo_dataset
from .utils import require_cuda, write_json


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_detector(dataset_root: Path, weights_path: Path, output_json: Path) -> Dict[str, Any]:
    require_cuda()
    dataset = prepare_yolo_dataset(dataset_root)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(dataset.yaml_path),
        imgsz=dataset.imgsz,
        device=0,
        plots=False,
        save=False,
        verbose=False,
    )

    results: Dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            results[key] = _float_or_none(value) if value is not None else None
    else:
        if hasattr(metrics, "box"):
            box = metrics.box
            results["box/map"] = _float_or_none(getattr(box, "map", None))
            results["box/map50"] = _float_or_none(getattr(box, "map50", None))
            results["box/map75"] = _float_or_none(getattr(box, "map75", None))
        results["fitness"] = _float_or_none(getattr(metrics, "fitness", None))

    speed = getattr(metrics, "speed", None)
    if speed is not None:
        results["speed"] = speed

    write_json(output_json, results)
    return results
