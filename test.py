from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

from src.pipeline.utils import configure_ultralytics, repo_root, require_cuda


def _newest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def resolve_weights() -> Path:
    models_root = repo_root() / "models"
    preferred = [
        models_root / "latest.pt",
        models_root / "yolo_fish" / "weights" / "best.pt",
        models_root / "yolo_fish" / "weights" / "last.pt",
    ]
    for path in preferred:
        if path.exists():
            return path

    best_candidate = _newest(list(models_root.glob("**/weights/best.pt")))
    if best_candidate:
        return best_candidate

    last_candidate = _newest(list(models_root.glob("**/weights/last.pt")))
    if last_candidate:
        return last_candidate

    base_names = {"yolov8n.pt", "yolo11n.pt"}
    pt_candidates = [
        p
        for p in models_root.glob("**/*.pt")
        if p.name.lower() not in base_names
    ]
    newest_pt = _newest(pt_candidates)
    if newest_pt:
        return newest_pt

    raise FileNotFoundError("No trained weights found in models/.")


def main() -> None:
    require_cuda()
    root = repo_root()
    configure_ultralytics(root / "models")

    frames_dir = root / "data" / "interim" / "aau-zebrafish-reid" / "vid1"
    image_paths = sorted(frames_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    image_path = image_paths[0]
    weights_path = resolve_weights()

    print(f"Using weights: {weights_path}")
    print(f"Input frame: {image_path}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        conf=0.25,
        iou=0.5,
        device=0,
        verbose=False,
    )

    result = results[0]
    annotated = result.plot()

    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"sanity_{image_path.name}"
    cv2.imwrite(str(output_path), annotated)
    print(f"Wrote annotated frame: {output_path}")

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("Detections: 0")
        return

    print(f"Detections: {len(boxes)}")
    xyxy = boxes.xyxy.cpu().tolist()
    conf = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    for idx, (coords, score) in enumerate(zip(xyxy, conf), start=1):
        x1, y1, x2, y2 = [int(v) for v in coords]
        print(f"  {idx}: ({x1}, {y1}) -> ({x2}, {y2}) conf={score:.4f}")


if __name__ == "__main__":
    main()
