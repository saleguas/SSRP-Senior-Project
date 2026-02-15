from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from ultralytics import YOLO

from .utils import require_cuda


def _ensure_pngs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.png"))


def track_folder(images_dir: Path, output_csv: Path, weights_path: Path) -> None:
    require_cuda()

    images_dir = images_dir.resolve()
    image_paths = _ensure_pngs(images_dir)
    if not image_paths:
        subdirs = [p.name for p in images_dir.iterdir() if p.is_dir()]
        hint = f" Available subfolders: {', '.join(subdirs)}" if subdirs else ""
        raise FileNotFoundError(
            f"No PNG frames found in {images_dir}. Select a video folder.{hint}"
        )

    model = YOLO(str(weights_path))

    results = model.track(
        source=[str(p) for p in image_paths],
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.5,
        device=0,
        verbose=False,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "track_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "xc",
                "yc",
                "w",
                "h",
                "conf",
            ]
        )

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().tolist()
            conf = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(xyxy)
            ids = boxes.id.cpu().tolist() if boxes.id is not None else [-1] * len(xyxy)

            frame_name = Path(result.path).name
            for (x1, y1, x2, y2), score, track_id in zip(xyxy, conf, ids):
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                writer.writerow(
                    [
                        frame_name,
                        int(track_id),
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        float(f"{xc:.2f}"),
                        float(f"{yc:.2f}"),
                        float(f"{w:.2f}"),
                        float(f"{h:.2f}"),
                        float(f"{score:.4f}") if score is not None else "",
                    ]
                )
