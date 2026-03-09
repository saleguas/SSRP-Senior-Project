from __future__ import annotations

import csv
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

from .utils import repo_root, require_cuda


def _ensure_pngs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.png"))


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    if track_id < 0:
        return (200, 200, 200)
    base = track_id * 37
    return (
        (base * 3) % 255,
        (base * 5) % 255,
        (base * 7) % 255,
    )


def _tracker_name() -> str:
    env_value = os.environ.get("FISH_TRACKER")
    if env_value:
        return env_value
    custom = repo_root() / "configs" / "trackers" / "botsort_fish.yaml"
    if custom.exists():
        return str(custom)
    return "botsort.yaml"


def _jump_threshold(frame_w: int, frame_h: int, bbox_w: float, bbox_h: float) -> float:
    env_value = os.environ.get("FISH_MAX_JUMP_PX")
    if env_value:
        try:
            return max(0.0, float(env_value))
        except ValueError:
            pass
    if bbox_w > 0 and bbox_h > 0:
        return max(30.0, 2.0 * max(bbox_w, bbox_h))
    diag = math.hypot(frame_w, frame_h)
    return max(30.0, 0.1 * diag)

def _assign_display_id(
    raw_id: int,
    center: Tuple[float, float],
    bbox_w: float,
    bbox_h: float,
    frame_w: int,
    frame_h: int,
    display_map: Dict[int, int],
    last_center: Dict[int, Tuple[float, float]],
    next_display_id: List[int],
) -> int:
    if raw_id < 0:
        display_id = next_display_id[0]
        next_display_id[0] += 1
        last_center[display_id] = center
        return display_id

    display_id = display_map.get(raw_id)
    if display_id is None:
        display_id = next_display_id[0]
        next_display_id[0] += 1
        display_map[raw_id] = display_id
        last_center[display_id] = center
        return display_id

    threshold = _jump_threshold(frame_w, frame_h, bbox_w, bbox_h)
    if threshold > 0:
        prev = last_center.get(display_id)
        if prev is not None:
            dist = math.hypot(center[0] - prev[0], center[1] - prev[1])
            if dist > threshold:
                display_id = next_display_id[0]
                next_display_id[0] += 1
                display_map[raw_id] = display_id

    last_center[display_id] = center
    return display_id

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
    tracker_name = _tracker_name()
    display_map: Dict[int, int] = {}
    last_center: Dict[int, Tuple[float, float]] = {}
    next_display_id = [1]

    results = model.track(
        source=[str(p) for p in image_paths],
        stream=True,
        persist=True,
        tracker=tracker_name,
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

            frame = result.orig_img
            if frame is None:
                frame = cv2.imread(str(result.path))
            frame_h, frame_w = frame.shape[:2] if frame is not None else (0, 0)

            xyxy = boxes.xyxy.cpu().tolist()
            conf = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(xyxy)
            ids = boxes.id.cpu().tolist() if boxes.id is not None else [-1] * len(xyxy)

            frame_name = Path(result.path).name
            for (x1, y1, x2, y2), score, track_id in zip(xyxy, conf, ids):
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                display_id = _assign_display_id(
                    int(track_id),
                    (xc, yc),
                    w,
                    h,
                    frame_w,
                    frame_h,
                    display_map,
                    last_center,
                    next_display_id,
                )
                writer.writerow(
                    [
                        frame_name,
                        int(display_id),
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
def visualize_folder(
    images_dir: Path,
    output_video: Path,
    weights_path: Path,
    fps: int = 30,
) -> None:
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
    tracker_name = _tracker_name()
    display_map: Dict[int, int] = {}
    last_center: Dict[int, Tuple[float, float]] = {}
    next_display_id = [1]
    total_frames = len(image_paths)
    print(f"Visualize: {total_frames} frames -> {output_video}", flush=True)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output_video.with_suffix("")
    frames_dir = frames_dir.parent / f"{frames_dir.name}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    saved_frames: List[Path] = []
    tracker_initialized = False
    for idx, image_path in enumerate(image_paths, start=1):
        frame_output = frames_dir / image_path.name
        track_args = {
            "source": str(image_path),
            "persist": True,
            "conf": 0.25,
            "iou": 0.5,
            "device": 0,
            "verbose": False,
        }
        if not tracker_initialized:
            track_args["tracker"] = tracker_name
            tracker_initialized = True
        results = model.track(**track_args)
        if not results:
            continue

        result = results[0]
        frame = result.orig_img
        if frame is None:
            frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            conf = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(xyxy)
            ids = boxes.id.cpu().tolist() if boxes.id is not None else [-1] * len(xyxy)
            for (x1, y1, x2, y2), score, track_id in zip(xyxy, conf, ids):
                x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                display_id = _assign_display_id(
                    int(track_id),
                    (xc, yc),
                    w,
                    h,
                    frame_w,
                    frame_h,
                    display_map,
                    last_center,
                    next_display_id,
                )
                color = _color_for_id(display_id)
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)

                label = f"ID {display_id}"
                if score is not None:
                    label = f"{label} {score:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                text_x = x1_i
                text_y = max(0, y1_i - text_h - baseline - 4)
                cv2.rectangle(
                    frame,
                    (text_x, text_y),
                    (text_x + text_w + 4, text_y + text_h + baseline + 4),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (text_x + 2, text_y + text_h + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        if not cv2.imwrite(str(frame_output), frame):
            continue
        saved_frames.append(frame_output)
        try:
            print(f"Saved frame {idx}/{total_frames}: {frame_output.name}", flush=True)
        except OSError:
            pass

    ordered_frames = [frames_dir / p.name for p in image_paths if (frames_dir / p.name).exists()]
    if not ordered_frames:
        raise RuntimeError("No frames were saved during visualization.")

    first = cv2.imread(str(ordered_frames[0]))
    if first is None:
        raise RuntimeError("Unable to read saved frames to build video.")

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    for idx, frame_path in enumerate(ordered_frames, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        writer.write(frame)
        if idx == 1 or idx == len(ordered_frames) or idx % 25 == 0:
            try:
                print(f"Stitch: {idx}/{len(ordered_frames)} frames", flush=True)
            except OSError:
                pass

    writer.release()
    shutil.rmtree(frames_dir, ignore_errors=True)






