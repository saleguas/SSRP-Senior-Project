from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Sequence, Tuple

import cv2
from ultralytics import YOLO

from .utils import list_image_files, repo_root, require_cuda

if TYPE_CHECKING:
    from src.liao_lab import CoordPoint

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv")


@dataclass(frozen=True)
class TrackingSource:
    path: Path
    kind: str
    source_arg: str | List[str]
    frame_names: List[str]
    total_frames: int | None
    fps: float | None


def _list_images(folder: Path) -> List[Path]:
    return list_image_files(folder)


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def _video_metadata(video_path: Path) -> Tuple[float | None, int | None]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(frame_count_value) if frame_count_value and frame_count_value > 0 else None
    capture.release()

    return (fps if fps > 0 else None), frame_count


def _missing_frames_error(images_dir: Path) -> FileNotFoundError:
    subdirs = [p.name for p in images_dir.iterdir() if p.is_dir()]
    hint = f" Available subfolders: {', '.join(subdirs)}" if subdirs else ""
    return FileNotFoundError(
        f"No supported image frames found in {images_dir}. Select a video folder.{hint}"
    )


def _resolve_source(input_path: Path, default_fps: int = 30) -> TrackingSource:
    source_path = input_path.resolve()

    if source_path.is_dir():
        image_paths = _list_images(source_path)
        if not image_paths:
            raise _missing_frames_error(source_path)
        return TrackingSource(
            path=source_path,
            kind="frames",
            source_arg=[str(path) for path in image_paths],
            frame_names=[path.name for path in image_paths],
            total_frames=len(image_paths),
            fps=float(default_fps),
        )

    if _is_video_file(source_path):
        fps, total_frames = _video_metadata(source_path)
        return TrackingSource(
            path=source_path,
            kind="video",
            source_arg=str(source_path),
            frame_names=[],
            total_frames=total_frames,
            fps=fps,
        )

    if source_path.exists():
        raise ValueError(
            f"Unsupported tracking input: {source_path}. Use a folder of PNG/JPG/JPEG frames "
            f"or a video file ({', '.join(VIDEO_EXTENSIONS)})."
        )

    raise FileNotFoundError(f"Tracking input not found: {source_path}")


def _iter_results(model: YOLO, source: TrackingSource):
    return model.track(
        source=source.source_arg,
        stream=True,
        persist=True,
        tracker=_tracker_name(),
        conf=0.25,
        iou=0.5,
        device=0,
        verbose=False,
    )


def _frame_name(source: TrackingSource, frame_index: int, result_path: str | None) -> str:
    if source.kind == "frames":
        if 0 <= frame_index < len(source.frame_names):
            return source.frame_names[frame_index]
        if result_path:
            return Path(result_path).name
    return f"{source.path.stem}_frame_{frame_index + 1:06d}"


def _frame_index_from_name(name: str) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(match.group(1)) if match else None


def _load_frame(source: TrackingSource, frame_index: int, fallback_path: str | None):
    if source.kind == "frames" and 0 <= frame_index < len(source.frame_names):
        return cv2.imread(str(source.path / source.frame_names[frame_index]))
    if fallback_path and source.kind == "frames":
        return cv2.imread(str(fallback_path))
    return None


def _should_log_progress(frame_index: int, total_frames: int | None) -> bool:
    if frame_index == 1:
        return True
    if total_frames is not None and frame_index == total_frames:
        return True
    return frame_index % 25 == 0


def _candidate_output_fps(preferred_fps: float | None, fallback_fps: float = 30.0) -> List[float]:
    values: List[float] = []
    for value in (preferred_fps, fallback_fps, 25.0, 24.0):
        if value is None or not math.isfinite(value):
            continue
        normalized = float(value)
        if normalized <= 0:
            continue
        if normalized > 60.0:
            normalized = 60.0
        if all(abs(existing - normalized) > 1e-6 for existing in values):
            values.append(normalized)
    return values or [30.0]


def _resolve_output_fps(
    source: TrackingSource,
    fallback_fps: float,
    target_duration_sec: float | None,
) -> float:
    if (
        source.kind == "frames"
        and target_duration_sec is not None
        and target_duration_sec > 0
        and source.total_frames
    ):
        return max(0.2, float(source.total_frames) / float(target_duration_sec))
    return float(source.fps or fallback_fps or 30.0)


def _open_video_writer(
    output_video: Path,
    fourcc: int,
    writer_size: Tuple[int, int],
    preferred_fps: float | None,
):
    for candidate_fps in _candidate_output_fps(preferred_fps):
        writer = cv2.VideoWriter(str(output_video), fourcc, candidate_fps, writer_size)
        if writer.isOpened():
            return writer, candidate_fps
        writer.release()
    raise RuntimeError(f"Unable to open video writer for {output_video}")


def _ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def _finalize_video_output(temp_video: Path, output_video: Path) -> None:
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        temp_video.replace(output_video)
        return

    reencoded_video = output_video.with_name(f"{output_video.stem}.__encoded__.mp4")
    if reencoded_video.exists():
        reencoded_video.unlink()

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(temp_video),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(reencoded_video),
    ]
    try:
        subprocess.run(command, check=True)
    except Exception:
        if reencoded_video.exists():
            reencoded_video.unlink()
        temp_video.replace(output_video)
        return

    temp_video.unlink(missing_ok=True)
    if output_video.exists():
        output_video.unlink()
    reencoded_video.replace(output_video)


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


def _result_tracks(result) -> Sequence[Tuple[List[float], float | None, float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().tolist()
    conf = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(xyxy)
    ids = boxes.id.cpu().tolist() if boxes.id is not None else [-1] * len(xyxy)
    return list(zip(xyxy, conf, ids))


def track_folder(images_dir: Path, output_csv: Path, weights_path: Path) -> None:
    require_cuda()

    source = _resolve_source(images_dir)
    model = YOLO(str(weights_path))
    display_map: Dict[int, int] = {}
    last_center: Dict[int, Tuple[float, float]] = {}
    next_display_id = [1]

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

        for frame_index, result in enumerate(_iter_results(model, source)):
            frame_tracks = _result_tracks(result)
            if not frame_tracks:
                continue

            frame = result.orig_img
            if frame is None:
                frame = _load_frame(source, frame_index, getattr(result, "path", None))
            frame_h, frame_w = frame.shape[:2] if frame is not None else (0, 0)
            frame_name = _frame_name(source, frame_index, getattr(result, "path", None))

            for (x1, y1, x2, y2), score, track_id in frame_tracks:
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
    fps: float = 30.0,
    target_duration_sec: float | None = None,
    coords_by_frame: Mapping[int, Sequence["CoordPoint"]] | None = None,
) -> None:
    require_cuda()

    source = _resolve_source(images_dir, default_fps=fps)
    model = YOLO(str(weights_path))
    display_map: Dict[int, int] = {}
    last_center: Dict[int, Tuple[float, float]] = {}
    next_display_id = [1]
    total_frames = source.total_frames
    total_label = total_frames if total_frames is not None else "unknown"
    print(f"Visualize: {source.path} ({total_label} frames) -> {output_video}", flush=True)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    writer_size: Tuple[int, int] | None = None
    frames_written = 0
    video_fps = _resolve_output_fps(source, fallback_fps=float(fps or 30.0), target_duration_sec=target_duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_output_video = output_video.with_name(f"{output_video.stem}.__raw__.mp4")
    temp_output_video.unlink(missing_ok=True)

    try:
        for frame_index, result in enumerate(_iter_results(model, source), start=1):
            frame = result.orig_img
            if frame is None:
                frame = _load_frame(source, frame_index - 1, getattr(result, "path", None))
            if frame is None:
                continue

            frame = frame.copy()
            frame_h, frame_w = frame.shape[:2]
            frame_name = _frame_name(source, frame_index - 1, getattr(result, "path", None))

            if writer is None:
                writer_size = (frame_w, frame_h)
                writer, actual_fps = _open_video_writer(temp_output_video, fourcc, writer_size, video_fps)
                if abs(actual_fps - video_fps) > 1e-6:
                    print(
                        f"Adjusted output FPS for {output_video.name}: {video_fps:.3f} -> {actual_fps:.3f}",
                        flush=True,
                    )

            for (x1, y1, x2, y2), score, track_id in _result_tracks(result):
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

            if coords_by_frame:
                coords_frame_index = _frame_index_from_name(frame_name) or frame_index
                for point in coords_by_frame.get(coords_frame_index, []):
                    center = (int(round(point.x)), int(round(point.y)))
                    cv2.drawMarker(
                        frame,
                        center,
                        (0, 255, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=18,
                        thickness=2,
                    )
                    cv2.putText(
                        frame,
                        f"P{point.fish_id}",
                        (center[0] + 6, max(18, center[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if writer_size is not None and (frame.shape[1], frame.shape[0]) != writer_size:
                frame = cv2.resize(frame, writer_size)

            writer.write(frame)
            frames_written += 1
            if _should_log_progress(frame_index, total_frames):
                try:
                    print(f"Rendered frame {frame_index}/{total_label}", flush=True)
                except OSError:
                    pass
    finally:
        if writer is not None:
            writer.release()

    if frames_written == 0:
        temp_output_video.unlink(missing_ok=True)
        raise RuntimeError("No frames were written during visualization.")

    _finalize_video_output(temp_output_video, output_video)
