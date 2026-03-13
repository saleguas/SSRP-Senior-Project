from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import (
    describe_detector,
    train_detector,
    track_folder,
    validate_detector,
    visualize_folder,
)
from src.pipeline.utils import repo_root


def _as_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _ensure_suffix(path: Path, suffix: str) -> Path:
    if path.suffix.lower() != suffix.lower():
        return path.with_suffix(suffix)
    return path


def _resolve_weights() -> Path:
    env_value = os.environ.get("FISH_WEIGHTS")
    if env_value:
        path = _as_path(env_value)
        if path.exists():
            return path
        raise FileNotFoundError(f"FISH_WEIGHTS was set but not found: {path}")

    default_path = repo_root() / "models" / "latest.pt"
    if default_path.exists():
        return default_path

    models_root = repo_root() / "models"
    preferred = [
        models_root / "runs" / "domain_general_fish" / "weights" / "best.pt",
        models_root / "yolo_fish" / "weights" / "best.pt",
        models_root / "yolo_fish" / "weights" / "last.pt",
    ]
    for path in preferred:
        if path.exists():
            return path

    def newest(paths: list[Path]) -> Optional[Path]:
        if not paths:
            return None
        return max(paths, key=lambda p: p.stat().st_mtime)

    best_candidates = newest(list(models_root.glob("**/weights/best.pt")))
    if best_candidates:
        return best_candidates

    last_candidates = newest(list(models_root.glob("**/weights/last.pt")))
    if last_candidates:
        return last_candidates

    runs_root = repo_root() / "runs"
    runs_best = newest(list(runs_root.glob("detect/**/weights/best.pt")))
    if runs_best:
        return runs_best
    runs_last = newest(list(runs_root.glob("detect/**/weights/last.pt")))
    if runs_last:
        return runs_last

    base_names = {"yolov8n.pt", "yolo11n.pt"}
    pt_candidates = [
        p
        for p in models_root.glob("**/*.pt")
        if p.name.lower() not in base_names
    ]
    newest_pt = newest(pt_candidates)
    if newest_pt:
        return newest_pt

    raise FileNotFoundError(
        "No trained weights found. Run train first or set FISH_WEIGHTS."
    )


def _default_train_data() -> Path:
    manifest = repo_root() / "configs" / "datasets" / "domain_general_fish.json"
    if manifest.exists():
        return manifest
    return repo_root() / "data" / "interim" / "aau-zebrafish-reid"


def _default_frames_dir(dataset_root: Path) -> Path:
    if dataset_root.is_file():
        return repo_root() / "data" / "interim" / "aau-zebrafish-reid" / "vid1"
    if (dataset_root / "vid1").exists():
        return dataset_root / "vid1"
    if dataset_root.exists():
        videos = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        if videos:
            return videos[0]
    return dataset_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fish tracking pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_dataset = _default_train_data()

    train_parser = subparsers.add_parser("train", help="Train detector")
    train_parser.add_argument(
        "data",
        help="Normalized dataset folder, processed YOLO dataset root, or JSON dataset manifest",
        nargs="?",
        default=str(default_dataset),
    )
    train_parser.add_argument(
        "output",
        help="Output weights file (.pt)",
        nargs="?",
        default=str(repo_root() / "models" / "domain_general_fish.pt"),
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Override the auto-selected epoch count.",
    )
    train_parser.add_argument(
        "--log",
        default="",
        help="Optional training log file path. Defaults to <output>.train.log.",
    )
    train_parser.add_argument(
        "--imgsz",
        type=int,
        default=0,
        help="Override the training image size. Defaults to a speed-aware auto choice.",
    )
    train_parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Override dataloader workers. Defaults to an auto choice.",
    )
    train_parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Override batch size. Defaults to Ultralytics AutoBatch.",
    )
    train_parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base detection model to fine-tune.",
    )
    train_parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training. This is slower; leave off for normal runs.",
    )

    run_parser = subparsers.add_parser("run", help="Run tracking")
    run_parser.add_argument(
        "data",
        help="Frames folder (PNG/JPG/JPEG images) or video file (.mp4/.avi/.mov/.mkv)",
    )
    run_parser.add_argument(
        "output",
        help="Output tracks CSV",
        nargs="?",
        default=str(repo_root() / "outputs" / "tracks.csv"),
    )
    run_parser.add_argument(
        "--weights",
        help="Optional weights file to use instead of the default resolver",
        default="",
    )

    visualize_parser = subparsers.add_parser(
        "visualize", help="Write annotated tracking video"
    )
    visualize_parser.add_argument(
        "data",
        help="Frames folder (PNG/JPG/JPEG images) or video file (.mp4/.avi/.mov/.mkv)",
        nargs="?",
        default=str(_default_frames_dir(default_dataset)),
    )
    visualize_parser.add_argument(
        "output",
        help="Output video (.mp4)",
        nargs="?",
        default=str(repo_root() / "outputs" / "visualization.mp4"),
    )
    visualize_parser.add_argument(
        "--weights",
        help="Optional weights file to use instead of the default resolver",
        default="",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate detector")
    validate_parser.add_argument(
        "data",
        help="Normalized dataset folder, processed YOLO dataset root, or JSON dataset manifest",
        nargs="?",
        default=str(default_dataset),
    )
    validate_parser.add_argument(
        "output",
        help="Output metrics JSON",
        nargs="?",
        default=str(repo_root() / "outputs" / "metrics.json"),
    )
    validate_parser.add_argument(
        "--weights",
        help="Optional weights file to use instead of the default resolver",
        default="",
    )

    info_parser = subparsers.add_parser("model-info", help="Show detector architecture")
    info_parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Model weights or YAML to inspect.",
    )
    info_parser.add_argument(
        "--classes",
        type=int,
        default=1,
        help="Number of detector classes to instantiate for the summary.",
    )

    return parser


def _run() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        data_root = _as_path(args.data)
        output_path = _ensure_suffix(_as_path(args.output), ".pt")
        log_path = _as_path(args.log) if args.log else None
        metadata = train_detector(
            data_root,
            output_path,
            epochs=args.epochs or None,
            log_path=log_path,
            imgsz=args.imgsz or None,
            workers=None if args.workers < 0 else args.workers,
            batch=args.batch or None,
            deterministic=args.deterministic,
            model_name=args.model,
        )
        print(f"Saved best weights: {metadata['output_best']}")
        print(f"Saved last weights: {metadata['output_last']}")
        print(f"Training log: {metadata['log_path']}")
        return 0

    if args.command == "run":
        data_root = _as_path(args.data)
        output_path = _ensure_suffix(_as_path(args.output), ".csv")
        weights_path = _as_path(args.weights) if args.weights else _resolve_weights()
        track_folder(data_root, output_path, weights_path)
        print(f"Wrote tracks: {output_path}")
        return 0

    if args.command == "visualize":
        data_root = _as_path(args.data)
        output_path = _ensure_suffix(_as_path(args.output), ".mp4")
        weights_path = _as_path(args.weights) if args.weights else _resolve_weights()
        visualize_folder(data_root, output_path, weights_path)
        print(f"Wrote video: {output_path}")
        return 0

    if args.command == "validate":
        data_root = _as_path(args.data)
        output_path = _ensure_suffix(_as_path(args.output), ".json")
        weights_path = _as_path(args.weights) if args.weights else _resolve_weights()
        validate_detector(data_root, weights_path, output_path)
        print(f"Wrote metrics: {output_path}")
        return 0

    if args.command == "model-info":
        info = describe_detector(args.model, classes=args.classes)
        print(json.dumps(info, indent=2))
        return 0

    parser.print_help()
    return 1


def main() -> None:
    raise SystemExit(_run())


if __name__ == "__main__":
    main()
