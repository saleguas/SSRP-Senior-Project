from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from gooey import Gooey, GooeyParser
except ImportError:
    Gooey = None
    GooeyParser = argparse.ArgumentParser

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import train_detector, track_folder, validate_detector, visualize_folder
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


def _default_frames_dir(dataset_root: Path) -> Path:
    if (dataset_root / "vid1").exists():
        return dataset_root / "vid1"
    if dataset_root.exists():
        videos = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        if videos:
            return videos[0]
    return dataset_root


def _add_arg(
    parser: argparse.ArgumentParser,
    *args: str,
    widget: Optional[str] = None,
    **kwargs: object,
) -> None:
    if Gooey and widget:
        parser.add_argument(*args, widget=widget, **kwargs)
    else:
        parser.add_argument(*args, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = GooeyParser(description="Fish tracking pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_dataset = repo_root() / "data" / "interim" / "aau-zebrafish-reid"

    train_parser = subparsers.add_parser("train", help="Train detector")
    _add_arg(
        train_parser,
        "data",
        help="Dataset folder (contains annotations.csv and video folders)",
        widget="DirChooser",
        nargs="?",
        default=str(default_dataset),
    )
    _add_arg(
        train_parser,
        "output",
        help="Output weights file (.pt)",
        widget="FileSaver",
        default=str(repo_root() / "models" / "fish_best.pt"),
    )

    run_parser = subparsers.add_parser("run", help="Run tracking")
    _add_arg(
        run_parser,
        "data",
        help="Frames folder (PNG images)",
        widget="DirChooser",
    )
    _add_arg(
        run_parser,
        "output",
        help="Output tracks CSV",
        widget="FileSaver",
        default=str(repo_root() / "outputs" / "tracks.csv"),
    )

    visualize_parser = subparsers.add_parser(
        "visualize", help="Write annotated tracking video"
    )
    _add_arg(
        visualize_parser,
        "data",
        help="Frames folder (PNG images)",
        widget="DirChooser",
        nargs="?",
        default=str(_default_frames_dir(default_dataset)),
    )
    _add_arg(
        visualize_parser,
        "output",
        help="Output video (.mp4)",
        widget="FileSaver",
        default=str(repo_root() / "outputs" / "visualization.mp4"),
    )

    validate_parser = subparsers.add_parser("validate", help="Validate detector")
    _add_arg(
        validate_parser,
        "data",
        help="Dataset folder (contains annotations.csv and video folders)",
        widget="DirChooser",
        nargs="?",
        default=str(default_dataset),
    )
    _add_arg(
        validate_parser,
        "output",
        help="Output metrics JSON",
        widget="FileSaver",
        default=str(repo_root() / "outputs" / "metrics.json"),
    )

    return parser


def _run() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        data_root = _as_path(args.data)
        output_path = _ensure_suffix(_as_path(args.output), ".pt")
        metadata = train_detector(data_root, output_path)
        print(f"Saved best weights: {metadata['output_best']}")
        print(f"Saved last weights: {metadata['output_last']}")
        return 0

    if args.command == "run":
        data_root = _as_path(args.data)
        output_path = _ensure_suffix(_as_path(args.output), ".csv")
        weights_path = _resolve_weights()
        track_folder(data_root, output_path, weights_path)
        print(f"Wrote tracks: {output_path}")
        return 0

    if args.command == "visualize":
        data_root = _as_path(args.data)
        output_path = _ensure_suffix(_as_path(args.output), ".mp4")
        weights_path = _resolve_weights()
        visualize_folder(data_root, output_path, weights_path)
        print(f"Wrote video: {output_path}")
        return 0

    if args.command == "validate":
        data_root = _as_path(args.data)
        output_path = _ensure_suffix(_as_path(args.output), ".json")
        weights_path = _resolve_weights()
        validate_detector(data_root, weights_path, output_path)
        print(f"Wrote metrics: {output_path}")
        return 0

    parser.print_help()
    return 1


if Gooey:

    @Gooey(program_name="Fish Tracking", default_size=(820, 620))
    def main() -> None:
        raise SystemExit(_run())

else:

    def main() -> None:
        raise SystemExit(_run())


if __name__ == "__main__":
    main()
