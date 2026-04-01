from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.liao_lab import (
    compare_tracks_to_points,
    discover_render_sources,
    find_fish_coords_xlsx,
    load_fish_coords_xlsx,
    render_source_slug,
)
from src.dataset_layout import AVAILABLE_TRAINING_MANIFEST, AVAILABLE_TRAINING_ROOT
from src.pipeline import (
    describe_detector,
    train_detector,
    track_folder,
    validate_detector,
    visualize_folder,
)
from src.pipeline.utils import repo_root, write_json


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


def _manifest_is_usable(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    source_specs = payload.get("sources")
    if not isinstance(source_specs, list) or not source_specs:
        return False

    for spec in source_specs:
        if not isinstance(spec, dict):
            return False
        path_value = spec.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return False
        source_root = (manifest_path.parent / path_value).resolve()
        if not source_root.exists():
            return False
        if source_root.is_file() and source_root.suffix.lower() == ".json":
            continue
        if (source_root / "data.yaml").exists():
            continue
        if (source_root / "annotations.csv").exists():
            continue
        return False
    return True


def _default_train_data() -> Path:
    canonical_training_root = (
        repo_root() / "data" / "training" / "domain-general-fish-all-yolo"
    )
    if canonical_training_root.exists():
        return canonical_training_root
    if AVAILABLE_TRAINING_ROOT.exists():
        return AVAILABLE_TRAINING_ROOT

    manifests = [
        AVAILABLE_TRAINING_MANIFEST,
        repo_root() / "configs" / "datasets" / "domain_general_fish_all.json",
        repo_root() / "configs" / "datasets" / "domain_general_fish_plus_noaa_psnf.json",
        repo_root() / "configs" / "datasets" / "domain_general_fish.json",
    ]
    for manifest in manifests:
        if _manifest_is_usable(manifest):
            return manifest
    processed_roots = sorted((repo_root() / "data" / "processed").glob("*-yolo"))
    for dataset_root in processed_roots:
        if (dataset_root / "data.yaml").exists():
            return dataset_root
    return repo_root() / "data" / "interim" / "aau-zebrafish-reid"


def _default_frames_dir(dataset_root: Path) -> Path:
    generative_root = repo_root() / "data" / "generative" / "liao-lab-videos"
    if generative_root.exists():
        return generative_root
    if dataset_root.is_file():
        return repo_root() / "data" / "interim" / "aau-zebrafish-reid" / "vid1"
    if (dataset_root / "images" / "train").exists():
        train_images = dataset_root / "images" / "train"
        nested = sorted([p for p in train_images.iterdir() if p.is_dir()])
        if nested:
            return nested[0]
    if (dataset_root / "vid1").exists():
        return dataset_root / "vid1"
    if dataset_root.exists():
        videos = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        if videos:
            return videos[0]
    return dataset_root


def _default_train_output(default_dataset: Path) -> Path:
    if default_dataset.is_file():
        return repo_root() / "models" / f"{default_dataset.stem}.pt"
    return repo_root() / "models" / "domain_general_fish.pt"


def _load_coords_xlsx(path_value: str) -> dict[int, list]:
    return load_fish_coords_xlsx(_as_path(path_value))


def _visualize_batch(
    batch_root: Path,
    output_root: Path,
    weights_path: Path,
    write_tracks: bool,
    fps: float,
    duration_sec: float | None,
) -> list[dict[str, object]]:
    sources = discover_render_sources(batch_root)
    if not sources:
        raise FileNotFoundError(
            f"No video files or frame folders found under {batch_root}"
        )

    videos_dir = output_root / "videos"
    tracks_dir = output_root / "tracks"
    checks_dir = output_root / "coords_checks"
    summary: list[dict[str, object]] = []

    for source in sources:
        slug = render_source_slug(source, batch_root)
        output_video = videos_dir / f"{slug}.mp4"
        coords_xlsx = find_fish_coords_xlsx(source) if source.is_dir() else None
        coords_by_frame = load_fish_coords_xlsx(coords_xlsx) if coords_xlsx else None

        print(f"Rendering {source} -> {output_video}")
        visualize_folder(
            source,
            output_video,
            weights_path,
            fps=fps,
            target_duration_sec=duration_sec,
            coords_by_frame=coords_by_frame,
        )

        item: dict[str, object] = {
            "source": str(source),
            "video": str(output_video),
        }
        if coords_xlsx is not None:
            item["coords_xlsx"] = str(coords_xlsx)

        if write_tracks or coords_by_frame:
            output_tracks = tracks_dir / f"{slug}.csv"
            track_folder(source, output_tracks, weights_path)
            item["tracks"] = str(output_tracks)

            if coords_by_frame:
                output_check = checks_dir / f"{slug}.json"
                metrics = compare_tracks_to_points(
                    output_tracks,
                    coords_by_frame,
                    output_path=output_check,
                )
                item["coords_check"] = str(output_check)
                item["matched_points"] = metrics["matched_points"]
                item["total_points"] = metrics["total_points"]
                item["mean_distance_px"] = metrics["mean_distance_px"]
                print(
                    "Coords check "
                    f"{source.name}: matched {metrics['matched_points']}/{metrics['total_points']} "
                    f"points, mean error={metrics['mean_distance_px']}"
                )

        summary.append(item)

    write_json(
        output_root / "batch_summary.json",
        {
            "root": str(batch_root),
            "weights": str(weights_path),
            "sources": summary,
        },
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
    description="Fish Tracking Interface - train, run tracking, visualize results, and validate models"
    )
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
        default=str(_default_train_output(default_dataset)),
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

    run_parser = subparsers.add_parser(
        "run",
        help="Run tracking on input data",
        description="Runs the tracking pipeline on a video file or folder of frames."
    )
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
        "visualize",
        help="Generate visualization video",
        description="Creates a video with tracking results overlaid on frames or video."
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
    visualize_parser.add_argument(
        "--coords-xlsx",
        help="Optional Fish coords.xlsx file to overlay reference points.",
        default="",
    )
    visualize_parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output FPS. For frame folders this controls playback speed when --duration-sec is not set.",
    )
    visualize_parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="For frame-folder inputs, stretch/compress playback to this many seconds.",
    )

    visualize_batch_parser = subparsers.add_parser(
        "visualize-batch",
        help="Run visualization on multiple inputs",
        description="Processes all valid sources in a folder and generates visualization outputs."
    )
    visualize_batch_parser.add_argument(
        "data",
        help="Folder containing video files and/or frame folders.",
    )
    visualize_batch_parser.add_argument(
        "output",
        help="Output directory for rendered videos and optional sidecar files.",
        nargs="?",
        default=str(repo_root() / "outputs" / "batch_visualizations"),
    )
    visualize_batch_parser.add_argument(
        "--weights",
        help="Optional weights file to use instead of the default resolver",
        default="",
    )
    visualize_batch_parser.add_argument(
        "--write-tracks",
        action="store_true",
        help="Also write tracks CSVs for every source. Coords clips always get CSV + check JSON.",
    )
    visualize_batch_parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output FPS. For frame folders this controls playback speed when --duration-sec is not set.",
    )
    visualize_batch_parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="For frame-folder inputs, stretch/compress playback to this many seconds.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate model performance",
        description="Evaluates the trained detector and outputs performance metrics."
    )
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

    info_parser = subparsers.add_parser(
        "model-info",
        help="Display model architecture",
        description="Prints details about the model structure and configuration."
    )
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
        if not data_root.exists():
            print("Error - input path doesn't exist")
            return 1
        output_path = _ensure_suffix(_as_path(args.output), ".csv")
        print("Running tracking...")
        if args.weights:
            weights_path = _as_path(args.weights)
        else:
            try:
                weights_path = _resolve_weights()
            except FileNotFoundError as e:
                print("Error:", e)
                print("Hint: run training first or set FISH_WEIGHTS.")
                return 1
        track_folder(data_root, output_path, weights_path)
        print(f"Wrote tracks: {output_path}")
        return 0

    if args.command == "visualize":
        data_root = _as_path(args.data)
        if not data_root.exists():
            print("Error - input path doesn't exist")
            return 1
        output_path = _ensure_suffix(_as_path(args.output), ".mp4")
        print("Generating visualization...")
        if args.weights:
            weights_path = _as_path(args.weights)
        else:
            try:
                weights_path = _resolve_weights()
            except FileNotFoundError as e:
                print("Error:", e)
                print("Hint: run training first or set FISH_WEIGHTS.")
                return 1
        coords_by_frame = _load_coords_xlsx(args.coords_xlsx) if args.coords_xlsx else None
        visualize_folder(
            data_root,
            output_path,
            weights_path,
            fps=args.fps,
            target_duration_sec=args.duration_sec or None,
            coords_by_frame=coords_by_frame,
        )
        print(f"Wrote video: {output_path}")
        return 0

    if args.command == "visualize-batch":
        data_root = _as_path(args.data)
        if not data_root.exists():
            print("Error - input path doesn't exist")
            return 1
        output_root = _as_path(args.output)
        print("Processing batch visualization...")
        if args.weights:
            weights_path = _as_path(args.weights)
        else:
            try:
                weights_path = _resolve_weights()
            except FileNotFoundError as e:
                print("Error:", e)
                print("Hint: run training first or set FISH_WEIGHTS.")
                return 1
        summary = _visualize_batch(
            data_root,
            output_root,
            weights_path,
            write_tracks=args.write_tracks,
            fps=args.fps,
            duration_sec=args.duration_sec or None,
        )
        print(f"Wrote batch outputs: {output_root}")
        print(f"Processed sources: {len(summary)}")
        return 0

    if args.command == "validate":
        data_root = _as_path(args.data)
        if not data_root.exists():
            print("Error - input path doesn't exist")
            return 1
        output_path = _ensure_suffix(_as_path(args.output), ".json")
        print("Validating model...")
        if args.weights:
            weights_path = _as_path(args.weights)
        else:
            try:
                weights_path = _resolve_weights()
            except FileNotFoundError as e:
                print("Error:", e)
                print("Hint: run training first or set FISH_WEIGHTS.")
                return 1
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
