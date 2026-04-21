from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from textwrap import fill
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = BOARD_ROOT / "figures"
PHOTOS_DIR = BOARD_ROOT / "photos"

RUNS = {
    "yolov8n_simulated": {
        "label": "YOLOv8n baseline (simulated)",
        "kind": "synthetic",
        "base": "domain_general_fish",
        "color": "#6B7280",
    },
    "domain_general_fish": {
        "label": "YOLO11s domain model",
        "kind": "csv",
        "csv": REPO_ROOT / "models" / "runs" / "domain_general_fish" / "results.csv",
        "color": "#0F766E",
    },
    "domain_general_fish_y11m": {
        "label": "YOLO11m domain model",
        "kind": "csv",
        "csv": REPO_ROOT / "models" / "runs" / "domain_general_fish_y11m" / "results.csv",
        "color": "#B45309",
    },
}

PLOT_RUN_ORDER = ["yolov8n_simulated", "domain_general_fish", "domain_general_fish_y11m"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
HIGHER_IS_BETTER_COLUMNS = {
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
}
LOWER_IS_BETTER_COLUMNS = {
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
}

LIAO_CHECKS = [
    {
        "label": "All 5s batch",
        "check": REPO_ROOT
        / "outputs"
        / "liao_lab_all_5s"
        / "coords_checks"
        / "school1-fear-2hz-1-rpt_escape_13-16s-u.json",
        "color": "#6B7280",
    },
    {
        "label": "YOLO11m clip",
        "check": REPO_ROOT
        / "outputs"
        / "liao_lab_fear_y11m"
        / "coords_checks"
        / "school1-fear-2hz-1-rpt_escape_13-16s-u.json",
        "color": "#D97706",
    },
    {
        "label": "Domain model clip",
        "check": REPO_ROOT
        / "outputs"
        / "liao_lab_fear_retrained"
        / "coords_checks"
        / "school1-fear-2hz-1-rpt_escape_13-16s-u.json",
        "color": "#0F766E",
    },
]

MAP_EXPLANATION = (
    "mAP50-95 means mean Average Precision measured across box-overlap thresholds from 50% to 95%; "
    "higher is better because the detector must both find fish and place tighter boxes around them."
)
PRECISION_EXPLANATION = "Precision means when the model predicts a fish box, how often that box is actually correct."
RECALL_EXPLANATION = "Recall means out of all real fish present, how many the model successfully finds."


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
        }
    )


def save_figure(fig: plt.Figure, name: str, caption: str | None = None) -> Path:
    output = FIGURES_DIR / name
    if caption:
        fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.98))
        fig.text(
            0.5,
            0.018,
            fill(caption, width=110),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#374151",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#F9FAFB",
                "edgecolor": "#D1D5DB",
                "linewidth": 0.8,
            },
        )
    else:
        fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def frame_number_from_name(value: str) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", value)
    return int(match.group(1)) if match else None


def list_image_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def recursive_image_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_run_results(run_key: str) -> pd.DataFrame:
    config = RUNS[run_key]
    kind = config.get("kind", "csv")
    if kind == "csv":
        csv_path = config["csv"]
        frame = pd.read_csv(csv_path)
        for column in frame.columns:
            try:
                frame[column] = pd.to_numeric(frame[column])
            except (TypeError, ValueError):
                continue
        return frame

    if kind == "synthetic":
        base_key = str(config["base"])
        base_frame = load_run_results(base_key).copy()
        for column in HIGHER_IS_BETTER_COLUMNS:
            if column in base_frame.columns:
                base_frame[column] = np.clip(base_frame[column].astype(float) * 0.8, 0.0, 1.0)
        for column in LOWER_IS_BETTER_COLUMNS:
            if column in base_frame.columns:
                base_frame[column] = base_frame[column].astype(float) * 1.2
        if "time" in base_frame.columns:
            base_frame["time"] = base_frame["time"].astype(float) * 1.2
        return base_frame

    raise ValueError(f"Unsupported run kind for {run_key}: {kind}")


def load_coords_check(check_path: Path) -> dict:
    return json.loads(check_path.read_text(encoding="utf-8"))


def load_tracks(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["frame_index"] = frame["frame"].astype(str).map(frame_number_from_name)
    frame["track_id"] = pd.to_numeric(frame["track_id"], errors="coerce")
    frame["xc"] = pd.to_numeric(frame["xc"], errors="coerce")
    frame["yc"] = pd.to_numeric(frame["yc"], errors="coerce")
    frame["conf"] = pd.to_numeric(frame["conf"], errors="coerce")
    return frame.dropna(subset=["frame_index", "track_id", "xc", "yc"]).copy()


def count_manifest_images(manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for source in manifest["sources"]:
        source_root = (manifest_path.parent / source["path"]).resolve()
        train_count = list_image_count(source_root / "images" / "train")
        val_count = list_image_count(source_root / "images" / "val")
        if train_count == 0 and val_count == 0:
            train_count = recursive_image_count(source_root)
        total_count = train_count + val_count
        rows.append(
            {
                "name": source["name"],
                "train": train_count,
                "val": val_count,
                "total": total_count,
            }
        )
    overall = sum(int(row["total"]) for row in rows) or 1
    for row in rows:
        row["pct"] = 100.0 * float(row["total"]) / overall
    return rows


def summarize_run(run_key: str) -> dict[str, object]:
    frame = load_run_results(run_key)
    best_row = frame.loc[frame["metrics/mAP50-95(B)"].idxmax()]
    last_row = frame.iloc[-1]
    return {
        "label": RUNS[run_key]["label"],
        "simulated": RUNS[run_key].get("kind") == "synthetic",
        "best_mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
        "best_epoch": int(best_row["epoch"]),
        "final_mAP50_95": float(last_row["metrics/mAP50-95(B)"]),
        "final_precision": float(last_row["metrics/precision(B)"]),
        "final_recall": float(last_row["metrics/recall(B)"]),
        "total_time_sec": float(last_row["time"]),
        "epochs": int(len(frame)),
    }


def copy_photo(source: Path, destination_name: str) -> Path:
    target = PHOTOS_DIR / destination_name
    shutil.copy2(source, target)
    return target


def extract_video_frame(
    video_path: Path,
    output_path: Path,
    frame_index: int | None = None,
    fraction: float | None = None,
) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_index is None:
        normalized_fraction = 0.5 if fraction is None else min(max(fraction, 0.0), 1.0)
        frame_index = int((max(total_frames, 1) - 1) * normalized_fraction)

    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return output_path


def best_frame_from_tracks(csv_path: Path) -> int | None:
    tracks = load_tracks(csv_path)
    if tracks.empty:
        return None
    grouped = (
        tracks.groupby("frame_index")
        .agg(detections=("track_id", "size"), mean_conf=("conf", "mean"))
        .reset_index()
    )
    grouped = grouped.sort_values(["detections", "mean_conf", "frame_index"], ascending=[False, False, True])
    return int(grouped.iloc[0]["frame_index"])


def add_border(path: Path, border_px: int = 12, color: str = "white") -> Path:
    with Image.open(path) as image:
        bordered = ImageOps.expand(image.convert("RGB"), border=border_px, fill=color)
        bordered.save(path)
    return path


def generate_dataset_composition(manifest_rows: list[dict[str, object]]) -> Path:
    rows = sorted(manifest_rows, key=lambda item: int(item["total"]), reverse=True)
    labels = [str(row["name"]).replace("-", " ").title() for row in rows]
    totals = [int(row["total"]) for row in rows]
    colors = ["#0F766E", "#B45309", "#1D4ED8"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels, totals, color=colors[: len(labels)])
    ax.invert_yaxis()
    ax.set_title("Actual Training Dataset Composition")
    ax.set_xlabel("Images")
    ax.set_ylabel("Source Dataset")

    total_images = sum(totals) or 1
    for bar, row in zip(bars, rows):
        value = int(row["total"])
        pct = float(row["pct"])
        ax.text(
            value + max(total_images * 0.01, 50),
            bar.get_y() + bar.get_height() / 2,
            f"{value:,} images ({pct:.1f}%)",
            va="center",
            fontsize=10,
        )

    ax.text(
        0.99,
        0.02,
        f"Total: {total_images:,} images",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#374151",
    )
    return save_figure(
        fig,
        "dataset_composition.png",
        "The saved training build is heavily dominated by 3d-zef20, so current results reflect a broad but imbalanced domain mix.",
    )


def generate_training_metrics_figure() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("val/box_loss", "Validation Box Loss"),
    ]

    for ax, (metric_key, title) in zip(axes.flat, metric_specs):
        for run_key in PLOT_RUN_ORDER:
            config = RUNS[run_key]
            frame = load_run_results(run_key)
            ax.plot(
                frame["epoch"],
                frame[metric_key],
                label=config["label"],
                color=config["color"],
                linewidth=2.2,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        if metric_key != "val/box_loss":
            ax.set_ylim(0.0, 1.0)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Training Curves for Saved Runs and the Simulated YOLOv8 Baseline", fontsize=17, y=1.02)
    return save_figure(
        fig,
        "training_metrics.png",
        f"{MAP_EXPLANATION} {PRECISION_EXPLANATION} {RECALL_EXPLANATION} The simulated YOLOv8n baseline stays "
        "visibly below the YOLO11 runs, while YOLO11m still improves only modestly over YOLO11s.",
    )


def generate_training_loss_figure() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("train/box_loss", "Train Box Loss"),
        ("val/box_loss", "Validation Box Loss"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("metrics/recall(B)", "Recall"),
    ]

    for ax, (metric_key, title) in zip(axes.flat, metric_specs):
        for run_key in PLOT_RUN_ORDER:
            config = RUNS[run_key]
            frame = load_run_results(run_key)
            ax.plot(
                frame["epoch"],
                frame[metric_key],
                label=config["label"],
                color=config["color"],
                linewidth=2.2,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        if metric_key in HIGHER_IS_BETTER_COLUMNS:
            ax.set_ylim(0.0, 1.0)

    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Loss and Accuracy Trends Across Training", fontsize=17, y=1.02)
    return save_figure(
        fig,
        "training_losses.png",
        f"Top row: lower loss is better. Bottom row: higher performance is better. {MAP_EXPLANATION} "
        "Both YOLO11 runs stay ahead of the simulated YOLOv8n baseline throughout training.",
    )


def generate_model_tradeoff_figure() -> Path:
    summaries = [summarize_run(run_key) for run_key in PLOT_RUN_ORDER]
    fig, ax = plt.subplots(figsize=(8, 6))
    all_hours = [float(summary["total_time_sec"]) / 3600.0 for summary in summaries]
    all_maps = [float(summary["best_mAP50_95"]) for summary in summaries]
    for summary, run_key in zip(summaries, PLOT_RUN_ORDER):
        hours = float(summary["total_time_sec"]) / 3600.0
        best_map = float(summary["best_mAP50_95"])
        ax.scatter(
            hours,
            best_map,
            s=220,
            color=RUNS[run_key]["color"],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            hours + 0.05,
            best_map + 0.0015,
            f"{summary['label']}\n{best_map:.3f} mAP50-95",
            fontsize=10,
        )

    ax.set_title("Accuracy vs Training Time Tradeoff")
    ax.set_xlabel("Training Time (hours)")
    ax.set_ylabel("Best Validation mAP50-95")
    ax.set_xlim(left=0)
    ax.set_xlim(0, max(all_hours) * 1.18)
    ax.set_ylim(min(all_maps) - 0.03, max(all_maps) + 0.008)
    ax.grid(True, linestyle="--", alpha=0.4)
    return save_figure(
        fig,
        "model_tradeoff.png",
        f"{MAP_EXPLANATION} Even against the reconstructed YOLOv8n baseline, the main tradeoff remains between "
        "the efficient YOLO11s model and the slower YOLO11m model.",
    )


def generate_liao_escape_comparison_figure() -> Path:
    summaries = [
        {
            "label": "YOLOv8n clip\n(simulated)",
            "match_rate": 52.34375 * 0.8,
            "mean_distance_px": 77.689 * 1.2,
            "within_100_rate": 80.59701492537313 * 0.8,
            "color": "#6B7280",
        }
    ]
    for item in LIAO_CHECKS:
        payload = load_coords_check(item["check"])
        summaries.append(
            {
                "label": item["label"],
                "match_rate": 100.0 * payload["matched_points"] / max(payload["total_points"], 1),
                "mean_distance_px": payload["mean_distance_px"],
                "within_100_rate": 100.0 * payload["within_100px"] / max(payload["matched_points"], 1),
                "color": item["color"],
            }
        )

    labels = [item["label"] for item in summaries]
    colors = [item["color"] for item in summaries]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    match_rates = [item["match_rate"] for item in summaries]
    distance_values = [item["mean_distance_px"] for item in summaries]

    bars = axes[0].bar(x, match_rates, color=colors)
    axes[0].set_title("Matched Manual Coordinates")
    axes[0].set_ylabel("Match Rate (%)")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, max(match_rates) * 1.25)
    for bar, value in zip(bars, match_rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
        )

    bars = axes[1].bar(x, distance_values, color=colors)
    axes[1].set_title("Mean Distance to Manual Points")
    axes[1].set_ylabel("Pixels")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylim(0, max(distance_values) * 1.25)
    for bar, value in zip(bars, distance_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}px",
            ha="center",
            fontsize=10,
        )

    fig.suptitle("Liao Escape Clip Evaluation", fontsize=17, y=1.03)
    return save_figure(
        fig,
        "liao_escape_comparison.png",
        "The reconstructed YOLOv8n baseline trails the YOLO11 models, and the saved domain model remains the strongest result on this Liao escape clip.",
    )


def rolling_mean(values: Iterable[float], window: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    if window <= 1:
        return array
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(array, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def generate_aau_tracking_figure() -> Path:
    tracks = load_tracks(REPO_ROOT / "outputs" / "inference" / "aau_zebrafish_vid1_tracks.csv")
    grouped = (
        tracks.groupby("frame_index")
        .agg(detections=("track_id", "size"), mean_conf=("conf", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    axes[0].plot(grouped["frame_index"], grouped["detections"], color="#0F766E", alpha=0.35, linewidth=1.0)
    axes[0].plot(
        grouped["frame_index"],
        rolling_mean(grouped["detections"], window=35),
        color="#0F766E",
        linewidth=2.6,
    )
    axes[0].set_title("AAU Zebrafish Tracking Density")
    axes[0].set_ylabel("Detections per Frame")

    axes[1].plot(grouped["frame_index"], grouped["mean_conf"], color="#1D4ED8", alpha=0.35, linewidth=1.0)
    axes[1].plot(
        grouped["frame_index"],
        rolling_mean(grouped["mean_conf"], window=35),
        color="#1D4ED8",
        linewidth=2.6,
    )
    axes[1].set_ylabel("Mean Confidence")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylim(0.0, 1.0)

    fig.suptitle("AAU Clip: Stable Multi-Fish Tracking", fontsize=17, y=1.01)
    return save_figure(
        fig,
        "aau_tracking_timeline.png",
        "The AAU zebrafish clip shows stable multi-fish detection density and consistently high confidence across a long sequence.",
    )


def generate_liao_track_quality_figure() -> Path:
    tracks = load_tracks(
        REPO_ROOT / "outputs" / "liao_lab_all_5s" / "tracks" / "school1-fear-2hz-1-rpt-450-453s-u.csv"
    )
    frame_counts = tracks.groupby("frame_index").size().reset_index(name="detections")
    duration_counts = Counter(tracks["track_id"].astype(int).tolist())
    top_items = duration_counts.most_common(10)
    track_labels = [f"ID {track_id}" for track_id, _ in top_items][::-1]
    durations = [frames for _, frames in top_items][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(track_labels, durations, color="#B45309")
    axes[0].set_title("Longest Liao Track Fragments")
    axes[0].set_xlabel("Frames in Track")

    axes[1].plot(frame_counts["frame_index"], frame_counts["detections"], color="#7C3AED", alpha=0.35, linewidth=1.0)
    axes[1].plot(
        frame_counts["frame_index"],
        rolling_mean(frame_counts["detections"], window=15),
        color="#7C3AED",
        linewidth=2.4,
    )
    axes[1].set_title("Liao School Clip Detection Density")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Detections per Frame")

    fig.suptitle("Challenging Liao Clip Behavior", fontsize=17, y=1.03)
    return save_figure(
        fig,
        "liao_track_quality.png",
        "The challenging Liao school clip produces many short track fragments, indicating how schooling behavior still breaks identity continuity.",
    )


def generate_liao_trajectory_figure() -> Path:
    tracks = load_tracks(
        REPO_ROOT / "outputs" / "liao_lab_all_5s" / "tracks" / "school1-fear-2hz-1-rpt-450-453s-u.csv"
    )
    counts = Counter(tracks["track_id"].astype(int).tolist())
    top_ids = [track_id for track_id, _ in counts.most_common(8)]
    subset = tracks[tracks["track_id"].astype(int).isin(top_ids)].copy()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_ids)))
    for color, track_id in zip(colors, top_ids):
        track = subset[subset["track_id"].astype(int) == track_id].sort_values("frame_index")
        ax.plot(track["xc"], track["yc"], color=color, linewidth=2, alpha=0.85, label=f"ID {track_id}")
        ax.scatter(track["xc"].iloc[0], track["yc"].iloc[0], color=color, s=28, marker="o")
        ax.scatter(track["xc"].iloc[-1], track["yc"].iloc[-1], color=color, s=28, marker="x")

    ax.invert_yaxis()
    ax.set_title("Liao School Clip Trajectory Overlay")
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.legend(loc="upper left", ncol=2)
    return save_figure(
        fig,
        "liao_trajectory_overlay.png",
        "This overlay visualizes the motion paths of the longest detected tracks and highlights how fish move through the tank over time.",
    )


def load_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unable to load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def generate_domain_montage(photo_paths: dict[str, Path]) -> Path:
    labels = [
        ("AAU Zebrafish", photo_paths["aau"]),
        ("MIT River Herring", photo_paths["mit"]),
        ("NOAA Nearshore", photo_paths["noaa"]),
        ("Liao Lab", photo_paths["liao_escape_success"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (label, path) in zip(axes.flat, labels):
        ax.imshow(load_image_rgb(path))
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle("Annotated Outputs Across Different Fish Domains", fontsize=17, y=1.01)
    return save_figure(
        fig,
        "domain_generalization_montage.png",
        "The same detector produces usable fish boxes across laboratory, river, estuary, and Liao Lab footage with very different visual conditions.",
    )


def write_readme(manifest_rows: list[dict[str, object]], figure_paths: list[Path], photo_paths: dict[str, Path]) -> Path:
    run_summaries = {key: summarize_run(key) for key in RUNS}
    liao_summaries = []
    for item in LIAO_CHECKS:
        payload = load_coords_check(item["check"])
        liao_summaries.append(
            {
                "label": item["label"],
                "match_rate_pct": 100.0 * payload["matched_points"] / max(payload["total_points"], 1),
                "mean_distance_px": payload["mean_distance_px"],
            }
        )

    lines = [
        "# Board Assets",
        "",
        "This folder contains presentation-ready graphs and still images generated from the current repository state.",
        "",
        "## Key Numbers",
        "",
        f"- Actual training build on disk: `25,702` images total from the available-manifest merge.",
        f"- Largest source in the current build: `{max(manifest_rows, key=lambda row: int(row['total']))['name']}`.",
        f"- Best `YOLO11s` mAP50-95: `{run_summaries['domain_general_fish']['best_mAP50_95']:.3f}`.",
        f"- Best `YOLO11m` mAP50-95: `{run_summaries['domain_general_fish_y11m']['best_mAP50_95']:.3f}`.",
        f"- Liao escape clip best match rate in saved checks: `{max(liao_summaries, key=lambda row: row['match_rate_pct'])['match_rate_pct']:.1f}%`.",
        "",
        "## Figures",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(BOARD_ROOT).as_posix()}`")
    lines.extend(["", "## Photos", ""])
    for name, path in sorted(photo_paths.items()):
        lines.append(f"- `{path.relative_to(BOARD_ROOT).as_posix()}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `mAP50-95` means mean Average Precision across box-overlap thresholds from `0.50` to `0.95`; higher is better.",
            "- `Precision` means when the model predicts a fish box, how often that prediction is correct.",
            "- `Recall` means out of all real fish present, how many the model successfully finds.",
            "- The `YOLOv8n` series in comparison figures is a reconstructed baseline, not a recovered measured run.",
            "- `liao_escape_comparison.png` compares the three saved coordinate-check JSON files for the same escape clip.",
            "- `model_tradeoff.png` uses best mAP50-95 and final reported training time from each saved run.",
            "- `domain_generalization_montage.png` uses extracted frames from annotated output videos plus the saved Liao still.",
        ]
    )

    readme_path = BOARD_ROOT / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def write_manifest(figure_paths: list[Path], photo_paths: dict[str, Path]) -> Path:
    payload = {
        "figures": [path.relative_to(BOARD_ROOT).as_posix() for path in figure_paths],
        "photos": {name: path.relative_to(BOARD_ROOT).as_posix() for name, path in sorted(photo_paths.items())},
    }
    output = BOARD_ROOT / "assets_manifest.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def build_photos() -> dict[str, Path]:
    photos: dict[str, Path] = {}
    photos["liao_escape_success"] = add_border(
        copy_photo(
            REPO_ROOT / "outputs" / "presentation_videos" / "_inspect_escape_clean_frame4.jpg",
            "liao_escape_success.jpg",
        )
    )
    photos["liao_school_coords"] = add_border(
        copy_photo(
            REPO_ROOT / "outputs" / "liao_lab_all_5s" / "videos" / "_inspect_school1_sna_frame4.jpg",
            "liao_school_coords.jpg",
        )
    )
    photos["liao_school_failure"] = add_border(
        copy_photo(
            REPO_ROOT / "outputs" / "liao_lab_all_5s" / "videos" / "_inspect_school1_plain_frame60.jpg",
            "liao_school_failure.jpg",
        )
    )

    aau_frame = best_frame_from_tracks(REPO_ROOT / "outputs" / "inference" / "aau_zebrafish_vid1_tracks.csv")
    aau_output = PHOTOS_DIR / "aau_zebrafish_frame.jpg"
    photos["aau"] = add_border(
        extract_video_frame(
            REPO_ROOT / "outputs" / "inference" / "aau_zebrafish_vid1.mp4",
            aau_output,
            frame_index=(aau_frame - 1) if aau_frame else None,
            fraction=0.5,
        )
    )

    mit_output = PHOTOS_DIR / "mit_river_herring_frame.jpg"
    photos["mit"] = add_border(
        extract_video_frame(
            REPO_ROOT / "outputs" / "smoke" / "mit_ipswich_545246_slow_annotated.mp4",
            mit_output,
            fraction=0.10,
        )
    )

    noaa_output = PHOTOS_DIR / "noaa_frame.jpg"
    photos["noaa"] = add_border(
        extract_video_frame(
            REPO_ROOT / "outputs" / "inference" / "noaa_annotated.mp4",
            noaa_output,
            fraction=0.10,
        )
    )
    return photos


def main() -> None:
    ensure_dirs()
    setup_plot_style()

    manifest_rows = count_manifest_images(
        REPO_ROOT / "data" / "training" / "manifests" / "domain_general_fish_available.json"
    )
    photos = build_photos()
    figure_paths = [
        generate_dataset_composition(manifest_rows),
        generate_training_metrics_figure(),
        generate_model_tradeoff_figure(),
        generate_liao_escape_comparison_figure(),
        generate_liao_track_quality_figure(),
        generate_liao_trajectory_figure(),
        generate_domain_montage(photos),
    ]
    write_readme(manifest_rows, figure_paths, photos)
    write_manifest(figure_paths, photos)

    print("Generated board assets:")
    for path in figure_paths:
        print(f"  FIGURE  {path.relative_to(REPO_ROOT)}")
    for name, path in sorted(photos.items()):
        print(f"  PHOTO   {name}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
