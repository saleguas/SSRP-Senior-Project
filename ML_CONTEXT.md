# ML Context for This Codebase
This is a concise, technical map of the repo for an ML agent or engineer.

## Project Goal
Track individual fish in fixed-camera footage. Output 2D trajectories and
consistent IDs over time. No 3D or pose.

## Stack
- Python, PyTorch, Ultralytics YOLO
- OpenCV for visualization
- Pure `argparse` CLI
- CUDA GPU required for train/track/visualize

## Data Layout (normalized)
`data/interim/aau-zebrafish-reid/`
- `annotations.csv`: ground-truth boxes + IDs
- `vid1/`, `vid2/`: PNG frames (`Vid1_*.png`, `Vid2_*.png`)

## Data Layout (raw download complete)
`data/raw/mit-sea-grant-river-herring/`
- `mit_river_herring.zip`: 40GB image archive from LILA
- `mit_sea_grant_river_herring.json.zip`: COCO metadata zip from LILA
- `metadata/mit_sea_grant_river_herring.json`: extracted COCO metadata JSON

`data/raw/deep-vision-fish/`
- `fishDatasetSimulationAlgorithm.zip`: Deep Vision archive from NMDC
- `unzipped/fish_dataset/`: extracted source images + CSV annotations

`data/raw/kakadu-fishai/`
- `202210-KakaduFishAI-TrainingData.zip`: Zenodo archive
- `unzipped/`: extracted images + `KakaduFishAI_boundingbox.json`

`data/raw/noaa-puget-sound-nearshore-fish/`
- `noaa_estuary_fish-images.zip`: LILA/GCP image archive
- `noaa_estuary_fish-annotations-2023.08.19.zip`: annotations zip
- `unzipped/annotations/noaa_estuary_fish-2023.08.19.json`: extracted annotation JSON
- `unzipped/images/`: extracted JPGs

## Data Layout (normalized via organizer)
`data/interim/mit-sea-grant-river-herring/`
- `annotations.csv`: normalized boxes with category labels
- `<clip_name>/`: PNG frames renamed to unique per-clip filenames

`data/processed/aau-zebrafish-reid-yolo/` (auto-generated)
- YOLO format train/val split by video
- `data.yaml` used by Ultralytics

`data/processed/deep-vision-fish-yolo/`
- YOLO-ready dataset built directly from the Deep Vision CSV annotations

`data/processed/kakadu-fishai-yolo/`
- YOLO-ready dataset built directly from the Kakadu COCO JSON
- deterministic split: image id modulo 10 -> val

`data/processed/noaa-puget-sound-nearshore-fish-yolo/`
- YOLO-ready dataset built from the NOAA COCO JSON
- keeps `fish` positives, keeps `empty`/`crab` as negatives, excludes `fish_or_crab` and `unknown`
- deterministic split by hashed `location` to reduce frame leakage

`data/processed/domain-general-fish-yolo/` (auto-generated from manifest)
- one-class fish detector training set composed from multiple normalized datasets
- built from `configs/datasets/domain_general_fish.json`

## Key Scripts
- `scripts/download_dataset.py`: unified downloader for all downloadable datasets by dataset name
- `scripts/organize_dataset.py`: unified organizer/builder for all normalized/training/generative datasets by dataset name
- `src/dataset_constants.py`: canonical dataset names and aliases
- `src/dataset_download.py`: shared download implementations
- `src/dataset_organize.py`: shared organize/build implementations
- `scripts/fish_cli.py`: CLI for train/run/visualize/validate
- `test.py`: sanity test on a single frame; writes annotated PNG

## Domain-General Training
- `src/pipeline/dataset.py` now accepts a JSON dataset manifest as well as a single normalized dataset root
- `configs/datasets/domain_general_fish.json` now mixes all four sources:
  - `data/interim/aau-zebrafish-reid`
  - `data/interim/mit-sea-grant-river-herring`
  - `data/processed/deep-vision-fish-yolo`
  - `data/processed/kakadu-fishai-yolo`
- `configs/datasets/domain_general_fish_plus_noaa_psnf.json` adds:
  - `data/processed/noaa-puget-sound-nearshore-fish-yolo`
- all boxes are still collapsed to a single detection class: `fish`

## Core Pipeline
`src/pipeline/dataset.py`
- Converts `annotations.csv` to YOLO labels
- Splits by video/clip, or combines multiple prepared YOLO datasets from a manifest
- Uses hardlinks where possible for speed/space
- Supports PNG, JPG, and JPEG inputs

`src/pipeline/train.py`
- Trains YOLO (`yolov8n.pt` base)
- Auto epochs from dataset size
- Saves per-epoch, best, last
- Writes a tee'd train log when `--log` is provided
- Uses `models/runs/<output_stem>/` for per-run artifacts
- Writes `models/latest.pt` + `models/latest.json`

`src/pipeline/track.py`
- `track_folder`: runs tracking on a frames folder or video file and writes CSV
- `visualize_folder`: tracks + draws boxes + IDs and writes an annotated MP4
- Default tracker: BoT-SORT with tuned config
- Jump gating: reassigns display ID if a track "teleports"

`src/pipeline/validate.py`
- Runs Ultralytics validation and writes metrics JSON

`src/data_registry.py`
- Helpers to list datasets/videos and iterate frames w/ GT boxes

## Tracking / ID Behavior
Default tracker is BoT-SORT (better than ByteTrack).
Config: `configs/trackers/botsort_fish.yaml`
- tuned for fixed-camera fish: stricter matching, shorter track buffer
- `with_reid: False` (no learned ReID yet)
This still allows occasional ID swaps when fish cross.

## Environment Overrides
- `FISH_WEIGHTS`: override weights path
- `FISH_TRACKER`: override tracker config (path or name)
- `FISH_MAX_JUMP_PX`: max per-frame jump before reassigning display ID

## Outputs
- `models/`: training runs and weights (ignored by git)
- `outputs/`: tracks CSV and visualization MP4 (ignored by git)

## CUDA Notes
Must use a PyTorch build that supports your GPU architecture.
For RTX 50-series, use a CUDA 12.x build.

## Known Limitations
- IDs can still swap without a fish-specific ReID model.
- Visualize uses predicted tracks, not ground-truth annotations.

## Common Commands
Train:
`python scripts/fish_cli.py train configs/datasets/domain_general_fish.json models/domain_general_fish.pt --log models/domain_general_fish.train.log`

Train with NOAA:
`python scripts/fish_cli.py train configs/datasets/domain_general_fish_plus_noaa_psnf.json models/domain_general_fish_plus_noaa_psnf.pt --log models/domain_general_fish_plus_noaa_psnf.train.log`

Track:
`python scripts/fish_cli.py run data/interim/aau-zebrafish-reid/vid1 outputs/tracks.csv`

Visualize (predictions):
`python scripts/fish_cli.py visualize data/interim/aau-zebrafish-reid/vid1 outputs/visualization.mp4`

Visualize a new video:
`python scripts/fish_cli.py visualize path/to/new_video.mp4 outputs/new_video_annotated.mp4`

Sanity:
`python test.py`
