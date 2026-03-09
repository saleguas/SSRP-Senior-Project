# ML Context for This Codebase
This is a concise, technical map of the repo for an ML agent or engineer.

## Project Goal
Track individual fish in fixed-camera footage. Output 2D trajectories and
consistent IDs over time. No 3D or pose.

## Stack
- Python, PyTorch, Ultralytics YOLO
- OpenCV for visualization
- Gooey for GUI (optional)
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

## Data Layout (normalized via organizer)
`data/interim/mit-sea-grant-river-herring/`
- `annotations.csv`: normalized boxes with category labels
- `<clip_name>/`: PNG frames renamed to unique per-clip filenames

`data/processed/aau-zebrafish-reid-yolo/` (auto-generated)
- YOLO format train/val split by video
- `data.yaml` used by Ultralytics

`data/processed/domain-general-fish-yolo/` (auto-generated from manifest)
- one-class fish detector training set composed from multiple normalized datasets
- built from `configs/datasets/domain_general_fish.json`

## Key Scripts
- `scripts/download_aau_zebrafish_reid.py`: download Kaggle dataset
- `scripts/download_mit_river_herring.py`: download the LILA river herring image + metadata zips
- `scripts/organize_aau_zebrafish_reid.py`: normalize into `data/interim/`
- `scripts/organize_mit_river_herring.py`: extract and normalize the LILA COCO dataset into the repo layout
- `scripts/fish_cli.py`: CLI/GUI for train/run/visualize/validate
- `test.py`: sanity test on a single frame; writes annotated PNG

## Domain-General Training
- `src/pipeline/dataset.py` now accepts a JSON dataset manifest as well as a single normalized dataset root
- `configs/datasets/domain_general_fish.json` currently mixes `data/interim/aau-zebrafish-reid` and `data/interim/mit-sea-grant-river-herring`
- all boxes are still collapsed to a single detection class: `fish`

## Core Pipeline
`src/pipeline/dataset.py`
- Converts `annotations.csv` to YOLO labels
- Splits by video/clip, or combines multiple prepared YOLO datasets from a manifest
- Uses hardlinks where possible for speed/space

`src/pipeline/train.py`
- Trains YOLO (`yolov8n.pt` base)
- Auto epochs from dataset size
- Saves per-epoch, best, last
- Writes `models/latest.pt` + `models/latest.json`

`src/pipeline/track.py`
- `track_folder`: runs tracking and writes CSV
- `visualize_folder`: tracks + draws boxes + IDs and stitches MP4
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
`python scripts/fish_cli.py train`

Track:
`python scripts/fish_cli.py run data/interim/aau-zebrafish-reid/vid1 outputs/tracks.csv`

Visualize (predictions):
`python scripts/fish_cli.py visualize data/interim/aau-zebrafish-reid/vid1 outputs/visualization.mp4`

Sanity:
`python test.py`
