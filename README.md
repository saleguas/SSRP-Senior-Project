# Fish Trajectory Tracking Application
This project tracks individual fish in fixed-camera videos using existing
detectors and multi-object tracking methods, and extracts 2D trajectories.

## Folder Layout
- `data/`: all datasets and artifacts (ignored by git)
- `data/raw/`: raw downloads and unmodified extractions
- `data/raw/aau-zebrafish-reid/`: raw Kaggle dataset folder
- `data/raw/aau-zebrafish-reid/aau-zebrafish-reid.zip`: original zip
- `data/raw/aau-zebrafish-reid/annotations.csv`: raw annotations after unzip
- `data/raw/aau-zebrafish-reid/data/`: raw frames after unzip
- `data/interim/`: normalized, consistent layout used by code
- `data/interim/aau-zebrafish-reid/`: normalized dataset root
- `data/interim/aau-zebrafish-reid/annotations.csv`: annotations for all frames
- `data/interim/aau-zebrafish-reid/vid1/`: frames for video 1 (`Vid1_*.png`)
- `data/interim/aau-zebrafish-reid/vid2/`: frames for video 2 (`Vid2_*.png`)
- `data/processed/`: derived artifacts (resized frames, caches, model outputs)
- `scripts/`: one-off utilities for downloading and organizing data
- `src/`: reusable Python code
- `notebooks/`: exploration and experiments
- `docs/`: project notes and references

## Data: AAU Zebrafish Re-Identification (Kaggle)
Dataset slug: `aalborguniversity/aau-zebrafish-reid`

### Kaggle API Setup
- Place `kaggle.json` in `%USERPROFILE%\\.kaggle\\` (Windows) or `~/.kaggle/`
- Or set `KAGGLE_CONFIG_DIR` to a folder containing `kaggle.json`

### Download and Organize
1. Download the zip to `data/raw/aau-zebrafish-reid/`:
```bash
python scripts/download_aau_zebrafish_reid.py
```
2. Unzip the dataset into the raw folder:
```bash
python scripts/download_aau_zebrafish_reid.py --unzip
```
3. Normalize into the repo layout (moves frames + annotations, removes zip):
```bash
python scripts/organize_aau_zebrafish_reid.py --clean-empty
```

Expected normalized layout:
- `data/interim/aau-zebrafish-reid/annotations.csv`
- `data/interim/aau-zebrafish-reid/vid1/*.png`
- `data/interim/aau-zebrafish-reid/vid2/*.png`

## Data Access (Python)
Use `src/data_registry.py` to list datasets and iterate frames with boxes:
```python
from src.data_registry import list_datasets_with_videos, iter_frames

print(list_datasets_with_videos())
for frame in iter_frames("aau-zebrafish-reid", "vid1"):
    print(frame.image_path, len(frame.annotations))
```

## CLI / GUI (Gooey)
Run `python scripts/fish_cli.py` to open the GUI or use it as a CLI.
CUDA GPU is required; the app fails fast if no GPU is available.

Modes:
- `train`: dataset folder -> output weights `.pt` (saves best + last)
- `run`: frames folder -> output tracks `.csv` (uses `models/latest.pt`)
- `validate`: dataset folder -> output metrics `.json` (uses `models/latest.pt`)

If training appears stuck, run from the terminal (not the Gooey window) to see live logs.
Train/validate default to `data/interim/aau-zebrafish-reid` if no dataset path is provided.
