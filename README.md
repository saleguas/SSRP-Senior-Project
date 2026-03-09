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
- `data/raw/mit-sea-grant-river-herring/`: raw LILA river herring downloads
- `data/raw/mit-sea-grant-river-herring/mit_river_herring.zip`: original image zip
- `data/raw/mit-sea-grant-river-herring/mit_sea_grant_river_herring.json.zip`: COCO metadata zip
- `data/raw/deep-vision-fish/`: raw Deep Vision fish dataset download
- `data/raw/deep-vision-fish/fishDatasetSimulationAlgorithm.zip`: original zip
- `data/raw/kakadu-fishai/`: raw Zenodo Kakadu FishAI download
- `data/raw/kakadu-fishai/202210-KakaduFishAI-TrainingData.zip`: original training zip
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
Dataset page: [AAU Zebrafish ReID (Kaggle)](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid)

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

## Data: MIT Sea Grant River Herring (LILA)
Dataset page: [MIT Sea Grant River Herring (LILA)](https://lila.science/datasets/mit-sea-grant-river-herring/)

Current status: raw download and normalization into `data/interim/` are supported.

Download the image zip and COCO metadata zip to `data/raw/mit-sea-grant-river-herring/`:
```bash
python scripts/download_mit_river_herring.py
```

Download only the metadata zip:
```bash
python scripts/download_mit_river_herring.py --metadata-only
```

Normalize the raw zip into the repo's interim layout:
```bash
python scripts/organize_mit_river_herring.py
```

Build a smaller development subset first:
```bash
python scripts/organize_mit_river_herring.py --location coonamessett --max-clips 5 --dest data/interim/mit-sea-grant-river-herring-sample
```

## Data: Deep Vision Fish Dataset
Dataset page: [Deep Vision fish dataset (NMDC)](https://metadata.nmdc.no/metadata-api/landingpage/01d102345aef4639f063a13ea20cd3f3)
Direct zip: [fishDatasetSimulationAlgorithm.zip](https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip)

Current status: raw download is supported; normalization is not wired into the repo yet.

Download the published zip to `data/raw/deep-vision-fish/`:
```bash
python scripts/download_deep_vision_fish.py
```

## Data: Kakadu FishAI Training Data
Dataset page: [A deep learning dataset for underwater object detection of tropical freshwater fish species in northern Australia (Zenodo)](https://zenodo.org/records/7250921)
Direct zip: [202210-KakaduFishAI-TrainingData.zip](https://zenodo.org/records/7250921/files/202210-KakaduFishAI-TrainingData.zip?download=1)

Current status: raw download is supported; normalization is not wired into the repo yet.

Download the training zip to `data/raw/kakadu-fishai/`:
```bash
python scripts/download_kakadu_fishai.py
```

## Domain-General Fish Detector
Current goal: train one detector that generalizes across multiple fish-video domains.

Current manifest:
- `configs/datasets/domain_general_fish.json`
- sources included now: `aau-zebrafish-reid` + `mit-sea-grant-river-herring`

Datasets currently in the project:
- active training sources: [AAU Zebrafish ReID (Kaggle)](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid), [MIT Sea Grant River Herring (LILA)](https://lila.science/datasets/mit-sea-grant-river-herring/)
- downloaded next sources to integrate: [Deep Vision fish dataset (NMDC)](https://metadata.nmdc.no/metadata-api/landingpage/01d102345aef4639f063a13ea20cd3f3), [Kakadu FishAI training data (Zenodo)](https://zenodo.org/records/7250921)

Train the combined one-class fish detector:
```bash
python scripts/fish_cli.py train configs/datasets/domain_general_fish.json models/domain_general_fish.pt
```

The combined YOLO dataset is built automatically at:
- `data/processed/domain-general-fish-yolo/`

Candidate future sources:
- curated repo: https://github.com/filippovarini/fish-datasets
- prioritize adding footage that expands camera/domain coverage, especially aquarium/tank footage if the deployment target is home or lab fish tanks

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
- `visualize`: frames folder -> output video `.mp4` with IDs (uses `models/latest.pt`)
- `validate`: dataset folder -> output metrics `.json` (uses `models/latest.pt`)

If training appears stuck, run from the terminal (not the Gooey window) to see live logs.
Train/validate default to `data/interim/aau-zebrafish-reid` if no dataset path is provided.
