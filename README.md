# Domain-General Fish Detection and Tracking

This project trains a one-class fish detector and uses multi-object tracking to keep IDs consistent within a video. The current stack uses YOLO for detection and BoT-SORT for tracking.

## Datasets In Use

These are the four datasets currently used in the active training manifest at `configs/datasets/domain_general_fish.json`.

- [AAU Zebrafish ReID](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid): tank footage with fish annotations and identity labels
- [MIT Sea Grant River Herring](https://lila.science/datasets/mit-sea-grant-river-herring/): river passage footage with fish bounding boxes
- [Deep Vision fish dataset](https://metadata.nmdc.no/metadata-api/landingpage/01d102345aef4639f063a13ea20cd3f3): fish detection dataset used to widen visual domain coverage
- [Kakadu FishAI Training Data](https://zenodo.org/records/7250921): underwater freshwater fish detection dataset

All training labels are collapsed into a single detection class: `fish`.

## Model

- Detector: YOLOv8n initialized from pretrained `yolov8n.pt`
- Tracking: BoT-SORT with the fixed-camera fish settings in `configs/trackers/botsort_fish.yaml`
- Identity handling: online tracking only; there is no separate learned ReID model in the current pipeline
- Current exported weights: `models/domain_general_fish.pt`

YOLOv8n details for the current setup:

- Classes: `1`
- Parameters: `3,011,043`
- Stages: `23`
- Strides: `8`, `16`, `32`
- Depth multiple: `0.33`
- Width multiple: `0.25`
- Backbone blocks: `Conv`, `C2f`, `SPPF`
- Detection head: multi-scale `Detect` head with upsample/concat feature fusion

## Fine-Tuning Setup

The current production run fine-tunes YOLOv8n on the merged 4-dataset manifest, built into `data/processed/domain-general-fish-yolo/`.

- Training source: `configs/datasets/domain_general_fish.json`
- Combined dataset size: `271,345` train images and `38,923` validation images
- Epochs: `20`
- Image size: `960`
- Batch: Ultralytics AutoBatch, which selected `21` on the final run
- Workers: `8`
- Patience: `10`
- Deterministic mode: `False`
- Optimizer: `auto`
- AMP: enabled

## Train / Validation Split

- AAU Zebrafish ReID and MIT Sea Grant River Herring are normalized as video datasets and split by video folder.
- If a normalized dataset has `5+` video folders, the last `~20%` of folders are used for validation.
- If a normalized dataset has `2-4` video folders, the last folder is used for validation.
- If a normalized dataset has only `1` video folder, it falls back to an `80/20` frame split.
- Deep Vision and Kakadu are prepared as YOLO datasets and keep their own train/validation splits.
- The manifest merge preserves each source split and combines them into one YOLO train/validation root.
- There is currently no separate held-out test set; evaluation is done on the merged validation split.

## CLI

Train the current domain-general detector and write a live log:

```powershell
python scripts/fish_cli.py train configs/datasets/domain_general_fish.json models/domain_general_fish.pt --log models/domain_general_fish.train.log
```

Monitor the training log:

```powershell
Get-Content models\domain_general_fish.train.log -Tail 50 -Wait
```

Run tracking on a folder of frames with the trained weights:

```powershell
python scripts/fish_cli.py run <frames_folder> <tracks_csv> --weights models/domain_general_fish.pt
```

Render a tracked video:

```powershell
python scripts/fish_cli.py visualize <frames_folder> <output_video> --weights models/domain_general_fish.pt
```
