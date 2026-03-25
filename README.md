# Domain-General Fish Detection and Tracking

This project trains a one-class fish detector for video footage and uses multi-object tracking to keep fish IDs consistent within a clip. The current stack uses YOLOv8n for detection and BoT-SORT for tracking.

## Datasets In Use

These are the four datasets currently used in the active training manifest at `configs/datasets/domain_general_fish.json`.

Find more at: https://github.com/filippovarini/fish-datasets?tab=readme-ov-file

- [AAU Zebrafish ReID](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid): tank footage with fish annotations and identity labels
- [MIT Sea Grant River Herring](https://lila.science/datasets/mit-sea-grant-river-herring/): river passage footage with fish bounding boxes
- [Deep Vision fish dataset](https://metadata.nmdc.no/metadata-api/landingpage/01d102345aef4639f063a13ea20cd3f3): fish detection dataset used to widen visual domain coverage
- [Kakadu FishAI Training Data](https://zenodo.org/records/7250921): underwater freshwater fish detection dataset

Optional fifth source now supported in a separate manifest:

- [NOAA Puget Sound Nearshore Fish 2017-2018](https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-images.zip): estuary images with fish/crab/empty labels; use the NOAA-specific build script to filter ambiguous labels and convert to YOLO

All training labels are collapsed into a single detection class: `fish`.

## Model

- Detector: pretrained `yolov8n.pt`, fine-tuned as a one-class `fish` detector
- Tracker: BoT-SORT with the settings in `configs/trackers/botsort_fish.yaml`
- Identity handling: online tracking only; there is no separate learned ReID model yet
- Current weights: `models/domain_general_fish.pt`
- Architecture summary: `23` stages, `3,011,043` parameters, stride levels `8/16/32`

## Fine-Tuning Setup

The current production run fine-tunes YOLOv8n on the merged 4-dataset manifest built into `data/processed/domain-general-fish-yolo/`.

- Training source: `configs/datasets/domain_general_fish.json`
- Combined dataset size: `271,345` train images and `38,923` validation images
- Run settings: `20` epochs, `960` image size, AutoBatch=`21`, `8` workers, patience=`10`
- Training mode: optimizer=`auto`, AMP enabled, deterministic=`False`

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

Run tracking on a folder of frames or a new video file:

```powershell
python scripts/fish_cli.py run path\to\new_video.mp4 outputs\new_video_tracks.csv --weights models\domain_general_fish.pt
```

Render an annotated video from a folder of frames or a new video file:

```powershell
python scripts/fish_cli.py visualize path\to\new_video.mp4 outputs\new_video_annotated.mp4 --weights models\domain_general_fish.pt
```

If you omit `--weights`, the CLI will use `models/latest.pt` automatically.

## NOAA Optional Dataset

Download the NOAA images and annotations:

```powershell
python scripts/download_dataset.py noaa
```

Build the YOLO-ready dataset:

```powershell
python scripts/organize_dataset.py noaa
```

Train with the NOAA-augmented manifest:

```powershell
python scripts/fish_cli.py train configs/datasets/domain_general_fish_plus_noaa_psnf.json models/domain_general_fish_plus_noaa_psnf.pt --log models/domain_general_fish_plus_noaa_psnf.train.log
```
