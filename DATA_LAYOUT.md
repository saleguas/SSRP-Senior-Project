# Data Layout

Use one storage path per dataset stage:

```text
data/
  raw/
    <dataset-name>/            downloaded zips or manually placed archives
  interim/
    <dataset-name>/            normalized non-YOLO datasets such as AAU or MIT
  processed/
    <dataset-name>-yolo/       per-dataset YOLO-ready roots
  training/
    manifests/                 generated multi-dataset manifests
    domain-general-fish-all-yolo/
    domain-general-fish-available-yolo/
  generative/
    <dataset-name>/            generation-only inputs such as Liao
```

## Workflow

Create the layout and inspect readiness:

```powershell
python scripts/dataset_status.py --init
```

Download or manually place archives into `data/raw/<dataset-name>/`.

Organize one dataset at a time:

```powershell
python scripts/organize_dataset.py aau
python scripts/organize_dataset.py mit
python scripts/organize_dataset.py deepvision
python scripts/organize_dataset.py kakadu
python scripts/organize_dataset.py noaa
python scripts/organize_dataset.py 3d-zef20 --source-zip "C:\Users\game\Downloads\3DZeF20.zip"
python scripts/organize_dataset.py liao --source-zip "C:\Users\game\Documents\quick\Liao-lab-videos.zip"
```

Build a merged training dataset from whatever training datasets are ready:

```powershell
python scripts/prepare_training_data.py --available-only --force
```

That generates:

- `data/training/manifests/domain_general_fish_available.json`
- `data/training/domain-general-fish-available-yolo`

Train against the merged root:

```powershell
python scripts/fish_cli.py train data/training/domain-general-fish-available-yolo models/domain_general_fish.pt
```
