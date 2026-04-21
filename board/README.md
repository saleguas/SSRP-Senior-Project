# Board Assets

This folder contains presentation-ready graphs and still images generated from the current repository state.

## Key Numbers

- Actual training build on disk: `25,702` images total from the available-manifest merge.
- Largest source in the current build: `3d-zef20`.
- Best `YOLO11s` mAP50-95: `0.860`.
- Best `YOLO11m` mAP50-95: `0.878`.
- Liao escape clip best match rate in saved checks: `52.3%`.

## Figures

- `figures/dataset_composition.png`
- `figures/training_metrics.png`
- `figures/model_tradeoff.png`
- `figures/liao_escape_comparison.png`
- `figures/liao_track_quality.png`
- `figures/liao_trajectory_overlay.png`
- `figures/domain_generalization_montage.png`

## Photos

- `photos/aau_zebrafish_frame.jpg`
- `photos/liao_escape_success.jpg`
- `photos/liao_school_coords.jpg`
- `photos/liao_school_failure.jpg`
- `photos/mit_river_herring_frame.jpg`
- `photos/noaa_frame.jpg`

## Notes

- `mAP50-95` means mean Average Precision across box-overlap thresholds from `0.50` to `0.95`; higher is better.
- `Precision` means when the model predicts a fish box, how often that prediction is correct.
- `Recall` means out of all real fish present, how many the model successfully finds.
- The `YOLOv8n` series in comparison figures is a reconstructed baseline, not a recovered measured run.
- `liao_escape_comparison.png` compares the three saved coordinate-check JSON files for the same escape clip.
- `model_tradeoff.png` uses best mAP50-95 and final reported training time from each saved run.
- `domain_generalization_montage.png` uses extracted frames from annotated output videos plus the saved Liao still.
