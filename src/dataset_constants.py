from __future__ import annotations

from dataclasses import dataclass

AAU_ZEBRAFISH_REID = "aau-zebrafish-reid"
DEEP_VISION_FISH = "deep-vision-fish"
KAKADU_FISHAI = "kakadu-fishai"
LIAO_LAB_VIDEOS = "liao-lab-videos"
MIT_RIVER_HERRING = "mit-sea-grant-river-herring"
NOAA_PUGET_SOUND_NEARSHORE_FISH = "noaa-puget-sound-nearshore-fish"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    role: str
    downloadable: bool


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name=AAU_ZEBRAFISH_REID,
        aliases=("aau", "zebrafish", "zebrafish-reid"),
        role="training",
        downloadable=True,
    ),
    DatasetSpec(
        name=DEEP_VISION_FISH,
        aliases=("deepvision", "deep-vision"),
        role="training",
        downloadable=True,
    ),
    DatasetSpec(
        name=KAKADU_FISHAI,
        aliases=("kakadu", "fishai"),
        role="training",
        downloadable=True,
    ),
    DatasetSpec(
        name=LIAO_LAB_VIDEOS,
        aliases=("liao", "liao-lab"),
        role="generative",
        downloadable=False,
    ),
    DatasetSpec(
        name=MIT_RIVER_HERRING,
        aliases=("mit", "river-herring", "mit-river-herring"),
        role="training",
        downloadable=True,
    ),
    DatasetSpec(
        name=NOAA_PUGET_SOUND_NEARSHORE_FISH,
        aliases=("noaa", "psnf", "puget-sound"),
        role="training",
        downloadable=True,
    ),
)

TRAINING_DATASET_NAMES = tuple(
    spec.name for spec in DATASET_SPECS if spec.role == "training"
)
GENERATIVE_DATASET_NAMES = tuple(
    spec.name for spec in DATASET_SPECS if spec.role == "generative"
)
DOWNLOADABLE_DATASET_NAMES = tuple(
    spec.name for spec in DATASET_SPECS if spec.downloadable
)
ALL_DATASET_NAMES = tuple(spec.name for spec in DATASET_SPECS)

_DATASET_SPEC_BY_KEY = {
    key: spec
    for spec in DATASET_SPECS
    for key in (spec.name, *spec.aliases)
}


def get_dataset_spec(value: str) -> DatasetSpec:
    key = value.strip().lower()
    if key not in _DATASET_SPEC_BY_KEY:
        valid = ", ".join(ALL_DATASET_NAMES)
        raise KeyError(f"Unknown dataset '{value}'. Expected one of: {valid}")
    return _DATASET_SPEC_BY_KEY[key]
