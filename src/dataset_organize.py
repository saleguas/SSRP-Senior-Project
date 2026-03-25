from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

from src.dataset_constants import (
    AAU_ZEBRAFISH_REID,
    DEEP_VISION_FISH,
    KAKADU_FISHAI,
    LIAO_LAB_VIDEOS,
    MIT_RIVER_HERRING,
    NOAA_PUGET_SOUND_NEARSHORE_FISH,
)
from src.liao_lab import discover_render_sources, find_fish_coords_xlsx, render_source_slug
from src.pipeline.utils import is_image_file, link_or_copy, write_json

AAU_ZIP_NAME = "aau-zebrafish-reid.zip"
DEEP_VISION_ZIP_NAME = "fishDatasetSimulationAlgorithm.zip"
KAKADU_ZIP_NAME = "202210-KakaduFishAI-TrainingData.zip"
KAKADU_ANNOTATIONS_NAME = "KakaduFishAI_boundingbox.json"
MIT_IMAGE_ZIP_NAME = "mit_river_herring.zip"
MIT_METADATA_ZIP_NAME = "mit_sea_grant_river_herring.json.zip"
MIT_METADATA_JSON_NAME = "mit_sea_grant_river_herring.json"
MIT_ARCHIVE_ROOT = "mit_river_herring"
MIT_COPY_CHUNK_SIZE = 8 * 1024 * 1024
NOAA_IMAGES_ZIP_NAME = "noaa_estuary_fish-images.zip"
NOAA_ANNOTATIONS_ZIP_NAME = "noaa_estuary_fish-annotations-2023.08.19.zip"
NOAA_ANNOTATIONS_JSON_NAME = "noaa_estuary_fish-2023.08.19.json"
LIAO_DEFAULT_SOURCE_ZIP = r"C:\Users\game\Documents\quick\Liao-lab-videos.zip"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_local_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def yolo_lines(
    boxes: Iterable[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> List[str]:
    lines: List[str] = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0.0, min(x1, width - 1))
        y1 = max(0.0, min(y1, height - 1))
        x2 = max(0.0, min(x2, width))
        y2 = max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        xc = (x1 + x2) / 2.0 / width
        yc = (y1 + y2) / 2.0 / height
        w = (x2 - x1) / float(width)
        h = (y2 - y1) / float(height)
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def write_label(label_path: Path, lines: List[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def organize_aau(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Reorganize AAU Zebrafish ReID dataset into vid folders."
    )
    parser.add_argument(
        "--source",
        default=f"data/raw/{AAU_ZEBRAFISH_REID}",
        help="Source dataset directory containing the raw unzip.",
    )
    parser.add_argument(
        "--dest",
        default=f"data/interim/{AAU_ZEBRAFISH_REID}",
        help="Destination dataset directory.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the zip file in the source directory.",
    )
    parser.add_argument(
        "--clean-empty",
        action="store_true",
        help="Remove empty source folders after organizing.",
    )
    args = parser.parse_args(argv)

    source_dir = resolve_local_path(args.source)
    dest_dir = resolve_local_path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def move_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

    moved = 0
    skipped = 0
    for prefix in ("Vid1", "Vid2"):
        target_dir = dest_dir / prefix.lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(source_dir.rglob(f"{prefix}_*.png")):
            dst = target_dir / src.name
            if dst.exists():
                skipped += 1
                continue
            move_or_copy(src, dst)
            moved += 1

    annotations_src = source_dir / "annotations.csv"
    annotations_dst = dest_dir / "annotations.csv"
    if annotations_src.exists() and not annotations_dst.exists():
        move_or_copy(annotations_src, annotations_dst)

    if not args.keep_zip:
        zip_path = source_dir / AAU_ZIP_NAME
        if zip_path.exists():
            zip_path.unlink()

    if args.clean_empty and source_dir.exists():
        for child in sorted(source_dir.rglob("*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            source_dir.rmdir()
        except OSError:
            pass

    print(
        f"Organized {AAU_ZEBRAFISH_REID}: moved {moved} files, skipped {skipped} existing files."
    )
    print(f"Destination: {dest_dir}")
    return 0


def organize_deep_vision(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset root for the Deep Vision fish dataset."
    )
    parser.add_argument(
        "--source",
        default=f"data/raw/{DEEP_VISION_FISH}",
        help="Raw dataset directory.",
    )
    parser.add_argument(
        "--dest",
        default=f"data/processed/{DEEP_VISION_FISH}-yolo",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the destination even if it already exists.",
    )
    args = parser.parse_args(argv)

    source_dir = resolve_local_path(args.source)
    output_root = resolve_local_path(args.dest)

    def ensure_extracted(raw_root: Path) -> Path:
        extracted_root = raw_root / "unzipped" / "fish_dataset"
        if extracted_root.exists():
            return extracted_root
        zip_path = raw_root / DEEP_VISION_ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"Deep Vision zip not found: {zip_path}")
        extracted_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(zip_path), str(extracted_root.parent))
        if not extracted_root.exists():
            raise FileNotFoundError(f"Deep Vision extraction failed: {extracted_root}")
        return extracted_root

    def normalize_rel_path(value: str) -> Path:
        return Path(value.strip().lstrip("/\\"))

    def read_rows(csv_path: Path) -> List[Tuple[Path, Tuple[int, int, int, int]]]:
        rows: List[Tuple[Path, Tuple[int, int, int, int]]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 5:
                    continue
                rel_path = normalize_rel_path(row[0])
                bbox = tuple(int(float(value)) for value in row[1:5])
                rows.append((rel_path, bbox))  # type: ignore[arg-type]
        return rows

    def relative_name(rel_path: Path) -> str:
        parts = [part.replace(" ", "-") for part in rel_path.parts]
        return "__".join(parts)

    def build_basename_index(dataset_root: Path) -> Tuple[Dict[str, List[Path]], List[Path]]:
        index: Dict[str, List[Path]] = defaultdict(list)
        all_images: List[Path] = []
        for path in dataset_root.rglob("*"):
            if is_image_file(path):
                index[path.name].append(path)
                all_images.append(path)
        return index, all_images

    def resolve_image_path(
        dataset_root: Path,
        basename_index: Dict[str, List[Path]],
        all_images: List[Path],
        rel_path: Path,
    ) -> Path:
        source_image = dataset_root / rel_path
        if source_image.exists():
            return source_image

        candidates = basename_index.get(rel_path.name, [])
        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1 and len(rel_path.parts) >= 2:
            year, split = rel_path.parts[0], rel_path.parts[1]
            narrowed = [path for path in candidates if year in path.parts and split in path.parts]
            if len(narrowed) == 1:
                return narrowed[0]

        suffix_matches = [path for path in all_images if path.name.endswith(rel_path.name)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        if len(suffix_matches) > 1 and len(rel_path.parts) >= 2:
            year, split = rel_path.parts[0], rel_path.parts[1]
            narrowed = [path for path in suffix_matches if year in path.parts and split in path.parts]
            if len(narrowed) == 1:
                return narrowed[0]

        raise FileNotFoundError(f"Missing Deep Vision image: {source_image}")

    def prepare_split_maps(
        dataset_root: Path,
    ) -> Tuple[Dict[Path, List[Tuple[int, int, int, int]]], Dict[Path, List[Tuple[int, int, int, int]]]]:
        train_csvs = [
            dataset_root / "2017" / "train" / "source-train2017-annotations.csv",
            dataset_root / "2018" / "train" / "source-train2018-annotations.csv",
            dataset_root / "2017" / "test" / "test_2017_annotations.csv",
            dataset_root / "2018" / "test" / "test_2018_annotations.csv",
        ]
        val_csv = dataset_root / "val_annotations.csv"

        val_rows = read_rows(val_csv)
        val_paths = {path for path, _ in val_rows}

        train_map: Dict[Path, List[Tuple[int, int, int, int]]] = defaultdict(list)
        val_map: Dict[Path, List[Tuple[int, int, int, int]]] = defaultdict(list)

        for rel_path, bbox in val_rows:
            val_map[rel_path].append(bbox)

        for csv_path in train_csvs:
            for rel_path, bbox in read_rows(csv_path):
                if rel_path in val_paths:
                    continue
                train_map[rel_path].append(bbox)

        return dict(train_map), dict(val_map)

    dataset_root = ensure_extracted(source_dir)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Destination already exists: {output_root}. Use --force to rebuild."
            )
        shutil.rmtree(output_root, ignore_errors=True)

    train_map, val_map = prepare_split_maps(dataset_root)
    basename_index, all_images = build_basename_index(dataset_root)
    counts = {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}

    for split_name, image_map in (("train", train_map), ("val", val_map)):
        images_dir = output_root / "images" / split_name
        labels_dir = output_root / "labels" / split_name
        for rel_path, boxes in sorted(image_map.items()):
            source_image = resolve_image_path(dataset_root, basename_index, all_images, rel_path)
            with Image.open(source_image) as img:
                width, height = img.size

            linked_name = relative_name(rel_path)
            dest_image = images_dir / linked_name
            link_or_copy(source_image, dest_image)

            label_path = labels_dir / f"{Path(linked_name).stem}.txt"
            write_label(label_path, yolo_lines(boxes, width, height))

            counts[f"{split_name}_images"] += 1
            counts[f"{split_name}_boxes"] += len(boxes)

    yaml_path = output_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: fish",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        output_root / "build_info.json",
        {
            "name": DEEP_VISION_FISH,
            "source_root": str(dataset_root),
            "counts": counts,
        },
    )
    print(f"Built {DEEP_VISION_FISH} YOLO dataset at {output_root}")
    print(
        f"train images={counts['train_images']} val images={counts['val_images']} "
        f"train boxes={counts['train_boxes']} val boxes={counts['val_boxes']}"
    )
    return 0


def organize_kakadu(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset root for the Kakadu FishAI dataset."
    )
    parser.add_argument(
        "--source",
        default=f"data/raw/{KAKADU_FISHAI}",
        help="Raw dataset directory.",
    )
    parser.add_argument(
        "--dest",
        default=f"data/processed/{KAKADU_FISHAI}-yolo",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the destination even if it already exists.",
    )
    args = parser.parse_args(argv)

    source_dir = resolve_local_path(args.source)
    output_root = resolve_local_path(args.dest)
    val_modulus = 10

    def find_7z() -> Path:
        candidates = [
            shutil.which("7z"),
            shutil.which("7z.exe"),
            str(Path("C:/Program Files/7-Zip/7z.exe")),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return Path(candidate)
        raise FileNotFoundError(
            "7-Zip was not found. Install 7-Zip or extract the Kakadu zip manually."
        )

    def ensure_extracted(raw_root: Path) -> Path:
        extracted_root = raw_root / "unzipped"
        annotations_path = extracted_root / KAKADU_ANNOTATIONS_NAME
        if annotations_path.exists():
            return extracted_root
        zip_path = raw_root / KAKADU_ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"Kakadu zip not found: {zip_path}")
        extracted_root.mkdir(parents=True, exist_ok=True)
        seven_zip = find_7z()
        subprocess.run(
            [str(seven_zip), "x", "-y", f"-o{extracted_root}", str(zip_path)],
            check=True,
        )
        if not annotations_path.exists():
            raise FileNotFoundError(f"Kakadu extraction failed: {annotations_path}")
        return extracted_root

    dataset_root = ensure_extracted(source_dir)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Destination already exists: {output_root}. Use --force to rebuild."
            )
        shutil.rmtree(output_root, ignore_errors=True)

    payload = json.loads((dataset_root / KAKADU_ANNOTATIONS_NAME).read_text(encoding="utf-8"))
    images = {image["id"]: image for image in payload["images"]}
    annotations_by_image: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        x, y, w, h = annotation["bbox"]
        annotations_by_image[int(annotation["image_id"])].append(
            (int(x), int(y), int(x + w), int(y + h))
        )

    counts = {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}
    for image_id, image in sorted(images.items()):
        split_name = "val" if int(image_id) % val_modulus == 0 else "train"
        image_name = str(image["file_name"])
        source_image = dataset_root / image_name
        if not source_image.exists():
            raise FileNotFoundError(f"Missing Kakadu image: {source_image}")

        images_dir = output_root / "images" / split_name
        labels_dir = output_root / "labels" / split_name
        dest_image = images_dir / image_name
        link_or_copy(source_image, dest_image)

        width = int(image["width"])
        height = int(image["height"])
        boxes = annotations_by_image.get(int(image_id), [])
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        write_label(label_path, yolo_lines(boxes, width, height))

        counts[f"{split_name}_images"] += 1
        counts[f"{split_name}_boxes"] += len(boxes)

    yaml_path = output_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: fish",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        output_root / "build_info.json",
        {
            "name": KAKADU_FISHAI,
            "source_root": str(dataset_root),
            "val_modulus": val_modulus,
            "counts": counts,
        },
    )
    print(f"Built {KAKADU_FISHAI} YOLO dataset at {output_root}")
    print(
        f"train images={counts['train_images']} val images={counts['val_images']} "
        f"train boxes={counts['train_boxes']} val boxes={counts['val_boxes']}"
    )
    return 0


@dataclass(frozen=True)
class MitImageRecord:
    image_id: str
    archive_name: str
    clip_key: str
    clip_name: str
    file_name: str
    dest_path: Path
    width: int
    height: int
    location: str


def organize_mit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize the MIT Sea Grant River Herring dataset."
    )
    parser.add_argument(
        "--source-zip",
        default=f"data/raw/{MIT_RIVER_HERRING}/{MIT_IMAGE_ZIP_NAME}",
        help="Path to the downloaded image zip.",
    )
    parser.add_argument(
        "--metadata-json",
        default=f"data/raw/{MIT_RIVER_HERRING}/metadata/{MIT_METADATA_JSON_NAME}",
        help="Path to the extracted metadata JSON.",
    )
    parser.add_argument(
        "--metadata-zip",
        default=f"data/raw/{MIT_RIVER_HERRING}/{MIT_METADATA_ZIP_NAME}",
        help="Path to the metadata zip (used if --metadata-json is missing).",
    )
    parser.add_argument(
        "--dest",
        default=f"data/interim/{MIT_RIVER_HERRING}",
        help="Destination dataset directory.",
    )
    parser.add_argument(
        "--location",
        action="append",
        default=[],
        help="Restrict to one or more locations (repeatable): coonamessett, ipswich, santuit.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit organization to the first N clips after filtering.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted images.",
    )
    args = parser.parse_args(argv)

    source_zip = resolve_local_path(args.source_zip)
    metadata_json = resolve_local_path(args.metadata_json)
    metadata_zip = resolve_local_path(args.metadata_zip)
    dest_root = resolve_local_path(args.dest)

    if not source_zip.exists():
        raise FileNotFoundError(f"Image zip not found: {source_zip}")

    def load_metadata(metadata_json_path: Path, metadata_zip_path: Path) -> dict:
        if metadata_json_path.exists():
            with metadata_json_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        if metadata_zip_path.exists():
            with zipfile.ZipFile(metadata_zip_path) as archive:
                with archive.open(MIT_METADATA_JSON_NAME) as handle:
                    with io.TextIOWrapper(handle, encoding="utf-8") as reader:
                        return json.load(reader)
        raise FileNotFoundError(
            f"Metadata not found. Checked {metadata_json_path} and {metadata_zip_path}."
        )

    def clip_key(file_name: str) -> str:
        parts = file_name.split("/")
        if len(parts) < 2:
            raise ValueError(f"Unexpected image path: {file_name}")
        return "/".join(parts[:2])

    def sanitize_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    def clip_name(value: str) -> str:
        left, right = value.split("/", 1)
        return f"{sanitize_name(left)}_{sanitize_name(right)}"

    def frame_name(clip_value: str, original_file_name: str) -> str:
        stem = Path(original_file_name).stem.lower()
        return f"{clip_value}_{stem}.png"

    def iter_selected_images(images: Iterable[dict]) -> List[MitImageRecord]:
        normalized_locations = {value.strip().lower() for value in args.location if value.strip()}
        filtered_images = [
            image
            for image in images
            if not normalized_locations
            or str(image.get("location", "")).strip().lower() in normalized_locations
        ]
        ordered_clip_keys = sorted({clip_key(image["file_name"]) for image in filtered_images})
        if args.max_clips is not None:
            ordered_clip_keys = ordered_clip_keys[: max(0, args.max_clips)]
        allowed_clips = set(ordered_clip_keys)

        records: List[MitImageRecord] = []
        for image in filtered_images:
            image_clip_key = clip_key(image["file_name"])
            if image_clip_key not in allowed_clips:
                continue
            normalized_clip_name = clip_name(image_clip_key)
            file_name = frame_name(normalized_clip_name, image["file_name"])
            records.append(
                MitImageRecord(
                    image_id=str(image["id"]),
                    archive_name=f"{MIT_ARCHIVE_ROOT}/{image['file_name']}",
                    clip_key=image_clip_key,
                    clip_name=normalized_clip_name,
                    file_name=file_name,
                    dest_path=dest_root / normalized_clip_name / file_name,
                    width=int(image["width"]),
                    height=int(image["height"]),
                    location=str(image.get("location", "")),
                )
            )
        return records

    def extract_images(image_records: List[MitImageRecord]) -> tuple[int, int]:
        extracted = 0
        skipped = 0
        missing: List[str] = []
        with zipfile.ZipFile(source_zip) as archive:
            total = len(image_records)
            for index, record in enumerate(image_records, start=1):
                record.dest_path.parent.mkdir(parents=True, exist_ok=True)
                if record.dest_path.exists() and not args.force:
                    skipped += 1
                else:
                    try:
                        with archive.open(record.archive_name) as src, record.dest_path.open(
                            "wb"
                        ) as dst:
                            shutil.copyfileobj(src, dst, MIT_COPY_CHUNK_SIZE)
                    except KeyError:
                        missing.append(record.archive_name)
                        continue
                    extracted += 1

                if index % 1000 == 0 or index == total:
                    print(
                        f"Processed {index}/{total} images "
                        f"(extracted {extracted}, skipped {skipped})"
                    )

        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} files in archive. First missing entries: {preview}"
            )
        return extracted, skipped

    def bbox_xyxy(annotation: dict, width: int, height: int) -> tuple[int, int, int, int]:
        x, y, w, h = annotation["bbox"]
        x1 = max(0, min(width - 1, int(math.floor(x))))
        y1 = max(0, min(height - 1, int(math.floor(y))))
        x2 = max(x1 + 1, min(width, int(math.ceil(x + w))))
        y2 = max(y1 + 1, min(height, int(math.ceil(y + h))))
        return x1, y1, x2, y2

    def write_annotations_csv(
        image_records: List[MitImageRecord],
        annotations: Iterable[dict],
        categories: Iterable[dict],
    ) -> int:
        csv_header = [
            "Filename",
            "Object ID",
            "Annotation tag",
            "Upper left corner X",
            "Upper left corner Y",
            "Lower right corner X",
            "Lower right corner Y",
            "Right,Turning,Occlusion,Glitch",
            "Buffer",
            "",
            "",
        ]
        category_names = {int(item["id"]): str(item["name"]) for item in categories}
        image_lookup = {record.image_id: record for record in image_records}
        annotations_by_image: Dict[str, List[dict]] = defaultdict(list)

        for annotation in annotations:
            image_id = str(annotation.get("image_id"))
            if image_id in image_lookup:
                annotations_by_image[image_id].append(annotation)

        annotations_path = dest_root / "annotations.csv"
        row_count = 0
        with annotations_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(csv_header)

            for record in image_records:
                image_annotations = sorted(
                    annotations_by_image.get(record.image_id, []),
                    key=lambda item: str(item.get("id", "")),
                )
                for object_id, annotation in enumerate(image_annotations, start=1):
                    x1, y1, x2, y2 = bbox_xyxy(annotation, record.width, record.height)
                    writer.writerow(
                        [
                            record.file_name,
                            object_id,
                            category_names.get(int(annotation["category_id"]), "fish"),
                            x1,
                            y1,
                            x2,
                            y2,
                            "0,0,0,0",
                            0,
                            "",
                            "",
                        ]
                    )
                    row_count += 1
        return row_count

    coco = load_metadata(metadata_json, metadata_zip)
    image_records = iter_selected_images(coco.get("images", []))
    if not image_records:
        raise SystemExit("No images matched the requested filters.")

    selected_clips = sorted({record.clip_name for record in image_records})
    print(
        f"Selected {len(image_records)} images across {len(selected_clips)} clips into {dest_root}"
    )

    extracted, skipped = extract_images(image_records)
    annotation_rows = write_annotations_csv(
        image_records=image_records,
        annotations=coco.get("annotations", []),
        categories=coco.get("categories", []),
    )

    print(
        f"Organized {MIT_RIVER_HERRING}: extracted {extracted} images, "
        f"skipped {skipped} existing images, wrote {annotation_rows} annotation rows."
    )
    return 0


def organize_noaa(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset root for the NOAA Puget Sound Nearshore Fish dataset."
    )
    parser.add_argument(
        "--source",
        default=f"data/raw/{NOAA_PUGET_SOUND_NEARSHORE_FISH}",
        help="Raw dataset directory.",
    )
    parser.add_argument(
        "--dest",
        default=f"data/processed/{NOAA_PUGET_SOUND_NEARSHORE_FISH}-yolo",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the destination even if it already exists.",
    )
    args = parser.parse_args(argv)

    source_dir = resolve_local_path(args.source)
    output_root = resolve_local_path(args.dest)
    positive_category = "fish"
    negative_categories = {"empty", "crab"}
    excluded_categories = {"fish_or_crab", "unknown"}
    val_location_modulus = 5

    def first_image(images_root: Path) -> Path | None:
        for path in images_root.rglob("*"):
            if is_image_file(path):
                return path
        return None

    def ensure_annotations_extracted(raw_root: Path) -> Path:
        annotations_root = raw_root / "unzipped" / "annotations"
        annotations_path = annotations_root / NOAA_ANNOTATIONS_JSON_NAME
        if annotations_path.exists():
            return annotations_path
        zip_path = raw_root / NOAA_ANNOTATIONS_ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"NOAA annotations zip not found: {zip_path}")
        annotations_root.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(zip_path), str(annotations_root))
        if not annotations_path.exists():
            raise FileNotFoundError(f"NOAA annotation extraction failed: {annotations_path}")
        return annotations_path

    def ensure_images_extracted(raw_root: Path) -> Path:
        images_root = raw_root / "unzipped" / "images"
        if first_image(images_root) is not None:
            return images_root
        zip_path = raw_root / NOAA_IMAGES_ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"NOAA images zip not found: {zip_path}")
        images_root.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(zip_path), str(images_root))
        if first_image(images_root) is None:
            raise FileNotFoundError(f"NOAA image extraction failed under {images_root}")
        return images_root

    def build_basename_index(images_root: Path) -> Dict[str, List[Path]]:
        index: Dict[str, List[Path]] = defaultdict(list)
        for path in images_root.rglob("*"):
            if is_image_file(path):
                index[path.name].append(path)
        return index

    def resolve_image_path(
        images_root: Path, basename_index: Dict[str, List[Path]], file_name: str
    ) -> Path:
        direct = images_root / file_name
        if direct.exists():
            return direct
        candidates = basename_index.get(Path(file_name).name, [])
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"Missing NOAA image: {file_name}")
        raise FileNotFoundError(
            f"Multiple NOAA images matched {file_name}; expected one, found {len(candidates)}"
        )

    def category_name(annotation: dict, categories: Dict[int, str]) -> str:
        category_id = int(annotation["category_id"])
        return categories.get(category_id, str(category_id))

    def select_val_locations(images: Iterable[dict]) -> Tuple[List[str], str]:
        import hashlib

        locations = sorted({str(image["location"]) for image in images})
        if not locations:
            return ([], "none")
        selected = [
            location
            for location in locations
            if int(hashlib.md5(location.encode("utf-8")).hexdigest(), 16)
            % val_location_modulus
            == 0
        ]
        if 0 < len(selected) < len(locations):
            return (sorted(selected), f"md5_mod_{val_location_modulus}")
        val_count = max(1, int(round(len(locations) * 0.2)))
        return (locations[-val_count:], "sorted_tail_fallback")

    def fish_boxes(
        annotations: Iterable[dict],
        categories: Dict[int, str],
    ) -> List[Tuple[float, float, float, float]]:
        boxes: List[Tuple[float, float, float, float]] = []
        for annotation in annotations:
            if category_name(annotation, categories) != positive_category:
                continue
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x, y, w, h = (float(value) for value in bbox[:4])
            boxes.append((x, y, x + w, y + h))
        return boxes

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Destination already exists: {output_root}. Use --force to rebuild."
            )
        shutil.rmtree(output_root, ignore_errors=True)

    annotations_path = ensure_annotations_extracted(source_dir)
    images_root = ensure_images_extracted(source_dir)
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {annotations_path}")

    categories = {int(category["id"]): str(category["name"]) for category in payload["categories"]}
    images = {str(image["id"]): image for image in payload["images"]}
    annotations_by_image: Dict[str, List[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[str(annotation["image_id"])].append(annotation)

    basename_index = build_basename_index(images_root)
    val_locations, split_policy = select_val_locations(images.values())
    val_location_set = set(val_locations)

    counts = Counter()
    excluded_by_reason = Counter()
    for image_id, image in sorted(images.items(), key=lambda item: str(item[1]["file_name"])):
        annotations = annotations_by_image.get(image_id, [])
        labels = {category_name(annotation, categories) for annotation in annotations}
        excluded_labels = sorted(label for label in labels if label in excluded_categories)
        if excluded_labels:
            counts["excluded_images"] += 1
            for label in excluded_labels:
                excluded_by_reason[label] += 1
            continue

        unexpected = sorted(
            label
            for label in labels
            if label and label not in negative_categories and label != positive_category
        )
        if unexpected:
            counts["excluded_images"] += 1
            for label in unexpected:
                excluded_by_reason[label] += 1
            continue

        split_name = "val" if str(image["location"]) in val_location_set else "train"
        source_image = resolve_image_path(images_root, basename_index, str(image["file_name"]))
        with Image.open(source_image) as handle:
            width, height = handle.size

        label_lines = yolo_lines(fish_boxes(annotations, categories), width, height)
        dest_image = output_root / "images" / split_name / source_image.name
        label_path = output_root / "labels" / split_name / f"{source_image.stem}.txt"
        link_or_copy(source_image, dest_image)
        write_label(label_path, label_lines)

        counts[f"{split_name}_images"] += 1
        counts[f"{split_name}_boxes"] += len(label_lines)
        if label_lines:
            counts[f"{split_name}_positive_images"] += 1
        else:
            counts[f"{split_name}_negative_images"] += 1

    yaml_path = output_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: fish",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_info = {
        "name": NOAA_PUGET_SOUND_NEARSHORE_FISH,
        "source_root": str(source_dir),
        "images_root": str(images_root),
        "annotations_path": str(annotations_path),
        "positive_category": positive_category,
        "negative_categories": sorted(negative_categories),
        "excluded_categories": sorted(excluded_categories),
        "split_policy": split_policy,
        "val_location_modulus": val_location_modulus,
        "val_locations": val_locations,
        "counts": dict(counts),
        "excluded_by_reason": dict(excluded_by_reason),
    }
    write_json(output_root / "build_info.json", build_info)
    print(f"Built {NOAA_PUGET_SOUND_NEARSHORE_FISH} YOLO dataset at {output_root}")
    print(
        f"train images={counts.get('train_images', 0)} val images={counts.get('val_images', 0)} "
        f"train boxes={counts.get('train_boxes', 0)} val boxes={counts.get('val_boxes', 0)} "
        f"excluded images={counts.get('excluded_images', 0)}"
    )
    print(
        f"train negatives={counts.get('train_negative_images', 0)} "
        f"val negatives={counts.get('val_negative_images', 0)}"
    )
    return 0


def organize_liao(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Extract the Liao lab archive into the project's generation-data layout."
    )
    parser.add_argument(
        "--source-zip",
        default=LIAO_DEFAULT_SOURCE_ZIP,
        help="Path to Liao-lab-videos.zip.",
    )
    parser.add_argument(
        "--dest",
        default=f"data/generative/{LIAO_LAB_VIDEOS}",
        help="Destination folder for generation data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination if it already exists.",
    )
    args = parser.parse_args(argv)

    source_zip = resolve_local_path(args.source_zip)
    dest_root = resolve_local_path(args.dest)

    if not source_zip.exists():
        raise FileNotFoundError(f"Liao archive not found: {source_zip}")

    if dest_root.exists():
        if args.force:
            shutil.rmtree(dest_root, ignore_errors=True)
        else:
            raise FileExistsError(
                f"Destination already exists: {dest_root}. Use --force to rebuild."
            )

    dest_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(source_zip), str(dest_root.parent))

    extracted_candidates = [
        path
        for path in dest_root.parent.iterdir()
        if path.is_dir() and path.name.lower() == LIAO_LAB_VIDEOS
    ]
    if not dest_root.exists() and len(extracted_candidates) == 1:
        extracted_candidates[0].rename(dest_root)
    if not dest_root.exists():
        raise FileNotFoundError(f"Archive extraction failed: {dest_root}")

    sources = discover_render_sources(dest_root)
    build_info = {
        "source_zip": str(source_zip),
        "generation_root": str(dest_root),
        "source_count": len(sources),
        "sources": [
            {
                "slug": render_source_slug(source, dest_root),
                "path": str(source),
                "kind": "frames" if source.is_dir() else "video",
                "coords_xlsx": str(find_fish_coords_xlsx(source))
                if source.is_dir() and find_fish_coords_xlsx(source) is not None
                else None,
            }
            for source in sources
        ],
    }
    write_json(dest_root / "build_info.json", build_info)

    print(f"Prepared Liao generation data at {dest_root}")
    print(f"Renderable sources: {len(sources)}")
    return 0


def run_organize(dataset_name: str, argv: list[str]) -> int:
    if dataset_name == AAU_ZEBRAFISH_REID:
        return organize_aau(argv)
    if dataset_name == DEEP_VISION_FISH:
        return organize_deep_vision(argv)
    if dataset_name == KAKADU_FISHAI:
        return organize_kakadu(argv)
    if dataset_name == LIAO_LAB_VIDEOS:
        return organize_liao(argv)
    if dataset_name == MIT_RIVER_HERRING:
        return organize_mit(argv)
    if dataset_name == NOAA_PUGET_SOUND_NEARSHORE_FISH:
        return organize_noaa(argv)
    raise SystemExit(f"Unsupported dataset: {dataset_name}")
