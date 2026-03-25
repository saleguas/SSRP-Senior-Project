from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from src.pipeline.utils import is_image_file, write_json

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv")
_XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass(frozen=True)
class CoordPoint:
    frame_index: int
    fish_id: int
    x: float
    y: float


def frame_index_from_name(name: str) -> Optional[int]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(match.group(1)) if match else None


def find_fish_coords_xlsx(folder: Path) -> Optional[Path]:
    folder = folder.resolve()
    candidates = sorted(
        [
            path
            for path in folder.glob("*.xlsx")
            if path.is_file()
            and "fish coords" in path.name.lower()
            and not path.name.startswith("~$")
        ]
    )
    return candidates[0] if candidates else None


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def _has_direct_images(folder: Path) -> bool:
    return any(is_image_file(path) for path in folder.iterdir())


def discover_render_sources(root: Path) -> List[Path]:
    root = root.resolve()
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Render root not found: {root}")

    sources: List[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            sources.append(resolved)

    if _has_direct_images(root):
        add(root)

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            try:
                if _has_direct_images(path):
                    add(path)
            except OSError:
                continue
        elif _is_video_file(path):
            add(path)

    return sources


def _sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower())
    return safe.strip("_.-") or "source"


def render_source_slug(source: Path, batch_root: Path) -> str:
    source = source.resolve()
    batch_root = batch_root.resolve()
    if batch_root.is_file():
        return _sanitize_name(batch_root.stem)
    if source == batch_root:
        return _sanitize_name(batch_root.name)
    relative = source.relative_to(batch_root)
    parts = [
        _sanitize_name(part if index < len(relative.parts) - 1 else Path(part).stem)
        for index, part in enumerate(relative.parts)
    ]
    return "_".join(part for part in parts if part) or _sanitize_name(source.stem)


def _col_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - 64)
    return max(0, value - 1)


def _shared_strings(xlsx: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []
    root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for item in root.findall("main:si", _XLSX_NS):
        values.append("".join(node.text or "" for node in item.iterfind(".//main:t", _XLSX_NS)))
    return values


def _sheet_targets(xlsx: zipfile.ZipFile) -> List[Tuple[str, str]]:
    workbook_root = ET.fromstring(xlsx.read("xl/workbook.xml"))
    rels_root = ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels_root.findall("rel:Relationship", _XLSX_NS)
    }
    targets: List[Tuple[str, str]] = []
    for sheet in workbook_root.findall("main:sheets/main:sheet", _XLSX_NS):
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", ""
        )
        target = rel_map.get(rel_id)
        if target:
            targets.append((name, "xl/" + target.lstrip("/")))
    return targets


def _sheet_rows(
    xlsx: zipfile.ZipFile,
    target: str,
    shared_strings: Sequence[str],
) -> List[Tuple[str, ...]]:
    sheet_root = ET.fromstring(xlsx.read(target))
    rows: List[Tuple[str, ...]] = []
    for row in sheet_root.findall(".//main:sheetData/main:row", _XLSX_NS):
        cells: Dict[int, str] = {}
        for cell in row.findall("main:c", _XLSX_NS):
            ref = cell.attrib.get("r", "")
            value_node = cell.find("main:v", _XLSX_NS)
            value = value_node.text.strip() if value_node is not None and value_node.text else ""
            if cell.attrib.get("t") == "s" and value:
                value = shared_strings[int(value)]
            cells[_col_to_index(ref)] = value
        if cells:
            max_index = max(cells)
            rows.append(tuple(cells.get(index, "") for index in range(max_index + 1)))
    return rows


def load_fish_coords_xlsx(xlsx_path: Path) -> Dict[int, List[CoordPoint]]:
    xlsx_path = xlsx_path.resolve()
    coords: Dict[int, List[CoordPoint]] = {}
    with zipfile.ZipFile(xlsx_path) as xlsx:
        shared_strings = _shared_strings(xlsx)
        candidate_rows: List[Tuple[str, ...]] = []
        for _, target in _sheet_targets(xlsx):
            rows = _sheet_rows(xlsx, target, shared_strings)
            if not rows:
                continue
            header = [value.strip().lower() for value in rows[0][:4]]
            if header == ["frame", "fishid", "x", "y"]:
                candidate_rows = rows
                break
        if not candidate_rows:
            raise ValueError(f"No usable Fish coords sheet found in {xlsx_path}")

    for row in candidate_rows[1:]:
        if len(row) < 4:
            continue
        frame_raw, fish_id_raw, x_raw, y_raw = row[:4]
        if not frame_raw or not fish_id_raw or not x_raw or not y_raw:
            continue
        frame_index = int(float(frame_raw))
        point = CoordPoint(
            frame_index=frame_index,
            fish_id=int(float(fish_id_raw)),
            x=float(x_raw),
            y=float(y_raw),
        )
        coords.setdefault(frame_index, []).append(point)

    for points in coords.values():
        points.sort(key=lambda point: point.fish_id)
    return coords


def _load_track_centers(csv_path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    centers: Dict[int, List[Tuple[int, float, float]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_name = str(row.get("frame") or "").strip()
            frame_index = frame_index_from_name(frame_name)
            if frame_index is None:
                continue
            try:
                track_id = int(float(str(row.get("track_id") or 0)))
                x_center = float(str(row.get("xc") or 0.0))
                y_center = float(str(row.get("yc") or 0.0))
            except ValueError:
                continue
            centers.setdefault(frame_index, []).append((track_id, x_center, y_center))
    return centers


def _greedy_match(
    points: Sequence[CoordPoint],
    predictions: Sequence[Tuple[int, float, float]],
) -> List[Dict[str, Any]]:
    pairs: List[Tuple[float, int, int]] = []
    for point_index, point in enumerate(points):
        for prediction_index, (track_id, x_center, y_center) in enumerate(predictions):
            distance = ((point.x - x_center) ** 2 + (point.y - y_center) ** 2) ** 0.5
            pairs.append((distance, point_index, prediction_index))
    pairs.sort(key=lambda item: item[0])

    used_points: set[int] = set()
    used_predictions: set[int] = set()
    matches: List[Dict[str, Any]] = []
    for distance, point_index, prediction_index in pairs:
        if point_index in used_points or prediction_index in used_predictions:
            continue
        point = points[point_index]
        track_id, x_center, y_center = predictions[prediction_index]
        used_points.add(point_index)
        used_predictions.add(prediction_index)
        matches.append(
            {
                "fish_id": point.fish_id,
                "track_id": track_id,
                "point_x": point.x,
                "point_y": point.y,
                "pred_x": x_center,
                "pred_y": y_center,
                "distance_px": float(f"{distance:.3f}"),
            }
        )
    return matches


def compare_tracks_to_points(
    tracks_csv: Path,
    coords_by_frame: Dict[int, List[CoordPoint]],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    tracks_csv = tracks_csv.resolve()
    centers_by_frame = _load_track_centers(tracks_csv)

    all_distances: List[float] = []
    matched_points = 0
    missing_points = 0
    extra_predictions = 0
    per_frame: List[Dict[str, Any]] = []

    for frame_index in sorted(coords_by_frame):
        points = coords_by_frame[frame_index]
        predictions = centers_by_frame.get(frame_index, [])
        matches = _greedy_match(points, predictions)
        distances = [float(match["distance_px"]) for match in matches]
        all_distances.extend(distances)
        matched_points += len(matches)
        missing_points += max(0, len(points) - len(matches))
        extra_predictions += max(0, len(predictions) - len(matches))
        per_frame.append(
            {
                "frame_index": frame_index,
                "points": len(points),
                "predictions": len(predictions),
                "matched": len(matches),
                "missing": max(0, len(points) - len(matches)),
                "extra_predictions": max(0, len(predictions) - len(matches)),
                "mean_distance_px": float(f"{(sum(distances) / len(distances)):.3f}") if distances else None,
                "matches": matches,
            }
        )

    sorted_distances = sorted(all_distances)
    median_distance = None
    if sorted_distances:
        middle = len(sorted_distances) // 2
        if len(sorted_distances) % 2 == 1:
            median_distance = sorted_distances[middle]
        else:
            median_distance = (sorted_distances[middle - 1] + sorted_distances[middle]) / 2.0

    summary = {
        "tracks_csv": str(tracks_csv),
        "frames_with_coords": len(coords_by_frame),
        "frames_with_predictions": sum(1 for frame_index in coords_by_frame if centers_by_frame.get(frame_index)),
        "total_points": sum(len(points) for points in coords_by_frame.values()),
        "matched_points": matched_points,
        "missing_points": missing_points,
        "extra_predictions": extra_predictions,
        "mean_distance_px": float(f"{(sum(all_distances) / len(all_distances)):.3f}") if all_distances else None,
        "median_distance_px": float(f"{median_distance:.3f}") if median_distance is not None else None,
        "max_distance_px": float(f"{max(all_distances):.3f}") if all_distances else None,
        "within_25px": sum(distance <= 25.0 for distance in all_distances),
        "within_50px": sum(distance <= 50.0 for distance in all_distances),
        "within_100px": sum(distance <= 100.0 for distance in all_distances),
        "per_frame": per_frame,
    }

    if output_path is not None:
        write_json(output_path.resolve(), summary)

    return summary
