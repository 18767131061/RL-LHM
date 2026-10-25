"""Strict parsers for public RL-LHM prediction artifacts."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_LABEL_ALIASES = {
    "HCC": "HCC",
    "ICC": "ICC",
    "FNH": "FNH",
    "HA": "HA_Hemangioma",
    "HH": "HA_Hemangioma",
    "HA_Hemangioma": "HA_Hemangioma",
    "NC": "nodule_CYST",
    "nodule_CYST": "nodule_CYST",
    "BCLM": "BCLM",
    "CRLM": "CRLM",
}


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Detection:
    phase: str
    label: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ParsedCompletion:
    detections: tuple[Detection, ...]
    background_phases: tuple[str, ...]


def canonical_label(
    value: object,
    aliases: Mapping[str, str] = DEFAULT_LABEL_ALIASES,
) -> str:
    original = str(value or "").strip()
    if original in aliases:
        return aliases[original]
    normalized = original.lower().replace("-", "_").replace(" ", "")
    normalized_aliases = {
        key.lower().replace("-", "_").replace(" ", ""): label
        for key, label in aliases.items()
    }
    if normalized in normalized_aliases:
        return normalized_aliases[normalized]
    raise SchemaError(f"Unknown label: {original!r}")


def _normalize_box(value: object) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        lowered = {str(key).lower(): item for key, item in value.items()}
        if all(key in lowered for key in ("x1", "y1", "x2", "y2")):
            value = [lowered[key] for key in ("x1", "y1", "x2", "y2")]
        elif all(key in lowered for key in ("xmin", "ymin", "xmax", "ymax")):
            value = [lowered[key] for key in ("xmin", "ymin", "xmax", "ymax")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SchemaError("bbox_2d must be xyxy list/tuple or named-coordinate object")
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise SchemaError("bbox_2d contains a non-numeric coordinate") from error
    if not all(math.isfinite(item) for item in box):
        raise SchemaError("bbox_2d coordinates must be finite")
    return box


def parse_completion(
    raw: str,
    classes: Iterable[str],
    aliases: Mapping[str, str] = DEFAULT_LABEL_ALIASES,
) -> ParsedCompletion:
    try:
        payload = json.loads(str(raw).strip())
    except (TypeError, json.JSONDecodeError) as error:
        raise SchemaError("Completion is not valid JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("detections"), list):
        raise SchemaError("Completion must be an object with a detections list")
    allowed = set(classes)
    detections: list[Detection] = []
    background_phases: set[str] = set()
    for item in payload["detections"]:
        if not isinstance(item, dict) or "label" not in item or "bbox_2d" not in item:
            raise SchemaError("Every detection needs label and bbox_2d")
        phase = str(item.get("phase", "")).strip().lower()
        if phase not in {"a", "p", "v"}:
            raise SchemaError(f"Unsupported phase: {phase!r}")
        box = _normalize_box(item["bbox_2d"])
        raw_label = str(item["label"] or "").strip().lower().replace("_", "-").replace(" ", "")
        if raw_label in {"no-lesion", "nolesion", "background", "bg"}:
            if box != (-1.0, -1.0, -1.0, -1.0) or phase in background_phases:
                raise SchemaError("Background requires one [-1,-1,-1,-1] sentinel per phase")
            background_phases.add(phase)
            continue
        label = canonical_label(item["label"], aliases)
        x1, y1, x2, y2 = box
        if label not in allowed or min(box) < 0.0 or max(box) > 1000.0 or x2 <= x1 or y2 <= y1:
            raise SchemaError("Detection label or normalized xyxy box is out of contract")
        detections.append(Detection(phase, label, box))
    if any(item.phase in background_phases for item in detections):
        raise SchemaError("A phase cannot contain both background and lesion detections")
    return ParsedCompletion(tuple(detections), tuple(sorted(background_phases)))


def read_csv_rows(path: str | Path, required_fields: Sequence[str]) -> list[dict[str, str]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise SchemaError(f"Duplicate CSV columns in {source}")
        missing = [field for field in required_fields if field not in fields]
        if missing:
            raise SchemaError(f"Missing CSV fields in {source}: {missing}")
        return [dict(row) for row in reader]
