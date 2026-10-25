"""Structured three-phase lesion-output contract, independent of evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

PHASES = ("a", "p", "v")


class SchemaError(ValueError):
    """A model completion or reference violates the structured output schema."""


@dataclass(frozen=True)
class Detection:
    phase: str
    label: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ParsedCompletion:
    detections: tuple[Detection, ...]
    background_phases: tuple[str, ...]


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SchemaError(f"Invalid JSON numeric constant: {value}")


def parse_completion(
    raw: str,
    classes: Iterable[str],
    aliases: Mapping[str, str] | None = None,
) -> ParsedCompletion:
    """Parse JSON; keep all valid lesions and reject any invalid record.

    Classes are caller-supplied exact canonical labels. Optional aliases map
    explicit spellings (e.g., HH) to those labels. Missing phases are allowed.
    A background phase has exactly one ``no-lesion`` [-1,-1,-1,-1] record.
    Lesion coordinates are numeric xyxy lists in the fixed [0,1000] schema.
    """
    allowed = tuple(classes)
    if (not allowed or len(allowed) != len(set(allowed))
            or any(not isinstance(label, str) or not label for label in allowed)):
        raise ValueError("classes must be non-empty, unique strings")
    lookup = {label: label for label in allowed}
    for alias, label in (aliases or {}).items():
        if label not in allowed or not isinstance(alias, str) or not alias:
            raise ValueError("aliases must map non-empty strings to active classes")
        if alias in lookup and lookup[alias] != label:
            raise ValueError("An alias cannot override a canonical label")
        lookup[alias] = label
    if "no-lesion" in lookup:
        raise ValueError("no-lesion is reserved for background records")
    if not isinstance(raw, str):
        raise SchemaError("Completion must be a JSON string")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
    except SchemaError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise SchemaError("Completion is not valid JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("detections"), list):
        raise SchemaError("Completion must be an object with a detections list")
    detections = []
    background = set()
    for item in payload["detections"]:
        if not isinstance(item, dict) or not {"phase", "label", "bbox_2d"} <= item.keys():
            raise SchemaError("Every detection needs phase, label, and bbox_2d")
        phase, label, coordinates = item["phase"], item["label"], item["bbox_2d"]
        if not isinstance(phase, str) or phase not in PHASES:
            raise SchemaError("Phase must be a, p, or v")
        if not isinstance(label, str):
            raise SchemaError("A diagnostic label must be a string")
        if (not isinstance(coordinates, list) or len(coordinates) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in coordinates)):
            raise SchemaError("bbox_2d must contain four numeric xyxy coordinates")
        try:
            box = tuple(float(value) for value in coordinates)
        except OverflowError as error:
            raise SchemaError("bbox_2d coordinates must be finite") from error
        if not all(math.isfinite(value) for value in box):
            raise SchemaError("bbox_2d coordinates must be finite")
        if label == "no-lesion":
            if box != (-1.0, -1.0, -1.0, -1.0) or phase in background:
                raise SchemaError("Background needs one [-1,-1,-1,-1] sentinel per phase")
            background.add(phase)
            continue
        if label not in lookup:
            raise SchemaError(f"Label {label!r} is outside the active vocabulary")
        x1, y1, x2, y2 = box
        if min(box) < 0 or max(box) > 1000 or x2 <= x1 or y2 <= y1:
            raise SchemaError("Lesion boxes need positive extent within [0,1000]")
        detections.append(Detection(phase, lookup[label], box))
    if any(item.phase in background for item in detections):
        raise SchemaError("A phase cannot contain both background and lesions")
    return ParsedCompletion(tuple(detections), tuple(sorted(background)))
