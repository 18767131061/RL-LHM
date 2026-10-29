"""Read prepared image groups with a fixed patient population."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rewards.schema import PHASES, parse_completion


@dataclass(frozen=True)
class Manifest:
    output_classes: tuple[str, ...]
    patients: dict[str, str]
    records: list[dict]
    metric_classes: tuple[str, ...] | None = None
    patient_centers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        from evaluation.classification import class_spaces
        output, metric = class_spaces(self.output_classes, self.metric_classes)
        object.__setattr__(self, "output_classes", output)
        object.__setattr__(self, "metric_classes", metric)
        if not isinstance(self.patient_centers, dict):
            raise ValueError("patient_centers must be a dictionary")
        if self.patient_centers and (set(self.patient_centers) != set(self.patients)
                or any(not isinstance(center, str) or not center.strip() for center in self.patient_centers.values())):
            raise ValueError("patient_centers must give a nonempty center for every patient")

    @property
    def classes(self):
        """Compatibility alias for the output vocabulary."""
        return self.output_classes

    def payload(self):
        return dict(output_classes=list(self.output_classes), metric_classes=list(self.metric_classes),
                    patients=self.patients, patient_centers=self.patient_centers,
                    records=[{k: v for k, v in row.items() if k != "patient_label"} for row in self.records])


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def load_manifest(path: str | Path) -> Manifest:
    """Load JSON; resolve image paths relative to its directory.

    ``patients`` defines the entire evaluation population, including patients
    with no input groups. Records contain ``uid``, ``patient_id``, ``images``
    keyed by a/p/v, ``prompt``, ``ground_truth`` as a JSON string, and an optional
    fixed ``visual_vector``. Loaded records also contain ``patient_label``.
    """
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream, object_pairs_hook=_unique_object,
                            parse_constant=_constant)
    if (not isinstance(payload, dict) or not {"patients", "records"} <= payload.keys()
            or set(payload) - {"classes", "output_classes", "metric_classes", "patients", "records", "patient_centers"}
            or ("classes" in payload) == ("output_classes" in payload)):
        raise ValueError("Manifest needs output_classes, patients, records; classes is a legacy alias")
    classes = payload.get("output_classes", payload.get("classes"))
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(c, str) or not c.strip() for c in classes)
            or len(set(classes)) != len(classes) or "no-lesion" in classes):
        raise ValueError("classes must be unique non-empty diagnostic labels")
    patients = payload["patients"]
    if not isinstance(patients, dict):
        raise ValueError("patients must map patient IDs to diagnostic labels")
    for patient, label in patients.items():
        _identifier(patient, "patient_id")
        if not isinstance(label, str) or label not in classes:
            raise ValueError("Every patient label must belong to classes")
    if not isinstance(payload["records"], list):
        raise ValueError("records must be a list")
    required = {"uid", "patient_id", "images", "prompt", "ground_truth"}
    records, seen = [], set()
    vector_dimension = None
    for raw in payload["records"]:
        if (not isinstance(raw, dict) or not required <= raw.keys()
                or set(raw) - required - {"visual_vector"}):
            raise ValueError("Each record needs uid, patient_id, images, prompt, and ground_truth")
        uid = _identifier(raw["uid"], "uid")
        patient = _identifier(raw["patient_id"], "patient_id")
        if uid in seen:
            raise ValueError(f"Duplicate input UID: {uid}")
        if patient not in patients:
            raise ValueError(f"Unknown patient ID: {patient}")
        seen.add(uid)
        _identifier(raw["prompt"], "prompt")
        parse_completion(raw["ground_truth"], classes)
        images = raw["images"]
        if not isinstance(images, dict) or set(images) != set(PHASES):
            raise ValueError("images must contain exactly a, p, and v")
        resolved_images = {}
        for phase in PHASES:
            image_path = Path(_identifier(images[phase], f"images.{phase}"))
            resolved_images[phase] = str((path.parent / image_path).resolve())
        record = dict(raw, images=resolved_images, patient_label=patients[patient])
        vector = raw.get("visual_vector")
        if vector is not None:
            if (not isinstance(vector, list) or not vector
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in vector)):
                raise ValueError("visual_vector must contain finite numbers")
            if not any(x != 0 for x in vector):
                raise ValueError("visual_vector must have nonzero norm")
            if vector_dimension is not None and len(vector) != vector_dimension:
                raise ValueError("All visual vectors must share one dimension")
            vector_dimension = len(vector)
        records.append(record)
    return Manifest(tuple(classes), dict(patients), records,
                    tuple(payload["metric_classes"]) if "metric_classes" in payload else None,
                    payload.get("patient_centers", {}))


def training_class_counts(manifest: Manifest) -> dict[str, int]:
    """Count positive training groups using the inference class-order tie rule.

    Each group contributes once. A background-only group contributes no count.
    Only observed dominant labels are keys in the returned dictionary.
    """
    counts: Counter[str] = Counter()
    for record in manifest.records:
        parsed = parse_completion(record["ground_truth"], manifest.output_classes)
        labels = Counter(detection.label for detection in parsed.detections)
        if labels:
            label = max(manifest.output_classes, key=lambda item: labels[item])
            counts[label] += 1
    return dict(counts)
