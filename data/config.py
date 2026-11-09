"""JSON configuration loading with unique keys and finite numbers."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"Configuration numbers must be finite: {value}")


def _float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Configuration numbers must be finite: {value}")
    return result


def load_config(path: str | Path) -> dict:
    """Read a JSON object containing explicit stage settings."""
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_object, parse_constant=_constant,
                          parse_float=_float)
    if not isinstance(value, dict):
        raise ValueError("A stage configuration must be a JSON object")
    return value
