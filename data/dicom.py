"""Read a scalar DICOM series and apply the intensity step in Appendix D.1.

This module does not register phases, select slices, construct lesion boxes,
or create the experiment manifests. SimpleITK is an optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DicomVolume:
    """Processed voxel values plus the unmodified physical image geometry.

    ``values`` is indexed in NumPy z/y/x order. ``spacing`` and ``origin`` use
    SimpleITK x/y/z order; ``direction`` is the flattened row-major 3x3 matrix.
    ``window_bounds`` contains the lower/upper clipping intensities.
    """

    values: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    series_uid: str
    filenames: tuple[str, ...]
    window_bounds: tuple[float, float]
    window_source: str
    normalized: bool


def _window_bounds(center: float, width: float) -> tuple[float, float]:
    if isinstance(center, bool) or isinstance(width, bool):
        raise ValueError("window center and width must be finite numbers")
    center, width = float(center), float(width)
    if not math.isfinite(center) or not math.isfinite(width) or width <= 0:
        raise ValueError("window center must be finite and width must be positive")
    lower, upper = center - width / 2, center + width / 2
    if (not math.isfinite(lower) or not math.isfinite(upper)
            or upper <= lower or not math.isfinite(upper - lower)):
        raise ValueError("window bounds must have finite positive width")
    return lower, upper


def _first_metadata_number(reader, tag: str) -> float | None:
    if not reader.HasMetaDataKey(0, tag):
        return None
    try:
        # Multi-valued DICOM decimal strings are separated by backslashes.
        value = float(reader.GetMetaData(0, tag).split("\\", 1)[0].strip())
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def read_dicom_series(
    directory: str | Path,
    *,
    series_uid: str | None = None,
    window_center: float | None = None,
    window_width: float | None = None,
    normalize: bool = False,
) -> DicomVolume:
    """Read one GDCM-discovered series without reorienting or resampling it.

    A directory containing multiple series requires an explicit ``series_uid``.
    Both explicit window parameters must be supplied together. They override
    the first GDCM-ordered DICOM's WindowCenter/WindowWidth (0028|1050/1051);
    the first component is used for multivalued metadata. Missing or unusable
    metadata falls back to the intensity range returned by SimpleITK.

    Window bounds are ``center +/- width/2``. No additional rescale slope or
    intercept is applied after reading. With ``normalize=False``, return clipped
    intensities as float64. With normalization, map the window linearly to
    [0,255], then truncate to uint8. A constant-range volume maps to zero.
    Native spacing, origin and direction are returned unchanged. Metadata APIs:
    https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageSeriesReader.html
    """
    if type(normalize) is not bool:
        raise ValueError("normalize must be a boolean")
    if (window_center is None) != (window_width is None):
        raise ValueError("supply both window_center and window_width")
    explicit = None if window_center is None else _window_bounds(window_center, window_width)
    path = Path(directory)
    if not path.is_dir():
        raise ValueError("DICOM input must be an existing directory")
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError("DICOM reading requires requirements-preprocessing.txt") from error

    series = tuple(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or ())
    if not series:
        raise ValueError("no DICOM series found in the directory")
    if series_uid is None:
        if len(series) != 1:
            raise ValueError("multiple DICOM series found; supply series_uid explicitly")
        series_uid = series[0]
    if not isinstance(series_uid, str) or series_uid not in series:
        raise ValueError("series_uid does not identify a discovered DICOM series")
    filenames = tuple(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_uid))
    if not filenames:
        raise ValueError("selected DICOM series contains no readable files")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(filenames)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    values = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float64)
    if image.GetDimension() != 3 or values.ndim != 3 or values.size == 0:
        raise ValueError("DICOM series must produce a nonempty scalar 3D volume")
    if not np.isfinite(values).all():
        raise ValueError("DICOM intensities must be finite")

    bounds, source = explicit, "explicit"
    if bounds is None:
        center = _first_metadata_number(reader, "0028|1050")
        width = _first_metadata_number(reader, "0028|1051")
        if center is not None and width is not None:
            try:
                bounds = _window_bounds(center, width)
            except ValueError:
                pass
        source = "dicom"
    if bounds is None:
        bounds, source = (float(values.min()), float(values.max())), "intensity_range"
        if not math.isfinite(bounds[1] - bounds[0]):
            raise ValueError("DICOM intensity range must be finite")
    lower, upper = bounds
    processed = np.clip(values, lower, upper)
    if normalize:
        processed = (np.zeros_like(processed) if upper == lower
                     else (processed - lower) / (upper - lower) * 255)
        processed = np.clip(processed, 0, 255).astype(np.uint8)

    return DicomVolume(
        values=processed, spacing=tuple(image.GetSpacing()), origin=tuple(image.GetOrigin()),
        direction=tuple(image.GetDirection()), series_uid=series_uid, filenames=filenames,
        window_bounds=bounds, window_source=source, normalized=normalize,
    )
