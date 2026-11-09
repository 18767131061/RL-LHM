"""DICOM-reader tests use synthetic arrays, with an optional real GDCM roundtrip."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from data.dicom import read_dicom_series


def install_reader(monkeypatch, *, values=None, metadata=None, series=("series-A",), files=None):
    pixels = np.array([[[-100, 0, 50, 100, 200]]], dtype=np.int16) if values is None else np.asarray(values)
    tags = metadata or {}
    calls = []
    filenames = ("slice-02.dcm", "slice-01.dcm") if files is None else files
    image = SimpleNamespace(GetDimension=lambda: 3, GetSpacing=lambda: (.7, .8, 2.5),
                            GetOrigin=lambda: (-10., 20., 30.),
                            GetDirection=lambda: (0., -1., 0., 1., 0., 0., 0., 0., 1.))

    class Reader:
        @staticmethod
        def GetGDCMSeriesIDs(path):
            calls.append(("ids", path))
            return series

        @staticmethod
        def GetGDCMSeriesFileNames(path, uid):
            calls.append(("files", path, uid))
            return filenames

        def SetFileNames(self, names):
            calls.append(("set_files", tuple(names)))

        def MetaDataDictionaryArrayUpdateOn(self):
            calls.append(("metadata_on",))

        def Execute(self):
            assert ("metadata_on",) in calls
            calls.append(("execute",))
            return image

        def HasMetaDataKey(self, index, key):
            assert index == 0
            return key in tags

        def GetMetaData(self, index, key):
            assert index == 0
            calls.append(("metadata", index, key))
            return tags[key]

    monkeypatch.setitem(sys.modules, "SimpleITK",
                        SimpleNamespace(ImageSeriesReader=Reader, GetArrayFromImage=lambda _: pixels))
    return calls, pixels, image


def test_explicit_window_overrides_metadata_and_preserves_physical_geometry(tmp_path, monkeypatch):
    calls, original, image = install_reader(monkeypatch, metadata={"0028|1050": "999", "0028|1051": "1"})
    result = read_dicom_series(tmp_path, window_center=50, window_width=100, normalize=True)
    assert result.values.tolist() == [[[0, 0, 127, 255, 255]]]
    assert result.values.dtype == np.uint8
    assert result.spacing == image.GetSpacing() and result.origin == image.GetOrigin()
    assert result.direction == image.GetDirection()
    assert result.window_bounds == (0, 100) and result.window_source == "explicit"
    assert result.filenames == ("slice-02.dcm", "slice-01.dcm")
    assert ("set_files", result.filenames) in calls
    assert not any(call[0] == "metadata" for call in calls)
    assert original.tolist() == [[[-100, 0, 50, 100, 200]]]


def test_first_dicom_first_multivalued_window_and_native_intensity_units(tmp_path, monkeypatch):
    install_reader(monkeypatch, metadata={"0028|1050": "50\\500", "0028|1051": "100\\800"})
    result = read_dicom_series(tmp_path)
    assert result.window_source == "dicom" and not result.normalized
    assert result.values.dtype == np.float64
    assert result.values.tolist() == [[[0, 0, 50, 100, 100]]]


@pytest.mark.parametrize("metadata", [{}, {"0028|1050": "50"},
                                      {"0028|1050": "nan", "0028|1051": "100"},
                                      {"0028|1050": "50", "0028|1051": "0"}])
def test_missing_or_unusable_window_falls_back_to_intensity_range(tmp_path, monkeypatch, metadata):
    install_reader(monkeypatch, metadata=metadata)
    result = read_dicom_series(tmp_path, normalize=True)
    assert result.window_source == "intensity_range" and result.window_bounds == (-100, 200)
    assert result.values.tolist() == [[[0, 85, 127, 170, 255]]]


def test_constant_volume_maps_to_zero_and_keeps_native_values_without_normalization(tmp_path, monkeypatch):
    install_reader(monkeypatch, values=np.full((2, 2, 2), 42))
    assert (read_dicom_series(tmp_path, normalize=True).values == 0).all()
    assert (read_dicom_series(tmp_path).values == 42).all()


def test_ambiguous_series_requires_selection_and_preserves_gdcm_file_order(tmp_path, monkeypatch):
    calls, _, _ = install_reader(monkeypatch, series=("series-A", "series-B"))
    with pytest.raises(ValueError, match="multiple"):
        read_dicom_series(tmp_path)
    result = read_dicom_series(tmp_path, series_uid="series-B")
    assert result.series_uid == "series-B"
    assert ("files", str(tmp_path), "series-B") in calls
    with pytest.raises(ValueError, match="does not identify"):
        read_dicom_series(tmp_path, series_uid="missing")


@pytest.mark.parametrize("kwargs", [{"window_center": 40}, {"window_width": 400},
                                    {"window_center": 40, "window_width": -1},
                                    {"window_center": float("nan"), "window_width": 400},
                                    {"normalize": 1}])
def test_invalid_explicit_settings_are_not_silently_replaced(tmp_path, monkeypatch, kwargs):
    install_reader(monkeypatch)
    with pytest.raises(ValueError):
        read_dicom_series(tmp_path, **kwargs)


def test_empty_series_and_nonfinite_intensity_are_rejected(tmp_path, monkeypatch):
    install_reader(monkeypatch, series=())
    with pytest.raises(ValueError, match="no DICOM series"):
        read_dicom_series(tmp_path)
    install_reader(monkeypatch, files=())
    with pytest.raises(ValueError, match="no readable"):
        read_dicom_series(tmp_path)
    install_reader(monkeypatch, values=[[[float("nan")]]])
    with pytest.raises(ValueError, match="finite"):
        read_dicom_series(tmp_path)


def test_real_synthetic_dicom_series_roundtrip(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    import uuid

    values = np.array([
        [[-100, 0, 50], [100, 150, 200]],
        [[-50, 25, 75], [125, 175, 225]],
        [[0, 10, 20], [30, 40, 50]],
    ], dtype=np.int16)
    volume = sitk.GetImageFromArray(values)
    volume.SetSpacing((.7, .8, 2.5))
    volume.SetOrigin((-10., 20., 30.))
    volume.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    study_uid, series_uid = ("2.25." + str(uuid.uuid4().int) for _ in range(2))
    direction = volume.GetDirection()
    orientation = "\\".join(str(direction[i]) for i in (0, 3, 6, 1, 4, 7))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for index in range(volume.GetDepth()):
        image = volume[:, :, index]
        tags = {
            "0008|0016": "1.2.840.10008.5.1.4.1.1.2",  # CT Image Storage
            "0008|0018": "2.25." + str(uuid.uuid4().int),
            "0008|0060": "CT", "0010|0010": "Synthetic^Fixture",
            "0010|0020": "SYNTHETIC_TEST", "0020|000d": study_uid,
            "0020|000e": series_uid, "0020|0011": "1", "0020|0013": str(index + 1),
            "0020|0032": "\\".join(str(v) for v in volume.TransformIndexToPhysicalPoint((0, 0, index))),
            "0020|0037": orientation, "0018|0050": "2.5",
            "0028|1050": "50\\500", "0028|1051": "100\\800",
        }
        for key, value in tags.items():
            image.SetMetaData(key, value)
        # Reverse the filename order to exercise GDCM's geometric slice ordering.
        writer.SetFileName(str(tmp_path / f"slice-{volume.GetDepth() - index}.dcm"))
        writer.Execute(image)

    result = read_dicom_series(tmp_path)
    assert result.series_uid == series_uid and result.window_source == "dicom"
    assert result.window_bounds == (0, 100)
    np.testing.assert_array_equal(result.values, np.clip(values, 0, 100))
    assert result.spacing == pytest.approx(volume.GetSpacing())
    assert result.origin == pytest.approx(volume.GetOrigin())
    assert result.direction == pytest.approx(volume.GetDirection())
    assert [name.rsplit("/", 1)[-1] for name in result.filenames] == ["slice-3.dcm", "slice-2.dcm", "slice-1.dcm"]

    explicit = read_dicom_series(tmp_path, series_uid=series_uid,
                                 window_center=0, window_width=200, normalize=True)
    expected = ((np.clip(values.astype(float), -100, 100) + 100) / 200 * 255).astype(np.uint8)
    np.testing.assert_array_equal(explicit.values, expected)
    assert explicit.window_source == "explicit" and explicit.window_bounds == (-100, 100)
    assert explicit.values.dtype == np.uint8
    assert explicit.spacing == result.spacing and explicit.origin == result.origin
    assert explicit.direction == result.direction
