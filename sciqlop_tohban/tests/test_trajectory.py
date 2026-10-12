import numpy as np
import pytest

from sciqlop_tohban.trajectory import fov_on_shell, hourly, lon_lat, mode_tracks, polylines


def test_mode_track_keeps_its_samples_and_the_next_one_so_runs_join():
    xy = np.arange(12.0).reshape(6, 2)
    tracks = mode_tracks(xy, np.array(["OFF", "HKM", "HKM", "OFF", "LM", "OFF"]))
    hkm_x = tracks["HKM"][:, 0]
    assert np.isnan(hkm_x[[0, 4, 5]]).all() and hkm_x[1:4].tolist() == [2.0, 4.0, 6.0]
    assert sorted(tracks) == ["HKM", "LM", "OFF"]


def test_hourly_interpolates_positions_on_full_hours():
    time = np.datetime64("2030-01-01T00:30:00") + np.arange(5) * np.timedelta64(30, "m")
    points = np.column_stack([np.arange(5.0)] * 3)
    assert hourly(time, points)[:, 0].tolist() == [1.0, 3.0]


def test_polylines_join_curves_with_nan_breaks():
    x, y = polylines([np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[4.0, 5.0]])])
    assert x[[0, 1, 3]].tolist() == [0.0, 2.0, 4.0] and np.isnan(x[2]) and np.isnan(y[2])


def test_lon_lat_in_degrees():
    lon, lat = lon_lat(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -2.0]]))
    assert lon[0] == pytest.approx(90.0) and lat.tolist() == pytest.approx([0.0, -90.0])


def test_fov_on_shell_samples_evenly_and_hits_the_shell():
    n = 50
    origin = np.zeros((n, 3))
    corners = np.tile(np.eye(3)[None, :, :], (n, 1, 1))
    picked, hits = fov_on_shell(origin, corners, radius=10.0, count=5)
    assert picked.tolist() == [0, 12, 24, 36, 49]
    assert np.linalg.norm(hits, axis=-1) == pytest.approx(np.full((5, 3), 10.0))
