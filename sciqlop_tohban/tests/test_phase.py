import numpy as np
import pytest

from sciqlop_tohban.evt import Interval
from sciqlop_tohban.phase import apoapses, clip, fold, folded_bars

T0 = np.datetime64("2030-01-01T00:00:00")


def hours(*values):
    return T0 + (np.array(values) * 3600).astype("timedelta64[s]")


def test_apoapses_are_the_radius_peaks_padded_by_one_orbit_each_side():
    time = T0 + np.arange(0, 30 * 60, 5) * np.timedelta64(1, "m")
    elapsed = (time - T0).astype(float) / 3600
    radius = 2.0 - np.cos(2 * np.pi * (elapsed - 2.0) / 10.0)  # peaks at 7, 17 and 27 h
    assert apoapses(time, radius).tolist() == hours(-3, 7, 17, 27, 37).tolist()


def test_apoapses_need_two_peaks_to_know_the_period():
    time = T0 + np.arange(5) * np.timedelta64(1, "h")
    assert len(apoapses(time, np.array([0.0, 1.0, 2.0, 1.0, 0.0]))) == 0


def test_fold_splits_an_interval_at_each_apoapsis():
    assert fold(*hours(8, 22), hours(0, 10, 20, 30)) == pytest.approx([(8, 10), (0, 10), (0, 2)])


def test_fold_drops_what_is_outside_the_known_orbits():
    assert fold(*hours(-2, 3), hours(0, 10)) == pytest.approx([(0, 3)])
    assert fold(*hours(12, 14), hours(0, 10)) == []


def test_folded_bars_put_each_lane_on_its_row_coloured_by_mode():
    intervals = [
        Interval("MGF_I", "HKM", *hours(1, 2)),
        Interval("DPU", "LM", *hours(3, 4)),
        Interval("DPU", "HKM", *hours(11, 12)),
    ]
    rows, bars = folded_bars(intervals, hours(0, 10, 20))
    assert rows == ["DPU", "MGF_I"]
    x, y = bars["HKM"]
    assert x[[0, 1, 3, 4]].tolist() == pytest.approx([1, 2, 1, 2]) and np.isnan(x[2])
    assert sorted({y[0], y[3]}) == [-1.0, 0.0]
    assert bars["LM"][1][0] == 0.0


def test_clip_keeps_the_overlapping_part_of_each_interval():
    intervals = [Interval("DPU", "LM", *hours(0, 4)), Interval("DPU", "HKM", *hours(5, 6))]
    assert clip(intervals, *hours(2, 5)) == [Interval("DPU", "LM", *hours(2, 4))]
