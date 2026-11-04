import numpy as np
import pytest
from spiceypy.utils.exceptions import SpiceCKINSUFFDATA

from sciqlop_tohban import ephemeris
from sciqlop_tohban.ephemeris import Ephemeris, MissingEphemeris, cached_sample, coverage, sample, shell_footprint
from sciqlop_tohban.kernels import CACHE


def synthetic(n=10):
    time = np.datetime64("2030-01-01T00:00:00") + np.arange(n) * np.timedelta64(5, "m")
    ones = np.ones((n, 3))
    return Ephemeris(time, ones * 2, ones, ones * 5, ones, np.ones((n, 4, 3)))


def test_window_keeps_the_samples_inside_and_one_on_each_side():
    eph = synthetic()
    window = eph.window(eph.time[3] + np.timedelta64(1, "m"), eph.time[5] - np.timedelta64(1, "m"))
    assert window.time.tolist() == eph.time[3:6].tolist()
    assert window.mio.shape == (3, 3) and window.msasi_corners.shape == (3, 4, 3)


def test_separation_and_radii():
    eph = synthetic(2)
    assert eph.separation.tolist() == pytest.approx([np.sqrt(27)] * 2)
    assert eph.mio_radius.tolist() == pytest.approx([np.sqrt(12)] * 2)


JUNE = np.datetime64("2029-06-01"), np.datetime64("2029-07-01")


def test_missing_coverage_names_what_is_missing_and_the_window():
    with pytest.raises(MissingEphemeris, match="no MSASI pointing between 2029-06-01 and 2029-07-01 in ESA's current plan"):
        with coverage("MSASI pointing", *JUNE):
            raise SpiceCKINSUFFDATA(long="Insufficient pointing data")


def test_bugs_are_not_reported_as_missing_coverage():
    with pytest.raises(KeyError):
        with coverage("MIO ephemeris", *JUNE):
            raise KeyError("not a SPICE error")


def test_shell_footprint_is_where_a_ray_leaves_the_sphere():
    origin = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    direction = np.array([[0.0, 0.0, 2.0], [-1.0, 0.0, 0.0]])
    assert shell_footprint(origin, direction, 10.0) == pytest.approx(np.array([[0, 0, 10.0], [-10.0, 0, 0]]))


def test_shell_footprint_is_nan_outside_the_sphere_looking_away():
    hit = shell_footprint(np.array([[20.0, 0, 0]]), np.array([[1.0, 0, 0]]), 10.0)
    assert np.isnan(hit).all()


@pytest.mark.skipif(not (CACHE / "tohban.tm").exists(), reason="needs the downloaded ESA kernels")
def test_sample_mio_orbit_from_the_esa_planning_kernels():
    eph = sample(CACHE / "tohban.tm", np.datetime64("2027-05-08T00:00:00"), np.datetime64("2027-05-11T00:00:00"))
    assert len(eph.time) == 3 * 24 * 12 + 1
    # MIO orbit: 590 x 11640 km altitude
    assert eph.mio_radius.min() == pytest.approx(2440 + 590, rel=0.05)
    assert eph.mio_radius.max() == pytest.approx(2440 + 11640, rel=0.05)
    assert np.linalg.norm(eph.msasi_boresight, axis=1) == pytest.approx(1.0)


@pytest.fixture
def counted_sampling(monkeypatch):
    calls = []

    def fake_sample(metakernel, start, stop):
        calls.append((metakernel.read_text(), start, stop))
        return synthetic()

    monkeypatch.setattr(ephemeris, "sample", fake_sample)
    return calls


def test_cached_sample_samples_spice_once_per_plan_and_kernel_set(tmp_path, counted_sampling):
    mk = tmp_path / "tohban.tm"
    mk.write_text("KERNELS_TO_LOAD = ( 'spk/bc_mmo_v01.bsp' )")
    start, stop = np.datetime64("2030-01-01T00:00:00"), np.datetime64("2030-01-02T00:00:00")
    first = cached_sample(mk, start, stop)
    again = cached_sample(mk, start, stop)
    assert len(counted_sampling) == 1
    np.testing.assert_array_equal(again.mio, first.mio)
    np.testing.assert_array_equal(again.time, first.time)


def test_new_kernels_or_another_window_sample_again(tmp_path, counted_sampling):
    mk = tmp_path / "tohban.tm"
    mk.write_text("KERNELS_TO_LOAD = ( 'spk/bc_mmo_v01.bsp' )")
    start, stop = np.datetime64("2030-02-01T00:00:00"), np.datetime64("2030-02-02T00:00:00")
    cached_sample(mk, start, stop)
    cached_sample(mk, start, stop + np.timedelta64(1, "D"))
    mk.write_text("KERNELS_TO_LOAD = ( 'spk/bc_mmo_v02.bsp' )")
    cached_sample(mk, start, stop)
    assert len(counted_sampling) == 3
