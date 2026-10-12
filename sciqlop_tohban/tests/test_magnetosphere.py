import numpy as np
import pytest

from sciqlop_tohban.magnetosphere import (
    BS,
    boundary_radius,
    DIPOLE_Z,
    MP,
    RM_KM,
    classify,
    meridian_curve,
    orbit_plane_basis,
    plane_section,
    region_runs,
    sunward_nose,
)


def at(x_rm, y_rm=0.0, dz_rm=0.0):
    return np.array([[x_rm * RM_KM, y_rm * RM_KM, DIPOLE_Z + dz_rm * RM_KM]])


def test_noses_match_winslow_2013():
    assert sunward_nose(MP) == pytest.approx(1.45 * RM_KM, rel=1e-3)
    assert sunward_nose(BS) == pytest.approx((0.5 + 2.75 * 1.04 / 2.04) * RM_KM, rel=1e-3)


@pytest.mark.parametrize(
    "point, region",
    [
        (at(1.2), "MSP"),
        (at(1.7), "MSH"),
        (at(2.5), "SW"),
        (at(-5.0), "MSP"),
        (at(-5.0, dz_rm=8.0), "SW"),
        (at(-12.0), "unknown"),
    ],
)
def test_classify_compares_the_distance_to_the_x_axis_with_each_boundary(point, region):
    assert classify(point).tolist() == [region]


def test_classification_is_axisymmetric_around_the_dipole_axis():
    assert classify(at(-5.0, y_rm=1.0)).tolist() == classify(at(-5.0, dz_rm=-1.0)).tolist() == ["MSP"]


def test_meridian_curve_is_open_on_the_nightside_and_closed_at_the_nose():
    curve = meridian_curve(MP)
    assert curve[:, 0].max() == pytest.approx(sunward_nose(MP), rel=1e-3)
    assert curve[0, 0] == pytest.approx(curve[-1, 0]) and curve[0, 0] < -9 * RM_KM
    assert curve[0, 1] < DIPOLE_Z < curve[-1, 1]


def test_orbit_plane_basis_puts_apoapsis_on_negative_first_axis():
    angle = np.linspace(0, 2 * np.pi, 200)
    radius = 5000 + 3000 * np.cos(angle)  # apoapsis at angle 0, along +Z
    position = np.column_stack([np.zeros_like(angle), radius * np.sin(angle), radius * np.cos(angle)])
    velocity = np.gradient(position, axis=0)
    basis = orbit_plane_basis(position, velocity)
    assert basis[0] == pytest.approx([0, 0, -1], abs=1e-6)
    assert abs(basis[1] @ [1, 0, 0]) == pytest.approx(0, abs=1e-6)


def test_section_in_the_xz_plane_reaches_the_nose_and_closes():
    contour = plane_section(MP, np.array([[1.0, 0, 0], [0, 0, 1.0]]))
    assert np.nanmax(contour[:, 0]) == pytest.approx(sunward_nose(MP), rel=0.01)
    assert contour[0] == pytest.approx(contour[-1])


def test_section_is_broken_where_rays_leave_through_the_tail_cut():
    contour = plane_section(BS, np.array([[1.0, 0, 0], [0, 0, 1.0]]))
    tailward = contour[:-1][np.cos(np.linspace(0, 2 * np.pi, len(contour) - 1, endpoint=False)) < -0.99]
    assert np.isnan(tailward).all()


def test_region_runs_merge_consecutive_samples():
    time = np.arange(5).astype("datetime64[m]")
    runs = region_runs(time, np.array(["SW", "SW", "MSH", "MSH", "SW"]))
    assert runs == [(time[0], time[2], "SW"), (time[2], time[4], "MSH"), (time[4], time[4], "SW")]


def test_boundary_radius_for_plots_is_undefined_sunward_of_the_nose():
    radius = boundary_radius(MP, np.vstack([at(1.0), at(2.0)]))
    assert radius[0] > 0 and np.isnan(radius[1])
