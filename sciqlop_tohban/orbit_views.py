"""SPICE-based plots of the plan: trajectories, MSASI field of view, distances and regions.

The XY plots have no time axis: they redraw the part of the orbit inside the
plan panel's time range whenever it changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PySide6.QtCore import QTimer
from SciQLop.user_api.plot import GraphLineStyle, PlotPanel, XYPlot, create_plot_panel
from SciQLopPlots import PlotType as SqpPlotType

from .ephemeris import Ephemeris
from .evt import Interval
from .magnetosphere import (
    BS,
    MP,
    RM_KM,
    axis_distance,
    boundary_radius,
    classify,
    meridian_curve,
    orbit_plane_basis,
    plane_section,
    region_runs,
)
from .timeline import MODE_COLORS, MODE_PRIORITY, OFF, OFF_COLOR, highest_mode_at
from .trajectory import fov_on_shell, hourly, lon_lat, mode_tracks, polylines

MIO_COLOR = "#7c3aed"
MPO_COLOR = "#6b7280"
BOUNDARY_COLOR = "#16a34a"
SEPARATION_COLOR = "#0891b2"
HOUR_MARK_COLOR = "#f9fafb"
MERCURY_COLOR = "#9ca3af"
REGION_NAMES = {"MSP": "magnetosphere", "MSH": "magnetosheath", "SW": "solar wind", "unknown": "outside model"}
REGION_COLORS = {"magnetosphere": "#86efac", "magnetosheath": "#fcd34d", "solar wind": "#93c5fd", "outside model": "#e5e7eb"}
# Drawn bottom to top, so the highest mode stays visible where tracks overlap.
TRACK_MODES = (OFF, *reversed(MODE_PRIORITY))
SHELL_RM = 10.0
FOV_COLORS = (  # viridis, early to late in the shown range
    "#440154", "#482173", "#433e85", "#38588c", "#2d708e", "#25858e",
    "#1e9b8a", "#2ab07f", "#52c569", "#86d549", "#c2df23", "#fde725",
)
# Coalesce the burst of range changes of a pan or zoom into one redraw.
REDRAW_DELAY_MS = 150
_EMPTY = (np.empty(0), np.empty(0))


def _raw(graph):
    return graph._impl


def _curve(plot, label: str, color: str, width: float | None = None, dashed: bool = False):
    style = GraphLineStyle.Dash if dashed else GraphLineStyle.Solid
    graph = plot.plot(*_EMPTY, labels=[label], colors=[color], line_style=style)
    if width is not None:
        _raw(graph).set_line_width(width)
    return graph


def alive(panel: PlotPanel) -> bool:
    """False once the user closed the panel."""
    return panel._impl is not None


def _xy_plot(panel: PlotPanel, x_label: str, y_label: str, equal_aspect: bool = True) -> XYPlot:
    # user_api only creates an XY plot together with its first graph; start empty instead.
    plot = XYPlot(panel._impl.create_plot(-1, SqpPlotType.BasicXY))
    plot.set_axis_label("x", x_label)
    plot.set_axis_label("y", y_label)
    _raw(plot).set_equal_aspect_ratio(equal_aspect)
    return plot


def _set(graph, xy: np.ndarray) -> None:
    graph.set_data(xy[:, 0], xy[:, 1]) if len(xy) else graph.set_data(*_EMPTY)


def _circle(radius: float) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi, 361)
    return radius * np.column_stack([np.cos(angle), np.sin(angle)])


def _mercury(plot):
    # A curve, not an Ellipse item: Ellipse items do not render on SciQLopPlots 0.51.
    mercury = _curve(plot, "Mercury", MERCURY_COLOR, width=2)
    _set(mercury, _circle(1.0))
    return mercury


@dataclass(frozen=True)
class Projection:
    """How a window of ephemeris maps to plot coordinates, with the matching boundaries."""

    x_label: str
    y_label: str
    # eph -> (to_xy, magnetopause, bow shock); positions in km, results in R_M.
    frame: Callable[[Ephemeris], tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]]


def _meridian_frame(eph: Ephemeris):
    return (lambda p: p[:, [0, 2]] / RM_KM), meridian_curve(MP) / RM_KM, meridian_curve(BS) / RM_KM


def _orbit_plane_frame(eph: Ephemeris):
    basis = orbit_plane_basis(eph.mio, eph.mio_velocity)
    return (lambda p: p @ basis.T / RM_KM), plane_section(MP, basis) / RM_KM, plane_section(BS, basis) / RM_KM


MSO_XZ = Projection("MSO X [R_M]", "MSO Z [R_M]", _meridian_frame)
ORBIT_PLANE = Projection("apoapsis axis [R_M]", "in-plane perpendicular axis [R_M]", _orbit_plane_frame)


class TrajectoryPlot:
    """MIO track coloured by the highest active operation mode, MPO, hourly marks, MP and BS."""

    def __init__(self, panel: PlotPanel, projection: Projection):
        self.projection = projection
        self.plot = _xy_plot(panel, projection.x_label, projection.y_label)
        self.magnetopause = _curve(self.plot, "magnetopause", BOUNDARY_COLOR, width=2)
        self.bow_shock = _curve(self.plot, "bow shock", BOUNDARY_COLOR, width=2, dashed=True)
        self.mpo = _curve(self.plot, "MPO", MPO_COLOR, dashed=True)
        self.tracks = {
            mode: _curve(self.plot, mode, MODE_COLORS.get(mode, OFF_COLOR), width=3) for mode in TRACK_MODES
        }
        self.hours = self.plot.scatter(*_EMPTY, labels=["MIO 1-hour marker"], colors=[HOUR_MARK_COLOR])
        self.mercury = _mercury(self.plot)

    def update(self, eph: Ephemeris, modes: np.ndarray) -> None:
        to_xy, magnetopause, bow_shock = self.projection.frame(eph)
        mio, mpo = to_xy(eph.mio), to_xy(eph.mpo)
        tracks = mode_tracks(mio, modes)
        for mode, graph in self.tracks.items():
            _set(graph, tracks.get(mode, np.empty((0, 2))))
        _set(self.mpo, mpo)
        marks = to_xy(hourly(eph.time, eph.mio))
        # An XY scatter misplaces points given in unsorted x order.
        _set(self.hours, marks[np.argsort(marks[:, 0])])
        _set(self.magnetopause, magnetopause)
        _set(self.bow_shock, bow_shock)


def _fov_label(time: np.datetime64) -> str:
    return f"FoV {str(time.astype('datetime64[m]')).replace('T', ' ')}"


class MsasiPlots:
    """MSASI FoV cone in MSO X–Z, and its footprint on a reference shell."""

    def __init__(self, panel: PlotPanel):
        self.cone = _xy_plot(panel, "MSO X [R_M]", "MSO Z [R_M]")
        self.mio = _curve(self.cone, "MIO", MERCURY_COLOR, width=2)
        self.mpo = _curve(self.cone, "MPO", MPO_COLOR, dashed=True)
        self.shell = _curve(self.cone, f"shell R={SHELL_RM:g} R_M", MPO_COLOR, dashed=True)
        self.mercury = _mercury(self.cone)
        self.rays = [_curve(self.cone, "", color) for color in FOV_COLORS]
        self.cone.legend_visible = False
        self.footprint = _xy_plot(panel, "Mercury-centred longitude [deg]", "latitude [deg]", equal_aspect=False)
        self.footprints = [_curve(self.footprint, "", color, width=2) for color in FOV_COLORS]

    def update(self, eph: Ephemeris) -> None:
        shell_km = max(SHELL_RM * RM_KM, 1.05 * eph.mio_radius.max())
        picked, hits = fov_on_shell(eph.mio, eph.msasi_corners, shell_km, len(FOV_COLORS))
        _set(self.mio, eph.mio[:, [0, 2]] / RM_KM)
        _set(self.mpo, eph.mpo[:, [0, 2]] / RM_KM)
        _set(self.shell, _circle(shell_km / RM_KM))
        for index, (rays, footprint) in enumerate(zip(self.rays, self.footprints)):
            if index >= len(picked):
                rays.set_data(*_EMPTY)
                footprint.set_data(*_EMPTY)
                continue
            origin, corners = eph.mio[picked[index], [0, 2]] / RM_KM, hits[index][:, [0, 2]] / RM_KM
            outline = np.vstack([corners, corners[:1]])
            rays.set_data(*polylines([np.vstack([origin, corner]) for corner in corners] + [outline]))
            lon, lat = lon_lat(np.vstack([hits[index], hits[index][:1]]))
            footprint.set_data(lon, lat)
            _raw(footprint).set_labels([_fov_label(eph.time[picked[index]])])


class OrbitViews:
    """Trajectory and MSASI panels following the plan panel's time range."""

    def __init__(self, plan_panel: PlotPanel, eph: Ephemeris, intervals: list[Interval]):
        self._eph = eph
        self._intervals = intervals
        self._plan_panel = plan_panel
        self.trajectory_panel = create_plot_panel("Tohban trajectory")
        self.trajectories = [TrajectoryPlot(self.trajectory_panel, p) for p in (MSO_XZ, ORBIT_PLANE)]
        self.msasi_panel = create_plot_panel("Tohban MSASI")
        self.msasi = MsasiPlots(self.msasi_panel)
        self._timer = QTimer(singleShot=True, interval=REDRAW_DELAY_MS, timeout=self.redraw)
        plan_panel._impl.time_range_changed.connect(lambda _: self._timer.start())
        self.redraw()

    def set_intervals(self, intervals: list[Interval]) -> None:
        self._intervals = intervals
        self.redraw()

    def redraw(self) -> None:
        if not alive(self._plan_panel):
            return
        shown = self._plan_panel.time_range
        eph = self._eph.window(_datetime64(shown.start()), _datetime64(shown.stop()))
        if len(eph.time) < 2:
            return
        if alive(self.trajectory_panel):
            modes = highest_mode_at(eph.time, self._intervals)
            for plot in self.trajectories:
                plot.update(eph, modes)
        if alive(self.msasi_panel):
            self.msasi.update(eph)


def _datetime64(epoch: float) -> np.datetime64:
    return np.datetime64(int(epoch), "s")


def _distance_series(eph: Ephemeris) -> list[tuple[str, np.ndarray, list[str], list[str]]]:
    """(y axis label, columns, legend labels, colours) of each distance plot."""
    mio, mpo = eph.mio, eph.mpo
    return [
        ("distance to Mercury", np.column_stack([eph.mio_radius, eph.mpo_radius]),
         ["MIO", "MPO"], [MIO_COLOR, MPO_COLOR]),
        ("distance to MSO X axis",
         np.column_stack([axis_distance(mio), boundary_radius(MP, mio), boundary_radius(BS, mio),
                          axis_distance(mpo), boundary_radius(MP, mpo), boundary_radius(BS, mpo)]),
         ["MIO rho", "MP at MIO X", "BS at MIO X", "MPO rho", "MP at MPO X", "BS at MPO X"],
         [MIO_COLOR, "#15803d", "#4ade80", MPO_COLOR, "#166534", "#86efac"]),
        ("separation", eph.separation, ["MIO–MPO"], [SEPARATION_COLOR]),
    ]


# Relative plot heights: the plan timeline needs room for its ~25 lanes.
PLAN_STRETCH = 4
DISTANCE_STRETCH = 2


def add_distance_plots(panel: PlotPanel, plan_plot, eph: Ephemeris) -> None:
    """Regions, radial distances, distance to the MSO X axis vs MP/BS, and separation."""
    add_region_timeline(panel, eph)
    panel._impl.set_plot_stretch(plan_plot._impl, PLAN_STRETCH)
    # plot_data reads any datetime64 as nanoseconds (speasy datetime64_to_epoch).
    time = eph.time.astype("datetime64[ns]")
    for y_label, columns, labels, colors in _distance_series(eph):
        plot, _ = panel.plot_data(time, columns, labels=labels, colors=colors)
        plot.set_axis_label("y", y_label, unit="km")
        panel._impl.set_plot_stretch(plot._impl, DISTANCE_STRETCH)


def add_region_timeline(panel: PlotPanel, eph: Ephemeris) -> None:
    runs = [
        (start, stop, lane, REGION_NAMES[region])
        for lane, points in (("MIO region", eph.mio), ("MPO region", eph.mpo))
        for start, stop, region in region_runs(eph.time, classify(points))
    ]
    plot, timeline = panel.add_timeline(plot_index=1)
    plot.legend_visible = False
    timeline.style = "bars"
    timeline.set_category_colors(REGION_COLORS)
    start, stop, lane, region = zip(*runs)
    timeline.set_intervals(np.array(start), np.array(stop), lane=list(lane), category=list(region))
    timeline.lanes = ["MIO region", "MPO region"]
