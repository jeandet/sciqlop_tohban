from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QFileDialog, QWidget
from shiboken6 import isValid
from SciQLop.user_api.plot import OverlayLevel

from .ephemeris import Ephemeris, cached_sample
from .evt import EvtFile, Interval, obs_mode_intervals, pair_intervals, parse
from .filter_panel import FilterPanel
from .kernels import metakernel
from .orbit_views import OrbitViews, add_distance_plots, alive
from .plan_view import PlanView
from .recent import RecentPlans
from .resources import ICON_PATH
from .timeline import ViewState, select


def read_plan(path: Path) -> EvtFile:
    # newline="" keeps CRLF so that writing the plan back is byte-identical.
    with open(path, encoding="ascii", newline="") as handle:
        return parse(handle.read())


def plan_intervals(evt: EvtFile) -> list[Interval]:
    intervals, points = pair_intervals(evt.records)
    return intervals + obs_mode_intervals(points, intervals)


def side_pane_title(path: Path) -> str:
    # ESA names every plan OBS_PLAN.evt: the folder tells two open plans apart.
    return f"Tohban: {path.parent.name}/{path.name}"


def show_side_pane(widget: QWidget) -> None:
    dock = widget
    while dock is not None and not hasattr(dock, "toggleView"):
        dock = dock.parentWidget()
    if dock is None:
        return
    dock.toggleView(True)
    if (container := dock.autoHideDockContainer()) is not None:
        container.collapseView(False)


def plan_span(intervals: list[Interval]) -> tuple[np.datetime64, np.datetime64]:
    return min(i.start for i in intervals), max(i.stop for i in intervals)


class EphemerisLoader(QObject):
    """Downloads kernels and samples SPICE off the GUI thread; signals arrive on the GUI thread."""

    loaded = Signal(object)  # Ephemeris
    failed = Signal(str)

    def start(self, start: np.datetime64, stop: np.datetime64) -> None:
        threading.Thread(target=self._run, args=(start, stop), daemon=True).start()

    def _run(self, start: np.datetime64, stop: np.datetime64) -> None:
        try:
            self.loaded.emit(cached_sample(metakernel(), start, stop))
        except Exception as error:  # noqa: BLE001 - any failure must reach the user, not die in the thread
            self.failed.emit(f"{type(error).__name__}: {error}")


class OpenPlan:
    """A plan on screen: its timeline, filters, and the SPICE views once the ephemeris is in."""

    def __init__(self, evt: EvtFile):
        self.evt = evt
        self.intervals = plan_intervals(evt)
        self.view = PlanView(self.intervals)
        self.filters = FilterPanel(self.view.instruments, self.view.modes)
        self.filters.changed.connect(self.show)
        self.orbit: OrbitViews | None = None
        if self.intervals:
            self._load_ephemeris()

    def show(self, state: ViewState) -> None:
        self.view.show(state)
        if self.orbit is not None:
            self.orbit.set_intervals(select(self.intervals, state.hidden_instruments, state.hidden_modes), state.view)

    def _load_ephemeris(self) -> None:
        self.view.plot.overlay.show("Loading SPICE ephemeris (first time: downloading ~230 MB of ESA kernels)…",
                                    level=OverlayLevel.Info)
        self.loader = EphemerisLoader()
        self.loader.loaded.connect(self._add_orbit_views)
        self.loader.failed.connect(
            lambda message: self.view.plot.overlay.show(f"No SPICE views: {message}", level=OverlayLevel.Error)
        )
        self.loader.start(*plan_span(self.intervals))

    def _add_orbit_views(self, eph: Ephemeris) -> None:
        if not alive(self.view.panel):
            return
        self.view.plot.overlay.clear()
        add_distance_plots(self.view.panel, self.view.plot, eph)
        state = self.filters.state()
        self.orbit = OrbitViews(
            self.view.panel, eph, select(self.intervals, state.hidden_instruments, state.hidden_modes), state.view
        )
        self._add_orbit_stepper()

    def _add_orbit_stepper(self) -> None:
        orbits = self.filters.orbits
        orbits.set_apoapses(self.orbit.apoapses)
        orbits.follow(self.view.middle())
        orbits.selected.connect(self.view.show_window)
        self.view.panel._impl.time_range_changed.connect(lambda _: orbits.follow(self.view.middle()))


class TohbanPlugin:
    def __init__(self, main_window):
        self._main_window = main_window
        self.plans: dict[Path, OpenPlan] = {}
        self.open_action = QAction(QIcon(str(ICON_PATH)), "Open Tohban observation plan…", main_window)
        self.open_action.setToolTip("Open a Mio observation plan (.evt) as a timeline")
        self.open_action.triggered.connect(self.open_plan)
        main_window.toolsMenu.addAction(self.open_action)
        main_window.toolBar.addAction(self.open_action)
        # QSettings() is SciQLop's own store: the application sets its organisation and name.
        self.recent = RecentPlans(QSettings(), main_window)
        self.recent.chosen.connect(self.open)
        main_window.toolsMenu.addMenu(self.recent.menu)

    def open_plan(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self._main_window, "Open observation plan", "", "Event files (*.evt);;All files (*)"
        )
        if name:
            self.open(Path(name))

    def open(self, path: Path) -> OpenPlan:
        path = path.resolve()
        if (plan := self._still_open(path)) is not None:
            show_side_pane(plan.filters)
            return plan
        plan = OpenPlan(read_plan(path))
        plan.filters.setWindowTitle(side_pane_title(path))
        self._main_window.add_side_pan(plan.filters, icon=str(ICON_PATH))
        self.plans[path] = plan
        self.recent.add(path)
        return plan

    def _still_open(self, path: Path) -> OpenPlan | None:
        plan = self.plans.get(path)
        return plan if plan is not None and isValid(plan.filters) and alive(plan.view.panel) else None

    async def close(self) -> None:
        self.plans.clear()


def load(main_window):
    return TohbanPlugin(main_window)
