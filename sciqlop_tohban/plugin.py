from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog

from .evt import EvtFile, Interval, obs_mode_intervals, pair_intervals, parse
from .filter_panel import FilterPanel
from .plan_view import PlanView


def read_plan(path: Path) -> EvtFile:
    # newline="" keeps CRLF so that writing the plan back is byte-identical.
    with open(path, encoding="ascii", newline="") as handle:
        return parse(handle.read())


def plan_intervals(evt: EvtFile) -> list[Interval]:
    intervals, points = pair_intervals(evt.records)
    return intervals + obs_mode_intervals(points, intervals)


def plot_plan(evt: EvtFile) -> tuple[PlanView, FilterPanel]:
    view = PlanView(plan_intervals(evt))
    filters = FilterPanel(view.instruments, view.modes)
    filters.changed.connect(view.show)
    return view, filters


class TohbanPlugin:
    def __init__(self, main_window):
        self._main_window = main_window
        self.plans: dict[Path, tuple[EvtFile, PlanView, FilterPanel]] = {}
        main_window.toolsMenu.addAction("Open Tohban observation plan…", self.open_plan)

    def open_plan(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self._main_window, "Open observation plan", "", "Event files (*.evt);;All files (*)"
        )
        if name:
            path = Path(name)
            evt = read_plan(path)
            view, filters = plot_plan(evt)
            filters.setWindowTitle(f"Tohban: {path.name}")
            self._main_window.add_side_pan(filters)
            self.plans[path] = (evt, view, filters)

    async def close(self) -> None:
        self.plans.clear()


def load(main_window):
    return TohbanPlugin(main_window)
