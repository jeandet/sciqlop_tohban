from __future__ import annotations

from itertools import groupby
from pathlib import Path

from PySide6.QtWidgets import QFileDialog
from SciQLop.user_api.plot import PlotPanel, TimeRange, create_plot_panel

from .evt import EvtFile, obs_mode_intervals, pair_intervals, parse
from .timeline import Lane, lanes


def read_plan(path: Path) -> EvtFile:
    # newline="" keeps CRLF so that writing the plan back is byte-identical.
    with open(path, encoding="ascii", newline="") as handle:
        return parse(handle.read())


def plan_lanes(evt: EvtFile) -> list[Lane]:
    intervals, points = pair_intervals(evt.records)
    return lanes(intervals + obs_mode_intervals(points, intervals))


def _show_whole_plan(panel: PlotPanel, all_lanes: list[Lane]) -> None:
    start = min(lane.x[0] for lane in all_lanes)
    stop = max(lane.x[-1] for lane in all_lanes)
    if 0 < panel.zoom_limit_seconds < stop - start:
        panel.zoom_limit_seconds = stop - start
    panel.time_range = TimeRange(start, stop)


def _plot_instrument(panel: PlotPanel, instrument: str, instrument_lanes: list[Lane]) -> None:
    first, *others = instrument_lanes
    plot, _ = panel.plot_data(first.x, first.y, name=first.mode, colors=[first.color])
    for lane in others:
        plot.plot(lane.x, lane.y, name=lane.mode, colors=[lane.color])
    plot.set_axis_label("y", instrument)
    plot.set_axis_range("y", 0.5 - len(instrument_lanes), 0.5)


def plot_plan(evt: EvtFile) -> PlotPanel:
    all_lanes = plan_lanes(evt)
    panel = create_plot_panel()
    for instrument, instrument_lanes in groupby(all_lanes, key=lambda lane: lane.instrument):
        _plot_instrument(panel, instrument, list(instrument_lanes))
    if all_lanes:
        _show_whole_plan(panel, all_lanes)
    return panel


class TohbanPlugin:
    def __init__(self, main_window):
        self._main_window = main_window
        self.plans: dict[Path, EvtFile] = {}
        main_window.toolsMenu.addAction("Open Tohban observation plan…", self.open_plan)

    def open_plan(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self._main_window, "Open observation plan", "", "Event files (*.evt);;All files (*)"
        )
        if name:
            path = Path(name)
            self.plans[path] = read_plan(path)
            plot_plan(self.plans[path])

    async def close(self) -> None:
        self.plans.clear()


def load(main_window):
    return TohbanPlugin(main_window)
