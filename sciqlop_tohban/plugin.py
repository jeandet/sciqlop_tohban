from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog
from SciQLop.user_api.plot import PlotPanel, TimeRange, create_plot_panel

from .evt import EvtFile, Interval, obs_mode_intervals, pair_intervals, parse
from .timeline import CATEGORY_ORDER, MODE_COLORS, TimelineData, lane_order, timeline_data

# Labels overflow below ~12 px; 14 px fits ~70 mode rows on a 1080p screen.
LANE_HEIGHT = 14


def read_plan(path: Path) -> EvtFile:
    # newline="" keeps CRLF so that writing the plan back is byte-identical.
    with open(path, encoding="ascii", newline="") as handle:
        return parse(handle.read())


def plan_intervals(evt: EvtFile) -> list[Interval]:
    intervals, points = pair_intervals(evt.records)
    return intervals + obs_mode_intervals(points, intervals)


def _epoch_seconds(time: np.datetime64) -> float:
    return float(time.astype("datetime64[s]").astype(np.int64))


def _show_whole_plan(panel: PlotPanel, data: TimelineData) -> None:
    # Epoch floats: TimeRange does not convert np.datetime64 (SciQLop 0.14.0.dev0).
    start, stop = _epoch_seconds(data.start.min()), _epoch_seconds(data.stop.max())
    if 0 < panel.zoom_limit_seconds < stop - start:
        panel.zoom_limit_seconds = stop - start
    panel.time_range = TimeRange(start, stop)


def plot_plan(evt: EvtFile) -> PlotPanel:
    data = timeline_data(plan_intervals(evt))
    panel = create_plot_panel()
    plot, timeline = panel.add_timeline(lane_height=LANE_HEIGHT)
    plot.legend_visible = False  # every mode row is named at its start
    timeline.stack = "category"
    timeline.category_order = list(CATEGORY_ORDER)
    timeline.set_category_colors(MODE_COLORS)
    timeline.set_intervals(data.start, data.stop, lane=data.lane, category=data.category)
    timeline.lanes = lane_order(data.lane)
    if data.lane:
        _show_whole_plan(panel, data)
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
