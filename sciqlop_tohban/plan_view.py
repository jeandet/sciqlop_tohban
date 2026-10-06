"""A plan shown as one SciQLop interval timeline, re-rendered on each filter change."""

from __future__ import annotations

import numpy as np
from SciQLop.user_api.plot import PlotPanel, TimeRange, create_plot_panel

from .evt import Interval
from .timeline import CATEGORY_ORDER, MODE_COLORS, ViewState, display_mode, lane_order, view_data

# Labels overflow below ~12 px; 14 px fits the ~70 rows of the detailed view on 1080p.
LANE_HEIGHT = 14
# A Tohban meeting reviews half a week of plan.
SESSION_SECONDS = 3.5 * 86400


def _epoch_seconds(time: np.datetime64) -> float:
    # Epoch floats: TimeRange does not convert np.datetime64[s] (SciQLop#150).
    return float(time.astype("datetime64[s]").astype(np.int64))


def _show_first_session(panel: PlotPanel, intervals: list[Interval]) -> None:
    start = _epoch_seconds(min(i.start for i in intervals))
    if 0 < panel.zoom_limit_seconds < SESSION_SECONDS:
        panel.zoom_limit_seconds = SESSION_SECONDS
    panel.time_range = TimeRange(start, start + SESSION_SECONDS)


class PlanView:
    def __init__(self, intervals: list[Interval]):
        self._intervals = intervals
        self.panel = create_plot_panel()
        self.plot, self.timeline = self.panel.add_timeline(lane_height=LANE_HEIGHT)
        self.plot.legend_visible = False  # lanes and mode rows are named on the plot
        self.timeline.category_order = list(CATEGORY_ORDER)
        self.timeline.set_category_colors(MODE_COLORS)
        self.show(ViewState())
        if intervals:
            _show_first_session(self.panel, intervals)

    @property
    def instruments(self) -> list[str]:
        return lane_order(i.instrument for i in self._intervals)

    @property
    def modes(self) -> list[str]:
        present = {display_mode(i.mode) for i in self._intervals}
        return [m for m in CATEGORY_ORDER if m in present] + sorted(present - set(CATEGORY_ORDER))

    def show(self, state: ViewState) -> None:
        data = view_data(self._intervals, state)
        self.timeline.stack = None if state.summary else "category"
        self.timeline.set_intervals(data.start, data.stop, lane=data.lane, category=data.category, label=data.label)
        self.timeline.lanes = lane_order(data.lane)
