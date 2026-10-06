"""Turn plan intervals into the columns of one SciQLop interval timeline.

Detailed view: one lane per instrument, modes stacked one fixed sub-row each.
Summary view: one lane per instrument showing its highest active mode, plus an
"<instrument> OBS" lane for observation modes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .evt import Interval

INSTRUMENT_ORDER = (
    "DPU", "MPPE_HEPi", "MPPE_HEPe", "MPPE_ENA", "MPPE_MIA", "MPPE_MEA", "MPPE_MSA",
    "MGF_I", "MGF_O", "MDM", "MSASI", "P_EWO_E", "P_EWO_B", "P_EFD", "P_AM2P", "P_SORBET",
)

MODE_COLORS = {
    "BASE": "#9ca3af",
    "HKM": "#2563eb",
    "LM": "#f59e0b",
    "MM": "#ef4444",
    "L_HKM": "#1d4ed8",
    "M_HKM": "#7c3aed",
    "Apoapsis": "#14b8a6",
    "Periapsis": "#10b981",
    "LowReso": "#f97316",
    "Mag": "#ec4899",
    "SW": "#06b6d4",
}
CATEGORY_ORDER = tuple(MODE_COLORS)
# Highest first; the default of the original quicklook script.
MODE_PRIORITY = ("M_HKM", "L_HKM", "HKM", "MM", "LM", "BASE")
OBS_PREFIX = "OBS:"
OBS_LANE_SUFFIX = " OBS"


@dataclass(frozen=True)
class TimelineData:
    start: np.ndarray
    stop: np.ndarray
    lane: list[str]
    category: list[str]


def is_obs(interval: Interval) -> bool:
    return interval.mode.startswith(OBS_PREFIX)


def display_mode(mode: str) -> str:
    return mode.removeprefix(OBS_PREFIX)


def _lane(interval: Interval, summary: bool) -> str:
    return interval.instrument + OBS_LANE_SUFFIX if summary and is_obs(interval) else interval.instrument


def timeline_data(intervals: Iterable[Interval], summary: bool = False) -> TimelineData:
    rows = sorted(intervals, key=lambda i: i.start)
    return TimelineData(
        start=np.array([i.start for i in rows], dtype="datetime64[s]"),
        stop=np.array([i.stop for i in rows], dtype="datetime64[s]"),
        lane=[_lane(i, summary) for i in rows],
        category=[display_mode(i.mode) for i in rows],
    )


def _instrument_rank(instrument: str) -> int:
    return INSTRUMENT_ORDER.index(instrument) if instrument in INSTRUMENT_ORDER else len(INSTRUMENT_ORDER)


def _lane_key(lane: str) -> tuple[int, str, bool]:
    instrument = lane.removesuffix(OBS_LANE_SUFFIX)
    return _instrument_rank(instrument), instrument, lane.endswith(OBS_LANE_SUFFIX)


def lane_order(lanes: Iterable[str]) -> list[str]:
    return sorted(set(lanes), key=_lane_key)


def _priority(mode: str) -> int:
    return -MODE_PRIORITY.index(mode) if mode in MODE_PRIORITY else -len(MODE_PRIORITY)


def _highest_mode_spans(instrument: str, windows: list[Interval]) -> list[Interval]:
    """Sweep the window edges, keeping the highest-priority active mode between them."""
    edges: dict[np.datetime64, list[tuple[str, int]]] = defaultdict(list)
    for window in windows:
        edges[window.start].append((window.mode, +1))
        edges[window.stop].append((window.mode, -1))
    active: Counter[str] = Counter()
    spans: list[Interval] = []
    times = sorted(edges)
    for time, next_time in zip(times, times[1:]):
        for mode, change in edges[time]:
            active[mode] += change
        modes = [mode for mode, count in active.items() if count > 0]
        if not modes:
            continue
        mode = max(modes, key=_priority)
        if spans and spans[-1].mode == mode and spans[-1].stop == time:
            spans[-1] = Interval(instrument, mode, spans[-1].start, next_time)
        else:
            spans.append(Interval(instrument, mode, time, next_time))
    return spans


def summarize(intervals: Iterable[Interval]) -> list[Interval]:
    """Highest active power mode per instrument; observation modes pass through.

    Zero-length windows disappear: they are active for no time.
    """
    power: dict[str, list[Interval]] = defaultdict(list)
    obs: list[Interval] = []
    for interval in intervals:
        if is_obs(interval):
            obs.append(interval)
        else:
            power[interval.instrument].append(interval)
    spans = [span for instrument, windows in power.items() for span in _highest_mode_spans(instrument, windows)]
    return sorted(spans + obs, key=lambda i: (i.start, is_obs(i)))


def select(intervals: Iterable[Interval], hidden_instruments: set[str], hidden_modes: set[str]) -> list[Interval]:
    return [
        i for i in intervals
        if i.instrument not in hidden_instruments and display_mode(i.mode) not in hidden_modes
    ]


@dataclass(frozen=True)
class ViewState:
    summary: bool = True
    hidden_instruments: frozenset[str] = frozenset()
    hidden_modes: frozenset[str] = frozenset()


def view_data(intervals: Iterable[Interval], state: ViewState) -> TimelineData:
    # Hide first: summarizing first would let a hidden mode mask the modes below it.
    shown = select(intervals, state.hidden_instruments, state.hidden_modes)
    return timeline_data(summarize(shown) if state.summary else shown, summary=state.summary)
