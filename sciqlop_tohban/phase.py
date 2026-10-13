"""Fold plan intervals onto one MIO orbit: time since the previous apoapsis."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .evt import Interval
from .timeline import lane_order, summarize, timeline_data
from .trajectory import polylines

HOUR = np.timedelta64(3600, "s")


def apoapses(time: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Times of the radius peaks, plus one orbit before the first and after the last.

    The padding lets the partial orbits at both ends of the plan fold too.
    """
    # simplify: peaks at sample resolution (5 min); refine with a parabola if phases need better.
    peaks = np.flatnonzero((radius[1:-1] > radius[:-2]) & (radius[1:-1] >= radius[2:])) + 1
    if len(peaks) < 2:
        return np.empty(0, dtype="datetime64[s]")
    times = time[peaks].astype("datetime64[s]")
    period = np.median(np.diff(times))
    return np.concatenate([[times[0] - period], times, [times[-1] + period]])


def fold(start: np.datetime64, stop: np.datetime64, apo: np.ndarray) -> list[tuple[float, float]]:
    """[start, stop] as (from, to) hours since apoapsis, one piece per orbit it spans."""
    edges = np.concatenate([[start], apo[(apo > start) & (apo < stop)], [stop]])
    pieces = []
    for left, right in zip(edges[:-1], edges[1:]):
        orbit = np.searchsorted(apo, left, side="right") - 1
        if orbit < 0 or orbit >= len(apo) - 1:
            continue
        pieces.append(((left - apo[orbit]) / HOUR, (right - apo[orbit]) / HOUR))
    return pieces


def clip(intervals: Iterable[Interval], start: np.datetime64, stop: np.datetime64) -> list[Interval]:
    return [
        Interval(i.instrument, i.mode, max(i.start, start), min(i.stop, stop))
        for i in intervals
        if i.start < stop and i.stop > start
    ]


def folded_bars(intervals: Iterable[Interval], apo: np.ndarray) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Summary lanes top to bottom, and per mode the x, y polylines of its folded bars.

    Lane i sits at y = -i, so the first lane is on top.
    """
    data = timeline_data(summarize(intervals), summary=True)
    rows = lane_order(data.lane)
    row_of = {lane: -float(index) for index, lane in enumerate(rows)}
    pieces: dict[str, list[np.ndarray]] = {}
    for start, stop, lane, mode in zip(data.start, data.stop, data.lane, data.category):
        for left, right in fold(start, stop, apo):
            pieces.setdefault(mode, []).append(np.array([[left, row_of[lane]], [right, row_of[lane]]]))
    return rows, {mode: polylines(curves) for mode, curves in pieces.items()}
