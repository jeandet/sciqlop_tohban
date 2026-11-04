"""Fold plan intervals onto one MIO orbit: time since the previous apoapsis."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from .evt import Interval
from .timeline import View, display_mode, is_obs, lane_order, states, summarize
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


def orbit_containing(apo: np.ndarray, time: np.datetime64) -> int:
    """Index of the orbit (apoapsis ``i`` to ``i + 1``) holding ``time``, clamped to the known orbits."""
    return int(np.clip(np.searchsorted(apo, time, side="right") - 1, 0, len(apo) - 2))


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


def power_and_obs(intervals: Iterable[Interval], view: View) -> tuple[list[Interval], list[Interval]]:
    """One row per instrument each: its power state as the view shows it, and its OBS mode.

    The detailed view stacks modes in sub-rows, which do not fold; it gets the summary.
    """
    intervals = list(intervals)
    power = [i for i in intervals if not is_obs(i)]
    obs = [replace(i, mode=display_mode(i.mode)) for i in intervals if is_obs(i)]
    return (states if view is View.STATE else summarize)(power), obs


def folded_bars(intervals: Iterable[Interval], apo: np.ndarray) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Instruments top to bottom, and per mode the x, y polylines of its folded bars.

    Instrument i sits at y = -i, so the first one is on top. Expects at most one mode per instrument at a time.
    """
    intervals = list(intervals)
    rows = lane_order(i.instrument for i in intervals)
    row_of = {lane: -float(index) for index, lane in enumerate(rows)}
    pieces: dict[str, list[np.ndarray]] = {}
    for interval in intervals:
        y = row_of[interval.instrument]
        for left, right in fold(interval.start, interval.stop, apo):
            pieces.setdefault(interval.mode, []).append(np.array([[left, y], [right, y]]))
    return rows, {mode: polylines(curves) for mode, curves in pieces.items()}
