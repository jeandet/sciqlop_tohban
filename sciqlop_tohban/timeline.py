"""Turn plan intervals into the columns of one SciQLop interval timeline.

One lane per instrument; modes are categories, stacked one fixed sub-row each.
"""

from __future__ import annotations

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
    "OBS:Apoapsis": "#14b8a6",
    "OBS:Periapsis": "#10b981",
    "OBS:LowReso": "#f97316",
    "OBS:Mag": "#ec4899",
    "OBS:SW": "#06b6d4",
}
CATEGORY_ORDER = tuple(MODE_COLORS)


@dataclass(frozen=True)
class TimelineData:
    start: np.ndarray
    stop: np.ndarray
    lane: list[str]
    category: list[str]


def timeline_data(intervals: Iterable[Interval]) -> TimelineData:
    rows = sorted(intervals, key=lambda i: i.start)
    return TimelineData(
        start=np.array([i.start for i in rows], dtype="datetime64[s]"),
        stop=np.array([i.stop for i in rows], dtype="datetime64[s]"),
        lane=[i.instrument for i in rows],
        category=[i.mode for i in rows],
    )


def lane_order(lanes: Iterable[str]) -> list[str]:
    present = set(lanes)
    known = [name for name in INSTRUMENT_ORDER if name in present]
    return known + sorted(present - set(known))
