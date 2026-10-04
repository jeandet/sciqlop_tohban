import numpy as np

from sciqlop_tohban.evt import Interval
from sciqlop_tohban.timeline import CATEGORY_ORDER, lane_order, timeline_data


def t(text):
    return np.datetime64(text, "s")


INTERVALS = [
    Interval("MPPE_MSA", "LM", t("2030-01-01T01:00:00"), t("2030-01-01T02:00:00")),
    Interval("DPU", "BASE", t("2030-01-01T00:00:00"), t("2030-01-01T03:00:00")),
    Interval("DPU", "L_HKM", t("2030-01-01T00:00:00"), t("2030-01-01T00:30:00")),
]


def test_timeline_data_has_one_aligned_row_per_interval_sorted_by_start():
    data = timeline_data(INTERVALS)
    assert data.start.tolist() == [t("2030-01-01T00:00:00")] * 2 + [t("2030-01-01T01:00:00")]
    assert data.stop.tolist() == [t("2030-01-01T03:00:00"), t("2030-01-01T00:30:00"), t("2030-01-01T02:00:00")]
    assert data.lane == ["DPU", "DPU", "MPPE_MSA"]
    assert data.category == ["BASE", "L_HKM", "LM"]


def test_timeline_data_of_nothing_is_empty():
    data = timeline_data([])
    assert len(data.start) == len(data.stop) == len(data.lane) == len(data.category) == 0


def test_lane_order_follows_mission_order_then_unknown_alphabetically():
    assert lane_order(["ZETA", "MPPE_MSA", "ALPHA", "DPU", "DPU"]) == ["DPU", "MPPE_MSA", "ALPHA", "ZETA"]


def test_category_order_starts_with_power_then_submodes():
    assert CATEGORY_ORDER[:6] == ("BASE", "HKM", "LM", "MM", "L_HKM", "M_HKM")
