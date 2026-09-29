import numpy as np

from sciqlop_tohban.evt import Interval
from sciqlop_tohban.timeline import lanes, merge_spans, to_epoch


def t(text):
    return np.datetime64(text, "s")


def test_merge_spans_unions_overlaps_and_sorts():
    spans = [(5.0, 8.0), (0.0, 2.0), (1.0, 3.0), (8.0, 9.0)]
    assert merge_spans(spans) == [(0.0, 3.0), (5.0, 9.0)]


INTERVALS = [
    Interval("MPPE_MSA", "LM", t("2030-01-01T00:00:00"), t("2030-01-01T01:00:00")),
    Interval("DPU", "BASE", t("2030-01-01T00:00:00"), t("2030-01-01T01:00:00")),
    Interval("DPU", "BASE", t("2030-01-01T02:00:00"), t("2030-01-01T03:00:00")),
    Interval("DPU", "L_HKM", t("2030-01-01T00:00:00"), t("2030-01-01T00:30:00")),
]


def test_lanes_group_by_instrument_in_mission_order():
    assert [(lane.instrument, lane.mode) for lane in lanes(INTERVALS, cadence=600)] == [
        ("DPU", "BASE"), ("DPU", "L_HKM"), ("MPPE_MSA", "LM"),
    ]


def test_lanes_are_on_off_traces_on_a_shared_regular_grid():
    # A regular grid: SciQLopPlots breaks a line wherever a step is >1.5x its neighbours'.
    base, hkm, _ = lanes(INTERVALS, cadence=600)
    t0 = to_epoch(t("2030-01-01T00:00:00"))
    assert base.x[0] == t0 and base.x[-1] == t0 + 3 * 3600
    assert np.all(np.diff(base.x) == 600)
    assert np.array_equal(base.x, hkm.x)
    low, high = base.y.min(), base.y.max()
    on = dict(zip(base.x - t0, base.y))
    assert on[0] == high and on[3000] == high and on[3600] == low
    assert on[5400] == low and on[7200] == high and on[10800] == low
    assert hkm.y.max() < low
    assert not np.isnan(base.y).any()
