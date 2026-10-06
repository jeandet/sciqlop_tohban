import numpy as np

from sciqlop_tohban.evt import Interval
from sciqlop_tohban.timeline import (
    CATEGORY_ORDER,
    ViewState,
    lane_order,
    select,
    summarize,
    timeline_data,
    view_data,
)


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


def test_categories_drop_the_obs_prefix():
    mag = Interval("MPPE_MIA", "OBS:Mag", t("2030-01-01T00:00:00"), t("2030-01-01T01:00:00"))
    assert timeline_data([mag]).category == ["Mag"]
    assert "Mag" in CATEGORY_ORDER and "OBS:Mag" not in CATEGORY_ORDER


def h(hours):
    return t("2030-01-01T00:00:00") + np.timedelta64(int(hours * 3600), "s")


def test_summary_keeps_the_highest_priority_mode_at_each_time():
    windows = [
        Interval("MGF_I", "BASE", h(0), h(3)),
        Interval("MGF_I", "HKM", h(0), h(1)),
        Interval("MGF_I", "LM", h(0.5), h(2)),
    ]
    assert summarize(windows) == [
        Interval("MGF_I", "HKM", h(0), h(1)),
        Interval("MGF_I", "LM", h(1), h(2)),
        Interval("MGF_I", "BASE", h(2), h(3)),
    ]


def test_summary_merges_touching_windows_of_the_same_mode():
    windows = [Interval("MDM", "BASE", h(0), h(1)), Interval("MDM", "BASE", h(1), h(2))]
    assert summarize(windows) == [Interval("MDM", "BASE", h(0), h(2))]


def test_summary_passes_obs_modes_through_on_their_own_lane():
    obs = Interval("MPPE_ENA", "OBS:Periapsis", h(0), h(1))
    power = Interval("MPPE_ENA", "BASE", h(0), h(2))
    assert summarize([obs, power]) == [power, obs]
    assert timeline_data(summarize([obs, power]), summary=True).lane == ["MPPE_ENA", "MPPE_ENA OBS"]


def test_detailed_view_keeps_obs_modes_on_the_instrument_lane():
    obs = Interval("MPPE_ENA", "OBS:Periapsis", h(0), h(1))
    assert timeline_data([obs]).lane == ["MPPE_ENA"]


def test_lane_order_puts_each_obs_lane_after_its_instrument():
    assert lane_order(["MPPE_MSA", "MPPE_ENA OBS", "MPPE_ENA", "DPU"]) == [
        "DPU", "MPPE_ENA", "MPPE_ENA OBS", "MPPE_MSA",
    ]


def test_select_hides_instruments_and_modes():
    windows = [
        Interval("DPU", "BASE", h(0), h(1)),
        Interval("MDM", "BASE", h(0), h(1)),
        Interval("MDM", "HKM", h(0), h(1)),
        Interval("MPPE_MIA", "OBS:Mag", h(0), h(1)),
    ]
    assert select(windows, hidden_instruments={"DPU"}, hidden_modes={"HKM", "Mag"}) == [windows[1]]


def test_view_hides_modes_before_summarizing():
    windows = [Interval("MGF_I", "BASE", h(0), h(2)), Interval("MGF_I", "HKM", h(0), h(1))]
    data = view_data(windows, ViewState(summary=True, hidden_modes=frozenset({"HKM"})))
    assert data.category == ["BASE"] and data.start.tolist() == [h(0)] and data.stop.tolist() == [h(2)]


def test_detailed_view_shows_every_window():
    windows = [Interval("MGF_I", "BASE", h(0), h(2)), Interval("MGF_I", "HKM", h(0), h(1))]
    assert view_data(windows, ViewState(summary=False)).category == ["BASE", "HKM"]


def test_summary_bars_are_labelled_with_their_mode():
    windows = [Interval("MGF_I", "BASE", h(0), h(2)), Interval("MPPE_MIA", "OBS:Mag", h(0), h(1))]
    assert view_data(windows, ViewState(summary=True)).label == ["BASE", "Mag"]


def test_detailed_bars_are_unlabelled_since_rows_carry_the_mode_name():
    windows = [Interval("MGF_I", "BASE", h(0), h(2))]
    assert view_data(windows, ViewState(summary=False)).label is None
