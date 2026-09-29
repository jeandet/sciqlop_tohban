import numpy as np
import pytest

from sciqlop_tohban.evt import (
    Command,
    Interval,
    Record,
    dump,
    obs_mode_intervals,
    pair_intervals,
    parse,
    parse_command,
)

# Synthetic plan: command names follow the .evt convention, times are made up.
SAMPLE = (
    "# Observation Plan Event \r\n"
    "# event time \r\n"
    "SI_DPU_ON 2030-01-01T00:00:00\r\n"
    "SI_DPU_L_HKM_ON 2030-01-01T00:00:00\r\n"
    "SI_MPPE_MSA_ON 2030-01-01T00:00:00\r\n"
    "SI_MPPE_MSA_LM_ON 2030-01-01T00:00:00\r\n"
    "SI_MPPE_ENA_OBS_MODE_Periapsis 2030-01-01T00:10:00\r\n"
    "SI_MPPE_MSA_LM_OFF 2030-01-01T01:00:00\r\n"
    "SI_MPPE_MSA_OFF 2030-01-01T01:00:00\r\n"
    "SI_DPU_L_HKM_OFF 2030-01-01T01:00:00\r\n"
    "SI_DPU_OFF 2030-01-01T02:00:00\r\n"
)


def t(text):
    return np.datetime64(text, "s")


def test_round_trip_is_byte_identical():
    assert dump(parse(SAMPLE)) == SAMPLE


def test_round_trip_keeps_lf_and_missing_final_newline():
    text = "# h\nSI_DPU_ON 2030-01-01T00:00:00\nSI_DPU_OFF 2030-01-01T01:00:00"
    assert dump(parse(text)) == text


def test_parse_splits_header_and_records():
    evt = parse(SAMPLE)
    assert evt.header == ("# Observation Plan Event ", "# event time ")
    assert evt.records[0] == Record("SI_DPU_ON", t("2030-01-01T00:00:00"))
    assert len(evt.records) == 9


def test_parse_rejects_malformed_line_with_its_number():
    with pytest.raises(ValueError, match="line 2"):
        parse("# h\nSI_DPU_ON 2030-01-01 00:00\n")


def test_parse_rejects_comment_after_records():
    with pytest.raises(ValueError, match="line 3"):
        parse("SI_DPU_ON 2030-01-01T00:00:00\nSI_DPU_OFF 2030-01-01T01:00:00\n# late\n")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("SI_DPU_ON", Command("DPU", "BASE", "ON")),
        ("SI_DPU_L_HKM_OFF", Command("DPU", "L_HKM", "OFF")),
        ("SI_MPPE_MSA_LM_ON", Command("MPPE_MSA", "LM", "ON")),
        ("SI_MSASI_HKM_ON", Command("MSASI", "HKM", "ON")),
        ("SI_P_EWO_B_MM_OFF", Command("P_EWO_B", "MM", "OFF")),
        ("SI_MPPE_ENA_OBS_MODE_Periapsis", Command("MPPE_ENA", "OBS:Periapsis", None)),
    ],
)
def test_parse_command(name, expected):
    assert parse_command(name) == expected


def test_pair_intervals_and_points():
    intervals, points = pair_intervals(parse(SAMPLE).records)
    assert Interval("MPPE_MSA", "LM", t("2030-01-01T00:00:00"), t("2030-01-01T01:00:00")) in intervals
    assert Interval("DPU", "BASE", t("2030-01-01T00:00:00"), t("2030-01-01T02:00:00")) in intervals
    assert len(intervals) == 4
    assert points == [Record("SI_MPPE_ENA_OBS_MODE_Periapsis", t("2030-01-01T00:10:00"))]


def test_pair_intervals_is_fifo_for_overlapping_on():
    records = parse(
        "SI_DPU_ON 2030-01-01T00:00:00\n"
        "SI_DPU_ON 2030-01-01T00:30:00\n"
        "SI_DPU_OFF 2030-01-01T01:00:00\n"
        "SI_DPU_OFF 2030-01-01T02:00:00\n"
    ).records
    intervals, _ = pair_intervals(records)
    assert [(i.start, i.stop) for i in intervals] == [
        (t("2030-01-01T00:00:00"), t("2030-01-01T01:00:00")),
        (t("2030-01-01T00:30:00"), t("2030-01-01T02:00:00")),
    ]


def test_unmatched_on_is_open_ended_at_last_record():
    records = parse("SI_DPU_ON 2030-01-01T00:00:00\nSI_MDM_ON 2030-01-01T03:00:00\n").records
    intervals, _ = pair_intervals(records)
    dpu = next(i for i in intervals if i.instrument == "DPU")
    assert dpu.stop == t("2030-01-01T03:00:00") and dpu.open_ended


def test_unmatched_off_is_ignored():
    intervals, points = pair_intervals(parse("SI_DPU_OFF 2030-01-01T00:00:00\n").records)
    assert intervals == [] and points == []


def test_obs_modes_hold_until_next_switch_or_instrument_off():
    evt = parse(
        "SI_MPPE_ENA_ON 2030-01-01T00:00:00\n"
        "SI_MPPE_ENA_OBS_MODE_Periapsis 2030-01-01T00:10:00\n"
        "SI_MPPE_ENA_OBS_MODE_LowReso 2030-01-01T00:40:00\n"
        "SI_MPPE_ENA_OFF 2030-01-01T01:00:00\n"
    )
    intervals, points = pair_intervals(evt.records)
    assert obs_mode_intervals(points, intervals) == [
        Interval("MPPE_ENA", "OBS:Periapsis", t("2030-01-01T00:10:00"), t("2030-01-01T00:40:00")),
        Interval("MPPE_ENA", "OBS:LowReso", t("2030-01-01T00:40:00"), t("2030-01-01T01:00:00")),
    ]


def test_obs_mode_outside_power_window_is_dropped():
    evt = parse("SI_MPPE_ENA_OBS_MODE_Periapsis 2030-01-01T00:10:00\n")
    intervals, points = pair_intervals(evt.records)
    assert obs_mode_intervals(points, intervals) == []
