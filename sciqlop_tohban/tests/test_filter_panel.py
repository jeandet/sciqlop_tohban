import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402

from sciqlop_tohban.filter_panel import FilterPanel  # noqa: E402
from sciqlop_tohban.timeline import ViewState  # noqa: E402


@pytest.fixture
def panel(qtbot):
    widget = FilterPanel(["DPU", "MDM"], ["BASE", "HKM", "Mag"])
    qtbot.addWidget(widget)
    return widget


def _item(check_list, name):
    return check_list.findItems(name, Qt.MatchExactly)[0]


def test_starts_in_summary_with_everything_shown(panel):
    assert panel.state() == ViewState()


def test_unchecking_an_instrument_hides_it_and_emits_the_new_state(panel, qtbot):
    with qtbot.waitSignal(panel.changed) as signal:
        _item(panel.instruments.list, "DPU").setCheckState(Qt.Unchecked)
    assert signal.args[0] == ViewState(hidden_instruments=frozenset({"DPU"}))


def test_none_then_all_buttons(panel):
    panel.modes.none_button.click()
    assert panel.state().hidden_modes == frozenset({"BASE", "HKM", "Mag"})
    panel.modes.all_button.click()
    assert panel.state().hidden_modes == frozenset()


def test_summary_toggle(panel):
    panel.summary.setChecked(False)
    assert panel.state().summary is False
