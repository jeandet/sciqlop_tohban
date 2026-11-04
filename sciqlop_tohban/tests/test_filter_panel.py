import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QProxyStyle, QStyle  # noqa: E402

from sciqlop_tohban.filter_panel import FilterPanel  # noqa: E402
from sciqlop_tohban.timeline import View, ViewState  # noqa: E402


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


def test_view_choice(panel):
    panel.view.setCurrentText(View.STATE.value)
    assert panel.state().view is View.STATE


APOAPSES = np.datetime64("2030-01-01T00:00:00") + np.array([0, 10, 20, 30]) * np.timedelta64(1, "h")


def test_orbit_slider_waits_for_the_ephemeris(panel):
    assert not panel.orbits.slider.isEnabled()


def test_orbit_slider_selects_one_apoapsis_to_apoapsis_window(panel, qtbot):
    panel.orbits.set_apoapses(APOAPSES)
    assert panel.orbits.slider.maximum() == 2
    with qtbot.waitSignal(panel.orbits.selected) as signal:
        panel.orbits.slider.setValue(1)
    assert signal.args == [APOAPSES[1], APOAPSES[2]]


class TabOnlyFocusStyle(QProxyStyle):
    """Like desktop styles whose sliders only take focus from Tab, never from a click."""

    def styleHint(self, hint, option=None, widget=None, data=None):
        if hint == QStyle.SH_Button_FocusPolicy:
            return int(Qt.TabFocus.value)
        return super().styleHint(hint, option, widget, data)


@pytest.fixture
def tab_only_focus_style(qapp):
    qapp.setStyle(TabOnlyFocusStyle("Fusion"))
    yield
    qapp.setStyle("Fusion")


def test_orbit_slider_takes_focus_on_click_so_arrow_keys_step_orbits(tab_only_focus_style, qtbot):
    panel = FilterPanel(["DPU"], ["BASE"])
    qtbot.addWidget(panel)
    panel.orbits.set_apoapses(APOAPSES)
    assert panel.orbits.slider.focusPolicy() & Qt.ClickFocus
    panel.orbits.slider.setFocus()
    with qtbot.waitSignal(panel.orbits.selected) as signal:
        qtbot.keyClick(panel.orbits.slider, Qt.Key_Right)
    assert signal.args == [APOAPSES[1], APOAPSES[2]]


def test_orbit_slider_follows_the_shown_time_silently(panel, qtbot):
    panel.orbits.set_apoapses(APOAPSES)
    with qtbot.assertNotEmitted(panel.orbits.selected):
        panel.orbits.follow(APOAPSES[2] + np.timedelta64(5, "h"))
    assert panel.orbits.slider.value() == 2
    assert "3 / 3" in panel.orbits.label.text()
