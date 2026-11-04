import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from sciqlop_tohban.recent import RecentPlans, remember  # noqa: E402


def test_remember_puts_the_latest_first_without_duplicates():
    assert remember(["a", "b", "c"], "b") == ["b", "a", "c"]


def test_remember_keeps_a_bounded_list():
    assert remember(["a", "b", "c"], "d", limit=3) == ["d", "a", "b"]


@pytest.fixture
def settings(tmp_path, qapp):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


@pytest.fixture
def plans(tmp_path):
    def make(name):
        path = tmp_path / name / "OBS_PLAN.evt"
        path.parent.mkdir()
        path.write_text("")
        return path
    return make


def _shown(recent):
    recent.menu.aboutToShow.emit()
    return recent.menu.actions()


def _entries(recent):
    return [a for a in _shown(recent) if a.data()]


def test_starts_with_a_disabled_placeholder(settings):
    recent = RecentPlans(settings)
    [placeholder] = _shown(recent)
    assert not placeholder.isEnabled()


def test_lists_opened_plans_latest_first_and_remembers_them_across_sessions(settings, plans):
    first, second = plans("week1"), plans("week2")
    recent = RecentPlans(settings)
    recent.add(first)
    recent.add(second)
    again = RecentPlans(settings)
    assert [a.data() for a in _entries(again)] == [str(second), str(first)]
    assert "week2" in _entries(again)[0].text()


def test_a_single_remembered_plan_survives_a_reload(settings, plans):
    # QSettings hands back a one-item list as a plain string.
    plan = plans("week1")
    RecentPlans(settings).add(plan)
    assert [a.data() for a in _entries(RecentPlans(settings))] == [str(plan)]


def test_choosing_an_entry_asks_to_open_it(settings, plans, qtbot):
    plan = plans("week1")
    recent = RecentPlans(settings)
    recent.add(plan)
    with qtbot.waitSignal(recent.chosen) as signal:
        _entries(recent)[0].trigger()
    assert signal.args == [Path(plan)]


def test_hides_plans_that_no_longer_exist(settings, plans):
    kept, gone = plans("week1"), plans("week2")
    recent = RecentPlans(settings)
    recent.add(kept)
    recent.add(gone)
    gone.unlink()
    assert [a.data() for a in _entries(recent)] == [str(kept)]


def test_clear_forgets_everything(settings, plans):
    recent = RecentPlans(settings)
    recent.add(plans("week1"))
    [clear] = [a for a in _shown(recent) if a.text().startswith("Clear")]
    clear.trigger()
    assert _entries(RecentPlans(settings)) == []
