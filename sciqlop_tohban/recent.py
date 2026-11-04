"""Recently opened plans, remembered across SciQLop sessions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QMenu, QWidget

KEY = "sciqlop_tohban/recent_plans"
LIMIT = 8


def remember(recent: list[str], path: str, limit: int = LIMIT) -> list[str]:
    return [path, *(p for p in recent if p != path)][:limit]


class RecentPlans(QObject):
    """A "recent plans" menu backed by QSettings; plans that vanished are hidden, not forgotten."""

    chosen = Signal(object)  # Path

    def __init__(self, settings: QSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self.menu = QMenu("Open recent Tohban plan", parent)
        # Rebuilt only as it opens: rebuilding from an entry's own click would delete that entry mid-signal.
        self.menu.aboutToShow.connect(self._fill)

    def paths(self) -> list[str]:
        value = self._settings.value(KEY, [])
        # QSettings hands back a one-item list as a plain string.
        return [value] if isinstance(value, str) else list(value or [])

    def add(self, path: Path) -> None:
        self._store(remember(self.paths(), str(path)))

    def _store(self, paths: list[str]) -> None:
        self._settings.setValue(KEY, paths)
        self._settings.sync()

    def _fill(self) -> None:
        self.menu.clear()
        existing = [p for p in self.paths() if Path(p).is_file()]
        for path in existing:
            # Plans often share a file name: the folder tells them apart.
            action = self.menu.addAction(f"{Path(path).name}  ({Path(path).parent})")
            action.setData(path)
            action.triggered.connect(lambda _=False, path=path: self.chosen.emit(Path(path)))
        if not existing:
            self.menu.addAction("No recent plans").setEnabled(False)
            return
        self.menu.addSeparator()
        self.menu.addAction("Clear recent plans").triggered.connect(lambda: self._store([]))
