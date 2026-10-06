"""Side panel to quickly hide instruments and modes, and switch summary/detailed view."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .timeline import ViewState


class CheckList(QWidget):
    """A titled list of checkable names with All / None buttons."""

    changed = Signal()

    def __init__(self, title: str, names: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.list = QListWidget()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list.addItem(item)
        self.all_button = QPushButton("All")
        self.none_button = QPushButton("None")
        self.all_button.clicked.connect(lambda: self._set_all(Qt.Checked))
        self.none_button.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        self.list.itemChanged.connect(self.changed)
        header = QHBoxLayout()
        header.addWidget(QLabel(title))
        header.addStretch()
        header.addWidget(self.all_button)
        header.addWidget(self.none_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.list)

    def _items(self) -> list[QListWidgetItem]:
        return [self.list.item(row) for row in range(self.list.count())]

    def _set_all(self, state: Qt.CheckState) -> None:
        self.list.blockSignals(True)
        for item in self._items():
            item.setCheckState(state)
        self.list.blockSignals(False)
        self.changed.emit()

    def unchecked(self) -> frozenset[str]:
        return frozenset(item.text() for item in self._items() if item.checkState() != Qt.Checked)


class FilterPanel(QWidget):
    changed = Signal(object)  # ViewState

    def __init__(self, instruments: list[str], modes: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Tohban plan")
        self.summary = QCheckBox("Summary: highest mode per instrument")
        self.summary.setChecked(True)
        self.instruments = CheckList("Instruments", instruments)
        self.modes = CheckList("Modes", modes)
        for source in (self.summary.toggled, self.instruments.changed, self.modes.changed):
            source.connect(lambda *_: self.changed.emit(self.state()))
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.instruments, stretch=3)
        layout.addWidget(self.modes, stretch=2)

    def state(self) -> ViewState:
        return ViewState(
            summary=self.summary.isChecked(),
            hidden_instruments=self.instruments.unchecked(),
            hidden_modes=self.modes.unchecked(),
        )
