"""Side panel to quickly hide instruments and modes, and pick how the plan is drawn."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .phase import orbit_containing
from .timeline import View, ViewState


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


class OrbitSlider(QWidget):
    """Steps the plan through one MIO orbit, apoapsis to apoapsis, at a time."""

    selected = Signal(object, object)  # orbit start, stop (datetime64)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.apoapses = np.empty(0, dtype="datetime64[s]")
        self.slider = QSlider(Qt.Horizontal, enabled=False)
        self.slider.setPageStep(10)
        self.label = QLabel("Orbit: waiting for SPICE…")
        self.slider.valueChanged.connect(self._select)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

    def set_apoapses(self, apo: np.ndarray) -> None:
        self.apoapses = apo
        self.slider.setEnabled(len(apo) >= 2)
        self.slider.setMaximum(max(len(apo) - 2, 0))

    def follow(self, time: np.datetime64) -> None:
        """Point at the orbit holding ``time`` without moving the plan."""
        if not self.slider.isEnabled():
            return
        self.slider.blockSignals(True)
        self.slider.setValue(orbit_containing(self.apoapses, time))
        self.slider.blockSignals(False)
        self._show_label()

    def _select(self, index: int) -> None:
        self._show_label()
        self.selected.emit(self.apoapses[index], self.apoapses[index + 1])

    def _show_label(self) -> None:
        index = self.slider.value()
        start = str(self.apoapses[index].astype("datetime64[m]")).replace("T", " ")
        self.label.setText(f"Orbit {index + 1} / {self.slider.maximum() + 1}, from apoapsis {start}")


class FilterPanel(QWidget):
    changed = Signal(object)  # ViewState

    def __init__(self, instruments: list[str], modes: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Tohban plan")
        self.orbits = OrbitSlider()
        self.view = QComboBox()
        self.view.addItems([view.value for view in View])
        self.instruments = CheckList("Instruments", instruments)
        self.modes = CheckList("Modes", modes)
        for source in (self.view.currentIndexChanged, self.instruments.changed, self.modes.changed):
            source.connect(lambda *_: self.changed.emit(self.state()))
        layout = QVBoxLayout(self)
        layout.addWidget(self.orbits)
        layout.addWidget(self.view)
        layout.addWidget(self.instruments, stretch=3)
        layout.addWidget(self.modes, stretch=2)

    def state(self) -> ViewState:
        return ViewState(
            view=View(self.view.currentText()),
            hidden_instruments=self.instruments.unchecked(),
            hidden_modes=self.modes.unchecked(),
        )
