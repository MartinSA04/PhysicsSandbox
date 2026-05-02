"""Top-level window that hosts every simulation behind a tabbed UI.

The window owns a single ``QTimer`` that drives the currently visible
simulation. Each simulation widget exposes an ``advance(dt)`` method which we
call once per tick, so adding a new simulation requires no changes here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from ..simulations import REGISTRY

FRAME_HZ = 60
FRAME_INTERVAL_MS = round(1000 / FRAME_HZ)


@runtime_checkable
class SimulationView(Protocol):
    """Contract for the widget side of a simulation module."""

    def advance(self, dt: float) -> None: ...
    def refresh(self) -> None: ...


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Physics Sandbox")
        self.resize(1200, 800)

        # Playback state must exist before tabs are added — adding the first
        # tab fires currentChanged, which reads _playing.
        self._playing = True
        self._speed = 1.0

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._views: list[SimulationView] = []
        for name, factory in REGISTRY.items():
            _, widget = factory()
            assert isinstance(widget, SimulationView), (
                f"{name} widget must implement advance/refresh"
            )
            self._views.append(widget)
            self._tabs.addTab(widget, name)

        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._build_toolbar()
        self.setStatusBar(QStatusBar(self))
        self._update_status()

    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Playback")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._play_action = QAction("Pause", self)
        self._play_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self._play_action.triggered.connect(self._toggle_play)
        toolbar.addAction(self._play_action)

        step_action = QAction("Step", self)
        step_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        step_action.triggered.connect(self._single_step)
        toolbar.addAction(step_action)

        toolbar.addSeparator()

        speed_widget = QWidget()
        speed_layout = QHBoxLayout(speed_widget)
        speed_layout.setContentsMargins(8, 0, 8, 0)
        speed_layout.addWidget(QLabel("Speed ×"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.05, 20.0)
        self._speed_spin.setSingleStep(0.1)
        self._speed_spin.setDecimals(2)
        self._speed_spin.setValue(self._speed)
        self._speed_spin.valueChanged.connect(self._set_speed)
        speed_layout.addWidget(self._speed_spin)
        toolbar.addWidget(speed_widget)

        toolbar.addSeparator()

        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self.close)
        toolbar.addWidget(quit_btn)

    # ------------------------------------------------------------------
    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self._play_action.setText("Pause" if self._playing else "Play")
        self._update_status()

    def _single_step(self) -> None:
        self._playing = False
        self._play_action.setText("Play")
        self._step_current(1.0 / FRAME_HZ * self._speed)
        self._update_status()

    def _set_speed(self, value: float) -> None:
        self._speed = value
        self._update_status()

    def _on_tab_changed(self, _: int) -> None:
        self._update_status()

    def _tick(self) -> None:
        if not self._playing:
            return
        dt = (FRAME_INTERVAL_MS / 1000.0) * self._speed
        self._step_current(dt)

    def _step_current(self, dt: float) -> None:
        index = self._tabs.currentIndex()
        if 0 <= index < len(self._views):
            try:
                self._views[index].advance(dt)
            except Exception as exc:
                self._playing = False
                self._play_action.setText("Play")
                self.statusBar().showMessage(f"Simulation error: {exc}", 5000)

    def _update_status(self) -> None:
        state = "running" if self._playing else "paused"
        name = self._tabs.tabText(self._tabs.currentIndex())
        self.statusBar().showMessage(f"{name} — {state} — speed ×{self._speed:.2f}")
