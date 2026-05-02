"""Chain (compound) pendulum with arbitrary number of links.

Each link is an idealised rigid rod of length ``l_i`` carrying a point-mass
``m_i`` at its end. Generalised coordinates are the angles ``θ_i`` of each rod
from vertical down. The equations of motion follow from the Lagrangian; with
``β_{ij} = Σ_{k ≥ max(i,j)} m_k`` we get

    M_{ij}(θ) = β_{ij} · l_i · l_j · cos(θ_i − θ_j)
    b_p(θ, θ̇) = −Σ_j β_{pj} l_p l_j sin(θ_p − θ_j) θ̇_j²
                − g · β_{pp} · l_p · sin(θ_p)
                − c_p · θ̇_p          # linear damping per link

and ``M(θ) · θ̈ = b``. We solve that linear system at every right-hand-side
evaluation and let ``solve_ivp`` (RK45, variable step) handle the rest. For
N=2 this reduces to the standard double pendulum, including the chaotic
regime when energies are large.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .base import IntegratorOptions, Simulation

_BOB_COLORS = (
    "#ffd166",
    "#4aa3ff",
    "#ef476f",
    "#06d6a0",
    "#c084fc",
    "#fb923c",
    "#22d3ee",
)

MAX_LINKS = 6


@dataclass
class PendulumLink:
    length: float = 1.0
    mass: float = 1.0
    damping: float = 0.0
    theta0: float = math.radians(120.0)
    omega0: float = 0.0


def _default_double_chaotic() -> list[PendulumLink]:
    return [
        PendulumLink(length=1.0, mass=1.0, damping=0.0, theta0=math.radians(120.0)),
        PendulumLink(length=1.0, mass=1.0, damping=0.0, theta0=math.radians(-10.0)),
    ]


@dataclass
class ChainPendulumParameters:
    g: float = 9.81
    links: list[PendulumLink] = field(default_factory=_default_double_chaotic)


@dataclass
class PendulumSimulation(Simulation):
    """Chain pendulum: N point-mass bobs connected by rigid massless rods."""

    params: ChainPendulumParameters = field(default_factory=ChainPendulumParameters)
    integrator: IntegratorOptions = field(
        default_factory=lambda: IntegratorOptions(method="RK45", rtol=1e-8, atol=1e-10)
    )

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.params.links)

    def reset(self) -> None:
        self.t = 0.0
        n = self.n
        if n == 0:
            self.state = np.zeros(0, dtype=np.float64)
            return
        thetas = np.array([link.theta0 for link in self.params.links], dtype=np.float64)
        omegas = np.array([link.omega0 for link in self.params.links], dtype=np.float64)
        self.state = np.concatenate([thetas, omegas])

    def _arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        masses = np.array([link.mass for link in self.params.links], dtype=np.float64)
        lengths = np.array([link.length for link in self.params.links], dtype=np.float64)
        damping = np.array([link.damping for link in self.params.links], dtype=np.float64)
        return masses, lengths, damping

    def _mass_matrix_and_bias(
        self,
        theta: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        masses, lengths, damping = self._arrays()
        # mass_above[i] = sum_{k >= i} m_k
        mass_above = np.cumsum(masses[::-1])[::-1]
        idx = np.arange(self.n)
        beta = mass_above[np.maximum(idx[:, None], idx[None, :])]

        dtheta = theta[:, None] - theta[None, :]
        outer_l = np.outer(lengths, lengths)

        M = beta * outer_l * np.cos(dtheta)
        coriolis = np.sum(beta * outer_l * np.sin(dtheta) * omega[None, :] ** 2, axis=1)
        gravity = self.params.g * mass_above * lengths * np.sin(theta)
        b = -coriolis - gravity - damping * omega
        return M, b

    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        n = self.n
        theta = y[:n]
        omega = y[n:]
        M, b = self._mass_matrix_and_bias(theta, omega)
        alpha = np.linalg.solve(M, b)
        return np.concatenate([omega, alpha])

    # ------------------------------------------------------------------
    @property
    def thetas(self) -> NDArray[np.float64]:
        return self.state[: self.n]

    @property
    def omegas(self) -> NDArray[np.float64]:
        return self.state[self.n :]

    def joint_positions(self) -> NDArray[np.float64]:
        """Return ``(N+1, 2)`` array: pivot + every bob position."""
        n = self.n
        if n == 0:
            return np.zeros((1, 2), dtype=np.float64)
        _, lengths, _ = self._arrays()
        theta = self.thetas
        x = np.cumsum(lengths * np.sin(theta))
        y = -np.cumsum(lengths * np.cos(theta))
        out = np.zeros((n + 1, 2), dtype=np.float64)
        out[1:, 0] = x
        out[1:, 1] = y
        return out

    def total_chain_length(self) -> float:
        return float(sum(link.length for link in self.params.links))

    def energy(self) -> float:
        """Mechanical energy with potential reference at the pivot."""
        n = self.n
        if n == 0:
            return 0.0
        theta = self.thetas
        omega = self.omegas
        masses, lengths, _ = self._arrays()
        mass_above = np.cumsum(masses[::-1])[::-1]
        M, _ = self._mass_matrix_and_bias(theta, omega)
        kinetic = 0.5 * float(omega @ M @ omega)
        potential = -self.params.g * float(np.sum(mass_above * lengths * np.cos(theta)))
        return kinetic + potential


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
@dataclass
class _LinkControls:
    box: QGroupBox
    length: QDoubleSpinBox
    mass: QDoubleSpinBox
    damping: QDoubleSpinBox
    theta0_deg: QDoubleSpinBox
    omega0: QDoubleSpinBox


class PendulumWidget(QWidget):
    """Chain pendulum viewer with per-link controls."""

    PHASE_TRAIL = 4000

    def __init__(self, sim: PendulumSimulation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sim = sim
        self._link_controls: list[_LinkControls] = []
        self._suppress_link_signals = False

        # World view ----------------------------------------------------
        self._world = pg.PlotWidget()
        self._world.setAspectLocked(True)
        self._world.setBackground("#101418")
        self._world.showGrid(x=True, y=True, alpha=0.2)
        self._world.setLabel("bottom", "x (m)")
        self._world.setLabel("left", "y (m)")

        self._rod = self._world.plot(pen=pg.mkPen("#cbd5e1", width=2))
        self._pivot = pg.ScatterPlotItem(size=8, brush=pg.mkBrush("#94a3b8"), pen=pg.mkPen(None))
        self._pivot.setData([0.0], [0.0])
        self._bobs = pg.ScatterPlotItem(pen=pg.mkPen(None))
        self._tip_trail: deque[tuple[float, float]] = deque(maxlen=self.PHASE_TRAIL)
        self._tip_curve = self._world.plot(pen=pg.mkPen("#ffd166", width=1.0))
        self._tip_curve.setZValue(-1)
        self._world.addItem(self._pivot)
        self._world.addItem(self._bobs)

        # Phase view (last bob) ----------------------------------------
        self._phase = pg.PlotWidget()
        self._phase.setBackground("#101418")
        self._phase.showGrid(x=True, y=True, alpha=0.2)
        self._phase.setLabel("bottom", "θ_last (rad)")
        self._phase.setLabel("left", "θ̇_last (rad/s)")
        self._phase_theta: deque[float] = deque(maxlen=self.PHASE_TRAIL)
        self._phase_omega: deque[float] = deque(maxlen=self.PHASE_TRAIL)
        self._phase_curve = self._phase.plot(pen=pg.mkPen("#4aa3ff", width=1.0))
        self._phase_marker = pg.ScatterPlotItem(
            size=8, brush=pg.mkBrush("#ffd166"), pen=pg.mkPen(None)
        )
        self._phase.addItem(self._phase_marker)

        plot_col = QVBoxLayout()
        plot_col.addWidget(self._world, stretch=2)
        plot_col.addWidget(self._phase, stretch=1)

        # Global controls ----------------------------------------------
        self._g_spin = QDoubleSpinBox()
        self._g_spin.setRange(0.0, 50.0)
        self._g_spin.setSingleStep(0.1)
        self._g_spin.setDecimals(3)
        self._g_spin.setValue(sim.params.g)
        self._g_spin.setKeyboardTracking(False)
        self._g_spin.valueChanged.connect(self._on_g_changed)

        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, MAX_LINKS)
        self._n_spin.setValue(sim.n)
        self._n_spin.valueChanged.connect(self._on_n_changed)

        global_box = QGroupBox("System")
        global_form = QFormLayout(global_box)
        global_form.addRow("g (m/s²)", self._g_spin)
        global_form.addRow("links N", self._n_spin)

        # Per-link controls inside a scroll area -----------------------
        self._links_container = QWidget()
        self._links_layout = QVBoxLayout(self._links_container)
        self._links_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._links_container)

        # Buttons -------------------------------------------------------
        reset_btn = QPushButton("Reset & apply")
        reset_btn.clicked.connect(self._reset)
        kick_btn = QPushButton("Random kick")
        kick_btn.clicked.connect(self._kick)
        clear_trails_btn = QPushButton("Clear trails")
        clear_trails_btn.clicked.connect(self._clear_trails)

        button_row = QHBoxLayout()
        button_row.addWidget(reset_btn)
        button_row.addWidget(kick_btn)

        # Presets -------------------------------------------------------
        presets_box = QGroupBox("Presets")
        presets_layout = QVBoxLayout(presets_box)
        for label, factory in (
            ("Single (45°)", _preset_single),
            ("Double (chaotic)", _default_double_chaotic),
            ("Triple", _preset_triple),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, f=factory: self._load_preset(f()))
            presets_layout.addWidget(btn)

        self._readout = QLabel("")
        self._readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        side = QVBoxLayout()
        side.addWidget(global_box)
        side.addWidget(scroll, stretch=1)
        side.addLayout(button_row)
        side.addWidget(clear_trails_btn)
        side.addWidget(presets_box)
        side.addWidget(self._readout)

        layout = QHBoxLayout(self)
        plots_container = QWidget()
        plots_container.setLayout(plot_col)
        layout.addWidget(plots_container, stretch=3)
        side_container = QWidget()
        side_container.setLayout(side)
        side_container.setMinimumWidth(340)
        side_container.setMaximumWidth(400)
        layout.addWidget(side_container, stretch=1)

        self._rebuild_link_controls()
        self._rebuild_visuals()
        self.refresh()

    # ------------------------------------------------------------------
    @staticmethod
    def _spin(minimum: float, maximum: float, step: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(4)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def _rebuild_link_controls(self) -> None:
        # Clear existing controls.
        for ctl in self._link_controls:
            ctl.box.setParent(None)
            ctl.box.deleteLater()
        self._link_controls.clear()

        for i, link in enumerate(self.sim.params.links):
            color = _BOB_COLORS[i % len(_BOB_COLORS)]
            box = QGroupBox(f"Link {i + 1}")
            box.setStyleSheet(f"QGroupBox::title {{ color: {color}; }}")
            form = QFormLayout(box)
            length = self._spin(0.05, 5.0, 0.05, link.length)
            mass = self._spin(0.01, 10.0, 0.05, link.mass)
            damping = self._spin(0.0, 5.0, 0.01, link.damping)
            theta0_deg = self._spin(-360.0, 360.0, 1.0, math.degrees(link.theta0))
            omega0 = self._spin(-20.0, 20.0, 0.1, link.omega0)
            form.addRow("length (m)", length)
            form.addRow("mass (kg)", mass)
            form.addRow("damping", damping)
            form.addRow("θ₀ (deg)", theta0_deg)
            form.addRow("θ̇₀ (rad/s)", omega0)

            controls = _LinkControls(box, length, mass, damping, theta0_deg, omega0)
            for spin in (length, mass, damping, theta0_deg, omega0):
                spin.valueChanged.connect(self._on_link_changed)
            self._link_controls.append(controls)
            self._links_layout.addWidget(box)
        self._links_layout.addStretch(1)

    def _on_link_changed(self) -> None:
        if self._suppress_link_signals:
            return
        for i, ctl in enumerate(self._link_controls):
            link = self.sim.params.links[i]
            link.length = ctl.length.value()
            link.mass = ctl.mass.value()
            link.damping = ctl.damping.value()
            link.theta0 = math.radians(ctl.theta0_deg.value())
            link.omega0 = ctl.omega0.value()

    def _on_g_changed(self, value: float) -> None:
        self.sim.params.g = value

    def _on_n_changed(self, n: int) -> None:
        current = self.sim.n
        if n == current:
            return
        if n > current:
            for _ in range(n - current):
                self.sim.params.links.append(
                    PendulumLink(
                        length=1.0,
                        mass=1.0,
                        damping=0.0,
                        theta0=math.radians(15.0 * (current + 1)),
                    )
                )
        else:
            del self.sim.params.links[n:]
        self._rebuild_link_controls()
        self._reset()

    def _load_preset(self, links: list[PendulumLink]) -> None:
        self.sim.params.links = [PendulumLink(**link.__dict__) for link in links]
        self._suppress_link_signals = True
        try:
            self._n_spin.setValue(len(self.sim.params.links))
        finally:
            self._suppress_link_signals = False
        self._rebuild_link_controls()
        self._reset()

    def _reset(self) -> None:
        self.sim.reset()
        self._rebuild_visuals()
        self._clear_trails()
        self.refresh()

    def _kick(self) -> None:
        rng = np.random.default_rng()
        n = self.sim.n
        if n == 0:
            return
        self.sim.state[n:] += rng.normal(0.0, 1.0, size=n)
        self.refresh()

    def _clear_trails(self) -> None:
        self._tip_trail.clear()
        self._phase_theta.clear()
        self._phase_omega.clear()
        self._tip_curve.setData([], [])
        self._phase_curve.setData([], [])
        self._phase_marker.setData([], [])

    # ------------------------------------------------------------------
    def _rebuild_visuals(self) -> None:
        # Resize world view to fit the chain plus a margin.
        L = max(self.sim.total_chain_length(), 0.1)
        self._world.setXRange(-1.3 * L, 1.3 * L)
        self._world.setYRange(-1.3 * L, 0.3 * L)

    # ------------------------------------------------------------------
    def advance(self, dt: float) -> None:
        self.sim.step(dt)
        joints = self.sim.joint_positions()
        if joints.shape[0] > 1:
            tip = joints[-1]
            self._tip_trail.append((float(tip[0]), float(tip[1])))
            self._phase_theta.append(float(self.sim.thetas[-1]))
            self._phase_omega.append(float(self.sim.omegas[-1]))
        self.refresh()

    def refresh(self) -> None:
        joints = self.sim.joint_positions()
        self._rod.setData(joints[:, 0].tolist(), joints[:, 1].tolist())

        if joints.shape[0] > 1:
            spots = []
            for i in range(self.sim.n):
                color = _BOB_COLORS[i % len(_BOB_COLORS)]
                size = 10.0 + 6.0 * float(np.cbrt(self.sim.params.links[i].mass))
                spots.append(
                    {
                        "pos": (float(joints[i + 1, 0]), float(joints[i + 1, 1])),
                        "size": size,
                        "brush": pg.mkBrush(color),
                        "pen": pg.mkPen(None),
                    }
                )
            self._bobs.setData(spots)
        else:
            self._bobs.setData([])

        if self._tip_trail:
            xs = [pt[0] for pt in self._tip_trail]
            ys = [pt[1] for pt in self._tip_trail]
            self._tip_curve.setData(xs, ys)
        if self._phase_theta:
            self._phase_curve.setData(list(self._phase_theta), list(self._phase_omega))
            self._phase_marker.setData([self._phase_theta[-1]], [self._phase_omega[-1]])

        if self.sim.n == 0:
            self._readout.setText("(no links)")
            return
        last_theta = float(self.sim.thetas[-1])
        last_omega = float(self.sim.omegas[-1])
        self._readout.setText(
            f"t        = {self.sim.t:8.3f}\n"
            f"N links  = {self.sim.n}\n"
            f"θ_last   = {math.degrees(last_theta):+7.2f}°\n"
            f"θ̇_last   = {last_omega:+7.3f} rad/s\n"
            f"E        = {self.sim.energy():+.4f} J"
        )


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------
def _preset_single() -> list[PendulumLink]:
    return [PendulumLink(length=1.0, mass=1.0, damping=0.05, theta0=math.radians(45.0))]


def _preset_triple() -> list[PendulumLink]:
    return [
        PendulumLink(length=0.8, mass=1.0, damping=0.0, theta0=math.radians(120.0)),
        PendulumLink(length=0.7, mass=0.7, damping=0.0, theta0=math.radians(-30.0)),
        PendulumLink(length=0.6, mass=0.5, damping=0.0, theta0=math.radians(60.0)),
    ]
