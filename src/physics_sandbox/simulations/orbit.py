"""Full N-body 2D gravity simulation.

Every body feels the gravitational pull of every other body. State layout is a
flat vector ``[positions(N,2).ravel(), velocities(N,2).ravel()]`` — flat so it
plugs straight into ``scipy.integrate.solve_ivp``, but reshaped inside ``rhs``
for vectorised pairwise force evaluation. The integrator stays RK45 (variable
step) and tolerances tighten for chaotic close encounters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .base import IntegratorOptions, Simulation

# Soften the singularity for very close approaches so the integrator doesn't
# have to take vanishingly small steps.
_SOFTENING_SQ = 1e-4

_TRAIL_COLORS = (
    "#ffd166",
    "#4aa3ff",
    "#ef476f",
    "#06d6a0",
    "#c084fc",
    "#fb923c",
    "#94a3b8",
    "#f472b6",
    "#22d3ee",
    "#facc15",
)


@dataclass
class Body:
    mass: float = 1.0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class NBodyParameters:
    G: float = 1.0
    bodies: list[Body] = field(default_factory=list)
    subtract_com_velocity: bool = True


def _default_two_body() -> list[Body]:
    """Two equal masses on a circular mutual orbit (G=1, separation 2)."""
    return [
        Body(mass=1.0, x=-1.0, y=0.0, vx=0.0, vy=-0.5),
        Body(mass=1.0, x=1.0, y=0.0, vx=0.0, vy=0.5),
    ]


@dataclass
class OrbitSimulation(Simulation):
    """Newtonian N-body gravity in 2D."""

    params: NBodyParameters = field(
        default_factory=lambda: NBodyParameters(bodies=_default_two_body())
    )
    integrator: IntegratorOptions = field(
        default_factory=lambda: IntegratorOptions(method="RK45", rtol=1e-9, atol=1e-11)
    )

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.params.bodies)

    def reset(self) -> None:
        self.t = 0.0
        n = self.n
        if n == 0:
            self.state = np.zeros(0, dtype=np.float64)
            return
        positions = np.array([[b.x, b.y] for b in self.params.bodies], dtype=np.float64)
        velocities = np.array([[b.vx, b.vy] for b in self.params.bodies], dtype=np.float64)
        if self.params.subtract_com_velocity:
            masses = self._masses()
            total_mass = masses.sum()
            if total_mass > 0:
                com_v = (masses[:, None] * velocities).sum(axis=0) / total_mass
                velocities -= com_v
        self.state = np.concatenate([positions.ravel(), velocities.ravel()])

    def _masses(self) -> NDArray[np.float64]:
        return np.array([b.mass for b in self.params.bodies], dtype=np.float64)

    def positions(self) -> NDArray[np.float64]:
        n = self.n
        return self.state[: 2 * n].reshape(n, 2)

    def velocities(self) -> NDArray[np.float64]:
        n = self.n
        return self.state[2 * n :].reshape(n, 2)

    # ------------------------------------------------------------------
    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        n = self.n
        pos = y[: 2 * n].reshape(n, 2)
        vel = y[2 * n :].reshape(n, 2)
        masses = self._masses()

        # diff[i, j, :] = pos[j] - pos[i]
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r2 = np.einsum("ijk,ijk->ij", diff, diff) + _SOFTENING_SQ
        # Self-interaction: zero out the diagonal so j==i contributes nothing.
        inv_r3 = r2 ** (-1.5)
        np.fill_diagonal(inv_r3, 0.0)

        # accel[i] = G * sum_j m_j * (pos[j] - pos[i]) / r_ij^3
        weights = masses[np.newaxis, :] * inv_r3
        accel = self.params.G * np.einsum("ij,ijk->ik", weights, diff)

        return np.concatenate([vel.ravel(), accel.ravel()])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def total_energy(self) -> float:
        if self.n == 0:
            return 0.0
        masses = self._masses()
        vel = self.velocities()
        pos = self.positions()
        kinetic = 0.5 * float(np.sum(masses * np.einsum("ij,ij->i", vel, vel)))

        if self.n < 2:
            return kinetic
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff) + _SOFTENING_SQ)
        # Sum over upper triangle to avoid double counting.
        iu = np.triu_indices(self.n, k=1)
        mij = masses[:, None] * masses[None, :]
        potential = -self.params.G * float(np.sum(mij[iu] / r[iu]))
        return kinetic + potential

    def total_momentum(self) -> tuple[float, float]:
        if self.n == 0:
            return 0.0, 0.0
        masses = self._masses()
        p = (masses[:, None] * self.velocities()).sum(axis=0)
        return float(p[0]), float(p[1])

    def center_of_mass(self) -> tuple[float, float]:
        if self.n == 0:
            return 0.0, 0.0
        masses = self._masses()
        total = masses.sum()
        com = (masses[:, None] * self.positions()).sum(axis=0) / total
        return float(com[0]), float(com[1])


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
_COLUMNS = ("mass", "x", "y", "vx", "vy")


def _figure_eight() -> list[Body]:
    """Chenciner-Montgomery figure-eight (G=1, m=1)."""
    p1 = (-0.97000436, 0.24308753)
    v1 = (0.4662036850, 0.4323657300)
    return [
        Body(1.0, p1[0], p1[1], v1[0], v1[1]),
        Body(1.0, -p1[0], -p1[1], v1[0], v1[1]),
        Body(1.0, 0.0, 0.0, -2 * v1[0], -2 * v1[1]),
    ]


def _sun_two_planets() -> list[Body]:
    """Heavy central body with two test planets on circular orbits (G=1)."""
    return [
        Body(mass=1.0, x=0.0, y=0.0, vx=0.0, vy=0.0),
        Body(mass=1e-3, x=1.0, y=0.0, vx=0.0, vy=1.0),
        Body(mass=1e-3, x=-1.5, y=0.0, vx=0.0, vy=-((1.0 / 1.5) ** 0.5)),
    ]


def _random_three() -> list[Body]:
    rng = np.random.default_rng()
    bodies: list[Body] = []
    for _ in range(3):
        bodies.append(
            Body(
                mass=float(rng.uniform(0.5, 1.5)),
                x=float(rng.uniform(-1.5, 1.5)),
                y=float(rng.uniform(-1.5, 1.5)),
                vx=float(rng.normal(0.0, 0.3)),
                vy=float(rng.normal(0.0, 0.3)),
            )
        )
    return bodies


_PRESETS: dict[str, list[Body]] = {
    "2-body circular": _default_two_body(),
    "Figure-8": _figure_eight(),
    "Sun + 2 planets": _sun_two_planets(),
}


class OrbitWidget(QWidget):
    """N-body editor and viewer."""

    TRAIL_LENGTH = 3000

    def __init__(self, sim: OrbitSimulation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sim = sim
        self._trails: list[deque[tuple[float, float]]] = []
        self._trail_curves: list[pg.PlotDataItem] = []

        # --- plot ------------------------------------------------------
        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.setBackground("#101418")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("left", "y")
        self._plot.setLabel("bottom", "x")
        self._plot.enableAutoRange(True)

        self._body_scatter = pg.ScatterPlotItem(pen=pg.mkPen(None))
        self._plot.addItem(self._body_scatter)

        # --- controls --------------------------------------------------
        self._g_spin = QDoubleSpinBox()
        self._g_spin.setRange(0.0, 1000.0)
        self._g_spin.setDecimals(4)
        self._g_spin.setSingleStep(0.1)
        self._g_spin.setValue(sim.params.G)
        self._g_spin.setKeyboardTracking(False)
        self._g_spin.valueChanged.connect(self._on_g_changed)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.verticalHeader().setDefaultSectionSize(22)
        header = self._table.horizontalHeader()
        for col in range(len(_COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._suppress_cell_signal = False

        add_btn = QPushButton("Add body")
        add_btn.clicked.connect(self._add_body)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected)
        reset_btn = QPushButton("Reset & apply")
        reset_btn.clicked.connect(self._reset)
        clear_trails_btn = QPushButton("Clear trails")
        clear_trails_btn.clicked.connect(self._clear_trails)

        body_btns = QHBoxLayout()
        body_btns.addWidget(add_btn)
        body_btns.addWidget(remove_btn)

        run_btns = QHBoxLayout()
        run_btns.addWidget(reset_btn)
        run_btns.addWidget(clear_trails_btn)

        params_box = QGroupBox("Parameters")
        params_form = QFormLayout(params_box)
        params_form.addRow("G", self._g_spin)

        bodies_box = QGroupBox("Bodies")
        bodies_layout = QVBoxLayout(bodies_box)
        bodies_layout.addWidget(self._table)
        bodies_layout.addLayout(body_btns)

        presets_box = QGroupBox("Presets")
        presets_layout = QVBoxLayout(presets_box)
        for name in _PRESETS:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _checked=False, n=name: self._load_preset(n))
            presets_layout.addWidget(btn)
        random_btn = QPushButton("Random 3-body")
        random_btn.clicked.connect(self._load_random)
        presets_layout.addWidget(random_btn)

        self._readout = QLabel("")
        self._readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        side = QVBoxLayout()
        side.addWidget(params_box)
        side.addWidget(bodies_box, stretch=1)
        side.addLayout(run_btns)
        side.addWidget(presets_box)
        side.addWidget(self._readout)

        layout = QHBoxLayout(self)
        layout.addWidget(self._plot, stretch=3)
        side_container = QWidget()
        side_container.setLayout(side)
        side_container.setMinimumWidth(340)
        side_container.setMaximumWidth(400)
        layout.addWidget(side_container, stretch=1)

        self._populate_table_from_params()
        self._rebuild_visuals()
        self.refresh()

    # ------------------------------------------------------------------
    # Body table sync
    # ------------------------------------------------------------------
    def _populate_table_from_params(self) -> None:
        self._suppress_cell_signal = True
        try:
            self._table.setRowCount(len(self.sim.params.bodies))
            for row, body in enumerate(self.sim.params.bodies):
                values = (body.mass, body.x, body.y, body.vx, body.vy)
                for col, value in enumerate(values):
                    item = QTableWidgetItem(f"{value:.4f}")
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                    self._table.setItem(row, col, item)
        finally:
            self._suppress_cell_signal = False

    def _on_cell_changed(self, row: int, col: int) -> None:
        if self._suppress_cell_signal:
            return
        if row >= len(self.sim.params.bodies):
            return
        item = self._table.item(row, col)
        if item is None:
            return
        try:
            value = float(item.text())
        except ValueError:
            # Revert to current stored value on parse error.
            self._populate_table_from_params()
            return
        body = self.sim.params.bodies[row]
        attr = _COLUMNS[col]
        setattr(body, attr, value)

    def _add_body(self) -> None:
        if len(self.sim.params.bodies) >= len(_TRAIL_COLORS):
            return
        # Place a new body in a free-ish region with a small tangential kick.
        rng = np.random.default_rng()
        n = len(self.sim.params.bodies)
        angle = float(rng.uniform(0.0, 2 * np.pi)) if n else 0.0
        radius = 1.0 + 0.3 * n
        new_body = Body(
            mass=0.5,
            x=radius * float(np.cos(angle)),
            y=radius * float(np.sin(angle)),
            vx=-0.3 * float(np.sin(angle)),
            vy=0.3 * float(np.cos(angle)),
        )
        self.sim.params.bodies.append(new_body)
        self._populate_table_from_params()
        self._reset()

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.sim.params.bodies):
                del self.sim.params.bodies[row]
        self._populate_table_from_params()
        self._reset()

    # ------------------------------------------------------------------
    # Presets and reset
    # ------------------------------------------------------------------
    def _load_preset(self, name: str) -> None:
        bodies = _PRESETS[name]
        # Deep copy via dataclass __dict__ so we don't share refs with the table.
        self.sim.params.bodies = [Body(**b.__dict__) for b in bodies]
        self._populate_table_from_params()
        self._reset()

    def _load_random(self) -> None:
        self.sim.params.bodies = _random_three()
        self._populate_table_from_params()
        self._reset()

    def _on_g_changed(self, value: float) -> None:
        self.sim.params.G = value

    def _reset(self) -> None:
        self.sim.reset()
        self._rebuild_visuals()
        self._clear_trails()
        self.refresh()

    def _clear_trails(self) -> None:
        for trail in self._trails:
            trail.clear()
        for curve in self._trail_curves:
            curve.setData([], [])

    # ------------------------------------------------------------------
    # Visuals
    # ------------------------------------------------------------------
    def _rebuild_visuals(self) -> None:
        for curve in self._trail_curves:
            self._plot.removeItem(curve)
        self._trail_curves.clear()
        self._trails.clear()
        for i in range(self.sim.n):
            color = _TRAIL_COLORS[i % len(_TRAIL_COLORS)]
            curve = self._plot.plot(pen=pg.mkPen(color, width=1.2))
            # Draw bodies on top of trails.
            curve.setZValue(-1)
            self._trail_curves.append(curve)
            self._trails.append(deque(maxlen=self.TRAIL_LENGTH))

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------
    def advance(self, dt: float) -> None:
        self.sim.step(dt)
        pos = self.sim.positions()
        for i, trail in enumerate(self._trails):
            trail.append((float(pos[i, 0]), float(pos[i, 1])))
        self.refresh()

    def refresh(self) -> None:
        if self.sim.n == 0:
            self._body_scatter.setData([], [])
            self._readout.setText("(no bodies)")
            return

        pos = self.sim.positions()
        masses = self.sim._masses()
        # Marker size ~ cube root of mass so it tracks "radius".
        sizes = 8.0 + 4.0 * np.cbrt(np.clip(masses, 1e-3, None))
        spots = []
        for i in range(self.sim.n):
            color = _TRAIL_COLORS[i % len(_TRAIL_COLORS)]
            spots.append(
                {
                    "pos": (float(pos[i, 0]), float(pos[i, 1])),
                    "size": float(sizes[i]),
                    "brush": pg.mkBrush(color),
                    "pen": pg.mkPen(None),
                }
            )
        self._body_scatter.setData(spots)

        for i, trail in enumerate(self._trails):
            if not trail:
                self._trail_curves[i].setData([], [])
                continue
            xs = [pt[0] for pt in trail]
            ys = [pt[1] for pt in trail]
            self._trail_curves[i].setData(xs, ys)

        E = self.sim.total_energy()
        px, py = self.sim.total_momentum()
        cx, cy = self.sim.center_of_mass()
        self._readout.setText(
            f"t   = {self.sim.t:8.3f}\n"
            f"N   = {self.sim.n}\n"
            f"E   = {E:+.6f}\n"
            f"|P| = {(px * px + py * py) ** 0.5:.3e}\n"
            f"COM = ({cx:+.3f}, {cy:+.3f})"
        )
