"""2D Eulerian smoke simulation (Stam's "Stable Fluids", 1999).

State is two velocity components ``u``, ``v`` and a scalar density field on
an ``(N+2, N+2)`` grid — one ring of ghost cells around the ``N×N`` interior
holds the boundary conditions. A substep does

    diffuse u, v  →  project (∇·v = 0)  →  advect u, v  →  project
    diffuse density  →  advect density

Diffusion and the pressure projection are both Poisson-like relaxations; we
use Jacobi iterations because they vectorise cleanly in NumPy. Velocity
walls are no-slip (component normal to the wall flips sign), density walls
just reflect.

Unlike the ODE-based simulations in this sandbox, the time step is fixed and
``rhs`` is unused — we override :meth:`step` directly instead of going
through ``solve_ivp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .base import Simulation

_RESOLUTIONS = (32, 48, 64, 96, 128)


@dataclass
class FluidParameters:
    N: int = 64
    dt: float = 0.1
    viscosity: float = 0.0
    diffusion: float = 0.0
    iterations: int = 16
    plume: bool = True
    plume_strength: float = 60.0
    plume_velocity: float = 4.0
    decay: float = 0.997


@dataclass
class FluidSimulation(Simulation):
    """2D smoke on a regular grid via the stable-fluids algorithm."""

    params: FluidParameters = field(default_factory=FluidParameters)

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.t = 0.0
        n = self.params.N
        shape = (n + 2, n + 2)
        self.u = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.density = np.zeros(shape, dtype=np.float64)
        # Base class wants a state vector; we don't use it.
        self.state = np.zeros(0, dtype=np.float64)

    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        # Unused — we don't drive this through solve_ivp.
        return np.zeros_like(y)

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        # Each frame advances ``params.dt`` in simulated time. The wall-clock
        # ``dt`` from the GUI (already scaled by the speed slider) tells us
        # how many fixed substeps to run this frame.
        n_sub = max(1, round(dt * 60.0))
        for _ in range(n_sub):
            self._substep()
        self.t += n_sub * self.params.dt

    def _substep(self) -> None:
        if self.params.plume:
            self._add_plume()
        self._velocity_step()
        self._density_step()

    def _add_plume(self) -> None:
        n = self.params.N
        cx = n // 2 + 1  # interior cells run 1..N
        cy = max(2, n // 16)
        radius = max(2, n // 18)
        ii, jj = np.indices(self.u.shape)
        mask = (ii - cx) ** 2 + (jj - cy) ** 2 <= radius * radius
        dt = self.params.dt
        self.density[mask] += self.params.plume_strength * dt
        self.v[mask] += self.params.plume_velocity * dt

    # ------------------------------------------------------------------
    def splash(self, i: int, j: int, vx: float = 0.0, vy: float = 0.0) -> None:
        """Drop a circular blob of ink and a velocity impulse at cell ``(i, j)``."""
        n = self.params.N
        if not (1 <= i <= n and 1 <= j <= n):
            return
        radius = max(2, n // 14)
        ii, jj = np.indices(self.u.shape)
        mask = (ii - i) ** 2 + (jj - j) ** 2 <= radius * radius
        self.density[mask] += 80.0 * self.params.dt
        self.u[mask] += vx
        self.v[mask] += vy

    def stir(self, strength: float = 6.0) -> None:
        """Inject a vortex at a random spot — fun visual stress test."""
        rng = np.random.default_rng()
        n = self.params.N
        ci = int(rng.integers(n // 4, 3 * n // 4))
        cj = int(rng.integers(n // 4, 3 * n // 4))
        ii, jj = np.indices(self.u.shape)
        di = ii - ci
        dj = jj - cj
        r2 = di * di + dj * dj
        sigma2 = (max(2, n // 10)) ** 2
        falloff = np.exp(-r2 / (2.0 * sigma2))
        # Solid-body rotation: v = ω × r, so (u, v) ∝ (-dj, di).
        self.u += -dj * falloff * strength * self.params.dt
        self.v += di * falloff * strength * self.params.dt

    # ------------------------------------------------------------------
    # Core stable-fluids primitives
    # ------------------------------------------------------------------
    def _velocity_step(self) -> None:
        nu = self.params.viscosity
        if nu > 0.0:
            self.u = self._diffuse(1, self.u, nu)
            self.v = self._diffuse(2, self.v, nu)
        self._project()
        u0 = self.u.copy()
        v0 = self.v.copy()
        self.u = self._advect(1, u0, u0, v0)
        self.v = self._advect(2, v0, u0, v0)
        self._project()

    def _density_step(self) -> None:
        if self.params.diffusion > 0.0:
            self.density = self._diffuse(0, self.density, self.params.diffusion)
        self.density = self._advect(0, self.density, self.u, self.v)
        if self.params.decay < 1.0:
            self.density *= self.params.decay

    def _diffuse(self, b: int, x0: NDArray[np.float64], diff: float) -> NDArray[np.float64]:
        n = self.params.N
        a = self.params.dt * diff * n * n
        x = x0.copy()
        denom = 1.0 + 4.0 * a
        for _ in range(self.params.iterations):
            x_new = x.copy()
            x_new[1:-1, 1:-1] = (
                x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
            ) / denom
            self._set_bnd(b, x_new)
            x = x_new
        return x

    def _advect(
        self,
        b: int,
        d0: NDArray[np.float64],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n = self.params.N
        dt0 = self.params.dt * n
        ii, jj = np.meshgrid(
            np.arange(1, n + 1, dtype=np.float64),
            np.arange(1, n + 1, dtype=np.float64),
            indexing="ij",
        )
        x = ii - dt0 * u[1:-1, 1:-1]
        y = jj - dt0 * v[1:-1, 1:-1]
        # Clip into [0.5, N+0.5] so all four bilinear neighbours are valid.
        x = np.clip(x, 0.5, n + 0.5)
        y = np.clip(y, 0.5, n + 0.5)
        i0 = np.floor(x).astype(np.int64)
        i1 = i0 + 1
        j0 = np.floor(y).astype(np.int64)
        j1 = j0 + 1
        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1
        d = np.zeros_like(d0)
        d[1:-1, 1:-1] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (
            t0 * d0[i1, j0] + t1 * d0[i1, j1]
        )
        self._set_bnd(b, d)
        return d

    def _project(self) -> None:
        n = self.params.N
        h = 1.0 / n
        u = self.u
        v = self.v
        div = np.zeros_like(u)
        div[1:-1, 1:-1] = -0.5 * h * (u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2])
        self._set_bnd(0, div)
        p = np.zeros_like(u)
        for _ in range(self.params.iterations):
            p_new = p.copy()
            p_new[1:-1, 1:-1] = (
                div[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
            ) / 4.0
            self._set_bnd(0, p_new)
            p = p_new
        u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        self._set_bnd(1, u)
        self._set_bnd(2, v)

    def _set_bnd(self, b: int, x: NDArray[np.float64]) -> None:
        # b == 1 → x-velocity (flip across left/right walls).
        # b == 2 → y-velocity (flip across bottom/top walls).
        # b == 0 → scalar (mirror).
        if b == 1:
            x[0, 1:-1] = -x[1, 1:-1]
            x[-1, 1:-1] = -x[-2, 1:-1]
        else:
            x[0, 1:-1] = x[1, 1:-1]
            x[-1, 1:-1] = x[-2, 1:-1]
        if b == 2:
            x[1:-1, 0] = -x[1:-1, 1]
            x[1:-1, -1] = -x[1:-1, -2]
        else:
            x[1:-1, 0] = x[1:-1, 1]
            x[1:-1, -1] = x[1:-1, -2]
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class FluidWidget(QWidget):
    """ImageItem viewer + parameter panel for the stable-fluids solver."""

    def __init__(self, sim: FluidSimulation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sim = sim

        # --- world view -----------------------------------------------
        self._world = pg.PlotWidget()
        self._world.setAspectLocked(True)
        self._world.setBackground("#101418")
        self._world.hideAxis("bottom")
        self._world.hideAxis("left")
        self._world.setMouseEnabled(x=False, y=False)
        self._world.setMenuEnabled(False)

        self._image = pg.ImageItem()
        cmap = pg.colormap.get("inferno")
        if cmap is not None:
            self._image.setColorMap(cmap)
        self._world.addItem(self._image)

        # --- controls --------------------------------------------------
        self._res_combo = QComboBox()
        for res in _RESOLUTIONS:
            self._res_combo.addItem(f"{res}×{res}", res)
        self._res_combo.setCurrentIndex(_RESOLUTIONS.index(sim.params.N))
        self._res_combo.currentIndexChanged.connect(self._on_resolution_changed)

        self._visc_spin = self._spin(0.0, 1e-2, 1e-5, sim.params.viscosity, decimals=6)
        self._visc_spin.valueChanged.connect(self._on_viscosity_changed)
        self._diff_spin = self._spin(0.0, 1e-2, 1e-5, sim.params.diffusion, decimals=6)
        self._diff_spin.valueChanged.connect(self._on_diffusion_changed)
        self._decay_spin = self._spin(0.90, 1.0, 0.001, sim.params.decay, decimals=4)
        self._decay_spin.valueChanged.connect(self._on_decay_changed)
        self._iter_spin = self._spin(4, 40, 1, float(sim.params.iterations), decimals=0)
        self._iter_spin.valueChanged.connect(self._on_iter_changed)

        self._plume_check = QCheckBox("Bottom plume")
        self._plume_check.setChecked(sim.params.plume)
        self._plume_check.toggled.connect(self._on_plume_toggled)
        self._plume_strength = self._spin(0.0, 200.0, 5.0, sim.params.plume_strength, decimals=1)
        self._plume_strength.valueChanged.connect(self._on_plume_strength_changed)
        self._plume_velocity = self._spin(0.0, 50.0, 0.5, sim.params.plume_velocity, decimals=2)
        self._plume_velocity.valueChanged.connect(self._on_plume_velocity_changed)

        params_box = QGroupBox("Solver")
        params_form = QFormLayout(params_box)
        params_form.addRow("Resolution", self._res_combo)
        params_form.addRow("Viscosity", self._visc_spin)
        params_form.addRow("Diffusion", self._diff_spin)
        params_form.addRow("Decay", self._decay_spin)
        params_form.addRow("Iterations", self._iter_spin)

        plume_box = QGroupBox("Plume")
        plume_form = QFormLayout(plume_box)
        plume_form.addRow(self._plume_check)
        plume_form.addRow("Density / s", self._plume_strength)
        plume_form.addRow("Velocity / s", self._plume_velocity)

        splash_btn = QPushButton("Splash random")
        splash_btn.clicked.connect(self._splash_random)
        stir_btn = QPushButton("Stir vortex")
        stir_btn.clicked.connect(lambda: self.sim.stir())
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)

        button_row = QHBoxLayout()
        button_row.addWidget(splash_btn)
        button_row.addWidget(stir_btn)

        self._readout = QLabel("")
        self._readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        side = QVBoxLayout()
        side.addWidget(params_box)
        side.addWidget(plume_box)
        side.addLayout(button_row)
        side.addWidget(reset_btn)
        side.addStretch(1)
        side.addWidget(self._readout)

        layout = QHBoxLayout(self)
        layout.addWidget(self._world, stretch=3)
        side_container = QWidget()
        side_container.setLayout(side)
        side_container.setMinimumWidth(300)
        side_container.setMaximumWidth(360)
        layout.addWidget(side_container, stretch=1)

        self._fit_view()
        self.refresh()

    # ------------------------------------------------------------------
    @staticmethod
    def _spin(
        minimum: float, maximum: float, step: float, value: float, *, decimals: int = 4
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def _fit_view(self) -> None:
        n = self.sim.params.N
        view = self._world.getViewBox()
        if view is not None:
            view.setRange(xRange=(0, n), yRange=(0, n), padding=0)

    # ------------------------------------------------------------------
    # Parameter handlers
    # ------------------------------------------------------------------
    def _on_resolution_changed(self, index: int) -> None:
        n = int(self._res_combo.itemData(index))
        self.sim.params.N = n
        self.sim.reset()
        self._fit_view()
        self.refresh()

    def _on_viscosity_changed(self, value: float) -> None:
        self.sim.params.viscosity = value

    def _on_diffusion_changed(self, value: float) -> None:
        self.sim.params.diffusion = value

    def _on_decay_changed(self, value: float) -> None:
        self.sim.params.decay = value

    def _on_iter_changed(self, value: float) -> None:
        self.sim.params.iterations = max(1, round(value))

    def _on_plume_toggled(self, on: bool) -> None:
        self.sim.params.plume = on

    def _on_plume_strength_changed(self, value: float) -> None:
        self.sim.params.plume_strength = value

    def _on_plume_velocity_changed(self, value: float) -> None:
        self.sim.params.plume_velocity = value

    def _splash_random(self) -> None:
        rng = np.random.default_rng()
        n = self.sim.params.N
        i = int(rng.integers(n // 6, 5 * n // 6))
        j = int(rng.integers(n // 6, 5 * n // 6))
        vx = float(rng.normal(0.0, 1.5))
        vy = float(rng.normal(0.0, 1.5))
        self.sim.splash(i, j, vx, vy)

    def _reset(self) -> None:
        self.sim.reset()
        self.refresh()

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------
    def advance(self, dt: float) -> None:
        self.sim.step(dt)
        self.refresh()

    def refresh(self) -> None:
        d = self.sim.density[1:-1, 1:-1]
        # Auto-scale levels with a floor so an empty box stays dark and a
        # building plume doesn't suddenly get clipped.
        d_max = float(d.max()) if d.size else 0.0
        upper = max(d_max, 0.5)
        self._image.setImage(d, levels=(0.0, upper), autoLevels=False)

        speed = float(np.hypot(self.sim.u, self.sim.v).max()) if d.size else 0.0
        self._readout.setText(
            f"t       = {self.sim.t:8.3f}\n"
            f"grid    = {self.sim.params.N}×{self.sim.params.N}\n"
            f"max ρ   = {d_max:7.3f}\n"
            f"max |v| = {speed:7.3f}"
        )
