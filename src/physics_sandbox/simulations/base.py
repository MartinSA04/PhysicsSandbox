"""Base classes shared by every simulation.

A ``Simulation`` owns a state vector and knows how to advance it in time. We
build everything on top of ``scipy.integrate.solve_ivp`` so individual modules
just need to provide a right-hand-side function and an initial state. The GUI
drives the simulation by calling :meth:`Simulation.step` once per frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


@dataclass
class IntegratorOptions:
    """Tolerances and method for ``scipy.integrate.solve_ivp``."""

    method: str = "RK45"
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = np.inf


@dataclass
class Simulation(ABC):
    """Abstract simulation driven by a fixed-frame stepper.

    Subclasses set ``state`` to the initial condition and implement
    :meth:`rhs`. The GUI calls :meth:`step` with a wall-clock ``dt``; we use
    a variable-step adaptive integrator inside that interval to keep the
    physics accurate even when frames are coarse.
    """

    integrator: IntegratorOptions = field(default_factory=IntegratorOptions)
    t: float = 0.0
    state: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    @abstractmethod
    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Right-hand side of the ODE: returns dy/dt."""

    @abstractmethod
    def reset(self) -> None:
        """Reset to the initial state defined by the current parameters."""

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds of simulated time."""
        if dt <= 0.0:
            return
        sol = solve_ivp(
            self.rhs,
            (self.t, self.t + dt),
            self.state,
            method=self.integrator.method,
            rtol=self.integrator.rtol,
            atol=self.integrator.atol,
            max_step=self.integrator.max_step,
            dense_output=False,
            t_eval=None,
        )
        if not sol.success:
            raise RuntimeError(f"Integrator failed: {sol.message}")
        self.t = float(sol.t[-1])
        self.state = np.asarray(sol.y[:, -1], dtype=np.float64)
