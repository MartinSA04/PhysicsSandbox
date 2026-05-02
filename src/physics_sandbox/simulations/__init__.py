"""Simulation modules.

Each simulation lives in its own submodule and exposes a ``Simulation`` subclass
plus a matching control widget. New simulations should be added to ``REGISTRY``
so the GUI picks them up automatically.
"""

from collections.abc import Callable

from PySide6.QtWidgets import QWidget

from .base import Simulation
from .orbit import OrbitSimulation, OrbitWidget
from .pendulum import PendulumSimulation, PendulumWidget

SimulationFactory = Callable[[], tuple[Simulation, QWidget]]


def _make_orbit() -> tuple[Simulation, QWidget]:
    sim = OrbitSimulation()
    return sim, OrbitWidget(sim)


def _make_pendulum() -> tuple[Simulation, QWidget]:
    sim = PendulumSimulation()
    return sim, PendulumWidget(sim)


REGISTRY: dict[str, SimulationFactory] = {
    "N-body Gravity": _make_orbit,
    "Pendulum Chain": _make_pendulum,
}

__all__ = [
    "REGISTRY",
    "OrbitSimulation",
    "OrbitWidget",
    "PendulumSimulation",
    "PendulumWidget",
    "Simulation",
    "SimulationFactory",
]
