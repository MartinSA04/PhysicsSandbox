# Physics Sandbox

Interactive playground for small physics simulations, with a PySide6 GUI.

## Stack

- [`uv`](https://docs.astral.sh/uv/) for environment & dependency management
- [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting
- [`ty`](https://docs.astral.sh/ty/) for type checking
- `numpy` + `scipy` for the math (`solve_ivp` with adaptive RK45)
- `PySide6` + `pyqtgraph` for the GUI

## Run

```bash
uv sync
uv run physics-sandbox
```

Hotkeys:

- `Space` — play/pause
- `→` — single step (auto-pauses)

## Layout

```text
src/physics_sandbox/
├── app.py                   # entry point
├── gui/
│   └── main_window.py       # tabbed window, frame timer, playback controls
└── simulations/
    ├── base.py              # Simulation ABC + IntegratorOptions
    ├── orbit.py             # 2D Kepler orbit (RK45 variable step) + widget
    ├── pendulum.py          # damped/driven pendulum + widget
    └── fluid.py             # 2D Eulerian smoke (stable fluids) + widget
```

Each simulation lives in its own submodule and exposes a `Simulation`
subclass plus a Qt widget. Register new simulations in
`simulations/__init__.py::REGISTRY` — the GUI picks them up automatically.

## Dev

```bash
uv run ruff check
uv run ruff format
uv run ty check
```
