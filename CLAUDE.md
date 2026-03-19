# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**py_pore_flow** is a Python pore-network two-phase flow simulator. It models drainage, imbibition, and second drainage cycles through pore networks loaded from Statoil-format files, computing capillary pressure (Pc), water saturation (Sw), and relative permeability curves (Krw, Kro).

## Commands

### Run tests
```bash
python -m pytest
```

### Run a single test
```bash
python -m pytest tests/unit/simulation/test_indexed_priority_tree.py -v
```

### Run simulation (CLI)
```bash
python -m py_pore_flow <network_path>       # automatic mode
python -m py_pore_flow --manual <network_path>  # interactive GUI mode
python -m py_pore_flow --debug <network_path>   # debug logging
python -m py_pore_flow --snapshot <snapshot_file>  # resume from snapshot
```

### Install
```bash
pip install .            # production
pip install ".[dev]"     # with dev dependencies (pytest)
```

### Docker
```bash
docker build -t py_pore_flow .
```

## Architecture

### Source layout (`src/py_pore_flow/`)

- **network/** — Pore network data model built on top of OpenPNM. `NetworkManager` wraps the raw OpenPNM network (`pn`) and provides `NetworkCore`, `PoresNetwork`, `ThroatsNetwork` accessors. `initialize_pn()` sets up default element properties (saturation, contact angles, conductivity arrays).

- **simulation/** — Core simulation engine:
  - `Simulation` (simulation.py) — Top-level orchestrator. Manages cycle sequencing (drainage → imbibition → 2nd drainage), parameter setup, and snapshot save/load. Exposes `run_next_cycle(return_generator=True)` for step-by-step execution.
  - `CycleSimulation` (cycle_simulation.py) — Abstract base for a single simulation cycle. Handles Pc updates, element invasion, trapping, cluster connectivity, and Krel computation via generators (`update_pc` yields progress).
  - `DrainageSimulation`, `ImbibitionSimulation`, `SecondDrainageSimulation` — Concrete cycle implementations inheriting from `CycleSimulation` (or `SubsequentCycleSimulation` for imbibition/2nd drainage).
  - Supporting modules: `solver.py` (linear system solve), `clusters.py` (phase connectivity), `trap_checker.py` (trapping logic), `conductance_calculator.py`, `saturation_calculator.py`, `krel_calculator.py`, `flow_rate_calculator.py`, `pc_incrementer.py`, `border_elements.py`.
  - `indexed_priority_tree.py` — Binary tree for efficient priority-queue operations during invasion.

- **equations/** — Physics equations: `geometry.py` (shape factors, half-angles, saturation), `conductance.py` (hydraulic conductance), `entry_pressure.py` (snap-off and piston-like displacement Pc).

- **parameters/** — `ParameterManager` holds simulation parameters (fluid properties, contact angles, Pc stepping) with defaults defined in `DEFAULT_PARAMETERS`.

- **graphic/** — Optional visualization (PySide2/pyqtgraph): manual simulation GUI, network plots, Krel plots.

### Key data flow

1. Network loaded from Statoil files via `openpnm.io.network_from_statoil()`
2. `initialize_pn(pn)` populates element properties on the raw OpenPNM dict
3. `Simulation` wraps network, applies `ParameterManager`, sets up geometry (shape factors, contact angles, conductance)
4. Each cycle runs as a generator yielding progress strings and a final `(Pc, Sw, Krw, Kro)` tuple
5. Results can be plotted or exported to VTU format

### Configuration

Simulation parameters are passed as a dict (see `tests/unit/simulation/resources/simulation_config.json` for the schema). Key sections:
- `parameters` — Pc stepping, saturation step targets, cycle skip flags
- `config` — runtime options (VTU generation, snapshot Pc values)

## Tech Stack

- Python 3.9, NumPy 1.23, Numba 0.60 (JIT-compiled equation kernels), OpenPNM 3.4
- Solver: pypardiso (Intel MKL sparse direct solver)
- GUI: PySide2 + pyqtgraph (optional)
- Tests: pytest 7.4 with `pythonpath = ["src"]`
- Formatter: black (line-length 120)

## Conventions

- The OpenPNM network object (`pn`) is the central data structure — a dict-like object with `"pore.*"` and `"throat.*"` property arrays.
- Element types are referenced via constants `PORE = 0`, `THROAT = 1` from `network/constants.py`.
- Fluid phases: `WATER = 0`, `OIL = 1` from `equations/geometry.py`.
- Performance-critical functions use `@numba.njit` decorators — keep these as pure numeric operations on NumPy arrays.

## Code Quality Guidelines

When modifying files, apply these improvements to the code being touched:
- Add **type hints** to modified or nearby function signatures and variables.
- Add or improve **docstrings** for modified classes and functions.
- Add clarifying **code comments** where logic is not self-evident.
- Make **minor readability refactors** (e.g., clearer variable names, small restructuring) when they improve clarity without changing behavior.
- Keep improvements scoped to the area being changed — do not do large-scale refactors unprompted.
