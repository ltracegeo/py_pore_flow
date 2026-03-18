# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run simulation (as module)
python -m py_pore_flow [network_path]

# Run simulation (via entry script)
python run_simulation.py [network_path]

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/unit/equations/equations_test.py

# Run a specific test
python -m pytest tests/unit/network_test.py::TestNetwork::test_name
```

**Simulation flags:**
- `--manual` — Interactive GUI mode (PySide2)
- `--vtu` — Generate VTU output files for each step (GeoSlicer visualization)
- `--debug` — Enable INFO/DEBUG logging
- `--nogui` — Skip final krel plot window
- `--prof` — cProfile profiling (outputs `output.prof`)
- `--snapshot` — Input is a snapshot file instead of Statoil format
- `--drainage_snapshot_pc` — Capillary pressure at which to save drainage snapshot (`"final"` or float)
- `--imbibition_snapshot_pc` — Capillary pressure at which to save imbibition snapshot
- `--second_drainage_snapshot_pc` — Capillary pressure at which to save second drainage snapshot

**Code style:** Black with 120-character line length.

## Architecture

`py_pore_flow` simulates capillary-controlled two-phase (oil/water) displacement in pore networks derived from rock images. Input is in Statoil format; output is capillary pressure curves, relative permeability, and optional VTU files for 3D Slicer.

### Simulation flow

Three sequential displacement cycles orchestrated by `simulation/simulation.py`:

1. **Drainage** (`drainage.py`) — Oil invades water-saturated network; capillary pressure increases
2. **Imbibition** (`imbibition.py`) — Water reinvades; capillary pressure decreases
3. **Second Drainage** (`second_drainage.py`) — Oil re-invades with Oren-Bakke hysteresis model

Each cycle is a subclass of `CycleSimulation` (`cycle_simulation.py`, 559 lines), which delegates to specialized calculators for conductance, flow rate, saturation, krel, and pressure stepping.

### Key modules

| Module | Role |
|--------|------|
| `network/network.py` | Wraps OpenPNM network; initializes per-element arrays (phase state, geometry, contact angles, pressures, saturations, conductances) |
| `network/pn_element.py` | Pore/throat element base class |
| `network/pores_network.py` / `throats_network.py` | Pore- and throat-specific network logic |
| `equations/geometry.py` | ~21 `@njit` functions: shape classification, beta angles, water/oil layer geometry, saturation per phase state |
| `equations/entry_pressure.py` | Piston-like and snap-off capillary pressure (Young-Laplace) for each cycle type |
| `equations/conductance.py` | Single-phase and multiphase conductances (Hagen-Poiseuille, corner layers, oil layers) |
| `parameters/parameter_manager.py` | All 70+ configurable parameters with defaults (contact angle models, pressure ranges, step sizes, pore-fill algorithm) |
| `simulation/clusters.py` | Phase cluster connectivity and trapping logic (1,022 lines) |
| `simulation/subsequent_cycle.py` | Generic subsequent cycle implementation (419 lines) |
| `simulation/conductance_calculator.py` | Multiphase conductance calculations |
| `simulation/krel_calculator.py` | Relative permeability calculations |
| `simulation/saturation_calculator.py` | Phase saturation calculations |
| `simulation/flow_rate_calculator.py` | Flow rate computations |
| `simulation/pc_incrementer.py` | Capillary pressure stepping logic |
| `simulation/trap_checker.py` | Trapping detection |
| `simulation/solver.py` | Sparse linear system solver (pypardiso) |
| `output.py` | VTU timestep files with GeoSlicer animation metadata |
| `graphic/` | PySide2 manual GUI, network plotting, krel curve rendering (pyqtgraph) |

### Performance

Geometry and pressure calculations use Numba `@njit(cache=True)` for JIT compilation. Sparse linear systems are solved with `pypardiso`. The `--prof` flag enables cProfile for identifying bottlenecks.

### Test resources

`tests/resources/` contains Statoil-format pore network samples. `tests/resources/bug_conductance/` holds regression data for conductance edge cases. `tests/benchmarks/` contains singlescale and multiscale networks for performance testing. `project/bentheimer_400/` is the development reference network (Bentheimer sandstone, 400³ voxels).

## Documentation

UML diagrams: https://drive.google.com/file/d/1V0Hd1MgIBn0si128PcMmybxgYtiEmbtO/view?usp=drive_link
