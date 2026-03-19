# Project Overview: py_pore_flow

This directory contains `py_pore_flow`, a Python-based scientific simulation project for modeling fluid flow through porous media (pore network modeling). The project utilizes OpenPNM for network modeling, Numba for performance optimization, and VTK/PyQtGraph for visualization.

It appears to be focused on simulating drainage and imbibition cycles, calculating relative permeability (Krw, Kro), saturation (Sw), and capillary pressure (Pc).

## Building and Running

### Prerequisites
The project uses standard Python packaging defined in `pyproject.toml`. You can install the dependencies using `pip`:

```bash
pip install -e .
```

To install development dependencies (like `pytest`):

```bash
pip install -e ".[dev]"
```

### Running the Application
The main entry point according to the `README.md` is:

```bash
python src/main.py
```

Alternatively, there is a dedicated script for step-by-step simulation execution:

```bash
python run_simulation.py
```

### Testing
Tests are managed using `pytest`. To run the unit test suite:

```bash
python -m pytest
```

## Directory Structure & Key Components

*   **`src/py_pore_flow/`**: The core Python package containing the application logic.
    *   **`simulation/`**: Core simulation algorithms including `drainage.py`, `imbibition.py`, `cycle_simulation.py`, and calculators for relative permeability and saturation.
    *   **`network/`**: Data structures and managers for the pore network (pores, throats, elements).
    *   **`equations/`**: Mathematical models for conductance, entry pressure, and geometry.
    *   **`graphic/`**: Visualization components using PyQtGraph and VTK.
    *   **`parameters/`**: Parameter management for the simulations.
*   **`tests/`**: Contains unit tests, graphical tests, and benchmarks for different network scales.
*   **`project/`**: Contains Geoslicer project data (e.g., `bentheimer_400`) used for development and testing.
*   **`run_simulation.py`**: A helper script to initialize a network, set up logging, load configurations, and run a simulation cycle step-by-step.
*   **`pyproject.toml`**: Project configuration, dependencies (numpy, numba, openpnm, vtk, etc.), and formatting settings (black).

## Development Conventions

*   **Code Formatting:** The project uses `black` for code formatting with a line length of 120 (as specified in `pyproject.toml`).
*   **Performance:** Performance-critical sections likely utilize `numba` for JIT compilation, given its inclusion in dependencies.
*   **Testing:** `pytest` is the standard testing framework. New features or bug fixes should be accompanied by tests in the `tests/` directory.

## Documentation

*   **UML Diagram:** A UML representation of the architecture is available at: [Google Drive Link](https://drive.google.com/file/d/1V0Hd1MgIBn0si128PcMmybxgYtiEmbtO/view?usp=drive_link)

## AI Agent Instructions

*   **Code Modification:** You may make minor refactoring on the code to improve readability, and add type hints, docstrings, and code comments.