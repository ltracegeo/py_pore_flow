import datetime
import json
from pathlib import Path
import re
import time

import openpnm
from PyQt6 import QtWidgets

import py_pore_flow as ppf
from py_pore_flow.graphic.manual_simulation import ManualSimulationPlot
from py_pore_flow.graphic.plot_krel import plot_krel


def print_props():
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", "a")
    print(pn.props())


def run(directory_path, parameters={}, manual=False, generate_vtu=False, debug=False):
    network_name = ""
    my_path = Path(directory_path)
    pattern = re.compile("(\S+)_node1.dat")
    for filepath in my_path.iterdir():
        re_match = pattern.match(filepath.name)
        if re_match:
            network_name = re_match[1]

    pn = openpnm.io.network_from_statoil(directory_path, network_name)

    if not parameters:
        try:
            with open(my_path / "input.txt", "r") as f:
                parameters = json.loads(f.read())
        except FileNotFoundError:
            pass

    if manual:
        manual_simulation(pn, parameters)
    else:
        automatic_simulation(pn, parameters, generate_vtu, debug)


def manual_simulation(pn, parameters):
    simulation = ManualSimulationPlot(pn, parameters)

    app = QtWidgets.QApplication([])
    simulation.show()
    app.exec()


def automatic_simulation(pn, parameters={}, generate_vtu=False, debug=False):
    start = time.time()

    ppf.log.configure(output=ppf.log.STDOUT, level=ppf.log.DEBUG if debug else ppf.log.INFO)

    cycle, Pc, Sw, Krw, Kro = ppf.run_two_phase_simulation(
        pn, parameters_dict=parameters, generate_vtu=generate_vtu, debug=True
    )

    time_elapsed = datetime.timedelta(seconds=int(time.time() - start))

    print(f"Time elapsed: {time_elapsed}")

    plot_krel(cycle, Sw, Krw, Kro)


if __name__ == "__main__":
    sample_directory = "tests/resources/a"
    manual = False
    generate_vtu = False
    debug = False

    run(sample_directory, {}, manual, generate_vtu, debug)
