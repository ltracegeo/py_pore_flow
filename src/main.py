import cProfile
import datetime
import json
from pathlib import Path
import re
import time

import openpnm
from PySide2 import QtWidgets

import py_pore_flow as ppf
from py_pore_flow.graphic.manual_simulation import ManualSimulationPlot
from py_pore_flow.graphic.plot_krel import plot_krel


def print_props():
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", "a")
    print(pn.props())


def run(sample_directory, manual, generate_vtu, debug, enable_profiling):
    if enable_profiling:
        prof = cProfile.Profile()
        prof.run(f'run_simulation("{sample_directory}", {{}}, {manual}, {generate_vtu}, {debug}, False)')
        prof.dump_stats("output.prof")
    else:
        run_simulation(sample_directory, {}, manual, generate_vtu, debug)


def run_simulation(directory_path, parameters={}, manual=False, generate_vtu=False, debug=False, show_final_plot=True):
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
        automatic_simulation(pn, parameters, generate_vtu, debug, show_final_plot)


def manual_simulation(pn, parameters):
    simulation = ManualSimulationPlot(pn, parameters)

    app = QtWidgets.QApplication([])
    simulation.show()
    app.exec_()


def automatic_simulation(pn, parameters={}, generate_vtu=False, debug=False, show_final_plot=True):
    start = time.time()

    ppf.log.configure(output=ppf.log.STDOUT, level=ppf.log.DEBUG if debug else ppf.log.INFO)

    cycle, Pc, Sw, Krw, Kro = ppf.run_two_phase_simulation(pn, parameters_dict=parameters, generate_vtu=generate_vtu)

    time_elapsed = datetime.timedelta(seconds=int(time.time() - start))

    print(f"Time elapsed: {time_elapsed}")

    if show_final_plot:
        plot_krel(cycle, Sw, Krw, Kro)


def snapshot(snapshot_file):
    with open(snapshot_file, "r") as fp:
        snapshot_dict = json.load(fp)

    pn = openpnm.network.Network()
    pn.update(snapshot_dict["network"])
    parameters = snapshot_dict["parameters"]

    automatic_simulation(pn, parameters, False, False, True)

if __name__ == "__main__":
    # sample_directory = "tests/resources/bug_conductance"
    # manual = True
    # generate_vtu = False
    # debug = False
    # enable_profiling = False

    # run(sample_directory, manual, generate_vtu, debug, enable_profiling)
    snapshot_file = "snapshot.json"
    snapshot(snapshot_file)
