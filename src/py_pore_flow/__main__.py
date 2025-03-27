import argparse
import cProfile
import datetime
import json
from pathlib import Path
import re
import time
import os

import openpnm
from PySide2 import QtWidgets

import py_pore_flow as ppf
from py_pore_flow.graphic.manual_simulation import ManualSimulationPlot
from py_pore_flow.graphic.plot_krel import plot_krel
from py_pore_flow.simulation.simulation import StepSimulation


def print_props():
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", "a")
    print(pn.props())


def run(sample_directory, manual, generate_vtu, debug, enable_profiling, drainage_snapshot_pc, imbibition_snapshot_pc):
    sample_directory = Path(sample_directory).as_posix()
    if enable_profiling:
        prof = cProfile.Profile()
        prof.run(f'run_simulation("{sample_directory}", {{}}, {manual}, {generate_vtu}, {debug})')
        prof.dump_stats("output.prof")
    else:
        run_simulation(
            sample_directory,
            {},
            manual,
            generate_vtu,
            debug,
            drainage_snapshot_pc=drainage_snapshot_pc,
            imbibition_snapshot_pc=imbibition_snapshot_pc,
        )


def run_simulation(
    directory_path,
    parameters={},
    manual=False,
    generate_vtu=False,
    debug=False,
    show_final_plot=True,
    drainage_snapshot_pc=None,
    imbibition_snapshot_pc=None,
):
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
        automatic_simulation(
            pn,
            parameters,
            generate_vtu,
            debug,
            show_final_plot,
            True,
            drainage_snapshot_pc,
            imbibition_snapshot_pc,
        )


def manual_simulation(pn, parameters):
    simulation = ManualSimulationPlot(pn, parameters)

    app = QtWidgets.QApplication([])
    simulation.show()
    app.exec_()


def automatic_simulation(
    pn,
    parameters={},
    generate_vtu=False,
    debug=False,
    show_final_plot=True,
    setup_network=True,
    drainage_snapshot_pc=None,
    imbibition_snapshot_pc=None,
    config=None,
):
    start = time.time()

    ppf.log.configure(output=ppf.log.STDOUT, level=ppf.log.DEBUG if debug else ppf.log.INFO)

    cycle, Pc, Sw, Krw, Kro = ppf.run_two_phase_simulation(
        pn,
        parameters_dict=parameters,
        generate_vtu=generate_vtu,
        generate_cas=setup_network,
        setup_network=setup_network,
        drainage_snapshot_pc=drainage_snapshot_pc,
        imbibition_snapshot_pc=imbibition_snapshot_pc,
        config=config,
    )

    time_elapsed = datetime.timedelta(seconds=int(time.time() - start))

    print(f"Time elapsed: {time_elapsed}")

    if show_final_plot:
        plot_krel(cycle, Sw, Krw, Kro)


def snapshot(snapshot_file):
    pn, parameters, config = StepSimulation.load_snapshot(snapshot_file)

    automatic_simulation(pn, parameters, False, False, True, setup_network=False, config=config)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="py_pore_flow - Pore network two-phase simulation")
    arg_parser.add_argument(
        "--manual", action="store_true", help="Run simulation via user inputs with a user interface"
    )
    arg_parser.add_argument("--prof", action="store_true", help="Enable profiling")
    arg_parser.add_argument("--vtu", action="store_true", help="Generate vtu files for each simulation step")
    arg_parser.add_argument("--debug", action="store_true", help="Enable debug messages")
    arg_parser.add_argument("--snapshot", action="store_true", help="Input is a snapshot instead of a statoil")
    arg_parser.add_argument(
        "--drainage_snapshot_pc", type=float, default=None, help="Pc to take snapshotin drainage cycle"
    )
    arg_parser.add_argument(
        "--imbibition_snapshot_pc", type=float, default=None, help="Pc to take snapshotin imbibition cycle"
    )
    arg_parser.add_argument("network_path", help="Path to a directory with the network statoil files")
    args = arg_parser.parse_args()

    is_snapshot = args.snapshot

    if is_snapshot:
        snapshot_file = args.network_path
        snapshot(snapshot_file)
    else:
        manual = args.manual
        generate_vtu = args.vtu
        debug = args.debug
        enable_profiling = args.prof
        drainage_snapshot_pc = args.drainage_snapshot_pc
        imbibition_snapshot_pc = args.imbibition_snapshot_pc

        sample_directory = args.network_path
        run(
            sample_directory,
            manual,
            generate_vtu,
            debug,
            enable_profiling,
            drainage_snapshot_pc,
            imbibition_snapshot_pc,
        )
