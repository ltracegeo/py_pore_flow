import argparse
import cProfile
import datetime
import json
from pathlib import Path
import re
import time
import os

import numpy as np
import openpnm
from PySide2 import QtWidgets

import py_pore_flow as ppf
from py_pore_flow.graphic.manual_simulation import ManualSimulationPlot
from py_pore_flow.graphic.plot_krel import plot_krel
from py_pore_flow.simulation.simulation import StepSimulation


def print_props():
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", "a")
    print(pn.props())


def load_multiscale_values_from_dir(directory, network_name, pn):
    link3_path = Path(directory) / f"{network_name}_link3.dat"
    node3_path = Path(directory) / f"{network_name}_node3.dat"

    if not link3_path.exists() or not node3_path.exists():
        return None, None

    with link3_path.open("r") as f:
        link3_content = f.read()
    with node3_path.open("r") as f:
        node3_content = f.read()

    num_pores = len(pn["pore.all"])
    num_throats = len(pn["throat.all"])
    pore_N = np.full(num_pores, 1.0, dtype=np.float64)
    throat_N = np.full(num_throats, 1.0, dtype=np.float64)
    for line in node3_content.strip().splitlines():
        pid_str, N_str = line.split()[:2]
        pid = int(pid_str)
        pore_N[pid - 1] = float(N_str)
    for line in link3_content.strip().splitlines():
        parts = line.split()
        tid = int(parts[0])
        if tid <= num_throats:
            throat_N[tid - 1] = float(parts[3])

    return pore_N, throat_N


def run(
    sample_directory,
    manual,
    generate_vtu,
    debug,
    show_final_plot,
    enable_profiling,
    drainage_snapshot_pc,
    imbibition_snapshot_pc,
    second_drainage_snapshot_pc,
):
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
            show_final_plot=show_final_plot,
            drainage_snapshot_pc=drainage_snapshot_pc,
            imbibition_snapshot_pc=imbibition_snapshot_pc,
            second_drainage_snapshot_pc=second_drainage_snapshot_pc,
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
    second_drainage_snapshot_pc=None,
):
    network_name = ""
    my_path = Path(directory_path)
    pattern = re.compile(r"(\S+)_node1.dat")
    for filepath in my_path.iterdir():
        re_match = pattern.match(filepath.name)
        if re_match:
            network_name = re_match[1]

    pn = openpnm.io.network_from_statoil(directory_path, network_name)

    pore_N, throat_N = load_multiscale_values_from_dir(directory_path, network_name, pn)
    if pore_N is not None and throat_N is not None:
        pn["pore.N"], pn["throat.N"] = pore_N, throat_N

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
            second_drainage_snapshot_pc,
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
    second_drainage_snapshot_pc=None,
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
        second_drainage_snapshot_pc=second_drainage_snapshot_pc,
        config=config,
    )

    time_elapsed = datetime.timedelta(seconds=int(time.time() - start))

    print(f"Time elapsed: {time_elapsed}")

    if show_final_plot:
        plot_krel(cycle, Sw, Krw, Kro)


def snapshot(snapshot_file, debug):
    pn, parameters, config = StepSimulation.load_snapshot(snapshot_file)

    automatic_simulation(pn, parameters, False, debug, True, setup_network=False, config=config)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="py_pore_flow - Pore network two-phase simulation")
    arg_parser.add_argument(
        "--manual", action="store_true", help="Run simulation via user inputs with a user interface"
    )
    arg_parser.add_argument("--prof", action="store_true", help="Enable profiling")
    arg_parser.add_argument("--vtu", action="store_true", help="Generate vtu files for each simulation step")
    arg_parser.add_argument("--debug", action="store_true", help="Enable debug messages")
    arg_parser.add_argument("--nogui", action="store_true", help="Disable graphical window")
    arg_parser.add_argument("--snapshot", action="store_true", help="Input is a snapshot instead of a statoil")
    arg_parser.add_argument("--drainage_snapshot_pc", default=None, help="Pc to take snapshotin drainage cycle")
    arg_parser.add_argument(
        "--imbibition_snapshot_pc", type=float, default=None, help="Pc to take snapshotin imbibition cycle"
    )
    arg_parser.add_argument(
        "--second_drainage_snapshot_pc", type=float, default=None, help="Pc to take snapshotin imbibition cycle"
    )
    arg_parser.add_argument("network_path", help="Path to a directory with the network statoil files")
    args = arg_parser.parse_args()

    is_snapshot = args.snapshot
    debug = args.debug

    if is_snapshot:
        snapshot_file = args.network_path
        snapshot(snapshot_file, debug)
    else:
        manual = args.manual
        generate_vtu = args.vtu
        show_final_plot = not args.nogui
        enable_profiling = args.prof
        drainage_snapshot_pc = args.drainage_snapshot_pc
        imbibition_snapshot_pc = args.imbibition_snapshot_pc
        second_drainage_snapshot_pc = args.second_drainage_snapshot_pc

        if drainage_snapshot_pc is not None and drainage_snapshot_pc != "final":
            drainage_snapshot_pc = float(drainage_snapshot_pc)

        sample_directory = args.network_path
        run(
            sample_directory,
            manual,
            generate_vtu,
            debug,
            show_final_plot,
            enable_profiling,
            drainage_snapshot_pc,
            imbibition_snapshot_pc,
            second_drainage_snapshot_pc,
        )
