from pathlib import Path
import pickle
import sys
import time

import numpy as np
import openpnm

import py_pore_flow.consts as c
from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
    calculate_2nd_drainage_oil_filled_multiphase,
    calculate_2nd_drainage_oil_layer_existance_network,
    calculate_area,
    calculate_beta,
    calculate_C,
    calculate_drainage_b,
    calculate_drainage_network_b,
    calculate_imbibion_oil_filled_multiphase,
    calculate_max_b,
    calculate_oil_layer_collapse_pc_network,
    calculate_r,
    calculate_shape,
    calculate_sides,
    calculate_water_corner_collapse_pc,
    calculate_water_filled_multiphase,
)
from py_pore_flow.equations.entry_pressure import (
    calculate_2nd_drainage_piston_like_displacement_pc_network,
    calculate_2nd_drainage_snap_off_pc_network,
    calculate_2nd_drainage_snap_off_viability,
    calculate_drainage_pc,
    calculate_imbibition_snap_off_viability,
    calculate_snap_off_pc_network,
    calculate_imbibition_piston_like_displacement_pc_network,
)
from py_pore_flow.network.network import initialize_pn, NetworkManager, PoreNetwork
from py_pore_flow.output.diagnostics import DiagnosticsWriter
from py_pore_flow.output.vtu import pn_to_vtu
from py_pore_flow.parameters.parameter_manager import ParameterManager
from py_pore_flow.simulation.second_drainage import SecondDrainageSimulation
from py_pore_flow.simulation.drainage import DrainageSimulation
from py_pore_flow.simulation.imbibition import ImbibitionSimulation
from py_pore_flow.simulation.conductance_calculator import ConductanceCalculator
from py_pore_flow.simulation.pc_incrementer import PcIncrementer

INITIAL = 0
FIRST_DRAINAGE = 1
IMBIBITION = 2
SECOND_DRAINAGE = 3
CYCLE_NAME = {
    INITIAL: "Initial state",
    FIRST_DRAINAGE: "First drainage",
    IMBIBITION: "First imbibition",
    SECOND_DRAINAGE: "Second drainage",
}

def run_two_phase_simulation(
    pn,
    parameters_dict={},
    generate_vtu=False,
    generate_cas=True,
    setup_network=True,
    drainage_snapshot_pc=None,
    imbibition_snapshot_pc=None,
    second_drainage_snapshot_pc=None,
    config=None,
    progress_callback=None,
    diagnostics_output_path=None,
):
    previous_limit = sys.getrecursionlimit()

    number_of_elements = len(pn["pore.all"]) + len(pn["throat.all"])
    sys.setrecursionlimit(previous_limit + number_of_elements)

    cycle = [[], [], []]
    Pc = [[], [], []]
    Sw = [[], [], []]
    Krw = [[], [], []]
    Kro = [[], [], []]

    log.debug(f"Parameters: {parameters_dict}")
    log.debug(f"Config: {config}")

    simulation = Simulation(
        pn,
        parameters_dict,
        generate_vtu,
        generate_cas,
        setup_network,
        drainage_snapshot_pc,
        imbibition_snapshot_pc,
        second_drainage_snapshot_pc,
        config,
        progress_callback,
        diagnostics_output_path,
    )

    skip_drainage = config is not None and config["skip_drainage"]
    skip_imbibition = simulation.parameter_manager.skip_imbibition or config is not None and config["skip_imbibition"]
    skip_2nd_drainage = simulation.parameter_manager.skip_2nd_drainage

    for i, cycle_skip_setting in enumerate([
        skip_drainage,
        skip_imbibition,
        skip_2nd_drainage,
    ]):
        if not cycle_skip_setting:
            Pc[i], Sw[i], Krw[i], Kro[i] = simulation.run_next_cycle()
            cycle[i] = [i+1] * len(Pc[i])
        else:
            simulation.skip_cycle()
    simulation.end()

    sys.setrecursionlimit(previous_limit)

    return (
        np.array(cycle[0] + cycle[1] + cycle[2]),
        np.array(Pc[0] + Pc[1] + Pc[2]),
        np.array(Sw[0] + Sw[1] + Sw[2]),
        np.array(Krw[0] + Krw[1] + Krw[2]),
        np.array(Kro[0] + Kro[1] + Kro[2]),
    )


def load_snapshot(snapshot_file):
    return StepSimulation.load_snapshot(snapshot_file)


class Simulation:
    def __init__(
        self,
        pn,
        parameters_dict={},
        generate_vtu=False,
        generate_cas=True,
        setup_network=True,
        drainage_snapshot_pc=None,
        imbibition_snapshot_pc=None,
        second_drainage_snapshot_pc=None,
        config=None,
        progress_callback=None,
        diagnostics_output_path=False,
    ):
        self.pn = pn
        self.drainage_snapshot_pc = drainage_snapshot_pc
        self.imbibition_snapshot_pc = imbibition_snapshot_pc
        self.second_drainage_snapshot_pc = second_drainage_snapshot_pc
        self.progress_callback = progress_callback
        generate_cas = generate_cas
        set_initial_values = setup_network
        self.current_cycle_number = INITIAL

        self.progress_time = 0

        self.step_simulation = MultiStepSimulation(
            self.pn,
            parameters_dict,
            config,
            generate_cas=generate_cas,
            calculate_full_conductance=generate_vtu,
            set_initial_values=set_initial_values,
            diagnostics_output_path=diagnostics_output_path,
        )
        self.parameter_manager = self.step_simulation.get_parameter_manager()

        self.generate_vtu = generate_vtu

        self.pc_incrementer = PcIncrementer(
            self.parameter_manager.pc_step,
            self.parameter_manager.pc_step_min,
            self.parameter_manager.pc_step_max,
            self.parameter_manager.drainage_sw_step,
            self.parameter_manager.pc_maximum,
            self.parameter_manager.pc_minimum,
            config["pc"] if config else self.parameter_manager.pc_initial,
            enable_control=self.parameter_manager.control_by_pc,
        )
        self.step_simulation.set_pc_incrementer(self.pc_incrementer)

        self.snapshot_not_taken = True

        if generate_vtu:
            pn_to_vtu(self.pn, self.pc_incrementer.get_pc(), 1, 0)

    def skip_cycle(self):
        self.current_cycle_number += 1

    def run_next_cycle(self, return_generator=False, verbose=False):
        if self.current_cycle_number == INITIAL:
            self.step_simulation.set_max_pc(self.parameter_manager.pc_maximum)
            generator = self._get_run_cycle_generator("drainage", verbose=verbose)
        elif self.current_cycle_number == FIRST_DRAINAGE:
            generator = self._get_run_cycle_generator("imbibition", verbose=verbose)
        elif self.current_cycle_number == IMBIBITION:
            generator = self._get_run_cycle_generator("2nd_drainage", verbose=verbose)
        else:
            raise Exception("All cycles have already been run")
        self.current_cycle_number += 1

        if not return_generator:
            Pc, Sw, Krw, Kro = None, None, None, None
            for item in generator:
                if isinstance(item, tuple) and len(item) == 4 and not isinstance(item[0], str):
                    Pc, Sw, Krw, Kro = item
            return Pc, Sw, Krw, Kro
        else:
            return generator

    def end(self):
        self.step_simulation.end_simulation()

    def _get_network_state_str(self):
        pn = self.pn
        res = "Network State:\n"
        res += "  Pores:\n"
        for i in range(len(pn['pore.all'])):
            trapped = " (Trapped)" if pn['pore.trapped'][i] else ""
            res += f"    Pore {i:3d}: Entry Pc={pn['pore.entry_pressure'][i]:8.2f}, Sw={pn['pore.water_saturation'][i]:.4f}, So={pn['pore.oil_saturation'][i]:.4f}, Filled={'Oil' if pn['pore.oil_filled'][i] else 'Water'}{trapped}\n"
        res += "  Throats:\n"
        for i in range(len(pn['throat.all'])):
            trapped = " (Trapped)" if pn['throat.trapped'][i] else ""
            res += f"    Throat {i:3d}: Entry Pc={pn['throat.entry_pressure'][i]:8.2f}, Sw={pn['throat.water_saturation'][i]:.4f}, So={pn['throat.oil_saturation'][i]:.4f}, Filled={'Oil' if pn['throat.oil_filled'][i] else 'Water'}{trapped}\n"

        # Add border elements information
        if hasattr(self.step_simulation.simulation, 'border_elements'):
            be = self.step_simulation.simulation.border_elements
            res += f"  Border Pores: {be.get_pores_ids().tolist()}\n"
            res += f"  Border Throats: {be.get_throats_ids().tolist()}\n"

        return res

    def _get_run_cycle_generator(self, cycle_name, verbose=False):
        def _verbose_msg(msg):
            return msg

        def _wrap(gen):
            try:
                item = next(gen)
                while True:
                    if verbose and isinstance(item, str):
                        yield _verbose_msg(item)
                    else:
                        yield item
                    item = gen.send(None)
            except StopIteration as e:
                return e.value

        if verbose: yield _verbose_msg(f"Start {cycle_name}")
        log.info(f"Start {cycle_name}")

        if cycle_name == "drainage":
            self.pc_incrementer.set_cycle(self.pc_incrementer.DRAINAGE_CYCLE)
        elif cycle_name == "imbibition":
            self.pc_incrementer.set_cycle(self.pc_incrementer.IMBIBITION_CYCLE)
        else:
            self.pc_incrementer.set_cycle(self.pc_incrementer.SECOND_DRAINAGE_CYCLE)
        self.step_simulation.set_cycle(cycle_name)

        self.Pc_array = []
        self.Sw_array = []
        self.Krw_array = []
        self.Kro_array = []

        current_pc = self.pc_incrementer.get_pc()

        yield from _wrap(self._update_step_simulation(current_pc, first_step=True, verbose=verbose))

        simulation_end = False

        while not simulation_end:
            last_pc = None
            current_pc = self.pc_incrementer.get_pc()

            while current_pc != last_pc:
                last_pc = current_pc

                if verbose: yield _verbose_msg("Checking snapshots")
                self._check_snapshot(cycle_name, current_pc)

                current_pc = yield from _wrap(self._update_step_simulation(current_pc, verbose=verbose))

                if self.generate_vtu:
                    pn_to_vtu(self.pn, current_pc, self.Sw_array[-1], self.current_cycle_number)

                self.pc_incrementer.set_pc(current_pc)

                if verbose: yield _verbose_msg("Checking end conditions")
                if self.parameter_manager.end_at_sor and cycle_name == "imbibition" and self.Kro_array[-1] <= 0:
                    if verbose: yield _verbose_msg("End condition met: SOR reached")
                    simulation_end = True
                    break
                elif self.step_simulation.cycle_should_end():
                    if verbose: yield _verbose_msg("End condition met: Cycle should end")
                    simulation_end = True
                    break
                elif self.Sw_array[-1] <= self.parameter_manager.enforced_swi_1 and (
                    cycle_name == "drainage" or cycle_name == "2nd_drainage"
                ):
                    if verbose: yield _verbose_msg(f"End condition met: Swi reached ({self.Sw_array[-1]:.6f})")
                    simulation_end = True
                    break
                elif self.Sw_array[-1] >= self.parameter_manager.enforced_swi_2 and cycle_name == "imbibition":
                    if verbose: yield _verbose_msg(f"End condition met: Swi reached ({self.Sw_array[-1]:.6f})")
                    simulation_end = True
                    break

                yield self.Pc_array, self.Sw_array, self.Krw_array, self.Kro_array

            if self.parameter_manager.control_by_pc:
                if verbose: yield _verbose_msg("Updating PC incrementer")
                self.pc_incrementer.update_pc(self.Sw_array[-1])

        self.step_simulation.end_cycle()
        self._check_snapshot_final(cycle_name, current_pc)

        if verbose: yield f"Final Cycle State:\n{self._get_network_state_str()}"

        yield self.Pc_array, self.Sw_array, self.Krw_array, self.Kro_array

    def _update_step_simulation(self, current_pc, first_step=False, verbose=False):
        (
            current_pc,
            Sw,
            Krw,
            Kro,
            pores_filled,
            total_pores,
            percentage,
        ) = yield from self.step_simulation.update(current_pc, skip_invasions=first_step, verbose=verbose)

        self.Pc_array.append(current_pc)
        self.Sw_array.append(Sw)
        self.Krw_array.append(Krw)
        self.Kro_array.append(Kro)

        percentage_100 = int(percentage * 100)

        log.info(
            f"Pc: {current_pc:8.2f}, Sw: {Sw:.6f}, Krw: {Krw:.6f}, Kro: {Kro:.6f}, Filled: {pores_filled:4}/{total_pores} ({percentage_100:2}%)."
        )

        if verbose:
            yield f"Network State at Pc={current_pc:.2f}:\n{self._get_network_state_str()}"

        if self.progress_callback is not None:
            if (time.time() - self.progress_time) >= 60:
                self.progress_time = time.time()
                self.progress_callback(self.current_cycle_number, percentage)

        return current_pc

    def _check_snapshot(self, cycle_name, current_pc):
        if (
            (
                isinstance(self.drainage_snapshot_pc, int)
                or isinstance(self.drainage_snapshot_pc, float)
                and self.snapshot_not_taken
                and cycle_name == "drainage"
                and current_pc >= self.drainage_snapshot_pc
            )
            or (
                self.imbibition_snapshot_pc is not None
                and self.snapshot_not_taken
                and cycle_name == "imbibition"
                and current_pc <= self.imbibition_snapshot_pc
            )
            or (
                self.second_drainage_snapshot_pc is not None
                and self.snapshot_not_taken
                and cycle_name == "2nd_drainage"
                and current_pc >= self.second_drainage_snapshot_pc
            )
        ):
            snapshot_dict = self.step_simulation.get_snapshot(current_pc)
            self._save_snapshot(snapshot_dict)

    def _check_snapshot_final(self, cycle_name, current_pc):
        if (
            isinstance(self.drainage_snapshot_pc, str)
            and self.drainage_snapshot_pc == "final"
            and self.snapshot_not_taken
            and cycle_name == "drainage"
        ):
            snapshot_dict = self.step_simulation.get_snapshot(current_pc)
            snapshot_dict["cycle"] = "imbibition"
            self._save_snapshot(snapshot_dict)

    def _save_snapshot(self, snapshot_dict):
        with open("snapshot.bin", "wb") as fp:
            pickle.dump(snapshot_dict, fp)
        self.snapshot_not_taken = False
        log.info("Snapshot created")


def calculate_initial_values(network):
    pore = network.pore
    for pore_id in network.get_pores_ids():
        calculate_element_initial_values(pore, pore_id)

    throat = network.throat
    for throat_id in network.get_throats_ids():
        calculate_element_initial_values(throat, throat_id)

    return network


def calculate_drainage_values(network, max_pc):
    pore = network.pore
    for pore_id in network.get_pores_ids():
        calculate_element_drainage_values(pore, pore_id, max_pc)

    throat = network.throat
    for throat_id in network.get_throats_ids():
        calculate_element_drainage_values(throat, throat_id, max_pc)

    conductance_calculator = ConductanceCalculator(network)
    conductance_calculator.calculate_singlephase_conductance(OIL)
    conductance_calculator.calculate_singlephase_conductance(WATER)

    network.pn_core["pore.trapped"] = False
    network.pn_core["throat.trapped"] = False
    network.pn_core["pore.trapped_evacuation"] = False
    network.pn_core["throat.trapped_evacuation"] = False
    network.pn_core.pore.hinging_contact_angle[:, :] = network.pn_core.pore.receding_contact_angle[:, np.newaxis]
    network.pn_core.throat.hinging_contact_angle[:, :] = network.pn_core.throat.receding_contact_angle[:, np.newaxis]
    network.pn_core.pore.filled[:] = False
    network.pn_core.throat.filled[:] = False

    return network


def calculate_drainage_pinned_b(network, max_pc):
    calculate_drainage_network_b(
        network.pn_core.pore.b,
        network.pn_core.pore.corner_half_angle,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.n_corners,
        network.pn_core.pore.max_b,
        max_pc,
        c.sigma,
    )
    calculate_drainage_network_b(
        network.pn_core.throat.b,
        network.pn_core.throat.corner_half_angle,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.n_corners,
        network.pn_core.throat.max_b,
        max_pc,
        c.sigma,
    )


def calculate_imbibition_values(network, max_pc):
    pore = network.pore
    for pore_id in network.get_pores_ids():
        calculate_element_imbibition_values(pore, pore_id, max_pc)

    throat = network.throat
    for throat_id in network.get_throats_ids():
        calculate_element_imbibition_values(throat, throat_id, max_pc)

    conductance_calculator = ConductanceCalculator(network)
    conductance_calculator.calculate_singlephase_conductance(OIL)
    conductance_calculator.calculate_singlephase_conductance(WATER)

    network.pn_core["pore.trapped"] = False
    network.pn_core["throat.trapped"] = False
    network.pn_core["pore.trapped_evacuation"] = False
    network.pn_core["throat.trapped_evacuation"] = False
    network.pn_core.pore.hinging_contact_angle_o[:, :] = network.pn_core.pore.advancing_contact_angle[:, np.newaxis]
    network.pn_core.throat.hinging_contact_angle_o[:, :] = network.pn_core.throat.advancing_contact_angle[:, np.newaxis]
    network.pn_core.pore.filled[:] = False
    network.pn_core.throat.filled[:] = False

    calculate_imbibion_oil_filled_multiphase(
        network.pn_core.pore.oil_filled_multiphase,
        network.pn_core.pore.n_corners,
        network.pn_core.pore.b,
    )
    calculate_imbibion_oil_filled_multiphase(
        network.pn_core.throat.oil_filled_multiphase,
        network.pn_core.throat.n_corners,
        network.pn_core.throat.b,
    )

    calculate_snap_off_values(network, max_pc)
    calculate_piston_like_values(network, max_pc)
    calculate_oil_layer_collapse_values(network, max_pc)

    return network


def calculate_2nd_drainage_values(network, min_pc):
    calculate_2nd_drainage_snap_off_viability(
        network.pn_core.pore.snap_off_viability,
        network.pn_core.pore.corner_oil_layer,
        network.pn_core.pore.n_conns,
    )
    calculate_2nd_drainage_snap_off_viability(
        network.pn_core.throat.snap_off_viability,
        network.pn_core.throat.corner_oil_layer,
        network.pn_core.throat.n_conns,
    )

    conductance_calculator = ConductanceCalculator(network)
    conductance_calculator.calculate_singlephase_conductance(OIL)
    conductance_calculator.calculate_singlephase_conductance(WATER)

    network.pn_core["pore.trapped"] = False
    network.pn_core["throat.trapped"] = False
    network.pn_core["pore.trapped_evacuation"] = False
    network.pn_core["throat.trapped_evacuation"] = False
    network.pn_core.pore.filled[:] = False
    network.pn_core.throat.filled[:] = False

    network.pn_core.pore.b_pinned[:] = network.pn_core.pore.b
    calculate_water_corner_collapse_pc(
        network.pn_core.pore.water_corner_collapse_pressure,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.n_corners,
        network.pn_core.pore.b,
        network.pn_core.pore.corner_half_angle,
        c.sigma,
    )
    calculate_2nd_drainage_oil_layer_existance_network(
        network.pn_core.pore.corner_oil_layer, network.pn_core.pore.bo, network.pn_core.pore.n_corners
    )
    calculate_water_filled_multiphase(
        network.pn_core.pore.water_filled_multiphase,
        network.pn_core.pore.n_corners,
        network.pn_core.pore.corner_oil_layer,
        network.pn_core.pore.bo,
    )
    calculate_2nd_drainage_oil_filled_multiphase(
        network.pn_core.pore.oil_filled_multiphase,
        network.pn_core.pore.n_corners,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.corner_half_angle,
        network.pn_core.pore.b,
        network.pn_core.pore.oil_filled,
    )
    network.pn_core.throat.b_pinned[:] = network.pn_core.throat.b
    calculate_water_corner_collapse_pc(
        network.pn_core.throat.water_corner_collapse_pressure,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.n_corners,
        network.pn_core.throat.b,
        network.pn_core.throat.corner_half_angle,
        c.sigma,
    )
    calculate_2nd_drainage_oil_layer_existance_network(
        network.pn_core.throat.corner_oil_layer, network.pn_core.throat.bo, network.pn_core.throat.n_corners
    )
    calculate_water_filled_multiphase(
        network.pn_core.throat.water_filled_multiphase,
        network.pn_core.throat.n_corners,
        network.pn_core.throat.corner_oil_layer,
        network.pn_core.throat.bo,
    )
    calculate_2nd_drainage_oil_filled_multiphase(
        network.pn_core.throat.oil_filled_multiphase,
        network.pn_core.throat.n_corners,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.corner_half_angle,
        network.pn_core.throat.b,
        network.pn_core.throat.oil_filled,
    )

    calculate_2nd_drainage_snap_off_values(network, min_pc)
    calculate_2nd_drainage_piston_like_values(network, min_pc)

    return network


def calculate_piston_like_values(network, max_pc):
    Rmin = calculate_r(max_pc, c.sigma, 100000)

    network.pn_core["throat.piston_like_pressure"] = calculate_imbibition_piston_like_displacement_pc_network(
        network.pn_core.throat.shape_factor,
        network.pn_core.throat.radius,
        network.pn_core.throat.advancing_contact_angle,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.shape,
        network.pn_core.throat.corner_half_angle,
        network.pn_core.throat.n_corners,
        Rmin,
        c.sigma,
    )

    network.pn_core["pore.piston_like_pressure"] = calculate_imbibition_piston_like_displacement_pc_network(
        network.pn_core.pore.shape_factor,
        network.pn_core.pore.radius,
        network.pn_core.pore.advancing_contact_angle,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.shape,
        network.pn_core.pore.corner_half_angle,
        network.pn_core.pore.n_corners,
        Rmin,
        c.sigma,
    )


def calculate_snap_off_values(network, max_pc):
    Rmin = calculate_r(max_pc, c.sigma, 100000)

    network.pn_core["throat.snap_off_pressure"] = calculate_snap_off_pc_network(
        network.pn_core.throat.radius,
        network.pn_core.throat.corner_half_angle,
        network.pn_core.throat.advancing_contact_angle,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.n_corners,
        Rmin,
        np.zeros(len(network.pn_core.throat.radius)),
        c.sigma,
        c.pw,
        c.po,
        c.g,
    )

    network.pn_core["pore.snap_off_pressure"] = calculate_snap_off_pc_network(
        network.pn_core.pore.radius,
        network.pn_core.pore.corner_half_angle,
        network.pn_core.pore.advancing_contact_angle,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.n_corners,
        Rmin,
        np.zeros(len(network.pn_core.pore.radius)),
        c.sigma,
        c.pw,
        c.po,
        c.g,
    )


def calculate_oil_layer_collapse_values(network, max_pc):
    Rmin = calculate_r(max_pc, c.sigma, 100000)

    calculate_oil_layer_collapse_pc_network(
        network.pn_core.throat.oil_layer_collapse_pressure,
        network.pn_core.throat.corner_half_angle,
        network.pn_core.throat.advancing_contact_angle,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.n_corners,
        Rmin,
        c.sigma,
    )

    calculate_oil_layer_collapse_pc_network(
        network.pn_core.pore.oil_layer_collapse_pressure,
        network.pn_core.pore.corner_half_angle,
        network.pn_core.pore.advancing_contact_angle,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.n_corners,
        Rmin,
        c.sigma,
    )


def calculate_2nd_drainage_piston_like_values(network, min_pc):
    Rmax = calculate_r(min_pc, c.sigma, 100000)
    network.pn_core["throat.piston_like_pressure"] = calculate_2nd_drainage_piston_like_displacement_pc_network(
        network.pn_core.throat.shape_factor,
        network.pn_core.throat.radius,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.advancing_contact_angle,
        network.pn_core.throat.corner_oil_layer,
        network.pn_core.throat.n_corners,
        network.pn_core.throat.bo,
        network.pn_core.throat.shape,
        network.pn_core.throat.corner_half_angle,
        Rmax,
        c.sigma,
    )

    network.pn_core["pore.piston_like_pressure"] = calculate_2nd_drainage_piston_like_displacement_pc_network(
        network.pn_core.pore.shape_factor,
        network.pn_core.pore.radius,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.advancing_contact_angle,
        network.pn_core.pore.corner_oil_layer,
        network.pn_core.pore.n_corners,
        network.pn_core.pore.bo,
        network.pn_core.pore.shape,
        network.pn_core.pore.corner_half_angle,
        Rmax,
        c.sigma,
    )
    # print(network.pn_core.pore.piston_like_pressure)


def calculate_2nd_drainage_snap_off_values(network, min_pc):
    Rmax = calculate_r(min_pc, c.sigma, 100000)

    network.pn_core["throat.snap_off_pressure"] = calculate_2nd_drainage_snap_off_pc_network(
        network.pn_core.throat.radius,
        network.pn_core.throat.corner_half_angle,
        network.pn_core.throat.advancing_contact_angle,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.n_corners,
        Rmax,
        np.zeros(len(network.pn_core.throat.radius)),
        c.sigma,
    )

    network.pn_core["pore.snap_off_pressure"] = calculate_2nd_drainage_snap_off_pc_network(
        network.pn_core.pore.radius,
        network.pn_core.pore.corner_half_angle,
        network.pn_core.pore.advancing_contact_angle,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.n_corners,
        Rmax,
        np.zeros(len(network.pn_core.pore.radius)),
        c.sigma,
    )


def calculate_element_initial_values(element, element_id):
    new_shape, new_G = calculate_shape(element.get_G(element_id))
    element.set_shape(element_id, new_shape)
    element.set_G(element_id, new_G)
    element_r = element.get_r(element_id)
    element.set_A(element_id, calculate_area(element_r, new_G))
    new_beta = calculate_beta(new_G, new_shape)
    element.set_beta(element_id, new_beta)
    element_n_corners = element.get_n_corners(element_id)
    new_sides = calculate_sides(new_beta, element_r, element.get_n_corners(element_id))
    element.set_sides(element_id, new_sides)
    element.set_max_b(element_id, calculate_max_b(new_sides, element_n_corners))


def calculate_element_drainage_values(element, element_id, max_pc):
    element.set_b(
        element_id,
        calculate_drainage_b(
            element.get_beta(element_id),
            np.full(
                element.get_n_corners(element_id),
                element.get_receding_contact_angle(element_id),
            ),
            element.get_n_corners(element_id),
            element.get_max_b(element_id),
            max_pc,
            c.sigma,
        ),
    )
    element.set_C_r(
        element_id,
        calculate_C(
            np.full(
                element.get_n_corners(element_id),
                element.get_receding_contact_angle(element_id),
            ),
            element.get_beta(element_id),
            element.get_n_corners(element_id),
        ),
    )
    element.set_C_a(
        element_id,
        calculate_C(
            np.full(
                element.get_n_corners(element_id),
                element.get_advancing_contact_angle(element_id),
            ),
            element.get_beta(element_id),
            element.get_n_corners(element_id),
        ),
    )
    element.set_pc(
        element_id,
        calculate_drainage_pc(
            element.get_G(element_id),
            element.get_r(element_id),
            element.get_receding_contact_angle(element_id),
            element.get_shape(element_id),
            element.get_beta(element_id),
            element.get_n_corners(element_id),
            c.sigma,
        ),
    )


def calculate_element_imbibition_values(element, element_id, max_pc):
    element.set_C_r(
        element_id,
        calculate_C(
            np.full(
                element.get_n_corners(element_id),
                element.get_receding_contact_angle(element_id),
            ),
            element.get_beta(element_id),
            element.get_n_corners(element_id),
        ),
    )
    element.set_C_a(
        element_id,
        calculate_C(
            np.full(
                element.get_n_corners(element_id),
                element.get_advancing_contact_angle(element_id),
            ),
            element.get_beta(element_id),
            element.get_n_corners(element_id),
        ),
    )

    element.set_snap_off_viability(
        element_id,
        calculate_imbibition_snap_off_viability(
            element.get_receding_contact_angle(element_id),
            element.get_beta(element_id),
            element.get_n_corners(element_id),
        ),
    )


def update_wettability_property(network):
    network.pn_core.pore.changed_wettability[:] = network.pn_core.pore.oil_filled
    network.pn_core.throat.changed_wettability[:] = network.pn_core.throat.oil_filled


class StepSimulation:
    def __init__(
        self,
        pn,
        parameters_dict={},
        config={},
        generate_cas=True,
        calculate_full_conductance=False,
        set_initial_values=True,
        diagnostics_output_path=False,
    ):
        self.simulation = None
        self.max_pc = config["max_pc"] if config else None
        self.min_pc = 0
        self.calculate_full_conductance = calculate_full_conductance
        self.is_first_step = True
        self.current_pc = 0
        self.cycle = "drainage"
        self.diagnostics_output_path = diagnostics_output_path

        self.parameter_manager = ParameterManager(parameters_dict)

        if set_initial_values:
            initialize_pn(pn)

        log_conductivity_file = config.get("log_conductivity_file") if config else None
        self.network = NetworkManager(pn, log_conductivity_file=log_conductivity_file)

        if generate_cas:
            self.parameter_manager.generate_contact_angles(self.network)

        if set_initial_values:
            calculate_initial_values(self.network)

        self.diagnostics_writer = None

    def set_max_pc(self, max_pc):
        self.max_pc = max_pc

    def set_cycle(self, cycle="imbibition"):
        self.cycle = cycle
        batch = self.parameter_manager.batch_invasions
        if cycle == "drainage":
            self.parameter_manager.update_initial_cas(self.network)
            calculate_drainage_values(self.network, self.max_pc)
            if self.diagnostics_output_path:
                Path(self.diagnostics_output_path).mkdir(parents=True, exist_ok=True)
                self.diagnostics_writer = DiagnosticsWriter(self.network, self.diagnostics_output_path)
            self.simulation = DrainageSimulation(self.network, self.calculate_full_conductance, batch_invasions=batch)
            self.max_pc = 0
        elif cycle == "imbibition":
            calculate_drainage_pinned_b(
                self.network, self.max_pc
            )  # Necessary because drainage doesn't calculate theta_h
            update_wettability_property(self.network)
            self.parameter_manager.update_equilibrium_cas(self.network)
            calculate_imbibition_values(self.network, self.max_pc)
            self.simulation = ImbibitionSimulation(self.network, self.max_pc, self.calculate_full_conductance, batch_invasions=batch)
            self.min_pc = self.max_pc
        elif cycle == "2nd_drainage":
            self.parameter_manager.update_2nd_drainage_ca(self.network)
            calculate_2nd_drainage_values(self.network, self.min_pc)
            self.simulation = SecondDrainageSimulation(self.network, self.max_pc, self.calculate_full_conductance, batch_invasions=batch)

    def update(self, pc, skip_invasions=False, verbose=False):
        if skip_invasions:
            if verbose: yield "Skipping invasions"
            self.current_pc = pc
        else:
            if verbose: yield "Performing simulation step"
            self.current_pc = yield from self._perform_step(pc, verbose=verbose)

        if verbose: yield "Calculating saturations"
        Sw = self.simulation.calculate_sw_total()

        if verbose: yield "Calculating relative permeability"
        Krw, Kro = self.simulation.calculate_krel()

        pores_filled, total_pores, percentage = self.simulation.get_oil_filled_percentage()

        if self.diagnostics_writer is not None:
            self.diagnostics_writer.save_state()
        return self.current_pc, Sw, Krw, Kro, pores_filled, total_pores, percentage

    def get_pores_left(self):
        return self.simulation.get_pores_left()

    def get_throats_left(self):
        return self.simulation.get_throats_left()

    def get_parameter_manager(self):
        return self.parameter_manager

    def get_snapshot(self, current_pc):
        snapshot_dict = {}
        snapshot_dict["parameters"] = self.parameter_manager.parameters_dict
        snapshot_dict["pc"] = current_pc
        snapshot_dict["max_pc"] = self.max_pc
        snapshot_dict["cycle"] = self.cycle
        snapshot_dict["network"] = self.network.pn.copy()
        return snapshot_dict

    def end_cycle(self):
        if self.diagnostics_writer is not None:
            self.diagnostics_writer.save_cycle_data()

    def end_simulation(self):
        if self.diagnostics_writer is not None:
            self.diagnostics_writer.close()

    @staticmethod
    def load_snapshot(snapshot_file: str) -> tuple:
        with open(snapshot_file, "rb") as fp:
            snapshot_dict = pickle.load(fp)

        pn = PoreNetwork()
        pn.update(snapshot_dict["network"])
        parameters = snapshot_dict["parameters"]
        parameters["pc_initial"] = snapshot_dict["pc"]

        config = {}
        if snapshot_dict["cycle"] == "imbibition":
            config["skip_drainage"] = True
            config["skip_imbibition"] = False
        elif snapshot_dict["cycle"] == "2nd_drainage":
            config["skip_drainage"] = True
            config["skip_imbibition"] = True
        else:
            config["skip_drainage"] = False
            config["skip_imbibition"] = False
        config["pc"] = snapshot_dict["pc"]
        config["max_pc"] = snapshot_dict["max_pc"]

        return pn, parameters, config

    def _perform_step(self, pc, verbose=False):
        return (yield from self._step_by_pc(pc, verbose=verbose))

    def _step_by_pc(self, pc, verbose=False):
        yield from self.simulation.update_pc(pc, verbose=verbose)
        self.max_pc = max(self.max_pc, pc)
        self.min_pc = min(self.min_pc, pc)
        return pc


class MultiStepSimulation(StepSimulation):
    def __init__(
        self,
        pn,
        parameters_dict={},
        config={},
        generate_cas=True,
        calculate_full_conductance=False,
        set_initial_values=True,
        diagnostics_output_path=False,
    ):
        super().__init__(
            pn,
            parameters_dict,
            config,
            generate_cas,
            calculate_full_conductance,
            set_initial_values,
            diagnostics_output_path,
        )
        self.pc_incrementer = None
        self.sw_step = None
        self.no_invasion_left = False

    def set_pc_incrementer(self, pc_incrementer):
        self.pc_incrementer = pc_incrementer

    def set_cycle(self, cycle="imbibition"):
        super().set_cycle(cycle)
        self.cycle = cycle
        if cycle == "drainage":
            self.sw_step = None if self.parameter_manager.control_by_pc else self.parameter_manager.drainage_sw_step
        else:
            self.sw_step = None if self.parameter_manager.control_by_pc else self.parameter_manager.imbibition_sw_step
        self.is_first_step = True
        self.no_invasion_left = False

    def cycle_should_end(self):
        return self.pc_incrementer.cycle_should_end() or (self.get_pores_left() == 0 and self.get_throats_left() == 0) or self.no_invasion_left

    def _perform_step(self, pc, verbose=False):
        if self.sw_step is None or self.is_first_step:
            current_pc = yield from super()._perform_step(pc, verbose=verbose)
        else:
            current_pc = yield from self._step_by_sw(pc, verbose=verbose)
        self.is_first_step = False
        return current_pc

    def _step_by_sw(self, pc, verbose=False):
        total_sw_delta = 0.0
        next_pc = pc
        while True:
            sw_delta, current_pc, next_pc, self.no_invasion_left = yield from self.simulation.update_pc(next_pc, self.sw_step, verbose=verbose)
            self.max_pc = max(self.max_pc, current_pc)
            self.min_pc = min(self.min_pc, current_pc)
            total_sw_delta += sw_delta
            if total_sw_delta >= self.sw_step or self.cycle_should_end():
                break
        return current_pc
