import numpy as np
import openpnm
from pyflowsolver import NetworkManager
from pypardiso import spsolve
from scipy.sparse import csr_matrix

try:
    from petsc4py import PETSc
except ImportError:
    PETSc = None

from py_pore_flow.equations.geometry import (
    WATER,
)
from py_pore_flow.simulation.conductance_calculator import ConductanceCalculator
from py_pore_flow.simulation.solver import Solver


# Flags
USE_OPENPNM_SOLVER = False
SOLVER = "pypardiso"  # "pypardiso" or "petsc"
PETSC_USE_PARDISO_PRECONDITIONER = False


class FlowRateCalculator:
    def __init__(self, network):
        self.network = network
        self.solver = Solver(self.network.pn)

    def calculate_singlephase_flow_rate(self, phase):
        if phase == WATER:
            phase_name = "water"
        else:
            phase_name = "oil"

        self.network.pn_core["throat.hydraulic_conductance"] = self.network.pn[
            f"throat.{phase_name}_singlephase_conductance"
        ]

        return self._solve_flow_rate(phase)

    def calculate_total_flow_rate(self, phase, update_hydraulic_conductance=True, initial_guess=None):
        if update_hydraulic_conductance:
            self.update_hydraulic_conductance(phase)
        return self._solve_flow_rate(phase, initial_guess=initial_guess)

    def update_hydraulic_conductance(self, phase):
        if phase == WATER:
            phase_name = "water"
        else:
            phase_name = "oil"

        conductance_calculator = ConductanceCalculator(self.network)
        conductance_calculator.calculate_multiphase_conductance(phase)

        self.network.pn_core["throat.hydraulic_conductance"] = self.network.pn[
            f"throat.{phase_name}_multiphase_conductance"
        ]

    def _solve_flow_rate(self, phase, initial_guess=None):
        if USE_OPENPNM_SOLVER:
            return self._openpnm_solver(phase)
        else:
            sub_network = self.solver.get_sub_network()
            if sub_network is False:
                return 0

            conn = sub_network["throat.conns"].astype(np.int32)
            cond = sub_network["throat.hydraulic_conductance"].astype(np.float64)
            inlets = sub_network["pore.inlets"]
            outlets = sub_network["pore.outlets"]

            if np.all(cond == 0):
                return 0

            min_cond = cond[cond > 0].min() if np.any(cond <= 0) else cond.min()
            max_cond = cond.max()

            if min_cond <= 0:
                positive_cond = cond[cond > 0]
                if len(positive_cond) == 0:
                    return 0
                min_cond = positive_cond.min()
                max_cond = positive_cond.max()

            # Prevent pypardiso from breaking with values too close to 0
            factor = 10 ** ((np.log10(min_cond) + np.log10(max_cond)) / 2)
            cond = cond / factor

            network_manager = NetworkManager(
                conn=conn,
                cond=cond,
                inlets=inlets,
                outlets=outlets,
            )
            network_manager.generate_sparse_system()
            a_sparse_array, b_array = network_manager.get_sparse_system()

            if len(a_sparse_array["val"]) == 0:
                return 1

            A_csr = csr_matrix(
                (
                    a_sparse_array["val"],
                    a_sparse_array["col_idx"],
                    np.append(a_sparse_array["row_ptr"], a_sparse_array["val"].size),
                )
            )

            if SOLVER == "petsc":
                x = self._petsc_solver(A_csr, b_array, initial_guess=initial_guess)
            else:
                x = np.atleast_1d(self._pypardiso_solver(A_csr, b_array))

            pressure = network_manager.get_pressure_list(x)
            flow_rate = network_manager.get_flow_rate(pressure)

            return flow_rate * factor

    def _pypardiso_solver(self, A_csr, b_array):
        return spsolve(A_csr, b_array)

    def _petsc_solver(self, A_csr, b_array, initial_guess=None):
        if PETSc is None:
            raise ImportError("petsc4py is not installed, which is required for using the PETSc solver.")

        A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        x_petsc, b_petsc = A_petsc.createVecs()
        b_petsc.setArray(b_array)
        if initial_guess is not None:
            x_petsc.setArray(initial_guess)

        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)

        # Removed custom PardisoPreconditioner for using PETSc's default or configured preconditioner.
        ksp.setFromOptions()
        ksp.solve(b_petsc, x_petsc)

        return x_petsc.getArray()

    def _openpnm_solver(self, phase):
        if phase == WATER:
            phase = openpnm.phase.Water(network=self.network.pn)
        else:
            phase = openpnm.phase.Air(network=self.network.pn)
        # water.add_model_collection(openpnm.models.collections.physics.standard)
        stokes_flow = openpnm.algorithms.StokesFlow(network=self.network.pn, phase=phase)

        stokes_flow.set_value_BC(
            pores=self.network.get_inlets_ids(), values=101325, mode="overwrite"
        )  # pressure in pa: 101325 pa = 1 atm
        stokes_flow.set_value_BC(pores=self.network.get_outlets_ids(), values=0, mode="overwrite")
        stokes_flow.run()
        inlet_flow = stokes_flow.rate(pores=self.network.get_inlets_ids())
        outlet_flow = stokes_flow.rate(pores=self.network.get_outlets_ids())
        flow_rate = (inlet_flow[0] - outlet_flow[0]) / 2
        return flow_rate
