"""
Relative permeability calculator.

Optimised version: avoids cloning the full OpenPNM network and calling
openpnm.topotools.trim(). Instead, builds a masked subnetwork dict
directly from the original network arrays, uses scipy for connected
component detection, and solves the sparse linear system directly
without creating temporary OpenPNM Network objects.
"""

from numba import njit
import numpy as np
from pypardiso import spsolve
from pyflowsolver import NetworkManager as FlowNetworkManager
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from py_pore_flow.equations.geometry import WATER, OIL
from py_pore_flow.equations.conductance import calculate_absolute_permeability
from py_pore_flow.simulation.flow_rate_calculator import FlowRateCalculator


class KrelCalculator:
    def __init__(self, network, calculate_full_conductance: bool = False):
        self.network = network
        self.calculate_full_conductance = calculate_full_conductance

        flow_rate_calculator = FlowRateCalculator(self.network)

        self.singlephase_flow_rate = [0.0] * 2
        self.singlephase_flow_rate[WATER] = flow_rate_calculator.calculate_singlephase_flow_rate(WATER)
        self.singlephase_flow_rate[OIL] = flow_rate_calculator.calculate_singlephase_flow_rate(OIL)
        self.absolute_permeability = self._calculate_K()

    # ── Public API ────────────────────────────────────────────────────────────

    def calculate_kro(self) -> float:
        return self._calculate_kr(OIL)

    def calculate_krw(self) -> float:
        return self._calculate_kr(WATER)

    def get_absolute_permeability(self) -> float:
        return self.absolute_permeability

    # ── Core computation ──────────────────────────────────────────────────────

    def _calculate_kr(self, phase: int) -> float:
        """Compute relative permeability for *phase* without cloning the network.

        Steps:
        1. Update multiphase hydraulic conductance on the original network.
        2. Build a pore/throat mask for elements with nonzero phase saturation.
        3. Keep only throats whose *both* endpoint pores survive the mask.
        4. Detect connected components on the masked sub-graph (scipy).
        5. For each component that spans inlets→outlets, solve flow directly.
        6. Sum flow rates and normalise by single-phase flow rate.
        """
        # Always compute conductance on the full network first.
        # Conductance is per-element, so the result is identical whether
        # computed on the full network or per-component.
        flow_rate_calculator = FlowRateCalculator(self.network)
        flow_rate_calculator.update_hydraulic_conductance(phase)

        pn = self.network.pn
        sat_key = "pore.oil_saturation" if phase == OIL else "pore.water_saturation"
        tsat_key = "throat.oil_saturation" if phase == OIL else "throat.water_saturation"
        phase_name = "oil" if phase == OIL else "water"

        pore_mask = pn[sat_key] > 0
        throat_mask = pn[tsat_key] > 0

        conns = pn["throat.conns"]
        # Also require both endpoint pores to be present
        throat_mask = triple_compare(
            throat_mask, 
            pore_mask,
            conns,
        )

        n_pores_keep = pore_mask.sum()
        n_throats_keep = throat_mask.sum()

        if n_pores_keep == 0 or n_throats_keep == 0:
            return 0.0

        # Find connected components on the masked subgraph
        pore_indices = np.where(pore_mask)[0]
        throat_indices = np.where(throat_mask)[0]

        # Build old→new pore index mapping
        old_to_new = np.full(len(pore_mask), -1, dtype=np.int64)
        old_to_new[pore_indices] = np.arange(n_pores_keep, dtype=np.int64)

        sub_conns = old_to_new[conns[throat_indices]]

        # Build adjacency for connected components
        rows = np.concatenate([sub_conns[:, 0], sub_conns[:, 1]])
        cols = np.concatenate([sub_conns[:, 1], sub_conns[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)
        adj = coo_matrix((data, (rows, cols)), shape=(n_pores_keep, n_pores_keep))

        n_components, labels = connected_components(adj, directed=False)

        sub_inlets = pn["pore.inlets"][pore_indices]
        sub_outlets = pn["pore.outlets"][pore_indices]

        inlet_labels = set(labels[sub_inlets].tolist())
        outlet_labels = set(labels[sub_outlets].tolist())
        valid_labels = inlet_labels & outlet_labels

        if not valid_labels:
            return 0.0

        cond_key = f"throat.{phase_name}_multiphase_conductance"

        total_flow_rate = 0.0
        for vlabel in valid_labels:
            component_pore_mask_sub = labels == vlabel
            component_pore_ids_orig = pore_indices[component_pore_mask_sub]

            # Throats: both endpoints in this component
            component_throat_mask = (
                np.isin(conns[throat_indices, 0], component_pore_ids_orig)
                & np.isin(conns[throat_indices, 1], component_pore_ids_orig)
            )
            component_throat_ids_orig = throat_indices[component_throat_mask]

            flow_rate = self._solve_direct(
                pn, component_pore_ids_orig, component_throat_ids_orig, cond_key,
            )
            total_flow_rate += flow_rate

        sp_flow = self.singlephase_flow_rate[phase]
        if sp_flow == 0:
            return 0.0
        return total_flow_rate / sp_flow

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _solve_direct(pn, pore_ids, throat_ids, cond_key):
        """Solve flow on a connected component directly from arrays.

        No OpenPNM Network objects are created — builds the sparse linear
        system from raw numpy arrays and solves with pypardiso.
        """
        if len(pore_ids) == 0 or len(throat_ids) == 0:
            return 0.0

        inlets = pn["pore.inlets"][pore_ids]
        outlets = pn["pore.outlets"][pore_ids]

        if not np.any(inlets) or not np.any(outlets):
            return 0.0

        # Remap pore indices to 0..n
        old_to_new = np.full(len(pn["pore.all"]), -1, dtype=np.int64)
        old_to_new[pore_ids] = np.arange(len(pore_ids), dtype=np.int64)

        conn = old_to_new[pn["throat.conns"][throat_ids]].astype(np.int32)
        cond = pn[cond_key][throat_ids].astype(np.float64)

        if np.all(cond == 0):
            return 0.0

        positive_cond = cond[cond > 0]
        if len(positive_cond) == 0:
            return 0.0
        min_cond = positive_cond.min()
        max_cond = positive_cond.max()

        # Prevent pypardiso from breaking with values too close to 0
        factor = 10 ** ((np.log10(min_cond) + np.log10(max_cond)) / 2)
        cond = cond / factor

        network_manager = FlowNetworkManager(
            conn=conn, cond=cond, inlets=inlets, outlets=outlets,
        )
        network_manager.generate_sparse_system()
        a_sparse_array, b_array = network_manager.get_sparse_system()

        if len(a_sparse_array["val"]) == 0:
            return factor

        A_csr = csr_matrix((
            a_sparse_array["val"],
            a_sparse_array["col_idx"],
            np.append(a_sparse_array["row_ptr"], a_sparse_array["val"].size),
        ))

        x = spsolve(A_csr, b_array)
        pressure = network_manager.get_pressure_list(x)
        flow_rate = network_manager.get_flow_rate(pressure)

        return flow_rate * factor

    def _calculate_K(self):
        A, L = self._get_network_size(self.network.pn)
        qtsp = self.singlephase_flow_rate[WATER]
        return calculate_absolute_permeability(WATER, qtsp, A, L, 101325, 0)

    @staticmethod
    def _get_network_size(pn):
        dimension_list = [0, 1, 2]
        element_max = [0.0] * len(dimension_list)
        element_min = [0.0] * len(dimension_list)

        for dimension in dimension_list:
            element_min[dimension] = pn["pore.coords"][:, dimension].min()
            element_max[dimension] = pn["pore.coords"][:, dimension].max()

        inlets = pn["pore.inlets"]
        outlets = pn["pore.outlets"]

        flow_dimension = None
        for dimension in dimension_list:
            inlet_min = pn["pore.coords"][inlets][:, dimension].min()
            inlet_max = pn["pore.coords"][inlets][:, dimension].max()
            if inlet_min == inlet_max:
                flow_dimension = dimension
                break

        dimension_list.remove(flow_dimension)
        cross_sectional_area = (
            element_max[dimension_list[0]]
            - element_min[dimension_list[0]] * element_max[dimension_list[1]]
            - element_min[dimension_list[1]]
        )
        length = element_max[flow_dimension] - element_min[flow_dimension]

        return cross_sectional_area, length

@njit(cache=True)
def triple_compare(
    throat_mask, 
    pore_mask,
    conns,
    ):
    n = throat_mask.size
    output = np.empty(n, dtype=np.bool_)
    for i in range(n):
        output[i] = throat_mask[i] \
        & pore_mask[conns[i, 0]] \
        & pore_mask[conns[i, 1]]
    return output
