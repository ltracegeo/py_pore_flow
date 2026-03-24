"""
Relative permeability calculator.

Optimised version: avoids cloning the full OpenPNM network and calling
openpnm.topotools.trim(). Instead, builds a masked subnetwork dict
directly from the original network arrays, uses scipy for connected
component detection, and solves on the masked subnetwork.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from py_pore_flow.equations.geometry import WATER, OIL
from py_pore_flow.equations.conductance import calculate_absolute_permeability
from py_pore_flow.network.network import NetworkManager
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
        1. Optionally update multiphase hydraulic conductance on the original network.
        2. Build a pore/throat mask for elements with nonzero phase saturation.
        3. Keep only throats whose *both* endpoint pores survive the mask.
        4. Detect connected components on the masked sub-graph (scipy).
        5. For each component that spans inlets→outlets, build a sub-dict and solve.
        6. Sum flow rates and normalise by single-phase flow rate.
        """
        if self.calculate_full_conductance:
            flow_rate_calculator = FlowRateCalculator(self.network)
            flow_rate_calculator.update_hydraulic_conductance(phase)
            update_hc = False
        else:
            update_hc = True

        pn = self.network.pn
        sat_key = "pore.oil_saturation" if phase == OIL else "pore.water_saturation"
        tsat_key = "throat.oil_saturation" if phase == OIL else "throat.water_saturation"

        pore_mask = pn[sat_key] > 0
        throat_mask = pn[tsat_key] > 0

        conns = pn["throat.conns"]
        # Also require both endpoint pores to be present
        throat_mask = throat_mask & pore_mask[conns[:, 0]] & pore_mask[conns[:, 1]]

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

            sub_pn = self._build_sub_dict(pn, component_pore_ids_orig, component_throat_ids_orig)
            if sub_pn is None:
                continue

            flow_rate = self._solve_sub_dict(sub_pn, phase, update_hc)
            total_flow_rate += flow_rate

        sp_flow = self.singlephase_flow_rate[phase]
        if sp_flow == 0:
            return 0.0
        return total_flow_rate / sp_flow

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sub_dict(pn, pore_ids, throat_ids):
        """Build a plain dict sub-network from the original pn using masks.

        Returns None if the sub-network would be trivially unsolvable.
        """
        if len(pore_ids) == 0 or len(throat_ids) == 0:
            return None

        old_to_new = np.full(len(pn["pore.all"]), -1, dtype=np.int64)
        old_to_new[pore_ids] = np.arange(len(pore_ids), dtype=np.int64)

        sub_pn = {}
        for key in pn.keys():
            prefix = key.split(".")[0]
            if prefix == "pore":
                sub_pn[key] = pn[key][pore_ids]
            elif prefix == "throat":
                sub_pn[key] = pn[key][throat_ids]

        # Remap connectivity
        sub_pn["throat.conns"] = old_to_new[pn["throat.conns"][throat_ids]].astype(np.int64)

        sub_pn["pore.all"] = np.ones(len(pore_ids), dtype=bool)
        sub_pn["throat.all"] = np.ones(len(throat_ids), dtype=bool)

        if not np.any(sub_pn.get("pore.inlets", np.array([]))):
            return None
        if not np.any(sub_pn.get("pore.outlets", np.array([]))):
            return None

        return sub_pn

    def _solve_sub_dict(self, sub_pn, phase, update_hydraulic_conductance):
        """Wrap a sub-dict into a temporary OpenPNM network and solve flow."""
        import openpnm

        temp_net = openpnm.network.Network()
        temp_net.update(sub_pn)

        if len(temp_net.throats()) == 0 or len(temp_net.pores()) == 0:
            return 0.0

        new_network = NetworkManager(temp_net, log_conductivity_file=self.network.log_conductivity_file)
        if len(new_network.get_inlets_ids()) == 0 or len(new_network.get_outlets_ids()) == 0:
            return 0.0

        flow_rate_calculator = FlowRateCalculator(new_network)
        return flow_rate_calculator.calculate_total_flow_rate(phase, update_hydraulic_conductance)

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
