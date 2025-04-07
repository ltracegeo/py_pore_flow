import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import OIL, WATER


# Constants
PORE = 0
THROAT = 1

TRAP_TOLERANCE = 0.0


class PhaseConnectionChecker:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase

        water_connection = PhaseConnectionChecker2(
            self.network.pn_core,
            self.network.get_inlets_ids(),
            self.network.get_outlets_ids(),
            WATER,
        )
        oil_connection = PhaseConnectionChecker2(
            self.network.pn_core,
            self.network.get_inlets_ids(),
            self.network.get_outlets_ids(),
            OIL,
        )

        if self.phase == WATER:
            self.invasion_connections = water_connection
            self.evasion_connections = oil_connection
        else:
            self.invasion_connections = oil_connection
            self.evasion_connections = water_connection

    def reset_phase_connections(self):
        self.invasion_connections.reset_connections()
        self.evasion_connections.reset_connections()

    def update_phase_connections(self, element_type, element_id):
        if element_type == PORE:
            pore_id_list = np.array([element_id], dtype=np.int64)
            throat_id_list = np.empty(0, dtype=np.int64)
        else:
            pore_id_list = np.empty(0, dtype=np.int64)
            throat_id_list = np.array([element_id], dtype=np.int64)

        self.invasion_connections.update_connections(pore_id_list, throat_id_list)
        self.evasion_connections.reset_connections()


class PhaseConnectionChecker2:
    def __init__(self, pn, pore_inlets_ids, pore_outlets_ids, phase):
        self.pn = pn
        self.pore_inlets_ids = pore_inlets_ids
        self.pore_outlets_ids = pore_outlets_ids
        self.phase = phase

        self.wqupc = Wqupc(len(self.pn.pore.all), len(self.pn.throat.all))

    def reset_connections(self):
        if self.phase == WATER:
            self.__reset_connections(
                self.pn.pore.water_connected_to_inlet,
                self.pn.pore.water_connected_to_outlet,
                self.pn.throat.water_connected_to_inlet,
                self.pn.throat.water_connected_to_outlet,
                self.pn.pore.water_saturation > 0.0 + TRAP_TOLERANCE,
                self.pn.throat.water_saturation > 0.0 + TRAP_TOLERANCE,
            )
        else:
            self.__reset_connections(
                self.pn.pore.oil_connected_to_inlet,
                self.pn.pore.oil_connected_to_outlet,
                self.pn.throat.oil_connected_to_inlet,
                self.pn.throat.oil_connected_to_outlet,
                self.pn.pore.oil_saturation > 0.0 + TRAP_TOLERANCE,
                self.pn.throat.oil_saturation > 0.0 + TRAP_TOLERANCE,
            )

    def update_connections(self, pore_id_list, throat_id_list):
        if self.phase == WATER:
            self.__update_connections(
                self.pn.pore.water_connected_to_inlet,
                self.pn.pore.water_connected_to_outlet,
                self.pn.throat.water_connected_to_inlet,
                self.pn.throat.water_connected_to_outlet,
                self.pn.pore.water_saturation > 0.0 + TRAP_TOLERANCE,
                self.pn.throat.water_saturation > 0.0 + TRAP_TOLERANCE,
                pore_id_list,
                throat_id_list,
            )
        else:
            self.__update_connections(
                self.pn.pore.oil_connected_to_inlet,
                self.pn.pore.oil_connected_to_outlet,
                self.pn.throat.oil_connected_to_inlet,
                self.pn.throat.oil_connected_to_outlet,
                self.pn.pore.oil_saturation > 0.0 + TRAP_TOLERANCE,
                self.pn.throat.oil_saturation > 0.0 + TRAP_TOLERANCE,
                pore_id_list,
                throat_id_list,
            )

    def __update_connections(
        self,
        pore_connected_to_inlet,
        pore_connected_to_outlet,
        throat_connected_to_inlet,
        throat_connected_to_outlet,
        pore_has_phase,
        throat_has_phase,
        pore_id_list,
        throat_id_list,
    ):
        self.__jit_update_connections(
            self.wqupc.parent_array,
            self.wqupc.size_array,
            self.wqupc.nP,
            self.wqupc.nT,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            pore_has_phase,
            self.pn.pore.conns,
            self.pn.pore.n_conns,
            throat_has_phase,
            self.pn.throat.conns,
            self.pore_inlets_ids,
            self.pore_outlets_ids,
            pore_id_list,
            throat_id_list,
        )

    def __reset_connections(
        self,
        pore_connected_to_inlet,
        pore_connected_to_outlet,
        throat_connected_to_inlet,
        throat_connected_to_outlet,
        pore_has_phase,
        throat_has_phase,
    ):
        self.wqupc.reset()
        self.__jit_reset_connections(
            self.wqupc.parent_array,
            self.wqupc.size_array,
            self.wqupc.nP,
            self.wqupc.nT,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            pore_has_phase,
            throat_has_phase,
            self.pn.throat.conns,
            self.pore_inlets_ids,
            self.pore_outlets_ids,
        )

    @staticmethod
    @nb.njit(cache=True)
    def __jit_update_connections(
        parent_array,
        size_array,
        nP,
        nT,
        pore_connected_to_inlet,
        pore_connected_to_outlet,
        throat_connected_to_inlet,
        throat_connected_to_outlet,
        pore_has_phase,
        pore_conns,
        pore_n_conns,
        throat_has_phase,
        throat_conns,
        pore_inlets_ids,
        pore_outlets_ids,
        pore_id_list,
        throat_id_list,
    ):
        cluster_add_union(
            parent_array,
            size_array,
            nT,
            pore_has_phase,
            pore_conns,
            pore_n_conns,
            throat_has_phase,
            throat_conns,
            pore_id_list,
            throat_id_list,
        )
        cluster_update_connections(
            parent_array,
            nP,
            nT,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            pore_inlets_ids,
            pore_outlets_ids,
        )

    @staticmethod
    @nb.njit(cache=True)
    def __jit_reset_connections(
        parent_array,
        size_array,
        nP,
        nT,
        pore_connected_to_inlet,
        pore_connected_to_outlet,
        throat_connected_to_inlet,
        throat_connected_to_outlet,
        pore_has_phase,
        throat_has_phase,
        throat_conns,
        pore_inlets_ids,
        pore_outlets_ids,
    ):
        cluster_create_union(
            parent_array,
            size_array,
            nT,
            pore_has_phase,
            throat_has_phase,
            throat_conns,
        )
        cluster_update_connections(
            parent_array,
            nP,
            nT,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            pore_inlets_ids,
            pore_outlets_ids,
        )


@nb.njit(cache=True)
def cluster_create_union(
    parent_array,
    size_array,
    nT,
    pore_has_phase,
    throat_has_phase,
    throat_conns,
):
    union = lambda p, q: wqupc_union(p, q, parent_array, size_array)

    for throat in range(nT):
        if throat_has_phase[throat]:
            pore_0, pore_1 = throat_conns[throat]
            if pore_has_phase[pore_0]:
                union(throat, (pore_0 + nT))
            if pore_has_phase[pore_1]:
                union(throat, (pore_1 + nT))


@nb.njit(cache=True)
def cluster_add_union(
    parent_array,
    size_array,
    nT,
    pore_has_phase,
    pore_conns,
    pore_n_conns,
    throat_has_phase,
    throat_conns,
    pore_id_list,
    throat_id_list,
):
    union = lambda p, q: wqupc_union(p, q, parent_array, size_array)

    for throat_id in throat_id_list:
        if throat_has_phase[throat_id]:
            pore_0, pore_1 = throat_conns[throat_id]
            if pore_has_phase[pore_0]:
                union(throat_id, (pore_0 + nT))
            if pore_has_phase[pore_1]:
                union(throat_id, (pore_1 + nT))
    for pore_id in pore_id_list:
        if pore_has_phase[pore_id]:
            neighbor_throat_ids = pore_conns[pore_id][: pore_n_conns[pore_id]]
            for throat_id in neighbor_throat_ids:
                if throat_has_phase[throat_id]:
                    union(throat_id, (pore_id + nT))


@nb.njit(cache=True)
def cluster_update_connections(
    parent_array,
    nP,
    nT,
    pore_connected_to_inlet,
    pore_connected_to_outlet,
    throat_connected_to_inlet,
    throat_connected_to_outlet,
    pore_inlets_ids,
    pore_outlets_ids,
):
    find = lambda p: wqupc_find(p, parent_array)

    inlets_clusters = set()
    outlets_clusters = set()

    for pore_id in pore_inlets_ids:
        cluster = find(pore_id + nT)
        inlets_clusters.add(cluster)
    for pore_id in pore_outlets_ids:
        cluster = find(pore_id + nT)
        outlets_clusters.add(cluster)

    for throat in range(nT):
        cluster = find(throat)
        throat_connected_to_inlet[throat] = cluster in inlets_clusters
        throat_connected_to_outlet[throat] = cluster in outlets_clusters

    for pore in range(nP):
        cluster = find(pore + nT)
        pore_connected_to_inlet[pore] = cluster in inlets_clusters
        pore_connected_to_outlet[pore] = cluster in outlets_clusters


class Wqupc:
    def __init__(self, nP, nT):
        self.nP = nP
        self.nT = nT
        self.parent_array = np.arange(self.nP + self.nT, dtype=np.uint32)
        self.size_array = np.ones((self.nP + self.nT))

    def union(self, p, q):
        wqupc_union(p, q + self.nT, self.parent_array, self.size_array)

    def find_throat(self, p):
        return wqupc_find(p, self.parent_array)

    def find_pore(self, p):
        return wqupc_find(p + self.nT, self.parent_array)

    def reset(self):
        wqupc_reset(
            self.parent_array,
            self.size_array,
            self.nP,
            self.nT,
        )


@nb.njit(cache=True)
def wqupc_find(p, parent_array):
    root = p
    while root != parent_array[root]:
        # parent_array[root] = parent_array[parent_array[root]]
        root = parent_array[root]

    while p != root:
        next_p = parent_array[p]
        parent_array[p] = root
        p = next_p

    return root


@nb.njit(cache=True)
def wqupc_union(p, q, parent_array, size_array):
    # Find the roots of the elements p and q
    rootP = wqupc_find(p, parent_array)
    rootQ = wqupc_find(q, parent_array)

    if rootP == rootQ:
        return

    if size_array[rootP] < size_array[rootQ]:
        parent_array[rootP] = rootQ
        size_array[rootQ] += size_array[rootP]
    else:
        parent_array[rootQ] = rootP
        size_array[rootP] += size_array[rootQ]


@nb.njit(cache=True)
def wqupc_reset(parent_array, size_array, nP, nT):
    for i in range(nP + nT):
        parent_array[i] = i
        size_array[i] = 1
