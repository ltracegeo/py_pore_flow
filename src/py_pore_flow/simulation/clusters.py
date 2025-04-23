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
            WATER,
        )
        oil_connection = PhaseConnectionChecker2(
            self.network.pn_core,
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
    def __init__(self, pn, phase):
        self.pn = pn
        self.phase = phase

        self.wqupc = Wqupc(len(self.pn.pore.all), len(self.pn.throat.all))

        self.pore_cluster = np.empty(len(self.pn.pore.all), np.int64)
        self.throat_cluster = np.empty(len(self.pn.throat.all), np.int64)

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

    def __delete_connections(self, element_type, element_id):
        if element_type == PORE:
            previous_cluster = self.pore_cluster[element_id]
            self.pore_cluster[element_id] = -1

            pore_to_scan = self.pore_cluster == previous_cluster
            pores_to_find = np.where((self.pn.pore.inlets | self.pn.pore.outlets) & pore_to_scan)[0]
            throat_to_scan = self.throat_cluster == previous_cluster
            throats_to_find = np.array([])

            self.max_cluster_id = reorder_clusters(
                element_id,
                self.pn.pore.conns,
                self.pn.pore.n_conns,
                pore_to_scan,
                self.pore_cluster,
                pores_to_find,
                self.pn.throat.conns,
                self.pn.throat.n_conns,
                throat_to_scan,
                self.throat_cluster,
                throats_to_find,
                self.max_cluster_id,
            )
        else:
            previous_cluster = self.throat_cluster[element_id]
            self.throat_cluster[element_id] = -1

            pore_to_scan = self.pore_cluster == previous_cluster
            pores_to_find = np.where((self.pn.pore.inlets | self.pn.pore.outlets) & pore_to_scan)[0]
            throat_to_scan = self.throat_cluster == previous_cluster
            throats_to_find = np.array([])

            self.max_cluster_id = reorder_clusters(
                element_id,
                self.pn.throat.conns,
                self.pn.throat.n_conns,
                throat_to_scan,
                self.throat_cluster,
                throats_to_find,
                self.pn.pore.conns,
                self.pn.pore.n_conns,
                pore_to_scan,
                self.pore_cluster,
                pores_to_find,
                self.max_cluster_id,
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
            self.pore_cluster,
            throat_has_phase,
            self.pn.throat.conns,
            self.throat_cluster,
            self.pn.pore.inlets,
            self.pn.pore.outlets,
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
            self.pore_cluster,
            self.throat_cluster,
            self.pn.pore.inlets,
            self.pn.pore.outlets,
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
        pore_cluster,
        throat_has_phase,
        throat_conns,
        throat_cluster,
        pore_inlets,
        pore_outlets,
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
            pore_cluster,
            pore_conns,
            pore_n_conns,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            throat_cluster,
            throat_conns,
            pore_inlets,
            pore_outlets,
            pore_id_list,
            throat_id_list,
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
        pore_cluster,
        throat_cluster,
        pore_inlets,
        pore_outlets,
    ):
        cluster_refresh_union(
            parent_array,
            size_array,
            nT,
            pore_has_phase,
            throat_has_phase,
            throat_conns,
        )
        cluster_refresh_connections(
            parent_array,
            nP,
            nT,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            pore_cluster,
            throat_cluster,
            pore_inlets,
            pore_outlets,
        )


@nb.njit(cache=True)
def cluster_refresh_union(
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
def cluster_refresh_connections(
    parent_array,
    nP,
    nT,
    pore_connected_to_inlet,
    pore_connected_to_outlet,
    throat_connected_to_inlet,
    throat_connected_to_outlet,
    pore_cluster,
    throat_cluster,
    pore_inlets,
    pore_outlets,
):
    find = lambda p: wqupc_find(p, parent_array)

    inlets_clusters = set()
    outlets_clusters = set()

    for pore in range(nP):
        if pore_inlets[pore] | pore_outlets[pore]:
            cluster = find(pore + nT)
            if pore_inlets[pore]:
                inlets_clusters.add(cluster)
            if pore_outlets[pore]:
                outlets_clusters.add(cluster)

    for throat in range(nT):
        cluster = find(throat)
        throat_cluster[throat] = cluster
        throat_connected_to_inlet[throat] = cluster in inlets_clusters
        throat_connected_to_outlet[throat] = cluster in outlets_clusters

    for pore in range(nP):
        cluster = find(pore + nT)
        pore_cluster[pore] = cluster
        pore_connected_to_inlet[pore] = cluster in inlets_clusters
        pore_connected_to_outlet[pore] = cluster in outlets_clusters


@nb.njit(cache=True)
def cluster_update_connections(
    parent_array,
    nP,
    nT,
    pore_connected_to_inlet,
    pore_connected_to_outlet,
    pore_cluster,
    pore_conns,
    pore_n_conns,
    throat_connected_to_inlet,
    throat_connected_to_outlet,
    throat_cluster,
    throat_conns,
    pore_inlets,
    pore_outlets,
    pore_id_list,
    throat_id_list,
):
    find = lambda p: wqupc_find(p, parent_array)

    inlets_clusters = set()
    outlets_clusters = set()

    for pore in range(nP):
        if pore_inlets[pore] | pore_outlets[pore]:
            cluster = find(pore + nT)
            if pore_inlets[pore]:
                inlets_clusters.add(cluster)
            if pore_outlets[pore]:
                outlets_clusters.add(cluster)

    for pore_id in pore_id_list:
        update_element_connection(
            parent_array,
            nT,
            0,
            pore_id,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            pore_cluster,
            pore_conns,
            pore_n_conns,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            throat_cluster,
            inlets_clusters,
            outlets_clusters,
        )
    for throat_id in throat_id_list:
        update_element_connection(
            parent_array,
            0,
            nT,
            throat_id,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            throat_cluster,
            throat_conns,
            np.full(len(throat_conns), 2, np.int64),
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            pore_cluster,
            inlets_clusters,
            outlets_clusters,
        )


@nb.njit(cache=True)
def update_element_connection(
    wqupc_parent_array,
    wqupc_element_id_offset,
    wqupc_neighbor_id_offset,
    changed_element_id,
    element_connected_to_inlet,
    element_connected_to_outlet,
    element_cluster,
    element_conns,
    element_n_conns,
    neighbor_connected_to_inlet,
    neighbor_connected_to_outlet,
    neighbor_cluster,
    inlets_clusters,
    outlets_clusters,
):
    find = lambda p: wqupc_find(p, wqupc_parent_array)

    cluster = find(changed_element_id + wqupc_element_id_offset)
    element_cluster[changed_element_id] = cluster

    cluster_has_inlets = cluster in inlets_clusters
    cluster_has_outlets = cluster in outlets_clusters

    for element_id in range(len(element_cluster)):
        if element_cluster[element_id] == cluster:
            element_connected_to_inlet[element_id] = cluster_has_inlets
            element_connected_to_outlet[element_id] = cluster_has_outlets
    for neighbor_id in range(len(neighbor_cluster)):
        if neighbor_cluster[neighbor_id] == cluster:
            neighbor_connected_to_inlet[neighbor_id] = cluster_has_inlets
            neighbor_connected_to_outlet[neighbor_id] = cluster_has_outlets

    for neighbor_id in element_conns[changed_element_id][: element_n_conns[changed_element_id]]:
        old_cluster = neighbor_cluster[neighbor_id]
        new_cluster = find(neighbor_id + wqupc_neighbor_id_offset)
        if new_cluster != old_cluster:
            cluster_has_inlets = new_cluster in inlets_clusters
            cluster_has_outlets = new_cluster in outlets_clusters
            for element_id in range(len(element_cluster)):
                if element_cluster[element_id] == old_cluster:
                    element_cluster[element_id] = new_cluster
                    element_connected_to_inlet[element_id] = cluster_has_inlets
                    element_connected_to_outlet[element_id] = cluster_has_outlets
            for neighbor_id in range(len(neighbor_cluster)):
                if neighbor_cluster[neighbor_id] == old_cluster:
                    neighbor_cluster[neighbor_id] = new_cluster
                    neighbor_connected_to_inlet[neighbor_id] = cluster_has_inlets
                    neighbor_connected_to_outlet[neighbor_id] = cluster_has_outlets


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


def scan_element(
    element_id,
    element_conns,
    element_n_conns,
    element_has_phase,
    element_scanned,
    element_connected,
    elements_to_find,
    neighbor_conns,
    neighbor_n_conns,
    neighbor_has_phase,
    neighbor_scanned,
    neighbor_connected,
    neighbors_to_find,
):
    conns = element_conns[element_id][:element_n_conns]
    for neighbor_id in conns:
        if neighbor_has_phase[neighbor_id] and not neighbor_scanned[neighbor_id]:
            neighbor_scanned[neighbor_id] = True

            if neighbor_connected[neighbor_id] or (
                len(neighbors_to_find) > 0 and neighbor_scanned[neighbors_to_find].all()
            ):
                neighbor_connected[neighbor_scanned] = True
                finished_cluster = True
            else:
                finished_cluster = scan_element(
                    neighbor_id,
                    neighbor_conns,
                    neighbor_n_conns,
                    neighbor_has_phase,
                    neighbor_scanned,
                    neighbor_connected,
                    neighbors_to_find,
                    element_conns,
                    element_n_conns,
                    element_has_phase,
                    element_scanned,
                    element_connected,
                    elements_to_find,
                )

            if finished_cluster:
                return True
    return False


def reorder_clusters(
    element_id,
    element_conns,
    element_n_conns,
    element_to_scan,
    element_cluster,
    elements_to_find,
    neighbor_conns,
    neighbor_n_conns,
    neighbor_to_scan,
    neighbor_cluster,
    neighbors_to_find,
    max_cluster_id,
):
    number_of_elements = len(element_to_scan)
    number_of_neighbors = len(neighbor_to_scan)

    element_scanned_general = np.full(number_of_elements, False, np.bool_)
    element_connected = np.full(number_of_elements, False, np.bool_)
    neighbor_scanned_general = np.full(number_of_neighbors, False, np.bool_)
    neighbor_connected = np.full(number_of_neighbors, False, np.bool_)

    conns = element_conns[element_id][:element_n_conns]
    for neighbor_id in conns:
        if not neighbor_scanned_general[neighbor_id]:
            element_scanned = np.full(number_of_elements, False, np.bool_)
            neighbor_scanned = np.full(number_of_neighbors, False, np.bool_)
            found = scan_element(
                neighbor_id,
                neighbor_conns,
                neighbor_n_conns,
                neighbor_to_scan,
                neighbor_scanned,
                neighbor_connected,
                neighbors_to_find,
                element_conns,
                element_n_conns,
                element_to_scan,
                element_scanned,
                element_connected,
                elements_to_find,
            )
            if not found:
                if element_scanned[elements_to_find].any() or neighbor_scanned[neighbors_to_find].any():
                    max_cluster_id += 1
                    element_cluster[element_scanned] = max_cluster_id
                    neighbor_cluster[neighbor_scanned] = max_cluster_id
                else:
                    element_cluster[element_scanned] = -1
                    neighbor_cluster[neighbor_scanned] = -1
            element_scanned_general |= element_scanned
            neighbor_scanned_general |= neighbor_scanned
    return max_cluster_id
