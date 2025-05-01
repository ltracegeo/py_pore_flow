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

        if self.phase == WATER:
            water_connection = PhaseConnectionChecker2(
                self.network.pn_core,
                WATER,
            )
            oil_connection = ReductiblePhaseConnectionChecker2(
                self.network.pn_core,
                OIL,
            )
            self.invasion_connections = water_connection
            self.evasion_connections = oil_connection
        else:
            water_connection = ReductiblePhaseConnectionChecker2(
                self.network.pn_core,
                WATER,
            )
            oil_connection = PhaseConnectionChecker2(
                self.network.pn_core,
                OIL,
            )
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
        self.evasion_connections.update_connections(pore_id_list, throat_id_list)

        water_connected_inlet = (
            self.network.pn_core.pore.water_connected_to_inlet.sum()
            + self.network.pn_core.throat.water_connected_to_inlet.sum()
        )
        water_connected_water = (
            self.network.pn_core.pore.water_connected_to_outlet.sum()
            + self.network.pn_core.throat.water_connected_to_outlet.sum()
        )
        oil_connected_inlet = (
            self.network.pn_core.pore.oil_connected_to_inlet.sum()
            + self.network.pn_core.throat.oil_connected_to_inlet.sum()
        )
        oil_connected_oil = (
            self.network.pn_core.pore.oil_connected_to_outlet.sum()
            + self.network.pn_core.throat.oil_connected_to_outlet.sum()
        )

        from collections import Counter

        counter = Counter(self.evasion_connections.pore_cluster)
        pores_filtered = {k: v for k, v in counter.items() if v > 1}
        counter = Counter(self.evasion_connections.throat_cluster)
        throats_filtered = {k: v for k, v in counter.items() if v > 1}

        # inlets_clusters, outlets_clusters = get_inlet_outlet_clusters(self.evasion_connections.pore_cluster, self.network.pn_core.pore.inlets, self.network.pn_core.pore.outlets)
        # print("inlets_clusters", inlets_clusters, "outlets_clusters", outlets_clusters, pores_filtered, throats_filtered)

        # print(
        #     f"water(in{water_connected_inlet},out{water_connected_water})|oil(in{oil_connected_inlet},out{oil_connected_oil}) |",
        #     pores_filtered,
        #     throats_filtered,
        # )
        # print(pores_filtered, throats_filtered)


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
        jit_reset_connections(
            self.wqupc.parent_array,
            self.wqupc.size_array,
            self.wqupc.nT,
            np.array(list(range(self.wqupc.nP))),
            np.array(list(range(self.wqupc.nT))),
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


@nb.njit(cache=True)
def jit_reset_connections(
    parent_array,
    size_array,
    nT,
    pore_ids,
    throat_ids,
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
        throat_ids,
        pore_has_phase,
        throat_has_phase,
        throat_conns,
    )
    cluster_refresh_connections(
        parent_array,
        nT,
        pore_ids,
        throat_ids,
        pore_connected_to_inlet,
        pore_connected_to_outlet,
        throat_connected_to_inlet,
        throat_connected_to_outlet,
        pore_cluster,
        throat_cluster,
        pore_inlets,
        pore_outlets,
    )


class ReductiblePhaseConnectionChecker2(PhaseConnectionChecker2):
    def __init__(self, pn, phase):
        super().__init__(pn, phase)

        self.pore_old_cluster = None
        self.throat_old_cluster = None
        self.max_cluster_id = None

    def reset_connections(self):
        if self.phase == WATER:
            self.pore_has_phase_old = self.pn.pore.water_saturation > 0.0 + TRAP_TOLERANCE
            self.throat_has_phase_old = self.pn.throat.water_saturation > 0.0 + TRAP_TOLERANCE
        else:
            self.pore_has_phase_old = self.pn.pore.oil_saturation > 0.0 + TRAP_TOLERANCE
            self.throat_has_phase_old = self.pn.throat.oil_saturation > 0.0 + TRAP_TOLERANCE
        super().reset_connections()

        self.max_cluster_id = max(self.pore_cluster.max(), self.throat_cluster.max())

    def update_connections(self, pore_id_list, throat_id_list):
        if False:
            super().reset_connections()
        else:
            # if len(pore_id_list) > 0:
            #     self.reverse_connections(PORE, pore_id_list[0])
            # elif len(throat_id_list) > 0:
            #     self.reverse_connections(THROAT, throat_id_list[0])
            self.reverse_connections()

    def reverse_connections(self):
        if self.phase == WATER:
            pore_has_phase = self.pn.pore.water_saturation > 0.0 + TRAP_TOLERANCE
            throat_has_phase = self.pn.throat.water_saturation > 0.0 + TRAP_TOLERANCE
        else:
            pore_has_phase = self.pn.pore.oil_saturation > 0.0 + TRAP_TOLERANCE
            throat_has_phase = self.pn.throat.oil_saturation > 0.0 + TRAP_TOLERANCE
        pore_has_phase_diff = self.pore_has_phase_old & ~pore_has_phase
        throat_has_phase_diff = self.throat_has_phase_old & ~throat_has_phase
        self.pore_has_phase_old = pore_has_phase.copy()
        self.throat_has_phase_old = throat_has_phase.copy()

        # self.pore_old_cluster = self.pore_cluster.copy()
        # self.throat_old_cluster = self.throat_cluster.copy()
        # for pore_id in np.where(pore_has_phase_diff)[0]:
        #     self.__delete_connections(PORE, pore_id)
        # for throat_id in np.where(throat_has_phase_diff)[0]:
        #     self.__delete_connections(THROAT, throat_id)
        self.__delete_connections(np.where(pore_has_phase_diff)[0], np.where(throat_has_phase_diff)[0])

        # self.__update_element_connection()

    def __delete_connections(self, pores_changed, throats_changed):
        if self.phase == WATER:
            self.max_cluster_id = self.__njit_delete_connections(
                pores_changed,
                throats_changed,
                self.pore_cluster,
                self.throat_cluster,
                self.pn.pore.water_connected_to_inlet,
                self.pn.pore.water_connected_to_outlet,
                self.pn.throat.water_connected_to_inlet,
                self.pn.throat.water_connected_to_outlet,
                self.pn.throat.conns,
                self.pn.pore.inlets,
                self.pn.pore.outlets,
                self.max_cluster_id,
            )
        else:
            self.max_cluster_id = self.__njit_delete_connections(
                pores_changed,
                throats_changed,
                self.pore_cluster,
                self.throat_cluster,
                self.pn.pore.oil_connected_to_inlet,
                self.pn.pore.oil_connected_to_outlet,
                self.pn.throat.oil_connected_to_inlet,
                self.pn.throat.oil_connected_to_outlet,
                self.pn.throat.conns,
                self.pn.pore.inlets,
                self.pn.pore.outlets,
                self.max_cluster_id,
            )

    @staticmethod
    @nb.njit(cache=True)
    def __njit_delete_connections(
        pores_changed,
        throats_changed,
        pore_cluster,
        throat_cluster,
        pore_connected_to_inlet,
        pore_connected_to_outlet,
        throat_connected_to_inlet,
        throat_connected_to_outlet,
        throat_conns,
        pore_inlets,
        pore_outlets,
        max_cluster_id,
    ):
        affected_clusters = []
        for pore_id in pores_changed:
            affected_clusters.append(pore_cluster[pore_id])
            remove_from_clusters(pore_cluster, pore_id)
        for throat_id in throats_changed:
            affected_clusters.append(throat_cluster[throat_id])
            remove_from_clusters(throat_cluster, throat_id)

        affected_clusters = np.array(list(set(affected_clusters)))
        pore_to_scan = njit_isin(pore_cluster, affected_clusters)
        throat_to_scan = njit_isin(throat_cluster, affected_clusters)
        pore_to_scan_ids = np.where(pore_to_scan)[0]
        throat_to_scan_ids = np.where(throat_to_scan)[0]

        wqupc_nP = len(pore_to_scan)
        wqupc_nT = len(throat_to_scan)
        wqupc_parent_array = np.arange(wqupc_nP + wqupc_nT, dtype=np.uint32)
        wqupc_size_array = np.ones((wqupc_nP + wqupc_nT))

        pore_cluster_temp = np.empty(wqupc_nP, dtype=np.int64)
        throat_cluster_temp = np.empty(wqupc_nT, dtype=np.int64)

        jit_reset_connections(
            wqupc_parent_array,
            wqupc_size_array,
            wqupc_nT,
            pore_to_scan_ids,
            throat_to_scan_ids,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            pore_to_scan,
            throat_to_scan,
            throat_conns,
            pore_cluster_temp,
            throat_cluster_temp,
            pore_inlets,
            pore_outlets,
        )
        new_pore_clusters = np.array(list(set(pore_cluster_temp[pore_to_scan])))
        new_throat_clusters = np.array(list(set(throat_cluster_temp[throat_to_scan])))
        new_clusters = np.array(list(set(np.concatenate((new_pore_clusters, new_throat_clusters)))))
        for i in range(len(new_clusters)):
            if i < len(affected_clusters):
                new_cluster_id = affected_clusters[i]
            else:
                max_cluster_id += 1
                new_cluster_id = max_cluster_id
            pore_cluster[pore_to_scan][pore_cluster_temp[pore_to_scan] == new_clusters[i]] = new_cluster_id
            throat_cluster[throat_to_scan][throat_cluster_temp[throat_to_scan] == new_clusters[i]] = new_cluster_id

        return max_cluster_id

    def __update_element_connection(self):
        if self.phase == WATER:
            self.__njit_update_element_connection(
                self.pn.pore.water_connected_to_inlet,
                self.pn.pore.water_connected_to_outlet,
                self.pore_old_cluster,
                self.pore_cluster,
                self.pn.throat.water_connected_to_inlet,
                self.pn.throat.water_connected_to_outlet,
                self.throat_old_cluster,
                self.throat_cluster,
                self.pn.pore.inlets,
                self.pn.pore.outlets,
            )
        else:
            self.__njit_update_element_connection(
                self.pn.pore.oil_connected_to_inlet,
                self.pn.pore.oil_connected_to_outlet,
                self.pore_old_cluster,
                self.pore_cluster,
                self.pn.throat.oil_connected_to_inlet,
                self.pn.throat.oil_connected_to_outlet,
                self.throat_old_cluster,
                self.throat_cluster,
                self.pn.pore.inlets,
                self.pn.pore.outlets,
            )

    @staticmethod
    @nb.njit(cache=True)
    def __njit_update_element_connection(
        pore_connected_to_inlet,
        pore_connected_to_outlet,
        pore_old_cluster,
        pore_cluster,
        throat_connected_to_inlet,
        throat_connected_to_outlet,
        throat_old_cluster,
        throat_cluster,
        pore_inlets,
        pore_outlets,
    ):
        modified_clusters = get_modified_clusters(pore_old_cluster, throat_old_cluster, pore_cluster, throat_cluster)

        # removed_clusters = modified_clusters[modified_clusters < 0]
        # removed_pores = isin_indexes(pore_cluster, removed_clusters)
        # removed_throats = isin_indexes(throat_cluster, removed_clusters)
        # for pore_id in removed_pores:
        #     pore_connected_to_inlet[pore_id] = False
        #     pore_connected_to_outlet[pore_id] = False
        # for throat_id in removed_throats:
        #     throat_connected_to_inlet[throat_id] = False
        #     throat_connected_to_outlet[throat_id] = False

        # modified_clusters = modified_clusters[modified_clusters > 0]

        # inlets_clusters = set()
        # outlets_clusters = set()

        # for pore_id in range(len(pore_connected_to_inlet)):
        #     if pore_inlets[pore_id] | pore_outlets[pore_id]:
        #         cluster = pore_cluster[pore_id]
        #         if pore_inlets[pore_id]:
        #             inlets_clusters.add(cluster)
        #         if pore_outlets[pore_id]:
        #             outlets_clusters.add(cluster)
        inlets_clusters, outlets_clusters = get_inlet_outlet_clusters(pore_cluster, pore_inlets, pore_outlets)

        non_wqupc_update_element_connection(
            modified_clusters,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            pore_cluster,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            throat_cluster,
            inlets_clusters,
            outlets_clusters,
        )


@nb.njit(cache=True)
def cluster_refresh_union(
    parent_array,
    size_array,
    nT,
    throat_ids,
    pore_has_phase,
    throat_has_phase,
    throat_conns,
):
    union = lambda p, q: wqupc_union(p, q, parent_array, size_array)

    for throat in throat_ids:
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
    nT,
    pore_ids,
    throat_ids,
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

    for pore in pore_ids:
        if pore_inlets[pore] | pore_outlets[pore]:
            cluster = find(pore + nT)
            if pore_inlets[pore]:
                inlets_clusters.add(cluster)
            if pore_outlets[pore]:
                outlets_clusters.add(cluster)

    for throat in throat_ids:
        cluster = find(throat)
        throat_cluster[throat] = cluster
        throat_connected_to_inlet[throat] = cluster in inlets_clusters
        throat_connected_to_outlet[throat] = cluster in outlets_clusters

    for pore in pore_ids:
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


@nb.njit(
    nb.bool_(
        nb.int32,
        nb.int32[:, :],
        nb.int32[:],
        nb.bool_[:],
        nb.bool_[:],
        nb.bool_[:],
        nb.int32[:],
        nb.int32[:, :],
        nb.int32[:],
        nb.bool_[:],
        nb.bool_[:],
        nb.bool_[:],
        nb.int32[:],
    ),
    cache=True,
)
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
    if element_has_phase[element_id] and not element_scanned[element_id]:
        element_scanned[element_id] = True
        if element_connected[element_id] or (len(elements_to_find) > 0 and element_scanned[elements_to_find].all()):
            element_connected[element_scanned] = True
            finished_cluster = True
        else:
            conns = element_conns[element_id][: element_n_conns[element_id]]
            for neighbor_id in conns:
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
                    break
    return finished_cluster

    # conns = element_conns[element_id][: element_n_conns[element_id]]
    # for neighbor_id in conns:
    #     if neighbor_has_phase[neighbor_id] and not neighbor_scanned[neighbor_id]:
    #         neighbor_scanned[neighbor_id] = True

    #         if neighbor_connected[neighbor_id] or (
    #             len(neighbors_to_find) > 0 and neighbor_scanned[neighbors_to_find].all()
    #         ):
    #             neighbor_connected[neighbor_scanned] = True
    #             finished_cluster = True
    #         else:
    #             finished_cluster = scan_element(
    #                 neighbor_id,
    #                 neighbor_conns,
    #                 neighbor_n_conns,
    #                 neighbor_has_phase,
    #                 neighbor_scanned,
    #                 neighbor_connected,
    #                 neighbors_to_find,
    #                 element_conns,
    #                 element_n_conns,
    #                 element_has_phase,
    #                 element_scanned,
    #                 element_connected,
    #                 elements_to_find,
    #             )

    #         if finished_cluster:
    #             return True
    # return False


@nb.njit(cache=True)
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

    conns = element_conns[element_id][: element_n_conns[element_id]]
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
                    # element_cluster[element_scanned] = -1
                    remove_from_clusters(element_cluster, np.where(element_scanned)[0])
                    # neighbor_cluster[neighbor_scanned] = -1
                    remove_from_clusters(neighbor_cluster, np.where(neighbor_scanned)[0])
            element_scanned_general |= element_scanned
            neighbor_scanned_general |= neighbor_scanned
    return max_cluster_id


@nb.njit(cache=True)
def non_wqupc_update_element_connection(
    modified_clusters,
    element_connected_to_inlet,
    element_connected_to_outlet,
    element_cluster,
    neighbor_connected_to_inlet,
    neighbor_connected_to_outlet,
    neighbor_cluster,
    inlets_clusters,
    outlets_clusters,
):
    for cluster in modified_clusters:
        element_in_changed_cluster = np.where(element_cluster == cluster)[0]
        neighbor_in_changed_cluster = np.where(neighbor_cluster == cluster)[0]

        inlets_in_cluster = cluster in inlets_clusters
        outlets_in_cluster = cluster in outlets_clusters

        element_connected_to_inlet[element_in_changed_cluster] = inlets_in_cluster
        element_connected_to_outlet[element_in_changed_cluster] = outlets_in_cluster

        neighbor_connected_to_inlet[neighbor_in_changed_cluster] = inlets_in_cluster
        neighbor_connected_to_outlet[neighbor_in_changed_cluster] = outlets_in_cluster


@nb.njit(cache=True)
def get_modified_values(original, modified):
    modified_values = set()

    for original_value, modified_value in zip(original.flatten(), modified.flatten()):
        if original_value != modified_value:
            modified_values.add(original_value)
            modified_values.add(modified_value)

    return modified_values


@nb.njit(cache=True)
def get_modified_clusters(pore_cluster_old, throat_cluster_old, pore_cluster, throat_cluster):
    modified_pore_clusters = get_modified_values(pore_cluster_old, pore_cluster)
    modified_throat_clusters = get_modified_values(throat_cluster_old, throat_cluster)
    return np.array(list(modified_pore_clusters | modified_throat_clusters), np.int64)


@nb.njit(cache=True)
def get_inlet_outlet_clusters(pore_cluster, pore_inlets, pore_outlets):
    inlets_clusters = set()
    outlets_clusters = set()

    for pore_id in range(len(pore_cluster)):
        if pore_inlets[pore_id] | pore_outlets[pore_id]:
            cluster = pore_cluster[pore_id]
            if pore_inlets[pore_id]:
                inlets_clusters.add(cluster)
            if pore_outlets[pore_id]:
                outlets_clusters.add(cluster)

    return inlets_clusters, outlets_clusters


@nb.njit(cache=True)
def remove_from_clusters(element_cluster_array, element_id_list):
    element_cluster_array[element_id_list] = -element_id_list


@nb.njit(cache=True)
def isin_indexes(array_a, array_b):
    result = nb.typed.List.empty_list(nb.int64)
    for i in range(array_a.size):
        for j in range(array_b.size):
            if array_a[i] == array_b[j]:
                result.append(i)
                break
    return result


@nb.njit(cache=True)
def njit_isin(array_a, array_b):
    result = np.full(array_a.size, False, nb.bool_)
    for i in range(array_a.size):
        for j in range(array_b.size):
            if array_a[i] == array_b[j]:
                result[i] = True
                break
    return result
