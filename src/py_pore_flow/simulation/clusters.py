"""Phase connectivity analysis for two-phase pore network flow.

Tracks which pores and throats are connected to inlet/outlet boundaries for
each fluid phase (oil/water).  This is essential for computing relative
permeability: only elements belonging to a connected cluster that spans
inlet-to-outlet contribute to flow.

Two checker variants exist:

* **AdditiveClusterTracker** — used for the *invading* phase.  New elements
  are added to the union-find incrementally via ``update_connections``.

* **ReductiveClusterTracker** — used for the *defending* phase.
  When the invading phase takes over an element, the defending phase loses it.
  Losing an element can split a cluster, so this checker detects which clusters
  are affected, re-clusters only those elements, and propagates the new
  inlet/outlet connectivity.

Both are wrapped by the top-level **PhaseConnectionChecker**, which pairs an
invading tracker with a defending tracker depending on the active cycle phase.

The heavy lifting is done in numba-jitted free functions to keep the inner
loops at native speed.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from py_pore_flow.data_structures.wqupc import Wqupc, wqupc_find, wqupc_union
from py_pore_flow.equations.geometry import OIL, WATER

# Element type constants (used as discriminator in update calls).
PORE: int = 0
THROAT: int = 1


@nb.njit(cache=True)
def _extract_inlet_outlet_ids(
    pore_inlets: NDArray[np.bool_],
    pore_outlets: NDArray[np.bool_],
) -> tuple:
    """Extract inlet and outlet pore IDs from boolean masks in a single pass."""
    inlet_count = 0
    outlet_count = 0
    for i in range(len(pore_inlets)):
        inlet_count += pore_inlets[i]
        outlet_count += pore_outlets[i]
    inlet_ids = np.empty(inlet_count, np.int64)
    outlet_ids = np.empty(outlet_count, np.int64)
    ii = 0
    oi = 0
    for i in range(len(pore_inlets)):
        if pore_inlets[i]:
            inlet_ids[ii] = i
            ii += 1
        if pore_outlets[i]:
            outlet_ids[oi] = i
            oi += 1
    return inlet_ids, outlet_ids


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


class PhaseConnectionChecker:
    """Pairs an invading-phase and a defending-phase connection checker.

    During drainage the invading phase is oil and the defending phase is water;
    during imbibition it is the reverse.
    """

    def __init__(self, network: object, phase: int) -> None:
        self.network = network
        self.phase: int = phase

        if self.phase == WATER:
            self.invasion_connections: AdditiveClusterTracker = AdditiveClusterTracker(
                self.network.pn_core, WATER
            )
            self.evasion_connections: ReductiveClusterTracker = ReductiveClusterTracker(
                self.network.pn_core, OIL
            )
        else:
            self.invasion_connections = AdditiveClusterTracker(self.network.pn_core, OIL)
            self.evasion_connections = ReductiveClusterTracker(self.network.pn_core, WATER)

    def reset_phase_connections(self) -> None:
        """Rebuild both checkers from scratch using current network state."""
        self.invasion_connections.reset_connections()
        self.evasion_connections.reset_connections()

    def update_phase_connections(self, element_type: int, element_id: int) -> None:
        """Incrementally update both checkers after a single element changed phase."""
        if element_type == PORE:
            pore_id_list = np.array([element_id], dtype=np.int64)
            throat_id_list = np.empty(0, dtype=np.int64)
        else:
            pore_id_list = np.empty(0, dtype=np.int64)
            throat_id_list = np.array([element_id], dtype=np.int64)

        self.invasion_connections.update_connections(pore_id_list, throat_id_list)
        self.evasion_connections.update_connections(pore_id_list, throat_id_list)


# ---------------------------------------------------------------------------
# Invading-phase connection checker (additive only)
# ---------------------------------------------------------------------------


class AdditiveClusterTracker:
    """Tracks connected clusters for a single phase using WQUPC.

    New elements are merged into existing clusters incrementally.  A full
    reset rebuilds the union-find from scratch.
    """

    def __init__(self, pn: object, phase: int) -> None:
        self.pn = pn
        self.phase: int = phase

        self.wqupc: Wqupc = Wqupc(len(self.pn.pore.all), len(self.pn.throat.all))

        self.pore_cluster: NDArray[np.int64] = np.empty(len(self.pn.pore.all), np.int64)
        self.throat_cluster: NDArray[np.int64] = np.empty(len(self.pn.throat.all), np.int64)

        # Pre-computed inlet/outlet pore IDs (fixed for the entire cycle).
        self.inlet_pore_ids, self.outlet_pore_ids = _extract_inlet_outlet_ids(
            self.pn.pore.inlets, self.pn.pore.outlets
        )

    # -- Phase-aware property helpers ---------------------------------------

    def _phase_masks(self) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Return (pore_has_phase, throat_has_phase) masks for the tracked phase."""
        if self.phase == WATER:
            return (
                self.pn.pore.is_multiphase | self.pn.pore.water_filled,
                self.pn.throat.is_multiphase | self.pn.throat.water_filled,
            )
        return (
            self.pn.pore.is_multiphase | self.pn.pore.oil_filled,
            self.pn.throat.is_multiphase | self.pn.throat.oil_filled,
        )

    def _connection_arrays(
        self,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        """Return (pore_inlet, pore_outlet, throat_inlet, throat_outlet) connectivity arrays."""
        if self.phase == WATER:
            return (
                self.pn.pore.water_connected_to_inlet,
                self.pn.pore.water_connected_to_outlet,
                self.pn.throat.water_connected_to_inlet,
                self.pn.throat.water_connected_to_outlet,
            )
        return (
            self.pn.pore.oil_connected_to_inlet,
            self.pn.pore.oil_connected_to_outlet,
            self.pn.throat.oil_connected_to_inlet,
            self.pn.throat.oil_connected_to_outlet,
        )

    # -- Public API ---------------------------------------------------------

    def reset_connections(self) -> None:
        """Rebuild union-find and connectivity from current network state."""
        pore_conn_in, pore_conn_out, throat_conn_in, throat_conn_out = self._connection_arrays()
        pore_has_phase, throat_has_phase = self._phase_masks()

        self.wqupc.reset()
        _jit_reset_connections(
            self.wqupc.parent_array,
            self.wqupc.size_array,
            self.wqupc.nT,
            np.arange(self.wqupc.nP, dtype=np.int64),
            np.arange(self.wqupc.nT, dtype=np.int64),
            pore_conn_in,
            pore_conn_out,
            throat_conn_in,
            throat_conn_out,
            pore_has_phase,
            throat_has_phase,
            self.pn.throat.conns,
            self.pore_cluster,
            self.throat_cluster,
            self.pn.pore.inlets,
            self.pn.pore.outlets,
        )

    def update_connections(self, pore_id_list: NDArray[np.int64], throat_id_list: NDArray[np.int64]) -> None:
        """Incrementally add newly-invaded elements to the union-find."""
        pore_conn_in, pore_conn_out, throat_conn_in, throat_conn_out = self._connection_arrays()
        pore_has_phase, throat_has_phase = self._phase_masks()

        self._jit_update_connections(
            self.wqupc.parent_array,
            self.wqupc.size_array,
            self.wqupc.nP,
            self.wqupc.nT,
            pore_conn_in,
            pore_conn_out,
            throat_conn_in,
            throat_conn_out,
            pore_has_phase,
            self.pn.pore.conns_values,
            self.pn.pore.conns_offsets,
            self.pn.pore.n_conns,
            self.pore_cluster,
            throat_has_phase,
            self.pn.throat.conns_values,
            self.pn.throat.conns_offsets,
            self.throat_cluster,
            self.pn.pore.inlets,
            self.pn.pore.outlets,
            pore_id_list,
            throat_id_list,
            self.inlet_pore_ids,
            self.outlet_pore_ids,
        )

    # -- JIT kernel ---------------------------------------------------------

    @staticmethod
    @nb.njit(cache=True)
    def _jit_update_connections(
        parent_array: NDArray[np.uint32],
        size_array: NDArray[np.float64],
        nP: int,
        nT: int,
        pore_connected_to_inlet: NDArray[np.bool_],
        pore_connected_to_outlet: NDArray[np.bool_],
        throat_connected_to_inlet: NDArray[np.bool_],
        throat_connected_to_outlet: NDArray[np.bool_],
        pore_has_phase: NDArray[np.bool_],
        pore_conns_values: NDArray[np.int64],
        pore_conns_offsets: NDArray[np.int64],
        pore_n_conns: NDArray[np.int64],
        pore_cluster: NDArray[np.int64],
        throat_has_phase: NDArray[np.bool_],
        throat_conns_values: NDArray[np.int64],
        throat_conns_offsets: NDArray[np.int64],
        throat_cluster: NDArray[np.int64],
        pore_inlets: NDArray[np.bool_],
        pore_outlets: NDArray[np.bool_],
        pore_id_list: NDArray[np.int64],
        throat_id_list: NDArray[np.int64],
        inlet_pore_ids: NDArray[np.int64],
        outlet_pore_ids: NDArray[np.int64],
    ) -> None:
        # Step 1: merge newly-invaded elements into union-find.
        _cluster_add_union(
            parent_array,
            size_array,
            nT,
            pore_has_phase,
            pore_conns_values,
            pore_conns_offsets,
            pore_n_conns,
            throat_has_phase,
            throat_conns_values,
            throat_conns_offsets,
            pore_id_list,
            throat_id_list,
        )
        # Step 2: propagate inlet/outlet reachability through the new clusters.
        _cluster_update_connections(
            parent_array,
            nT,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            pore_cluster,
            pore_conns_values,
            pore_conns_offsets,
            pore_n_conns,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            throat_cluster,
            throat_conns_values,
            throat_conns_offsets,
            pore_inlets,
            pore_outlets,
            pore_id_list,
            throat_id_list,
            inlet_pore_ids,
            outlet_pore_ids,
        )


# ---------------------------------------------------------------------------
# Defending-phase connection checker (supports element removal / cluster split)
# ---------------------------------------------------------------------------


class ReductiveClusterTracker(AdditiveClusterTracker):
    """Connection checker that handles element *removal* (cluster splitting).

    When the opposing phase invades an element, this checker detects which
    clusters lost members, re-runs union-find on just those clusters, and
    assigns new cluster IDs where splits occurred.

    Uses linked-list arrays (per-cluster element chains) so that the set of
    elements in a given cluster can be traversed in O(cluster_size) without
    scanning the entire network.
    """

    def __init__(self, pn: object, phase: int) -> None:
        super().__init__(pn, phase)

        nP: int = len(pn.pore.all)
        nT: int = len(pn.throat.all)

        # Cluster membership linked lists: cluster_id -> head element, then
        # element -> next element in the same cluster.
        max_clusters: int = 3 * (nP + nT)
        self.pore_cluster_next: NDArray[np.int64] = np.empty(nP, dtype=np.int64)
        self.throat_cluster_next: NDArray[np.int64] = np.empty(nT, dtype=np.int64)
        self.cluster_pore_head: NDArray[np.int64] = np.full(max_clusters, -1, dtype=np.int64)
        self.cluster_throat_head: NDArray[np.int64] = np.full(max_clusters, -1, dtype=np.int64)

        # Pre-allocated work buffers for _njit_delete_connections (avoids
        # repeated allocation inside the hot jitted path).
        self._work_pore_scan: NDArray[np.bool_] = np.zeros(nP, dtype=np.bool_)
        self._work_throat_scan: NDArray[np.bool_] = np.zeros(nT, dtype=np.bool_)
        self._work_pore_ids: NDArray[np.int64] = np.empty(nP, dtype=np.int64)
        self._work_throat_ids: NDArray[np.int64] = np.empty(nT, dtype=np.int64)
        self._work_wqupc_parent: NDArray[np.uint32] = np.empty(nP + nT, dtype=np.uint32)
        self._work_wqupc_size: NDArray[np.float64] = np.empty(nP + nT, dtype=np.float64)
        self._work_pore_cluster_temp: NDArray[np.int64] = np.empty(nP, dtype=np.int64)
        self._work_throat_cluster_temp: NDArray[np.int64] = np.empty(nT, dtype=np.int64)

        self.max_cluster_id: int = 0
        self.reset_connections()

    def reset_connections(self) -> None:
        """Full rebuild: recompute clusters and linked lists from scratch."""
        self.pore_has_phase_old: NDArray[np.bool_]
        self.throat_has_phase_old: NDArray[np.bool_]
        pore_mask, throat_mask = self._phase_masks()
        self.pore_has_phase_old = pore_mask.copy()
        self.throat_has_phase_old = throat_mask.copy()

        super().reset_connections()

        self.max_cluster_id = max(self.pore_cluster.max(), self.throat_cluster.max())
        _build_cluster_lists(
            self.pore_cluster,
            self.throat_cluster,
            self.pore_cluster_next,
            self.throat_cluster_next,
            self.cluster_pore_head,
            self.cluster_throat_head,
        )

    def update_connections(self, pore_id_list: NDArray[np.int64], throat_id_list: NDArray[np.int64]) -> None:
        """Detect removed elements and re-cluster the affected subset."""
        self._reverse_connections()

    def _reverse_connections(self) -> None:
        """Compare current phase masks to the previous snapshot, find removed
        elements, and re-cluster the clusters they belonged to."""
        pore_has_phase, throat_has_phase = self._phase_masks()

        pore_has_phase_diff: NDArray[np.bool_] = self.pore_has_phase_old & ~pore_has_phase
        throat_has_phase_diff: NDArray[np.bool_] = self.throat_has_phase_old & ~throat_has_phase
        self.pore_has_phase_old = pore_has_phase.copy()
        self.throat_has_phase_old = throat_has_phase.copy()

        self._delete_connections(np.where(pore_has_phase_diff)[0], np.where(throat_has_phase_diff)[0])

    def _delete_connections(self, pores_changed: NDArray[np.intp], throats_changed: NDArray[np.intp]) -> None:
        """Re-cluster after elements were removed from this phase."""
        pore_conn_in, pore_conn_out, throat_conn_in, throat_conn_out = self._connection_arrays()
        pore_has_phase, throat_has_phase = self._phase_masks()

        self.max_cluster_id = self._njit_delete_connections(
            pores_changed,
            throats_changed,
            self.pore_cluster,
            self.throat_cluster,
            pore_conn_in,
            pore_conn_out,
            throat_conn_in,
            throat_conn_out,
            self.pn.throat.conns,
            self.pn.pore.inlets,
            self.pn.pore.outlets,
            self.max_cluster_id,
            self.pore_cluster_next,
            self.throat_cluster_next,
            self.cluster_pore_head,
            self.cluster_throat_head,
            pore_has_phase,
            throat_has_phase,
            self._work_pore_scan,
            self._work_throat_scan,
            self._work_pore_ids,
            self._work_throat_ids,
            self._work_wqupc_parent,
            self._work_wqupc_size,
            self._work_pore_cluster_temp,
            self._work_throat_cluster_temp,
        )

    @staticmethod
    @nb.njit(cache=True)
    def _njit_delete_connections(
        pores_changed: NDArray[np.intp],
        throats_changed: NDArray[np.intp],
        pore_cluster: NDArray[np.int64],
        throat_cluster: NDArray[np.int64],
        pore_connected_to_inlet: NDArray[np.bool_],
        pore_connected_to_outlet: NDArray[np.bool_],
        throat_connected_to_inlet: NDArray[np.bool_],
        throat_connected_to_outlet: NDArray[np.bool_],
        throat_conns: NDArray[np.int64],
        pore_inlets: NDArray[np.bool_],
        pore_outlets: NDArray[np.bool_],
        max_cluster_id: int,
        pore_cluster_next: NDArray[np.int64],
        throat_cluster_next: NDArray[np.int64],
        cluster_pore_head: NDArray[np.int64],
        cluster_throat_head: NDArray[np.int64],
        pore_has_phase: NDArray[np.bool_],
        throat_has_phase: NDArray[np.bool_],
        pore_to_scan: NDArray[np.bool_],
        throat_to_scan: NDArray[np.bool_],
        pore_to_scan_buf: NDArray[np.int64],
        throat_to_scan_buf: NDArray[np.int64],
        wqupc_parent: NDArray[np.uint32],
        wqupc_size: NDArray[np.float64],
        pore_cluster_temp: NDArray[np.int64],
        throat_cluster_temp: NDArray[np.int64],
    ) -> int:
        if len(pores_changed) == 0 and len(throats_changed) == 0:
            return max_cluster_id

        # Phase 1: Collect affected cluster IDs, mark changed elements.
        affected_set = set()
        for pore_id in pores_changed:
            affected_set.add(pore_cluster[pore_id])
            pore_cluster[pore_id] = -pore_id
        for throat_id in throats_changed:
            affected_set.add(throat_cluster[throat_id])
            throat_cluster[throat_id] = -throat_id

        # Phase 2: Walk linked lists to find remaining members of affected clusters.
        pore_scan_count = np.int64(0)
        throat_scan_count = np.int64(0)

        for c in affected_set:
            pid = cluster_pore_head[c]
            while pid >= 0:
                nxt = pore_cluster_next[pid]
                if pore_cluster[pid] == c:
                    pore_to_scan_buf[pore_scan_count] = pid
                    pore_to_scan[pid] = True
                    pore_scan_count += 1
                pid = nxt
            cluster_pore_head[c] = -1

            tid = cluster_throat_head[c]
            while tid >= 0:
                nxt = throat_cluster_next[tid]
                if throat_cluster[tid] == c:
                    throat_to_scan_buf[throat_scan_count] = tid
                    throat_to_scan[tid] = True
                    throat_scan_count += 1
                tid = nxt
            cluster_throat_head[c] = -1

        pore_scan_ids = pore_to_scan_buf[:pore_scan_count]
        throat_scan_ids = throat_to_scan_buf[:throat_scan_count]

        # Phase 3: Initialize WQUPC only for affected elements.
        nT = len(throat_to_scan)
        for i in range(pore_scan_count):
            idx = pore_scan_ids[i] + nT
            wqupc_parent[idx] = np.uint32(idx)
            wqupc_size[idx] = 1.0
        for i in range(throat_scan_count):
            idx = throat_scan_ids[i]
            wqupc_parent[idx] = np.uint32(idx)
            wqupc_size[idx] = 1.0

        # Phase 4: Re-cluster the affected subset.
        _jit_reset_connections(
            wqupc_parent,
            wqupc_size,
            nT,
            pore_scan_ids,
            throat_scan_ids,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            pore_has_phase,
            throat_has_phase,
            throat_conns,
            pore_cluster_temp,
            throat_cluster_temp,
            pore_inlets,
            pore_outlets,
        )

        # Phase 5: Collect unique temp cluster IDs and map to real IDs.
        temp_cluster_set = set()
        for i in range(pore_scan_count):
            temp_cluster_set.add(pore_cluster_temp[pore_scan_ids[i]])
        for i in range(throat_scan_count):
            temp_cluster_set.add(throat_cluster_temp[throat_scan_ids[i]])

        new_temp_clusters = np.array(list(temp_cluster_set))
        affected_clusters = np.array(list(affected_set))
        new_clusters_ids = np.empty(len(new_temp_clusters), np.int64)

        for i in range(len(new_temp_clusters)):
            if i < len(affected_clusters):
                new_clusters_ids[i] = affected_clusters[i]
            else:
                max_cluster_id += 1
                new_clusters_ids[i] = max_cluster_id

        # Phase 6: Map temp clusters to real IDs.
        for i in range(pore_scan_count):
            pid = pore_scan_ids[i]
            temp_id = pore_cluster_temp[pid]
            for j in range(len(new_temp_clusters)):
                if temp_id == new_temp_clusters[j]:
                    pore_cluster[pid] = new_clusters_ids[j]
                    break

        for i in range(throat_scan_count):
            tid = throat_scan_ids[i]
            temp_id = throat_cluster_temp[tid]
            for j in range(len(new_temp_clusters)):
                if temp_id == new_temp_clusters[j]:
                    throat_cluster[tid] = new_clusters_ids[j]
                    break

        # Phase 7: Rebuild linked lists for new cluster IDs.
        for i in range(pore_scan_count):
            pid = pore_scan_ids[i]
            c = pore_cluster[pid]
            pore_cluster_next[pid] = cluster_pore_head[c]
            cluster_pore_head[c] = pid

        for i in range(throat_scan_count):
            tid = throat_scan_ids[i]
            c = throat_cluster[tid]
            throat_cluster_next[tid] = cluster_throat_head[c]
            cluster_throat_head[c] = tid

        # Phase 8: Cleanup boolean work buffers.
        for i in range(pore_scan_count):
            pore_to_scan[pore_scan_ids[i]] = False
        for i in range(throat_scan_count):
            throat_to_scan[throat_scan_ids[i]] = False

        return max_cluster_id


# ---------------------------------------------------------------------------
# JIT helper: union-find construction
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _cluster_refresh_union(
    parent_array: NDArray[np.uint32],
    size_array: NDArray[np.float64],
    nT: int,
    throat_ids: NDArray[np.int64],
    pore_has_phase: NDArray[np.bool_],
    throat_has_phase: NDArray[np.bool_],
    throat_conns: NDArray[np.int64],
) -> None:
    """Build union-find by iterating over throats and merging with adjacent pores."""
    union = lambda p, q: wqupc_union(p, q, parent_array, size_array)

    for throat in throat_ids:
        if throat_has_phase[throat]:
            pore_0, pore_1 = throat_conns[throat]
            if pore_has_phase[pore_0]:
                union(throat, pore_0 + nT)
            if pore_has_phase[pore_1]:
                union(throat, pore_1 + nT)


@nb.njit(cache=True)
def _cluster_add_union(
    parent_array: NDArray[np.uint32],
    size_array: NDArray[np.float64],
    nT: int,
    pore_has_phase: NDArray[np.bool_],
    pore_conns_values: NDArray[np.int64],
    pore_conns_offsets: NDArray[np.int64],
    pore_n_conns: NDArray[np.int64],
    throat_has_phase: NDArray[np.bool_],
    throat_conns_values: NDArray[np.int64],
    throat_conns_offsets: NDArray[np.int64],
    pore_id_list: NDArray[np.int64],
    throat_id_list: NDArray[np.int64],
) -> None:
    """Incrementally merge newly-invaded elements into the existing union-find."""
    union = lambda p, q: wqupc_union(p, q, parent_array, size_array)

    for throat_id in throat_id_list:
        if throat_has_phase[throat_id]:
            t_start = throat_conns_offsets[throat_id]
            pore_0 = throat_conns_values[t_start]
            pore_1 = throat_conns_values[t_start + 1]
            if pore_has_phase[pore_0]:
                union(throat_id, pore_0 + nT)
            if pore_has_phase[pore_1]:
                union(throat_id, pore_1 + nT)
    for pore_id in pore_id_list:
        if pore_has_phase[pore_id]:
            p_start = pore_conns_offsets[pore_id]
            p_end = p_start + pore_n_conns[pore_id]
            neighbor_throat_ids = pore_conns_values[p_start:p_end]
            for throat_id in neighbor_throat_ids:
                if throat_has_phase[throat_id]:
                    union(throat_id, pore_id + nT)


# ---------------------------------------------------------------------------
# JIT helper: connection propagation (inlet/outlet reachability)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _cluster_refresh_connections(
    parent_array: NDArray[np.uint32],
    nT: int,
    pore_ids: NDArray[np.int64],
    throat_ids: NDArray[np.int64],
    pore_connected_to_inlet: NDArray[np.bool_],
    pore_connected_to_outlet: NDArray[np.bool_],
    throat_connected_to_inlet: NDArray[np.bool_],
    throat_connected_to_outlet: NDArray[np.bool_],
    pore_cluster: NDArray[np.int64],
    throat_cluster: NDArray[np.int64],
    pore_inlets: NDArray[np.bool_],
    pore_outlets: NDArray[np.bool_],
) -> None:
    """Assign cluster IDs and inlet/outlet flags for all given pores and throats."""
    find = lambda p: wqupc_find(p, parent_array)

    # Determine which clusters touch inlet/outlet boundaries.
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
def _cluster_update_connections(
    parent_array: NDArray[np.uint32],
    nT: int,
    pore_connected_to_inlet: NDArray[np.bool_],
    pore_connected_to_outlet: NDArray[np.bool_],
    pore_cluster: NDArray[np.int64],
    pore_conns_values: NDArray[np.int64],
    pore_conns_offsets: NDArray[np.int64],
    pore_n_conns: NDArray[np.int64],
    throat_connected_to_inlet: NDArray[np.bool_],
    throat_connected_to_outlet: NDArray[np.bool_],
    throat_cluster: NDArray[np.int64],
    throat_conns_values: NDArray[np.int64],
    throat_conns_offsets: NDArray[np.int64],
    pore_inlets: NDArray[np.bool_],
    pore_outlets: NDArray[np.bool_],
    pore_id_list: NDArray[np.int64],
    throat_id_list: NDArray[np.int64],
    inlet_pore_ids: NDArray[np.int64],
    outlet_pore_ids: NDArray[np.int64],
) -> None:
    """Propagate inlet/outlet connectivity after incremental union updates."""
    find = lambda p: wqupc_find(p, parent_array)

    inlets_clusters = set()
    outlets_clusters = set()

    for i in range(len(inlet_pore_ids)):
        inlets_clusters.add(find(inlet_pore_ids[i] + nT))
    for i in range(len(outlet_pore_ids)):
        outlets_clusters.add(find(outlet_pore_ids[i] + nT))

    for pore_id in pore_id_list:
        _update_element_connection(
            parent_array,
            nT,
            0,
            pore_id,
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            pore_cluster,
            pore_conns_values,
            pore_conns_offsets,
            pore_n_conns,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            throat_cluster,
            inlets_clusters,
            outlets_clusters,
        )
    for throat_id in throat_id_list:
        _update_element_connection(
            parent_array,
            0,
            nT,
            throat_id,
            throat_connected_to_inlet,
            throat_connected_to_outlet,
            throat_cluster,
            throat_conns_values,
            throat_conns_offsets,
            np.full(len(throat_conns_offsets), 2, np.int64),
            pore_connected_to_inlet,
            pore_connected_to_outlet,
            pore_cluster,
            inlets_clusters,
            outlets_clusters,
        )


@nb.njit(cache=True)
def _update_element_connection(
    wqupc_parent_array: NDArray[np.uint32],
    wqupc_element_id_offset: int,
    wqupc_neighbor_id_offset: int,
    changed_element_id: int,
    element_connected_to_inlet: NDArray[np.bool_],
    element_connected_to_outlet: NDArray[np.bool_],
    element_cluster: NDArray[np.int64],
    element_conns_values: NDArray[np.int64],
    element_conns_offsets: NDArray[np.int64],
    element_n_conns: NDArray[np.int64],
    neighbor_connected_to_inlet: NDArray[np.bool_],
    neighbor_connected_to_outlet: NDArray[np.bool_],
    neighbor_cluster: NDArray[np.int64],
    inlets_clusters: set,
    outlets_clusters: set,
) -> None:
    """Update cluster ID and connectivity flags for a changed element and its neighbors."""
    find = lambda p: wqupc_find(p, wqupc_parent_array)

    cluster = find(changed_element_id + wqupc_element_id_offset)
    element_cluster[changed_element_id] = cluster

    cluster_has_inlets = cluster in inlets_clusters
    cluster_has_outlets = cluster in outlets_clusters

    # Update all elements sharing this cluster.
    for element_id in range(len(element_cluster)):
        if element_cluster[element_id] == cluster:
            element_connected_to_inlet[element_id] = cluster_has_inlets
            element_connected_to_outlet[element_id] = cluster_has_outlets
    for neighbor_id in range(len(neighbor_cluster)):
        if neighbor_cluster[neighbor_id] == cluster:
            neighbor_connected_to_inlet[neighbor_id] = cluster_has_inlets
            neighbor_connected_to_outlet[neighbor_id] = cluster_has_outlets

    # Also update neighbors whose cluster root may have changed due to the union.
    e_start = element_conns_offsets[changed_element_id]
    e_end = e_start + element_n_conns[changed_element_id]
    for neighbor_id in element_conns_values[e_start:e_end]:
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


# ---------------------------------------------------------------------------
# JIT helper: combined reset (union + connection propagation)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _jit_reset_connections(
    parent_array: NDArray[np.uint32],
    size_array: NDArray[np.float64],
    nT: int,
    pore_ids: NDArray[np.int64],
    throat_ids: NDArray[np.int64],
    pore_connected_to_inlet: NDArray[np.bool_],
    pore_connected_to_outlet: NDArray[np.bool_],
    throat_connected_to_inlet: NDArray[np.bool_],
    throat_connected_to_outlet: NDArray[np.bool_],
    pore_has_phase: NDArray[np.bool_],
    throat_has_phase: NDArray[np.bool_],
    throat_conns: NDArray[np.int64],
    pore_cluster: NDArray[np.int64],
    throat_cluster: NDArray[np.int64],
    pore_inlets: NDArray[np.bool_],
    pore_outlets: NDArray[np.bool_],
) -> None:
    """Full reset: build union-find from scratch, then propagate connectivity."""
    _cluster_refresh_union(
        parent_array,
        size_array,
        nT,
        throat_ids,
        pore_has_phase,
        throat_has_phase,
        throat_conns,
    )
    _cluster_refresh_connections(
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


# ---------------------------------------------------------------------------
# JIT helper: cluster linked-list maintenance
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _build_cluster_lists(
    pore_cluster: NDArray[np.int64],
    throat_cluster: NDArray[np.int64],
    pore_next: NDArray[np.int64],
    throat_next: NDArray[np.int64],
    cluster_pore_head: NDArray[np.int64],
    cluster_throat_head: NDArray[np.int64],
) -> None:
    """Build intrusive linked lists mapping cluster IDs to their member elements."""
    cluster_pore_head[:] = -1
    cluster_throat_head[:] = -1
    for pid in range(len(pore_cluster)):
        c = pore_cluster[pid]
        if c >= 0:
            pore_next[pid] = cluster_pore_head[c]
            cluster_pore_head[c] = pid
    for tid in range(len(throat_cluster)):
        c = throat_cluster[tid]
        if c >= 0:
            throat_next[tid] = cluster_throat_head[c]
            cluster_throat_head[c] = tid
