"""Single-pass percolation fast path for the first-drainage cycle.

First drainage has constant entry pressures (see DrainageSimulation), so the
invasion order can be computed once via a Union-Find sweep over elements
sorted by entry pressure -- the same trick used by openpnm's percolation
algorithms -- instead of the general step-by-step avalanche + cluster
recheck loop used by CycleSimulation (which also has to support imbibition's
dynamic trapping). Physics (entry pressure, corner saturation) are untouched;
only the invasion-order/connectivity computation changes.
"""

import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import calculate_drainage_network_saturation
from py_pore_flow.simulation.simulation import Simulation
import py_pore_flow.consts as c


@nb.njit(cache=True)
def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        nxt = parent[i]
        parent[i] = root
        i = nxt
    return root


@nb.njit(cache=True)
def _mark_component(list_head, list_next, invasion_pc, root, pc, n):
    node = list_head[root]
    while node != -1:
        if node < n:
            invasion_pc[node] = pc
        node = list_next[node]


@nb.njit(cache=True)
def _union(parent, rank, connected, list_next, list_head, list_tail, invasion_pc, n, a, b, pc):
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra == rb:
        return

    a_conn = connected[ra]
    b_conn = connected[rb]

    if a_conn and not b_conn:
        _mark_component(list_head, list_next, invasion_pc, rb, pc, n)
    elif b_conn and not a_conn:
        _mark_component(list_head, list_next, invasion_pc, ra, pc, n)

    head_a, tail_a = list_head[ra], list_tail[ra]
    head_b, tail_b = list_head[rb], list_tail[rb]
    list_next[tail_a] = head_b
    combined_connected = a_conn or b_conn

    if rank[ra] < rank[rb]:
        parent[ra] = rb
        list_head[rb] = head_a
        list_tail[rb] = tail_b
        connected[rb] = combined_connected
    else:
        parent[rb] = ra
        list_head[ra] = head_a
        list_tail[ra] = tail_b
        connected[ra] = combined_connected
        if rank[ra] == rank[rb]:
            rank[ra] += 1


@nb.njit(cache=True)
def _run_percolation(
    entry_pressure,
    is_inlet_pore,
    pore_conns_values,
    pore_conns_offsets,
    throat_conns_values,
    n_pores,
    order,
):
    n = entry_pressure.shape[0]
    size = n + 1
    SOURCE = n

    parent = np.arange(size, dtype=np.int64)
    rank = np.zeros(size, dtype=np.int64)
    connected = np.zeros(size, dtype=np.bool_)
    connected[SOURCE] = True
    list_next = np.full(size, -1, dtype=np.int64)
    list_head = np.arange(size, dtype=np.int64)
    list_tail = np.arange(size, dtype=np.int64)
    invasion_pc = np.full(n, np.inf, dtype=np.float64)
    activated = np.zeros(n, dtype=np.bool_)

    for idx in range(n):
        i = order[idx]
        pc_i = entry_pressure[i]
        activated[i] = True

        if i < n_pores:
            if is_inlet_pore[i]:
                _union(parent, rank, connected, list_next, list_head, list_tail, invasion_pc, n, i, SOURCE, pc_i)
            start = pore_conns_offsets[i]
            end = pore_conns_offsets[i + 1]
            for k in range(start, end):
                t = pore_conns_values[k]
                j = n_pores + t
                if activated[j]:
                    _union(parent, rank, connected, list_next, list_head, list_tail, invasion_pc, n, i, j, pc_i)
        else:
            t = i - n_pores
            p1 = throat_conns_values[2 * t]
            p2 = throat_conns_values[2 * t + 1]
            if activated[p1]:
                _union(parent, rank, connected, list_next, list_head, list_tail, invasion_pc, n, i, p1, pc_i)
            if activated[p2]:
                _union(parent, rank, connected, list_next, list_head, list_tail, invasion_pc, n, i, p2, pc_i)

    return invasion_pc


def compute_drainage_invasion_pc(network):
    """Compute, for every pore and throat, the Pc at which it is invaded by
    the non-wetting phase during first drainage -- equivalent to running the
    full step-by-step DrainageSimulation to completion, but in a single
    O(N log N) pass instead of O(N x avalanche-recheck) per pressure step.

    Elements never invaded (disconnected from the inlet even at pc_maximum)
    keep invasion_pc = +inf.
    """
    pore = network.pn_core.pore
    throat = network.pn_core.throat

    n_pores = len(pore.entry_pressure)
    n_throats = len(throat.entry_pressure)

    entry_pressure = np.concatenate((pore.entry_pressure, throat.entry_pressure))
    order = np.argsort(entry_pressure, kind="stable")

    invasion_pc = _run_percolation(
        entry_pressure,
        pore.inlets,
        pore.conns_values,
        pore.conns_offsets,
        throat.conns_values,
        n_pores,
        order,
    )

    return invasion_pc[:n_pores], invasion_pc[n_pores:]


def run_fast_first_drainage(pn, parameters_dict=None, generate_cas=True, setup_network=True, n_curve_points=100):
    """Fast path for the first-drainage-only case (e.g. MICP): same entry
    pressure physics (MS-P/Young-Laplace via calculate_drainage_pc, corner
    film saturation) as the regular engine, but the invasion order is
    computed with a single Union-Find percolation sweep instead of the
    general avalanche-based step loop. Only valid for a plain first drainage
    (no imbibition/2nd drainage/Krel -- those need dynamic trapping, which
    this fast path does not support).

    Returns (Pc, Sw) arrays, analogous to the drainage cycle output of
    run_two_phase_simulation.
    """
    parameters_dict = parameters_dict or {}

    simulation = Simulation(
        pn,
        parameters_dict,
        generate_vtu=False,
        generate_cas=generate_cas,
        setup_network=setup_network,
        calculate_krel=False,
        calculate_trapping=False,
    )
    step_simulation = simulation.step_simulation
    pc_maximum = step_simulation.get_parameter_manager().pc_maximum

    step_simulation.set_max_pc(pc_maximum)
    step_simulation.set_cycle("drainage")
    network = step_simulation.network

    pore_invasion_pc, throat_invasion_pc = compute_drainage_invasion_pc(network)

    finite_pc = np.concatenate((pore_invasion_pc[np.isfinite(pore_invasion_pc)],
                                 throat_invasion_pc[np.isfinite(throat_invasion_pc)]))
    if finite_pc.size == 0:
        return np.array([0.0]), np.array([0.0])

    pc_min = min(0.0, finite_pc.min())
    pc_max = max(finite_pc.max(), pc_min + 1e-12)
    pc_samples = np.linspace(pc_min, pc_max, n_curve_points)

    pore = network.pn_core.pore
    throat = network.pn_core.throat

    pore_volume = (pore.volume + pore.clay_volume) * pore.number_of_capilaries
    throat_volume = (throat.volume + throat.clay_volume) * throat.number_of_capilaries
    total_volume = pore_volume.sum() + throat_volume.sum()

    Sw = np.empty(n_curve_points, dtype=np.float64)
    for i, pc in enumerate(pc_samples):
        pore_oil_filled = pore_invasion_pc <= pc
        throat_oil_filled = throat_invasion_pc <= pc

        pore_sw, _ = calculate_drainage_network_saturation(
            pore.shape, pore.C_r, pc, pore.area, pore_oil_filled, c.sigma
        )
        throat_sw, _ = calculate_drainage_network_saturation(
            throat.shape, throat.C_r, pc, throat.cross_sectional_area, throat_oil_filled, c.sigma
        )

        Sw[i] = (pore_sw * pore_volume).sum() / total_volume + (throat_sw * throat_volume).sum() / total_volume

    # Leave the network in the same final state (at pc_maximum) that the
    # regular step-by-step engine would, so callers reading pn["pore.oil_filled"]
    # / pn["throat.oil_filled"] afterwards see consistent results.
    pore.oil_filled[:] = pore_invasion_pc <= pc_max
    throat.oil_filled[:] = throat_invasion_pc <= pc_max

    return pc_samples, Sw
