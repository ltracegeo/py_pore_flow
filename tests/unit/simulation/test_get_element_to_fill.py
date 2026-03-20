import numpy as np
import pytest

from py_pore_flow.network.network import PORE, THROAT
from py_pore_flow.simulation.border_elements import BorderElements
from py_pore_flow.simulation.drainage import DrainageSimulation
from py_pore_flow.simulation.imbibition import ImbibitionSimulation
from tests.unit.simulation.resources.simple_network import get_simple_network
from py_pore_flow.network.network import initialize_pn, NetworkManager


def test_border_elements_peek_best_drainage():
    n_pores = 10
    n_throats = 10
    pore_pc = np.array([1000.0, 2000.0, 1500.0, 500.0, 500.0] + [0.0]*5)
    throat_pc = np.array([800.0, 1200.0, 500.0, 400.0, 2500.0] + [0.0]*5)

    # In drainage, we do NOT negate PC (we want minimum PC)
    be = BorderElements(n_pores, n_throats, pore_pc, throat_pc, negate_pc=False)

    be.add_pores_ids(np.array([0, 1, 2]))
    be.add_throats_ids(np.array([0, 1]))

    # Pores: 0 (1000), 1 (2000), 2 (1500)
    # Throats: 0 (800), 1 (1200)
    # Min is Throat 0 (800)
    best = be.peek_best()
    assert best == (THROAT, 0, 800.0)

    be.add_pores_ids(np.array([3])) # pore 3 has PC 500
    best = be.peek_best()
    assert best == (PORE, 3, 500.0)

    be.add_throats_ids(np.array([2])) # throat 2 has PC 500
    # Tie break: Pore should be preferred when PC is equal
    best = be.peek_best()
    assert best == (PORE, 3, 500.0)

    be.remove_pores_ids(np.array([3]))
    best = be.peek_best()
    assert best == (THROAT, 2, 500.0)


def test_border_elements_peek_best_imbibition():
    n_pores = 10
    n_throats = 10
    pore_pc = np.array([1000.0, 2000.0, 1500.0, 2500.0, 2500.0] + [0.0]*5)
    throat_pc = np.array([800.0, 1200.0, 2500.0, 3000.0, 2500.0] + [0.0]*5)

    # In imbibition, we negate PC (we want maximum PC)
    be = BorderElements(n_pores, n_throats, pore_pc, throat_pc, negate_pc=True)

    be.add_pores_ids(np.array([0, 1, 2]))
    be.add_throats_ids(np.array([0, 1]))

    # Pores: 0 (1000), 1 (2000), 2 (1500)
    # Throats: 0 (800), 1 (1200)
    # Max is Pore 1 (2000)
    best = be.peek_best()
    assert best == (PORE, 1, 2000.0)

    be.add_throats_ids(np.array([2])) # throat 2 has PC 2500
    best = be.peek_best()
    assert best == (THROAT, 2, 2500.0)

    be.add_pores_ids(np.array([3])) # pore 3 has PC 2500
    # Tie break: Pore should be preferred when PC is equal
    best = be.peek_best()
    assert best == (PORE, 3, 2500.0)


def test_subsequent_cycle_get_element_to_fill():
    pn = get_simple_network(initial_saturation="water_filled")
    initialize_pn(pn)
    network = NetworkManager(pn)
    
    from py_pore_flow.parameters.parameter_manager import ParameterManager
    pm = ParameterManager({})
    
    # Create imbibition simulation
    sim = ImbibitionSimulation(network, 10000.0)
    
    # Manually populate trees to test _get_element_to_fill
    # Imbibition (WATER phase) negate_pc=True, meaning it looks for MAX PC
    
    # 1. Setup border elements
    # set pore 0 PC to 1000, throat 0 PC to 1500
    sim.network.pn_core.pore.entry_pressure[0] = 1000.0
    sim.network.pn_core.throat.entry_pressure[0] = 1500.0
    sim.border_elements.add_pores_ids(np.array([0]))
    sim.border_elements.add_throats_ids(np.array([0]))
    
    elem = sim._get_element_to_fill(2000.0, sim.border_elements)
    # Throat 0 is max (1500) vs Pore 0 (1000)
    assert elem == (THROAT, 0, 1500.0)  # for imbibition, returns min(true_pc, pc) -> min(1500, 2000) = 1500
    
    # 2. Add a snap_off pore with higher PC
    sim.network.pn_core.pore.entry_pressure[1] = 1800.0
    sim.network.pn_core.pore.water_filled[1] = False
    sim.snap_off_pore_tree.insert(1, sim.snap_off_sign * 1800.0)
    
    elem = sim._get_element_to_fill(2000.0, sim.border_elements)
    assert elem == (PORE, 1, 1800.0)
    
    # 3. Add a snap_off throat with tie-breaker
    sim.network.pn_core.throat.entry_pressure[1] = 1800.0
    sim.network.pn_core.throat.water_filled[1] = False
    sim.snap_off_throat_tree.insert(1, sim.snap_off_sign * 1800.0)
    
    # PORE should win ties against THROAT, so it should still be PORE 1
    elem = sim._get_element_to_fill(2000.0, sim.border_elements)
    assert elem == (PORE, 1, 1800.0)
    
    # 4. Add a border pore with tie-breaker against snap_off
    sim.network.pn_core.pore.entry_pressure[2] = 1800.0
    sim.border_elements.add_pores_ids(np.array([2]))
    
    # Tie between border pore 2 and snap_off pore 1.
    # The minimum ID should win in both old and new logic, but since they are in different trees,
    # my logic checks min_pc == best_p_pc FIRST, meaning border pores take precedence.
    elem = sim._get_element_to_fill(2000.0, sim.border_elements)
    assert elem == (PORE, 2, 1800.0)

    # 5. Check min clamping with PC parameter
    elem = sim._get_element_to_fill(1700.0, sim.border_elements)
    assert elem == (PORE, 2, 1700.0)
