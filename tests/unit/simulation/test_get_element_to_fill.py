import numpy as np
import pytest

from py_pore_flow.network.network import PORE, THROAT
from py_pore_flow.simulation.border_elements import BorderElements
from py_pore_flow.simulation.drainage import DrainageSimulation
from py_pore_flow.simulation.imbibition import ImbibitionSimulation
from tests.unit.simulation.resources.simple_network import get_simple_network
from py_pore_flow.network.network import initialize_pn, NetworkManager


def test_border_elements_peek_best_drainage():
    be = BorderElements()

    be.add_pores_ids(np.array([0, 1, 2]))
    be.add_throats_ids(np.array([0, 1]))

    assert set(be.get_pores_ids()) == {0, 1, 2}
    assert set(be.get_throats_ids()) == {0, 1}

    be.add_pores_ids(np.array([3]))
    assert set(be.get_pores_ids()) == {0, 1, 2, 3}

    be.add_throats_ids(np.array([2]))
    assert set(be.get_throats_ids()) == {0, 1, 2}

    be.remove_pores_ids(np.array([3]))
    assert set(be.get_pores_ids()) == {0, 1, 2}

def test_border_elements_peek_best_imbibition():
    be = BorderElements()

    be.add_pores_ids(np.array([0, 1, 2]))
    be.add_throats_ids(np.array([0, 1]))

    assert set(be.get_pores_ids()) == {0, 1, 2}
    assert set(be.get_throats_ids()) == {0, 1}

    be.add_throats_ids(np.array([2])) 
    assert set(be.get_throats_ids()) == {0, 1, 2}

    be.add_pores_ids(np.array([3]))
    assert set(be.get_pores_ids()) == {0, 1, 2, 3}
