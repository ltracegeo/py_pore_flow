import shutil
import tempfile
import unittest

import numpy as np

from py_pore_flow.graphic.diagnostics.diagnostics_properties import DiagnosticsProperties
from py_pore_flow.network.network import initialize_pn, NetworkManager, PoreNetwork
from py_pore_flow.output.diagnostics.diagnostics import Diagnostics, DiagnosticsWriter
from py_pore_flow.parameters.parameter_manager import ParameterManager
from py_pore_flow.simulation.simulation import calculate_initial_values

N_STEPS_PER_CYCLE = 3
N_CYCLES = 2


def make_network():
    pn = PoreNetwork.from_statoil("tests/resources/a", "a")
    initialize_pn(pn)
    network = NetworkManager(pn)
    parameter_manager = ParameterManager()
    parameter_manager.update_initial_cas(network)
    calculate_initial_values(network)
    return network


class DiagnosticsPropertiesTests(unittest.TestCase):
    """DiagnosticsProperties is the catalogue every view reads through, so it is covered
    here rather than in a widget test: it has no Qt dependency."""

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix="diagnostics_properties_test_")
        network = make_network()
        cls.n_pores = len(network.pn_core.pore.id)
        cls.n_throats = len(network.pn_core.throat.id)

        writer = DiagnosticsWriter(network, cls.path)
        for _ in range(N_CYCLES):
            for _ in range(N_STEPS_PER_CYCLE):
                writer.save_state()
            writer.save_cycle_data()
        writer.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def setUp(self):
        self.diagnostics = Diagnostics(self.path)
        self.properties = DiagnosticsProperties(self.diagnostics)

    def tearDown(self):
        self.diagnostics.close()

    def test_property_names_cover_every_source(self):
        names = self.properties.get_property_names()

        # per step loader
        self.assertIn("throat.water_saturation", names)
        self.assertIn("throat.oil_flow_rate", names)
        self.assertIn("pore.water_filled", names)
        # static network dump
        self.assertIn("throat.radius", names)
        self.assertIn("pore.volume", names)
        # derived conversors
        self.assertIn("pore.oil_saturation", names)
        self.assertIn("pore.x", names)
        # per cycle loader: resolvable through _sample_cycle_element but previously listed
        # by nothing, so unselectable in every combo
        self.assertIn("throat.entry_pressure", names)
        self.assertIn("throat.invasion_method", names)
        self.assertIn("pore.advancing_contact_angle", names)

        # 2-D corner arrays are not plottable and must stay out
        self.assertNotIn("throat.b", names)
        self.assertNotIn("pore.corner_water_layer", names)

    def test_property_names_are_computed_once(self):
        first = self.properties.get_property_names()
        self.assertIs(self.properties.get_property_names(), first)

    def test_cycle_properties_resolve_to_values(self):
        values = self.properties.get_values_in_step("throat.entry_pressure", 0, 0)
        self.assertEqual(values.shape, (self.n_throats,))

    def test_element_values_in_step_concatenates_both_types(self):
        # regression: this used to reference an undefined `step` and test ndarray truthiness
        values = self.properties.get_element_values_in_step("water_saturation", 0, 0)
        self.assertEqual(values.shape, (self.n_pores + self.n_throats,))

    def test_element_values_in_step_falls_back_to_the_type_that_has_it(self):
        # oil_flow_rate exists on throats only
        values = self.properties.get_element_values_in_step("oil_flow_rate", 0, 0)
        self.assertEqual(values.shape, (self.n_throats,))

    def test_element_values_in_step_raises_for_an_unknown_property(self):
        with self.assertRaises(KeyError):
            self.properties.get_element_values_in_step("not_a_property", 0, 0)

    def test_values_of_cycle_returns_one_array_per_step(self):
        steps = self.properties.get_throat_values_of_cycle("water_saturation", 0)
        self.assertEqual(len(steps), N_STEPS_PER_CYCLE)
        for step_values in steps:
            self.assertEqual(step_values.shape, (self.n_throats,))
            self.assertTrue(np.issubdtype(step_values.dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
