"""Directional flow: the throat flow rates projected onto the coordinate axes.

The projection hangs on reading `throat.conns` with the right convention, so the tests
check that independently rather than trusting the helper.
"""

import os
import shutil
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from numpy.testing import assert_allclose

from PySide2 import QtWidgets

from py_pore_flow.graphic.diagnostics.diagnostics_crossplot_widget import CrossPlotWidget
from py_pore_flow.graphic.diagnostics.diagnostics_heatmap_widget import HeatmapWidget
from py_pore_flow.graphic.diagnostics.diagnostics_properties import DiagnosticsProperties
from py_pore_flow.network.network import initialize_pn, NetworkManager, PoreNetwork
from py_pore_flow.output.diagnostics import (
    DIRECTION_AXES,
    Diagnostics,
    DiagnosticsWriter,
    throat_axis_cosines,
    zero_based_conns,
)
from py_pore_flow.output.diagnostics_aggregate import (
    AGG_SUM,
    AggregateDiagnosticsWriter,
    DiagnosticsAggregates,
)
from py_pore_flow.parameters.parameter_manager import ParameterManager
from py_pore_flow.simulation.simulation import calculate_initial_values

N_STEPS = 3
N_BINS = 20


def make_network():
    pn = PoreNetwork.from_statoil("tests/resources/a", "a")
    initialize_pn(pn)
    network = NetworkManager(pn)
    parameter_manager = ParameterManager()
    parameter_manager.update_initial_cas(network)
    calculate_initial_values(network)
    return network


def water_flow_rate(step, n):
    return (np.arange(n) % 32) * 0.5 + step


def oil_flow_rate(step, n):
    return (np.arange(n) % 16) * 0.25 + step * 2.0


def run(make_writer, path):
    network = make_network()
    writer = make_writer(network, path)
    n_throats = int(network.pn_core.throat.water_saturation.size)
    for step in range(N_STEPS):
        network.pn_core.throat.water_flow_rate = water_flow_rate(step, n_throats)
        network.pn_core.throat.oil_flow_rate = oil_flow_rate(step, n_throats)
        writer.save_state(10.0 * (step + 1), 1.0 - 0.1 * step)
    writer.save_cycle_data()
    writer.close()


class ConnsConventionTests(unittest.TestCase):
    """`throat.conns` is read as pore indices; the source table uses one based labels."""

    def test_zero_based_conns_are_passed_through(self):
        conns = np.array([[0, 1], [1, 2], [2, 3]])
        assert_allclose(zero_based_conns(conns, n_pores=4), conns)

    def test_one_based_labels_are_shifted(self):
        conns = np.array([[1, 2], [2, 3], [3, 4]])
        assert_allclose(zero_based_conns(conns, n_pores=4), conns - 1)

    def test_out_of_range_connections_are_refused(self):
        # better no direction at all than every throat attached to the wrong pore
        self.assertIsNone(zero_based_conns(np.array([[0, 9]]), n_pores=4))
        self.assertIsNone(zero_based_conns(np.array([[-2, 1]]), n_pores=4))
        self.assertIsNone(zero_based_conns(np.array([]), n_pores=4))

    def test_the_real_network_is_zero_based(self):
        # the decisive check: read with the right convention, the distance between the two
        # pore centres tracks the throat length; read one off, it does not even fit the array
        pn = make_network().pn
        coords = np.asarray(pn["pore.coords"])
        conns = np.asarray(pn["throat.conns"])
        n_pores = coords.shape[0]

        self.assertEqual(int(conns.min()), 0)
        self.assertEqual(int(conns.max()), n_pores - 1)

        distance = np.linalg.norm(coords[conns[:, 1]] - coords[conns[:, 0]], axis=1)
        length = np.asarray(pn["throat.length"])
        self.assertGreater(np.corrcoef(distance, length)[0, 1], 0.8)
        self.assertTrue((distance > 0).all())


class DirectionalPropertyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix="diagnostics_directional_")
        run(lambda network, path: DiagnosticsWriter(network, path), cls.path)

        pn = make_network().pn
        coords = np.asarray(pn["pore.coords"], dtype=float)
        conns = np.asarray(pn["throat.conns"])
        cls.n_throats = conns.shape[0]

        # computed here without the helper under test
        vectors = coords[conns[:, 1], :3] - coords[conns[:, 0], :3]
        cls.expected_cosines = np.abs(vectors / np.linalg.norm(vectors, axis=1)[:, None])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def setUp(self):
        self.diagnostics = Diagnostics(self.path)
        self.properties = DiagnosticsProperties(self.diagnostics)

    def tearDown(self):
        self.diagnostics.close()

    def test_axis_cosines_match_the_pore_coordinates(self):
        assert_allclose(throat_axis_cosines(self.diagnostics.network), self.expected_cosines, rtol=1e-12)

    def test_the_components_are_the_flow_projected_onto_each_axis(self):
        for step in range(N_STEPS):
            flow = water_flow_rate(step, self.n_throats)
            for axis, axis_name in enumerate(DIRECTION_AXES):
                values = self.properties.get_values_in_step(f"throat.water_flow_rate_{axis_name}", 0, step)
                assert_allclose(values, flow * self.expected_cosines[:, axis], rtol=1e-12)

    def test_oil_is_decomposed_as_well(self):
        flow = oil_flow_rate(1, self.n_throats)
        for axis, axis_name in enumerate(DIRECTION_AXES):
            values = self.properties.get_values_in_step(f"throat.oil_flow_rate_{axis_name}", 0, 1)
            assert_allclose(values, flow * self.expected_cosines[:, axis], rtol=1e-12)

    def test_the_components_recompose_into_the_original_flow(self):
        flow = water_flow_rate(0, self.n_throats)
        squares = sum(
            self.properties.get_values_in_step(f"throat.water_flow_rate_{name}", 0, 0) ** 2 for name in DIRECTION_AXES
        )
        assert_allclose(np.sqrt(squares), np.abs(flow), rtol=1e-10)

    def test_the_angles_span_a_quarter_turn_and_agree_with_the_components(self):
        for axis, axis_name in enumerate(DIRECTION_AXES):
            angles = self.properties.get_values_in_step(f"throat.angle_{axis_name}", 0, 0)
            self.assertEqual(angles.shape, (self.n_throats,))
            self.assertTrue(((angles >= 0.0) & (angles <= 90.0)).all())
            assert_allclose(np.cos(np.radians(angles)), self.expected_cosines[:, axis], atol=1e-12)

    def test_the_new_properties_are_offered_for_selection(self):
        names = self.properties.get_throat_property_names()
        for axis_name in DIRECTION_AXES:
            self.assertIn(f"water_flow_rate_{axis_name}", names)
            self.assertIn(f"oil_flow_rate_{axis_name}", names)
            self.assertIn(f"angle_{axis_name}", names)

        # they are throat geometry, so pores must not offer them
        self.assertNotIn("water_flow_rate_x", self.properties.get_pore_property_names())

    def test_they_are_unavailable_when_the_dump_has_no_geometry(self):
        stripped = Diagnostics(self.path)
        try:
            del stripped.network["pore.coords"]
            properties = DiagnosticsProperties(stripped)
            self.assertNotIn("water_flow_rate_x", properties.get_throat_property_names())
            with self.assertRaises(KeyError):
                properties.get_values_in_step("throat.water_flow_rate_x", 0, 0)
        finally:
            stripped.close()


class DirectionalInTheViewsTests(unittest.TestCase):
    """Both views the properties were asked for offer them."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.path = tempfile.mkdtemp(prefix="diagnostics_directional_views_")
        run(lambda network, path: DiagnosticsWriter(network, path), cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def setUp(self):
        self.diagnostics = Diagnostics(self.path)

    def tearDown(self):
        self.diagnostics.close()

    @staticmethod
    def _items(combo):
        return [combo.itemText(i) for i in range(combo.count())]

    def test_the_scatter_plot_offers_them_on_every_axis(self):
        widget = CrossPlotWidget(self.diagnostics)
        widget.element_type_combo.setCurrentText("Throat")

        for combo in (widget.x_combo, widget.y_combo):
            items = self._items(combo)
            self.assertIn("water_flow_rate_z", items)
            self.assertIn("oil_flow_rate_x", items)
            self.assertIn("angle_z", items)

    def test_the_heatmap_offers_them_as_axis_and_colour(self):
        widget = HeatmapWidget(self.diagnostics)
        widget.element_type_combo.setCurrentText("Throat")

        self.assertIn("angle_z", self._items(widget.property_combo))
        self.assertIn("water_flow_rate_z", self._items(widget.color_combo))

    def test_the_heatmap_can_bin_by_angle_and_colour_by_a_component(self):
        widget = HeatmapWidget(self.diagnostics)
        widget.element_type_combo.setCurrentText("Throat")
        widget.property_combo.setCurrentText("angle_z")
        widget.color_combo.setCurrentText("water_flow_rate_z")
        widget.aggregation_combo.setCurrentText(AGG_SUM)
        widget.bins_spinbox.setValue(N_BINS)
        widget._apply()

        self.assertEqual(np.asarray(widget.view._image.image).shape, (N_STEPS, N_BINS))


class DirectionalAggregateTests(unittest.TestCase):
    """The aggregates derive the components from the spooled source, not from a spool of
    their own."""

    PAIRS = [("throat", "angle_z", "water_flow_rate_z"), ("throat", "radius", "oil_flow_rate_x")]

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix="diagnostics_directional_agg_")
        run(
            lambda network, path: AggregateDiagnosticsWriter(network, path, pairs=cls.PAIRS, n_bins=N_BINS),
            cls.path,
        )

        pn = make_network().pn
        coords = np.asarray(pn["pore.coords"], dtype=float)
        conns = np.asarray(pn["throat.conns"])
        cls.n_throats = conns.shape[0]
        vectors = coords[conns[:, 1], :3] - coords[conns[:, 0], :3]
        cls.cosines = np.abs(vectors / np.linalg.norm(vectors, axis=1)[:, None])
        cls.radius = np.asarray(pn["throat.radius"], dtype=float)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def setUp(self):
        self.aggregates = DiagnosticsAggregates(self.path)

    def test_the_directional_pairs_are_stored(self):
        self.assertEqual(self.aggregates.color_property_names("throat", "angle_z"), ["water_flow_rate_z"])
        self.assertEqual(self.aggregates.color_property_names("throat", "radius"), ["oil_flow_rate_x"])

    def test_the_angle_bins_carry_the_projected_flow(self):
        angles = np.degrees(np.arccos(np.clip(self.cosines[:, 2], 0.0, 1.0)))
        edges = np.linspace(angles.min(), angles.max(), N_BINS + 1)

        matrix, _ = self.aggregates.matrix(0, "throat", "angle_z", "water_flow_rate_z", AGG_SUM, N_BINS)

        for step in range(N_STEPS):
            weights = water_flow_rate(step, self.n_throats) * self.cosines[:, 2]
            expected, _ = np.histogram(angles, bins=edges, weights=weights)
            count, _ = np.histogram(angles, bins=edges)
            assert_allclose(matrix[step], np.where(count > 0, expected, np.nan), rtol=1e-5)

    def test_a_component_can_colour_a_geometric_axis(self):
        edges = np.linspace(self.radius.min(), self.radius.max(), N_BINS + 1)

        matrix, _ = self.aggregates.matrix(0, "throat", "radius", "oil_flow_rate_x", AGG_SUM, N_BINS)

        for step in range(N_STEPS):
            weights = oil_flow_rate(step, self.n_throats) * self.cosines[:, 0]
            expected, _ = np.histogram(self.radius, bins=edges, weights=weights)
            count, _ = np.histogram(self.radius, bins=edges)
            assert_allclose(matrix[step], np.where(count > 0, expected, np.nan), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
