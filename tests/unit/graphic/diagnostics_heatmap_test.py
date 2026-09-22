"""The heatmap has two data paths: the raw per-element dump and the precomputed aggregates.

They must agree, since `basic` and `complete` runs of the same simulation should show the
same picture.
"""

import os
import shutil
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from numpy.testing import assert_allclose
from PySide2 import QtWidgets

from py_pore_flow.graphic.diagnostics.diagnostics_heatmap_widget import AGG_SUM, COLOR_COUNT, HeatmapWidget
from py_pore_flow.graphic.diagnostics.diagnostics_widget import DiagnosticsWidget
from py_pore_flow.network.network import initialize_pn, NetworkManager, PoreNetwork
from py_pore_flow.output.diagnostics import Diagnostics, DiagnosticsWriter
from py_pore_flow.output.diagnostics_aggregate import AggregateDiagnosticsWriter, DiagnosticsAggregates
from py_pore_flow.parameters.parameter_manager import ParameterManager
from py_pore_flow.simulation.simulation import calculate_initial_values

N_STEPS = 4
STORED_N_BINS = 100
VIEW_N_BINS = 50

PAIRS = [("throat", "radius", "water_flow_rate"), ("throat", "water_saturation", None)]


def make_network():
    pn = PoreNetwork.from_statoil("tests/resources/a", "a")
    initialize_pn(pn)
    network = NetworkManager(pn)
    parameter_manager = ParameterManager()
    parameter_manager.update_initial_cas(network)
    calculate_initial_values(network)
    return network


def water_flow_rate(step, n):
    # exact in float32, so the aggregate spool round trip is lossless
    return (np.arange(n) % 32) * 0.5 + step


def water_saturation(step, n):
    # reaches exactly zero, so this property can never be binned logarithmically
    return ((np.arange(n) + step) % 64) / 64.0


def run_fake_simulation(make_writer):
    """Drive a writer over a network whose properties change every step.

    The writer has to be built from the same network the loop mutates: it keeps a reference
    to pn_core, so feeding it a second network would record only the initial state.
    """
    network = make_network()
    writer = make_writer(network)

    n_throats = int(network.pn_core.throat.water_saturation.size)
    for step in range(N_STEPS):
        network.pn_core.throat.water_flow_rate = water_flow_rate(step, n_throats)
        network.pn_core.throat.water_saturation = water_saturation(step, n_throats)
        writer.save_state()
    writer.save_cycle_data()
    writer.close()


class HeatmapSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        cls.complete_path = tempfile.mkdtemp(prefix="diagnostics_heatmap_complete_")
        cls.basic_path = tempfile.mkdtemp(prefix="diagnostics_heatmap_basic_")

        run_fake_simulation(lambda network: DiagnosticsWriter(network, cls.complete_path))
        run_fake_simulation(
            lambda network: AggregateDiagnosticsWriter(network, cls.basic_path, pairs=PAIRS, n_bins=STORED_N_BINS)
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.complete_path, ignore_errors=True)
        shutil.rmtree(cls.basic_path, ignore_errors=True)

    @staticmethod
    def _configure(widget, color, aggregation, log_y=False, log_color=False):
        widget.element_type_combo.setCurrentText("Throat")
        widget.property_combo.setCurrentText("radius")
        widget.color_combo.setCurrentText(color)
        widget.aggregation_combo.setCurrentText(aggregation)
        widget.bins_spinbox.setValue(VIEW_N_BINS)
        widget.cycle_combo.setCurrentIndex(0)
        widget.log_y_check.setChecked(log_y)
        widget.log_color_check.setChecked(log_color)
        widget._apply()
        return np.asarray(widget.view._image.image)

    def _raw_image(self, color, aggregation, log_y=False, log_color=False):
        diagnostics = Diagnostics(self.complete_path)
        try:
            widget = HeatmapWidget(diagnostics)
            return self._configure(widget, color, aggregation, log_y, log_color)
        finally:
            diagnostics.close()

    def _aggregate_image(self, color, aggregation, log_y=False, log_color=False):
        widget = HeatmapWidget(None, DiagnosticsAggregates(self.basic_path))
        return self._configure(widget, color, aggregation, log_y, log_color)

    def test_counts_agree_between_the_two_sources(self):
        raw = self._raw_image(COLOR_COUNT, AGG_SUM)
        aggregated = self._aggregate_image(COLOR_COUNT, AGG_SUM)

        self.assertEqual(aggregated.shape, (N_STEPS, VIEW_N_BINS))
        assert_allclose(aggregated, raw, rtol=1e-6)

    def test_sums_agree_between_the_two_sources(self):
        raw = self._raw_image("water_flow_rate", AGG_SUM)
        aggregated = self._aggregate_image("water_flow_rate", AGG_SUM)

        self.assertEqual(aggregated.shape, (N_STEPS, VIEW_N_BINS))
        assert_allclose(aggregated, raw, rtol=1e-5)

    def test_log_binned_counts_agree_between_the_two_sources(self):
        raw = self._raw_image(COLOR_COUNT, AGG_SUM, log_y=True)
        aggregated = self._aggregate_image(COLOR_COUNT, AGG_SUM, log_y=True)

        self.assertEqual(aggregated.shape, (N_STEPS, VIEW_N_BINS))
        assert_allclose(aggregated, raw, rtol=1e-6)

    def test_log_binned_sums_agree_between_the_two_sources(self):
        raw = self._raw_image("water_flow_rate", AGG_SUM, log_y=True)
        aggregated = self._aggregate_image("water_flow_rate", AGG_SUM, log_y=True)

        assert_allclose(aggregated, raw, rtol=1e-5)

    def test_log_and_linear_binning_produce_different_pictures(self):
        # guards against the log option silently doing nothing
        linear = self._aggregate_image(COLOR_COUNT, AGG_SUM, log_y=False)
        logarithmic = self._aggregate_image(COLOR_COUNT, AGG_SUM, log_y=True)

        self.assertEqual(linear.shape, logarithmic.shape)
        self.assertFalse(np.allclose(linear, logarithmic))
        # no element is lost by rebinning
        assert_allclose(linear.sum(axis=1), logarithmic.sum(axis=1))

    def test_the_log_color_scale_is_the_log_of_the_linear_one(self):
        linear = self._aggregate_image(COLOR_COUNT, AGG_SUM, log_color=False)
        logarithmic = self._aggregate_image(COLOR_COUNT, AGG_SUM, log_color=True)

        positive = linear > 0
        self.assertTrue(positive.any())
        assert_allclose(logarithmic[positive], np.log10(linear[positive]), rtol=1e-6)

    def test_the_log_color_scale_applies_to_the_raw_source_too(self):
        linear = self._raw_image(COLOR_COUNT, AGG_SUM, log_color=False)
        logarithmic = self._raw_image(COLOR_COUNT, AGG_SUM, log_color=True)

        positive = linear > 0
        self.assertTrue(positive.any())
        assert_allclose(logarithmic[positive], np.log10(linear[positive]), rtol=1e-6)

    def test_log_y_is_disabled_for_a_property_that_reaches_zero(self):
        widget = HeatmapWidget(None, DiagnosticsAggregates(self.basic_path))
        widget.element_type_combo.setCurrentText("Throat")

        widget.property_combo.setCurrentText("radius")
        self.assertTrue(widget.log_y_check.isEnabled())

        widget.log_y_check.setChecked(True)
        widget.property_combo.setCurrentText("water_saturation")
        self.assertFalse(widget.log_y_check.isEnabled())
        # and the stale selection is cleared, so Apply cannot ask for bins that do not exist
        self.assertFalse(widget.log_y_check.isChecked())

    def test_log_y_stays_available_on_the_raw_source(self):
        # the raw source can bin anything, it just drops the non positive values
        diagnostics = Diagnostics(self.complete_path)
        try:
            widget = HeatmapWidget(diagnostics)
            widget.element_type_combo.setCurrentText("Throat")
            widget.property_combo.setCurrentText("water_saturation")
            self.assertTrue(widget.log_y_check.isEnabled())
        finally:
            diagnostics.close()

    def test_the_aggregate_source_only_offers_the_precomputed_pairs(self):
        widget = HeatmapWidget(None, DiagnosticsAggregates(self.basic_path))

        element_types = [widget.element_type_combo.itemText(i) for i in range(widget.element_type_combo.count())]
        self.assertEqual(element_types, ["Pore", "Throat"])

        widget.element_type_combo.setCurrentText("Throat")
        properties = [widget.property_combo.itemText(i) for i in range(widget.property_combo.count())]
        self.assertEqual(properties, ["radius", "water_saturation"])
        colors = [widget.color_combo.itemText(i) for i in range(widget.color_combo.count())]
        self.assertEqual(colors, [COLOR_COUNT, "water_flow_rate"])

    def test_requesting_more_bins_than_were_stored_is_capped(self):
        widget = HeatmapWidget(None, DiagnosticsAggregates(self.basic_path))
        widget.element_type_combo.setCurrentText("Throat")
        widget.property_combo.setCurrentText("radius")
        widget.bins_spinbox.setValue(1000)
        widget._apply()

        self.assertEqual(np.asarray(widget.view._image.image).shape, (N_STEPS, STORED_N_BINS))


class DiagnosticsWidgetTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.basic_path = tempfile.mkdtemp(prefix="diagnostics_tabs_basic_")
        run_fake_simulation(
            lambda network: AggregateDiagnosticsWriter(network, cls.basic_path, pairs=PAIRS, n_bins=STORED_N_BINS)
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.basic_path, ignore_errors=True)

    def test_a_basic_run_only_shows_the_heatmap(self):
        widget = DiagnosticsWidget()
        widget.set_diagnostics(self.basic_path)

        self.assertEqual(widget.tabs.count(), 1)
        self.assertEqual(widget.tabs.tabText(0), "Heatmap")


if __name__ == "__main__":
    unittest.main()
