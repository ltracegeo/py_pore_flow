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

from py_pore_flow.graphic.diagnostics.diagnostics_heatmap_widget import (
    AGG_SUM,
    COLOR_COUNT,
    X_AXIS_STEP,
    HeatmapWidget,
)
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


# one decade a step, so a log X grid of N_STEPS bins holds exactly one step per column
def step_pc(step):
    return 10.0 ** (step + 1)


# reaches exactly zero, so Sw can never be placed on a log X axis
def step_sw(step):
    return 0.75 - 0.25 * step


X_AXIS_PC = "capillary pressure (Pc)"
X_AXIS_SW = "water saturation (Sw)"


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
        writer.save_state(step_pc(step), step_sw(step))
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
    def _configure(widget, color, aggregation, log_y=False, log_color=False, x_axis=X_AXIS_STEP, log_x=False):
        widget.element_type_combo.setCurrentText("Throat")
        widget.property_combo.setCurrentText("radius")
        widget.color_combo.setCurrentText(color)
        widget.aggregation_combo.setCurrentText(aggregation)
        widget.bins_spinbox.setValue(VIEW_N_BINS)
        widget.cycle_combo.setCurrentIndex(0)
        widget.log_y_check.setChecked(log_y)
        widget.log_color_check.setChecked(log_color)
        widget.x_axis_combo.setCurrentText(x_axis)
        widget.log_x_check.setChecked(log_x)
        widget._apply()
        return np.asarray(widget.view._image.image)

    def _raw_image(self, color, aggregation, log_y=False, log_color=False, x_axis=X_AXIS_STEP, log_x=False):
        diagnostics = Diagnostics(self.complete_path)
        try:
            widget = HeatmapWidget(diagnostics)
            return self._configure(widget, color, aggregation, log_y, log_color, x_axis, log_x)
        finally:
            diagnostics.close()

    def _aggregate_image(self, color, aggregation, log_y=False, log_color=False, x_axis=X_AXIS_STEP, log_x=False):
        widget = HeatmapWidget(None, DiagnosticsAggregates(self.basic_path))
        return self._configure(widget, color, aggregation, log_y, log_color, x_axis, log_x)

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

    # ------------------------------------------------------------------ X axis

    def test_a_log_x_axis_keeps_one_step_per_column(self):
        # Pc rises a decade a step, so N_STEPS log spaced columns hold exactly one step each
        # and the picture is the step indexed one, only differently spaced
        by_step = self._aggregate_image(COLOR_COUNT, AGG_SUM)
        by_pc = self._aggregate_image(COLOR_COUNT, AGG_SUM, x_axis=X_AXIS_PC, log_x=True)

        assert_allclose(by_pc, by_step)

    def test_a_linear_x_axis_groups_the_crowded_steps(self):
        # on a linear Pc grid the first three steps share the first column and the last step
        # takes the last one; the two columns in between are filled from their nearest
        by_step = self._aggregate_image(COLOR_COUNT, AGG_SUM)
        by_pc = self._aggregate_image(COLOR_COUNT, AGG_SUM, x_axis=X_AXIS_PC, log_x=False)

        crowded = by_step[:3].mean(axis=0)
        assert_allclose(by_pc[0], crowded)
        assert_allclose(by_pc[1], crowded)
        assert_allclose(by_pc[2], by_step[3])
        assert_allclose(by_pc[3], by_step[3])

    def test_the_x_axis_agrees_between_the_two_sources(self):
        raw = self._raw_image("water_flow_rate", AGG_SUM, x_axis=X_AXIS_PC, log_x=True)
        aggregated = self._aggregate_image("water_flow_rate", AGG_SUM, x_axis=X_AXIS_PC, log_x=True)

        assert_allclose(aggregated, raw, rtol=1e-5)

    def test_log_x_follows_the_sign_of_the_selected_coordinate(self):
        widget = HeatmapWidget(None, DiagnosticsAggregates(self.basic_path))

        widget.x_axis_combo.setCurrentText(X_AXIS_STEP)
        self.assertFalse(widget.log_x_check.isEnabled())

        widget.x_axis_combo.setCurrentText(X_AXIS_PC)
        self.assertTrue(widget.log_x_check.isEnabled())

        widget.log_x_check.setChecked(True)
        # Sw reaches zero, so it cannot go on a log axis and the stale choice is cleared
        widget.x_axis_combo.setCurrentText(X_AXIS_SW)
        self.assertFalse(widget.log_x_check.isEnabled())
        self.assertFalse(widget.log_x_check.isChecked())

    def test_the_x_axis_falls_back_to_the_step_index_when_no_scalar_was_saved(self):
        stripped = tempfile.mkdtemp(prefix="diagnostics_no_scalars_")
        try:
            for name in os.listdir(self.complete_path):
                if not name.startswith("step_scalars"):
                    shutil.copy(os.path.join(self.complete_path, name), stripped)

            diagnostics = Diagnostics(stripped)
            try:
                widget = HeatmapWidget(diagnostics)
                widget.x_axis_combo.setCurrentText(X_AXIS_PC)
                self.assertFalse(widget.log_x_check.isEnabled())

                image = self._configure(widget, COLOR_COUNT, AGG_SUM, x_axis=X_AXIS_PC)
                assert_allclose(image, self._raw_image(COLOR_COUNT, AGG_SUM))
            finally:
                diagnostics.close()
        finally:
            shutil.rmtree(stripped, ignore_errors=True)

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
