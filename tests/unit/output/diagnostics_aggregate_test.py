import glob
import os
import shutil
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from py_pore_flow.network.network import initialize_pn, NetworkManager, PoreNetwork
from py_pore_flow.output.diagnostics.data.diagnostics_network import load_network
from py_pore_flow.output.diagnostics.diagnostics_aggregate import (
    AGG_COUNT,
    AGG_MEAN,
    AGG_SUM,
    SCALE_LINEAR,
    SCALE_LOG,
    AGGREGATE_FILE_NAME,
    AggregateDiagnosticsWriter,
    DiagnosticsAggregates,
    LEVEL_BASIC,
    LEVEL_COMPLETE,
    MANIFEST_FILE_NAME,
    diagnostics_level,
    read_manifest,
    resolve_n_bins,
)
from py_pore_flow.output.diagnostics.utils.diagnostics_file_cache import FILE_EXTENSION
from py_pore_flow.parameters.parameter_manager import ParameterManager
from py_pore_flow.simulation.simulation import calculate_initial_values

N_BINS = 100
CYCLES = [4, 3]

PAIRS = [
    ("throat", "radius", None),
    ("throat", "radius", "water_flow_rate"),
    ("throat", "water_saturation", "oil_flow_rate"),
]


def make_network():
    pn = PoreNetwork.from_statoil("tests/resources/a", "a")
    initialize_pn(pn)
    network = NetworkManager(pn)
    parameter_manager = ParameterManager()
    parameter_manager.update_initial_cas(network)
    calculate_initial_values(network)
    return network


# values chosen to be exact in float32, so the spool round trip introduces no error
def water_saturation(step, n):
    return ((np.arange(n) + step) % 64) / 64.0


def water_flow_rate(step, n):
    return (np.arange(n) % 32) * 0.5 + step


def oil_flow_rate(step, n):
    return (np.arange(n) % 16) * 0.25 + step * 2.0


def step_pc(step):
    return 10.0 ** (step + 1)


def step_sw(step):
    return 1.0 - 0.05 * step


class DiagnosticsAggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix="diagnostics_aggregate_test_")

        network = make_network()
        cls.n_throats = int(network.pn_core.throat.water_saturation.size)

        writer = AggregateDiagnosticsWriter(network, cls.path, pairs=PAIRS, n_bins=N_BINS)
        step = 0
        for n_steps in CYCLES:
            for _ in range(n_steps):
                throat = network.pn_core.throat
                throat.water_saturation = water_saturation(step, cls.n_throats)
                throat.water_flow_rate = water_flow_rate(step, cls.n_throats)
                throat.oil_flow_rate = oil_flow_rate(step, cls.n_throats)
                writer.save_state(step_pc(step), step_sw(step))
                step += 1
            writer.save_cycle_data()
        writer.close()

        cls.radius = np.asarray(load_network(cls.path)["throat.radius"], dtype=np.float64)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def setUp(self):
        self.aggregates = DiagnosticsAggregates(self.path)

    def _radius_edges(self):
        return np.linspace(self.radius.min(), self.radius.max(), N_BINS + 1)

    def _reference(self, weights_of_step, n_steps, step_offset, bins):
        """Direct np.histogram over the raw values, one row per step."""
        counts = np.zeros((n_steps, len(bins) - 1 if np.ndim(bins) else bins))
        sums = np.zeros_like(counts)
        for local_step in range(n_steps):
            weights = weights_of_step(step_offset + local_step, self.n_throats)
            counts[local_step], _ = np.histogram(self.radius, bins=bins)
            sums[local_step], _ = np.histogram(self.radius, bins=bins, weights=weights)
        return counts, sums

    # ---------------------------------------------------------------- artefacts

    def test_only_the_aggregates_survive_the_run(self):
        self.assertTrue(os.path.exists(os.path.join(self.path, AGGREGATE_FILE_NAME)))
        self.assertTrue(os.path.exists(os.path.join(self.path, MANIFEST_FILE_NAME)))
        # the spool exists only to get the exact ranges and is deleted afterwards
        self.assertEqual(glob.glob(os.path.join(self.path, "diagnostics_spool_*" + FILE_EXTENSION)), [])

    def test_manifest_describes_the_level_and_the_stored_pairs(self):
        manifest = read_manifest(self.path)
        self.assertEqual(manifest["level"], LEVEL_BASIC)
        self.assertEqual(diagnostics_level(self.path), LEVEL_BASIC)
        self.assertEqual(manifest["cycle_n_steps"], {"0": 4, "1": 3})

        self.assertEqual(self.aggregates.y_property_names("throat"), ["radius", "water_saturation"])
        self.assertEqual(self.aggregates.color_property_names("throat", "radius"), ["water_flow_rate"])
        self.assertEqual(self.aggregates.color_property_names("throat", "water_saturation"), ["oil_flow_rate"])

    def test_a_directory_without_a_manifest_is_a_complete_one(self):
        empty = tempfile.mkdtemp(prefix="diagnostics_legacy_test_")
        try:
            self.assertIsNone(read_manifest(empty))
            self.assertEqual(diagnostics_level(empty), LEVEL_COMPLETE)
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    # ------------------------------------------------------------- correctness

    def test_count_matches_a_direct_histogram(self):
        edges = self._radius_edges()
        matrix, _ = self.aggregates.matrix(0, "throat", "radius", None, AGG_COUNT, N_BINS)

        expected, _ = self._reference(water_flow_rate, CYCLES[0], 0, edges)
        self.assertEqual(matrix.shape, (CYCLES[0], N_BINS))
        assert_allclose(matrix, expected, rtol=1e-6)

    def test_sum_matches_a_direct_histogram(self):
        edges = self._radius_edges()
        matrix, _ = self.aggregates.matrix(0, "throat", "radius", "water_flow_rate", AGG_SUM, N_BINS)

        counts, sums = self._reference(water_flow_rate, CYCLES[0], 0, edges)
        assert_allclose(matrix, np.where(counts > 0, sums, np.nan), rtol=1e-5)

    def test_mean_matches_a_direct_histogram(self):
        edges = self._radius_edges()
        matrix, _ = self.aggregates.matrix(0, "throat", "radius", "water_flow_rate", AGG_MEAN, N_BINS)

        counts, sums = self._reference(water_flow_rate, CYCLES[0], 0, edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = np.where(counts > 0, sums / counts, np.nan)
        assert_allclose(matrix, expected, rtol=1e-5)

    def test_later_cycles_use_their_own_steps(self):
        edges = self._radius_edges()
        matrix, _ = self.aggregates.matrix(1, "throat", "radius", "water_flow_rate", AGG_SUM, N_BINS)

        counts, sums = self._reference(water_flow_rate, CYCLES[1], CYCLES[0], edges)
        self.assertEqual(matrix.shape, (CYCLES[1], N_BINS))
        assert_allclose(matrix, np.where(counts > 0, sums, np.nan), rtol=1e-5)

    def test_rebinning_matches_binning_the_raw_data_directly(self):
        # 100 stored bins merged down to 50, against a direct 50 bin histogram
        coarse = 50
        low, high = self.radius.min(), self.radius.max()
        matrix, value_range = self.aggregates.matrix(0, "throat", "radius", "water_flow_rate", AGG_SUM, coarse)

        counts = np.zeros((CYCLES[0], coarse))
        sums = np.zeros_like(counts)
        for step in range(CYCLES[0]):
            weights = water_flow_rate(step, self.n_throats)
            counts[step], _ = np.histogram(self.radius, bins=coarse, range=(low, high))
            sums[step], _ = np.histogram(self.radius, bins=coarse, range=(low, high), weights=weights)

        self.assertEqual(matrix.shape, (CYCLES[0], coarse))
        assert_allclose(value_range, (low, high), rtol=1e-12)
        assert_allclose(matrix, np.where(counts > 0, sums, np.nan), rtol=1e-5)

    def test_a_dynamic_property_can_be_the_binned_axis(self):
        # water_saturation changes every step, so its exact range is only known once the run
        # is over: this is what the spool buys
        low, high = self.aggregates.value_range(0, "throat", "water_saturation")
        expected_values = [water_saturation(step, self.n_throats) for step in range(CYCLES[0])]
        assert_allclose(low, min(values.min() for values in expected_values))
        assert_allclose(high, max(values.max() for values in expected_values))

        edges = np.linspace(low, high, N_BINS + 1)
        matrix, _ = self.aggregates.matrix(0, "throat", "water_saturation", "oil_flow_rate", AGG_SUM, N_BINS)
        for step, values in enumerate(expected_values):
            expected_sum, _ = np.histogram(values, bins=edges, weights=oil_flow_rate(step, self.n_throats))
            expected_count, _ = np.histogram(values, bins=edges)
            assert_allclose(matrix[step], np.where(expected_count > 0, expected_sum, np.nan), rtol=1e-5)

    # -------------------------------------------------------------- log scale

    def test_log_bins_are_stored_only_for_properties_that_stay_positive(self):
        # radius comes from the network and is strictly positive
        self.assertTrue(self.aggregates.log_scale_available("throat", "radius"))
        self.assertEqual(self.aggregates.scales("throat", "radius"), [SCALE_LINEAR, SCALE_LOG])

        # water_saturation reaches exactly zero, so log bins are impossible
        self.assertFalse(self.aggregates.log_scale_available("throat", "water_saturation"))
        self.assertEqual(self.aggregates.scales("throat", "water_saturation"), [SCALE_LINEAR])

    def test_log_bins_match_a_direct_log_histogram(self):
        edges = np.logspace(np.log10(self.radius.min()), np.log10(self.radius.max()), N_BINS + 1)
        matrix, _ = self.aggregates.matrix(0, "throat", "radius", "water_flow_rate", AGG_SUM, N_BINS, scale=SCALE_LOG)

        counts, sums = self._reference(water_flow_rate, CYCLES[0], 0, edges)
        self.assertEqual(matrix.shape, (CYCLES[0], N_BINS))
        assert_allclose(matrix, np.where(counts > 0, sums, np.nan), rtol=1e-5)

    def test_log_rebinning_matches_binning_the_raw_data_directly(self):
        # 100 stored log bins merged down to 50, against a direct 50 log bin histogram
        coarse = 50
        low, high = self.radius.min(), self.radius.max()
        edges = np.logspace(np.log10(low), np.log10(high), coarse + 1)

        matrix, value_range = self.aggregates.matrix(
            0, "throat", "radius", "water_flow_rate", AGG_SUM, coarse, scale=SCALE_LOG
        )

        counts, sums = self._reference(water_flow_rate, CYCLES[0], 0, edges)
        self.assertEqual(matrix.shape, (CYCLES[0], coarse))
        assert_allclose(value_range, (low, high), rtol=1e-12)
        assert_allclose(matrix, np.where(counts > 0, sums, np.nan), rtol=1e-5)

    def test_log_and_linear_bins_hold_the_same_elements_in_total(self):
        # rebinning changes where the elements land, never how many there are
        linear, _ = self.aggregates.matrix(0, "throat", "radius", None, AGG_COUNT, N_BINS, scale=SCALE_LINEAR)
        logarithmic, _ = self.aggregates.matrix(0, "throat", "radius", None, AGG_COUNT, N_BINS, scale=SCALE_LOG)

        assert_allclose(linear.sum(axis=1), logarithmic.sum(axis=1))
        assert_allclose(linear.sum(axis=1), np.full(CYCLES[0], self.n_throats))

    def test_the_log_range_is_reported_from_the_log_edges(self):
        low, high = self.aggregates.value_range(0, "throat", "radius", SCALE_LOG)
        assert_allclose((low, high), (self.radius.min(), self.radius.max()), rtol=1e-12)
        self.assertEqual(self.aggregates.stored_n_bins(0, "throat", "radius", SCALE_LOG), N_BINS)

    def test_step_axes_are_stored_per_cycle(self):
        offset = 0
        for cycle, n_steps in enumerate(CYCLES):
            assert_allclose(self.aggregates.step_axis(cycle, "pc"), [step_pc(offset + step) for step in range(n_steps)])
            assert_allclose(self.aggregates.step_axis(cycle, "sw"), [step_sw(offset + step) for step in range(n_steps)])
            offset += n_steps

    def test_an_unknown_step_axis_is_reported_as_missing(self):
        self.assertIsNone(self.aggregates.step_axis(0, "not_a_scalar"))

    def test_stored_resolution_is_reported(self):
        self.assertEqual(self.aggregates.stored_n_bins(0, "throat", "radius"), N_BINS)
        self.assertEqual(self.aggregates.cycle_n_steps, {0: CYCLES[0], 1: CYCLES[1]})
        self.assertEqual(self.aggregates.n_cycles, 2)


class ResolveBinsTests(unittest.TestCase):
    def test_bins_are_capped_by_the_element_count(self):
        self.assertEqual(resolve_n_bins(10), 16)  # floor
        self.assertEqual(resolve_n_bins(400), 100)
        self.assertEqual(resolve_n_bins(10**6), 500)  # ceiling


if __name__ == "__main__":
    unittest.main()
