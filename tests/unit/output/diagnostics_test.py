import glob
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

from py_pore_flow.output.diagnostics import (
    CORNER_SHAPE,
    Diagnostics,
    DiagnosticsInfo,
    DiagnosticsStateLoader,
    DiagnosticsStorer,
    DiagnosticsWriter,
    FILE_EXTENSION,
)

N_CORNERS = CORNER_SHAPE[0]

# Properties dumped every step, in the exact order DiagnosticsWriter.save_state writes them.
# Each entry is (attribute_name, seed); the seed only makes the generated values distinct
# between properties so a wrong property/axis mapping would be caught.
THROAT_STEP_FLOAT = [
    ("water_saturation", 1),
    ("oil_multiphase_conductivity", 2),
    ("water_multiphase_conductivity", 3),
    ("oil_multiphase_conductance", 4),
    ("water_multiphase_conductance", 5),
]
THROAT_BOOL = [("water_filled", 6), ("trapped", 7)]  # (N,)
THROAT_CORNER = [("corner_water_layer", 8), ("corner_oil_layer", 9)]  # (N, N_CORNERS)
PORE_STEP_FLOAT = [
    ("water_saturation", 11),
    ("oil_multiphase_conductivity", 12),
    ("water_multiphase_conductivity", 13),
]
PORE_BOOL = [("water_filled", 16), ("trapped", 17)]  # (N,)
PORE_CORNER = [("corner_water_layer", 18), ("corner_oil_layer", 19)]  # (N, N_CORNERS)
THROAT_ENTRY_PRESSURE_SEED = 21
PORE_ENTRY_PRESSURE_SEED = 22


def gen_float(seed, idx, n):
    # exact in float64, distinct per (seed, idx, element)
    return seed * 1e6 + idx * 1e3 + np.arange(n, dtype=np.float64)


def gen_bool(seed, idx, n):
    return ((np.arange(n) + idx + seed) % 3) == 0


def gen_corner(seed, idx, n):
    base = np.arange(n * N_CORNERS).reshape(n, N_CORNERS)
    return ((base + idx + seed) % 3) == 0


def make_pn(n_throats, n_pores):
    pn = SimpleNamespace(throat=SimpleNamespace(), pore=SimpleNamespace())
    # DiagnosticsWriter.__init__ reads `.water_saturation.size` to size the storers.
    pn.throat.water_saturation = np.zeros(n_throats)
    pn.pore.water_saturation = np.zeros(n_pores)
    return pn


def set_step_state(pn, gstep, n_throats, n_pores):
    for element, n, floats, bools, corners in (
        (pn.throat, n_throats, THROAT_STEP_FLOAT, THROAT_BOOL, THROAT_CORNER),
        (pn.pore, n_pores, PORE_STEP_FLOAT, PORE_BOOL, PORE_CORNER),
    ):
        for attr, seed in floats:
            setattr(element, attr, gen_float(seed, gstep, n))
        for attr, seed in bools:
            setattr(element, attr, gen_bool(seed, gstep, n))
        for attr, seed in corners:
            setattr(element, attr, gen_corner(seed, gstep, n))


def set_cycle_state(pn, cycle, n_throats, n_pores):
    pn.throat.entry_pressure = gen_float(THROAT_ENTRY_PRESSURE_SEED, cycle, n_throats)
    pn.pore.entry_pressure = gen_float(PORE_ENTRY_PRESSURE_SEED, cycle, n_pores)


class DiagnosticsTests(unittest.TestCase):
    # use distinct counts so any throat/pore (or row/column) axis confusion is caught
    N_THROATS = 7
    N_PORES = 5

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix="diagnostics_test_")

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _write(self, cycles, chunk_size):
        """Run a fake simulation of `cycles` (list of step counts per cycle), writing
        every step with DiagnosticsStorer.N_STEPS patched to `chunk_size`."""
        pn = make_pn(self.N_THROATS, self.N_PORES)
        with mock.patch.object(DiagnosticsStorer, "N_STEPS", chunk_size):
            writer = DiagnosticsWriter(pn, self.path)
            gstep = 0
            for cycle, n_steps in enumerate(cycles):
                for _ in range(n_steps):
                    set_step_state(pn, gstep, self.N_THROATS, self.N_PORES)
                    writer.save_state()
                    gstep += 1
                set_cycle_state(pn, cycle, self.N_THROATS, self.N_PORES)
                writer.save_cycle_data()
            writer.close()

    def _assert_element_step(self, element, n, floats, bools, corners, gstep):
        for attr, seed in floats:
            value = getattr(element, attr)
            self.assertEqual(value.shape, (n,))
            assert_array_equal(value, gen_float(seed, gstep, n))
        for attr, seed in bools:
            value = getattr(element, attr)
            self.assertEqual(value.shape, (n,))
            assert_array_equal(value, gen_bool(seed, gstep, n))
        for attr, seed in corners:
            value = getattr(element, attr)
            self.assertEqual(value.shape, (n, N_CORNERS))
            assert_array_equal(value, gen_corner(seed, gstep, n))

    def _assert_step(self, cycle_step, gstep):
        self._assert_element_step(
            cycle_step.throat, self.N_THROATS, THROAT_STEP_FLOAT, THROAT_BOOL, THROAT_CORNER, gstep
        )
        self._assert_element_step(cycle_step.pore, self.N_PORES, PORE_STEP_FLOAT, PORE_BOOL, PORE_CORNER, gstep)

    def _assert_all(self, cycles):
        diagnostics = Diagnostics(self.path)
        offset = 0
        for cycle, n_steps in enumerate(cycles):
            for step in range(n_steps):
                self._assert_step(diagnostics.cycle[cycle][step], offset + step)
            cycle_data = diagnostics.cycle[cycle]
            assert_array_equal(
                cycle_data.throat.entry_pressure,
                gen_float(THROAT_ENTRY_PRESSURE_SEED, cycle, self.N_THROATS),
            )
            assert_array_equal(
                cycle_data.pore.entry_pressure,
                gen_float(PORE_ENTRY_PRESSURE_SEED, cycle, self.N_PORES),
            )
            offset += n_steps

    def test_info_roundtrip(self):
        cycles = [4, 5]
        self._write(cycles, chunk_size=100)

        info = DiagnosticsInfo(self.path)
        self.assertEqual(info.n_throats, self.N_THROATS)
        self.assertEqual(info.n_pores, self.N_PORES)
        self.assertEqual(info.cycle_n_steps, {0: 4, 1: 5})

    def test_single_chunk_roundtrip(self):
        # chunk_size larger than the total number of steps -> one chunk per property
        cycles = [4, 5]
        self._write(cycles, chunk_size=100)
        self._assert_all(cycles)

    def test_step_indexing_within_chunk(self):
        # Regression test for the data_view bug: every step (not just step 0) must read back
        # the values written for that step.
        cycles = [6]
        self._write(cycles, chunk_size=100)
        self._assert_all(cycles)

    def test_multi_chunk_roundtrip(self):
        # chunk_size smaller than the per-cycle step counts forces chunk rollover and the
        # loader's cross-chunk concatenation. 4 + 5 = 9 steps with chunk_size 3 yields a
        # boundary that lands exactly on a multiple of the chunk size.
        cycles = [4, 5]
        self._write(cycles, chunk_size=3)

        chunks = glob.glob(os.path.join(self.path, "diagnostics_throats_*" + FILE_EXTENSION))
        self.assertGreater(len(chunks), 1, "expected the step data to span multiple chunks")

        self._assert_all(cycles)

    def test_cycle_offsets_across_cycles(self):
        # Steps of later cycles must map to the correct global row, including across chunks.
        cycles = [2, 3, 4]
        self._write(cycles, chunk_size=2)
        self._assert_all(cycles)

    def test_named_cycle_accessors(self):
        cycles = [2, 3, 4]
        self._write(cycles, chunk_size=100)
        diagnostics = Diagnostics(self.path)

        # the convenience properties map to the right cycle indices
        self._assert_step(diagnostics.drainage[0], 0)
        self._assert_step(diagnostics.imbibition[0], 2)
        self._assert_step(diagnostics.second_drainage[0], 5)

    def test_corner_arrays_are_2d(self):
        # the corner geometry arrays must round-trip with their (N, N_CORNERS) shape intact
        self._write([3], chunk_size=100)
        diagnostics = Diagnostics(self.path)
        step = diagnostics.cycle[0][1]
        self.assertEqual(step.throat.corner_water_layer.shape, (self.N_THROATS, N_CORNERS))
        self.assertEqual(step.pore.corner_oil_layer.shape, (self.N_PORES, N_CORNERS))

    def test_close_truncates_to_written_rows(self):
        cycles = [4]
        self._write(cycles, chunk_size=100)

        # throats step file: 4 rows, 5 data blocks * N_THROATS columns, float64
        path = os.path.join(self.path, "diagnostics_throats_0" + FILE_EXTENSION)
        expected = 4 * (5 * self.N_THROATS) * np.dtype(np.float64).itemsize
        self.assertEqual(os.path.getsize(path), expected)

        # corner geometry file: 4 rows, 2 data blocks * N_THROATS * N_CORNERS columns, bool
        geo_path = os.path.join(self.path, "diagnostics_throats_geometry_0" + FILE_EXTENSION)
        expected_geo = 4 * (2 * self.N_THROATS * N_CORNERS) * np.dtype(np.bool_).itemsize
        self.assertEqual(os.path.getsize(geo_path), expected_geo)

    def test_arbitrary_item_shape_roundtrip(self):
        # DiagnosticsStorer / DiagnosticsStateLoader must support any trailing per-element shape,
        # not just () and (N_CORNERS,).
        n_elements = 6
        item_shape = (2, 3)
        n_data = 2
        n_steps = 4

        def value(step, data_index):
            base = step * 1000 + data_index * 100 + np.arange(n_elements * 2 * 3)
            return base.reshape((n_elements,) + item_shape).astype(np.float64)

        with mock.patch.object(DiagnosticsStorer, "N_STEPS", 100):
            storer = DiagnosticsStorer("shaped", n_elements, n_data, np.float64, self.path, item_shape=item_shape)
            for step in range(n_steps):
                storer.save_state([value(step, 0), value(step, 1)])
            storer.close()

        for step in range(n_steps):
            loader = DiagnosticsStateLoader(self.path, step, n_elements)
            data = loader.load_state("shaped", n_data, np.float64, item_shape)
            for data_index in range(n_data):
                view = loader.data_view(data, data_index, item_shape)
                self.assertEqual(view.shape, (n_elements,) + item_shape)
                assert_array_equal(view, value(step, data_index))


if __name__ == "__main__":
    unittest.main()
