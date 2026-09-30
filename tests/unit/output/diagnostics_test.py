import glob
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

from py_pore_flow.network.network import initialize_pn, NetworkManager, PoreNetwork
from py_pore_flow.output.diagnostics import (
    CORNER_SHAPE,
    STEP_SCALAR_NAMES,
    ConcatenatedMemMaps,
    Diagnostics,
    DiagnosticsFileCache,
    DiagnosticsInfo,
    DiagnosticsStateLoader,
    DiagnosticsStorer,
    DiagnosticsWriter,
    FILE_EXTENSION,
    NETWORK_ARRAYS_NAME,
    THROATS_FLOATS_NAME,
    THROATS_FLOATS_NDATA,
)
from py_pore_flow.parameters.parameter_manager import ParameterManager
from py_pore_flow.simulation.simulation import calculate_initial_values

N_CORNERS = CORNER_SHAPE[0]

# Properties dumped every step, in the exact order DiagnosticsWriter.save_state writes them.
# Each entry is (attribute_name, seed); the seed only makes the generated values distinct
# between properties so a wrong property/axis mapping would be caught.
THROAT_FLOAT = [
    ("water_saturation", 1),
    ("oil_multiphase_conductivity", 2),
    ("water_multiphase_conductivity", 3),
    ("oil_multiphase_conductance", 4),
    ("water_multiphase_conductance", 5),
    ("oil_pressure", 6),
    ("water_pressure", 7),
    ("oil_flow_rate", 8),
    ("water_flow_rate", 9),
]
THROAT_BOOL = [
    ("water_filled", 10),
    ("trapped", 11),
    ("filled", 12),
]
THROAT_CORNERS_BOOL = [
    ("corner_water_layer", 13),
    ("corner_oil_layer", 14),
]
THROAT_CORNERS_FLOAT = [
    ("b", 15),
    ("bo", 16),
    ("hinging_contact_angle", 17),
    ("hinging_contact_angle_o", 18),
]
PORE_FLOAT = [
    ("water_saturation", 19),
    ("oil_multiphase_conductivity", 20),
    ("water_multiphase_conductivity", 21),
    ("oil_pressure", 22),
    ("water_pressure", 23),
]
PORE_BOOL = [
    ("water_filled", 24),
    ("trapped", 25),
    ("filled", 26),
]
PORE_CORNERS_BOOL = [
    ("corner_water_layer", 27),
    ("corner_oil_layer", 28),
]
PORE_CORNERS_FLOAT = [
    ("b", 29),
    ("bo", 30),
    ("hinging_contact_angle", 31),
    ("hinging_contact_angle_o", 32),
]

THROAT_ENTRY_PRESSURE_SEED = 21
PORE_ENTRY_PRESSURE_SEED = 22


def gen_float(seed, idx, n):
    # exact in float64, distinct per (seed, idx, element)
    return seed * 1e6 + idx * 1e3 + np.arange(n, dtype=np.float64)


def gen_bool(seed, idx, n):
    return ((np.arange(n) + idx + seed) % 3) == 0


def gen_corner_bool(seed, idx, n):
    base = np.arange(n * N_CORNERS).reshape(n, N_CORNERS)
    return ((base + idx + seed) % 3) == 0


def gen_corner_float(seed, idx, n):
    base = np.arange(n * N_CORNERS, dtype=np.float64).reshape(n, N_CORNERS)
    return seed * 1e6 + idx * 1e3 + base


def make_network():
    pn = PoreNetwork.from_statoil("tests/resources/a", "a")
    initialize_pn(pn)
    network = NetworkManager(pn)
    parameter_manager = ParameterManager()
    parameter_manager.update_initial_cas(network)
    calculate_initial_values(network)
    return network


def set_step_state(pn, gstep, n_throats, n_pores):
    for element, n, floats, bools, corners_bools, corners_floats in (
        (pn.throat, n_throats, THROAT_FLOAT, THROAT_BOOL, THROAT_CORNERS_BOOL, THROAT_CORNERS_FLOAT),
        (pn.pore, n_pores, PORE_FLOAT, PORE_BOOL, PORE_CORNERS_BOOL, PORE_CORNERS_FLOAT),
    ):
        for attr, seed in floats:
            setattr(element, attr, gen_float(seed, gstep, n))
        for attr, seed in bools:
            setattr(element, attr, gen_bool(seed, gstep, n))
        for attr, seed in corners_bools:
            setattr(element, attr, gen_corner_bool(seed, gstep, n))
        for attr, seed in corners_floats:
            setattr(element, attr, gen_corner_float(seed, gstep, n))


def step_pc(gstep):
    return 10.0 ** (gstep + 1)


def step_sw(gstep):
    return 1.0 - 0.05 * gstep


def set_cycle_state(pn, cycle, n_throats, n_pores):
    pn.throat.entry_pressure = gen_float(THROAT_ENTRY_PRESSURE_SEED, cycle, n_throats)
    pn.pore.entry_pressure = gen_float(PORE_ENTRY_PRESSURE_SEED, cycle, n_pores)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp(prefix="diagnostics_test_")

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _write(self, cycles, chunk_size):
        """Run a fake simulation of `cycles` (list of step counts per cycle), writing
        every step with DiagnosticsStorer.N_STEPS patched to `chunk_size`."""
        network = make_network()
        self.n_pores = len(network.pn_core.pore.id)
        self.n_throats = len(network.pn_core.throat.id)
        with mock.patch.object(DiagnosticsStorer, "N_STEPS", chunk_size):
            writer = DiagnosticsWriter(network, self.path)
            gstep = 0
            for cycle, n_steps in enumerate(cycles):
                for _ in range(n_steps):
                    set_step_state(network.pn_core, gstep, self.n_throats, self.n_pores)
                    writer.save_state(step_pc(gstep), step_sw(gstep))
                    gstep += 1
                set_cycle_state(network.pn_core, cycle, self.n_throats, self.n_pores)
                writer.save_cycle_data()
            writer.close()

    def _assert_element_step(self, element, n, floats, bools, corners_bools, corners_floats, gstep):
        for attr, seed in floats:
            value = getattr(element, attr)
            self.assertEqual(value.shape, (n,))
            assert_array_equal(value, gen_float(seed, gstep, n))
        for attr, seed in bools:
            value = getattr(element, attr)
            self.assertEqual(value.shape, (n,))
            assert_array_equal(value, gen_bool(seed, gstep, n))
        for attr, seed in corners_bools:
            value = getattr(element, attr)
            self.assertEqual(value.shape, (n, N_CORNERS))
            assert_array_equal(value, gen_corner_bool(seed, gstep, n))
        for attr, seed in corners_floats:
            value = getattr(element, attr)
            self.assertEqual(value.shape, (n, N_CORNERS))
            assert_array_equal(value, gen_corner_float(seed, gstep, n))

    def _assert_step(self, cycle_step, gstep):
        self._assert_element_step(
            cycle_step.throat,
            self.n_throats,
            THROAT_FLOAT,
            THROAT_BOOL,
            THROAT_CORNERS_BOOL,
            THROAT_CORNERS_FLOAT,
            gstep
        )
        self._assert_element_step(
            cycle_step.pore,
            self.n_pores,
            PORE_FLOAT,
            PORE_BOOL,
            PORE_CORNERS_BOOL,
            PORE_CORNERS_FLOAT,
            gstep
        )

    def _assert_all(self, cycles):
        diagnostics = Diagnostics(self.path)
        offset = 0
        for cycle, n_steps in enumerate(cycles):
            for step in range(n_steps):
                self._assert_step(diagnostics.cycle[cycle][step], offset + step)
            cycle_data = diagnostics.cycle[cycle]
            assert_array_equal(
                cycle_data.throat.entry_pressure,
                gen_float(THROAT_ENTRY_PRESSURE_SEED, cycle, self.n_throats),
            )
            assert_array_equal(
                cycle_data.pore.entry_pressure,
                gen_float(PORE_ENTRY_PRESSURE_SEED, cycle, self.n_pores),
            )
            offset += n_steps

    def test_info_roundtrip(self):
        cycles = [4, 5]
        self._write(cycles, chunk_size=100)

        info = DiagnosticsInfo(self.path)
        self.assertEqual(info.n_throats, self.n_throats)
        self.assertEqual(info.n_pores, self.n_pores)
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

        chunks = glob.glob(os.path.join(self.path, "diagnostics_throats_floats_*" + FILE_EXTENSION))
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
        self.assertEqual(step.throat.corner_water_layer.shape, (self.n_throats, N_CORNERS))
        self.assertEqual(step.pore.corner_oil_layer.shape, (self.n_pores, N_CORNERS))

    def test_close_truncates_to_written_rows(self):
        cycles = [4]
        self._write(cycles, chunk_size=100)

        # throats float file: 4 rows, 5 data blocks * n_throats columns, float64
        path = os.path.join(self.path, "diagnostics_throats_floats_0" + FILE_EXTENSION)
        expected = 4 * (len(THROAT_FLOAT) * self.n_throats) * np.dtype(np.float64).itemsize
        self.assertEqual(os.path.getsize(path), expected)

        # corner bool file: 4 rows, 2 data blocks * n_throats * N_CORNERS columns, bool
        corners_bools_path = os.path.join(self.path, "diagnostics_throats_corners_bools_0" + FILE_EXTENSION)
        expected_corners_bools = 4 * (len(THROAT_CORNERS_BOOL) * self.n_throats * N_CORNERS) * np.dtype(np.bool_).itemsize
        self.assertEqual(os.path.getsize(corners_bools_path), expected_corners_bools)

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

    def _chunk_paths(self, name):
        paths = []
        index = 0
        while True:
            path = os.path.join(self.path, f"diagnostics_{name}_{index}" + FILE_EXTENSION)
            if not os.path.exists(path):
                return paths
            paths.append(path)
            index += 1

    def test_chunked_rows_match_a_full_concatenation(self):
        # The loader used to np.concatenate every chunk on each read, which copied the whole
        # run into RAM. The lazy row view must return exactly the same values, including
        # across a chunk boundary.
        self._write([4, 5], chunk_size=3)

        chunk_paths = self._chunk_paths(THROATS_FLOATS_NAME)
        self.assertGreater(len(chunk_paths), 1, "expected the step data to span multiple chunks")

        cols = self.n_throats * THROATS_FLOATS_NDATA
        concatenated = np.concatenate(
            [np.fromfile(path, dtype=np.float64).reshape(-1, cols) for path in chunk_paths], axis=0
        )

        rows = DiagnosticsFileCache(self.path).rows(
            THROATS_FLOATS_NAME, self.n_throats, THROATS_FLOATS_NDATA, np.float64
        )
        self.assertEqual(rows.shape, concatenated.shape)
        for step in range(concatenated.shape[0]):
            assert_array_equal(rows[step], concatenated[step])
            assert_array_equal(rows[step, 0 : self.n_throats], concatenated[step, 0 : self.n_throats])

    def test_chunk_files_are_opened_once_per_storer(self):
        # Regression guard for the read path: every property read used to rescan the whole
        # diagnostics directory, 12 times per read.
        self._write([4, 5], chunk_size=3)

        with mock.patch("py_pore_flow.output.diagnostics.os.listdir", wraps=os.listdir) as listdir:
            diagnostics = Diagnostics(self.path)
            diagnostics.cycle[0][0]
            after_first_step = listdir.call_count
            self.assertGreater(after_first_step, 0)

            for step in range(4):
                cycle_step = diagnostics.cycle[0][step]
                for _ in range(5):
                    cycle_step.throat.water_saturation
                    cycle_step.pore.oil_pressure
            for step in range(5):
                diagnostics.cycle[1][step].throat.oil_flow_rate
            diagnostics.cycle[1].throat.entry_pressure

            self.assertEqual(listdir.call_count, after_first_step)

    def test_chunk_rows_are_capped_by_the_byte_budget(self):
        # A chunk file is preallocated in full, so a step count alone would make a large
        # network reserve several GB per storer before the first step is written.
        n_elements = 200000
        storer = DiagnosticsStorer("budget", n_elements, THROATS_FLOATS_NDATA, np.float64, self.path)
        row_bytes = n_elements * THROATS_FLOATS_NDATA * np.dtype(np.float64).itemsize

        self.assertGreaterEqual(storer._chunk_n_steps, 1)
        self.assertLess(storer._chunk_n_steps, DiagnosticsStorer.N_STEPS)
        self.assertLessEqual(storer._chunk_n_steps * row_bytes, DiagnosticsStorer.TARGET_CHUNK_BYTES)

        # a small network stays on the step cap
        small = DiagnosticsStorer("budget_small", 10, THROATS_FLOATS_NDATA, np.float64, self.path)
        self.assertEqual(small._chunk_n_steps, DiagnosticsStorer.N_STEPS)

    def test_network_properties_are_stored_as_arrays(self):
        self._write([3], chunk_size=100)

        self.assertTrue(os.path.exists(os.path.join(self.path, NETWORK_ARRAYS_NAME)))

        network = Diagnostics(self.path).network
        self.assertIn("throat.radius", network)
        self.assertIsInstance(network["throat.radius"], np.ndarray)
        self.assertEqual(network["throat.radius"].shape, (self.n_throats,))
        # the sparse conns accessor cannot be densified and must be skipped, not raise
        self.assertNotIn("pore.conns", network)

    def test_step_scalars_are_saved_per_step(self):
        cycles = [4, 5]
        self._write(cycles, chunk_size=3)

        diagnostics = Diagnostics(self.path)
        self.assertEqual(diagnostics.step_scalars.shape, (sum(cycles), len(STEP_SCALAR_NAMES)))

        offset = 0
        for cycle, n_steps in enumerate(cycles):
            expected_pc = [step_pc(offset + step) for step in range(n_steps)]
            expected_sw = [step_sw(offset + step) for step in range(n_steps)]
            assert_array_equal(diagnostics.step_axis(cycle, "pc"), expected_pc)
            assert_array_equal(diagnostics.step_axis(cycle, "sw"), expected_sw)
            offset += n_steps

    def test_step_scalars_are_absent_from_diagnostics_that_predate_them(self):
        self._write([3], chunk_size=100)
        os.remove(os.path.join(self.path, "step_scalars" + FILE_EXTENSION))

        diagnostics = Diagnostics(self.path)
        self.assertEqual(diagnostics.step_scalars.size, 0)
        self.assertIsNone(diagnostics.step_axis(0, "pc"))
        self.assertIsNone(diagnostics.step_axis(0, "not_a_scalar"))

    def test_close_releases_the_chunk_files(self):
        # the viewer has to let the mappings go: on Windows they lock the files, and the
        # diagnostics directory is moved when the project is saved
        self._write([3], chunk_size=2)

        diagnostics = Diagnostics(self.path)
        diagnostics.cycle[0][0].throat.water_saturation
        diagnostics.network
        diagnostics.close()

        shutil.rmtree(self.path)
        self.assertFalse(os.path.exists(self.path))


# chunk row counts of the pieces a ConcatenatedMemMaps is built from: uneven on purpose, so
# a boundary falls in the middle of the slices the tests take
CONCATENATED_PIECES = [3, 4, 2]
CONCATENATED_COLS = 5


class ConcatenatedMemMapsTests(unittest.TestCase):
    """Every supported index has to give what np.concatenate would have given."""

    def setUp(self):
        start = 0
        self.pieces = []
        for n_rows in CONCATENATED_PIECES:
            size = n_rows * CONCATENATED_COLS
            self.pieces.append(np.arange(start, start + size).reshape(n_rows, CONCATENATED_COLS))
            start += size

        self.expected = np.concatenate(self.pieces, axis=0)
        self.rows = ConcatenatedMemMaps(self.pieces)

    def _assert_matches(self, key):
        assert_array_equal(np.asarray(self.rows[key]), self.expected[key])

    def test_shape_and_length_match_the_concatenation(self):
        self.assertEqual(self.rows.shape, self.expected.shape)
        self.assertEqual(len(self.rows), len(self.expected))

    def test_iterating_yields_every_row_in_order(self):
        assert_array_equal(np.asarray(list(self.rows)), self.expected)

    def test_a_single_row_matches_whichever_piece_holds_it(self):
        for step in range(len(self.expected)):
            self._assert_matches(step)

    def test_a_negative_row_counts_back_from_the_end(self):
        for step in range(-len(self.expected), 0):
            self._assert_matches(step)

    def test_a_row_past_the_end_raises(self):
        # a negative index below -len is NOT checked and still reads from the last piece
        with self.assertRaises(IndexError):
            self.rows[len(self.expected)]

    def test_a_row_and_a_column_slice_is_the_read_the_loaders_make(self):
        # DiagnosticsStateLoader.data_view indexes exactly like this, once per property
        for step in range(len(self.expected)):
            self._assert_matches((step, slice(1, 4)))
            self._assert_matches((step, 2))

    def test_a_column_slice_over_every_row_keeps_both_axes(self):
        # get_full_data reads a column block of every step at once; the column key used to
        # be applied to the rows a second time, which silently returned the wrong block
        self._assert_matches((slice(None), slice(1, 4)))
        self._assert_matches((slice(1, 8), slice(1, 4)))
        self._assert_matches((slice(1, 8), 2))

    def test_a_row_slice_spans_the_pieces(self):
        for start in range(len(self.expected) + 1):
            for stop in range(start, len(self.expected) + 1):
                self._assert_matches(slice(start, stop))

    def test_an_empty_slice_keeps_the_column_axis(self):
        self._assert_matches(slice(3, 3))

    def test_a_slice_of_a_slice_still_matches(self):
        sliced = self.rows[2:8]
        assert_array_equal(np.asarray(sliced[1:4]), self.expected[2:8][1:4])
        assert_array_equal(np.asarray(sliced[1]), self.expected[2:8][1])
        assert_array_equal(np.asarray(sliced[:, 1:4]), self.expected[2:8][:, 1:4])

    def test_a_numpy_integer_indexes_like_a_python_one(self):
        # isinstance(np.int64(0), int) is False, and an unhandled key used to return None
        for dtype in (np.int32, np.int64):
            self._assert_matches(dtype(4))
            assert_array_equal(np.asarray(self.rows[dtype(4), 1:4]), self.expected[4, 1:4])

    def test_an_unsupported_key_raises_instead_of_returning_none(self):
        for key in ([1, 2], Ellipsis, "step", 1.5):
            with self.assertRaises(NotImplementedError):
                self.rows[key]

    def test_a_step_other_than_one_is_refused(self):
        with self.assertRaises(NotImplementedError):
            self.rows[::2]


if __name__ == "__main__":
    unittest.main()
