"""Widget-layer pieces that carry the per-step cost, covered without a render window."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from numpy.testing import assert_array_equal
from PySide2.QtCore import Qt
from vtk.util.numpy_support import vtk_to_numpy

from py_pore_flow.graphic.diagnostics.diagnostics_crossplot_widget import CrossPlotView, MAX_PLOTTED_POINTS
from py_pore_flow.graphic.diagnostics.diagnostics_network_widget import _ExpandedColors
from py_pore_flow.graphic.diagnostics.diagnostics_tabs import DiagnosticsTableModel


class ExpandedColorsTests(unittest.TestCase):
    """The 3D view expands per element colours over the glyph geometry on every step."""

    def setUp(self):
        # two glyph points per element, three elements
        self.index = np.array([0, 0, 1, 1, 2, 2], dtype=int)
        self.colors = _ExpandedColors(self.index)

    def test_update_expands_element_colors_over_the_geometry(self):
        element_colors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        self.colors.update(element_colors)

        assert_array_equal(vtk_to_numpy(self.colors.vtk_array), element_colors[self.index])

    def test_the_vtk_array_wraps_the_buffer_instead_of_copying_it(self):
        first = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        self.colors.update(first)
        vtk_array = self.colors.vtk_array

        second = np.array([[9, 9, 9], [8, 8, 8], [7, 7, 7]], dtype=np.uint8)
        self.colors.update(second)

        # same VTK object, new contents: no reallocation and no deep copy per step
        self.assertIs(self.colors.vtk_array, vtk_array)
        assert_array_equal(vtk_to_numpy(vtk_array), second[self.index])


class DiagnosticsTableModelTests(unittest.TestCase):
    """The tables used to allocate one Qt item per cell; the model reads the arrays directly."""

    def setUp(self):
        self.model = DiagnosticsTableModel()
        self.model.set_content(
            3, ["radius", "water_filled"], [np.array([1.5, 2.5, 3.5]), np.array([True, False, True])]
        )

    def test_shape_and_headers(self):
        self.assertEqual(self.model.rowCount(), 3)
        self.assertEqual(self.model.columnCount(), 2)
        self.assertEqual(self.model.headerData(0, Qt.Horizontal, Qt.DisplayRole), "radius")
        self.assertEqual(self.model.headerData(2, Qt.Vertical, Qt.DisplayRole), 2)

    def test_cells_read_from_the_column_arrays(self):
        self.assertEqual(self.model.data(self.model.index(0, 0), Qt.DisplayRole), "1.5")
        self.assertEqual(self.model.data(self.model.index(2, 1), Qt.DisplayRole), "True")

    def test_a_missing_column_renders_empty_instead_of_raising(self):
        self.model.set_content(2, ["missing"], [None])
        self.assertIsNone(self.model.data(self.model.index(0, 0), Qt.DisplayRole))

    def test_set_content_replaces_the_previous_step(self):
        self.model.set_content(2, ["radius"], [np.array([9.0, 8.0])])
        self.assertEqual(self.model.rowCount(), 2)
        self.assertEqual(self.model.columnCount(), 1)
        self.assertEqual(self.model.data(self.model.index(1, 0), Qt.DisplayRole), "8.0")


class CrossPlotSubsampleTests(unittest.TestCase):
    def test_small_scatters_are_plotted_whole(self):
        self.assertIsNone(CrossPlotView._subsample(MAX_PLOTTED_POINTS))

    def test_large_scatters_are_capped_and_deterministic(self):
        kept = CrossPlotView._subsample(MAX_PLOTTED_POINTS + 1)
        self.assertEqual(kept.size, MAX_PLOTTED_POINTS)
        self.assertEqual(np.unique(kept).size, MAX_PLOTTED_POINTS)
        self.assertLess(kept.max(), MAX_PLOTTED_POINTS + 1)
        assert_array_equal(kept, CrossPlotView._subsample(MAX_PLOTTED_POINTS + 1))


if __name__ == "__main__":
    unittest.main()
