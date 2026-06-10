import numpy as np
import pyqtgraph as pg
from PySide2 import QtWidgets

from py_pore_flow.graphic.diagnostics.diagnostics_properties import DiagnosticsProperties

ELEMENT_TYPES = ["Pore", "Throat"]
CYCLE_NAMES = ["drainage", "imbibition", "2nd_drainage"]
NONE_LABEL = "(none)"
COLORMAP_NAME = "viridis"


class CrossPlotView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self._colormap = pg.colormap.get(COLORMAP_NAME)
        self._colorbar = None

        self._graphics = pg.GraphicsLayoutWidget()
        self._plot = self._graphics.addPlot(row=0, col=0)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._scatter = pg.ScatterPlotItem(pxMode=True, symbol="o", size=8)
        self._plot.addItem(self._scatter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._graphics)

    def plot(self, x, y, z, x_label, y_label, z_label):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        valid = ~(np.isnan(x) | np.isnan(y))
        if z is not None:
            z = np.asarray(z, dtype=float).ravel()
            valid &= ~np.isnan(z)
        x, y = x[valid], y[valid]
        z = z[valid] if z is not None else None

        if z is not None and z.size:
            brush = self._z_brushes(z)
            self._show_colorbar(float(np.min(z)), float(np.max(z)), z_label)
        else:
            brush = pg.mkBrush(80, 120, 220, 180)
            self._hide_colorbar()

        self._scatter.setData(x=x, y=y, brush=brush, pen=pg.mkPen(0, 0, 0, 60), size=8)
        self._plot.setLabel("bottom", x_label)
        self._plot.setLabel("left", y_label)

    def clear(self):
        self._scatter.setData(x=[], y=[])
        self._hide_colorbar()
        self._plot.setLabel("bottom", "")
        self._plot.setLabel("left", "")

    def _z_brushes(self, z):
        z_min, z_max = float(np.min(z)), float(np.max(z))
        if z_max > z_min:
            normalized = (z - z_min) / (z_max - z_min)
        else:
            normalized = np.zeros_like(z)
        return list(self._colormap.mapToQColor(normalized))

    def _show_colorbar(self, z_min, z_max, label):
        self._hide_colorbar()
        if z_max <= z_min:  # ColorBarItem needs a non-degenerate range
            z_max = z_min + 1.0
        self._colorbar = pg.ColorBarItem(
            values=(z_min, z_max), colorMap=self._colormap, label=label, interactive=False, pen="k"
        )
        self._graphics.addItem(self._colorbar, row=0, col=1)

    def _hide_colorbar(self):
        if self._colorbar is not None:
            self._graphics.removeItem(self._colorbar)
            self._colorbar = None


class CrossPlotWidget(QtWidgets.QWidget):
    def __init__(self, diagnostics, parent=None):
        super().__init__(parent)
        self._diagnostics = diagnostics
        self._properties = DiagnosticsProperties(diagnostics)

        self.cycle_combo = QtWidgets.QComboBox()
        self.cycle_combo.addItems(CYCLE_NAMES)
        self.step_spinbox = QtWidgets.QSpinBox()
        self.element_type_combo = QtWidgets.QComboBox()
        self.element_type_combo.addItems(ELEMENT_TYPES)
        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = QtWidgets.QComboBox()
        self.z_combo = QtWidgets.QComboBox()

        controls = QtWidgets.QFormLayout()
        controls.addRow("Cycle", self.cycle_combo)
        controls.addRow("Step", self.step_spinbox)
        controls.addRow("Element", self.element_type_combo)
        controls.addRow("X-axis", self.x_combo)
        controls.addRow("Y-axis", self.y_combo)
        controls.addRow("Z-axis (color)", self.z_combo)

        self.view = CrossPlotView()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.view)

        self._populate_axis_combos()
        self._update_step_range()

        self.cycle_combo.currentIndexChanged.connect(self._on_cycle_changed)
        self.step_spinbox.valueChanged.connect(self._replot)
        self.element_type_combo.currentTextChanged.connect(self._on_element_type_changed)
        self.x_combo.currentTextChanged.connect(self._replot)
        self.y_combo.currentTextChanged.connect(self._replot)
        self.z_combo.currentTextChanged.connect(self._replot)

        self._replot()

    def refresh(self):
        self._replot()

    def _populate_axis_combos(self):
        if self.element_type_combo.currentText() == "Pore":
            properties = self._properties.get_pore_property_names()
        else:
            properties = self._properties.get_throat_property_names()
        for combo, include_none in ((self.x_combo, False), (self.y_combo, False), (self.z_combo, True)):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if include_none:
                combo.addItem(NONE_LABEL)
            combo.addItems(properties)
            index = combo.findText(current)
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def _on_element_type_changed(self, _text):
        self._populate_axis_combos()
        self._replot()

    def _on_cycle_changed(self, _index):
        self._update_step_range()
        self._replot()

    def _update_step_range(self):
        cycle_id = self.cycle_combo.currentIndex()
        self.step_spinbox.blockSignals(True)
        self.step_spinbox.setMinimum(0)
        if cycle_id >= self._diagnostics.info.n_cycles:
            self.step_spinbox.setMaximum(0)
        else:
            n_steps = self._diagnostics.info.cycle_n_steps[cycle_id]
            self.step_spinbox.setMaximum(max(0, n_steps - 1))
        self.step_spinbox.blockSignals(False)

    def _property_values(self, property_name):
        cycle_id = self.cycle_combo.currentIndex()
        step = self.step_spinbox.value()
        if self.element_type_combo.currentText() == "Pore":
            return self._properties.get_pore_property_values(property_name, cycle_id=cycle_id, step=step)
        return self._properties.get_throat_property_values(property_name, cycle_id=cycle_id, step=step)

    def _replot(self):
        x_property = self.x_combo.currentText()
        y_property = self.y_combo.currentText()
        z_property = self.z_combo.currentText()
        if not x_property or not y_property:
            self.view.clear()
            return

        x = self._property_values(x_property)
        y = self._property_values(y_property)
        if x is None or y is None:
            self.view.clear()
            return

        if z_property and z_property != NONE_LABEL:
            z = self._property_values(z_property)
            z_label = z_property
        else:
            z = None
            z_label = ""

        self.view.plot(x, y, z, x_property, y_property, z_label)
