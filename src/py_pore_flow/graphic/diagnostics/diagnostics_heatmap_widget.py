import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtGui, QtWidgets

from py_pore_flow.graphic.diagnostics.diagnostics_properties import DiagnosticsProperties

CYCLE_NAMES = ["drainage", "imbibition", "2nd_drainage"]
ELEMENT_TYPES = ["All", "Any", "Pore", "Throat"]
COLORMAP_NAME = "viridis"
N_BINS = 50
COLOR_COUNT = "count"
AGG_MEAN = "mean"
AGG_SUM = "sum"


class HeatmapView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self._colormap = pg.colormap.get(COLORMAP_NAME)

        self._graphics = pg.GraphicsLayoutWidget()
        self._plot = self._graphics.addPlot(row=0, col=0)
        self._plot.setLabel("bottom", "step")
        self._image = pg.ImageItem()
        self._plot.addItem(self._image)

        self._colorbar = pg.ColorBarItem(colorMap=self._colormap, interactive=False, pen="k", label="count")
        self._colorbar.setImageItem(self._image, insert_in=self._plot)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._graphics)

    def set_histogram_image(self, values, value_min, value_max, y_label, color_label):
        self._image.setImage(values, autoLevels=False)
        n_steps = values.shape[0]
        self._image.setRect(QtCore.QRectF(0.0, value_min, float(n_steps), float(value_max - value_min)))

        finite = values[np.isfinite(values)]
        color_min = float(finite.min()) if finite.size else 0.0
        color_max = float(finite.max()) if finite.size else 1.0
        if color_label == COLOR_COUNT:
            color_min = 0.0
        if color_max <= color_min:
            color_max = color_min + 1.0
        self._colorbar.setLevels((color_min, color_max))
        self._colorbar.setLabel("left", color_label)

        self._plot.setLabel("left", y_label)
        self._plot.setTitle(f"{y_label}  (color: {color_label})")
        self._plot.autoRange()

    def clear(self):
        self._image.clear()
        self._plot.setLabel("left", "")
        self._plot.setTitle("")


class HeatmapWidget(QtWidgets.QWidget):
    def __init__(self, diagnostics, parent=None):
        super().__init__(parent)
        self._diagnostics = diagnostics
        self._properties = DiagnosticsProperties(diagnostics)

        self.element_type_combo = QtWidgets.QComboBox()
        self.element_type_combo.addItems(ELEMENT_TYPES)
        self.property_combo = QtWidgets.QComboBox()
        self.color_combo = QtWidgets.QComboBox()
        self.aggregation_combo = QtWidgets.QComboBox()
        self.aggregation_combo.addItems([AGG_MEAN, AGG_SUM])
        self.cycle_combo = QtWidgets.QComboBox()
        self.cycle_combo.addItems(CYCLE_NAMES)

        self.bins_spinbox = QtWidgets.QSpinBox()
        self.bins_spinbox.setRange(2, 1000)
        self.bins_spinbox.setValue(N_BINS)

        self.auto_range_check = QtWidgets.QCheckBox("Auto Y range")
        self.auto_range_check.setChecked(True)
        self.auto_range_check.toggled.connect(self._on_auto_range_toggled)

        self.y_min_edit = QtWidgets.QLineEdit()
        self.y_max_edit = QtWidgets.QLineEdit()
        for edit in (self.y_min_edit, self.y_max_edit):
            validator = QtGui.QDoubleValidator(edit)
            validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
            edit.setValidator(validator)
            edit.setEnabled(False)  # enabled only when Auto is off

        y_range_layout = QtWidgets.QHBoxLayout()
        y_range_layout.addWidget(QtWidgets.QLabel("min"))
        y_range_layout.addWidget(self.y_min_edit)
        y_range_layout.addWidget(QtWidgets.QLabel("max"))
        y_range_layout.addWidget(self.y_max_edit)

        self.apply_button = QtWidgets.QPushButton("Apply")

        controls = QtWidgets.QGridLayout()
        controls.addWidget(QtWidgets.QLabel("Element type"), 0, 0)
        controls.addWidget(self.element_type_combo, 0, 1, 1, 2)
        controls.addWidget(QtWidgets.QLabel("Y-axis (property)"), 1, 0)
        controls.addWidget(self.property_combo, 1, 1, 1, 2)
        controls.addWidget(QtWidgets.QLabel("Color"), 2, 0)
        controls.addWidget(self.color_combo, 2, 1)
        controls.addWidget(self.aggregation_combo, 2, 2)
        controls.addWidget(QtWidgets.QLabel("Cycle"), 3, 0)
        controls.addWidget(self.cycle_combo, 3, 1, 1, 2)
        controls.addWidget(self.apply_button, 4, 1)
        controls.setColumnStretch(1, 2)
        controls.setColumnStretch(2, 1)

        self.view = HeatmapView()

        y_controls = QtWidgets.QFormLayout()
        y_controls.addRow("Bins", self.bins_spinbox)
        y_controls.addRow(self.auto_range_check)
        y_controls.addRow(y_range_layout)

        layout = QtWidgets.QGridLayout(self)
        layout.addLayout(controls, 0, 0, 1, 2)
        layout.addLayout(y_controls, 1, 0)
        layout.addWidget(self.view, 1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 10)

        self._populate_property_combos()
        self.element_type_combo.currentTextChanged.connect(self._populate_property_combos)
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        self.apply_button.clicked.connect(self._apply)
        self._on_color_changed(self.color_combo.currentText())

    def _on_color_changed(self, text):
        # aggregation (mean/sum) only applies when coloring by a property, not by count
        self.aggregation_combo.setEnabled(text != COLOR_COUNT)

    def _on_auto_range_toggled(self, checked):
        self.y_min_edit.setEnabled(not checked)
        self.y_max_edit.setEnabled(not checked)

    def _populate_property_combos(self):
        """Fill the Y-axis and Color combos with the property names for the current element
        type: All -> "<type>.<property>", Any/Pore/Throat -> bare property names."""
        element_type = self.element_type_combo.currentText()
        if element_type == "Pore":
            names = self._properties.get_pore_property_names()
        elif element_type == "Throat":
            names = self._properties.get_throat_property_names()
        elif element_type == "Any":
            names = self._properties.get_element_property_names()
        else:  # "All"
            names = self._properties.get_property_names()

        self.property_combo.clear()
        self.property_combo.addItems(names)
        self.color_combo.clear()
        self.color_combo.addItem(COLOR_COUNT)  # default: cell = element count in bin
        self.color_combo.addItems(names)

    def _property_values(self, name, cycle_id):
        """Per-step value arrays for `name` under the current element type (routes to the
        matching DiagnosticsProperties getter). Returns a list of arrays, or None."""
        element_type = self.element_type_combo.currentText()
        try:
            if element_type == "Pore":
                return self._properties.get_pore_property_values(name, cycle_id=cycle_id)
            if element_type == "Throat":
                return self._properties.get_throat_property_values(name, cycle_id=cycle_id)
            if element_type == "Any":
                return self._properties.get_element_property_values(name, cycle_id=cycle_id)
            return self._properties.get_property_values(name, cycle_id=cycle_id)  # "All"
        except KeyError:
            return None

    def _apply(self):
        text = self.property_combo.currentText()
        cycle_id = self.cycle_combo.currentIndex()
        if not text or cycle_id >= self._diagnostics.info.n_cycles:
            self.view.clear()
            return

        y_steps = self._property_values(text, cycle_id)
        if not y_steps:
            self.view.clear()
            return

        all_y = np.concatenate(y_steps)
        finite = all_y[np.isfinite(all_y)]
        if finite.size == 0:
            self.view.clear()
            return

        value_min, value_max = self._resolve_range(float(finite.min()), float(finite.max()))
        n_bins = self.bins_spinbox.value()
        bins = np.linspace(value_min, value_max, n_bins + 1)

        color = self.color_combo.currentText()
        if color == COLOR_COUNT:
            matrix = self._count_matrix(y_steps, bins, n_bins)
            color_label = COLOR_COUNT
        else:
            color_steps = self._property_values(color, cycle_id)
            if not color_steps or color_steps[0].shape[0] != y_steps[0].shape[0]:
                matrix = self._count_matrix(y_steps, bins, n_bins)
                color_label = COLOR_COUNT
            elif self.aggregation_combo.currentText() == AGG_SUM:
                matrix = self._sum_matrix(y_steps, color_steps, bins, n_bins)
                color_label = f"{color} (sum)"
            else:
                matrix = self._mean_matrix(y_steps, color_steps, bins, n_bins)
                color_label = f"{color} (mean)"

        self.view.set_histogram_image(matrix, value_min, value_max, text, color_label)

    @staticmethod
    def _count_matrix(y_steps, bins, n_bins):
        """Per (step, bin) count of elements whose binned property falls in the bin."""
        matrix = np.zeros((len(y_steps), n_bins), dtype=float)
        for step, y in enumerate(y_steps):
            matrix[step], _ = np.histogram(y[np.isfinite(y)], bins=bins)
        return matrix

    @staticmethod
    def _mean_matrix(y_steps, color_steps, bins, n_bins):
        """
        Per (step, bin) mean of the color property over the elements in that bin
        (NaN where a bin has no elements).
        """
        matrix = np.full((len(y_steps), n_bins), np.nan, dtype=float)
        for step, (y, color) in enumerate(zip(y_steps, color_steps)):
            mask = np.isfinite(y) & np.isfinite(color)
            if not mask.any():
                continue
            total, _ = np.histogram(y[mask], bins=bins, weights=color[mask])
            count, _ = np.histogram(y[mask], bins=bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                matrix[step] = np.where(count > 0, total / count, np.nan)
        return matrix

    @staticmethod
    def _sum_matrix(y_steps, color_steps, bins, n_bins):
        """
        Per (step, bin) sum of the color property over the elements in that bin
        (NaN where a bin has no elements).
        """
        matrix = np.full((len(y_steps), n_bins), np.nan, dtype=float)
        for step, (y, color) in enumerate(zip(y_steps, color_steps)):
            mask = np.isfinite(y) & np.isfinite(color)
            if not mask.any():
                continue
            total, _ = np.histogram(y[mask], bins=bins, weights=color[mask])
            count, _ = np.histogram(y[mask], bins=bins)
            matrix[step] = np.where(count > 0, total, np.nan)
        return matrix

    def _resolve_range(self, data_min, data_max):
        """
        The Y range to bin over: the data range in Auto mode (also shown in the fields),
        otherwise the (validated) values typed in the fields.
        """
        if self.auto_range_check.isChecked():
            value_min, value_max = data_min, data_max
            self.y_min_edit.setText(f"{value_min:.6g}")
            self.y_max_edit.setText(f"{value_max:.6g}")
        else:
            value_min = self._parse_field(self.y_min_edit, data_min)
            value_max = self._parse_field(self.y_max_edit, data_max)
            if value_max < value_min:
                value_min, value_max = value_max, value_min

        if value_max <= value_min:
            value_max = value_min + 1.0
        return value_min, value_max

    @staticmethod
    def _parse_field(edit, fallback):
        try:
            return float(edit.text())
        except ValueError:
            return fallback
