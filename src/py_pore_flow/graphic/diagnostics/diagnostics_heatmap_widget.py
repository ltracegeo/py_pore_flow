import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtGui, QtWidgets

from py_pore_flow.graphic.diagnostics.diagnostics_properties import DiagnosticsProperties
from py_pore_flow.output.diagnostics_aggregate import AGG_COUNT, SCALE_LINEAR, SCALE_LOG

CYCLE_NAMES = ["drainage", "imbibition", "2nd_drainage"]
ELEMENT_TYPES = ["All", "Any", "Pore", "Throat"]
# aggregates are stored per element type, so the combined views need raw data
AGGREGATE_ELEMENT_TYPES = ["Pore", "Throat"]
COLORMAP_NAME = "viridis"
N_BINS = 50
COLOR_COUNT = "count"
AGG_MEAN = "mean"
AGG_SUM = "sum"

NO_BIN_VALUE = 0

X_AXIS_STEP = "step"
X_AXIS_PC = "pc"
X_AXIS_SW = "sw"
# label -> name of the per step simulation scalar the diagnostics stored
X_AXIS_SOURCES = {"capillary pressure (Pc)": X_AXIS_PC, "water saturation (Sw)": X_AXIS_SW}
X_AXIS_LABELS = [X_AXIS_STEP] + list(X_AXIS_SOURCES)


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

    def set_histogram_image(
        self,
        values,
        value_min,
        value_max,
        y_label,
        color_label,
        log_y=False,
        log_color=False,
        x_range=None,
        x_label=X_AXIS_STEP,
        log_x=False,
    ):
        # Get min/max
        finite = values[np.isfinite(values)]
        color_min = float(finite.min()) if finite.size else 0.0
        color_max = float(finite.max()) if finite.size else 1.0
        if color_label == COLOR_COUNT:
            color_min = 0.0
        if color_max <= color_min:
            color_max = color_min + 1.0

        # Replace nans with NO_BIN_VALUE. On a log color scale an empty cell is -inf, so it
        # belongs at the bottom of the scale rather than at zero, which sits in the middle.
        fill_source = color_min if log_color else NO_BIN_VALUE
        fill_value = float(np.clip(fill_source, color_min, color_max))
        image = np.where(np.isfinite(values), values, fill_value)

        # Plot. A log Y axis places the image in log10 coordinates and lets the axis render
        # the ticks as powers of ten: the rows of an ImageItem are evenly spaced, so the bins
        # themselves have to be the log spaced ones.
        self._image.setImage(image, autoLevels=False)
        y_min, y_max = (np.log10(value_min), np.log10(value_max)) if log_y else (value_min, value_max)
        x_min, x_max = (0.0, float(image.shape[0])) if x_range is None else x_range
        if log_x:
            x_min, x_max = np.log10(x_min), np.log10(x_max)
        self._image.setRect(QtCore.QRectF(float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)))
        self._plot.getAxis("left").setLogMode(log_y)
        self._plot.getAxis("bottom").setLogMode(log_x)
        self._plot.setLabel("bottom", x_label)

        self._colorbar.setLevels((color_min, color_max))
        self._colorbar.setLabel("left", color_label)

        self._plot.setLabel("left", y_label)
        self._plot.setTitle(f"{y_label}  (color: {color_label})")
        self._plot.autoRange()

    def clear(self):
        self._image.clear()
        self._plot.getAxis("left").setLogMode(False)
        self._plot.getAxis("bottom").setLogMode(False)
        self._plot.setLabel("left", "")
        self._plot.setTitle("")


class HeatmapWidget(QtWidgets.QWidget):
    def __init__(self, diagnostics, aggregates=None, parent=None):
        super().__init__(parent)
        self._diagnostics = diagnostics
        self._aggregates = aggregates
        self._properties = DiagnosticsProperties(diagnostics) if diagnostics is not None else None

        self.x_axis_combo = QtWidgets.QComboBox()
        self.x_axis_combo.addItems(X_AXIS_LABELS)
        self.log_x_check = QtWidgets.QCheckBox("Log")
        self.log_x_check.setToolTip("Space the X axis logarithmically (positive values only)")
        self.log_x_check.setEnabled(False)

        self.element_type_combo = QtWidgets.QComboBox()
        self.element_type_combo.addItems(AGGREGATE_ELEMENT_TYPES if self._aggregates else ELEMENT_TYPES)
        self.property_combo = QtWidgets.QComboBox()
        self.color_combo = QtWidgets.QComboBox()
        self.aggregation_combo = QtWidgets.QComboBox()
        self.aggregation_combo.addItems([AGG_MEAN, AGG_SUM])

        self.log_y_check = QtWidgets.QCheckBox("Log")
        self.log_y_check.setToolTip("Bin the Y axis property logarithmically (positive values only)")
        self.log_color_check = QtWidgets.QCheckBox("Log")
        self.log_color_check.setToolTip("Color the cells by the base 10 logarithm of the aggregated value")
        self.cycle_combo = QtWidgets.QComboBox()
        self.cycle_combo.addItems(CYCLE_NAMES)

        self.bins_spinbox = QtWidgets.QSpinBox()
        self.bins_spinbox.setRange(2, 1000)
        self.bins_spinbox.setValue(N_BINS)

        self.manual_range_check = QtWidgets.QCheckBox()
        self.manual_range_check.setChecked(False)
        self.manual_range_check.toggled.connect(self._on_manual_range_toggled)

        self.y_min_edit = QtWidgets.QLineEdit()
        self.y_max_edit = QtWidgets.QLineEdit()
        for edit in (self.y_min_edit, self.y_max_edit):
            validator = QtGui.QDoubleValidator(edit)
            validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
            edit.setValidator(validator)
            edit.setEnabled(False)  # enabled only when Auto is off

        y_range_layout = QtWidgets.QHBoxLayout()
        y_range_layout.addWidget(QtWidgets.QLabel("Bins:"))
        y_range_layout.addWidget(self.bins_spinbox)
        y_range_layout.addWidget(QtWidgets.QLabel("Y Range:"))
        y_range_layout.addWidget(self.manual_range_check)
        y_range_layout.addWidget(self.y_min_edit)
        y_range_layout.addWidget(QtWidgets.QLabel(":"))
        y_range_layout.addWidget(self.y_max_edit)

        line_widget = QtWidgets.QFrame()
        line_widget.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line_widget.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)

        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setMinimumWidth(250)

        apply_button_layout = QtWidgets.QHBoxLayout()
        apply_button_layout.addWidget(self.apply_button)
        apply_button_layout.addStretch(5)
        apply_button_layout.addLayout(y_range_layout)

        controls = QtWidgets.QGridLayout()
        controls.addWidget(QtWidgets.QLabel("X-axis"), 0, 0)
        controls.addWidget(self.x_axis_combo, 0, 1, 1, 2)
        controls.addWidget(self.log_x_check, 0, 3)
        controls.addWidget(QtWidgets.QLabel("Element type"), 1, 0)
        controls.addWidget(self.element_type_combo, 1, 1, 1, 3)
        controls.addWidget(QtWidgets.QLabel("Y-axis (property)"), 2, 0)
        controls.addWidget(self.property_combo, 2, 1, 1, 2)
        controls.addWidget(self.log_y_check, 2, 3)
        controls.addWidget(QtWidgets.QLabel("Color"), 3, 0)
        controls.addWidget(self.color_combo, 3, 1)
        controls.addWidget(self.aggregation_combo, 3, 2)
        controls.addWidget(self.log_color_check, 3, 3)
        controls.addWidget(QtWidgets.QLabel("Cycle"), 4, 0)
        controls.addWidget(self.cycle_combo, 4, 1, 1, 3)
        controls.addWidget(line_widget, 5, 0, 1, 4)
        controls.addLayout(apply_button_layout, 6, 1, 1, 3)
        controls.setColumnStretch(1, 2)
        controls.setColumnStretch(2, 1)

        self.view = HeatmapView()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.view)

        self._populate_property_combos()
        self.element_type_combo.currentTextChanged.connect(self._populate_property_combos)
        if self._aggregates is not None:
            # only the pairs that were precomputed for this Y property can be coloured by
            self.property_combo.currentTextChanged.connect(self._populate_color_combo)
        self.x_axis_combo.currentTextChanged.connect(self._update_x_axis_availability)
        self.cycle_combo.currentIndexChanged.connect(self._update_x_axis_availability)
        self.property_combo.currentTextChanged.connect(self._update_log_availability)
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        self.apply_button.clicked.connect(self._apply)
        self._on_color_changed(self.color_combo.currentText())
        self._update_log_availability()
        self._update_x_axis_availability()

    def _on_color_changed(self, text):
        # aggregation (mean/sum) only applies when coloring by a property, not by count
        self.aggregation_combo.setEnabled(text != COLOR_COUNT)

    def _x_axis_values(self, cycle_id):
        """Per step values of the selected X coordinate, or None for the step index."""
        source = X_AXIS_SOURCES.get(self.x_axis_combo.currentText())
        if source is None:
            return None

        if self._aggregates is not None:
            values = self._aggregates.step_axis(cycle_id, source)
        else:
            if source == X_AXIS_PC:
                values = self._diagnostics.cycle[cycle_id].pc
            else:
                values = self._diagnostics.cycle[cycle_id].sw

        if values is None or not np.isfinite(values).any():
            return None
        return np.asarray(values, dtype=float)

    def _update_x_axis_availability(self):
        """Enable log X only for a coordinate that exists and stays positive in this cycle."""
        values = self._x_axis_values(self.cycle_combo.currentIndex())
        available = values is not None and np.all(values[np.isfinite(values)] > 0)
        if not available:
            self.log_x_check.setChecked(False)
        self.log_x_check.setEnabled(bool(available))

    def _place_on_x_axis(self, matrix, cycle_id):
        """Resample the per step rows onto the selected X coordinate.

        Returns (image, x_range, x_label, log_x). An ImageItem draws evenly spaced rows, so
        anything other than the step index has to be resampled onto a uniform grid: steps
        sharing a cell are averaged, and a cell no step landed in takes the nearest one,
        which is how an irregularly sampled series is normally drawn.
        """
        values = self._x_axis_values(cycle_id)
        if values is None or values.shape[0] != matrix.shape[0]:
            return matrix, None, X_AXIS_STEP, False

        finite = np.isfinite(values)
        if not finite.any():
            return matrix, None, X_AXIS_STEP, False

        log_x = self.log_x_check.isChecked() and np.all(values[finite] > 0)
        low, high = float(values[finite].min()), float(values[finite].max())
        if high <= low:
            high = low * 10.0 if log_x else low + 1.0

        n_x = matrix.shape[0]
        edges = np.logspace(np.log10(low), np.log10(high), n_x + 1) if log_x else np.linspace(low, high, n_x + 1)
        index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_x - 1)

        image = np.full((n_x, matrix.shape[1]), np.nan, dtype=float)
        for target in range(n_x):
            rows = matrix[finite & (index == target)]
            if rows.shape[0] == 0:
                continue
            valid = np.isfinite(rows)
            total = np.where(valid, rows, 0.0).sum(axis=0)
            n_valid = valid.sum(axis=0)
            image[target] = np.where(n_valid > 0, total / np.maximum(n_valid, 1), np.nan)

        image = self._fill_empty_columns(image)
        return image, (low, high), self.x_axis_combo.currentText(), log_x

    @staticmethod
    def _fill_empty_columns(image):
        """Carry the nearest sampled column into the cells no step landed in."""
        occupied = np.nonzero(np.isfinite(image).any(axis=1))[0]
        if occupied.size == 0 or occupied.size == image.shape[0]:
            return image
        targets = np.arange(image.shape[0])
        nearest = occupied[np.abs(targets[:, None] - occupied[None, :]).argmin(axis=1)]
        return image[nearest]

    def _update_log_availability(self):
        """Log binning needs a strictly positive property.

        The aggregates record which properties qualify; the raw source only finds out when
        it reads the values, so there the option stays enabled.
        """
        if self._aggregates is None:
            self.log_y_check.setEnabled(True)
            return

        available = self._aggregates.log_scale_available(self._element_type(), self.property_combo.currentText())
        if not available:
            self.log_y_check.setChecked(False)
        self.log_y_check.setEnabled(available)

    def _apply_color_scale(self, matrix, color_label):
        """Color by log10 of the aggregated value; cells that are not positive drop out."""
        if not self.log_color_check.isChecked():
            return matrix, color_label
        with np.errstate(invalid="ignore"):
            return np.log10(np.where(matrix > 0, matrix, np.nan)), f"log10({color_label})"

    def _on_manual_range_toggled(self, checked):
        self.y_min_edit.setEnabled(checked)
        self.y_max_edit.setEnabled(checked)

    def _element_type(self):
        return self.element_type_combo.currentText().lower()

    def _populate_color_combo(self):
        """Colour options of the selected Y property (aggregate source only)."""
        y_property = self.property_combo.currentText()
        names = self._aggregates.color_property_names(self._element_type(), y_property) if y_property else []
        self.color_combo.clear()
        self.color_combo.addItem(COLOR_COUNT)  # default: cell = element count in bin
        self.color_combo.addItems(sorted(names))

    def _populate_property_combos(self):
        """Fill the Y-axis and Color combos with the property names for the current element
        type: All -> "<type>.<property>", Any/Pore/Throat -> bare property names."""
        if self._aggregates is not None:
            self.property_combo.clear()
            self.property_combo.addItems(self._aggregates.y_property_names(self._element_type()))
            self._populate_color_combo()
            return

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
                return self._properties.get_pore_values_of_cycle(name, cycle_id=cycle_id)
            if element_type == "Throat":
                return self._properties.get_throat_values_of_cycle(name, cycle_id=cycle_id)
            if element_type == "Any":
                return self._properties.get_element_values_of_cycle(name, cycle_id=cycle_id)
            return self._properties.get_values_of_cycle(name, cycle_id=cycle_id)  # "All"
        except KeyError:
            return None

    def _apply(self):
        if self._aggregates is not None:
            self._apply_aggregates()
            return
        self._apply_raw()

    def _apply_aggregates(self):
        """Read a precomputed matrix and merge its fine bins down to the requested count."""
        y_property = self.property_combo.currentText()
        cycle_id = self.cycle_combo.currentIndex()
        element_type = self._element_type()
        if not y_property or not self._aggregates.cycle_n_steps.get(cycle_id):
            # nothing was aggregated for a cycle that was skipped or ran no steps
            self.view.clear()
            return

        color = self.color_combo.currentText()
        color_property = None if color == COLOR_COUNT else color
        aggregation = AGG_COUNT if color_property is None else self.aggregation_combo.currentText()

        log_y = self.log_y_check.isChecked() and self._aggregates.log_scale_available(element_type, y_property)
        scale = SCALE_LOG if log_y else SCALE_LINEAR

        stored_min, stored_max = self._aggregates.value_range(cycle_id, element_type, y_property, scale)
        requested_range = self._resolve_range(stored_min, stored_max, log_y)
        # the stored resolution is the finest the aggregate can be split into
        n_bins = min(
            self.bins_spinbox.value(),
            self._aggregates.stored_n_bins(cycle_id, element_type, y_property, scale),
        )

        matrix, (value_min, value_max) = self._aggregates.matrix(
            cycle_id, element_type, y_property, color_property, aggregation, n_bins, requested_range, scale
        )
        color_label = COLOR_COUNT if color_property is None else f"{color} ({aggregation})"
        matrix, x_range, x_label, log_x = self._place_on_x_axis(matrix, cycle_id)
        matrix, color_label = self._apply_color_scale(matrix, color_label)
        self.view.set_histogram_image(
            matrix,
            value_min,
            value_max,
            y_property,
            color_label,
            log_y,
            self.log_color_check.isChecked(),
            x_range,
            x_label,
            log_x,
        )

    def _apply_raw(self):
        text = self.property_combo.currentText()
        cycle_id = self.cycle_combo.currentIndex()
        if not text or cycle_id >= self._diagnostics.info.n_cycles:
            self.view.clear()
            return

        y_steps = self._property_values(text, cycle_id)
        if not y_steps:
            self.view.clear()
            return

        log_y = self.log_y_check.isChecked()
        all_y = np.concatenate(y_steps)
        finite = all_y[np.isfinite(all_y)]
        if log_y:
            # a log axis cannot show zero or negative values, so they leave the picture
            finite = finite[finite > 0]
        if finite.size == 0:
            self.view.clear()
            return

        value_min, value_max = self._resolve_range(float(finite.min()), float(finite.max()), log_y)
        n_bins = self.bins_spinbox.value()
        bins = (
            np.logspace(np.log10(value_min), np.log10(value_max), n_bins + 1)
            if log_y
            else np.linspace(value_min, value_max, n_bins + 1)
        )

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

        matrix, x_range, x_label, log_x = self._place_on_x_axis(matrix, cycle_id)
        matrix, color_label = self._apply_color_scale(matrix, color_label)
        self.view.set_histogram_image(
            matrix,
            value_min,
            value_max,
            text,
            color_label,
            log_y,
            self.log_color_check.isChecked(),
            x_range,
            x_label,
            log_x,
        )

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

    def _resolve_range(self, data_min, data_max, log_scale=False):
        """
        The Y range to bin over: the data range in Auto mode (also shown in the fields),
        otherwise the (validated) values typed in the fields.
        """
        if not self.manual_range_check.isChecked():
            value_min, value_max = data_min, data_max
            self.y_min_edit.setText(f"{value_min:.6g}")
            self.y_max_edit.setText(f"{value_max:.6g}")
        else:
            value_min = self._parse_field(self.y_min_edit, data_min)
            value_max = self._parse_field(self.y_max_edit, data_max)
            if value_max < value_min:
                value_min, value_max = value_max, value_min

        if log_scale and value_min <= 0:
            # data_min is already the smallest positive value in log mode
            value_min = data_min
        if value_max <= value_min:
            value_max = value_min * 10.0 if log_scale else value_min + 1.0
        return value_min, value_max

    @staticmethod
    def _parse_field(edit, fallback):
        try:
            return float(edit.text())
        except ValueError:
            return fallback
