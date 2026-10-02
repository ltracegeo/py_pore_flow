import math
import re

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide2 import QtCore, QtGui, QtWidgets

from py_pore_flow.graphic.element_graphic import ElementPlotter
from py_pore_flow.graphic.network_plot import ElementNotFoundException, PorePlot, ThroatPlot


# attributes DiagnosticsStateLoader sets on itself that are not per-element step data
_LOADER_ATTRS = {"path", "step", "n_elements"}

CYCLE_NAMES = ["drainage", "imbibition", "2nd_drainage"]

# krel curve colors: water reads as blue, oil as a dark red
KRW_COLOR = "#1565C0"
KRO_COLOR = "#8B1A1A"

# capillary pressure belongs to neither phase, so it stays a neutral slate gray
PC_COLOR = "#455A64"


class DiagnosticsElementWidget(QtWidgets.QFrame):
    networkChanged = QtCore.Signal()
    poreSelected = QtCore.Signal(int)
    throatSelected = QtCore.Signal(int)

    def __init__(self, diagnostics, diagnostics_properties, parent=None):
        super().__init__(parent)
        self._diagnostics = diagnostics
        self._diagnostics_properties = diagnostics_properties
        self._diagnostics_properties.set_cycle(0)
        self._diagnostics_properties.set_step(0)

        # Krel
        self.krel_widget = KrelWidget(self._diagnostics)

        # Pressure curve
        self.pressure_widget = PressureCurveWidget(self._diagnostics)

        # Element geometry
        self.element_widget = ElementWidget(self._diagnostics_properties)
        self.element_widget.poreSelected.connect(lambda id: self.poreSelected.emit(id))
        self.element_widget.throatSelected.connect(lambda id: self.throatSelected.emit(id))

        # Tabs
        plots_tab = QtWidgets.QTabWidget()
        plots_tab.addTab(self.element_widget, "Element")
        plots_tab.addTab(self.krel_widget, "Krel")
        plots_tab.addTab(self.pressure_widget, "Pressure curve")
        plots_tab.currentChanged.connect(self._on_tab_changed)

        # Step control
        self.cycle_combobox = QtWidgets.QComboBox()
        for name in CYCLE_NAMES:
            self.cycle_combobox.addItem(name)
        self.cycle_combobox.currentIndexChanged.connect(self._on_cycle_changed)

        self.step_input = QtWidgets.QSpinBox()
        self.step_input.valueChanged.connect(self._on_step_changed)

        self.status_label = QtWidgets.QLabel()
        font = QtGui.QFont("Courier New")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.status_label.setFont(font)

        step_control_layout = QtWidgets.QFormLayout()
        step_control_layout.addRow("Cycle:", self.cycle_combobox)
        step_control_layout.addRow("Step:", self.step_input)
        step_control_layout.addRow("", self.status_label)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(plots_tab)
        layout.addLayout(step_control_layout)

        self._update_step_range()
        self._refresh()

    def select_pore(self, pore_id):
        self.element_widget.select_pore(pore_id)

    def select_throat(self, throat_id):
        self.element_widget.select_throat(throat_id)

    def _on_cycle_changed(self, cycle_id):
        self._update_step_range()
        self._refresh()
        self.krel_widget.set_cycle(cycle_id)
        self.pressure_widget.set_cycle(cycle_id)

    def _on_step_changed(self, value):
        self._refresh()
        self.krel_widget.set_step(value)
        self.pressure_widget.set_step(value)

    def _update_step_range(self):
        cycle_id = self.cycle_combobox.currentIndex()
        self.step_input.blockSignals(True)
        self.step_input.setMinimum(0)
        if cycle_id >= self._diagnostics.info.n_cycles:
            self.step_input.setMaximum(0)
        else:
            n_steps = self._diagnostics.info.cycle_n_steps[cycle_id]
            self.step_input.setMaximum(max(0, n_steps - 1))
        self.step_input.blockSignals(False)

    def _refresh(self):
        cycle_id = self.cycle_combobox.currentIndex()
        info = self._diagnostics.info
        if cycle_id >= info.n_cycles:
            self.status_label.setText("No data for this cycle.")
            return

        n_steps = info.cycle_n_steps[cycle_id]
        step = self.step_input.value()

        self._diagnostics_properties.set_cycle(cycle_id)
        if n_steps > 0:
            self._diagnostics_properties.set_step(step)

        pc = self._diagnostics.cycle[cycle_id][step].pc
        sw = self._diagnostics.cycle[cycle_id][step].sw
        krw = self._diagnostics.cycle[cycle_id][step].krw
        kro = self._diagnostics.cycle[cycle_id][step].kro
        self.element_widget.update()
        self.status_label.setText(
            f"{self.cycle_combobox.currentText()}: step {step:3} / {max(0, n_steps - 1)} - Pc: {pc:.2f}, Sw: {sw:.2f}, Krw: {krw:.2f}, Kro: {kro:.2f}"
        )
        self.networkChanged.emit()

    def _on_tab_changed(self, tab_index):
        if tab_index == 1:
            self.krel_widget.initialize_plot()
        else:
            self.krel_widget.delete_plot()

        if tab_index == 2:
            self.pressure_widget.initialize_plot()
        else:
            self.pressure_widget.delete_plot()


class KrelWidget(QtWidgets.QWidget):
    """Relative permeability of one cycle, with the current step marked on both curves.

    The plot is built when the tab is first shown and thrown away when it is left.
    """

    def __init__(self, diagnostics):
        super().__init__()
        self._diagnostics = diagnostics
        self.cycle_id = 0
        self.step = 0
        self.plot_initialized = False

        self._graphics = None
        self._plot = None
        self._krw_curve = None
        self._kro_curve = None
        self._krw_highlight = None
        self._kro_highlight = None

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def set_cycle(self, cycle_id):
        self.cycle_id = cycle_id
        self._update_curves(self.cycle_id)
        self._update_highlight(self.step)

    def set_step(self, step):
        self.step = step
        self._update_highlight(self.step)

    def initialize_plot(self):
        if self.plot_initialized:
            return

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self._graphics = pg.GraphicsLayoutWidget()
        self._plot = self._graphics.addPlot(left="Krel", bottom="Sw")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend(offset=(-10, 10))

        self._krw_curve = self._add_curve(KRW_COLOR, "Krw")
        self._kro_curve = self._add_curve(KRO_COLOR, "Kro")
        self._krw_highlight = self._add_highlight(KRW_COLOR)
        self._kro_highlight = self._add_highlight(KRO_COLOR)

        self._layout.addWidget(self._graphics)
        self.plot_initialized = True

        self._update_curves(self.cycle_id)
        self._update_highlight(self.step)

    def delete_plot(self):
        if not self.plot_initialized:
            return

        self.plot_initialized = False

        self._layout.removeWidget(self._graphics)
        self._graphics.clear()
        self._graphics.setParent(None)
        self._graphics.deleteLater()

        self._graphics = None
        self._plot = None
        self._krw_curve = None
        self._kro_curve = None
        self._krw_highlight = None
        self._kro_highlight = None

    def _add_curve(self, color, name):
        curve = self._plot.plot(pen=pg.mkPen(color, width=2), name=name)
        curve.curve.setSegmentedLineMode("off")
        return curve

    def _add_highlight(self, color):
        highlight = pg.ScatterPlotItem(
            pxMode=True, symbol="o", size=11, brush=pg.mkBrush(color), pen=pg.mkPen("w", width=2)
        )
        highlight.setZValue(10)
        self._plot.addItem(highlight)
        return highlight

    def _update_curves(self, cycle_id):
        if not self.plot_initialized:
            return

        x = self._diagnostics.cycle[cycle_id].sw

        y = self._diagnostics.cycle[cycle_id].krw
        self._plot_krw(x, y)
        y = self._diagnostics.cycle[cycle_id].kro
        self._plot_kro(x, y)

    def _update_highlight(self, step):
        if not self.plot_initialized:
            return

        sw = self._diagnostics.cycle[self.cycle_id].sw
        if not 0 <= step < len(sw):
            self._highlight_krw(np.nan, np.nan)
            self._highlight_kro(np.nan, np.nan)
            return

        x = sw[step]

        y = self._diagnostics.cycle[self.cycle_id].krw[step]
        self._highlight_krw(x, y)
        y = self._diagnostics.cycle[self.cycle_id].kro[step]
        self._highlight_kro(x, y)

    def _plot_krw(self, x, y):
        self._set_curve(self._krw_curve, x, y)

    def _plot_kro(self, x, y):
        self._set_curve(self._kro_curve, x, y)

    def _highlight_krw(self, x, y):
        self._set_highlight(self._krw_highlight, x, y)

    def _highlight_kro(self, x, y):
        self._set_highlight(self._kro_highlight, x, y)

    @staticmethod
    def _set_curve(curve, x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        finite = np.isfinite(x) & np.isfinite(y)
        curve.setData(x[finite], y[finite])

    @staticmethod
    def _set_highlight(highlight, x, y):
        if not (np.isfinite(x) and np.isfinite(y)):
            highlight.setData(np.empty(0), np.empty(0))
            return
        highlight.setData(np.array([float(x)]), np.array([float(y)]))


class PressureCurveWidget(QtWidgets.QWidget):
    """Capillary pressure of one cycle, with the current step marked on the curve.

    Built when the tab is first shown and thrown away when it is left.
    """

    def __init__(self, diagnostics):
        super().__init__()
        self._diagnostics = diagnostics
        self.cycle_id = 0
        self.step = 0
        self.plot_initialized = False

        self._graphics = None
        self._plot = None
        self._pc_curve = None
        self._pc_highlight = None

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def set_cycle(self, cycle_id):
        self.cycle_id = cycle_id
        self._update_curve(self.cycle_id)
        self._update_highlight(self.step)

    def set_step(self, step):
        self.step = step
        self._update_highlight(self.step)

    def initialize_plot(self):
        if self.plot_initialized:
            return

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self._graphics = pg.GraphicsLayoutWidget()
        self._plot = self._graphics.addPlot(left="Pc", bottom="Sw")
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        self._pc_curve = self._add_curve(PC_COLOR)
        self._pc_highlight = self._add_highlight(PC_COLOR)

        self._layout.addWidget(self._graphics)
        self.plot_initialized = True

        self._update_curve(self.cycle_id)
        self._update_highlight(self.step)

    def delete_plot(self):
        if not self.plot_initialized:
            return

        self.plot_initialized = False

        self._layout.removeWidget(self._graphics)
        self._graphics.clear()
        self._graphics.setParent(None)
        self._graphics.deleteLater()

        self._graphics = None
        self._plot = None
        self._pc_curve = None
        self._pc_highlight = None

    def _add_curve(self, color):
        curve = self._plot.plot(pen=pg.mkPen(color, width=2))
        curve.curve.setSegmentedLineMode("off")
        return curve

    def _add_highlight(self, color):
        highlight = pg.ScatterPlotItem(
            pxMode=True, symbol="o", size=11, brush=pg.mkBrush(color), pen=pg.mkPen("w", width=2)
        )
        highlight.setZValue(10)
        self._plot.addItem(highlight)
        return highlight

    def _update_curve(self, cycle_id):
        if not self.plot_initialized:
            return

        x = self._diagnostics.cycle[cycle_id].sw
        y = self._diagnostics.cycle[cycle_id].pc

        self._plot_pc(x, y)

    def _update_highlight(self, step):
        if not self.plot_initialized:
            return

        sw = self._diagnostics.cycle[self.cycle_id].sw
        if not 0 <= step < len(sw):
            self._highlight_pc(np.nan, np.nan)
            return

        x = sw[step]
        y = self._diagnostics.cycle[self.cycle_id].pc[step]

        self._highlight_pc(x, y)

    def _plot_pc(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        finite = np.isfinite(x) & np.isfinite(y)
        self._pc_curve.setData(x[finite], y[finite])

    def _highlight_pc(self, x, y):
        if not (np.isfinite(x) and np.isfinite(y)):
            self._pc_highlight.setData(np.empty(0), np.empty(0))
            return
        self._pc_highlight.setData(np.array([float(x)]), np.array([float(y)]))


class ElementWidget(QtWidgets.QWidget):
    poreSelected = QtCore.Signal(int)
    throatSelected = QtCore.Signal(int)

    def __init__(self, diagnostics_properties):
        super().__init__()

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.element_type_combobox = QtWidgets.QComboBox()
        self.element_type_combobox.addItem("Throat")
        self.element_type_combobox.addItem("Pore")
        self.element_type_combobox.currentTextChanged.connect(
            lambda text: self._on_element_changed(self.element_id_input.text())
        )
        self.element_id_input = QtWidgets.QLineEdit()
        self.element_id_input.setValidator(QtGui.QIntValidator())
        self.element_id_input.editingFinished.connect(lambda: self._on_element_changed(self.element_id_input.text()))
        self.element_conn_combobox = QtWidgets.QComboBox()
        self.element_conn_combobox.setPlaceholderText("Go to connection")
        self.element_conn_combobox.currentTextChanged.connect(self._on_element_conn_changed)

        element_selection_layout = QtWidgets.QHBoxLayout()
        element_selection_layout.addWidget(self.element_type_combobox)
        element_selection_layout.addWidget(QtWidgets.QLabel("ID:"))
        element_selection_layout.addWidget(self.element_id_input)
        element_selection_layout.addWidget(self.element_conn_combobox)

        self.element_geometry_widget = ElementGeometryWidget()
        self.element_geometry_widget.set_pn(diagnostics_properties)

        element_layout = QtWidgets.QVBoxLayout()
        element_layout.addLayout(element_selection_layout)
        element_layout.addWidget(self.element_geometry_widget)

        self.setLayout(element_layout)

    def _on_element_changed(self, text):
        if not text:
            return

        try:
            element_id = int(text)
            if self.element_type_combobox.currentText() == "Throat":
                self.set_throat(element_id)
                self.element_geometry_widget.update()
                self.throatSelected.emit(element_id)
            else:
                self.set_pore(element_id)
                self.element_geometry_widget.update()
                self.poreSelected.emit(element_id)
        except ElementNotFoundException:
            self.element_id_input.blockSignals(True)
            self.element_id_input.setText("")
            self.element_id_input.blockSignals(False)

    def _on_element_conn_changed(self, text):
        pattern = re.compile(r"(Throat|Pore) (\d+)")
        re_match = pattern.match(text)
        if re_match:
            element_type = re_match[1]
            element_id = int(re_match[2])
            self.element_type_combobox.blockSignals(True)
            self.element_type_combobox.setCurrentText(element_type)
            self.element_type_combobox.blockSignals(False)
            self.element_id_input.setText(str(element_id))
            self._on_element_changed(str(element_id))

    def select_pore(self, pore_id):
        self._select_element("Pore", pore_id)

    def select_throat(self, throat_id):
        self._select_element("Throat", throat_id)

    def set_pore(self, pore_id):
        self.element_geometry_widget.set_pore(pore_id)
        self._update_conn_combobox(self.element_geometry_widget.get_conns(), "Throat")

    def set_throat(self, throat_id):
        self.element_geometry_widget.set_throat(throat_id)
        self._update_conn_combobox(self.element_geometry_widget.get_conns(), "Pore")

    def update(self):
        self.element_geometry_widget.update()

    def _update_conn_combobox(self, conns, element_type):
        self.element_conn_combobox.blockSignals(True)
        self.element_conn_combobox.clear()
        for conn_id in conns:
            self.element_conn_combobox.addItem(f"{element_type} {conn_id}")
        self.element_conn_combobox.blockSignals(False)

    def _select_element(self, element_type, element_id):
        self.element_type_combobox.blockSignals(True)
        self.element_type_combobox.setCurrentText(element_type)
        self.element_type_combobox.blockSignals(False)
        self.element_id_input.setText(str(element_id))
        self._on_element_changed(str(element_id))


# --- copied from py_pore_flow.graphic.manual_simulation ---------------------------------
# ElementWidget / ConductanceBarWidget are duplicated here because manual_simulation must
# not depend on PySide2 (it is shared with another project), so it cannot be imported.


class ElementGeometryWidget(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()

        self.pn = None

        self.element_plotter = ElementPlotter()
        self.element_plot = None

        self.graph = pg.GraphicsLayoutWidget()
        self.table = QtWidgets.QTableWidget()
        self.water_conductance_bar = ConductanceBarWidget("#05B8CC")
        self.oil_conductance_bar = ConductanceBarWidget("#efafaf")
        self.plot = self.graph.addPlot()

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.graph)
        splitter.addWidget(self.table)
        splitter.addWidget(self.water_conductance_bar)
        splitter.addWidget(self.oil_conductance_bar)
        splitter.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(splitter)

    def set_pn(self, pn):
        self.pn = pn

    def set_pore(self, pore_id):
        self.element_plot = PorePlot(self.pn, pore_id)

    def set_throat(self, throat_id):
        self.element_plot = ThroatPlot(self.pn, throat_id)

    def update(self):
        if not self.element_plot:
            return

        self.element_plot.update()
        self.element_plotter.plot_pyqtgraph_element(self.plot, self.element_plot.get_graphic())
        self._update_table(self.element_plot)

    def get_conns(self):
        if not self.element_plot:
            return []
        element_type = self.element_plot.element_type
        id = self.element_plot.id
        return np.resize(self.pn[f"{element_type}.conns"][id], self.pn[f"{element_type}.n_conns"][id])

    def _update_table(self, element_plot):
        element_type = element_plot.element_type
        id = element_plot.id
        n_corners = element_plot.n_corners

        G = self._pad_with_nan(self.pn[f"{element_type}.shape_factor"][id], n_corners)
        radius = self._pad_with_nan(self.pn[f"{element_type}.radius"][id], n_corners)
        beta = np.degrees(np.resize(self.pn[f"{element_type}.corner_half_angle"][id], n_corners))
        conns = np.resize(self.pn[f"{element_type}.conns"][id], self.pn[f"{element_type}.n_conns"][id])
        oil_filled = self.pn[f"{element_type}.oil_filled"][id]
        sw = self._pad_with_nan(self.pn[f"{element_type}.water_saturation"][id], n_corners)
        theta_h = np.degrees(np.resize(self.pn[f"{element_type}.hinging_contact_angle"][id], n_corners))
        theta_ho = np.degrees(np.resize(self.pn[f"{element_type}.hinging_contact_angle_o"][id], n_corners))
        theta_a = np.degrees(np.resize(self.pn[f"{element_type}.advancing_contact_angle"][id], max(n_corners, 1)))
        theta_r = np.degrees(np.resize(self.pn[f"{element_type}.receding_contact_angle"][id], max(n_corners, 1)))
        corner_oil_layer = np.resize(self.pn[f"{element_type}.corner_oil_layer"][id], n_corners)
        corner_water_layer = np.resize(self.pn[f"{element_type}.corner_water_layer"][id], n_corners)
        if oil_filled:
            b = np.resize(self.pn[f"{element_type}.b"][id], n_corners)
        else:
            b = np.full(n_corners, 0.0, np.float64)
            b[corner_oil_layer] = np.resize(self.pn[f"{element_type}.b"][id], n_corners)[corner_oil_layer]
        bo = np.resize(self.pn[f"{element_type}.bo"][id], n_corners)
        oil_layer_collapse_pressure = self.pn[f"{element_type}.oil_layer_collapse_pressure"][id][:n_corners]
        entry_pressure = self._pad_with_nan(self.pn[f"{element_type}.entry_pressure"][id], n_corners)
        piston_like_pressure = self._pad_with_nan(self.pn[f"{element_type}.piston_like_pressure"][id], n_corners)
        snap_off_pressure = self._pad_with_nan(self.pn[f"{element_type}.snap_off_pressure"][id], n_corners)
        water_single_conductivity = self.pn[f"{element_type}.water_singlephase_conductivity"][id]
        water_multi_conductivity = self.pn[f"{element_type}.water_multiphase_conductivity"][id]
        oil_single_conductivity = self.pn[f"{element_type}.oil_singlephase_conductivity"][id]
        oil_multi_conductivity = self.pn[f"{element_type}.oil_multiphase_conductivity"][id]

        if element_type == "pore":
            if self.pn["pore.inlets"][id]:
                conns = ["inlet"] + conns.tolist()
            elif self.pn["pore.outlets"][id]:
                conns = ["outlet"] + conns.tolist()
            water_single_conductance = ""
            water_multi_conductance = ""
            oil_single_conductance = ""
            oil_multi_conductance = ""
        else:
            water_single_conductance = self.pn[f"throat.water_singlephase_conductance"][id]
            water_multi_conductance = self.pn[f"throat.water_multiphase_conductance"][id]
            oil_single_conductance = self.pn[f"throat.oil_singlephase_conductance"][id]
            oil_multi_conductance = self.pn[f"throat.oil_multiphase_conductance"][id]

        df = pd.DataFrame.from_dict(
            {
                "G:": G,
                "radius:": radius,
                "beta:": beta,
                "conns": [str(conns)],
                "oil_filled": [str(oil_filled)],
                "Sw": sw,
                "theta_h": theta_h,
                "theta_ho": theta_ho,
                "theta_a": theta_a,
                "theta_r": theta_r,
                "bi": b,
                "bo": bo,
                "corner_oil_layer": corner_oil_layer,
                "corner_water_layer": corner_water_layer,
                "entry_pressure": entry_pressure,
                "piston_like_pressure": piston_like_pressure,
                "snap_off_pressure": snap_off_pressure,
                "water_single_conductivity": [water_single_conductivity],
                "water_multi_conductivity": [water_multi_conductivity],
                "oil_single_conductivity": [oil_single_conductivity],
                "oil_multi_conductivity": [oil_multi_conductivity],
                "water_single_conductance": [water_single_conductance],
                "water_multi_conductance": [water_multi_conductance],
                "oil_single_conductance": [oil_single_conductance],
                "oil_multi_conductance": [oil_multi_conductance],
                "oil_layer_collapse_pressure": oil_layer_collapse_pressure,
            },
            orient="index",
        )

        self._df_to_widget(df, self.table)

        if element_type == "throat":
            self.water_conductance_bar.set_max_conductance(water_single_conductance)
            self.water_conductance_bar.set_current_conductance(water_multi_conductance)
            self.water_conductance_bar.update()
            self.oil_conductance_bar.set_max_conductance(oil_single_conductance)
            self.oil_conductance_bar.set_current_conductance(oil_multi_conductance)
            self.oil_conductance_bar.update()

    @staticmethod
    def _df_to_widget(dataframe, table_widget):
        num_rows, num_cols = dataframe.shape
        table_widget.setRowCount(num_rows)
        table_widget.setColumnCount(num_cols)

        for row_idx, row_item in enumerate(dataframe.iterrows()):
            row_name, row_data = row_item
            table_widget.setVerticalHeaderItem(row_idx, QtWidgets.QTableWidgetItem(row_name))
            for col_idx, value in enumerate(row_data):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    item = QtWidgets.QTableWidgetItem("")
                else:
                    item = QtWidgets.QTableWidgetItem(str(value))
                table_widget.setItem(row_idx, col_idx, item)

        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()

    @staticmethod
    def _pad_with_nan(value, size):
        if size > 1:
            return np.pad(np.array([value], dtype="float64"), (0, size - 1), constant_values=np.nan)
        else:
            return np.array([value])


class ConductanceBarWidget(QtWidgets.QProgressBar):
    def __init__(self, color="#05B8CC"):
        super().__init__()
        self.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.setTextVisible(False)
        self.setRange(0, 100)
        self.setValue(0)
        self.setStyleSheet(
            """
QProgressBar::chunk {
    background-color: %s;
}
"""
            % (color)
        )

    def set_max_conductance(self, max_conductance):
        self.max_conductance = max_conductance

    def set_current_conductance(self, current_conductance):
        self.current_conductance = current_conductance

    def update(self):
        if self.max_conductance == 0:
            self.setValue(0)
        else:
            ratio = int(100 * (self.current_conductance / self.max_conductance))
            self.setValue(ratio)
