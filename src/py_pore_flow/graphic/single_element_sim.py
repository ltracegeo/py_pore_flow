import math

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from py_pore_flow.graphic.element_graphic import ElementPlotter
from py_pore_flow.graphic.network_plot import PorePlot, ThroatPlot
from py_pore_flow.network import initialize_pn, Network
from py_pore_flow.parameters.parameter_manager import DEFAULT_PARAMETERS, ParameterManager
from py_pore_flow.simulation.drainage import DrainageSimulation
from py_pore_flow.simulation.imbibition import ImbibitionSimulation
from py_pore_flow.simulation.simulation import (
    calculate_drainage_values,
    calculate_imbibition_values,
    calculate_initial_values,
)


class ManualSimulationPlot:
    def __init__(self, pn, parameters_dict=DEFAULT_PARAMETERS):
        self.pn = pn
        self.manual_simulation = ManualSimulation(pn, parameters_dict)
        self.manual_simulation.set_max_pc(40000)

    def set_pore(self, pore_id):
        self.element_widget.set_pore(pore_id)

    def set_throat(self, throat_id):
        self.element_widget.set_throat(throat_id)

    def show(self):
        app = QtWidgets.QApplication([])

        self.element_type_combobox = QtWidgets.QComboBox()
        self.element_type_combobox.addItem("Throat")
        self.element_type_combobox.addItem("Pore")
        self.element_id_input = QtWidgets.QLineEdit()
        self.element_id_input.setValidator(QtGui.QIntValidator())
        self.element_id_input.editingFinished.connect(lambda: self.__on_element_changed(self.element_id_input.text()))
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.cycle_combobox = QtWidgets.QComboBox()
        self.cycle_combobox.addItem("drainage")
        self.cycle_combobox.addItem("imbibition")
        self.cycle_combobox.currentTextChanged.connect(self.__on_cycle_changed)
        self.pc_input = QtWidgets.QLineEdit()
        self.pc_input.setValidator(QtGui.QDoubleValidator())
        self.pc_input.editingFinished.connect(lambda: self.__on_pc_changed(self.pc_input.text()))

        element_selection_layout = QtWidgets.QHBoxLayout()
        element_selection_layout.addWidget(self.element_type_combobox)
        element_selection_layout.addWidget(QtWidgets.QLabel("ID:"))
        element_selection_layout.addWidget(self.element_id_input)

        self.element_widget = ElementWidget(self.pn)

        central_widget = QtWidgets.QFrame()
        layout = QtWidgets.QFormLayout(central_widget)
        layout.addRow(element_selection_layout)
        layout.addRow(self.element_widget)
        layout.addRow("Cycle:", self.cycle_combobox)
        layout.addRow("Pc:", self.pc_input)

        win = QtWidgets.QMainWindow()
        win.setCentralWidget(central_widget)
        win.show()

        self.__on_cycle_changed("drainage")
        self.__on_pc_changed("0.0")

        app.exec()

    def __on_element_changed(self, text):
        if self.element_type_combobox.currentText() == "Throat":
            self.set_throat(int(text))
        else:
            self.set_pore(int(text))
        self.element_widget.update()

    def __on_cycle_changed(self, text):
        self.manual_simulation.set_cycle(text)
        self.element_widget.update()

    def __on_pc_changed(self, text):
        self.manual_simulation.update(float(text))
        self.element_widget.update()


class ElementWidget(QtWidgets.QFrame):
    def __init__(self, pn):
        super().__init__()

        self.pn = pn

        self.element_plotter = ElementPlotter()
        self.element_plot = None

        self.graph = pg.GraphicsLayoutWidget()
        self.table = QtWidgets.QTableWidget()
        self.plot = self.graph.addPlot()

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.graph)
        splitter.addWidget(self.table)
        splitter.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(splitter)

    def set_pore(self, pore_id):
        self.element_plot = PorePlot(self.pn, pore_id)

    def set_throat(self, throat_id):
        self.element_plot = ThroatPlot(self.pn, throat_id)

    def update(self):
        if not self.element_plot:
            return

        self.element_plot.update()
        self.element_plotter.plot_pyqtgraph_element(self.plot, self.element_plot.get_graphic())
        self.__update_table(self.element_plot)

    def __update_table(self, element_plot):
        element_type = element_plot.element_type
        id = element_plot.id
        n_corners = element_plot.n_corners

        radius = self.__pad_with_nan(self.pn[f"{element_type}.radius"][id], n_corners)
        beta = np.degrees(np.resize(self.pn[f"{element_type}.corner_half_angle"][id], n_corners))
        conns = np.resize(self.pn[f"{element_type}.conns"][id], self.pn[f"{element_type}.n_conns"][id])
        sw = self.__pad_with_nan(self.pn[f"{element_type}.water_saturation"][id], n_corners)
        theta_h = np.degrees(np.resize(self.pn[f"{element_type}.hinging_contact_angle"][id], n_corners))
        theta_a = np.degrees(np.resize(self.pn[f"{element_type}.advancing_contact_angle"][id], n_corners))
        theta_r = np.degrees(np.resize(self.pn[f"{element_type}.receding_contact_angle"][id], n_corners))
        b = np.resize(self.pn[f"{element_type}.b"][id], n_corners)
        bo = np.resize(self.pn[f"{element_type}.bo"][id], n_corners)
        corner_oil_layer = np.resize(self.pn[f"{element_type}.corner_oil_layer"][id], n_corners)
        entry_pressure = self.__pad_with_nan(self.pn[f"{element_type}.entry_pressure"][id], n_corners)

        if element_type == "pore":
            if self.pn["pore.inlets"][id]:
                conns = ["inlet"] + conns.tolist()
            elif self.pn["pore.outlets"][id]:
                conns = ["outlet"] + conns.tolist()

        df = pd.DataFrame.from_dict(
            {
                "radius:": radius,
                "beta:": beta,
                "conns": [str(conns)],
                "Sw": sw,
                "theta_h": theta_h,
                "theta_a": theta_a,
                "theta_r": theta_r,
                "bi": b,
                "bo": bo,
                "corner_oil_layer": corner_oil_layer,
                "entry_pressure": entry_pressure,
            },
            orient="index",
        )

        self.__df_to_widget(df, self.table)

    @staticmethod
    def __df_to_widget(dataframe, table_widget: QtWidgets.QTableWidget):
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
    def __pad_with_nan(value, size):
        if size > 1:
            return np.pad(np.array([value], dtype="float64"), (0, size - 1), constant_values=np.nan)
        else:
            return np.array([value])


class ManualSimulation:
    def __init__(self, pn, parameters_dict=DEFAULT_PARAMETERS):
        self.simulation = None
        self.max_pc = None

        self.parameter_manager = ParameterManager(parameters_dict)

        initialize_pn(pn)
        self.network = Network(pn)

        self.parameter_manager.generate_contact_angles(self.network)

        calculate_initial_values(self.network)

    def set_max_pc(self, max_pc):
        self.max_pc = max_pc

    def set_cycle(self, cycle="imbibition"):
        if cycle == "drainage":
            self.parameter_manager.update_initial_cas(self.network)
            calculate_drainage_values(self.network, self.max_pc)
            self.simulation = DrainageSimulation(self.network)
        else:
            self.parameter_manager.update_equilibrium_cas(self.network)
            calculate_imbibition_values(self.network, self.max_pc)
            self.simulation = ImbibitionSimulation(self.network, self.max_pc)

    def update(self, pc):
        # 17125.8354
        self.simulation.update_pc(pc)
        Sw = self.simulation.calculate_sw_total()
        Krw, Kro = self.simulation.calculate_krel()
        pores_filled, total_pores, percentage = self.simulation.get_oil_filled_percentage()
        print(
            f"Pc: {pc:8.2f}, Sw: {Sw:.6f}, Krw: {Krw:.6f}, Kro: {Kro:.6f}, Filled: {pores_filled:4}/{total_pores} ({percentage:2}%)."
        )
