import numpy as np
import openpnm as op
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets
import pyqtgraph as pg

from py_pore_flow.graphic.single_element_sim import ElementWidget, ManualSimulationPlot, ManualSimulation
from py_pore_flow.parameters.parameter_manager import DEFAULT_PARAMETERS


class TestDisplay:
    def __init__(self, pn, parameters_dict=DEFAULT_PARAMETERS):
        self.pn = pn
        self.manual_simulation = ManualSimulation(pn, parameters_dict, generate_cas=False)
        self.manual_simulation.set_max_pc(40000)
        self.win = None

    def set_pore(self, pore_id):
        self.element_widget.set_pore(pore_id)

    def set_throat(self, throat_id):
        self.element_widget.set_throat(throat_id)

    def set_cycle(self, cycle_name, wait_m=0):
        self.manual_simulation.set_cycle(cycle_name)
        self.element_widget.update()
        QtTest.QTest.qWait(wait_m)

    def update_pc(self, pc, wait_m=0):
        self.manual_simulation.update(pc)
        self.element_widget.update()
        QtTest.QTest.qWait(wait_m)

    def show(self):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.element_widget = ElementWidget(self.pn)

        central_widget = QtWidgets.QFrame()
        layout = QtWidgets.QFormLayout(central_widget)
        layout.addRow(self.element_widget)

        self.win = QtWidgets.QMainWindow()
        self.win.setCentralWidget(central_widget)
        self.win.show()


class GraphicalTests:
    def __init__(self):
        pass

    def run(self):
        # self.oil_layer()
        self.snap_off_water_layer()

    def oil_layer(self):
        pn = op.network.Network()
        pn_dict = {
            "throat.conns": np.array([[0, 1]]),
            "throat.radius": np.array([1.0e-05]),
            "throat.shape_factor": np.array([0.04]),
            "throat.total_length": np.array([2.2e-05]),
            "throat.length": np.array([2.00000e-06]),
            "throat.conduit_lengths.throat": np.array([2.00000e-06]),
            "throat.volume": np.array([6.26186e-15]),
            "throat.conduit_lengths.pore1": np.array([1.0e-05]),
            "throat.conduit_lengths.pore2": np.array([1.0e-05]),
            "throat.clay_volume": np.array([0], dtype=np.int64),
            "pore.coords": np.array([[1.45e-05, 1.45e-05, 0.00e00], [1.45e-05, 1.45e-05, 2.20e-05]]),
            "pore.volume": np.array([5.78308e-15, 5.97317e-15]),
            "pore.radius": np.array([1.0e-05, 1.0e-05]),
            "pore.shape_factor": np.array([0.071, 0.06]),
            "pore.clay_volume": np.array([0, 0], dtype=np.int64),
            "throat.cross_sectional_area": np.array([3.52112676e-10]),
            "pore.area": np.array([4.15606178e-10, 4.15606178e-10]),
            "pore.outlets": np.array([False, True]),
            "pore.inlets": np.array([True, False]),
            "pore.initial_advancing_ca": np.array([np.radians(40), np.radians(40)]),
            "pore.initial_receding_ca": np.array([np.radians(20), np.radians(20)]),
            "pore.equilibrium_advancing_ca": np.array([np.radians(140), np.radians(140)]),
            "pore.equilibrium_receding_ca": np.array([np.radians(70), np.radians(110)]),
            "throat.initial_advancing_ca": np.array([np.radians(40)]),
            "throat.initial_receding_ca": np.array([np.radians(20)]),
            "throat.equilibrium_advancing_ca": np.array([np.radians(140)]),
            "throat.equilibrium_receding_ca": np.array([np.radians(60)]),
        }

        pn.update(pn_dict)
        simulation = TestDisplay(pn)
        simulation.show()

        simulation.set_throat(0)
        simulation.set_cycle("drainage")

        # Invasion drainage
        simulation.update_pc(6000, 1000)

        simulation.set_cycle("imbibition", 1000)
        simulation.update_pc(1000, 1000)
        simulation.update_pc(0, 1000)
        simulation.update_pc(-1000, 1000)
        simulation.update_pc(-4000, 1000)

        # Invasion imbibition
        simulation.update_pc(-5000, 1000)
        simulation.update_pc(-6000, 500)
        simulation.update_pc(-7000, 500)
        simulation.update_pc(-8000, 500)
        simulation.update_pc(-9000, 500)
        simulation.update_pc(-10000, 500)
        simulation.update_pc(-11000, 500)
        simulation.update_pc(-12000, 5000)

    def snap_off_water_layer(self):
        pn = op.network.Network()
        pn_dict = {
            "throat.conns": np.array([[0, 1]]),
            "throat.radius": np.array([1.0e-05]),
            "throat.shape_factor": np.array([0.04]),
            "throat.total_length": np.array([2.2e-05]),
            "throat.length": np.array([2.00000e-06]),
            "throat.conduit_lengths.throat": np.array([2.00000e-06]),
            "throat.volume": np.array([6.26186e-15]),
            "throat.conduit_lengths.pore1": np.array([1.0e-05]),
            "throat.conduit_lengths.pore2": np.array([1.0e-05]),
            "throat.clay_volume": np.array([0], dtype=np.int64),
            "pore.coords": np.array([[1.45e-05, 1.45e-05, 0.00e00], [1.45e-05, 1.45e-05, 2.20e-05]]),
            "pore.volume": np.array([5.78308e-15, 5.97317e-15]),
            "pore.radius": np.array([1.0e-06, 1.0e-06]),
            "pore.shape_factor": np.array([0.071, 0.06]),
            "pore.clay_volume": np.array([0, 0], dtype=np.int64),
            "throat.cross_sectional_area": np.array([3.52112676e-10]),
            "pore.area": np.array([4.15606178e-10, 4.15606178e-10]),
            "pore.outlets": np.array([False, True]),
            "pore.inlets": np.array([True, False]),
            "pore.initial_advancing_ca": np.array([np.radians(40), np.radians(40)]),
            "pore.initial_receding_ca": np.array([np.radians(20), np.radians(20)]),
            "pore.equilibrium_advancing_ca": np.array([np.radians(140), np.radians(140)]),
            "pore.equilibrium_receding_ca": np.array([np.radians(70), np.radians(110)]),
            "throat.initial_advancing_ca": np.array([np.radians(40)]),
            "throat.initial_receding_ca": np.array([np.radians(20)]),
            "throat.equilibrium_advancing_ca": np.array([np.radians(40)]),
            "throat.equilibrium_receding_ca": np.array([np.radians(20)]),
        }

        pn.update(pn_dict)
        simulation = TestDisplay(pn)
        simulation.show()

        simulation.set_throat(0)
        simulation.set_cycle("drainage")
        simulation.update_pc(55000, 500)
        simulation.set_cycle("imbibition", 500)
        simulation.update_pc(5000, 500)
        simulation.update_pc(4000, 500)
        simulation.update_pc(3000, 500)
        simulation.update_pc(2000, 500)
        simulation.update_pc(1900, 500)
        simulation.update_pc(1800, 500)
        simulation.update_pc(1700, 500)
        simulation.update_pc(1600, 2000)
        simulation.update_pc(1500, 2000)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    graphical_test = GraphicalTests()
    QtCore.QTimer.singleShot(1, graphical_test.run)
    app.exec()
