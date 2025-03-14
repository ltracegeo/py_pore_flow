import numpy as np
import openpnm as op
from PySide2 import QtCore, QtGui, QtTest, QtWidgets
import pyqtgraph as pg

from py_pore_flow.graphic.manual_simulation import ElementWidget
from py_pore_flow.simulation.simulation import StepSimulation


class TestDisplay:
    def __init__(self):
        self.pn = None
        self.manual_simulation = None
        self.win = None

    def set_pn(self, pn, parameters_dict={}):
        self.pn = pn
        self.manual_simulation = StepSimulation(pn, parameters_dict, generate_cas=False)
        self.manual_simulation.set_max_pc(40000)
        self.element_widget.set_pn(self.pn)

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
        self.element_widget = ElementWidget()

        central_widget = QtWidgets.QFrame()
        layout = QtWidgets.QFormLayout(central_widget)
        layout.addRow(self.element_widget)

        self.win = QtWidgets.QMainWindow()
        self.win.setCentralWidget(central_widget)
        self.win.show()

    def close(self):
        self.win.close()


class GraphicalTests:
    def __init__(self):
        self.test_display = TestDisplay()

    def run(self):
        self.test_display.show()

        # self.oil_layer()
        # self.snap_off_water_layer()
        # self.collapse_oil_layer()
        # self.snap_off_2nd_drainage()
        self.oil_layer_imbibition_1()
        self.oil_layer_imbibition_2()

        self.test_display.close()

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
        self.test_display.set_pn(pn)

        self.test_display.set_throat(0)
        self.test_display.set_cycle("drainage")

        # Invasion drainage
        self.test_display.update_pc(6000, 1000)

        self.test_display.set_cycle("imbibition", 1000)
        self.test_display.update_pc(-4500, 1000)
        self.test_display.update_pc(-4600, 1000)
        self.test_display.update_pc(-6000, 1000)
        self.test_display.update_pc(-8000, 1000)
        self.test_display.update_pc(-15000, 1000)

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
            "pore.shape_factor": np.array([0.04, 0.06]),
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
        self.test_display.set_pn(pn)

        self.test_display.set_throat(0)
        self.test_display.set_cycle("drainage")
        self.test_display.update_pc(55000, 500)
        self.test_display.set_cycle("imbibition", 500)
        self.test_display.update_pc(5000, 500)
        self.test_display.update_pc(4000, 500)
        self.test_display.update_pc(3000, 500)
        self.test_display.update_pc(2000, 500)
        self.test_display.update_pc(1900, 500)
        self.test_display.update_pc(1800, 500)
        self.test_display.update_pc(1700, 500)
        self.test_display.update_pc(1600, 1000)
        self.test_display.update_pc(1549, 2000)
        self.test_display.update_pc(1546, 2000)

    def collapse_oil_layer(self):
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
            "pore.shape_factor": np.array([0.04, 0.06]),
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
            "throat.equilibrium_advancing_ca": np.array([np.radians(106)]),
            "throat.equilibrium_receding_ca": np.array([np.radians(60)]),
        }

        pn.update(pn_dict)
        self.test_display.set_pn(pn)

        self.test_display.set_pore(0)
        self.test_display.set_cycle("drainage")

        # Invasion drainage
        self.test_display.update_pc(6000, 1000)

        self.test_display.set_cycle("imbibition", 1000)
        self.test_display.update_pc(-2007, 1000)
        self.test_display.update_pc(-2300, 500)
        self.test_display.update_pc(-2500, 500)
        self.test_display.update_pc(-2669, 1000)
        self.test_display.update_pc(-2670, 500)
        self.test_display.update_pc(-3000, 500)
        self.test_display.update_pc(-3521, 1000)
        self.test_display.update_pc(-3522, 1000)

    def snap_off_2nd_drainage(self):
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
            "pore.shape_factor": np.array([0.04, 0.04]),
            "pore.clay_volume": np.array([0, 0], dtype=np.int64),
            "throat.cross_sectional_area": np.array([3.52112676e-10]),
            "pore.area": np.array([4.15606178e-10, 4.15606178e-10]),
            "pore.outlets": np.array([False, True]),
            "pore.inlets": np.array([True, False]),
            "pore.initial_advancing_ca": np.array([np.radians(40), np.radians(40)]),
            "pore.initial_receding_ca": np.array([np.radians(20), np.radians(20)]),
            "pore.equilibrium_advancing_ca": np.array([np.radians(140), np.radians(140)]),
            "pore.equilibrium_receding_ca": np.array([np.radians(20), np.radians(70)]),
            "throat.initial_advancing_ca": np.array([np.radians(70)]),
            "throat.initial_receding_ca": np.array([np.radians(20)]),
            "throat.equilibrium_advancing_ca": np.array([np.radians(140)]),
            "throat.equilibrium_receding_ca": np.array([np.radians(60)]),
        }

        pn.update(pn_dict)
        self.test_display.set_pn(pn)

        self.test_display.set_pore(1)
        self.test_display.set_cycle("drainage")
        self.test_display.update_pc(6000, 1000)

        self.test_display.set_cycle("imbibition", 1000)
        self.test_display.update_pc(-2007, 1000)

        self.test_display.set_cycle("2nd_drainage", 1000)
        self.test_display.update_pc(-1000, 500)
        self.test_display.update_pc(0, 500)
        self.test_display.update_pc(500, 1000)
        self.test_display.update_pc(1000, 500)
        self.test_display.update_pc(1500, 500)
        self.test_display.update_pc(2000, 1000)
        self.test_display.update_pc(2080, 1000)

    def oil_layer_imbibition_1(self):
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
            "pore.shape_factor": np.array([0.04, 0.04]),
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

        self.test_display.set_pn(pn)

        self.test_display.set_pore(0)
        self.test_display.set_cycle("drainage")
        self.test_display.update_pc(5000, 1000)

        self.test_display.set_cycle("imbibition", 1000)
        self.test_display.update_pc(-100, 1000)
        self.test_display.update_pc(-1500, 1000)
        self.test_display.update_pc(-4300, 1000)

    def oil_layer_imbibition_2(self):
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
            "pore.shape_factor": np.array([0.04, 0.04]),
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

        self.test_display.set_pn(pn)

        self.test_display.set_pore(0)
        self.test_display.set_cycle("drainage")
        self.test_display.update_pc(10000, 1000)

        self.test_display.set_cycle("imbibition", 1000)
        self.test_display.update_pc(-100, 1000)
        self.test_display.update_pc(-3032, 1000)
        self.test_display.update_pc(-4205, 1000)

        self.test_display.set_cycle("2nd_drainage", 1000)
        self.test_display.update_pc(-1000, 500)
        self.test_display.update_pc(0, 500)
        self.test_display.update_pc(2060, 500)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    graphical_test = GraphicalTests()
    QtCore.QTimer.singleShot(1, graphical_test.run)
    app.exec_()
