import datetime
import time

import openpnm
from PyQt6 import QtWidgets, QtGui
import pyqtgraph as pg

import py_pore_flow as ppf
from py_pore_flow.graphic.element_graphic import WATER, OIL, ElementGraphic, ElementPlotter
from py_pore_flow.graphic.single_element_sim import ManualSimulationPlot


def see_props():
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", "a")
    print(pn.props())


def show_graph(cycle, Sw, Krw, Kro):
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget()
    plot_item = win.addPlot(left="Krel", bottom="Sw")

    for i, green in enumerate([0, 100]):
        indexes = cycle == (i + 1)
        krw_curve_item = pg.PlotCurveItem(Sw[indexes], Krw[indexes], pen=pg.mkPen(QtGui.QColor(0, green, 255), width=2))
        kro_curve_item = pg.PlotCurveItem(Sw[indexes], Kro[indexes], pen=pg.mkPen(QtGui.QColor(255, green, 0), width=2))
        plot_item.addItem(krw_curve_item)
        plot_item.addItem(kro_curve_item)

    win.show()
    app.exec()


def run(sample="a", generate_vtu=False, plot_elements=False, debug=True):
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", sample)

    start = time.time()

    ppf.log.configure(output=ppf.log.STDOUT, level=ppf.log.DEBUG)

    cycle, Pc, Sw, Krw, Kro = ppf.run_two_phase_simulation(
        pn, generate_vtu=generate_vtu, debug=debug, plot_elements=plot_elements
    )

    time_elapsed = datetime.timedelta(seconds=int(time.time() - start))

    print(f"Time elapsed: {time_elapsed}")

    if debug:
        show_graph(cycle, Sw, Krw, Kro)


def draw():
    initial_element = ElementGraphic(
        [80, 70, 30], 1, [0, 0, 0], [80, 80, 80], [0, 0, 0], [20, 20, 20], phase_in_middle=WATER
    )
    initial_element.set_name("initial state")
    initial_element.update()

    oil_filled_element = ElementGraphic(
        [80, 70, 30], 1, [0.5, 0.2, 0.1], [80, 80, 80], [0, 0, 0], [20, 20, 20], phase_in_middle=OIL
    )
    oil_filled_element.set_name("oil filled")
    oil_filled_element.update()

    water_filled_element = ElementGraphic(
        [80, 70, 30], 1, [0.5, 0.2, 0.1], [80, 80, 80], [1.2, 0.9, 0.8], [20, 20, 20], phase_in_middle=WATER
    )
    water_filled_element.set_name("water filled")
    water_filled_element.update()

    square_element = ElementGraphic(
        [90, 90, 90, 90], 1, [0.5, 0.2, 0.1, 0.1], [80, 80, 80, 80], [0, 0, 0, 0], [20, 20, 20, 20], phase_in_middle=OIL
    )
    square_element.set_name("square")
    square_element.update()

    circle_element = ElementGraphic([], 1, [], [], [], [], phase_in_middle=OIL)
    circle_element.set_name("circle")
    circle_element.update()

    element_plotter = ElementPlotter()
    element_plotter.add_element_graphic(initial_element)
    element_plotter.add_element_graphic(oil_filled_element)
    element_plotter.add_element_graphic(water_filled_element)
    element_plotter.add_element_graphic(square_element)
    element_plotter.add_element_graphic(circle_element)

    for i in range(4):
        oil_filled_element.distance = [0.5 + 0.1 * i, 0.2, 0.1]
        oil_filled_element.update()
        element_plotter.show(pause=1, block=False)


def manual_sim():
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", "Image")
    simulation = ManualSimulationPlot(pn)

    app = QtWidgets.QApplication([])
    simulation.show()
    app.exec()


def profile():
    import cProfile

    cProfile.run("run(False)", filename="profile_results")


if __name__ == "__main__":
    run()
