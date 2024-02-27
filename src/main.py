import datetime
import time

import openpnm
from PyQt6 import QtWidgets, QtGui
import pyqtgraph as pg

from py_pore_flow.simulation.simulation import Simulation


def see_props():
    pn = openpnm.io.network_from_statoil("statoil", "a")
    print(pn.props())


def show_graph(Sw, Krw, Kro):
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget()
    plot_item = win.addPlot(left="Krel", bottom="Sw")

    for i, green in enumerate([0, 100]):
        krw_curve_item = pg.PlotCurveItem(Sw[i], Krw[i], pen=pg.mkPen(QtGui.QColor(0, green, 255), width=2))
        kro_curve_item = pg.PlotCurveItem(Sw[i], Kro[i], pen=pg.mkPen(QtGui.QColor(255, green, 0), width=2))
        plot_item.addItem(krw_curve_item)
        plot_item.addItem(kro_curve_item)

    win.show()
    app.exec()


def run():
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", "a")

    Pc = [None] * 2
    Sw = [None] * 2
    Krw = [None] * 2
    Kro = [None] * 2

    start = time.time()
    simulation = Simulation(pn, generate_vtu=True, debug=True)
    Pc[0], Sw[0], Krw[0], Kro[0] = simulation.run_drainage()
    Pc[1], Sw[1], Krw[1], Kro[1] = simulation.run_imbibition()
    time_elapsed = datetime.timedelta(seconds=int(time.time() - start))

    print(f"Time elapsed: {time_elapsed}")

    show_graph(Sw, Krw, Kro)


if __name__ == "__main__":
    run()
