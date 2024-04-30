import datetime
import time

import openpnm
from PyQt6 import QtWidgets, QtGui
import pyqtgraph as pg

import py_pore_flow as ppf


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


def run(debug=True):
    pn = openpnm.io.network_from_statoil("tests/resources/statoil", "a")

    start = time.time()

    ppf.log.configure(output=ppf.log.STDOUT, level=ppf.log.DEBUG)

    cycle, Pc, Sw, Krw, Kro = ppf.run_two_phase_simulation(pn, generate_vtu=debug, debug=debug)

    time_elapsed = datetime.timedelta(seconds=int(time.time() - start))

    print(f"Time elapsed: {time_elapsed}")

    if debug:
        show_graph(cycle, Sw, Krw, Kro)


def profile():
    import cProfile

    cProfile.run("run(False)", filename="profile_results")


if __name__ == "__main__":
    run()
