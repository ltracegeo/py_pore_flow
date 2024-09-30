from PySide2 import QtWidgets, QtGui
import pyqtgraph as pg


def plot_krel(cycle, Sw, Krw, Kro):
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
    app.exec_()
