import math

import matplotlib.pyplot as plt
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from py_pore_flow.graphic.element_graphic import WATER, OIL


class ElementPlotter:
    def __init__(self):
        self.oil_color = "#ffbfbf"
        self.water_color = "#abcbff"
        self.line_color = "#FFFFFF"

        self.element_graphic_list = []
        self.axis_list = []

    def add_element_graphic(self, element_graphic):
        self.element_graphic_list.append(element_graphic)

    def show(self, pause=0, block=False):
        number_of_plots = len(self.element_graphic_list)
        if number_of_plots != len(self.axis_list):
            self.axis_list = self.reset_axes(number_of_plots)

        for i, element_graphic in enumerate(self.element_graphic_list):
            self.plot_element(self.axis_list[i], element_graphic)

        plt.show(block=block)
        if pause > 0:
            plt.pause(pause)

    def plot_element(self, axis, element_graphic):
        center_color = self.oil_color if element_graphic.phase_in_middle == OIL else self.water_color
        inner_layers_color = self.water_color
        outter_layers_color = self.oil_color

        axis.clear()
        self.plot_graphic(
            axis,
            element_graphic.name,
            element_graphic.vertices,
            element_graphic.inner_arcs,
            element_graphic.outter_arcs,
            center_color,
            inner_layers_color,
            outter_layers_color,
            self.line_color,
        )

        axis.text(0.0, -0.2, element_graphic.description, size=12, ha="left", transform=axis.transAxes)

    def plot_pyqtgraph_element(self, plot, element_graphic):
        center_color = self.oil_color if element_graphic.phase_in_middle == OIL else self.water_color
        inner_layers_color = self.water_color
        outter_layers_color = self.oil_color

        plot.clear()
        self.plot_pyqtgraph_graphic(
            plot,
            element_graphic.name,
            element_graphic.vertices,
            element_graphic.inner_arcs,
            element_graphic.outter_arcs,
            center_color,
            inner_layers_color,
            outter_layers_color,
            self.line_color,
        )

    @staticmethod
    def reset_axes(number_of_plots):
        plt.close()
        fig, ax = plt.subplots(*ElementPlotter.find_2d_dimensions(number_of_plots))
        fig.tight_layout(pad=1)

        if number_of_plots > 1:
            axis_list = ax.flat
        else:
            axis_list = [ax]
        fig.suptitle("Elements")
        plt.grid(False)

        for axis in axis_list[number_of_plots:]:
            axis.axis("off")

        return axis_list[:number_of_plots]

    @staticmethod
    def plot_graphic(
        axis, name, vertices, inner_arcs, outter_arcs, center_color, inner_layers_color, outter_layers_color, line_color
    ):
        axis.fill(vertices[:, 0], vertices[:, 1], center_color)
        borders_x = np.concatenate((vertices[:, 0], [vertices[0][0]]))
        borders_y = np.concatenate((vertices[:, 1], [vertices[0][1]]))
        axis.plot(borders_x, borders_y, line_color, linewidth=1)  # Plot vertices in blue

        # draw oil layeres
        for arc_x, arc_y in outter_arcs:
            if arc_x is None:
                continue
            axis.fill(arc_x, arc_y, outter_layers_color)
            axis.plot(arc_x, arc_y, line_color, linewidth=1)

        # draw water layers
        for arc_x, arc_y in inner_arcs:
            if arc_x is None:
                continue
            axis.fill(arc_x, arc_y, inner_layers_color)
            axis.plot(arc_x, arc_y, line_color, linewidth=1)

        axis.set_title(name)
        axis.set_aspect("equal", adjustable="box")
        axis.label_outer()

    @staticmethod
    def plot_pyqtgraph_graphic(
        plot, name, vertices, inner_arcs, outter_arcs, center_color, inner_layers_color, outter_layers_color, line_color
    ):
        element_item = ElementItem(
            name, vertices, inner_arcs, outter_arcs, center_color, inner_layers_color, outter_layers_color, line_color
        )
        plot.setAspectLocked(lock=True, ratio=1)
        plot.addItem(element_item)

    @staticmethod
    def find_2d_dimensions(n):
        """
        Find the 2D dimensions (rows, cols) for a given number of elements n such that the grid
        is as close to square as possible.

        Args:
        n (int): The number of elements.

        Returns:
        tuple: A tuple (rows, cols) representing the dimensions of the grid.
        """
        if n <= 0:
            return (0, 0)

        # Start with the assumption that the grid is as square as possible
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        return (rows, cols)


class ElementItem(pg.GraphicsObject):
    def __init__(
        self, name, vertices, inner_arcs, outter_arcs, center_color, inner_layers_color, outter_layers_color, line_color
    ):
        super().__init__()

        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)

        polygon = QtGui.QPolygonF()
        for x, y in zip(vertices[:, 0], vertices[:, 1]):
            polygon.append(QtCore.QPointF(x, y))
        painter.setPen(pg.mkPen(QtGui.QColor(line_color), width=2))
        painter.setBrush(pg.mkBrush(QtGui.QColor(center_color)))
        painter.drawPolygon(polygon)

        for arc_x, arc_y in outter_arcs:
            if arc_x is None:
                continue
            polygon = QtGui.QPolygonF()
            for i in range(len(arc_x)):
                polygon.append(QtCore.QPointF(arc_x[i], arc_y[i]))
            painter.setPen(pg.mkPen(QtGui.QColor(line_color), width=2))
            painter.setBrush(pg.mkBrush(QtGui.QColor(outter_layers_color)))
            painter.drawPolygon(polygon)

        for arc_x, arc_y in inner_arcs:
            if arc_x is None:
                continue
            polygon = QtGui.QPolygonF()
            for i in range(len(arc_x)):
                polygon.append(QtCore.QPointF(arc_x[i], arc_y[i]))
            painter.setPen(pg.mkPen(QtGui.QColor(line_color), width=2))
            painter.setBrush(pg.mkBrush(QtGui.QColor(inner_layers_color)))
            painter.drawPolygon(polygon)

        painter.end()

        self.rect = QtCore.QRectF(
            QtCore.QPointF(np.min(vertices[:, 0]), np.min(vertices[:, 1])),
            QtCore.QPointF(np.max(vertices[:, 0]), np.max(vertices[:, 1])),
        )

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return self.rect
