import numpy as np

from py_pore_flow.graphic.element_graphic import WATER, OIL, ElementGraphic, ElementPlotter


class NetworkPlot:
    def __init__(self, pn):
        """
        pn(network): openpnm network
        """
        self.pn = pn
        self.plot_list = []

        self.element_plotter = ElementPlotter()

    def create_pore_plot(self, pore_id):
        new_plot = PorePlot(self.pn, pore_id)
        self.plot_list.append(new_plot)
        self.element_plotter.add_element_graphic(new_plot.get_graphic())
        return new_plot

    def create_throat_plot(self, throat_id):
        new_plot = ThroatPlot(self.pn, throat_id)
        self.plot_list.append(new_plot)
        self.element_plotter.add_element_graphic(new_plot.get_graphic())
        return new_plot

    def update(self):
        for plot in self.plot_list:
            plot.update()

        self.element_plotter.show(pause=0.001)


class ElementPlot:
    def __init__(self, pn, element_type, element_id):
        self.pn = pn
        self.element_type = element_type
        self.id = element_id

        self.n_corners = self.pn[f"{self.element_type}.n_corners"][self.id]

        self.graphic = ElementGraphic()
        self.graphic.angles = np.resize(self.pn[f"{self.element_type}.corner_half_angle"][self.id] * 2, self.n_corners)
        self.graphic.radius = self.pn[f"{self.element_type}.radius"][self.id]
        self.update()

    def update(self):
        self.graphic.distance = np.abs(np.resize(self.pn[f"{self.element_type}.b"][self.id], self.n_corners))
        self.graphic.theta = np.resize(self.pn[f"{self.element_type}.hinging_contact_angle"][self.id], self.n_corners)
        self.graphic.distance_o = np.abs(np.resize(self.pn[f"{self.element_type}.bo"][self.id], self.n_corners))
        self.graphic.theta_o = np.resize(
            self.pn[f"{self.element_type}.advancing_contact_angle"][self.id], self.n_corners
        )
        self.graphic.phase_in_middle = OIL if self.pn[f"{self.element_type}.oil_filled"][self.id] else WATER

        self.graphic.update()

    def get_graphic(self):
        return self.graphic


class PorePlot(ElementPlot):
    def __init__(self, pn, pore_id):
        super().__init__(pn, "pore", pore_id)
        if self.pn["pore.inlets"][pore_id]:
            self.graphic.set_name(f"inlet pore #{pore_id}")
        elif self.pn["pore.outlets"][pore_id]:
            self.graphic.set_name(f"outlet pore #{pore_id}")
        else:
            self.graphic.set_name(f"pore #{pore_id}")


class ThroatPlot(ElementPlot):
    def __init__(self, pn, throat_id):
        super().__init__(pn, "throat", throat_id)
        self.graphic.set_name(f"throat #{throat_id}")
