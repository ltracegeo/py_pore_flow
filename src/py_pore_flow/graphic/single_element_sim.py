import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox

from py_pore_flow.graphic.network_plot import NetworkPlot
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
        self.manuel_simulation = ManualSimulation(pn, parameters_dict)
        self.manuel_simulation.set_max_pc(40000)
        self.network_plot = NetworkPlot(pn)

    def add_pore(self, pore_id):
        self.network_plot.create_pore_plot(pore_id)

    def add_throat(self, throat_id):
        self.network_plot.create_throat_plot(throat_id)

    def show(self):
        plt.close()
        fig, self.axis = plt.subplot_mosaic("A;B;C", gridspec_kw=dict(height_ratios=[1, 0.2, 0.2]))
        fig.tight_layout(pad=1)

        fig.suptitle("Elements")
        plt.grid(False)

        self.__update_plot()

        self.radio_buttons = RadioButtons(self.axis["B"], ["drainage", "imbibition"])
        self.radio_buttons.on_clicked(self.__on_cycle_changed)

        self.text_box = TextBox(self.axis["C"], "Pc", initial="0.0")
        self.text_box.on_submit(self.__on_pc_changed)

        self.__on_cycle_changed("drainage")
        self.__on_pc_changed("0.0")

        plt.show()

    def __on_cycle_changed(self, text):
        self.manuel_simulation.set_cycle(text)
        self.__update_plot()

    def __on_pc_changed(self, text):
        self.manuel_simulation.update(float(text))
        self.__update_plot()

    def __update_plot(self):
        self.network_plot.plot_list[0].update()
        element_plotter = self.network_plot.element_plotter
        element_plotter.plot_element(self.axis["A"], element_plotter.element_graphic_list[0])
        plt.draw()


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
