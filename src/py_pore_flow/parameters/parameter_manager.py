import math
from math import pi

from numba import njit
import numpy as np

from py_pore_flow import consts
from py_pore_flow import global_random
from py_pore_flow.global_random import random, weibull, gaussian


UNCORRELATED = 0
POSITIVE_RADIUS = 1
NEGATIVE_RADIUS = 2


DEFAULT_PARAMETERS = {
    "seed": 0,
    "water_viscosity": 0.00105,
    "water_density": 1000,
    "oil_viscosity": 0.00139,
    "oil_density": 900,
    "interfacial_tension": 0.03,
    "initial_ca_center": 30,
    "initial_ca_range": 15,
    "initial_ca_gamma": -1,
    "initial_ca_delta": -1,
    "initial_ca_sigma": 5,
    "initial_ca_correlation": UNCORRELATED,
    "initial_ca_model": 3,
    "initial_ca_distribution": "weibull",
    "initial_ca_separation": 25.2,
    "equilibrium_ca_center": 30,
    "equilibrium_ca_range": 15,
    "equilibrium_ca_gamma": -1,
    "equilibrium_ca_delta": -1,
    "equilibrium_ca_sigma": 5,
    "equilibrium_ca_correlation": UNCORRELATED,
    "equilibrium_ca_model": 3,
    "equilibrium_ca_distribution": "weibull",
    "equilibrium_ca_separation": 25.2,
    "second_ca_center": 140,
    "second_ca_range": 15,
    "second_ca_gamma": -1,
    "second_ca_delta": -1,
    "second_ca_sigma": 5,
    "second_ca_correlation": UNCORRELATED,
    "second_ca_distribution": "weibull",
    "second_ca_fraction": 0.5,
    "pc_initial": 0,
    "pc_maximum": 40000,
    "pc_minimum": -40000,
    "pc_step": 200,
    "pc_step_min": 1,
    "pc_step_max": 1000,
    "drainage_sw_step": 0.005,
    "imbibition_sw_step": 0.005,
    "use_pore_fill": True,
    "pore_fill_algorithm": "oren2",
    "pore_fill_weight_a1": 0.0,
    "pore_fill_weight_a2": 0.5,
    "pore_fill_weight_a3": 1.0,
    "pore_fill_weight_a4": 2.0,
    "pore_fill_weight_a5": 5.0,
    "pore_fill_weight_a6": 10.0,
    "end_at_sor": True,
    "control_by_pc": False,
    "skip_imbibition": False,
    "skip_2nd_drainage": False,
    "enforced_swi_1": 0.0,
    "enforced_swi_2": 1.0,
    "batch_invasions": "none",
}


class ParameterManager:
    def __init__(self, parameters_dict={}):
        self.parameters_dict = DEFAULT_PARAMETERS
        self.parameters_dict.update(parameters_dict)

        if self.parameters_dict["seed"] is None:
            seed = np.random.randint(0, 999999999)
        else:
            seed = self.parameters_dict["seed"]
        global_random.set_seed(seed)

        consts.pw = self.parameters_dict["water_density"]
        consts.po = self.parameters_dict["oil_density"]
        consts.sigma = self.parameters_dict["interfacial_tension"]
        consts.miw = self.parameters_dict["water_viscosity"]
        consts.mio = self.parameters_dict["oil_viscosity"]

        consts.use_pore_fill = self.parameters_dict["use_pore_fill"]
        consts.pore_fill_alg = self.parameters_dict["pore_fill_algorithm"]
        consts.pore_fill_weights[0] = self.parameters_dict["pore_fill_weight_a1"]
        consts.pore_fill_weights[1] = self.parameters_dict["pore_fill_weight_a2"]
        consts.pore_fill_weights[2] = self.parameters_dict["pore_fill_weight_a3"]
        consts.pore_fill_weights[3] = self.parameters_dict["pore_fill_weight_a4"]
        consts.pore_fill_weights[4] = self.parameters_dict["pore_fill_weight_a5"]
        consts.pore_fill_weights[5] = self.parameters_dict["pore_fill_weight_a6"]

    @property
    def pc_initial(self):
        return self.parameters_dict["pc_initial"]

    @property
    def pc_maximum(self):
        return self.parameters_dict["pc_maximum"]

    @property
    def pc_minimum(self):
        return self.parameters_dict["pc_minimum"]

    @property
    def pc_step(self):
        return self.parameters_dict["pc_step"]

    @property
    def pc_step_min(self):
        return self.parameters_dict["pc_step_min"]

    @property
    def pc_step_max(self):
        return self.parameters_dict["pc_step_max"]

    @property
    def drainage_sw_step(self):
        return self.parameters_dict["drainage_sw_step"]

    @property
    def imbibition_sw_step(self):
        return self.parameters_dict["imbibition_sw_step"]

    @property
    def end_at_sor(self):
        return self.parameters_dict["end_at_sor"]

    @property
    def control_by_pc(self):
        return self.parameters_dict["control_by_pc"]

    @property
    def skip_imbibition(self):
        return self.parameters_dict["skip_imbibition"]

    @property
    def skip_2nd_drainage(self):
        return self.parameters_dict["skip_2nd_drainage"]

    @property
    def batch_invasions(self):
        return self.parameters_dict["batch_invasions"]

    @property
    def enforced_swi_1(self):
        return self.parameters_dict["enforced_swi_1"]

    @property
    def enforced_swi_2(self):
        return self.parameters_dict["enforced_swi_2"]

    def generate_contact_angles(self, network):
        self._update_cas(network, "initial")
        self._update_cas(network, "equilibrium")

    def update_initial_cas(self, network):
        network.pn_core.pore.advancing_contact_angle[:] = network.pn_core.pore.initial_advancing_ca
        network.pn_core.pore.receding_contact_angle[:] = network.pn_core.pore.initial_receding_ca
        network.pn_core.throat.advancing_contact_angle[:] = network.pn_core.throat.initial_advancing_ca
        network.pn_core.throat.receding_contact_angle[:] = network.pn_core.throat.initial_receding_ca

    def update_equilibrium_cas(self, network):
        network.pn_core.pore.advancing_contact_angle[
            network.pn_core.pore.changed_wettability
        ] = network.pn_core.pore.equilibrium_advancing_ca[network.pn_core.pore.changed_wettability]
        network.pn_core.pore.receding_contact_angle[network.pn_core.pore.changed_wettability] = np.minimum(
            network.pn_core.pore.receding_contact_angle, network.pn_core.pore.equilibrium_receding_ca
        )[network.pn_core.pore.changed_wettability]
        network.pn_core.throat.advancing_contact_angle[
            network.pn_core.throat.changed_wettability
        ] = network.pn_core.throat.equilibrium_advancing_ca[network.pn_core.throat.changed_wettability]
        network.pn_core.throat.receding_contact_angle[network.pn_core.throat.changed_wettability] = np.minimum(
            network.pn_core.throat.receding_contact_angle, network.pn_core.throat.equilibrium_receding_ca
        )[network.pn_core.throat.changed_wettability]

    def update_2nd_drainage_ca(self, network):
        network.pn_core.pore.receding_contact_angle[
            network.pn_core.pore.changed_wettability
        ] = network.pn_core.pore.equilibrium_receding_ca[network.pn_core.pore.changed_wettability]
        network.pn_core.throat.receding_contact_angle[
            network.pn_core.throat.changed_wettability
        ] = network.pn_core.throat.equilibrium_receding_ca[network.pn_core.throat.changed_wettability]

    def _update_cas(self, network, prefix):
        advancing_ca_array, receding_ca_array = self._generate_cas(
            prefix, len(network.pn_core.pore.radius), network.pn_core.pore.radius
        )
        network.pn_core[f"pore.{prefix}_advancing_ca"] = advancing_ca_array
        network.pn_core[f"pore.{prefix}_receding_ca"] = receding_ca_array

        advancing_ca_array, receding_ca_array = self._generate_cas(
            prefix, len(network.pn_core.throat.radius), network.pn_core.throat.radius
        )
        network.pn_core[f"throat.{prefix}_advancing_ca"] = advancing_ca_array
        network.pn_core[f"throat.{prefix}_receding_ca"] = receding_ca_array

    def _generate_cas(self, prefix, size, radius_array):
        intrinsic_ca_array = self._generate_intrinsic_cas(prefix, size, radius_array)
        model_class = self.parameters_dict[f"{prefix}_ca_model"]
        separation_angle = math.radians(self.parameters_dict[f"{prefix}_ca_separation"])
        advancing_ca_array, receding_ca_array = self._calculate_cas(
            intrinsic_ca_array,
            model_class,
            separation_angle,
        )
        # ParameterManager.show_histogram(np.degrees(intrinsic_ca_array))
        # ParameterManager.show_points(np.degrees(intrinsic_ca_array), np.degrees(receding_ca_array), np.degrees(advancing_ca_array))
        return advancing_ca_array, receding_ca_array

    def _generate_intrinsic_cas(self, prefix, size, radius_array):
        if prefix == "equilibrium":
            fraction = self.parameters_dict[f"second_ca_fraction"]
            if fraction == 0:
                intrisic_ca_array = self._generate_fraction_cas("equilibrium", size, radius_array)
            else:
                fraction_size = round(size * fraction)
                indexes = list(range(size))
                random.shuffle(indexes)
                first_dist_indexes = indexes[fraction_size:]
                second_dist_indexes = indexes[0:fraction_size]
                intrisic_ca_array = np.empty(size, dtype=np.float64)
                intrisic_ca_array[first_dist_indexes] = self._generate_fraction_cas(
                    "equilibrium", len(first_dist_indexes), radius_array[first_dist_indexes]
                )
                intrisic_ca_array[second_dist_indexes] = self._generate_fraction_cas(
                    "second", len(second_dist_indexes), radius_array[second_dist_indexes]
                )
        else:
            intrisic_ca_array = self._generate_fraction_cas(prefix, size, radius_array)
        return intrisic_ca_array

    def _generate_fraction_cas(self, prefix, size, radius_array):
        center = math.radians(self.parameters_dict[f"{prefix}_ca_center"])
        range = math.radians(self.parameters_dict[f"{prefix}_ca_range"])
        gamma = self.parameters_dict[f"{prefix}_ca_gamma"]
        delta = self.parameters_dict[f"{prefix}_ca_delta"]
        sigma = self.parameters_dict[f"{prefix}_ca_sigma"]
        intrinsic_ca_array = self._generate_ca_distribution(
            delta,
            gamma,
            sigma,
            center,
            range,
            size,
            radius_array,
            self.parameters_dict[f"{prefix}_ca_correlation"],
            self.parameters_dict[f"{prefix}_ca_distribution"],
        )
        return intrinsic_ca_array

    @staticmethod
    def _generate_ca_distribution(delta, gamma, sigma, center, range, size, radius_array, correlation, distribution):
        if distribution == "weibull":
            reference_ca_array = weibull(size, delta, gamma, center, range)
        elif distribution == "gaussian":
            reference_ca_array = gaussian(size, sigma, center, range)
        reference_ca_array[reference_ca_array < 0] = 0
        reference_ca_array[reference_ca_array > pi] = pi

        contact_angles = np.empty(size, dtype=np.float64)
        if correlation == UNCORRELATED:
            contact_angles = reference_ca_array
        elif correlation == POSITIVE_RADIUS:
            reference_ca_array.sort()
            pore_sorted_index_array = radius_array.argsort()
            contact_angles[pore_sorted_index_array] = reference_ca_array
        else:
            reference_ca_array.sort()
            pore_sorted_index_array = radius_array.argsort()
            contact_angles[pore_sorted_index_array] = reference_ca_array[::-1]

        return contact_angles

    @staticmethod
    @njit(cache=True)
    def _calculate_cas(intrinsic_ca_array, model_class, separation_angle):
        if model_class == 1:
            advancing_ca_array = intrinsic_ca_array
            receding_ca_array = intrinsic_ca_array
        elif model_class == 2:
            growth = (pi + separation_angle) / pi
            receding_ca_array = np.maximum(0.0, growth * intrinsic_ca_array - separation_angle)
            advancing_ca_array = np.minimum(pi, growth * intrinsic_ca_array)
        else:
            receding_ca_array = np.empty(len(intrinsic_ca_array), dtype=np.float64)
            mask = intrinsic_ca_array < 0.38349
            receding_ca_array[mask] = 0.0
            mask = (0.38349 < intrinsic_ca_array) & (intrinsic_ca_array < 1.5289)
            receding_ca_array[mask] = (0.5 * np.exp(0.05 * intrinsic_ca_array[mask] * 180.0 / pi) - 1.5) * pi / 180.0
            mask = (1.5289 < intrinsic_ca_array) & (intrinsic_ca_array < 2.7646)
            receding_ca_array[mask] = 2.0 * (intrinsic_ca_array[mask] - 1.19680)
            mask = 2.7646 < intrinsic_ca_array
            receding_ca_array[mask] = pi

            advancing_ca_array = np.empty(len(intrinsic_ca_array), dtype=np.float64)
            mask = intrinsic_ca_array < 0.38349
            advancing_ca_array[mask] = 0.0
            mask = (0.38349 < intrinsic_ca_array) & (intrinsic_ca_array < 1.61268)
            advancing_ca_array[mask] = 2.0 * (intrinsic_ca_array[mask] - 0.38349)
            mask = (1.61268 < intrinsic_ca_array) & (intrinsic_ca_array < 2.75805)
            advancing_ca_array[mask] = (
                (181.5 - 4051.0 * np.exp(-0.05 * intrinsic_ca_array[mask] * 180.0 / pi)) * pi / 180.0
            )
            mask = 2.75805 < intrinsic_ca_array
            advancing_ca_array[mask] = pi

        return advancing_ca_array, receding_ca_array

    @staticmethod
    def show_points(intrinsic, receding, advancing):
        from PySide2 import QtWidgets, QtGui
        import pyqtgraph as pg

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget()
        plot_item = win.addPlot(left="Adv/Rec Cas", bottom="Intrinsic CAs")

        curve_item0 = pg.ScatterPlotItem(
            intrinsic, advancing, stepMode=False, pen=pg.mkPen(QtGui.QColor(0, 255, 255), width=2)
        )
        curve_item1 = pg.ScatterPlotItem(
            intrinsic, receding, stepMode=False, pen=pg.mkPen(QtGui.QColor(255, 0, 255), width=2)
        )
        plot_item.addItem(curve_item0)
        plot_item.addItem(curve_item1)

        win.show()
        app.exec_()
        exit()

    @staticmethod
    def show_histogram(data):
        from PySide2 import QtWidgets, QtGui
        import pyqtgraph as pg

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget()
        plot_item = win.addPlot(left="quant", bottom="angle")

        hist, edges = np.histogram(data, bins=90)
        curve_item0 = pg.PlotCurveItem(edges, hist, stepMode="center", pen=pg.mkPen(QtGui.QColor(0, 255, 255), width=2))
        # hist, edges = np.histogram(data2, bins=50)
        # curve_item1 = pg.PlotCurveItem(edges, hist, stepMode="center", pen=pg.mkPen(QtGui.QColor(255, 0, 255), width=2))
        plot_item.addItem(curve_item0)
        # plot_item.addItem(curve_item1)

        win.show()
        app.exec_()
        exit()
