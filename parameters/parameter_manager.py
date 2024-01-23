import math
from math import exp, pi

import numpy as np
from scipy.stats import weibull_min

import constants


UNCORRELATED = 0
POSITIVE_RADIUS = 1
NEGATIVE_RADIUS = 2


DEFAULT_PARAMETERS = {
    "water_viscosity": 0.00105,
    "water_density": 1000,
    "oil_viscosity": 0.00139,
    "oil_density": 900,
    "interfacial_tension": 0.03,
    "initial_ca_center": 30,
    "initial_ca_range": 15,
    "initial_ca_k": 3.5,
    "initial_ca_correlation": UNCORRELATED,
    "initial_ca_model": 3,
    "initial_ca_separation": 25.2,
    "equilibrium_ca_center": 30,
    "equilibrium_ca_range": 15,
    "equilibrium_ca_k": 3.5,
    "equilibrium_ca_correlation": UNCORRELATED,
    "equilibrium_ca_model": 3,
    "equilibrium_ca_separation": 25.2,
    "pc_maximum": 40000,
    "pc_minimum": 0,
    "pc_step": 200,
    "pc_step_min": 1,
    "pc_step_max": 1000,
    "sw_step": 0.01,
}


class ParameterManager:
    def __init__(self, parameters_dict=DEFAULT_PARAMETERS):
        self.parameters_dict = parameters_dict

        constants.pw = self.parameters_dict["water_density"]
        constants.po = self.parameters_dict["oil_density"]
        constants.sigma = self.parameters_dict["interfacial_tension"]
        constants.miw = self.parameters_dict["water_viscosity"]
        constants.mio = self.parameters_dict["oil_viscosity"]

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
    def sw_step(self):
        return self.parameters_dict["sw_step"]

    def update_initial_cas(self, network):
        self.__update_cas(network, "initial")

    def update_equilibrium_cas(self, network):
        self.__update_cas(network, "equilibrium")

    def __update_cas(self, network, prefix):
        network.reset_property("advancing_contact_angle")
        network.reset_property("receding_contact_angle")

        advancing_ca_array, receding_ca_array = self.__generate_cas(
            prefix, len(network.pn["pore.radius"]), network.pn["pore.radius"]
        )
        network.pn["pore.advancing_contact_angle"] = advancing_ca_array
        network.pn["pore.receding_contact_angle"] = receding_ca_array

        advancing_ca_array, receding_ca_array = self.__generate_cas(
            prefix, len(network.pn["throat.radius"]), network.pn["throat.radius"]
        )
        network.pn["throat.advancing_contact_angle"] = advancing_ca_array
        network.pn["throat.receding_contact_angle"] = receding_ca_array

    def __generate_cas(self, prefix, size, radius_array):
        center = math.radians(self.parameters_dict[f"{prefix}_ca_center"])
        range = math.radians(self.parameters_dict[f"{prefix}_ca_range"])
        k = self.parameters_dict[f"{prefix}_ca_k"]
        intrinsic_ca_array = self.__generate_intrinsic_cas(
            center,
            range,
            k,
            size,
            radius_array,
            self.parameters_dict[f"{prefix}_ca_correlation"],
        )
        model_class = self.parameters_dict[f"{prefix}_ca_model"]
        separation_angle = math.radians(self.parameters_dict[f"{prefix}_ca_separation"])
        advancing_ca_array, receding_ca_array = self.__calculate_cas(
            intrinsic_ca_array,
            model_class,
            separation_angle,
        )
        return advancing_ca_array, receding_ca_array

    @staticmethod
    def __generate_intrinsic_cas(center, range, k, size, radius_array, correlation):
        loc = center - range / 2
        scale = center - loc
        reference_ca_array = weibull_min.rvs(k, loc=loc, scale=scale, size=size)
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
    def __calculate_cas(intrinsic_ca_array, model_class, separation_angle):
        if model_class == 1:
            advancing_ca_array = intrinsic_ca_array
            receding_ca_array = intrinsic_ca_array
        elif model_class == 2:
            growth = (pi + separation_angle) / pi
            receding_ca_array = np.maximum(0.0, growth * intrinsic_ca_array - separation_angle)
            advancing_ca_array = np.minimum(pi, growth * intrinsic_ca_array)
        else:
            receding_ca_array = np.empty(len(intrinsic_ca_array), dtype=np.float64)
            receding_ca_array[intrinsic_ca_array < 0.38349] = 0.0
            receding_ca_array[(0.38349 < intrinsic_ca_array) & (intrinsic_ca_array < 1.5289)] = (
                (
                    0.5
                    * np.exp(
                        0.05
                        * intrinsic_ca_array[(0.38349 < intrinsic_ca_array) & (intrinsic_ca_array < 1.5289)]
                        * 180.0
                        / pi
                    )
                    - 1.5
                )
                * pi
                / 180.0
            )
            receding_ca_array[(1.5289 < intrinsic_ca_array) & (intrinsic_ca_array < 2.7646)] = 2.0 * (
                intrinsic_ca_array[(1.5289 < intrinsic_ca_array) & (intrinsic_ca_array < 2.7646)] - 1.19680
            )
            receding_ca_array[2.7646 < intrinsic_ca_array] = pi

            advancing_ca_array = np.empty(len(intrinsic_ca_array), dtype=np.float64)
            advancing_ca_array[intrinsic_ca_array < 0.38349] = 0.0
            advancing_ca_array[(0.38349 < intrinsic_ca_array) & (intrinsic_ca_array < 1.61268)] = 2.0 * (
                intrinsic_ca_array[(0.38349 < intrinsic_ca_array) & (intrinsic_ca_array < 1.61268)] - 0.38349
            )
            advancing_ca_array[(1.61268 < intrinsic_ca_array) & (intrinsic_ca_array < 2.75805)] = (
                (
                    181.5
                    - 4051.0
                    * np.exp(
                        -0.05
                        * intrinsic_ca_array[(1.61268 < intrinsic_ca_array) & (intrinsic_ca_array < 2.75805)]
                        * 180.0
                        / pi
                    )
                )
                * pi
                / 180.0
            )
            advancing_ca_array[2.75805 < intrinsic_ca_array] = pi

        return advancing_ca_array, receding_ca_array
