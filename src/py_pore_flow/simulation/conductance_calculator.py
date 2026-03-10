import numpy as np
import traceback
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
    MAX_NUMBER_OF_CORNERS,
)
from py_pore_flow.equations.conductance import (
    calculate_network_conductance,
    calculate_network_multiphase_conductivity,
    calculate_network_singlephase_conductivity,
)
from py_pore_flow.network.network import PORE, THROAT


class ConductanceCalculator:
    def __init__(self, network, log_conductivity_file=None):
        self.network = network
        self.log_conductivity_file = log_conductivity_file if log_conductivity_file is not None else getattr(network, "log_conductivity_file", None)
        if self.log_conductivity_file:
            # Only write header if the file is empty or doesn't exist
            import os
            if not os.path.exists(self.log_conductivity_file) or os.path.getsize(self.log_conductivity_file) == 0:
                with open(self.log_conductivity_file, "w") as f:
                    header = [
                        "element_type", "element_id", "phase", "conductivity_type",
                        "A", "G", "shape", "Sw", "So", "b", "bo",
                        "theta_h", "beta", "n_corners",
                        "has_water_in_center", "has_oil_in_center", "conductivity"
                    ]
                    f.write("\t".join(header) + "\n")

    def _log_conductivity(self, element_type, phase, cond_type, gp, A, G, shape, Sw=None, So=None, b=None, bo=None, theta_h=None, beta=None, n_corners=None, has_water_in_center=None, has_oil_in_center=None):
        if not self.log_conductivity_file:
            return
        
        print("--- Capturing traceback as a string ---")
        traceback.print_stack()

        with open(self.log_conductivity_file, "a") as f:
            for n in range(len(gp)):
                def format_array(arr):
                    if arr is None: return "0.0"
                    if isinstance(arr[n], (np.ndarray, list)):
                        return "[" + ",".join([f"{x:.6e}" if isinstance(x, (float, np.float64)) else str(x) for x in arr[n]]) + "]"
                    return f"{arr[n]:.6e}"

                row = [
                    "PORE" if element_type == PORE else "THROAT",
                    str(n),
                    "WATER" if phase == WATER else "OIL",
                    cond_type,
                    f"{A[n]:.6e}",
                    f"{G[n]:.6e}",
                    str(shape[n]),
                    f"{Sw[n]:.6e}" if Sw is not None else "0.0",
                    f"{So[n]:.6e}" if So is not None else "0.0",
                    format_array(b),
                    format_array(bo),
                    format_array(theta_h),
                    format_array(beta),
                    str(n_corners[n]) if n_corners is not None else "0",
                    str(has_water_in_center[n]) if has_water_in_center is not None else "False",
                    str(has_oil_in_center[n]) if has_oil_in_center is not None else "False",
                    f"{gp[n]:.6e}"
                ]
                f.write("\t".join(row) + "\n")

    def calculate_singlephase_conductance(self, phase):
        self._calculate_singlephase_conductivity(phase)
        self._calculate_singlephase_conductance(phase)

    def calculate_multiphase_conductance(self, phase):
        self._calculate_multiphase_conductivity(phase)
        self._calculate_multiphase_conductance(phase)

    def _calculate_singlephase_conductance(self, phase):
        if phase == WATER:
            calculate_network_conductance(
                self.network.pn_core.throat.water_singlephase_conductance,
                self.network.pn_core.throat.water_singlephase_conductivity,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.conns,
                self.network.pn_core.pore.water_singlephase_conductivity,
                self.network.pn_core.pore.radius,
            )
        else:
            calculate_network_conductance(
                self.network.pn_core.throat.oil_singlephase_conductance,
                self.network.pn_core.throat.oil_singlephase_conductivity,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.conns,
                self.network.pn_core.pore.oil_singlephase_conductivity,
                self.network.pn_core.pore.radius,
            )

    def _calculate_multiphase_conductance(self, phase):
        if phase == WATER:
            calculate_network_conductance(
                self.network.pn_core.throat.water_multiphase_conductance,
                self.network.pn_core.throat.water_multiphase_conductivity,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.conns,
                self.network.pn_core.pore.water_multiphase_conductivity,
                self.network.pn_core.pore.radius,
            )
        else:
            calculate_network_conductance(
                self.network.pn_core.throat.oil_multiphase_conductance,
                self.network.pn_core.throat.oil_multiphase_conductivity,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.conns,
                self.network.pn_core.pore.oil_multiphase_conductivity,
                self.network.pn_core.pore.radius,
            )

    def _calculate_singlephase_conductivity(self, phase):
        for element_type in [PORE, THROAT]:
            if element_type == PORE:
                pn_element = self.network.pn_core.pore
                area = self.network.pn_core.pore.area
            else:  # element_type == THROAT:
                pn_element = self.network.pn_core.throat
                area = self.network.pn_core.throat.cross_sectional_area

            if phase == WATER:
                singlephase_conductivity = pn_element.water_singlephase_conductivity
            else:  # phase == OIL:
                singlephase_conductivity = pn_element.oil_singlephase_conductivity

            calculate_network_singlephase_conductivity(
                singlephase_conductivity,
                area,
                pn_element.shape_factor,
                pn_element.shape,
                phase,
            )
            
            if self.log_conductivity_file:
                self._log_conductivity(
                    element_type, phase, "SINGLEPHASE", 
                    singlephase_conductivity, area, pn_element.shape_factor, pn_element.shape
                )

            singlephase_conductivity *= pn_element.N

    def _calculate_multiphase_conductivity(self, phase):
        for element_type in [PORE, THROAT]:
            if element_type == PORE:
                pn_element = self.network.pn_core.pore
                area = self.network.pn_core.pore.area
            else:  # element_type == THROAT:
                pn_element = self.network.pn_core.throat
                area = self.network.pn_core.throat.cross_sectional_area

            if phase == WATER:
                multiphase_conductivity = pn_element.water_multiphase_conductivity
            else:  # phase == OIL:
                multiphase_conductivity = pn_element.oil_multiphase_conductivity

            calculate_network_multiphase_conductivity(
                multiphase_conductivity,
                area,
                pn_element.shape_factor,
                pn_element.shape,
                pn_element.water_saturation,
                pn_element.oil_saturation,
                pn_element.b,
                pn_element.bo,
                pn_element.corner_half_angle,
                pn_element.hinging_contact_angle,
                np.tile(pn_element.advancing_contact_angle[:, np.newaxis], MAX_NUMBER_OF_CORNERS),
                pn_element.corner_oil_layer,
                pn_element.n_corners,
                pn_element.water_filled,
                pn_element.oil_filled,
                phase,
            )
            
            if self.log_conductivity_file:
                self._log_conductivity(
                    element_type, phase, "MULTIPHASE",
                    multiphase_conductivity, area, pn_element.shape_factor, pn_element.shape,
                    Sw=pn_element.water_saturation,
                    So=pn_element.oil_saturation,
                    b=pn_element.b,
                    bo=pn_element.bo,
                    theta_h=pn_element.hinging_contact_angle,
                    beta=pn_element.corner_half_angle,
                    n_corners=pn_element.n_corners,
                    has_water_in_center=pn_element.water_filled,
                    has_oil_in_center=pn_element.oil_filled
                )

            multiphase_conductivity *= pn_element.N
