from abc import abstractmethod
import numpy as np
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
    calculate_sw_total,
)
from py_pore_flow.network.network import PORE


class SaturationCalculator:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase

        self.total_volume = (self.network.pn_core.pore.volume * self.network.pn_core.pore.N).sum() + (
            self.network.pn_core.throat.volume * self.network.pn_core.throat.N
        ).sum()

    def update_saturations(self, pc, verbose=False):
        pore_sw, pore_so = self._get_pores_saturations(pc, verbose=verbose)
        if self.phase == WATER:
            update_mask = pore_sw > self.network.pn_core.pore.water_saturation
        else:
            update_mask = pore_so > self.network.pn_core.pore.oil_saturation
        self.network.pn_core.pore.water_saturation[update_mask] = pore_sw[update_mask]
        self.network.pn_core.pore.oil_saturation[update_mask] = pore_so[update_mask]

        throat_sw, throat_so = self._get_throats_saturations(pc, verbose=verbose)
        if self.phase == WATER:
            update_mask = throat_sw > self.network.pn_core.throat.water_saturation
        else:
            update_mask = throat_so > self.network.pn_core.throat.oil_saturation
        self.network.pn_core.throat.water_saturation[update_mask] = throat_sw[update_mask]
        self.network.pn_core.throat.oil_saturation[update_mask] = throat_so[update_mask]

    @abstractmethod
    def _get_pores_saturations(self, pc, verbose=False):
        pass

    @abstractmethod
    def _get_throats_saturations(self, pc, verbose=False):
        pass

    @abstractmethod
    def _get_element_saturations(self, element_type, id, pc, verbose=False):
        pass

    def update_element_saturations(self, element_type, id, pc, verbose=False):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
        else:
            element_data = self.network.pn_core.throat

        sw, so = self._get_element_saturations(element_type, id, pc, verbose=verbose)
        if self.phase == WATER:
            should_update = sw > element_data.water_saturation[id]
        else:
            should_update = so > element_data.oil_saturation[id]

        if not should_update:
            # error
            return 0

        sw_diff = sw - element_data.water_saturation[id]
        so_diff = so - element_data.oil_saturation[id]
        element_data.water_saturation[id] = sw
        element_data.oil_saturation[id] = so
        if sw > 0 and so > 0:
            element_data.is_multiphase[id] = True
        else:
            element_data.is_multiphase[id] = False

        if self.phase == OIL:
            return so_diff * (element_data.volume[id] * element_data.N[id]) / self.total_volume
        else:
            return sw_diff * (element_data.volume[id] * element_data.N[id]) / self.total_volume

    def calculate_sw_total(self):
        return calculate_sw_total(
            np.concatenate(
                (
                    self.network.pn_core.throat.water_saturation,
                    self.network.pn_core.pore.water_saturation,
                )
            ),
            np.concatenate(
                (
                    self.network.pn_core.throat.volume,
                    self.network.pn_core.pore.volume,
                )
            ),
            np.concatenate(
                (
                    self.network.pn_core.throat.N,
                    self.network.pn_core.pore.N,
                )
            ),
            self.total_volume,
        )
