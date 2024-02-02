from abc import abstractmethod
from math import pi

import numba as nb
from numba.experimental import jitclass
import numpy as np
import openpnm

from equations import (
    SQUARE,
    TRIANGLE,
    WATER,
)


def initialize_pn(pn):
    for element_type in ["pore", "throat"]:
        pn[f"{element_type}.oil_filled"] = False
        pn[f"{element_type}.water_filled"] = True
        pn[f"{element_type}.oil_saturation"] = 0.0
        pn[f"{element_type}.water_saturation"] = 1.0
        pn[f"{element_type}.corner_half_angle"] = np.empty((len(pn[f"{element_type}.all"]), 4), np.float64)
        pn[f"{element_type}.hinging_contact_angle"] = np.empty((len(pn[f"{element_type}.all"]), 4), np.float64)
        pn[f"{element_type}.b"] = np.empty((len(pn[f"{element_type}.all"]), 4), np.float64)
        pn[f"{element_type}.C_r"] = 0.0
        pn[f"{element_type}.C_a"] = 0.0
        pn[f"{element_type}.advancing_contact_angle"] = 0.0
        pn[f"{element_type}.receding_contact_angle"] = 0.0
        pn[f"{element_type}.oil_singlephase_conductance"] = 0.0
        pn[f"{element_type}.water_singlephase_conductance"] = 0.0
        pn[f"{element_type}.shape"] = 0
        pn[f"{element_type}.trapped"] = False


class Network:
    def __init__(self, network):
        self.pn = network

        number_of_pores = len(self.pn.pores())
        number_of_throats = max(self.pn.throats()) + 1

        self.pore_list = [None] * number_of_pores
        self.throat_list = [None] * number_of_throats
        pore_core_list = [None] * number_of_pores
        throat_core_list = [None] * number_of_throats

        for pore_id in self.pn.pores():
            self.pore_list[pore_id] = Pore(self.pn, pore_id)
            pore_core_list[pore_id] = ElementCore(pore_id)
        self.pore_core_list = nb.typed.List(pore_core_list)

        for throat_id in self.pn.throats():
            self.throat_list[throat_id] = Throat(self.pn, throat_id)
            throat_core_list[throat_id] = ElementCore(throat_id)
        self.throat_core_list = nb.typed.List(throat_core_list)

    def get_pore(self, id):
        return self.pore_list[id]

    def get_throat(self, id):
        return self.throat_list[id]

    def get_pore_list(self):
        return self.pore_list

    def get_throat_list(self):
        return self.throat_list

    def get_pore_core_list(self):
        return self.pore_core_list

    def get_throat_core_list(self):
        return self.throat_core_list

    def get_inlets_ids(self):
        return list(np.where(self.pn["pore.inlets"] == True)[0])

    def get_outlets_ids(self):
        return list(np.where(self.pn["pore.outlets"] == True)[0])

    def get_pores_ids(self):
        return list(self.pn.pores())

    def get_throats_ids(self):
        return list(self.pn.throats())

    def get_number_of_pores(self):
        return len(self.pn.pores())

    def get_number_of_throats(self):
        return len(self.pn.pores())

    def get_neighbor_pores(self, throat_id):
        neighbor_pores_list = []
        for pore_id in self.pn.find_connected_pores(throat_id)[0]:
            neighbor_pores_list.append(self.pore_list[pore_id])
        return neighbor_pores_list

    def get_neighbor_throats(self, pore_id):
        neighbor_throats_list = []
        for throat_id in self.pn.find_neighbor_throats(pore_id):
            neighbor_throats_list.append(self.throat_list[throat_id])
        return neighbor_throats_list

    def get_neighbors(self, element):
        if element.is_pore():
            return self.get_neighbor_throats(element.get_id())
        else:
            return self.get_neighbor_pores(element.get_id())

    def reset_property(self, property_name):
        for pore in self.pore_list:
            setattr(pore, property_name, None)
        for throat in self.throat_list:
            setattr(throat, property_name, None)

    def update_core_elements(self):
        for pore_id in self.pn.pores():
            element = self.pore_list[pore_id]
            core = self.pore_core_list[pore_id]
            self.__update_core(core, element)

        for throat_id in self.pn.throats():
            element = self.throat_list[throat_id]
            core = self.throat_core_list[throat_id]
            self.__update_core(core, element)

    @staticmethod
    def __update_core(core, element):
        core.G = element.G
        core.r = element.r
        core.A = element.A
        core.V = element.V
        core.Vclay = element.Vclay
        core.Sw = element.Sw
        core.So = element.So
        core.theta_a = element.advancing_contact_angle
        core.theta_r = element.receding_contact_angle
        core.C_r = element.C_r
        core.C_a = element.C_a
        core.shape = element.shape
        core.trapped = element.trapped
        core.beta = element.beta
        core.theta_h = element.theta_h
        core.b = element.b
        core.oil_filled = element.is_oil_filled()


class PnElement:
    def __init__(self, network, type_name, id):
        self.pn = network
        self.type_name = type_name
        self.id = id

        self._G = None
        self._r = None
        self._A = None
        self._V = None
        self._Vclay = None
        self._Sw = None
        self._So = None
        self._advancing_contact_angle = None
        self._receding_contact_angle = None
        self._C_r = None
        self._C_a = None
        self._shape = None
        self._trapped = None
        self._beta = None
        self._theta_h = None
        self._b = None

        self.pc = None
        self.oil_filled = False
        self.water_filled = True

    @property
    def G(self):
        if self._G is None:
            self._G = self.pn[f"{self.type_name}.shape_factor"][self.id]
        return self._G

    @property
    def r(self):
        if self._r is None:
            self._r = self.pn[f"{self.type_name}.radius"][self.id]
        return self._r

    @property
    def V(self):
        if self._V is None:
            self._V = self.pn[f"{self.type_name}.volume"][self.id]
        return self._V

    @property
    def Vclay(self):
        if self._Vclay is None:
            self._Vclay = self.pn[f"{self.type_name}.clay_volume"][self.id]
        return self._Vclay

    @property
    def Sw(self):
        if self._Sw is None:
            self._Sw = self.pn[f"{self.type_name}.water_saturation"][self.id]
        return self._Sw

    @property
    def So(self):
        if self._So is None:
            self._So = self.pn[f"{self.type_name}.oil_saturation"][self.id]
        return self._So

    @property
    def advancing_contact_angle(self):
        if self._advancing_contact_angle is None:
            self._advancing_contact_angle = self.pn[f"{self.type_name}.advancing_contact_angle"][self.id]
        return self._advancing_contact_angle

    @property
    def receding_contact_angle(self):
        if self._receding_contact_angle is None:
            self._receding_contact_angle = self.pn[f"{self.type_name}.receding_contact_angle"][self.id]
        return self._receding_contact_angle

    @property
    def C_r(self):
        if self._C_r is None:
            self._C_r = self.pn[f"{self.type_name}.C_r"][self.id]
        return self._C_r

    @property
    def C_a(self):
        if self._C_a is None:
            self._C_a = self.pn[f"{self.type_name}.C_a"][self.id]
        return self._C_a

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.pn[f"{self.type_name}.shape"][self.id]
        return self._shape

    @property
    def trapped(self):
        if self._trapped is None:
            self._trapped = self.pn[f"{self.type_name}.trapped"][self.id]
        return self._trapped

    @property
    def beta(self):
        if self._beta is None:
            if self.shape == TRIANGLE:
                n = 3
            elif self.shape == SQUARE:
                n = 4
            else:
                n = 0
            self._beta = np.resize(self.pn[f"{self.type_name}.corner_half_angle"][self.id], n)
        return self._beta

    @property
    def theta_h(self):
        if self._theta_h is None:
            if self.shape == TRIANGLE:
                n = 3
            elif self.shape == SQUARE:
                n = 4
            else:
                n = 0
            self._theta_h = np.resize(self.pn[f"{self.type_name}.hinging_contact_angle"][self.id], n)
        return self._theta_h

    @property
    def b(self):
        if self._b is None:
            if self.shape == TRIANGLE:
                n = 3
            elif self.shape == SQUARE:
                n = 4
            else:
                n = 0
            self._b = np.resize(self.pn[f"{self.type_name}.b"][self.id], n)
        return self._b

    @Sw.setter
    def Sw(self, value):
        self._Sw = value
        self.pn[f"{self.type_name}.water_saturation"][self.id] = value

    @So.setter
    def So(self, value):
        self._So = value
        self.pn[f"{self.type_name}.oil_saturation"][self.id] = value

    @advancing_contact_angle.setter
    def advancing_contact_angle(self, value):
        self._advancing_contact_angle = value
        self.pn[f"{self.type_name}.advancing_contact_angle"][self.id] = value

    @receding_contact_angle.setter
    def receding_contact_angle(self, value):
        self._receding_contact_angle = value
        self.pn[f"{self.type_name}.receding_contact_angle"][self.id] = value

    @C_r.setter
    def C_r(self, C_r):
        self._C_r = C_r
        self.pn[f"{self.type_name}.C_r"][self.id] = C_r

    @C_a.setter
    def C_a(self, C_a):
        self._C_a = C_a
        self.pn[f"{self.type_name}.C_a"][self.id] = C_a

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.pn[f"{self.type_name}.shape"][self.id] = shape

    @trapped.setter
    def trapped(self, trapped):
        self._trapped = trapped
        self.pn[f"{self.type_name}.trapped"][self.id] = trapped

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self.pn[f"{self.type_name}.corner_half_angle"][self.id] = np.resize(beta, 4)

    @theta_h.setter
    def theta_h(self, theta_h):
        self._theta_h = theta_h
        self.pn[f"{self.type_name}.hinging_contact_angle"][self.id] = np.resize(theta_h, 4)

    @b.setter
    def b(self, b):
        self._b = b
        self.pn[f"{self.type_name}.b"][self.id] = np.resize(b, 4)

    def get_id(self):
        return self.id

    def get_pc(self):
        return self.pc

    def set_oil_filled(self, filled: bool):
        self.oil_filled = filled
        self.water_filled = not filled

    def set_water_filled(self, filled: bool):
        self.water_filled = filled
        self.oil_filled = not filled

    def set_filled(self, phase, filled: bool):
        if phase == WATER:
            self.water_filled = filled
            self.oil_filled = not filled
        else:
            self.oil_filled = filled
            self.water_filled = not filled

    def is_oil_filled(self) -> bool:
        return self.oil_filled

    def is_water_filled(self) -> bool:
        return self.water_filled

    def is_filled(self, phase):
        if phase == WATER:
            return self.water_filled
        else:
            return self.oil_filled

    def set_pc(self, pc):
        self.pc = pc

    @abstractmethod
    def is_pore(self):
        pass

    @abstractmethod
    def is_throat(self):
        pass


class Pore(PnElement):
    def __init__(self, network, id):
        super().__init__(network, "pore", id)

    @property
    def A(self):
        if self._A is None:
            self._A = self.pn[f"{self.type_name}.area"][self.id]
        return self._A

    def is_pore(self):
        return True

    def is_throat(self):
        return False


class Throat(PnElement):
    def __init__(self, network, id):
        super().__init__(network, "throat", id)
        self._L = None

    @property
    def A(self):
        if self._A is None:
            self._A = self.pn[f"{self.type_name}.cross_sectional_area"][self.id]
        return self._A

    @property
    def L(self):
        if self._L is None:
            self._L = self.pn[f"{self.type_name}.length"][self.id]
        return self._L

    def is_pore(self):
        return False

    def is_throat(self):
        return True


spec = [
    ("id", nb.types.int32),
    ("G", nb.types.float64),
    ("r", nb.types.float64),
    ("A", nb.types.float64),
    ("V", nb.types.float64),
    ("Vclay", nb.types.float64),
    ("Sw", nb.types.float64),
    ("So", nb.types.float64),
    ("theta_a", nb.types.float64),
    ("theta_r", nb.types.float64),
    ("C_r", nb.types.float64),
    ("C_a", nb.types.float64),
    ("shape", nb.types.int32),
    ("trapped", nb.types.boolean),
    ("beta", nb.types.float64[:]),
    ("theta_h", nb.types.float64[:]),
    ("b", nb.types.float64[:]),
    ("oil_filled", nb.types.boolean),
    ("n_corners", nb.types.int32),
]


@jitclass(spec)
class ElementCore:
    def __init__(self, id):
        self.id = id
        self.G = 0.0
        self.r = 0.0
        self.A = 0.0
        self.V = 0.0
        self.Vclay = 0.0
        self.Sw = 0.0
        self.So = 0.0
        self.theta_a = 0.0
        self.theta_r = 0.0
        self.C_r = 0.0
        self.C_a = 0.0
        self.shape = 0
        self.trapped = False
        self.beta = np.empty(4, dtype=np.float64)
        self.theta_h = np.empty(4, dtype=np.float64)
        self.b = np.empty(4, dtype=np.float64)
        self.oil_filled = True
        self.n_corners = 0
