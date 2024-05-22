from abc import abstractmethod
from math import pi

import numba as nb
from numba.experimental import jitclass
import numpy as np
import openpnm

from py_pore_flow.equations import (
    MAX_NUMBER_OF_CORNERS,
    SQUARE,
    TRIANGLE,
    WATER,
)


def initialize_pn(pn):
    for element_type in ["pore", "throat"]:
        number_of_elements = len(pn[f"{element_type}.all"])

        pn[f"{element_type}.oil_filled"] = False
        pn[f"{element_type}.water_filled"] = True
        pn[f"{element_type}.oil_saturation"] = 0.0
        pn[f"{element_type}.water_saturation"] = 1.0
        pn[f"{element_type}.corner_half_angle"] = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
        pn[f"{element_type}.hinging_contact_angle"] = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
        pn[f"{element_type}.b"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), 0.0, np.float64)
        pn[f"{element_type}.bo"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), 0.0, np.float64)
        pn[f"{element_type}.corner_oil_layer"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), False, np.bool_)
        pn[f"{element_type}.C_r"] = 0.0
        pn[f"{element_type}.C_a"] = 0.0
        pn[f"{element_type}.C_h"] = 0.0
        pn[f"{element_type}.advancing_contact_angle"] = 0.0
        pn[f"{element_type}.receding_contact_angle"] = 0.0
        pn[f"{element_type}.oil_singlephase_conductance"] = 0.0
        pn[f"{element_type}.water_singlephase_conductance"] = 0.0
        pn[f"{element_type}.shape"] = 0
        pn[f"{element_type}.trapped"] = False
        pn[f"{element_type}.n_corners"] = int(0)
        pn[f"{element_type}.spontaneous_imbibition"] = True

    # Initialize connection properties
    pn["throat.n_conns"] = int(2)
    pn["pore.n_conns"] = int(0)

    number_of_pores = len(pn["pore.all"])

    pore_n_conns_list = [0] * number_of_pores
    pore_conns_list = [None] * number_of_pores
    for pore_id in pn.pores():
        neighbor_throats_list = pn.find_neighbor_throats(pore_id)
        pore_n_conns_list[pore_id] = len(neighbor_throats_list)
        pore_conns_list[pore_id] = neighbor_throats_list

    max_n_conns = max(pore_n_conns_list)
    pn["pore.conns"] = np.empty((number_of_pores, max_n_conns), np.int32)
    for pore_id in pn.pores():
        pn["pore.n_conns"][pore_id] = pore_n_conns_list[pore_id]
        pn["pore.conns"][pore_id] = np.resize(pore_conns_list[pore_id], max_n_conns)


class Network:
    def __init__(self, network):
        self.pn = network

        number_of_pores = len(self.pn[f"pore.all"])
        number_of_throats = len(self.pn[f"throat.all"])

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
        for throat_id in np.resize(self.pn["pore.conns"][pore_id], self.pn["pore.n_conns"][pore_id]):
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
        core.C_h = element.C_h
        core.shape = element.shape
        core.trapped = element.trapped
        core.beta = element.beta
        core.theta_h = element.theta_h
        core.b = element.b
        core.bo = element.bo
        core.corner_oil_layer = element.corner_oil_layer
        core.oil_filled = element.oil_filled
        core.n_corners = element.n_corners
        core.spontaneous_imbibition = element.spontaneous_imbibition


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
        self._advancing_contact_angle = None
        self._receding_contact_angle = None
        self._C_r = None
        self._C_a = None
        self._C_h = None
        self._shape = None
        self._trapped = None
        self._water_filled = None
        self._oil_filled = None
        self._n_corners = None
        self._spontaneous_imbibition = None
        self._beta = None
        self._theta_h = None
        self._b = None
        self._bo = None
        self._corner_oil_layer = None

        self.pc = None

    @property
    def G(self):
        return self.pn[f"{self.type_name}.shape_factor"][self.id]

    @property
    def r(self):
        return self.pn[f"{self.type_name}.radius"][self.id]

    @property
    def V(self):
        return self.pn[f"{self.type_name}.volume"][self.id]

    @property
    def Vclay(self):
        return self.pn[f"{self.type_name}.clay_volume"][self.id]

    @property
    def Sw(self):
        return self.pn[f"{self.type_name}.water_saturation"][self.id]

    @property
    def So(self):
        return self.pn[f"{self.type_name}.oil_saturation"][self.id]

    @property
    def advancing_contact_angle(self):
        return self.pn[f"{self.type_name}.advancing_contact_angle"][self.id]

    @property
    def receding_contact_angle(self):
        return self.pn[f"{self.type_name}.receding_contact_angle"][self.id]

    @property
    def C_r(self):
        return self.pn[f"{self.type_name}.C_r"][self.id]

    @property
    def C_a(self):
        return self.pn[f"{self.type_name}.C_a"][self.id]

    @property
    def C_h(self):
        return self.pn[f"{self.type_name}.C_h"][self.id]

    @property
    def shape(self):
        return self.pn[f"{self.type_name}.shape"][self.id]

    @property
    def trapped(self):
        return self.pn[f"{self.type_name}.trapped"][self.id]

    @property
    def oil_filled(self):
        return self.pn[f"{self.type_name}.oil_filled"][self.id]

    @property
    def water_filled(self):
        return self.pn[f"{self.type_name}.water_filled"][self.id]

    @property
    def n_corners(self):
        return self.pn[f"{self.type_name}.n_corners"][self.id]

    @property
    def spontaneous_imbibition(self):
        return self.pn[f"{self.type_name}.spontaneous_imbibition"][self.id]

    @property
    def beta(self):
        return self.pn[f"{self.type_name}.corner_half_angle"][self.id]

    @property
    def theta_h(self):
        return self.pn[f"{self.type_name}.hinging_contact_angle"][self.id]

    @property
    def b(self):
        return self.pn[f"{self.type_name}.b"][self.id]

    @property
    def bo(self):
        return self.pn[f"{self.type_name}.bo"][self.id]

    @property
    def corner_oil_layer(self):
        return self.pn[f"{self.type_name}.corner_oil_layer"][self.id]

    @Sw.setter
    def Sw(self, value):
        self.pn[f"{self.type_name}.water_saturation"][self.id] = value

    @So.setter
    def So(self, value):
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

    @C_h.setter
    def C_h(self, C_h):
        self._C_h = C_h
        self.pn[f"{self.type_name}.C_h"][self.id] = C_h

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.pn[f"{self.type_name}.shape"][self.id] = shape

        if shape == TRIANGLE:
            self._n_corners = int(3)
        elif shape == SQUARE:
            self._n_corners = int(4)
        else:
            self._n_corners = int(0)
        self.pn[f"{self.type_name}.n_corners"][self.id] = self._n_corners

    @trapped.setter
    def trapped(self, trapped):
        self._trapped = trapped
        self.pn[f"{self.type_name}.trapped"][self.id] = trapped

    @oil_filled.setter
    def oil_filled(self, oil_filled):
        self._oil_filled = oil_filled
        self.pn[f"{self.type_name}.oil_filled"][self.id] = oil_filled

    @water_filled.setter
    def water_filled(self, water_filled):
        self._water_filled = water_filled
        self.pn[f"{self.type_name}.water_filled"][self.id] = water_filled

    @spontaneous_imbibition.setter
    def spontaneous_imbibition(self, spontaneous_imbibition):
        self._spontaneous_imbibition = spontaneous_imbibition
        self.pn[f"{self.type_name}.spontaneous_imbibition"][self.id] = spontaneous_imbibition

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self.pn[f"{self.type_name}.corner_half_angle"][self.id] = beta

    @theta_h.setter
    def theta_h(self, theta_h):
        self._theta_h = theta_h
        self.pn[f"{self.type_name}.hinging_contact_angle"][self.id] = theta_h

    @b.setter
    def b(self, b):
        self._b = b
        self.pn[f"{self.type_name}.b"][self.id] = b

    @b.setter
    def bo(self, bo):
        self._bo = bo
        self.pn[f"{self.type_name}.bo"][self.id] = bo

    @corner_oil_layer.setter
    def corner_oil_layer(self, corner_oil_layer):
        self._corner_oil_layer = corner_oil_layer
        self.pn[f"{self.type_name}.corner_oil_layer"][self.id] = corner_oil_layer

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
            self.set_water_filled(filled)
        else:
            self.set_oil_filled(filled)

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
    ("C_h", nb.types.float64),
    ("shape", nb.types.int32),
    ("trapped", nb.types.boolean),
    ("beta", nb.types.float64[:]),
    ("theta_h", nb.types.float64[:]),
    ("b", nb.types.float64[:]),
    ("bo", nb.types.float64[:]),
    ("corner_oil_layer", nb.types.boolean[:]),
    ("oil_filled", nb.types.boolean),
    ("n_corners", nb.types.int32),
    ("spontaneous_imbibition", nb.types.boolean),
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
        self.C_h = 0.0
        self.shape = 0
        self.trapped = False
        self.beta = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
        self.theta_h = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
        self.b = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
        self.bo = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
        self.corner_oil_layer = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.bool_)
        self.oil_filled = True
        self.n_corners = 0
        self.spontaneous_imbibition = True
