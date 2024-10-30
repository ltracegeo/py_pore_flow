from abc import abstractmethod
from math import pi

import numba as nb
from numba.experimental import jitclass
import numpy as np
import openpnm

from py_pore_flow.equations.geometry import (
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
        pn[f"{element_type}.sides"] = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
        pn[f"{element_type}.hinging_contact_angle"] = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
        pn[f"{element_type}.b"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), 0.0, np.float64)
        pn[f"{element_type}.bo"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), 0.0, np.float64)
        pn[f"{element_type}.max_b"] = 0.0
        pn[f"{element_type}.corner_water_layer"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), False, np.bool_)
        pn[f"{element_type}.corner_oil_layer"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), False, np.bool_)
        pn[f"{element_type}.C_r"] = 0.0
        pn[f"{element_type}.C_a"] = 0.0
        pn[f"{element_type}.C_h"] = 0.0
        pn[f"{element_type}.advancing_contact_angle"] = 0.0
        pn[f"{element_type}.receding_contact_angle"] = 0.0
        pn[f"{element_type}.oil_singlephase_conductance"] = 0.0
        pn[f"{element_type}.water_singlephase_conductance"] = 0.0
        pn[f"{element_type}.oil_multiphase_conductance"] = 0.0
        pn[f"{element_type}.water_multiphase_conductance"] = 0.0
        pn[f"{element_type}.shape"] = 0
        pn[f"{element_type}.trapped"] = False
        pn[f"{element_type}.n_corners"] = int(0)
        pn[f"{element_type}.spontaneous_imbibition"] = True
        pn[f"{element_type}.snap_off_viability"] = True
        pn[f"{element_type}.entry_pressure"] = 0.0
        pn[f"{element_type}.snap_off_pressure"] = 0.0
        pn[f"{element_type}.initial_advancing_ca"] = 0.0
        pn[f"{element_type}.initial_receding_ca"] = 0.0
        pn[f"{element_type}.equilibrium_advancing_ca"] = 0.0
        pn[f"{element_type}.equilibrium_receding_ca"] = 0.0
        pn[f"{element_type}.second_advancing_ca"] = 0.0
        pn[f"{element_type}.second_receding_ca"] = 0.0
        pn[f"{element_type}.id"] = list(range(len(pn[f"{element_type}.all"])))

    pn["throat.hydraulic_conductance"] = 0.0

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
        self.pn_core = NetworkCore(self.pn)

        number_of_pores = len(self.pn_core.pore.all)
        number_of_throats = len(self.pn_core.throat.all)

        self.pore_list = [None] * number_of_pores
        self.throat_list = [None] * number_of_throats

        for pore_id in self.pn.pores():
            self.pore_list[pore_id] = Pore(self.pn_core.pore, pore_id)

        for throat_id in self.pn.throats():
            self.throat_list[throat_id] = Throat(self.pn_core.throat, throat_id)

    def get_pore(self, id):
        return self.pore_list[id]

    def get_throat(self, id):
        return self.throat_list[id]

    def get_pore_list(self):
        return self.pore_list

    def get_throat_list(self):
        return self.throat_list

    def get_inlets_ids(self):
        return list(np.where(self.pn_core.pore.inlets == True)[0])

    def get_outlets_ids(self):
        return list(np.where(self.pn_core.pore.outlets == True)[0])

    def get_pores_ids(self):
        return list(self.pn.pores())

    def get_throats_ids(self):
        return list(self.pn.throats())

    def get_number_of_pores(self):
        return len(self.pn.pores())

    def get_number_of_throats(self):
        return len(self.pn.pores())

    def get_neighbor_throats_ids(self, pore_id):
        return self.pn_core.pore.conns[pore_id][: self.pn_core.pore.n_conns[pore_id]]

    def get_neighbor_pores_ids(self, throat_id):
        return self.pn_core.throat.conns[throat_id]

    def get_neighbor_pores(self, throat_id):
        neighbor_pores_list = []
        for pore_id in self.pn.find_connected_pores(throat_id)[0]:
            neighbor_pores_list.append(self.pore_list[pore_id])
        return neighbor_pores_list

    def get_neighbor_throats(self, pore_id):
        neighbor_throats_list = []
        for throat_id in self.pn_core.pore.conns[pore_id][: self.pn_core.pore.n_conns[pore_id]]:
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


class PnElement:
    def __init__(self, element, type_name, id):
        self.element = element
        self.type_name = type_name
        self.id = id

    @property
    def G(self):
        return self.element.shape_factor[self.id]

    @property
    def r(self):
        return self.element.radius[self.id]

    @property
    def V(self):
        return self.element.volume[self.id]

    @property
    def Vclay(self):
        return self.element.clay_volume[self.id]

    @property
    def Sw(self):
        return self.element.water_saturation[self.id]

    @property
    def So(self):
        return self.element.oil_saturation[self.id]

    @property
    def advancing_contact_angle(self):
        return self.element.advancing_contact_angle[self.id]

    @property
    def receding_contact_angle(self):
        return self.element.receding_contact_angle[self.id]

    @property
    def C_r(self):
        return self.element.C_r[self.id]

    @property
    def C_a(self):
        return self.element.C_a[self.id]

    @property
    def C_h(self):
        return self.element.C_h[self.id]

    @property
    def shape(self):
        return self.element.shape[self.id]

    @property
    def trapped(self):
        return self.element.trapped[self.id]

    @property
    def oil_filled(self):
        return self.element.oil_filled[self.id]

    @property
    def water_filled(self):
        return self.element.water_filled[self.id]

    @property
    def n_corners(self):
        return self.element.n_corners[self.id]

    @property
    def spontaneous_imbibition(self):
        return self.element.spontaneous_imbibition[self.id]

    @property
    def snap_off_viability(self):
        return self.element.snap_off_viability[self.id]

    @property
    def beta(self):
        return self.element.corner_half_angle[self.id]

    @property
    def theta_h(self):
        return self.element.hinging_contact_angle[self.id]

    @property
    def b(self):
        return self.element.b[self.id]

    @property
    def bo(self):
        return self.element.bo[self.id]

    @property
    def max_b(self):
        return self.element.max_b[self.id]

    @property
    def corner_water_layer(self):
        return self.element.corner_water_layer[self.id]

    @property
    def corner_oil_layer(self):
        return self.element.corner_oil_layer[self.id]

    @property
    def pc(self):
        return self.element.entry_pressure[self.id]

    @property
    def sides(self):
        return self.element.sides[self.id]

    @sides.setter
    def sides(self, value):
        self.element.sides[self.id] = value

    @G.setter
    def G(self, value):
        self.element.shape_factor[self.id] = value

    @Sw.setter
    def Sw(self, value):
        self.element.water_saturation[self.id] = value

    @So.setter
    def So(self, value):
        self.element.oil_saturation[self.id] = value

    @advancing_contact_angle.setter
    def advancing_contact_angle(self, value):
        self.element.advancing_contact_angle[self.id] = value

    @receding_contact_angle.setter
    def receding_contact_angle(self, value):
        self.element.receding_contact_angle[self.id] = value

    @C_r.setter
    def C_r(self, C_r):
        self.element.C_r[self.id] = C_r

    @C_a.setter
    def C_a(self, C_a):
        self.element.C_a[self.id] = C_a

    @C_h.setter
    def C_h(self, C_h):
        self.element.C_h[self.id] = C_h

    @shape.setter
    def shape(self, shape):
        self.element.shape[self.id] = shape

        if shape == TRIANGLE:
            n_corners = int(3)
        elif shape == SQUARE:
            n_corners = int(4)
        else:
            n_corners = int(0)
        self.element.n_corners[self.id] = n_corners

    @trapped.setter
    def trapped(self, trapped):
        self.element.trapped[self.id] = trapped

    @oil_filled.setter
    def oil_filled(self, oil_filled):
        self.element.oil_filled[self.id] = oil_filled

    @water_filled.setter
    def water_filled(self, water_filled):
        self.element.water_filled[self.id] = water_filled

    @spontaneous_imbibition.setter
    def spontaneous_imbibition(self, spontaneous_imbibition):
        self.element.spontaneous_imbibition[self.id] = spontaneous_imbibition

    @snap_off_viability.setter
    def snap_off_viability(self, snap_off_viability):
        self.element.snap_off_viability[self.id] = snap_off_viability

    @beta.setter
    def beta(self, beta):
        self.element.corner_half_angle[self.id] = beta

    @theta_h.setter
    def theta_h(self, theta_h):
        self.element.hinging_contact_angle[self.id] = theta_h

    @b.setter
    def b(self, b):
        self.element.b[self.id] = b

    @bo.setter
    def bo(self, bo):
        self.element.bo[self.id] = bo

    @max_b.setter
    def max_b(self, max_b):
        self.element.max_b[self.id] = max_b

    @corner_oil_layer.setter
    def corner_oil_layer(self, corner_oil_layer):
        self.element.corner_oil_layer[self.id] = corner_oil_layer

    @corner_water_layer.setter
    def corner_water_layer(self, corner_water_layer):
        self.element.corner_water_layer[self.id] = corner_water_layer

    @pc.setter
    def pc(self, entry_pressure):
        self.element.entry_pressure[self.id] = entry_pressure

    def get_id(self):
        return self.id

    def get_pc(self):
        return self.element.entry_pressure[self.id]

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

    def set_pc(self, entry_pressure):
        self.element.entry_pressure[self.id] = entry_pressure

    @abstractmethod
    def is_pore(self):
        pass

    @abstractmethod
    def is_throat(self):
        pass


class Pore(PnElement):
    def __init__(self, element, id):
        super().__init__(element, "pore", id)

    @property
    def A(self):
        return self.element.area[self.id]

    @A.setter
    def A(self, value):
        self.element.area[self.id] = value

    def is_pore(self):
        return True

    def is_throat(self):
        return False


class Throat(PnElement):
    def __init__(self, element, id):
        super().__init__(element, "throat", id)
        self._L = None

    @property
    def A(self):
        return self.element.cross_sectional_area[self.id]

    @property
    def L(self):
        return self.element.length[self.id]

    @A.setter
    def A(self, value):
        self.element.cross_sectional_area[self.id] = value

    def is_pore(self):
        return False

    def is_throat(self):
        return True


class ElementNetwork:
    def __init__(self):
        self.pn = None

    def __setitem__(self, idx, value):
        self.pn[idx] = value
        setattr(self, idx.split(".")[1], value)


class PoresNetwork(ElementNetwork):
    def __init__(self, pn):
        self.pn = pn

        self.all = pn["pore.all"]
        self.shape_factor = pn["pore.shape_factor"]
        self.radius = pn["pore.radius"]
        self.volume = pn["pore.volume"]
        self.clay_volume = pn["pore.clay_volume"]
        self.oil_filled = pn["pore.oil_filled"]
        self.water_filled = pn["pore.water_filled"]
        self.oil_saturation = pn["pore.oil_saturation"]
        self.water_saturation = pn["pore.water_saturation"]
        self.corner_half_angle = pn["pore.corner_half_angle"]
        self.sides = pn["pore.sides"]
        self.hinging_contact_angle = pn["pore.hinging_contact_angle"]
        self.b = pn["pore.b"]
        self.bo = pn["pore.bo"]
        self.max_b = pn["pore.max_b"]
        self.corner_water_layer = pn["pore.corner_water_layer"]
        self.corner_oil_layer = pn["pore.corner_oil_layer"]
        self.C_r = pn["pore.C_r"]
        self.C_a = pn["pore.C_a"]
        self.C_h = pn["pore.C_h"]
        self.advancing_contact_angle = pn["pore.advancing_contact_angle"]
        self.receding_contact_angle = pn["pore.receding_contact_angle"]
        self.oil_singlephase_conductance = pn["pore.oil_singlephase_conductance"]
        self.water_singlephase_conductance = pn["pore.water_singlephase_conductance"]
        self.oil_multiphase_conductance = pn["pore.oil_multiphase_conductance"]
        self.water_multiphase_conductance = pn["pore.water_multiphase_conductance"]
        self.shape = pn["pore.shape"]
        self.trapped = pn["pore.trapped"]
        self.n_corners = pn["pore.n_corners"]
        self.spontaneous_imbibition = pn["pore.spontaneous_imbibition"]
        self.snap_off_viability = pn["pore.snap_off_viability"]
        self.entry_pressure = pn["pore.entry_pressure"]
        self.snap_off_pressure = pn["pore.snap_off_pressure"]
        self.conns = pn["pore.conns"]
        self.n_conns = pn["pore.n_conns"]
        self.area = pn["pore.area"]
        self.inlets = pn["pore.inlets"]
        self.outlets = pn["pore.outlets"]
        self.initial_advancing_ca = pn["pore.initial_advancing_ca"]
        self.initial_receding_ca = pn["pore.initial_receding_ca"]
        self.equilibrium_advancing_ca = pn["pore.equilibrium_advancing_ca"]
        self.equilibrium_receding_ca = pn["pore.equilibrium_receding_ca"]
        self.second_advancing_ca = pn["pore.second_advancing_ca"]
        self.second_receding_ca = pn["pore.second_receding_ca"]


class ThroatsNetwork(ElementNetwork):
    def __init__(self, pn):
        self.pn = pn

        self.all = pn["throat.all"]
        self.shape_factor = pn["throat.shape_factor"]
        self.radius = pn["throat.radius"]
        self.volume = pn["throat.volume"]
        self.clay_volume = pn["throat.clay_volume"]
        self.oil_filled = pn["throat.oil_filled"]
        self.water_filled = pn["throat.water_filled"]
        self.oil_saturation = pn["throat.oil_saturation"]
        self.water_saturation = pn["throat.water_saturation"]
        self.corner_half_angle = pn["throat.corner_half_angle"]
        self.sides = pn["throat.sides"]
        self.hinging_contact_angle = pn["throat.hinging_contact_angle"]
        self.b = pn["throat.b"]
        self.bo = pn["throat.bo"]
        self.max_b = pn["throat.max_b"]
        self.corner_water_layer = pn["throat.corner_water_layer"]
        self.corner_oil_layer = pn["throat.corner_oil_layer"]
        self.C_r = pn["throat.C_r"]
        self.C_a = pn["throat.C_a"]
        self.C_h = pn["throat.C_h"]
        self.advancing_contact_angle = pn["throat.advancing_contact_angle"]
        self.receding_contact_angle = pn["throat.receding_contact_angle"]
        self.oil_singlephase_conductance = pn["throat.oil_singlephase_conductance"]
        self.water_singlephase_conductance = pn["throat.water_singlephase_conductance"]
        self.oil_multiphase_conductance = pn["throat.oil_multiphase_conductance"]
        self.water_multiphase_conductance = pn["throat.water_multiphase_conductance"]
        self.shape = pn["throat.shape"]
        self.trapped = pn["throat.trapped"]
        self.n_corners = pn["throat.n_corners"]
        self.spontaneous_imbibition = pn["throat.spontaneous_imbibition"]
        self.snap_off_viability = pn["throat.snap_off_viability"]
        self.entry_pressure = pn["throat.entry_pressure"]
        self.snap_off_pressure = pn["throat.snap_off_pressure"]
        self.conns = pn["throat.conns"]
        self.n_conns = pn["throat.n_conns"]
        self.cross_sectional_area = pn["throat.cross_sectional_area"]
        self.length = pn["throat.length"]
        self.hydraulic_conductance = pn["throat.hydraulic_conductance"]
        self.initial_advancing_ca = pn["throat.initial_advancing_ca"]
        self.initial_receding_ca = pn["throat.initial_receding_ca"]
        self.equilibrium_advancing_ca = pn["throat.equilibrium_advancing_ca"]
        self.equilibrium_receding_ca = pn["throat.equilibrium_receding_ca"]
        self.second_advancing_ca = pn["throat.second_advancing_ca"]
        self.second_receding_ca = pn["throat.second_receding_ca"]


class NetworkCore:
    def __init__(self, pn):
        self.pn = pn
        self.throat = ThroatsNetwork(pn)
        self.pore = PoresNetwork(pn)

    def __setitem__(self, idx, value):
        self.pn[idx] = value

        splitted = idx.split(".")
        if splitted[0] == "pore":
            setattr(self.pore, splitted[1], self.pn[idx])
        else:
            setattr(self.throat, splitted[1], self.pn[idx])
