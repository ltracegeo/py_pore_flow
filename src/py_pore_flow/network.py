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

        self.pn_core = NetworkCore()
        self.pn_core.pore.shape_factor = network["pore.shape_factor"]
        self.pn_core.pore.radius = network["pore.radius"]
        self.pn_core.pore.volume = network["pore.volume"]
        self.pn_core.pore.clay_volume = network["pore.clay_volume"]
        self.pn_core.pore.oil_filled = network["pore.oil_filled"]
        self.pn_core.pore.water_filled = network["pore.water_filled"]
        self.pn_core.pore.oil_saturation = network["pore.oil_saturation"]
        self.pn_core.pore.water_saturation = network["pore.water_saturation"]
        self.pn_core.pore.corner_half_angle = network["pore.corner_half_angle"]
        self.pn_core.pore.sides = network["pore.sides"]
        self.pn_core.pore.hinging_contact_angle = network["pore.hinging_contact_angle"]
        self.pn_core.pore.b = network["pore.b"]
        self.pn_core.pore.bo = network["pore.bo"]
        self.pn_core.pore.max_b = network["pore.max_b"]
        self.pn_core.pore.corner_water_layer = network["pore.corner_water_layer"]
        self.pn_core.pore.corner_oil_layer = network["pore.corner_oil_layer"]
        self.pn_core.pore.C_r = network["pore.C_r"]
        self.pn_core.pore.C_a = network["pore.C_a"]
        self.pn_core.pore.C_h = network["pore.C_h"]
        self.pn_core.pore.advancing_contact_angle = network["pore.advancing_contact_angle"]
        self.pn_core.pore.receding_contact_angle = network["pore.receding_contact_angle"]
        self.pn_core.pore.oil_singlephase_conductance = network["pore.oil_singlephase_conductance"]
        self.pn_core.pore.water_singlephase_conductance = network["pore.water_singlephase_conductance"]
        self.pn_core.pore.oil_multiphase_conductance = network["pore.oil_multiphase_conductance"]
        self.pn_core.pore.water_multiphase_conductance = network["pore.water_multiphase_conductance"]
        self.pn_core.pore.shape = network["pore.shape"]
        self.pn_core.pore.trapped = network["pore.trapped"]
        self.pn_core.pore.n_corners = network["pore.n_corners"]
        self.pn_core.pore.spontaneous_imbibition = network["pore.spontaneous_imbibition"]
        self.pn_core.pore.snap_off_viability = network["pore.snap_off_viability"]
        self.pn_core.pore.entry_pressure = network["pore.entry_pressure"]
        self.pn_core.pore.conns = network["pore.conns"]
        self.pn_core.pore.n_conns = network["pore.n_conns"]
        self.pn_core.pore.area = network["pore.area"]
        self.pn_core.throat.shape_factor = network["throat.shape_factor"]
        self.pn_core.throat.radius = network["throat.radius"]
        self.pn_core.throat.volume = network["throat.volume"]
        self.pn_core.throat.clay_volume = network["throat.clay_volume"]
        self.pn_core.throat.oil_filled = network["throat.oil_filled"]
        self.pn_core.throat.water_filled = network["throat.water_filled"]
        self.pn_core.throat.oil_saturation = network["throat.oil_saturation"]
        self.pn_core.throat.water_saturation = network["throat.water_saturation"]
        self.pn_core.throat.corner_half_angle = network["throat.corner_half_angle"]
        self.pn_core.throat.sides = network["throat.sides"]
        self.pn_core.throat.hinging_contact_angle = network["throat.hinging_contact_angle"]
        self.pn_core.throat.b = network["throat.b"]
        self.pn_core.throat.bo = network["throat.bo"]
        self.pn_core.throat.max_b = network["throat.max_b"]
        self.pn_core.throat.corner_water_layer = network["throat.corner_water_layer"]
        self.pn_core.throat.corner_oil_layer = network["throat.corner_oil_layer"]
        self.pn_core.throat.C_r = network["throat.C_r"]
        self.pn_core.throat.C_a = network["throat.C_a"]
        self.pn_core.throat.C_h = network["throat.C_h"]
        self.pn_core.throat.advancing_contact_angle = network["throat.advancing_contact_angle"]
        self.pn_core.throat.receding_contact_angle = network["throat.receding_contact_angle"]
        self.pn_core.throat.oil_singlephase_conductance = network["throat.oil_singlephase_conductance"]
        self.pn_core.throat.water_singlephase_conductance = network["throat.water_singlephase_conductance"]
        self.pn_core.throat.oil_multiphase_conductance = network["throat.oil_multiphase_conductance"]
        self.pn_core.throat.water_multiphase_conductance = network["throat.water_multiphase_conductance"]
        self.pn_core.throat.shape = network["throat.shape"]
        self.pn_core.throat.trapped = network["throat.trapped"]
        self.pn_core.throat.n_corners = network["throat.n_corners"]
        self.pn_core.throat.spontaneous_imbibition = network["throat.spontaneous_imbibition"]
        self.pn_core.throat.snap_off_viability = network["throat.snap_off_viability"]
        self.pn_core.throat.entry_pressure = network["throat.entry_pressure"]
        self.pn_core.throat.conns = network["throat.conns"]
        self.pn_core.throat.n_conns = network["throat.n_conns"]
        self.pn_core.throat.cross_sectional_area = network["throat.cross_sectional_area"]
        self.pn_core.throat.length = network["throat.length"]
        self.pn_core.throat.hydraulic_conductance = network["throat.hydraulic_conductance"]

        number_of_pores = len(self.pn[f"pore.all"])
        number_of_throats = len(self.pn[f"throat.all"])

        self.pore_list = [None] * number_of_pores
        self.throat_list = [None] * number_of_throats
        pore_core_list = [None] * number_of_pores
        throat_core_list = [None] * number_of_throats

        for pore_id in self.pn.pores():
            self.pore_list[pore_id] = Pore(self.pn_core, pore_id)
            pore_core_list[pore_id] = ElementCore(pore_id)
        self.pore_core_list = nb.typed.List(pore_core_list)

        for throat_id in self.pn.throats():
            self.throat_list[throat_id] = Throat(self.pn_core, throat_id)
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

    @property
    def G(self):
        return self.pn.shape_factor[self.id]

    @property
    def r(self):
        return self.pn.radius[self.id]

    @property
    def V(self):
        return self.pn.volume[self.id]

    @property
    def Vclay(self):
        return self.pn.clay_volume[self.id]

    @property
    def Sw(self):
        return self.pn.water_saturation[self.id]

    @property
    def So(self):
        return self.pn.oil_saturation[self.id]

    @property
    def advancing_contact_angle(self):
        return self.pn.advancing_contact_angle[self.id]

    @property
    def receding_contact_angle(self):
        return self.pn.receding_contact_angle[self.id]

    @property
    def C_r(self):
        return self.pn.C_r[self.id]

    @property
    def C_a(self):
        return self.pn.C_a[self.id]

    @property
    def C_h(self):
        return self.pn.C_h[self.id]

    @property
    def shape(self):
        return self.pn.shape[self.id]

    @property
    def trapped(self):
        return self.pn.trapped[self.id]

    @property
    def oil_filled(self):
        return self.pn.oil_filled[self.id]

    @property
    def water_filled(self):
        return self.pn.water_filled[self.id]

    @property
    def n_corners(self):
        return self.pn.n_corners[self.id]

    @property
    def spontaneous_imbibition(self):
        return self.pn.spontaneous_imbibition[self.id]

    @property
    def snap_off_viability(self):
        return self.pn.snap_off_viability[self.id]

    @property
    def beta(self):
        return self.pn.corner_half_angle[self.id]

    @property
    def theta_h(self):
        return self.pn.hinging_contact_angle[self.id]

    @property
    def b(self):
        return self.pn.b[self.id]

    @property
    def bo(self):
        return self.pn.bo[self.id]

    @property
    def max_b(self):
        return self.pn.max_b[self.id]

    @property
    def corner_water_layer(self):
        return self.pn.corner_water_layer[self.id]

    @property
    def corner_oil_layer(self):
        return self.pn.corner_oil_layer[self.id]

    @property
    def pc(self):
        return self.pn.entry_pressure[self.id]

    @G.setter
    def G(self, value):
        self.pn.shape_factor[self.id] = value

    @Sw.setter
    def Sw(self, value):
        self.pn.water_saturation[self.id] = value

    @So.setter
    def So(self, value):
        self.pn.oil_saturation[self.id] = value

    @advancing_contact_angle.setter
    def advancing_contact_angle(self, value):
        self.pn.advancing_contact_angle[self.id] = value

    @receding_contact_angle.setter
    def receding_contact_angle(self, value):
        self.pn.receding_contact_angle[self.id] = value

    @C_r.setter
    def C_r(self, C_r):
        self.pn.C_r[self.id] = C_r

    @C_a.setter
    def C_a(self, C_a):
        self.pn.C_a[self.id] = C_a

    @C_h.setter
    def C_h(self, C_h):
        self.pn.C_h[self.id] = C_h

    @shape.setter
    def shape(self, shape):
        self.pn.shape[self.id] = shape

        if shape == TRIANGLE:
            self.pn.n_corners[self.id] = int(3)
        elif shape == SQUARE:
            self.pn.n_corners[self.id] = int(4)
        else:
            self.pn.n_corners[self.id] = int(0)

    @trapped.setter
    def trapped(self, trapped):
        self.pn.trapped[self.id] = trapped

    @oil_filled.setter
    def oil_filled(self, oil_filled):
        self.pn.oil_filled[self.id] = oil_filled

    @water_filled.setter
    def water_filled(self, water_filled):
        self.pn.water_filled[self.id] = water_filled

    @spontaneous_imbibition.setter
    def spontaneous_imbibition(self, spontaneous_imbibition):
        self.pn.spontaneous_imbibition[self.id] = spontaneous_imbibition

    @snap_off_viability.setter
    def snap_off_viability(self, snap_off_viability):
        self.pn.snap_off_viability[self.id] = snap_off_viability

    @beta.setter
    def beta(self, beta):
        self.pn.corner_half_angle[self.id] = beta

    @theta_h.setter
    def theta_h(self, theta_h):
        self.pn.hinging_contact_angle[self.id] = theta_h

    @b.setter
    def b(self, b):
        self.pn.b[self.id] = b

    @bo.setter
    def bo(self, bo):
        self.pn.bo[self.id] = bo

    @max_b.setter
    def max_b(self, max_b):
        self.pn.max_b[self.id] = max_b

    @corner_oil_layer.setter
    def corner_oil_layer(self, corner_oil_layer):
        self.pn.corner_oil_layer[self.id] = corner_oil_layer

    @corner_water_layer.setter
    def corner_water_layer(self, corner_water_layer):
        self.pn.corner_water_layer[self.id] = corner_water_layer

    def get_id(self):
        return self.id

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
        self.pn.entry_pressure[self.id] = pc

    @abstractmethod
    def is_pore(self):
        pass

    @abstractmethod
    def is_throat(self):
        pass


class Pore(PnElement):
    def __init__(self, network, id):
        super().__init__(network.pore, "pore", id)

    @property
    def A(self):
        return self.pn.area[self.id]

    @A.setter
    def A(self, value):
        self.pn.area[self.id] = value

    def is_pore(self):
        return True

    def is_throat(self):
        return False


class Throat(PnElement):
    def __init__(self, network, id):
        super().__init__(network.throat, "throat", id)
        self._L = None

    @property
    def A(self):
        return self.pn.cross_sectional_area[self.id]

    @property
    def L(self):
        return self.pn.length[self.id]

    @A.setter
    def A(self, value):
        self.pn.cross_sectional_area[self.id] = value

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

NUMBA_ENABLED = False
if NUMBA_ENABLED:
    spec = [
        # Default
        ("shape_factor", nb.types.float64[:]),
        ("radius", nb.types.float64[:]),
        ("volume", nb.types.float64[:]),
        ("clay_volume", nb.types.int64[:]),
        # Custom
        ("oil_filled", nb.types.boolean[:]),
        ("water_filled", nb.types.boolean[:]),
        ("oil_saturation", nb.types.float64[:]),
        ("water_saturation", nb.types.float64[:]),
        ("corner_half_angle", nb.types.float64[:, :]),
        ("sides", nb.types.float64[:, :]),
        ("hinging_contact_angle", nb.types.float64[:, :]),
        ("b", nb.types.float64[:, :]),
        ("bo", nb.types.float64[:, :]),
        ("max_b", nb.types.float64[:]),
        ("corner_water_layer", nb.types.boolean[:, :]),
        ("corner_oil_layer", nb.types.boolean[:, :]),
        ("C_r", nb.types.float64[:]),
        ("C_a", nb.types.float64[:]),
        ("C_h", nb.types.float64[:]),
        ("advancing_contact_angle", nb.types.float64[:]),
        ("receding_contact_angle", nb.types.float64[:]),
        ("oil_singlephase_conductance", nb.types.float64[:]),
        ("water_singlephase_conductance", nb.types.float64[:]),
        ("oil_multiphase_conductance", nb.types.float64[:]),
        ("water_multiphase_conductance", nb.types.float64[:]),
        ("shape", nb.types.int32[:]),
        ("trapped", nb.types.boolean[:]),
        ("n_corners", nb.types.int32[:]),
        ("spontaneous_imbibition", nb.types.boolean[:]),
        ("snap_off_viability", nb.types.boolean[:]),
        ("entry_pressure", nb.types.float64[:]),
        ("conns", nb.types.int32[:, :]),
        ("n_conns", nb.types.int32[:]),
        # Pore exclusive
        ("area", nb.types.float64[:]),
    ]


    @jitclass(spec)
    class PoresNetwork:
        def __init__(self):
            pass


    spec = [
        # Default
        ("shape_factor", nb.types.float64[:]),
        ("radius", nb.types.float64[:]),
        ("volume", nb.types.float64[:]),
        ("clay_volume", nb.types.int64[:]),
        # Custom
        ("oil_filled", nb.types.boolean[:]),
        ("water_filled", nb.types.boolean[:]),
        ("oil_saturation", nb.types.float64[:]),
        ("water_saturation", nb.types.float64[:]),
        ("corner_half_angle", nb.types.float64[:, :]),
        ("sides", nb.types.float64[:, :]),
        ("hinging_contact_angle", nb.types.float64[:, :]),
        ("b", nb.types.float64[:, :]),
        ("bo", nb.types.float64[:, :]),
        ("max_b", nb.types.float64[:]),
        ("corner_water_layer", nb.types.boolean[:, :]),
        ("corner_oil_layer", nb.types.boolean[:, :]),
        ("C_r", nb.types.float64[:]),
        ("C_a", nb.types.float64[:]),
        ("C_h", nb.types.float64[:]),
        ("advancing_contact_angle", nb.types.float64[:]),
        ("receding_contact_angle", nb.types.float64[:]),
        ("oil_singlephase_conductance", nb.types.float64[:]),
        ("water_singlephase_conductance", nb.types.float64[:]),
        ("oil_multiphase_conductance", nb.types.float64[:]),
        ("water_multiphase_conductance", nb.types.float64[:]),
        ("shape", nb.types.int32[:]),
        ("trapped", nb.types.boolean[:]),
        ("n_corners", nb.types.int32[:]),
        ("spontaneous_imbibition", nb.types.boolean[:]),
        ("snap_off_viability", nb.types.boolean[:]),
        ("entry_pressure", nb.types.float64[:]),
        ("conns", nb.types.int32[:, :]),
        ("n_conns", nb.types.int32[:]),
        # Throat exclusive
        ("cross_sectional_area", nb.types.float64[:]),
        ("length", nb.types.float64[:]),
        ("hydraulic_conductance", nb.types.float64[:]),
    ]


    @jitclass(spec)
    class ThroatsNetwork:
        def __init__(self):
            pass


    spec = [
        ("throat", ThroatsNetwork.class_type.instance_type),
        ("pore", PoresNetwork.class_type.instance_type),
    ]


    @jitclass(spec)
    class NetworkCore:
        def __init__(self):
            self.throat = ThroatsNetwork()
            self.pore = PoresNetwork()
else:
    class PoresNetwork:
        def __init__(self):
            self.shape_factor = None
            self.radius = None
            self.volume = None
            self.clay_volume = None
            self.oil_filled = None
            self.water_filled = None
            self.oil_saturation = None
            self.water_saturation = None
            self.corner_half_angle = None
            self.sides = None
            self.hinging_contact_angle = None
            self.b = None
            self.bo = None
            self.max_b = None
            self.corner_water_layer = None
            self.corner_oil_layer = None
            self.C_r = None
            self.C_a = None
            self.C_h = None
            self.advancing_contact_angle = None
            self.receding_contact_angle = None
            self.oil_singlephase_conductance = None
            self.water_singlephase_conductance = None
            self.oil_multiphase_conductance = None
            self.water_multiphase_conductance = None
            self.shape = None
            self.trapped = None
            self.n_corners = None
            self.spontaneous_imbibition = None
            self.snap_off_viability = None
            self.entry_pressure = None
            self.conns = None
            self.n_conns = None
            self.area = None

    class ThroatsNetwork:
        def __init__(self):
            self.shape_factor = None
            self.radius = None
            self.volume = None
            self.clay_volume = None
            self.oil_filled = None
            self.water_filled = None
            self.oil_saturation = None
            self.water_saturation = None
            self.corner_half_angle = None
            self.sides = None
            self.hinging_contact_angle = None
            self.b = None
            self.bo = None
            self.max_b = None
            self.corner_water_layer = None
            self.corner_oil_layer = None
            self.C_r = None
            self.C_a = None
            self.C_h = None
            self.advancing_contact_angle = None
            self.receding_contact_angle = None
            self.oil_singlephase_conductance = None
            self.water_singlephase_conductance = None
            self.oil_multiphase_conductance = None
            self.water_multiphase_conductance = None
            self.shape = None
            self.trapped = None
            self.n_corners = None
            self.spontaneous_imbibition = None
            self.snap_off_viability = None
            self.entry_pressure = None
            self.conns = None
            self.n_conns = None
            self.cross_sectional_area = None
            self.length = None
            self.hydraulic_conductance = None

    class NetworkCore:
        def __init__(self):
            self.throat = ThroatsNetwork()
            self.pore = PoresNetwork()
