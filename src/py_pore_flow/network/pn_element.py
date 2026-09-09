from abc import abstractmethod, ABC
from py_pore_flow.equations.geometry import (
    SQUARE,
    TRIANGLE,
    WATER,
)
from .constants import PORE, THROAT


class PnElement(ABC):
    def __init__(self, element, element_type):
        self.element = element
        self.type = element_type
        if self.type == PORE:
            self.type_name = "pore"
        else:
            self.type_name = "throat"

    def get_G(self, id):
        return self.element.shape_factor[id]

    def get_r(self, id):
        return self.element.radius[id]

    def get_V(self, id):
        return self.element.volume[id]

    def get_Vclay(self, id):
        return self.element.clay_volume[id]

    def get_Sw(self, id):
        return self.element.water_saturation[id]

    def get_So(self, id):
        return self.element.oil_saturation[id]

    def get_advancing_contact_angle(self, id):
        return self.element.advancing_contact_angle[id]

    def get_receding_contact_angle(self, id):
        return self.element.receding_contact_angle[id]

    def get_C_r(self, id):
        return self.element.C_r[id]

    def get_C_a(self, id):
        return self.element.C_a[id]

    def get_C_h(self, id):
        return self.element.C_h[id]

    def get_shape(self, id):
        return self.element.shape[id]

    def get_trapped(self, id):
        return self.element.trapped[id]

    def get_oil_filled(self, id):
        return self.element.oil_filled[id]

    def get_water_filled(self, id):
        return self.element.water_filled[id]

    def get_n_corners(self, id):
        return self.element.n_corners[id]

    def get_snap_off_viability(self, id):
        return self.element.snap_off_viability[id]

    def get_beta(self, id):
        return self.element.corner_half_angle[id]

    def get_theta_h(self, id):
        return self.element.hinging_contact_angle[id]

    def get_b(self, id):
        return self.element.b[id]

    def get_bo(self, id):
        return self.element.bo[id]

    def get_max_b(self, id):
        return self.element.max_b[id]

    def get_corner_water_layer(self, id):
        return self.element.corner_water_layer[id]

    def get_corner_oil_layer(self, id):
        return self.element.corner_oil_layer[id]

    def get_pc(self, id):
        return self.element.entry_pressure[id]

    def get_sides(self, id):
        return self.element.sides[id]

    def set_sides(self, id, value):
        self.element.sides[id] = value

    def set_G(self, id, value):
        self.element.shape_factor[id] = value

    def set_Sw(self, id, value):
        self.element.water_saturation[id] = value

    def set_So(self, id, value):
        self.element.oil_saturation[id] = value

    def set_advancing_contact_angle(self, id, value):
        self.element.advancing_contact_angle[id] = value

    def set_receding_contact_angle(self, id, value):
        self.element.receding_contact_angle[id] = value

    def set_C_r(self, id, C_r):
        self.element.C_r[id] = C_r

    def set_C_a(self, id, C_a):
        self.element.C_a[id] = C_a

    def set_C_h(self, id, C_h):
        self.element.C_h[id] = C_h

    def set_shape(self, id, shape):
        self.element.shape[id] = shape

        if shape == TRIANGLE:
            n_corners = int(3)
        elif shape == SQUARE:
            n_corners = int(4)
        else:
            n_corners = int(0)
        self.element.n_corners[id] = n_corners

    def set_trapped(self, id, trapped):
        self.element.trapped[id] = trapped

    def set_snap_off_viability(self, id, snap_off_viability):
        self.element.snap_off_viability[id] = snap_off_viability

    def set_beta(self, id, beta):
        self.element.corner_half_angle[id] = beta

    def set_theta_h(self, id, theta_h):
        self.element.hinging_contact_angle[id] = theta_h

    def set_b(self, id, b):
        self.element.b[id] = b

    def set_bo(self, id, bo):
        self.element.bo[id] = bo

    def set_max_b(self, id, max_b):
        self.element.max_b[id] = max_b

    def set_corner_oil_layer(self, id, corner_oil_layer):
        self.element.corner_oil_layer[id] = corner_oil_layer

    def set_corner_water_layer(self, id, corner_water_layer):
        self.element.corner_water_layer[id] = corner_water_layer

    def set_pc(self, id, entry_pressure):
        self.element.entry_pressure[id] = entry_pressure

    def get_id(self):
        return self.id

    def set_oil_filled(self, id, filled: bool):
        self.element.oil_filled[id] = filled
        self.element.water_filled[id] = not filled

    def set_water_filled(self, id, filled: bool):
        self.element.water_filled[id] = filled
        self.element.oil_filled[id] = not filled

    def set_filled(self, id, phase, filled: bool):
        if phase == WATER:
            self.set_water_filled(id, filled)
        else:
            self.set_oil_filled(id, filled)
        self.element.filled[id] = filled

    def is_filled(self, id, phase):
        if phase == WATER:
            return self.element.water_filled[id]
        else:
            return self.element.oil_filled[id]

    @abstractmethod
    def is_pore(self):
        pass

    @abstractmethod
    def is_throat(self):
        pass
