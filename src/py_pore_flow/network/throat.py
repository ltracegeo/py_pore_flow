from .pn_element import PnElement
from .constants import THROAT


class Throat(PnElement):
    def __init__(self, element):
        super().__init__(element, THROAT)
        self._L = None

    def get_A(self, id):
        return self.element.cross_sectional_area[id]

    def get_L(self, id):
        return self.element.length[id]

    def set_A(self, id, value):
        self.element.cross_sectional_area[id] = value

    def is_pore(self):
        return False

    def is_throat(self):
        return True
