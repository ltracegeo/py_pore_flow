from .pn_element import PnElement
from .constants import PORE


class PoresManager(PnElement):
    def __init__(self, element):
        super().__init__(element, PORE)

    def get_A(self, id):
        return self.element.area[id]

    def set_A(self, id, value):
        self.element.area[id] = value

    def is_pore(self):
        return True

    def is_throat(self):
        return False
