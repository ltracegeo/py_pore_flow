from .pores_network import PoresNetwork
from .throats_network import ThroatsNetwork


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
