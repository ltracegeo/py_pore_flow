import numpy as np
from .network_core import NetworkCore
from .pores_manager import PoresManager
from .throats_manager import ThroatsManager


class NetworkManager:
    def __init__(self, network, log_conductivity_file=None):
        self.pn = network
        self.log_conductivity_file = log_conductivity_file
        self.pn_core = NetworkCore(self.pn)
        self.pore = PoresManager(self.pn_core.pore)
        self.throat = ThroatsManager(self.pn_core.throat)

    def get_pore(self):
        return self.pore

    def get_throat(self):
        return self.throat

    def get_inlets_ids(self):
        return np.array(np.where(self.pn_core.pore.inlets == True)[0], np.int64)

    def get_outlets_ids(self):
        return np.array(np.where(self.pn_core.pore.outlets == True)[0], np.int64)

    def get_pores_ids(self):
        return self.pn.pores()

    def get_throats_ids(self):
        return self.pn.throats()

    def get_number_of_pores(self):
        return len(self.pn.pores())

    def get_number_of_throats(self):
        return len(self.pn.throats())

    def get_neighbor_throats_ids(self, pore_id):
        return self.pn_core.pore.conns[pore_id][: self.pn_core.pore.n_conns[pore_id]]

    def get_neighbor_pores_ids(self, throat_id):
        return self.pn_core.throat.conns[throat_id]

    def reset_property(self, property_name):
        for pore in self.pore_list:
            setattr(pore, property_name, None)
        for throat in self.throat_list:
            setattr(throat, property_name, None)
