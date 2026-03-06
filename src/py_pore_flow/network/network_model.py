import numpy as np
from .network_core import NetworkCore
from .pore import Pore
from .throat import Throat


class Network:
    def __init__(self, network, log_conductivity_file=None):
        self.pn = network
        self.log_conductivity_file = log_conductivity_file
        self.pn_core = NetworkCore(self.pn)
        self.pore = Pore(self.pn_core.pore)
        self.throat = Throat(self.pn_core.throat)

        # number_of_pores = len(self.pn_core.pore.all)
        # number_of_throats = len(self.pn_core.throat.all)

        # self.pore_list = [None] * number_of_pores
        # self.throat_list = [None] * number_of_throats

        # for pore_id in self.pn.pores():
        #     self.pore_list[pore_id] = Pore(self.pn_core.pore, pore_id)

        # for throat_id in self.pn.throats():
        #     self.throat_list[throat_id] = Throat(self.pn_core.throat, throat_id)

    def get_pore(self):
        return self.pore

    def get_throat(self):
        return self.throat

    # def get_pore_list(self):
    #     return self.pore_list

    # def get_throat_list(self):
    #     return self.throat_list

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

    # def get_neighbor_pores(self, throat_id):
    #     neighbor_pores_list = []
    #     for pore_id in self.pn.find_connected_pores(throat_id)[0]:
    #         neighbor_pores_list.append(self.pore_list[pore_id])
    #     return neighbor_pores_list

    # def get_neighbor_throats(self, pore_id):
    #     neighbor_throats_list = []
    #     for throat_id in self.pn_core.pore.conns[pore_id][: self.pn_core.pore.n_conns[pore_id]]:
    #         neighbor_throats_list.append(self.throat_list[throat_id])
    #     return neighbor_throats_list

    # def get_neighbors(self, element):
    #     if element.is_pore():
    #         return self.get_neighbor_throats(element.get_id())
    #     else:
    #         return self.get_neighbor_pores(element.get_id())

    def reset_property(self, property_name):
        for pore in self.pore_list:
            setattr(pore, property_name, None)
        for throat in self.throat_list:
            setattr(throat, property_name, None)
