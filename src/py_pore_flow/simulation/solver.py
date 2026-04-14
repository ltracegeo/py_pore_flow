import numpy as np
from scipy.sparse import csgraph as csg


class Solver:
    def __init__(self, spy_network):
        self.spy_network = spy_network
        self.connected_pores, self.connected_throats = self._get_connected_spy_network(
            self.spy_network,
        )

    def get_sub_network(self):
        return self._get_sub_spy(self.spy_network, self.connected_pores, self.connected_throats)

    @staticmethod
    def _get_sub_spy(spy_network, sub_pores, sub_throats):
        sub_pn = {}
        for prop in spy_network.keys():
            if prop.split(".")[0] == "pore":
                sub_pn[prop] = spy_network[prop][sub_pores]
            else:
                sub_pn[prop] = spy_network[prop][sub_throats]

        def _counter():
            i = -1
            while True:
                i += 1
                yield i

        counter = _counter()
        f_counter = lambda x: next(counter) if x else 0
        new_pore_index = np.fromiter(map(f_counter, sub_pores), dtype="int")

        if len(sub_pn["throat.conns"]) == 0:
            return False
        for i in np.nditer(sub_pn["throat.conns"], op_flags=["readwrite"]):
            i[...] = new_pore_index[i]
        return sub_pn

    @staticmethod
    def _get_connected_spy_network(network):
        _, cluster_labels = Solver._get_clusters(network)
        in_labels = np.unique(cluster_labels[network[f"pore.inlets"]])
        out_labels = np.unique(cluster_labels[network[f"pore.outlets"]])
        common_labels = np.intersect1d(in_labels, out_labels, assume_unique=True)

        connected_pores = network.pores()[np.isin(cluster_labels, common_labels)]
        # connected_throats = network.throats()[
        #    np.isin(network["throat.conns"], connected_pores).all(axis=1)
        # ]

        return np.isin(cluster_labels, common_labels), np.isin(network["throat.conns"], connected_pores).all(axis=1)

    @staticmethod
    def _get_clusters(network):
        """
        clusters are numbered starting at 0
        """
        from scipy.sparse import csgraph as csg

        am = network.create_adjacency_matrix(fmt="coo", triu=True, drop_zeros=True)
        N, Cs = csg.connected_components(am, directed=False)
        return N, Cs
