import numpy as np
from typing import Literal

from py_pore_flow.network.network import PoreNetwork


def get_simple_network(
    initial_saturation: Literal["water_filled", "oil_filled", "mix_filled"] = "water_filled"
    ) -> PoreNetwork:
    """
    Returns a simple network for testing purposes.
    The network structure is as follows:
    #  -> 0-----0----0---O---O-->
    #            \
    #             0----O
    #            /      \
    #  -> O--O--O        O---O-->
    #                   /
    #  -> 0--@--@--0--O
    #
    #  -> 0-----1----2---3---4------>
    #            \
    #             5-------6
    #            /         \
    #  -> 7--8--9          10---11-->
    #                      /
    #  -> 12--13--14--15--16
    """
    Np = 17
    Nt = 16
    pn_dict = {
        "throat.conns": np.array(
            [
                [0, 1], [1, 2], [2, 3], [3, 4],
                [1, 5], [5, 6], [5, 9], [6, 10],
                [7, 8], [8, 9], [10, 11], [16, 10],
                [12, 13], [13, 14], [14, 15], [15, 16],
            ]
        ),
        "pore.coords": np.array(
            [
                [0, 0, 0], [6, 0, 0], [11, 0, 0], [15, 0, 0], [19, 0, 0],
                [8, 2, 0], [13, 2, 0],
                [0, 4, 0], [3, 4, 0], [6, 4, 0], [15, 4, 0], [19, 4, 0],
                [0, 6, 0], [3, 6, 0], [6, 6, 0], [9, 6, 0], [12, 6, 0],
            ]
        ) * 1e-5,
        "pore.outlets": np.array(
            [False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False]
        ),
        "pore.inlets": np.array(
            [True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False]
        ),
        "throat.shape_factor": np.ones(Nt)*0.05,
        "throat.radius": np.ones(Nt) * 5e-6,
        "throat.volume": np.ones(Nt) * 1e-15,
        "throat.cross_sectional_area": np.ones(Nt) * 1e-10,
        "throat.length": np.ones(Nt) * 1e-5,
        "pore.shape_factor": np.ones(Np)*0.05,
        "pore.radius": np.ones(Np) * 1e-5,
        "pore.volume": np.ones(Np) * 1e-14,
        "pore.area": np.ones(Np) * 1e-9,
        "pore.length": np.ones(Np) * 1e-5,
    }
    if initial_saturation == "water_filled":
        pn_dict["pore.water_saturation"] = np.zeros(Np)
        pn_dict["throat.water_saturation"] = np.ones(Nt)
        pn_dict["pore.water_filled"] = np.array([True,] * Np)
        pn_dict["throat.water_filled"] = np.array([True,] * Nt)
    elif initial_saturation == "mix_filled":
        pn_dict["pore.water_saturation"] = np.array(
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0]
        )
        pn_dict["throat.water_saturation"] = np.array(
            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        )
        pn_dict["pore.water_filled"] = np.array(
            [True, True, True, False, False, True, False, False, False, False, False, False, True, False, False, True, False]
        )
        pn_dict["throat.water_filled"] = np.array(
            [True, True, True, False, True, True, True, False, False, False, False, False, True, True, True, True]
        )
    else:
        raise ValueError(f"Invalid initial_saturation value: {initial_saturation}")
    
    pn = PoreNetwork()
    pn.update(pn_dict)
    return pn
