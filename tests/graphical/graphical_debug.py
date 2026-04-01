import numpy as np

from py_pore_flow.graphic.manual_simulation import ManualSimulationPlot
from py_pore_flow.network.network import PoreNetwork


if __name__ == "__main__":
    from PySide2 import QtWidgets
    pn = PoreNetwork()
    pn_dict = {
        "throat.conns": np.array([[0, 1]]),
        "throat.radius": np.array([1.0e-05]),
        "throat.shape_factor": np.array([0.04]),
        "throat.total_length": np.array([2.2e-05]),
        "throat.length": np.array([2.00000e-06]),
        "throat.conduit_lengths.throat": np.array([2.00000e-06]),
        "throat.volume": np.array([6.26186e-15]),
        "throat.conduit_lengths.pore1": np.array([1.0e-05]),
        "throat.conduit_lengths.pore2": np.array([1.0e-05]),
        "throat.clay_volume": np.array([0], dtype=np.int64),
        "pore.coords": np.array([[1.45e-05, 1.45e-05, 0.00e00], [1.45e-05, 1.45e-05, 2.20e-05]]),
        "pore.volume": np.array([5.78308e-15, 5.97317e-15]),
        "pore.radius": np.array([1.0e-05, 1.0e-05]),
        "pore.shape_factor": np.array([0.04, 0.04]),
        "pore.clay_volume": np.array([0, 0], dtype=np.int64),
        "throat.cross_sectional_area": np.array([3.52112676e-10]),
        "pore.area": np.array([4.15606178e-10, 4.15606178e-10]),
        "pore.outlets": np.array([False, True]),
        "pore.inlets": np.array([True, False]),
        "pore.initial_advancing_ca": np.array([np.radians(40), np.radians(40)]),
        "pore.initial_receding_ca": np.array([np.radians(20), np.radians(20)]),
        "pore.equilibrium_advancing_ca": np.array([np.radians(140), np.radians(140)]),
        "pore.equilibrium_receding_ca": np.array([np.radians(70), np.radians(110)]),
        "throat.initial_advancing_ca": np.array([np.radians(40)]),
        "throat.initial_receding_ca": np.array([np.radians(20)]),
        "throat.equilibrium_advancing_ca": np.array([np.radians(140)]),
        "throat.equilibrium_receding_ca": np.array([np.radians(60)]),
    }

    pn.update(pn_dict)

    simulation = ManualSimulationPlot(pn, generate_cas=False)
    app = QtWidgets.QApplication([])
    simulation.show()
    app.exec_()
