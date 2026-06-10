import numpy as np
from PySide2.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QTabWidget,
)

from py_pore_flow.graphic.diagnostics.diagnostics_tabs import (
    DiagnosticsTab,
    ElementGeometryTab,
    CrossPlotTab,
    HeatmapTab,
)
from py_pore_flow.output.diagnostics import Diagnostics


class DiagnosticsWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.pores_tab = DiagnosticsTab()
        self.throats_tab = DiagnosticsTab()
        self.element_tab = ElementGeometryTab()
        self.crossplot_tab = CrossPlotTab()
        self.heatmap_tab = HeatmapTab()

        tabs = QTabWidget()
        tabs.addTab(self.pores_tab, "Pores")
        tabs.addTab(self.throats_tab, "Throats")
        tabs.addTab(self.element_tab, "Element")
        tabs.addTab(self.crossplot_tab, "Cross Plot")
        tabs.addTab(self.heatmap_tab, "Heatmap")

        layout = QHBoxLayout(self)
        layout.addWidget(tabs)

    def set_diagnostics(self, diagnostics_path):
        self._diagnostics = Diagnostics(diagnostics_path)
        self._ensure_numpy_network(self._diagnostics.network)
        self._ensure_pore_conns(self._diagnostics.network, self._diagnostics.info.n_pores)
        self._ensure_pore_aspect_ratio(self._diagnostics.network, self._diagnostics.info.n_pores)

        self.pores_tab.set_diagnostics(self._diagnostics, "pore")
        self.throats_tab.set_diagnostics(self._diagnostics, "throat")
        self.element_tab.set_diagnostics(self._diagnostics)
        self.crossplot_tab.set_diagnostics(self._diagnostics)
        self.heatmap_tab.set_diagnostics(self._diagnostics)

    @staticmethod
    def _ensure_numpy_network(network):
        for key, value in network.items():
            if isinstance(value, list):
                try:
                    network[key] = np.array(value)
                except ValueError:
                    pass

    @staticmethod
    def _ensure_pore_conns(network, n_pores):
        if "pore.conns" in network or "throat.conns" not in network:
            return

        throat_conns = np.asarray(network["throat.conns"])
        incident_throats = [[] for _ in range(n_pores)]
        for throat_id, (pore_a, pore_b) in enumerate(throat_conns):
            incident_throats[int(pore_a)].append(throat_id)
            incident_throats[int(pore_b)].append(throat_id)

        network["pore.conns"] = [np.array(throats, dtype=int) for throats in incident_throats]

    @staticmethod
    def _ensure_pore_aspect_ratio(network, n_pores):
        if "pore.aspect_ratio" in network or not {"throat.conns", "throat.radius", "pore.radius"} <= network.keys():
            return

        throat_conns = np.asarray(network["throat.conns"])
        throat_radius = np.asarray(network["throat.radius"], dtype=float)
        pore_radius = np.asarray(network["pore.radius"], dtype=float)

        # mean radius of the throats incident to each pore
        incident_pores = throat_conns.reshape(-1).astype(int)  # both endpoints of every throat
        incident_radius = np.repeat(throat_radius, 2)
        radius_sum = np.zeros(n_pores, dtype=float)
        throat_count = np.zeros(n_pores, dtype=float)
        np.add.at(radius_sum, incident_pores, incident_radius)
        np.add.at(throat_count, incident_pores, 1.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_throat_radius = np.where(throat_count > 0, radius_sum / throat_count, np.nan)
            network["pore.aspect_ratio"] = pore_radius / mean_throat_radius
