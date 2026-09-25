import numpy as np
from PySide2.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QTabWidget,
    QWidget,
)

from py_pore_flow.graphic.diagnostics.diagnostics_tabs import (
    CrossPlotTab,
    DiagnosticsTab,
    ElementGeometryTab,
    HeatmapTab,
)
from py_pore_flow.output.diagnostics import Diagnostics
from py_pore_flow.output.diagnostics_aggregate import DiagnosticsAggregates, LEVEL_BASIC, diagnostics_level


class DiagnosticsWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._diagnostics = None
        self._aggregates = None

        self.tabs = QTabWidget()
        layout = QHBoxLayout(self)
        layout.addWidget(self.tabs)

    def deleteLater(self):
        self.__close_vtk_window()
        self.__close_diagnostics()
        super().deleteLater()

    def __clear_tabs(self):
        while self.tabs.count():
            tab = self.tabs.widget(0)
            self.tabs.removeTab(0)
            tab.deleteLater()

    def __close_diagnostics(self):
        """Release the memmapped chunk files: the directory is moved on project save."""
        if self._diagnostics is not None:
            self._diagnostics.close()
            self._diagnostics = None

    def __close_vtk_window(self):
        network_container_widget = self.findChild(QWidget, "network_container_widget")
        if network_container_widget is not None:
            network_vtk_window = network_container_widget.property("embeddedVtk")
            if network_vtk_window is not None:
                network_vtk_window.close()

    def set_diagnostics(self, diagnostics_path):
        """Build the views the diagnostics level of `diagnostics_path` can feed.

        A basic run only stores the heatmap aggregates, so the element-level views have no
        data to show and the embedded VTK network is never built.
        """
        self.__close_diagnostics()
        self.__clear_tabs()

        if diagnostics_level(diagnostics_path) == LEVEL_BASIC:
            self._aggregates = DiagnosticsAggregates(diagnostics_path)
            self.heatmap_tab = HeatmapTab()
            self.heatmap_tab.set_diagnostics(None, self._aggregates)
            self.tabs.addTab(self.heatmap_tab, "Heatmap")
            return

        self._diagnostics = Diagnostics(diagnostics_path)
        self._ensure_numpy_network(self._diagnostics.network)
        self._ensure_pore_conns(self._diagnostics.network, self._diagnostics.info.n_pores)
        self._ensure_pore_aspect_ratio(self._diagnostics.network, self._diagnostics.info.n_pores)

        self.pores_tab = DiagnosticsTab()
        self.throats_tab = DiagnosticsTab()
        self.element_tab = ElementGeometryTab()
        self.crossplot_tab = CrossPlotTab()
        self.heatmap_tab = HeatmapTab()

        self.pores_tab.set_diagnostics(self._diagnostics, "pore")
        self.throats_tab.set_diagnostics(self._diagnostics, "throat")
        self.element_tab.set_diagnostics(self._diagnostics)
        self.crossplot_tab.set_diagnostics(self._diagnostics)
        self.heatmap_tab.set_diagnostics(self._diagnostics)

        self.tabs.addTab(self.pores_tab, "Pores")
        self.tabs.addTab(self.throats_tab, "Throats")
        self.tabs.addTab(self.element_tab, "Element")
        self.tabs.addTab(self.crossplot_tab, "Cross Plot")
        self.tabs.addTab(self.heatmap_tab, "Heatmap")

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
