from typing import ClassVar

from PySide2.QtCore import QAbstractTableModel, QEvent, QModelIndex, Qt, QTimer, Signal
from PySide2.QtGui import QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from py_pore_flow.graphic.diagnostics.diagnostics_crossplot_widget import CrossPlotWidget
from py_pore_flow.graphic.diagnostics.diagnostics_element_widget import DiagnosticsElementWidget
from py_pore_flow.graphic.diagnostics.diagnostics_heatmap_widget import HeatmapWidget
from py_pore_flow.graphic.diagnostics.diagnostics_network_widget import DiagnosticsNetworkWidget
from py_pore_flow.graphic.diagnostics.diagnostics_properties import DiagnosticsProperties

CYCLE_NAMES = ["Drainage", "Imbibition", "2nd Drainage"]


class CheckableComboBox(QComboBox):
    itemToggled = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setEditable(True)
        self.setCompleter(None)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setText("Columns")
        self.view().viewport().installEventFilter(self)
        self.model().itemChanged.connect(self._on_item_changed)

    def set_items(self, names, checked_names):
        checked = set(checked_names)
        model = self.model()
        model.blockSignals(True)  # avoid itemToggled while (re)populating
        model.clear()
        for name in names:
            item = QStandardItem(name)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if name in checked else Qt.Unchecked)
            model.appendRow(item)
        model.blockSignals(False)
        self.lineEdit().setText("Columns")

    def eventFilter(self, watched, event):
        if watched is self.view().viewport() and event.type() == QEvent.MouseButtonRelease:
            item = self.model().itemFromIndex(self.view().indexAt(event.pos()))
            if item is not None:
                item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
            return True
        return super().eventFilter(watched, event)

    def _on_item_changed(self, item):
        self.lineEdit().setText("Columns")
        self.itemToggled.emit(item.text(), item.checkState() == Qt.Checked)


class DiagnosticsTableModel(QAbstractTableModel):
    """Reads cell values straight from the per-step memmap views.

    The table used to be a QTableWidget, which allocates one QTableWidgetItem per cell: a
    134k throat network with 9 columns meant 1.2 million Qt objects on every step change.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n_rows = 0
        self._columns = []
        self._values = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._n_rows

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._columns[section]
        return section

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or not index.isValid():
            return None
        values = self._values[index.column()]
        if values is None:
            return None
        return str(values[index.row()])

    def set_content(self, n_rows, columns, values):
        self.beginResetModel()
        self._n_rows = n_rows
        self._columns = list(columns)
        self._values = list(values)
        self.endResetModel()


class DiagnosticsTableWidget(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._properties = None
        self._element_type = None
        self._cycle_step = None
        self._n_elements = 0
        self._shown = []
        self._model = DiagnosticsTableModel(self)
        self.setModel(self._model)
        self.column_selector = CheckableComboBox()
        self.column_selector.itemToggled.connect(self._set_column_shown)
        self.horizontalHeader().setContextMenuPolicy(Qt.NoContextMenu)
        self.horizontalHeader().viewport().installEventFilter(self)
        self.setWordWrap(False)

    def eventFilter(self, watched, event):
        if (
            watched is self.horizontalHeader().viewport()
            and event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.RightButton
        ):
            QTimer.singleShot(0, self.column_selector.showPopup)
            return True
        return super().eventFilter(watched, event)

    def set_source(self, properties, element_type, n_elements, default_columns, cycle_step_getter):
        self._properties = properties
        self._element_type = element_type
        self._n_elements = n_elements
        self._cycle_step = cycle_step_getter
        available = self._available_columns()
        self._shown = [name for name in default_columns if name in set(available)]
        self.column_selector.set_items(available, self._shown)

    def refresh(self):
        values = [self._column_values(name) for name in self._shown]
        self._model.set_content(self._n_elements, self._shown, values)

    def _available_columns(self):
        if self._properties is None:
            return []
        if self._element_type == "pore":
            return self._properties.get_pore_property_names()
        return self._properties.get_throat_property_names()

    def _column_values(self, name):
        cycle_id, step = self._cycle_step()
        try:
            if self._element_type == "pore":
                return self._properties.get_pore_values_in_step(name, cycle_id=cycle_id, step=step)
            return self._properties.get_throat_values_in_step(name, cycle_id=cycle_id, step=step)
        except (KeyError, IndexError):
            return None

    def _set_column_shown(self, name, shown):
        if shown and name not in self._shown:
            self._shown.append(name)
        elif not shown and name in self._shown:
            self._shown.remove(name)
        else:
            return
        self.refresh()


class DiagnosticsTab(QWidget):
    PORE_COLUMNS: ClassVar[list[str]] = [
        "water_saturation",
        "oil_multiphase_conductivity",
        "water_multiphase_conductivity",
        "water_filled",
        "trapped",
        "corner_water_layer",
        "corner_oil_layer",
    ]

    THROAT_COLUMNS: ClassVar[list[str]] = [
        "water_saturation",
        "oil_multiphase_conductivity",
        "water_multiphase_conductivity",
        "oil_multiphase_conductance",
        "water_multiphase_conductance",
        "water_filled",
        "trapped",
        "corner_water_layer",
        "corner_oil_layer",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cycle_combo = QComboBox()
        self.cycle_combo.addItems(CYCLE_NAMES)

        self.step_spinbox = QSpinBox()

        self.table = DiagnosticsTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.cycle_combo)
        top_bar.addWidget(QLabel("Step:"))
        top_bar.addWidget(self.step_spinbox)
        top_bar.addStretch()
        top_bar.addWidget(self.table.column_selector)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def set_diagnostics(self, diagnostics, element_type):
        self._diagnostics = diagnostics
        properties = DiagnosticsProperties(diagnostics)
        if element_type == "pore":
            n_elements = diagnostics.info.n_pores
            default_columns = self.PORE_COLUMNS
        else:
            n_elements = diagnostics.info.n_throats
            default_columns = self.THROAT_COLUMNS

        self.table.set_source(
            properties,
            element_type,
            n_elements,
            default_columns,
            lambda: (self.cycle_combo.currentIndex(), self.step_spinbox.value()),
        )

        self.cycle_combo.currentIndexChanged.connect(self._on_cycle_changed)
        self.step_spinbox.valueChanged.connect(self._on_step_changed)

        self._update_step_range()
        self.table.refresh()

    def _on_cycle_changed(self, _):
        self._update_step_range()
        self.table.refresh()

    def _on_step_changed(self, _):
        self.table.refresh()

    def _update_step_range(self):
        cycle_id = self.cycle_combo.currentIndex()
        self.step_spinbox.blockSignals(True)
        self.step_spinbox.setMinimum(0)
        if cycle_id >= self._diagnostics.info.n_cycles:
            self.step_spinbox.setMaximum(0)
        else:
            n_steps = self._diagnostics.info.cycle_n_steps[cycle_id]
            self.step_spinbox.setMaximum(max(0, n_steps - 1))
        self.step_spinbox.blockSignals(False)


class NetworkTab(QWidget):
    def __init__(self, diagnostics=None, parent=None):
        super().__init__(parent)

        self.element_widget = None
        self.network_widget = None

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        if diagnostics is not None:
            self.set_diagnostics(diagnostics)

    def set_diagnostics(self, diagnostics):
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

        diagnostics_properties = DiagnosticsProperties(diagnostics)
        self.element_widget = DiagnosticsElementWidget(diagnostics, diagnostics_properties)
        self.network_widget = DiagnosticsNetworkWidget(diagnostics_properties)
        self.network_widget.poreClicked.connect(self.element_widget.select_pore)
        self.network_widget.throatClicked.connect(self.element_widget.select_throat)
        self.element_widget.poreSelected.connect(self.network_widget.highlight_pore)
        self.element_widget.throatSelected.connect(self.network_widget.highlight_throat)
        self.element_widget.networkChanged.connect(self.network_widget.refresh_colors)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.network_widget)
        splitter.addWidget(self.element_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self._layout.addWidget(splitter)


class CrossPlotTab(QWidget):
    def __init__(self, diagnostics=None, parent=None):
        super().__init__(parent)

        self.crossplot_widget = None

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        if diagnostics is not None:
            self.set_diagnostics(diagnostics)

    def set_diagnostics(self, diagnostics):
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

        self.crossplot_widget = CrossPlotWidget(diagnostics)
        self._layout.addWidget(self.crossplot_widget)


class HeatmapTab(QWidget):
    def __init__(self, diagnostics=None, aggregates=None, parent=None):
        super().__init__(parent)

        self.heatmap_widget = None

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        if diagnostics is not None or aggregates is not None:
            self.set_diagnostics(diagnostics, aggregates)

    def set_diagnostics(self, diagnostics, aggregates=None):
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

        self.heatmap_widget = HeatmapWidget(diagnostics, aggregates)
        self._layout.addWidget(self.heatmap_widget)
