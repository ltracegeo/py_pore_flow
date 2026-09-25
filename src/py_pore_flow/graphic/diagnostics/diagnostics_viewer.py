from PySide2.QtWidgets import QMainWindow

PORE = 0
THROAT = 1


class DiagnosticsViewer(QMainWindow):
    def __init__(self, diagnostics_path):
        super().__init__()
        from py_pore_flow.graphic.diagnostics.diagnostics_widget import DiagnosticsWidget

        self.diagnostics_widget = DiagnosticsWidget()
        self.diagnostics_widget.set_diagnostics(diagnostics_path)

        self.setCentralWidget(self.diagnostics_widget)

    def closeEvent(self, event):
        self.diagnostics_widget.deleteLater()


class ElementTabManager:
    def __init__(self, tab_widget, columns, diagnostics, element_type):
        self.tab_widget = tab_widget
        self.columns = columns
        self.diagnostics = diagnostics
        self.element_type = element_type

        self.tab_widget.table.setColumnCount(len(self.columns))
        self.tab_widget.table.setHorizontalHeaderLabels(self.columns)
        self.tab_widget.table.setRowCount(self.diagnostics.info.n_pores)

        self.tab_widget.cycle_combo.currentIndexChanged.connect(self._on_cycle_changed)
        self.tab_widget.step_spinbox.valueChanged.connect(self._on_step_changed)

        self._update_step_range()
        self._update_table()

    def _on_cycle_changed(self, _):
        self._update_step_range()
        self._update_table()

    def _on_step_changed(self, _):
        self._update_table()

    def _update_step_range(self):
        cycle_id = self.tab_widget.cycle_combo.currentIndex()
        if cycle_id >= self.diagnostics.info.n_cycles:
            self.tab_widget.step_spinbox.setMaximum(0)
            return
        n_steps = self.diagnostics.info.cycle_n_steps[cycle_id]
        self.tab_widget.step_spinbox.setMinimum(0)
        self.tab_widget.step_spinbox.setMaximum(max(0, n_steps - 1))

    def _update_table(self):
        from PySide2.QtWidgets import QTableWidgetItem

        cycle_id = self.tab_widget.cycle_combo.currentIndex()
        step = self.tab_widget.step_spinbox.value()

        if cycle_id >= self.diagnostics.info.n_cycles:
            return

        if self.element_type == PORE:
            element = self.diagnostics.cycle[cycle_id][step].pore
        else:
            element = self.diagnostics.cycle[cycle_id][step].throat

        table = self.tab_widget.table
        table.setUpdatesEnabled(False)
        for col, attr in enumerate(self.columns):
            values = getattr(element, attr)
            for row, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(str(value)))
        table.setUpdatesEnabled(True)
