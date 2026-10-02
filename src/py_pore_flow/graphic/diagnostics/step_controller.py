from PySide2.QtCore import QObject, Signal
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QComboBox, QLabel, QSpinBox

CYCLE_NAMES = ["Drainage", "Imbibition", "2nd Drainage"]


class StepController(QObject):
    cycleChanged = Signal(int)
    stepChanged = Signal(int)

    def __init__(self, diagnostics=None):
        super().__init__()

        self.cycle_combobox = QComboBox()
        for name in CYCLE_NAMES:
            self.cycle_combobox.addItem(name)
        self.cycle_combobox.currentIndexChanged.connect(self._on_cycle_changed)

        self.step_input = QSpinBox()
        self.step_input.valueChanged.connect(self._on_step_changed)

        self.status_label = QLabel()
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        self.status_label.setFont(font)

        self.diagnostics = None
        if diagnostics is not None:
            self.set_diagnostics(diagnostics)

    def set_diagnostics(self, diagnostics):
        self.diagnostics = diagnostics
        cycle_id = self.cycle_combobox.currentIndex()
        self._on_cycle_changed(cycle_id)

    def get_cycle_widget(self):
        return self.cycle_combobox

    def get_step_widget(self):
        return self.step_input

    def get_status_widget(self):
        return self.status_label

    def get_cycle(self) -> int:
        return self.cycle_combobox.currentIndex()

    def get_step(self) -> int:
        return self.step_input.value()

    def _on_cycle_changed(self, cycle_id):
        self.step_input.blockSignals(True)
        self.step_input.setMinimum(0)
        if cycle_id >= self.diagnostics.info.n_cycles:
            self.step_input.setMaximum(0)
        else:
            n_steps = self.diagnostics.info.cycle_n_steps[cycle_id]
            self.step_input.setMaximum(max(0, n_steps - 1))
        self.step_input.blockSignals(False)

        self._update_status_label()
        self.cycleChanged.emit(cycle_id)

    def _on_step_changed(self, step):
        self._update_status_label()
        self.stepChanged.emit(step)

    def _update_status_label(self):
        cycle_id = self.get_cycle()
        step = self.get_step()
        n_steps = self.diagnostics.info.cycle_n_steps[cycle_id]

        pc = self.diagnostics.cycle[cycle_id][step].pc
        sw = self.diagnostics.cycle[cycle_id][step].sw
        krw = self.diagnostics.cycle[cycle_id][step].krw
        kro = self.diagnostics.cycle[cycle_id][step].kro
        self.status_label.setText(
            f"{self.cycle_combobox.currentText()}: step {step:3} / {max(0, n_steps - 1)} - "
            + f"Pc: {pc:.2f}, Sw: {sw:.2f}, Krw: {krw:.2f}, Kro: {kro:.2f}"
        )
