import numpy as np

from py_pore_flow.output.diagnostics.utils.diagnostics_io import (
    DiagnosticsStateLoader,
    DiagnosticsStorer,
)

PORES_CYCLE_FLOATS_NAME = "pores_cycle_floats"
PORES_CYCLE_FLOATS_NDATA = 3
PORES_CYCLE_INTS_NAME = "pores_cycle_ints"
PORES_CYCLE_INTS_NDATA = 1


class DiagnosticsPoresCycleFloats(DiagnosticsStorer):
    def __init__(self, n_pores, path):
        super().__init__(PORES_CYCLE_FLOATS_NAME, n_pores, PORES_CYCLE_FLOATS_NDATA, np.float64, path)

    def save_state(self, pore):
        super()._save_state([pore.entry_pressure, pore.advancing_contact_angle, pore.receding_contact_angle])


class DiagnosticsPoresCycleInts(DiagnosticsStorer):
    def __init__(self, n_pores, path):
        super().__init__(PORES_CYCLE_INTS_NAME, n_pores, PORES_CYCLE_INTS_NDATA, np.int8, path)

    def save_state(self, pore):
        super()._save_state([pore.invasion_method])


class DiagnosticsPoresCycle(DiagnosticsStateLoader):
    def __init__(self, path, cycle, n_pores, cache=None):
        super().__init__(path, cycle, n_pores, cache)

        data = self.load_state(PORES_CYCLE_FLOATS_NAME, PORES_CYCLE_FLOATS_NDATA, np.float64)
        self.entry_pressure = self.data_view(data, 0)
        self.advancing_contact_angle = self.data_view(data, 1)
        self.receding_contact_angle = self.data_view(data, 2)

        data = self.load_state(PORES_CYCLE_INTS_NAME, PORES_CYCLE_INTS_NDATA, np.int8)
        self.invasion_method = self.data_view(data, 0)

    def __str__(self):
        return f"DiagnosticsPoresCycle (cycle: {self.step}, pores: {self.n_elements})"
