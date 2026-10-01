import numpy as np

from py_pore_flow.output.diagnostics.utils.diagnostics_io import (
    DiagnosticsStateLoader,
    DiagnosticsStorer,
)

THROATS_CYCLE_FLOATS_NAME = "throats_cycle_floats"
THROATS_CYCLE_FLOATS_NDATA = 3
THROATS_CYCLE_INTS_NAME = "throats_cycle_ints"
THROATS_CYCLE_INTS_NDATA = 1


class DiagnosticsThroatsCycleFloats(DiagnosticsStorer):
    def __init__(self, n_throats, path):
        super().__init__(THROATS_CYCLE_FLOATS_NAME, n_throats, THROATS_CYCLE_FLOATS_NDATA, np.float64, path)

    def save_state(self, throat):
        super()._save_state([throat.entry_pressure, throat.advancing_contact_angle, throat.receding_contact_angle])


class DiagnosticsThroatsCycleInts(DiagnosticsStorer):
    def __init__(self, n_throats, path):
        super().__init__(THROATS_CYCLE_INTS_NAME, n_throats, THROATS_CYCLE_INTS_NDATA, np.int8, path)

    def save_state(self, throat):
        super()._save_state([throat.invasion_method])


class DiagnosticsThroatsCycle(DiagnosticsStateLoader):
    def __init__(self, path, cycle, n_throats, cache=None):
        super().__init__(path, cycle, n_throats, cache)

        data = self.load_state(THROATS_CYCLE_FLOATS_NAME, THROATS_CYCLE_FLOATS_NDATA, np.float64)
        self.entry_pressure = self.data_view(data, 0)
        self.advancing_contact_angle = self.data_view(data, 1)
        self.receding_contact_angle = self.data_view(data, 2)

        data = self.load_state(THROATS_CYCLE_INTS_NAME, THROATS_CYCLE_INTS_NDATA, np.int8)
        self.invasion_method = self.data_view(data, 0)

    def __str__(self):
        return f"DiagnosticsThroatsCycle (cycle: {self.step}, throats: {self.n_elements})"
