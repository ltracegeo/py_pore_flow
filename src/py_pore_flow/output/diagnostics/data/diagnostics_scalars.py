import numpy as np

from py_pore_flow.output.diagnostics.utils.diagnostics_io import (
    DiagnosticsStateLoader,
    DiagnosticsStorer,
)

STEP_SCALARS_NAME = "step_scalars"
STEP_SCALARS_NDATA = 4


class DiagnosticsScalars(DiagnosticsStorer):
    def __init__(self, path):
        super().__init__(STEP_SCALARS_NAME, 1, STEP_SCALARS_NDATA, np.float64, path)

    def save_state(self, pc, sw, krw, kro):
        super()._save_state([pc, sw, krw, kro])


class DiagnosticsScalarsStep(DiagnosticsStateLoader):
    def __init__(self, path, step, cache=None):
        super().__init__(path, step, 1, cache)

        floats = self.load_state(STEP_SCALARS_NAME, STEP_SCALARS_NDATA, np.float64)
        self.pc = self.data_view(floats, 0)[0]
        self.sw = self.data_view(floats, 1)[0]
        self.krw = self.data_view(floats, 2)[0]
        self.kro = self.data_view(floats, 3)[0]

    def __str__(self):
        return f"DiagnosticsScalarsStep (step: {self.step} - pc: {self.pc},  sw: {self.sw})"


class DiagnosticsScalarsCycle(DiagnosticsStateLoader):
    def __init__(self, path, cycle_offset: int, cycle_n_steps: int, cache=None):
        super().__init__(path, 0, 1, cache)

        cycle_offset_end = cycle_offset + cycle_n_steps
        scalar_floats = self.load_state(STEP_SCALARS_NAME, STEP_SCALARS_NDATA, np.float64)

        self.pc = scalar_floats[cycle_offset:cycle_offset_end, 0]
        self.sw = scalar_floats[cycle_offset:cycle_offset_end, 1]
        self.krw = scalar_floats[cycle_offset:cycle_offset_end, 2]
        self.kro = scalar_floats[cycle_offset:cycle_offset_end, 3]

    def __str__(self):
        return f"DiagnosticsScalarsCycle (cycle: {self.step})"
