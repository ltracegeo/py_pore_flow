import numpy as np

from py_pore_flow.output.diagnostics.utils.diagnostics_io import (
    CORNER_SHAPE,
    DiagnosticsStateLoader,
    DiagnosticsStorer,
)

PORES_FLOATS_NAME = "pores_floats"
PORES_FLOATS_NDATA = 5
PORES_BOOLS_NAME = "pores_bools"
PORES_BOOLS_NDATA = 3
PORES_CORNERS_BOOLS_NAME = "pores_corners_bools"
PORES_CORNERS_BOOLS_NDATA = 2
PORES_CORNERS_FLOATS_NAME = "pores_corners_floats"
PORES_CORNERS_FLOATS_NDATA = 4


class DiagnosticsPoresFloats(DiagnosticsStorer):
    def __init__(self, n_pores, path):
        super().__init__(PORES_FLOATS_NAME, n_pores, PORES_FLOATS_NDATA, np.float64, path)

    def save_state(self, pore):
        super()._save_state(
            [
                pore.water_saturation,
                pore.oil_multiphase_conductivity,
                pore.water_multiphase_conductivity,
                pore.oil_pressure,
                pore.water_pressure,
            ]
        )


class DiagnosticsPoresBools(DiagnosticsStorer):
    def __init__(self, n_pores, path):
        super().__init__(PORES_BOOLS_NAME, n_pores, PORES_BOOLS_NDATA, np.bool_, path)

    def save_state(self, pore):
        super()._save_state([pore.water_filled, pore.trapped, pore.filled])


class DiagnosticsPoresCornersBools(DiagnosticsStorer):
    def __init__(self, n_pores, path):
        super().__init__(
            PORES_CORNERS_BOOLS_NAME, n_pores, PORES_CORNERS_BOOLS_NDATA, np.bool_, path, item_shape=CORNER_SHAPE
        )

    def save_state(self, pore):
        super()._save_state([pore.corner_water_layer, pore.corner_oil_layer])


class DiagnosticsPoresCornersFloats(DiagnosticsStorer):
    def __init__(self, n_pores, path):
        super().__init__(
            PORES_CORNERS_FLOATS_NAME, n_pores, PORES_CORNERS_FLOATS_NDATA, np.float64, path, item_shape=CORNER_SHAPE
        )

    def save_state(self, pore):
        super()._save_state([pore.b, pore.bo, pore.hinging_contact_angle, pore.hinging_contact_angle_o])


class DiagnosticsPoresStep(DiagnosticsStateLoader):
    def __init__(self, path, step, n_pores, cache=None):
        super().__init__(path, step, n_pores, cache)

        floats = self.load_state(PORES_FLOATS_NAME, PORES_FLOATS_NDATA, np.float64)
        self.water_saturation = self.data_view(floats, 0)
        self.oil_multiphase_conductivity = self.data_view(floats, 1)
        self.water_multiphase_conductivity = self.data_view(floats, 2)
        self.oil_pressure = self.data_view(floats, 3)
        self.water_pressure = self.data_view(floats, 4)

        corners_bools = self.load_state(PORES_CORNERS_BOOLS_NAME, PORES_CORNERS_BOOLS_NDATA, np.bool_, CORNER_SHAPE)
        self.corner_water_layer = self.data_view(corners_bools, 0, CORNER_SHAPE)
        self.corner_oil_layer = self.data_view(corners_bools, 1, CORNER_SHAPE)

        bools = self.load_state(PORES_BOOLS_NAME, PORES_BOOLS_NDATA, np.bool_)
        self.water_filled = self.data_view(bools, 0)
        self.trapped = self.data_view(bools, 1)
        self.filled = self.data_view(bools, 2)

        corners_floats = self.load_state(
            PORES_CORNERS_FLOATS_NAME, PORES_CORNERS_FLOATS_NDATA, np.float64, CORNER_SHAPE
        )
        self.b = self.data_view(corners_floats, 0, CORNER_SHAPE)
        self.bo = self.data_view(corners_floats, 1, CORNER_SHAPE)
        self.hinging_contact_angle = self.data_view(corners_floats, 2, CORNER_SHAPE)
        self.hinging_contact_angle_o = self.data_view(corners_floats, 3, CORNER_SHAPE)

    def __str__(self):
        return f"DiagnosticsPoresStep (step: {self.step}, pores: {self.n_elements})"
