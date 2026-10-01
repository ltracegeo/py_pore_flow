import numpy as np

from py_pore_flow.output.diagnostics.utils.diagnostics_io import (
    CORNER_SHAPE,
    DiagnosticsStateLoader,
    DiagnosticsStorer,
)

THROATS_FLOATS_NAME = "throats_floats"
THROATS_FLOATS_NDATA = 9
THROATS_BOOLS_NAME = "throats_bools"
THROATS_BOOLS_NDATA = 3
THROATS_CORNERS_BOOLS_NAME = "throats_corners_bools"
THROATS_CORNERS_BOOLS_NDATA = 2
THROATS_CORNERS_FLOATS_NAME = "throats_corners_floats"
THROATS_CORNERS_FLOATS_NDATA = 4


class DiagnosticsThroatsFloats(DiagnosticsStorer):
    def __init__(self, n_throats, path):
        super().__init__(THROATS_FLOATS_NAME, n_throats, THROATS_FLOATS_NDATA, np.float64, path)

    def save_state(self, throat):
        super()._save_state(
            [
                throat.water_saturation,
                throat.oil_multiphase_conductivity,
                throat.water_multiphase_conductivity,
                throat.oil_multiphase_conductance,
                throat.water_multiphase_conductance,
                throat.oil_pressure,
                throat.water_pressure,
                throat.oil_flow_rate,
                throat.water_flow_rate,
            ]
        )


class DiagnosticsThroatsBools(DiagnosticsStorer):
    def __init__(self, n_throats, path):
        super().__init__(THROATS_BOOLS_NAME, n_throats, THROATS_BOOLS_NDATA, np.bool_, path)

    def save_state(self, throat):
        super()._save_state([throat.water_filled, throat.trapped, throat.filled])


class DiagnosticsThroatsCornersBools(DiagnosticsStorer):
    def __init__(self, n_throats, path):
        super().__init__(
            THROATS_CORNERS_BOOLS_NAME, n_throats, THROATS_CORNERS_BOOLS_NDATA, np.bool_, path, item_shape=CORNER_SHAPE
        )

    def save_state(self, throat):
        super()._save_state([throat.corner_water_layer, throat.corner_oil_layer])


class DiagnosticsThroatsCornersFloats(DiagnosticsStorer):
    def __init__(self, n_throats, path):
        super().__init__(
            THROATS_CORNERS_FLOATS_NAME,
            n_throats,
            THROATS_CORNERS_FLOATS_NDATA,
            np.float64,
            path,
            item_shape=CORNER_SHAPE,
        )

    def save_state(self, throat):
        super()._save_state([throat.b, throat.bo, throat.hinging_contact_angle, throat.hinging_contact_angle_o])


class DiagnosticsThroatsStep(DiagnosticsStateLoader):
    def __init__(self, path, step, n_throats, cache=None):
        super().__init__(path, step, n_throats, cache)

        floats = self.load_state(THROATS_FLOATS_NAME, THROATS_FLOATS_NDATA, np.float64)
        self.water_saturation = self.data_view(floats, 0)
        self.oil_multiphase_conductivity = self.data_view(floats, 1)
        self.water_multiphase_conductivity = self.data_view(floats, 2)
        self.oil_multiphase_conductance = self.data_view(floats, 3)
        self.water_multiphase_conductance = self.data_view(floats, 4)
        self.oil_pressure = self.data_view(floats, 5)
        self.water_pressure = self.data_view(floats, 6)
        self.oil_flow_rate = self.data_view(floats, 7)
        self.water_flow_rate = self.data_view(floats, 8)

        corners_bools = self.load_state(THROATS_CORNERS_BOOLS_NAME, THROATS_CORNERS_BOOLS_NDATA, np.bool_, CORNER_SHAPE)
        self.corner_water_layer = self.data_view(corners_bools, 0, CORNER_SHAPE)
        self.corner_oil_layer = self.data_view(corners_bools, 1, CORNER_SHAPE)

        bools = self.load_state(THROATS_BOOLS_NAME, THROATS_BOOLS_NDATA, np.bool_)
        self.water_filled = self.data_view(bools, 0)
        self.trapped = self.data_view(bools, 1)
        self.filled = self.data_view(bools, 2)

        corners_floats = self.load_state(
            THROATS_CORNERS_FLOATS_NAME, THROATS_CORNERS_FLOATS_NDATA, np.float64, CORNER_SHAPE
        )
        self.b = self.data_view(corners_floats, 0, CORNER_SHAPE)
        self.bo = self.data_view(corners_floats, 1, CORNER_SHAPE)
        self.hinging_contact_angle = self.data_view(corners_floats, 2, CORNER_SHAPE)
        self.hinging_contact_angle_o = self.data_view(corners_floats, 3, CORNER_SHAPE)

    def __str__(self):
        return f"DiagnosticsThroatsStep (step: {self.step}, throats: {self.n_elements})"
