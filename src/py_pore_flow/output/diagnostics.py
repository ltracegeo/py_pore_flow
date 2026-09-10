import json
import os
import re

import numpy as np

from py_pore_flow.equations.geometry import MAX_NUMBER_OF_CORNERS

FILE_EXTENSION = ".dat"
INFO_FILE_NAME = "info"
NETWORK_JSON_NAME = "network.json"

# trailing per-element shape of the corner layer arrays (one value per corner)
CORNER_SHAPE = (MAX_NUMBER_OF_CORNERS,)

THROATS_FLOATS_NAME = "throats_floats"
THROATS_FLOATS_NDATA = 5
THROATS_BOOLS_NAME = "throats_bools"
THROATS_BOOLS_NDATA = 3
THROATS_CORNERS_BOOL_NAME = "throats_corners_bools"
THROATS_CORNERS_BOOL_NDATA = 2
THROATS_CORNERS_FLOAT_NAME = "throats_corners_floats"
THROATS_CORNERS_FLOAT_NDATA = 4
THROATS_CYCLE_FLOATS_NAME = "throats_cycle_floats"
THROATS_CYCLE_FLOATS_NDATA = 3
THROATS_CYCLE_INTS_NAME = "throats_cycle_ints"
THROATS_CYCLE_INTS_NDATA = 1
PORES_FLOATS_NAME = "pores_floats"
PORES_FLOATS_NDATA = 5
PORES_BOOLS_NAME = "pores_bools"
PORES_BOOLS_NDATA = 3
PORES_CORNERS_BOOL_NAME = "pores_corners_bools"
PORES_CORNERS_BOOL_NDATA = 2
PORES_CORNERS_FLOAT_NAME = "pores_corners_floats"
PORES_CORNERS_FLOAT_NDATA = 4
PORES_CYCLE_FLOATS_NAME = "pores_cycle_floats"
PORES_CYCLE_FLOATS_NDATA = 3
PORES_CYCLE_INTS_NAME = "pores_cycle_ints"
PORES_CYCLE_INTS_NDATA = 1


class DiagnosticsWriter:
    def __init__(self, network, path="."):
        self.pn = network.pn_core

        self.cycle_id = 0
        self.n_saved_sates = 0

        n_throats = self.pn.throat.water_saturation.size
        n_pores = self.pn.pore.water_saturation.size

        self.info_storer = DiagnosticsInfoStorer(path)
        self.info_storer.save_number_of_elements(n_throats, n_pores)
        self.storer_throats_floats = DiagnosticsStorer(THROATS_FLOATS_NAME, n_throats, THROATS_FLOATS_NDATA, np.float64, path)
        self.storer_throats_bools = DiagnosticsStorer(
            THROATS_BOOLS_NAME, n_throats, THROATS_BOOLS_NDATA, np.bool_, path
        )
        self.storer_throats_corners_bool = DiagnosticsStorer(
            THROATS_CORNERS_BOOL_NAME, n_throats, THROATS_CORNERS_BOOL_NDATA, np.bool_, path, item_shape=CORNER_SHAPE
        )
        self.storer_throats_corners_float = DiagnosticsStorer(
            THROATS_CORNERS_FLOAT_NAME,
            n_throats,
            THROATS_CORNERS_FLOAT_NDATA,
            np.float64,
            path,
            item_shape=CORNER_SHAPE,
        )
        self.storer_throats_cycle_floats = DiagnosticsStorer(
            THROATS_CYCLE_FLOATS_NAME, n_throats, THROATS_CYCLE_FLOATS_NDATA, np.float64, path
        )
        self.storer_throats_cycle_ints = DiagnosticsStorer(
            THROATS_CYCLE_INTS_NAME, n_throats, THROATS_CYCLE_INTS_NDATA, np.int8, path
        )
        self.storer_pores = DiagnosticsStorer(PORES_FLOATS_NAME, n_pores, PORES_FLOATS_NDATA, np.float64, path)
        self.storer_pores_bools = DiagnosticsStorer(PORES_BOOLS_NAME, n_pores, PORES_BOOLS_NDATA, np.bool_, path)
        self.storer_pores_corners_bool = DiagnosticsStorer(
            PORES_CORNERS_BOOL_NAME, n_pores, PORES_CORNERS_BOOL_NDATA, np.bool_, path, item_shape=CORNER_SHAPE
        )
        self.storer_pores_corners_float = DiagnosticsStorer(
            PORES_CORNERS_FLOAT_NAME, n_pores, PORES_CORNERS_FLOAT_NDATA, np.float64, path, item_shape=CORNER_SHAPE
        )
        self.storer_pores_cycle_floats = DiagnosticsStorer(
            PORES_CYCLE_FLOATS_NAME, n_pores, PORES_CYCLE_FLOATS_NDATA, np.float64, path
        )
        self.storer_pores_cycle_ints = DiagnosticsStorer(
            PORES_CYCLE_INTS_NAME, n_pores, PORES_CYCLE_INTS_NDATA, np.int8, path
        )

        self._export_pn_to_json(network.pn, path + "/" + NETWORK_JSON_NAME)

    # stores elements properties of a single step
    def save_state(self):
        self.n_saved_sates += 1

        throat = self.pn.throat
        pore = self.pn.pore

        self.storer_throats_floats.save_state(
            [
                throat.water_saturation,
                throat.oil_multiphase_conductivity,
                throat.water_multiphase_conductivity,
                throat.oil_multiphase_conductance,
                throat.water_multiphase_conductance,
            ]
        )
        self.storer_throats_bools.save_state([throat.water_filled, throat.trapped, throat.filled])
        self.storer_throats_corners_bool.save_state([throat.corner_water_layer, throat.corner_oil_layer])
        self.storer_throats_corners_float.save_state(
            [throat.b, throat.bo, throat.hinging_contact_angle, throat.hinging_contact_angle_o]
        )
        self.storer_pores.save_state(
            [
                pore.water_saturation,
                pore.oil_multiphase_conductivity,
                pore.water_multiphase_conductivity,
                pore.oil_flow_rate,
                pore.water_flow_rate,
            ]
        )
        self.storer_pores_bools.save_state([pore.water_filled, pore.trapped, pore.filled])
        self.storer_pores_corners_bool.save_state([pore.corner_water_layer, pore.corner_oil_layer])
        self.storer_pores_corners_float.save_state(
            [pore.b, pore.bo, pore.hinging_contact_angle, pore.hinging_contact_angle_o]
        )

    # stores some elements properties at the end of a cycle
    def save_cycle_data(self):
        self.info_storer.save_number_of_steps(self.cycle_id, self.n_saved_sates)
        self.cycle_id += 1
        self.n_saved_sates = 0

        throat = self.pn.throat
        pore = self.pn.pore

        self.storer_throats_cycle_floats.save_state(
            [throat.entry_pressure, throat.advancing_contact_angle, throat.receding_contact_angle]
        )
        self.storer_throats_cycle_ints.save_state([throat.invasion_method])
        self.storer_pores_cycle_floats.save_state(
            [pore.entry_pressure, pore.advancing_contact_angle, pore.receding_contact_angle]
        )
        self.storer_pores_cycle_ints.save_state([pore.invasion_method])

    def close(self):
        self.storer_throats_floats.close()
        self.storer_throats_bools.close()
        self.storer_throats_corners_bool.close()
        self.storer_throats_corners_float.close()
        self.storer_throats_cycle_floats.close()
        self.storer_throats_cycle_ints.close()
        self.storer_pores.close()
        self.storer_pores_bools.close()
        self.storer_pores_corners_bool.close()
        self.storer_pores_corners_float.close()
        self.storer_pores_cycle_floats.close()
        self.storer_pores_cycle_ints.close()

    @staticmethod
    def _export_pn_to_json(pn, path):
        data = {}
        for key, value in pn.items():
            try:
                data[key] = value.tolist()  # convert numpy arrays
            except AttributeError:
                data[key] = value  # keep as-is if already serializable

        with open(path, "w") as f:
            json.dump(data, f)


class DiagnosticsStorer:
    N_STEPS = 500

    def __init__(self, name, n_elements, n_data, dtype, path=".", item_shape=()):
        self._name = name
        self._n_elements = n_elements
        self._n_data = n_data
        self._dtype = dtype
        self.path = path
        self._item_size = int(np.prod(item_shape, dtype=int))
        self._block = n_elements * self._item_size
        self._row_width = n_data * self._block
        self._memmap = None
        self._step = 0
        self._chunk = 0
        self._slices = [slice(i * self._block, (i + 1) * self._block) for i in range(n_data)]

    def _memmap_path(self):
        return os.path.join(self.path, f"diagnostics_{self._name}_{self._chunk}" + FILE_EXTENSION)

    def _init_memmap(self):
        self._memmap = np.memmap(
            self._memmap_path(), dtype=self._dtype, mode="w+", shape=(self.N_STEPS, self._row_width)
        )
        self._step = 0

    def save_state(self, arrays):
        if self._memmap is None:
            self._init_memmap()

        row = self._memmap[self._step]
        for s, arr in zip(self._slices, arrays):
            row[s] = np.ravel(arr)
        self._step += 1

        if self._step == self.N_STEPS:
            self._memmap.flush()
            del self._memmap
            self._memmap = None
            self._chunk += 1

    def close(self):
        if self._memmap is None:
            return
        self._memmap.flush()
        path = self._memmap_path()
        del self._memmap
        self._memmap = None
        truncated_size = self._step * self._row_width * np.dtype(self._dtype).itemsize
        with open(path, "r+b") as f:
            f.truncate(truncated_size)


class DiagnosticsInfoStorer:
    def __init__(self, path="."):
        self._path = os.path.join(path, INFO_FILE_NAME + FILE_EXTENSION)

    def save_number_of_elements(self, n_throats, n_pores):
        with open(self._path, "wb") as file:
            np.array([n_throats, n_pores], dtype=np.int64).tofile(file)

    def save_number_of_steps(self, cycle, n_steps):
        with open(self._path, "ab") as file:
            np.array([cycle, n_steps], dtype=np.int64).tofile(file)


class DiagnosticsInfo:
    def __init__(self, path):
        data = np.fromfile(os.path.join(path, INFO_FILE_NAME + FILE_EXTENSION), dtype=np.int64)
        self.n_throats = int(data[0])
        self.n_pores = int(data[1])
        self.cycle_n_steps = {int(data[i]): int(data[i + 1]) for i in range(2, len(data), 2)}
        self.n_cycles = len(self.cycle_n_steps)


# read diagnossis data related to a single simulation step
class DiagnosticsStateLoader:
    def __init__(self, path, step, n_elements):
        self.path = path
        self.step = step
        self.n_elements = n_elements

    # return a memory map for a "saved state" file
    def load_state(self, name, n_data, dtype, item_shape=()):
        return self._load_state(self.path, name, self.n_elements, n_data, dtype, item_shape)

    # return data[index] related to a single simulation step, shaped (n_elements, *item_shape)
    def data_view(self, data, index, item_shape=()):
        item_size = int(np.prod(item_shape, dtype=int))
        block = self.n_elements * item_size
        flat = data[self.step, index * block : (index + 1) * block]
        return flat.reshape((self.n_elements, *item_shape))

    @staticmethod
    def _load_state(path, name, n_elements, n_data, dtype, item_shape=()):
        chunk_regex = re.compile(rf"^diagnostics_{re.escape(name)}_(\d+){re.escape(FILE_EXTENSION)}$")
        chunks = []
        for file_name in os.listdir(path):
            match = chunk_regex.match(file_name)
            if match:
                chunks.append((int(match.group(1)), os.path.join(path, file_name)))
        chunk_paths = [chunk_path for _, chunk_path in sorted(chunks)]

        cols = n_elements * n_data * int(np.prod(item_shape, dtype=int))
        itemsize = np.dtype(dtype).itemsize

        arrays = []
        for chunk_path in chunk_paths:
            n_rows = os.path.getsize(chunk_path) // (cols * itemsize)
            if n_rows > 0:
                arrays.append(np.memmap(chunk_path, dtype=dtype, mode="r", shape=(n_rows, cols)))

        if not arrays:
            return np.empty((0, cols), dtype=dtype)
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays, axis=0)


class DiagnosticsThroatsStep(DiagnosticsStateLoader):
    def __init__(self, path, step, n_throats):
        super().__init__(path, step, n_throats)

        data = self.load_state(THROATS_FLOATS_NAME, THROATS_FLOATS_NDATA, np.float64)
        self.water_saturation = self.data_view(data, 0)
        self.oil_multiphase_conductivity = self.data_view(data, 1)
        self.water_multiphase_conductivity = self.data_view(data, 2)
        self.oil_multiphase_conductance = self.data_view(data, 3)
        self.water_multiphase_conductance = self.data_view(data, 4)

        geo = self.load_state(THROATS_CORNERS_BOOL_NAME, THROATS_CORNERS_BOOL_NDATA, np.bool_, CORNER_SHAPE)
        self.corner_water_layer = self.data_view(geo, 0, CORNER_SHAPE)
        self.corner_oil_layer = self.data_view(geo, 1, CORNER_SHAPE)

        bools = self.load_state(THROATS_BOOLS_NAME, THROATS_BOOLS_NDATA, np.bool_)
        self.water_filled = self.data_view(bools, 0)
        self.trapped = self.data_view(bools, 1)
        self.filled = self.data_view(bools, 2)

        corners = self.load_state(THROATS_CORNERS_FLOAT_NAME, THROATS_CORNERS_FLOAT_NDATA, np.float64, CORNER_SHAPE)
        self.b = self.data_view(corners, 0, CORNER_SHAPE)
        self.bo = self.data_view(corners, 1, CORNER_SHAPE)
        self.hinging_contact_angle = self.data_view(corners, 2, CORNER_SHAPE)
        self.hinging_contact_angle_o = self.data_view(corners, 3, CORNER_SHAPE)

    def __str__(self):
        return f"DiagnosticsThroatsStep (step: {self.step}, throats: {self.n_elements})"


class DiagnosticsPoresStep(DiagnosticsStateLoader):
    def __init__(self, path, step, n_pores):
        super().__init__(path, step, n_pores)

        data = self.load_state(PORES_FLOATS_NAME, PORES_FLOATS_NDATA, np.float64)
        self.water_saturation = self.data_view(data, 0)
        self.oil_multiphase_conductivity = self.data_view(data, 1)
        self.water_multiphase_conductivity = self.data_view(data, 2)
        self.oil_flow_rate = self.data_view(data, 3)
        self.water_flow_rate = self.data_view(data, 4)

        geo = self.load_state(PORES_CORNERS_BOOL_NAME, PORES_CORNERS_BOOL_NDATA, np.bool_, CORNER_SHAPE)
        self.corner_water_layer = self.data_view(geo, 0, CORNER_SHAPE)
        self.corner_oil_layer = self.data_view(geo, 1, CORNER_SHAPE)

        bools = self.load_state(PORES_BOOLS_NAME, PORES_BOOLS_NDATA, np.bool_)
        self.water_filled = self.data_view(bools, 0)
        self.trapped = self.data_view(bools, 1)
        self.filled = self.data_view(bools, 2)

        corners = self.load_state(PORES_CORNERS_FLOAT_NAME, PORES_CORNERS_FLOAT_NDATA, np.float64, CORNER_SHAPE)
        self.b = self.data_view(corners, 0, CORNER_SHAPE)
        self.bo = self.data_view(corners, 1, CORNER_SHAPE)
        self.hinging_contact_angle = self.data_view(corners, 2, CORNER_SHAPE)
        self.hinging_contact_angle_o = self.data_view(corners, 3, CORNER_SHAPE)

    def __str__(self):
        return f"DiagnosticsPoresStep (step: {self.step}, pores: {self.n_elements})"


class DiagnosticsThroatCycle(DiagnosticsStateLoader):
    def __init__(self, path, cycle, n_throats):
        super().__init__(path, cycle, n_throats)

        data = self.load_state(THROATS_CYCLE_FLOATS_NAME, THROATS_CYCLE_FLOATS_NDATA, np.float64)
        self.entry_pressure = self.data_view(data, 0)
        self.advancing_contact_angle = self.data_view(data, 1)
        self.receding_contact_angle = self.data_view(data, 2)

        data = self.load_state(THROATS_CYCLE_INTS_NAME, THROATS_CYCLE_INTS_NDATA, np.int8)
        self.invasion_method = self.data_view(data, 0)

    def __str__(self):
        return f"DiagnosticsThroatCycle (cycle: {self.step}, throats: {self.n_elements})"


class DiagnosticsPoreCycle(DiagnosticsStateLoader):
    def __init__(self, path, cycle, n_pores):
        super().__init__(path, cycle, n_pores)

        data = self.load_state(PORES_CYCLE_FLOATS_NAME, PORES_CYCLE_FLOATS_NDATA, np.float64)
        self.entry_pressure = self.data_view(data, 0)
        self.advancing_contact_angle = self.data_view(data, 1)
        self.receding_contact_angle = self.data_view(data, 2)

        data = self.load_state(PORES_CYCLE_INTS_NAME, PORES_CYCLE_INTS_NDATA, np.int8)
        self.invasion_method = self.data_view(data, 0)

    def __str__(self):
        return f"DiagnosticsPoreCycle (cycle: {self.step}, pores: {self.n_elements})"


class DiagnosticsCycleStep:
    def __init__(self, path, step, n_throats, n_pores):
        self.step = step
        self.throat = DiagnosticsThroatsStep(path, step, n_throats)
        self.pore = DiagnosticsPoresStep(path, step, n_pores)

    def __str__(self):
        return f"DiagnosticsCycleStep (step: {self.step})"


class DiagnosticsCycle:
    def __init__(self, path, cycle, info: DiagnosticsInfo):
        self.path = path
        self.cycle = cycle
        self.info = info
        self.throat = DiagnosticsThroatCycle(self.path, self.cycle, self.info.n_throats)
        self.pore = DiagnosticsPoreCycle(self.path, self.cycle, self.info.n_pores)

    # get data for a single step
    def __getitem__(self, step):
        step_offset = 0
        for cycle in range(0, self.cycle):
            step_offset += self.info.cycle_n_steps[cycle]
        return DiagnosticsCycleStep(self.path, step_offset + step, self.info.n_throats, self.info.n_pores)

    def __len__(self):
        return self.info.cycle_n_steps[self.cycle]

    def __str__(self):
        if self.cycle == Diagnostics.DRAINAGE:
            cycle_name = "drainage"
        elif self.cycle == Diagnostics.IMBIBITION:
            cycle_name = "imbibition"
        elif self.cycle == Diagnostics.SECOND_DRAINAGE:
            cycle_name = "2nd drainage"
        return f"DiagnosticsCycle ({cycle_name}, steps: {self.info.cycle_n_steps[self.cycle]})"


class DiagnosticsCycles:
    def __init__(self, path, info: DiagnosticsInfo):
        self.path = path
        self.info = info

    # get data for a single cycle
    def __getitem__(self, cycle):
        return DiagnosticsCycle(self.path, cycle, self.info)

    def __len__(self):
        return self.info.n_cycles

    def __str__(self):
        total_n_steps = 0
        for _, n_steps in self.info.cycle_n_steps:
            total_n_steps += n_steps
        return f"DiagnosticsCycles (cycles: {self.info.n_cyles}, steps: {total_n_steps})"


# read diagnostics files
# usage:
#   diagnostics.cycle[cycle_id][step].pore.water_saturation[pore_id]
#   diagnostics.cycle[cycle_id].throat.water_saturation[throat_id]
class Diagnostics:
    DRAINAGE = 0
    IMBIBITION = 1
    SECOND_DRAINAGE = 2

    def __init__(self, path):
        self.info = DiagnosticsInfo(path)
        self.cycle = DiagnosticsCycles(path, self.info)

        with open(path + "/" + NETWORK_JSON_NAME, "r") as file:
            self.network = json.load(file)

    @property
    def drainage(self):
        return self.cycle[self.DRAINAGE]

    @property
    def imbibition(self):
        return self.cycle[self.IMBIBITION]

    @property
    def second_drainage(self):
        return self.cycle[self.SECOND_DRAINAGE]

    def __str__(self):
        total_n_steps = 0
        for _, n_steps in self.info.cycle_n_steps:
            total_n_steps += n_steps
        return f"Diagnostics (cycles: {self.info.n_cyles}, steps: {total_n_steps})"
