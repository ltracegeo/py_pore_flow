import json
import os
import re

import numpy as np

from py_pore_flow.equations.geometry import MAX_NUMBER_OF_CORNERS

FILE_EXTENSION = ".dat"
INFO_FILE_NAME = "info"
NETWORK_JSON_NAME = "network.json"
NETWORK_ARRAYS_NAME = "network.npz"
NETWORK_EXTRAS_NAME = "network_extras.json"

# trailing per-element shape of the corner layer arrays (one value per corner)
CORNER_SHAPE = (MAX_NUMBER_OF_CORNERS,)

THROATS_FLOATS_NAME = "throats_floats"
THROATS_FLOATS_NDATA = 9
THROATS_BOOLS_NAME = "throats_bools"
THROATS_BOOLS_NDATA = 3
THROATS_CORNERS_BOOLS_NAME = "throats_corners_bools"
THROATS_CORNERS_BOOLS_NDATA = 2
THROATS_CORNERS_FLOATS_NAME = "throats_corners_floats"
THROATS_CORNERS_FLOATS_NDATA = 4
THROATS_CYCLE_FLOATS_NAME = "throats_cycle_floats"
THROATS_CYCLE_FLOATS_NDATA = 3
THROATS_CYCLE_INTS_NAME = "throats_cycle_ints"
THROATS_CYCLE_INTS_NDATA = 1
PORES_FLOATS_NAME = "pores_floats"
PORES_FLOATS_NDATA = 5
PORES_BOOLS_NAME = "pores_bools"
PORES_BOOLS_NDATA = 3
PORES_CORNERS_BOOLS_NAME = "pores_corners_bools"
PORES_CORNERS_BOOLS_NDATA = 2
PORES_CORNERS_FLOATS_NAME = "pores_corners_floats"
PORES_CORNERS_FLOATS_NDATA = 4
PORES_CYCLE_FLOATS_NAME = "pores_cycle_floats"
PORES_CYCLE_FLOATS_NDATA = 3
PORES_CYCLE_INTS_NAME = "pores_cycle_ints"
PORES_CYCLE_INTS_NDATA = 1

STATIC_PROPERTIES = [
    "coords",
    "shape_factor",
    "radius",
    "volume",
    "clay_volume",
    "corner_half_angle",
    "sides",
    "oil_singlephase_conductivity",
    "water_singlephase_conductivity",
    "oil_singlephase_conductance",
    "water_singlephase_conductance",
    "shape",
    "n_corners",
    "conns",
    "conns_values",
    "conns_offsets",
    "n_conns",
    "area",
    "inlets",
    "outlets",
    "initial_advancing_ca",
    "initial_receding_ca",
    "equilibrium_advancing_ca",
    "equilibrium_receding_ca",
    "second_advancing_ca",
    "second_receding_ca",
    "id",
    "number_of_capilaries",
    "cross_sectional_area",
    "length",
    # Not static but needed for now
    "oil_layer_collapse_pressure",
    "piston_like_pressure",
    "snap_off_pressure",
]


class DiagnosticsWriter:
    def __init__(self, network, path="."):
        self.pn = network.pn_core

        self.cycle_id = 0
        self.n_saved_sates = 0

        n_throats = self.pn.throat.water_saturation.size
        n_pores = self.pn.pore.water_saturation.size

        self.info_storer = DiagnosticsInfoStorer(path)
        self.info_storer.save_number_of_elements(n_throats, n_pores)
        self.storer_throats_floats = DiagnosticsStorer(
            THROATS_FLOATS_NAME, n_throats, THROATS_FLOATS_NDATA, np.float64, path
        )
        self.storer_throats_bools = DiagnosticsStorer(
            THROATS_BOOLS_NAME, n_throats, THROATS_BOOLS_NDATA, np.bool_, path
        )
        self.storer_throats_corners_bool = DiagnosticsStorer(
            THROATS_CORNERS_BOOLS_NAME, n_throats, THROATS_CORNERS_BOOLS_NDATA, np.bool_, path, item_shape=CORNER_SHAPE
        )
        self.storer_throats_corners_float = DiagnosticsStorer(
            THROATS_CORNERS_FLOATS_NAME,
            n_throats,
            THROATS_CORNERS_FLOATS_NDATA,
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
            PORES_CORNERS_BOOLS_NAME, n_pores, PORES_CORNERS_BOOLS_NDATA, np.bool_, path, item_shape=CORNER_SHAPE
        )
        self.storer_pores_corners_float = DiagnosticsStorer(
            PORES_CORNERS_FLOATS_NAME, n_pores, PORES_CORNERS_FLOATS_NDATA, np.float64, path, item_shape=CORNER_SHAPE
        )
        self.storer_pores_cycle_floats = DiagnosticsStorer(
            PORES_CYCLE_FLOATS_NAME, n_pores, PORES_CYCLE_FLOATS_NDATA, np.float64, path
        )
        self.storer_pores_cycle_ints = DiagnosticsStorer(
            PORES_CYCLE_INTS_NAME, n_pores, PORES_CYCLE_INTS_NDATA, np.int8, path
        )

        self._export_pn(network.pn, path)

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
                throat.oil_pressure,
                throat.water_pressure,
                throat.oil_flow_rate,
                throat.water_flow_rate,
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
                pore.oil_pressure,
                pore.water_pressure,
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
    def _export_pn(pn, path):
        """Dump the static network properties as a compressed npz.

        The previous JSON dump went through ndarray.tolist(), which for a 100k+ element
        network builds hundreds of MB of Python floats on write and pays the same cost
        again parsing them back when the viewer opens.
        """
        arrays = {}
        extras = {}
        for property in STATIC_PROPERTIES:
            for prefix in ["pore", "throat"]:
                key = f"{prefix}.{property}"
                try:
                    value = pn[key]
                except KeyError:
                    continue
                if isinstance(value, np.ndarray):
                    array = value
                else:
                    try:
                        array = np.asarray(value)
                    except Exception:
                        # sparse or ragged accessors (pore.conns) refuse to densify
                        continue
                if array.dtype == object:
                    continue
                if array.ndim == 0:
                    extras[key] = array.item()
                    continue
                arrays[key] = array

        np.savez_compressed(os.path.join(path, NETWORK_ARRAYS_NAME), **arrays)
        if extras:
            with open(os.path.join(path, NETWORK_EXTRAS_NAME), "w") as file:
                json.dump(extras, file)


class DiagnosticsStorer:
    # upper bound on the number of rows of a single chunk file
    N_STEPS = 500
    # a chunk file is preallocated in full, so it is also capped by a byte budget: bounded by
    # a step count alone, a 134k throat network reserves several GB per storer up front.
    TARGET_CHUNK_BYTES = 64 * 1024 * 1024

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
        self._chunk_n_steps = self._resolve_chunk_n_steps()

    def _resolve_chunk_n_steps(self):
        row_bytes = self._row_width * np.dtype(self._dtype).itemsize
        if row_bytes <= 0:
            return self.N_STEPS
        return max(1, min(self.N_STEPS, self.TARGET_CHUNK_BYTES // row_bytes))

    def _memmap_path(self):
        return os.path.join(self.path, f"diagnostics_{self._name}_{self._chunk}" + FILE_EXTENSION)

    def _init_memmap(self):
        self._memmap = np.memmap(
            self._memmap_path(), dtype=self._dtype, mode="w+", shape=(self._chunk_n_steps, self._row_width)
        )
        self._step = 0

    def save_state(self, arrays):
        if self._memmap is None:
            self._init_memmap()

        row = self._memmap[self._step]
        for s, arr in zip(self._slices, arrays):
            row[s] = np.ravel(arr)
        self._step += 1

        if self._step == self._chunk_n_steps:
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


class ChunkedRows:
    """Row addressable, lazy view over the ordered chunk files of a single storer.

    Replaces the np.concatenate the loader used to do when a run spans more than one chunk:
    that copied every step of every element into RAM on *each* property read, and a run
    longer than one chunk is the normal case.
    """

    __slots__ = ("_chunks", "_starts", "shape", "dtype")

    def __init__(self, chunks, n_cols, dtype):
        self._chunks = [chunk for chunk in chunks if chunk.shape[0] > 0]
        starts = [0]
        for chunk in self._chunks:
            starts.append(starts[-1] + chunk.shape[0])
        self._starts = np.array(starts, dtype=np.int64)
        self.shape = (int(self._starts[-1]), int(n_cols))
        self.dtype = np.dtype(dtype)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row = self._row(key[0])
            rest = key[1:]
            return row[rest[0]] if len(rest) == 1 else row[rest]
        return self._row(key)

    def _row(self, step):
        step = int(step)
        if step < 0:
            step += self.shape[0]
        if not 0 <= step < self.shape[0]:
            raise IndexError(f"step {step} is out of range for {self.shape[0]} saved steps")
        chunk_index = int(np.searchsorted(self._starts, step, side="right")) - 1
        return self._chunks[chunk_index][step - int(self._starts[chunk_index])]

    def close(self):
        self._chunks = []
        self._starts = np.zeros(1, dtype=np.int64)
        self.shape = (0, self.shape[1])


def open_chunks(path, name, n_elements, n_data, dtype, item_shape=()):
    """Open every chunk file of a storer as a memmap, wrapped in a ChunkedRows."""
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

    return ChunkedRows(arrays, cols, dtype)


class DiagnosticsFileCache:
    """Opens the chunk files of each storer once per Diagnostics instance.

    Without it every property read re-listed the diagnostics directory and re-opened every
    chunk file, 12 times per read - and the element view issues around 25 reads per step
    change.
    """

    def __init__(self, path):
        self.path = path
        self._rows = {}

    def rows(self, name, n_elements, n_data, dtype, item_shape=()):
        rows = self._rows.get(name)
        if rows is None:
            rows = open_chunks(self.path, name, n_elements, n_data, dtype, item_shape)
            self._rows[name] = rows
        return rows

    def close(self):
        for rows in self._rows.values():
            rows.close()
        self._rows.clear()


# read diagnossis data related to a single simulation step
class DiagnosticsStateLoader:
    def __init__(self, path, step, n_elements, cache=None):
        self.path = path
        self.step = step
        self.n_elements = n_elements
        # a standalone loader still works, it just gets a cache of its own
        self._cache = cache if cache is not None else DiagnosticsFileCache(path)

    # return a lazy row view over the chunk files of a "saved state" storer
    def load_state(self, name, n_data, dtype, item_shape=()):
        return self._cache.rows(name, self.n_elements, n_data, dtype, item_shape)

    # return data[index] related to a single simulation step, shaped (n_elements, *item_shape)
    def data_view(self, data, index, item_shape=()):
        item_size = int(np.prod(item_shape, dtype=int))
        block = self.n_elements * item_size
        flat = data[self.step, index * block : (index + 1) * block]
        return flat.reshape((self.n_elements, *item_shape))

    @staticmethod
    def _load_state(path, name, n_elements, n_data, dtype, item_shape=()):
        return open_chunks(path, name, n_elements, n_data, dtype, item_shape)


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


class DiagnosticsThroatCycle(DiagnosticsStateLoader):
    def __init__(self, path, cycle, n_throats, cache=None):
        super().__init__(path, cycle, n_throats, cache)

        data = self.load_state(THROATS_CYCLE_FLOATS_NAME, THROATS_CYCLE_FLOATS_NDATA, np.float64)
        self.entry_pressure = self.data_view(data, 0)
        self.advancing_contact_angle = self.data_view(data, 1)
        self.receding_contact_angle = self.data_view(data, 2)

        data = self.load_state(THROATS_CYCLE_INTS_NAME, THROATS_CYCLE_INTS_NDATA, np.int8)
        self.invasion_method = self.data_view(data, 0)

    def __str__(self):
        return f"DiagnosticsThroatCycle (cycle: {self.step}, throats: {self.n_elements})"


class DiagnosticsPoreCycle(DiagnosticsStateLoader):
    def __init__(self, path, cycle, n_pores, cache=None):
        super().__init__(path, cycle, n_pores, cache)

        data = self.load_state(PORES_CYCLE_FLOATS_NAME, PORES_CYCLE_FLOATS_NDATA, np.float64)
        self.entry_pressure = self.data_view(data, 0)
        self.advancing_contact_angle = self.data_view(data, 1)
        self.receding_contact_angle = self.data_view(data, 2)

        data = self.load_state(PORES_CYCLE_INTS_NAME, PORES_CYCLE_INTS_NDATA, np.int8)
        self.invasion_method = self.data_view(data, 0)

    def __str__(self):
        return f"DiagnosticsPoreCycle (cycle: {self.step}, pores: {self.n_elements})"


class DiagnosticsCycleStep:
    def __init__(self, path, step, n_throats, n_pores, cache=None):
        self.step = step
        self.throat = DiagnosticsThroatsStep(path, step, n_throats, cache)
        self.pore = DiagnosticsPoresStep(path, step, n_pores, cache)

    def __str__(self):
        return f"DiagnosticsCycleStep (step: {self.step})"


class DiagnosticsCycle:
    def __init__(self, path, cycle, info: DiagnosticsInfo, cache=None):
        self.path = path
        self.cycle = cycle
        self.info = info
        self._cache = cache
        self._steps = {}
        self._step_offset = sum(info.cycle_n_steps.get(c, 0) for c in range(cycle))
        self.throat = DiagnosticsThroatCycle(self.path, self.cycle, self.info.n_throats, cache)
        self.pore = DiagnosticsPoreCycle(self.path, self.cycle, self.info.n_pores, cache)

    # get data for a single step
    def __getitem__(self, step):
        cycle_step = self._steps.get(step)
        if cycle_step is None:
            cycle_step = DiagnosticsCycleStep(
                self.path, self._step_offset + step, self.info.n_throats, self.info.n_pores, self._cache
            )
            self._steps[step] = cycle_step
        return cycle_step

    def __len__(self):
        return self.info.cycle_n_steps[self.cycle]

    def __str__(self):
        cycle_name = {
            Diagnostics.DRAINAGE: "drainage",
            Diagnostics.IMBIBITION: "imbibition",
            Diagnostics.SECOND_DRAINAGE: "2nd drainage",
        }.get(self.cycle, str(self.cycle))
        return f"DiagnosticsCycle ({cycle_name}, steps: {self.info.cycle_n_steps[self.cycle]})"


class DiagnosticsCycles:
    def __init__(self, path, info: DiagnosticsInfo, cache=None):
        self.path = path
        self.info = info
        self._cache = cache
        self._cycles = {}

    # get data for a single cycle
    def __getitem__(self, cycle):
        cycle_data = self._cycles.get(cycle)
        if cycle_data is None:
            cycle_data = DiagnosticsCycle(self.path, cycle, self.info, self._cache)
            self._cycles[cycle] = cycle_data
        return cycle_data

    def __len__(self):
        return self.info.n_cycles

    def __str__(self):
        total_n_steps = sum(self.info.cycle_n_steps.values())
        return f"DiagnosticsCycles (cycles: {self.info.n_cycles}, steps: {total_n_steps})"


# read diagnostics files
# usage:
#   diagnostics.cycle[cycle_id][step].pore.water_saturation[pore_id]
#   diagnostics.cycle[cycle_id].throat.water_saturation[throat_id]
class Diagnostics:
    DRAINAGE = 0
    IMBIBITION = 1
    SECOND_DRAINAGE = 2

    def __init__(self, path):
        self.path = path
        self.info = DiagnosticsInfo(path)
        self._cache = DiagnosticsFileCache(path)
        self.cycle = DiagnosticsCycles(path, self.info, self._cache)
        self._network = None

    @property
    def network(self):
        """Static network properties, read on first use."""
        if self._network is None:
            self._network = self._load_network(self.path)
        return self._network

    @staticmethod
    def _load_network(path):
        arrays_path = os.path.join(path, NETWORK_ARRAYS_NAME)
        if os.path.exists(arrays_path):
            with np.load(arrays_path) as data:
                network = {key: data[key] for key in data.files}
            extras_path = os.path.join(path, NETWORK_EXTRAS_NAME)
            if os.path.exists(extras_path):
                with open(extras_path, "r") as file:
                    network.update(json.load(file))
            return network

        # diagnostics written before the binary dump
        with open(os.path.join(path, NETWORK_JSON_NAME), "r") as file:
            return json.load(file)

    def close(self):
        """Release the memmapped chunk files.

        Windows holds a lock on a mapped file and the diagnostics directory is moved when
        the project is saved, so the viewer has to let the handles go when it closes.
        """
        self._cache.close()
        self.cycle = DiagnosticsCycles(self.path, self.info, self._cache)
        self._network = None

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
        total_n_steps = sum(self.info.cycle_n_steps.values())
        return f"Diagnostics (cycles: {self.info.n_cycles}, steps: {total_n_steps})"
