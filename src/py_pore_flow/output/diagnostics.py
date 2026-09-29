import json
import os
import re

import numpy as np

from py_pore_flow import log
from py_pore_flow.equations.geometry import MAX_NUMBER_OF_CORNERS

FILE_EXTENSION = ".dat"
INFO_FILE_NAME = "info"
STEP_SCALARS_NAME = "step_scalars"
STEP_SCALARS_NDATA = 4

# simulation wide scalars saved once per step, in the order they are written. They are what
# lets the heatmap put something other than the step index on its X axis.
STEP_SCALAR_NAMES = ("pc", "sw")
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

DIRECTION_AXES = ("x", "y", "z")
ANGLE_NAME = "angle"
# throat quantities that can be projected onto the coordinate axes
DIRECTIONAL_SOURCES = ("water_flow_rate", "oil_flow_rate")


def _build_directional_specs():
    """{property name: (source property, axis index)}; a None source is the angle itself."""
    specs = {}
    for axis, axis_name in enumerate(DIRECTION_AXES):
        specs[f"{ANGLE_NAME}_{axis_name}"] = (None, axis)
        for source in DIRECTIONAL_SOURCES:
            specs[f"{source}_{axis_name}"] = (source, axis)
    return specs


DIRECTIONAL_SPECS = _build_directional_specs()


def zero_based_conns(conns, n_pores):
    """`throat.conns` as indices into the pore arrays, or None if it cannot be read as such.

    A py_pore_flow network stores zero based pore indices, but the GeoSlicer throat table it
    is built from carries one based pore labels, so the convention is checked against the
    pore count rather than assumed. Guessing wrong would silently attach every throat to a
    neighbouring pore and produce plausible but wrong directions.
    """
    conns = np.asarray(conns)
    if conns.ndim != 2 or conns.shape[1] < 2 or conns.size == 0:
        return None

    conns = conns[:, :2].astype(np.int64)
    low, high = int(conns.min()), int(conns.max())
    if low >= 1 and high == n_pores:
        return conns - 1
    if low >= 0 and high < n_pores:
        return conns
    return None


def throat_unit_vectors(network):
    """Unit vector of every throat, from the coordinates of the two pores it connects.

    Returns None when the dump has no usable geometry. A throat whose pores share a centre
    has no direction and comes back as NaN.
    """
    try:
        coords = np.asarray(network["pore.coords"], dtype=np.float64)
        conns = network["throat.conns"]
    except KeyError:
        return None

    if coords.ndim != 2 or coords.shape[1] < 3:
        return None

    conns = zero_based_conns(conns, coords.shape[0])
    if conns is None:
        return None

    vectors = coords[conns[:, 1], :3] - coords[conns[:, 0], :3]
    lengths = np.linalg.norm(vectors, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = vectors / lengths[:, None]
    unit[~np.isfinite(unit)] = np.nan
    return unit


def throat_axis_cosines(network):
    """Unsigned direction cosine of every throat against each axis, or None.

    Unsigned because which of its two pores a throat is measured from is arbitrary, so only
    the axis it lies along is meaningful, not the direction along it.
    """
    unit = throat_unit_vectors(network)
    return None if unit is None else np.abs(unit)


def throat_axis_angles(cosines):
    """Angle in degrees between each throat and each axis, in [0, 90]."""
    return np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0)))


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
        self.step_scalars_storer = DiagnosticsStorer(STEP_SCALARS_NAME, 1, STEP_SCALARS_NDATA, np.float64, path)
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
    def save_state(self, pc=np.nan, sw=np.nan, krw=np.nan, kro=np.nan):
        self.n_saved_sates += 1
        self.step_scalars_storer.save_state([pc, sw, krw, kro])

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
        self.step_scalars_storer.close()
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


class ConcatenatedMemMaps:
    def __init__(self, memmaps):
        self.__memmaps = memmaps
        self.__offsets = [0]
        for memmap in self.__memmaps[:-1]:
            last_offset = self.__offsets[-1]
            self.__offsets.append(last_offset + len(memmap))

        self.__length = self.__offsets[-1] + len(self.__memmaps[-1])
        self.__shape = (self.__length,) + self.__memmaps[0].shape[1:]

    @property
    def shape(self):
        return self.__shape

    def __len__(self):
        return self.__length

    def __array__(self, dtype=None, copy=None):
        array = np.concatenate(self.__memmaps, dtype=dtype)
        log.warning("Converting ConcatenatedMemMaps to np.array is not optimal")
        return array

    def __getitem__(self, key):
        if key is None:
            return self
        elif isinstance(key, (int, np.integer)):
            return self.__get_value_at(key)
        elif isinstance(key, slice):
            return self.__get_value_slice(key)
        elif isinstance(key, tuple):
            if len(key) == 1:
                return self.__getitem__(key[0])
            elif isinstance(key[0], (int, np.integer)):
                return self.__getitem__(key[0])[key[1:]]
            elif isinstance(key[0], slice):
                result = self.__getitem__(key[0])
                if isinstance(result, ConcatenatedMemMaps):
                    return result._get_axis_1(key[1:])
                else:
                    return result[:, *key[1:]]
            else:
                raise NotImplementedError(f"Key type not implemented: {key}")
        else:
            raise NotImplementedError(f"Key type not implemented: {key}")

    def __str__(self):
        representation = []
        for memmap in self.__memmaps:
            representation.append(str(memmap))
        return " || \n".join(representation)

    def __iter__(self):
        for memmap in self.__memmaps:
            yield from memmap

    def __get_value_at(self, index: int | np.integer):
        if index < 0:
            index += self.__length

        memmap_index = np.searchsorted(self.__offsets, index, side="right") - 1
        value_index = index - self.__offsets[memmap_index]
        return self.__memmaps[memmap_index][value_index]

    def __get_value_slice(self, array_slice: slice):
        array_slice = self.__normalize_slice(array_slice)
        if array_slice.step != 1:
            raise NotImplementedError("Only step=1 is supported")

        start_memmap_index = np.searchsorted(self.__offsets, array_slice.start, side="right") - 1
        start_value_index = array_slice.start - self.__offsets[start_memmap_index]

        stop_memmap_index = np.searchsorted(self.__offsets, array_slice.stop, side="right") - 1
        stop_value_index = array_slice.stop - self.__offsets[stop_memmap_index]

        if start_memmap_index == stop_memmap_index:
            return self.__memmaps[start_memmap_index][slice(start_value_index, stop_value_index)]
        else:
            memmap_pieces = []
            for memmap_index in range(start_memmap_index, stop_memmap_index + 1):
                if memmap_index == start_memmap_index:
                    memmap_pieces.append(self.__memmaps[memmap_index][slice(start_value_index, None)])
                elif memmap_index == stop_memmap_index:
                    memmap_pieces.append(self.__memmaps[memmap_index][slice(None, stop_value_index)])
                else:
                    memmap_pieces.append(self.__memmaps[memmap_index])

            return ConcatenatedMemMaps(memmap_pieces)

    def __normalize_slice(self, array_slice: slice):
        return slice(*array_slice.indices(self.__length))

    def _get_axis_1(self, key):
        memmap_pieces = []
        for memmap in self.__memmaps:
            memmap_pieces.append(memmap[:, *key])
        return ConcatenatedMemMaps(memmap_pieces)


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
            rows = self._open_chunks(self.path, name, n_elements, n_data, dtype, item_shape)
            self._rows[name] = rows
        return rows

    def close(self):
        self._rows.clear()

    @staticmethod
    def _open_chunks(path, name, n_elements, n_data, dtype, item_shape=()):
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

        if len(arrays) > 0:
            return ConcatenatedMemMaps(arrays)
        else:
            return np.empty((0, cols), dtype=dtype)


# read diagnossis data related to a single simulation step
class DiagnosticsStateLoader:
    def __init__(self, path, step, n_elements, cache=None):
        self.path = path
        self.step = step
        self.n_elements = n_elements
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


class DiagnosticsCycleStep:
    def __init__(self, path, step, n_throats, n_pores, cache=None):
        self.step = step
        self.scalars = DiagnosticsScalarsStep(path, step, cache)
        self.throat = DiagnosticsThroatsStep(path, step, n_throats, cache)
        self.pore = DiagnosticsPoresStep(path, step, n_pores, cache)

    @property
    def pc(self):
        return self.scalars.pc

    @property
    def sw(self):
        return self.scalars.sw

    @property
    def krw(self):
        return self.scalars.krw

    @property
    def kro(self):
        return self.scalars.kro

    def __str__(self):
        return f"DiagnosticsCycleStep (step: {self.step})"


class DiagnosticsCycle:
    def __init__(self, path, cycle, info: DiagnosticsInfo, cache=None):
        self._cache = cache
        self._steps = {}
        self._step_offset = sum(info.cycle_n_steps.get(c, 0) for c in range(cycle))

        self.path = path
        self.cycle = cycle
        self.info = info
        self.throat = DiagnosticsThroatCycle(self.path, self.cycle, self.info.n_throats, cache)
        self.pore = DiagnosticsPoreCycle(self.path, self.cycle, self.info.n_pores, cache)
        self.scalars = DiagnosticsScalarsCycle(self.path, self._step_offset, info.cycle_n_steps[cycle], cache)

    @property
    def pc(self):
        return self.scalars.pc

    @property
    def sw(self):
        return self.scalars.sw

    @property
    def krw(self):
        return self.scalars.krw

    @property
    def kro(self):
        return self.scalars.kro

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
#   diagnostics.cycle[cycle_id][step].pc
#   diagnostics.cycle[cycle_id][step].pore.water_saturation[pore_id]
#   diagnostics.cycle[cycle_id].throat.water_saturation[throat_id]
#   diagnostics.cycle[cycle_id].pc
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
