"""Precomputed heatmap aggregates.

The complete diagnostics dump grows with the network: every property of every element of
every step. The heatmap, which is the view most people actually use, only ever needs, per
(step, bin of a property Y), the count of elements and the sum of a second property C - a
payload whose size does not depend on the number of elements at all.

Computing those bins needs the exact range of Y over the whole cycle, which is not known
while the simulation is running, so the run spools the properties it needs to disk (scalars
only, no corner arrays), and ``close()`` walks the spool once to build the matrices and then
deletes it. Peak memory stays at one step, and what is kept is a few tens of MB.

Only sum and count are stored: mean is sum / count, so no aggregation mode needs its own
payload.
"""

import json
import os

import numpy as np

from py_pore_flow.output.diagnostics import (
    ANGLE_NAME,
    DIRECTION_AXES,
    DIRECTIONAL_SPECS,
    Diagnostics,
    DiagnosticsFileCache,
    DiagnosticsInfo,
    DiagnosticsInfoStorer,
    DiagnosticsStorer,
    DiagnosticsWriter,
    FILE_EXTENSION,
    STEP_SCALAR_NAMES,
    throat_axis_angles,
    throat_axis_cosines,
)

LEVEL_OFF = "off"
LEVEL_BASIC = "basic"
LEVEL_COMPLETE = "complete"
LEVELS = (LEVEL_OFF, LEVEL_BASIC, LEVEL_COMPLETE)

MANIFEST_FILE_NAME = "diagnostics_manifest.json"
AGGREGATE_FILE_NAME = "aggregates.npz"

SPOOL_NAMES = {"pore": "spool_pores", "throat": "spool_throats"}
ELEMENT_TYPES = ("pore", "throat")

# stored bin resolution: the viewer merges these down to whatever bin count is asked for
AGGREGATE_N_BINS = 500
MIN_AGGREGATE_N_BINS = 16
# fine bins are pointless when there are fewer elements than bins
ELEMENTS_PER_BIN = 4

KEY_SEPARATOR = "|"
COUNT_KEY = "count"
EDGES_KEY = "edges"
SUM_KEY = "sum"
# per step simulation scalars, kept alongside the matrices so the heatmap can put them on
# its X axis. "x" cannot collide with an element type.
X_AXIS_KEY = "x"

# Bin spacing of the Y axis. Both are stored: linear fine bins cannot be merged into log
# bins without collapsing the bottom of the axis into a single cell, and vice versa.
SCALE_LINEAR = "linear"
SCALE_LOG = "log"

AGG_COUNT = "count"
AGG_MEAN = "mean"
AGG_SUM = "sum"

# kept in step with DiagnosticsProperties.CONVERSORS
CONVERSORS = {
    "oil_saturation": ("water_saturation", lambda values: 1.0 - values),
    "oil_filled": ("water_filled", lambda values: ~values),
}

# (element type, binned property, aggregated property or None for a plain element count).
# Deliberately a short, extendable list: an aggregate is the same size whatever the network,
# so covering every property pair would cost more than the raw dump it replaces.
DEFAULT_PAIRS = [
    ("throat", "radius", None),
    ("throat", "radius", "water_flow_rate"),
    ("throat", "radius", "oil_flow_rate"),
    ("throat", "radius", "water_multiphase_conductance"),
    ("throat", "radius", "oil_multiphase_conductance"),
    ("throat", "radius", "water_saturation"),
    ("throat", "radius", "oil_saturation"),
    ("throat", "radius", "water_filled"),
    ("throat", "radius", "water_pressure"),
    ("throat", "radius", "oil_pressure"),
    ("throat", "volume", None),
    ("throat", "volume", "water_saturation"),
    ("throat", "volume", "water_filled"),
    ("throat", "volume", "water_flow_rate"),
    ("throat", "volume", "oil_flow_rate"),
    ("throat", "shape_factor", None),
    ("throat", "shape_factor", "water_saturation"),
    ("throat", "shape_factor", "water_filled"),
    ("throat", "length", None),
    ("throat", "length", "water_saturation"),
    ("throat", "length", "water_filled"),
    ("throat", "cross_sectional_area", None),
    ("throat", "cross_sectional_area", "water_flow_rate"),
    ("throat", "cross_sectional_area", "oil_flow_rate"),
    ("throat", "water_saturation", None),
    ("throat", "water_saturation", "radius"),
    ("throat", "water_saturation", "water_flow_rate"),
    ("throat", "water_saturation", "oil_flow_rate"),
    ("throat", "water_multiphase_conductivity", None),
    ("throat", "water_multiphase_conductivity", "radius"),
    ("throat", "angle_z", None),
    ("throat", "angle_z", "water_flow_rate_z"),
    ("throat", "angle_z", "oil_flow_rate_z"),
    ("throat", "radius", "water_flow_rate_x"),
    ("throat", "radius", "water_flow_rate_y"),
    ("throat", "radius", "water_flow_rate_z"),
    ("throat", "radius", "oil_flow_rate_x"),
    ("throat", "radius", "oil_flow_rate_y"),
    ("throat", "radius", "oil_flow_rate_z"),
    ("pore", "radius", None),
    ("pore", "radius", "water_saturation"),
    ("pore", "radius", "oil_saturation"),
    ("pore", "radius", "water_filled"),
    ("pore", "radius", "water_pressure"),
    ("pore", "radius", "oil_pressure"),
    ("pore", "radius", "volume"),
    ("pore", "volume", None),
    ("pore", "volume", "water_saturation"),
    ("pore", "volume", "water_filled"),
    ("pore", "shape_factor", None),
    ("pore", "shape_factor", "water_saturation"),
    ("pore", "shape_factor", "water_filled"),
    ("pore", "n_conns", None),
    ("pore", "n_conns", "water_saturation"),
    ("pore", "n_conns", "water_filled"),
    ("pore", "water_saturation", None),
    ("pore", "water_saturation", "radius"),
    ("pore", "water_saturation", "volume"),
]


def aggregate_key(element_type, y_property, scale, kind, color_property=None):
    parts = [element_type, y_property, scale, kind]
    if color_property is not None:
        parts.append(color_property)
    return KEY_SEPARATOR.join(parts)


def scale_edges(scale, low, high, n_bins):
    if scale == SCALE_LOG:
        return np.logspace(np.log10(low), np.log10(high), n_bins + 1)
    return np.linspace(low, high, n_bins + 1)


def cycle_key(cycle, key):
    return f"{cycle}{KEY_SEPARATOR}{key}"


def step_axis_key(cycle, name):
    return cycle_key(cycle, KEY_SEPARATOR.join([X_AXIS_KEY, name]))


def read_manifest(path):
    """The manifest of a diagnostics directory, or None for one written before levels."""
    manifest_path = os.path.join(path, MANIFEST_FILE_NAME)
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as file:
        return json.load(file)


def diagnostics_level(path):
    """Detail level of a diagnostics directory. Directories predating levels are complete."""
    manifest = read_manifest(path)
    if manifest is None:
        return LEVEL_COMPLETE
    return manifest.get("level", LEVEL_COMPLETE)


def write_manifest(path, level, extra=None):
    manifest = {"level": level}
    if extra:
        manifest.update(extra)
    with open(os.path.join(path, MANIFEST_FILE_NAME), "w") as file:
        json.dump(manifest, file)


def resolve_n_bins(n_elements):
    return int(max(MIN_AGGREGATE_N_BINS, min(AGGREGATE_N_BINS, n_elements // ELEMENTS_PER_BIN)))


class AggregateDiagnosticsWriter:
    """Writes the heatmap aggregates, and nothing else.

    Interchangeable with DiagnosticsWriter: same save_state / save_cycle_data / close.
    """

    def __init__(self, network, path=".", pairs=None, n_bins=None):
        self.pn = network.pn_core
        self.path = path
        self._pairs = list(DEFAULT_PAIRS if pairs is None else pairs)
        self._requested_n_bins = n_bins

        DiagnosticsWriter._export_pn(network.pn, path)
        self._static = Diagnostics._load_network(path)
        self._setup_throat_directions()

        self._n_elements = {
            "pore": int(self.pn.pore.water_saturation.size),
            "throat": int(self.pn.throat.water_saturation.size),
        }

        self._info_storer = DiagnosticsInfoStorer(path)
        self._info_storer.save_number_of_elements(self._n_elements["throat"], self._n_elements["pore"])

        self._dynamic = {element_type: self._dynamic_properties(element_type) for element_type in ELEMENT_TYPES}
        self._spools = {}
        for element_type, names in self._dynamic.items():
            if not names:
                continue
            self._spools[element_type] = DiagnosticsStorer(
                SPOOL_NAMES[element_type], self._n_elements[element_type], len(names), np.float32, path
            )

        self._cycle_id = 0
        self._n_saved_states = 0
        self._step_scalars = []

    def _setup_throat_directions(self):
        """Publish the throat angles as static properties, so they need no spool.

        Without usable geometry the directional pairs cannot be built at all, so they are
        dropped rather than silently aggregated from the wrong numbers.
        """
        self._throat_cosines = throat_axis_cosines(self._static)
        if self._throat_cosines is None:
            self._pairs = [pair for pair in self._pairs if not self._is_directional(pair)]
            return

        for axis, axis_name in enumerate(DIRECTION_AXES):
            self._static[f"throat.{ANGLE_NAME}_{axis_name}"] = throat_axis_angles(self._throat_cosines[:, axis])

    @staticmethod
    def _is_directional(pair):
        element_type, y_property, color_property = pair
        if element_type != "throat":
            return False
        return y_property in DIRECTIONAL_SPECS or color_property in DIRECTIONAL_SPECS

    # ------------------------------------------------------------------ writing

    def save_state(self, pc=np.nan, sw=np.nan, krw=np.nan, kro=np.nan):
        self._n_saved_states += 1
        self._step_scalars.append((pc, sw))
        for element_type, storer in self._spools.items():
            element = getattr(self.pn, element_type)
            storer.save_state([self._live_values(element_type, element, name) for name in self._dynamic[element_type]])

    def save_cycle_data(self):
        self._info_storer.save_number_of_steps(self._cycle_id, self._n_saved_states)
        self._cycle_id += 1
        self._n_saved_states = 0

    def close(self):
        for storer in self._spools.values():
            storer.close()

        info = DiagnosticsInfo(self.path)
        cache = DiagnosticsFileCache(self.path)
        try:
            arrays, aggregates = self._build_aggregates(info, cache)
        finally:
            cache.close()

        np.savez_compressed(os.path.join(self.path, AGGREGATE_FILE_NAME), **arrays)
        write_manifest(
            self.path,
            LEVEL_BASIC,
            {
                "n_pores": info.n_pores,
                "n_throats": info.n_throats,
                "cycle_n_steps": {str(cycle): n for cycle, n in info.cycle_n_steps.items()},
                "aggregates": aggregates,
            },
        )
        self._delete_spool()

    # ------------------------------------------------------------- property access

    def _is_static(self, element_type, name):
        """A property is static when the network dump already carries it."""
        return f"{element_type}.{name}" in self._static

    def _dynamic_properties(self, element_type):
        names = set()
        for pair_element_type, y_property, color_property in self._pairs:
            if pair_element_type != element_type:
                continue
            for name in (y_property, color_property):
                if name is None or self._is_static(element_type, name):
                    continue
                names.add(self._spool_source(element_type, name))
        return sorted(names)

    @staticmethod
    def _spool_source(element_type, name):
        """The property actually spooled for `name`.

        A projection onto an axis is its source times a constant cosine, so only the source
        is spooled and all three components are derived from it.
        """
        spec = DIRECTIONAL_SPECS.get(name) if element_type == "throat" else None
        if spec is None or spec[0] is None:
            return name
        return spec[0]

    def _live_values(self, element_type, element, name):
        spec = DIRECTIONAL_SPECS.get(name) if element_type == "throat" else None
        if spec is not None:
            source, axis = spec
            if source is None:
                return throat_axis_angles(self._throat_cosines[:, axis])
            return np.asarray(getattr(element, source), dtype=np.float64) * self._throat_cosines[:, axis]

        source, convert = CONVERSORS.get(name, (name, None))
        values = getattr(element, source)
        return values if convert is None else convert(values)

    def _spooled_values(self, reader, element_type, name, step):
        """Per step values of `name`, projecting the spooled source when it is directional."""
        spec = DIRECTIONAL_SPECS.get(name) if element_type == "throat" else None
        if spec is None or spec[0] is None:
            return reader.values(name, step).astype(np.float64, copy=False)
        source, axis = spec
        return reader.values(source, step).astype(np.float64, copy=False) * self._throat_cosines[:, axis]

    def _delete_spool(self):
        for file_name in os.listdir(self.path):
            for spool_name in SPOOL_NAMES.values():
                if file_name.startswith(f"diagnostics_{spool_name}_") and file_name.endswith(FILE_EXTENSION):
                    os.remove(os.path.join(self.path, file_name))

    # ---------------------------------------------------------------- aggregating

    def _build_aggregates(self, info, cache):
        readers = {}
        for element_type, names in self._dynamic.items():
            if names:
                readers[element_type] = _SpoolReader(
                    cache, SPOOL_NAMES[element_type], self._n_elements[element_type], names
                )

        value_ranges, log_scales = self._scan_ranges(readers, info)

        scalars = np.asarray(self._step_scalars, dtype=np.float64).reshape(-1, len(STEP_SCALAR_NAMES))

        arrays = {}
        aggregates = {}
        step_offset = 0
        for cycle in range(info.n_cycles):
            n_steps = info.cycle_n_steps.get(cycle, 0)
            for index, name in enumerate(STEP_SCALAR_NAMES):
                values = scalars[step_offset : step_offset + n_steps, index]
                if values.size:
                    arrays[step_axis_key(cycle, name)] = values
            for element_type in ELEMENT_TYPES:
                n_bins = self._requested_n_bins or resolve_n_bins(self._n_elements[element_type])
                for y_property, color_properties in self._grouped_pairs(element_type).items():
                    scales = [SCALE_LINEAR]
                    if log_scales.get((element_type, y_property), False):
                        scales.append(SCALE_LOG)
                    aggregates.setdefault(element_type, {})[y_property] = {
                        "colors": sorted(color_properties),
                        "scales": scales,
                    }
                    if n_steps == 0:
                        continue
                    self._aggregate_one(
                        arrays,
                        readers.get(element_type),
                        cycle,
                        element_type,
                        y_property,
                        sorted(color_properties),
                        step_offset,
                        n_steps,
                        n_bins,
                        value_ranges[(cycle, element_type, y_property)],
                        scales,
                    )
            step_offset += n_steps
        return arrays, aggregates

    def _scan_ranges(self, readers, info):
        """Exact per cycle range of every binned property, plus whether log bins are possible.

        This is what the spool exists for: the range of a per step property is only known
        once the run is over. Log bins additionally need the property to be positive in
        every cycle, so that the option cannot change meaning when the cycle is switched.
        """
        value_ranges = {}
        log_scales = {}
        step_offset = 0
        for cycle in range(info.n_cycles):
            n_steps = info.cycle_n_steps.get(cycle, 0)
            if n_steps == 0:
                continue
            for element_type in ELEMENT_TYPES:
                for y_property in self._grouped_pairs(element_type):
                    y_static = self._static_values(element_type, y_property)
                    low, high = self._value_range(
                        readers.get(element_type), element_type, y_property, y_static, step_offset, n_steps
                    )
                    value_ranges[(cycle, element_type, y_property)] = (low, high)
                    key = (element_type, y_property)
                    log_scales[key] = log_scales.get(key, True) and low > 0.0
            step_offset += n_steps
        return value_ranges, log_scales

    def _grouped_pairs(self, element_type):
        grouped = {}
        for pair_element_type, y_property, color_property in self._pairs:
            if pair_element_type != element_type:
                continue
            colors = grouped.setdefault(y_property, set())
            if color_property is not None:
                colors.add(color_property)
        return grouped

    def _aggregate_one(
        self,
        arrays,
        reader,
        cycle,
        element_type,
        y_property,
        color_properties,
        step_offset,
        n_steps,
        n_bins,
        value_range,
        scales,
    ):
        low, high = value_range
        y_static = self._static_values(element_type, y_property)

        per_scale = {}
        for scale in scales:
            edges = scale_edges(scale, low, high, n_bins)
            per_scale[scale] = {
                "edges": edges,
                "static_bins": self._bin_indices(y_static, edges) if y_static is not None else None,
                "count": np.zeros((n_steps, n_bins), dtype=np.float32),
                "sums": {color: np.zeros((n_steps, n_bins), dtype=np.float32) for color in color_properties},
            }

        for local_step in range(n_steps):
            step = step_offset + local_step
            y_values = None if y_static is not None else self._spooled_values(reader, element_type, y_property, step)
            color_values = {color: self._color_values(reader, element_type, color, step) for color in color_properties}

            for data in per_scale.values():
                if data["static_bins"] is not None:
                    bins, valid = data["static_bins"]
                else:
                    bins, valid = self._bin_indices(y_values, data["edges"])

                data["count"][local_step] = np.bincount(bins[valid], minlength=n_bins)

                for color, values in color_values.items():
                    finite = valid & np.isfinite(values)
                    data["sums"][color][local_step] = np.bincount(
                        bins[finite], weights=values[finite], minlength=n_bins
                    )

        for scale, data in per_scale.items():
            prefix = (cycle, element_type, y_property, scale)
            arrays[self._array_key(*prefix, EDGES_KEY)] = data["edges"]
            arrays[self._array_key(*prefix, COUNT_KEY)] = data["count"]
            for color, matrix in data["sums"].items():
                arrays[self._array_key(*prefix, SUM_KEY, color)] = matrix

    @staticmethod
    def _array_key(cycle, element_type, y_property, scale, kind, color_property=None):
        return cycle_key(cycle, aggregate_key(element_type, y_property, scale, kind, color_property))

    def _static_values(self, element_type, name):
        if not self._is_static(element_type, name):
            return None
        return np.asarray(self._static[f"{element_type}.{name}"], dtype=np.float64).ravel()

    def _color_values(self, reader, element_type, name, step):
        static = self._static_values(element_type, name)
        if static is not None:
            return static
        return self._spooled_values(reader, element_type, name, step)

    def _value_range(self, reader, element_type, y_property, y_static, step_offset, n_steps):
        """Exact range of the binned property, which is what the spool exists for."""
        if y_static is not None:
            finite = y_static[np.isfinite(y_static)]
            return self._non_degenerate(finite)

        low, high = np.inf, -np.inf
        for local_step in range(n_steps):
            values = self._spooled_values(reader, element_type, y_property, step_offset + local_step)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            low = min(low, float(finite.min()))
            high = max(high, float(finite.max()))
        if not np.isfinite(low):
            return 0.0, 1.0
        return self._non_degenerate(np.array([low, high]))

    @staticmethod
    def _non_degenerate(finite):
        if finite.size == 0:
            return 0.0, 1.0
        low, high = float(finite.min()), float(finite.max())
        if high <= low:
            high = low + 1.0
        return low, high

    @staticmethod
    def _bin_indices(values, edges):
        """Bin membership with np.histogram semantics: last bin closed, outside excluded."""
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
        bins = np.searchsorted(edges, values, side="right") - 1
        np.clip(bins, 0, edges.size - 2, out=bins)
        return bins.astype(np.intp), valid


class _SpoolReader:
    """Per-step views into the spooled scalars of one element type."""

    def __init__(self, cache, name, n_elements, names):
        self._rows = cache.rows(name, n_elements, len(names), np.float32)
        self._n_elements = n_elements
        self._index = {name: position for position, name in enumerate(names)}

    def values(self, name, step):
        position = self._index[name]
        return self._rows[step, position * self._n_elements : (position + 1) * self._n_elements]


class DiagnosticsAggregates:
    """Reads the precomputed aggregates and rebins them to the requested resolution."""

    def __init__(self, path):
        self.path = path
        self._manifest = read_manifest(path) or {}
        self._aggregates = self._manifest.get("aggregates", {})
        self._arrays = None

    @staticmethod
    def exists(path):
        return os.path.exists(os.path.join(path, AGGREGATE_FILE_NAME))

    @property
    def cycle_n_steps(self):
        return {int(cycle): n for cycle, n in self._manifest.get("cycle_n_steps", {}).items()}

    @property
    def n_cycles(self):
        return len(self.cycle_n_steps)

    def element_types(self):
        return sorted(self._aggregates)

    def step_axis(self, cycle, name):
        """Per step values of a simulation scalar over one cycle, or None when not stored."""
        try:
            return self._array(step_axis_key(cycle, name))
        except KeyError:
            return None

    def y_property_names(self, element_type):
        return sorted(self._aggregates.get(element_type, {}))

    def _entry(self, element_type, y_property):
        return self._aggregates.get(element_type, {}).get(y_property, {})

    def color_property_names(self, element_type, y_property):
        return list(self._entry(element_type, y_property).get("colors", []))

    def scales(self, element_type, y_property):
        return list(self._entry(element_type, y_property).get("scales", [SCALE_LINEAR]))

    def log_scale_available(self, element_type, y_property):
        """Log bins are only stored for properties that stay positive through the run."""
        return SCALE_LOG in self.scales(element_type, y_property)

    def stored_n_bins(self, cycle, element_type, y_property, scale=SCALE_LINEAR):
        return self._array(cycle_key(cycle, aggregate_key(element_type, y_property, scale, COUNT_KEY))).shape[1]

    def value_range(self, cycle, element_type, y_property, scale=SCALE_LINEAR):
        edges = self._array(cycle_key(cycle, aggregate_key(element_type, y_property, scale, EDGES_KEY)))
        return float(edges[0]), float(edges[-1])

    def matrix(
        self,
        cycle,
        element_type,
        y_property,
        color_property,
        aggregation,
        n_bins,
        value_range=None,
        scale=SCALE_LINEAR,
    ):
        """A (n_steps, n_bins) matrix, rebinned from the stored fine bins of `scale`."""
        edges = self._array(cycle_key(cycle, aggregate_key(element_type, y_property, scale, EDGES_KEY)))
        count = self._array(cycle_key(cycle, aggregate_key(element_type, y_property, scale, COUNT_KEY)))

        low, high = (float(edges[0]), float(edges[-1])) if value_range is None else value_range
        coarse_edges = scale_edges(scale, *self._usable_range(low, high, scale), n_bins)
        merge = self._merge_matrix(edges, coarse_edges)
        resolved_range = (float(coarse_edges[0]), float(coarse_edges[-1]))

        coarse_count = count @ merge
        if aggregation == AGG_COUNT or color_property is None:
            return coarse_count, resolved_range

        total = self._array(cycle_key(cycle, aggregate_key(element_type, y_property, scale, SUM_KEY, color_property)))
        coarse_sum = total @ merge
        if aggregation == AGG_SUM:
            return np.where(coarse_count > 0, coarse_sum, np.nan), resolved_range

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(coarse_count > 0, coarse_sum / coarse_count, np.nan)
        return mean, resolved_range

    @staticmethod
    def _usable_range(low, high, scale):
        if scale == SCALE_LOG:
            if low <= 0.0:
                low = np.nextafter(0.0, 1.0)
            if high <= low:
                high = low * 10.0
        elif high <= low:
            high = low + 1.0
        return low, high

    @staticmethod
    def _merge_matrix(edges, coarse_edges):
        """Maps the stored fine bins onto the requested coarse bins, by fine bin centre.

        searchsorted rather than arithmetic, so the same code merges linear and log bins.
        """
        centers = (edges[:-1] + edges[1:]) / 2.0
        inside = (centers >= coarse_edges[0]) & (centers <= coarse_edges[-1])
        coarse = np.searchsorted(coarse_edges, centers, side="right") - 1
        np.clip(coarse, 0, coarse_edges.size - 2, out=coarse)

        merge = np.zeros((centers.size, coarse_edges.size - 1), dtype=np.float64)
        merge[np.nonzero(inside)[0], coarse[inside]] = 1.0
        return merge

    def _array(self, key):
        if self._arrays is None:
            with np.load(os.path.join(self.path, AGGREGATE_FILE_NAME)) as data:
                self._arrays = {name: data[name] for name in data.files}
        return self._arrays[key]
