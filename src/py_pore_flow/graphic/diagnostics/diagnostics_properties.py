import numpy as np

from py_pore_flow.output.diagnostics.utils.aggregate import DIRECTIONAL_SPECS, throat_axis_angles, throat_axis_cosines

# attributes a step/cycle loader sets on itself that are not per-element property data
_LOADER_ATTRS = {"path", "step", "n_elements", "_cache"}

PORE_PREFIX = "pore."
THROAT_PREFIX = "throat."


class DiagnosticsProperties:
    """Property catalogue for a :class:`Diagnostics`.

    Exposes the plottable per-element properties (from both the per-step loaders and the
    static ``diagnostics.network`` dump) and their value arrays. Names are the ``pore.<x>`` /
    ``throat.<y>`` union; per-element helpers strip the prefix. Values come from
    ``diagnostics.network`` (the current applied state), falling back to a sample step loader
    for properties not (yet) copied into the network dump.
    """

    CONVERSORS = {
        "oil_filled": ["water_filled", lambda x: ~x],
        "oil_saturation": ["water_saturation", lambda x: 1.0 - x],
        "x": ["coords", lambda x: x[:, 0]],
        "y": ["coords", lambda x: x[:, 1]],
        "z": ["coords", lambda x: x[:, 2]],
    }

    def __init__(self, diagnostics):
        self._diagnostics = diagnostics
        self._cycle_id = 0
        self._step = 0
        self._property_names = None
        self._axis_cosines = None
        self._axis_cosines_loaded = False

    def __getitem__(self, key):
        return self.get_values_in_step(key, self._cycle_id, self._step)

    def set_cycle(self, cycle_id):
        self._cycle_id = cycle_id

    def set_step(self, step):
        self._step = step

    def get_property_names(self):
        # recomputed on every element type change by both the heatmap and the cross plot
        if self._property_names is None:
            step_names = set(self._step_scalar_properties())
            names = step_names | set(self._cycle_scalar_properties()) | set(self._network_properties())
            names |= set(self._directional_property_names(step_names))
            self._property_names = sorted(names)
        return self._property_names

    def get_pore_property_names(self):
        return [name[len(PORE_PREFIX) :] for name in self.get_property_names() if name.startswith(PORE_PREFIX)]

    def get_throat_property_names(self):
        return [name[len(THROAT_PREFIX) :] for name in self.get_property_names() if name.startswith(THROAT_PREFIX)]

    def get_element_property_names(self):
        pore_names = set(self.get_pore_property_names())
        throat_names = set(self.get_throat_property_names())
        return sorted(pore_names & throat_names)

    def get_values_in_step(self, property_name, cycle_id, step):
        prefix, _, prop = property_name.partition(".")

        directional = self._directional_values(prefix, prop, cycle_id, step)
        if directional is not None:
            return directional

        prop, conversor = self._get_conversor(prop)

        values = self._sample_step_element(prefix, prop, cycle_id, step)
        if values is not None:
            return conversor(values)
        values = self._sample_cycle_element(prefix, prop, cycle_id)
        if values is not None:
            return conversor(values)
        try:
            values = self._diagnostics.network[f"{prefix}.{prop}"]
            return conversor(values)
        except KeyError:
            raise KeyError(property_name)

    def get_pore_values_in_step(self, property_name, cycle_id, step):
        return self.get_values_in_step(PORE_PREFIX + property_name, cycle_id, step)

    def get_throat_values_in_step(self, property_name, cycle_id, step):
        return self.get_values_in_step(THROAT_PREFIX + property_name, cycle_id, step)

    def get_element_values_in_step(self, property_name, cycle_id, step):
        pore_values = self._try_values_in_step(PORE_PREFIX + property_name, cycle_id, step)
        throat_values = self._try_values_in_step(THROAT_PREFIX + property_name, cycle_id, step)

        if pore_values is not None and throat_values is not None:
            return self._concatenate_element_values(pore_values, throat_values)
        if pore_values is not None:
            return pore_values
        if throat_values is not None:
            return throat_values
        raise KeyError(property_name)

    def _try_values_in_step(self, property_name, cycle_id, step):
        try:
            return self.get_values_in_step(property_name, cycle_id, step)
        except KeyError:
            return None

    def get_values_of_cycle(self, property_name, cycle_id):
        n_steps = self._diagnostics.info.cycle_n_steps[cycle_id]
        cycle_data = []
        for step in range(n_steps):
            cycle_data.append(self.get_values_in_step(property_name, cycle_id, step))
        return cycle_data

    def get_pore_values_of_cycle(self, property_name, cycle_id):
        return self.get_values_of_cycle(PORE_PREFIX + property_name, cycle_id)

    def get_throat_values_of_cycle(self, property_name, cycle_id):
        return self.get_values_of_cycle(THROAT_PREFIX + property_name, cycle_id)

    def get_element_values_of_cycle(self, property_name, cycle_id):
        pore_values = self._try_values_of_cycle(PORE_PREFIX + property_name, cycle_id)
        throat_values = self._try_values_of_cycle(THROAT_PREFIX + property_name, cycle_id)

        if pore_values and throat_values:
            return self._concatenate_element_values(pore_values, throat_values)
        if pore_values:
            return pore_values
        if throat_values:
            return throat_values
        raise KeyError(property_name)

    def _try_values_of_cycle(self, property_name, cycle_id):
        try:
            return self.get_values_of_cycle(property_name, cycle_id)
        except KeyError:
            return None

    @staticmethod
    def _concatenate_element_values(pore_values, throat_values):
        if isinstance(pore_values, list) and isinstance(throat_values, list):
            return [np.concatenate([pore, throat]) for pore, throat in zip(pore_values, throat_values)]
        return np.concatenate([pore_values, throat_values])

    def _sample_cycle(self):
        """First cycle that actually ran, or None when nothing was saved."""
        info = self._diagnostics.info
        return next((c for c in range(info.n_cycles) if info.cycle_n_steps.get(c, 0) > 0), None)

    @staticmethod
    def _scalar_attributes(prefix, element):
        """The 1-D numeric/bool arrays a loader exposes, as "<prefix>.<attribute>" names."""
        names = []
        for attr, value in vars(element).items():
            if attr in _LOADER_ATTRS:
                continue
            array = np.asarray(value)
            if array.ndim == 1 and (np.issubdtype(array.dtype, np.number) or array.dtype == bool):
                names.append(f"{prefix}.{attr}")
        return names

    def _throat_axis_cosines(self):
        """Unsigned direction cosines of every throat, read from the network dump once."""
        if not self._axis_cosines_loaded:
            self._axis_cosines = throat_axis_cosines(self._diagnostics.network)
            self._axis_cosines_loaded = True
        return self._axis_cosines

    def _directional_values(self, prefix, prop, cycle_id, step):
        """A throat flow projected onto a coordinate axis, or the throat angle itself."""
        spec = DIRECTIONAL_SPECS.get(prop) if prefix == THROAT_PREFIX[:-1] else None
        if spec is None:
            return None

        cosines = self._throat_axis_cosines()
        if cosines is None:
            return None

        source, axis = spec
        if source is None:
            return throat_axis_angles(cosines[:, axis])

        values = np.asarray(self.get_values_in_step(THROAT_PREFIX + source, cycle_id, step), dtype=float)
        return values * cosines[:, axis]

    def _directional_property_names(self, step_names):
        if self._throat_axis_cosines() is None:
            return []
        return [
            THROAT_PREFIX + name
            for name, (source, _axis) in DIRECTIONAL_SPECS.items()
            if source is None or THROAT_PREFIX + source in step_names
        ]

    def _step_scalar_properties(self):
        sample_cycle = self._sample_cycle()
        if sample_cycle is None:
            return []

        sample_step = self._diagnostics.cycle[sample_cycle][0]
        properties = []
        for prefix, element in (("pore", sample_step.pore), ("throat", sample_step.throat)):
            properties.extend(self._scalar_attributes(prefix, element))

            for prop in self.CONVERSORS.keys():
                property_name = f"{prefix}.{prop}"
                try:
                    self.get_values_in_step(property_name, sample_cycle, 0)  # Check if property exists
                    properties.append(property_name)
                except KeyError:
                    pass

        return sorted(properties)

    def _cycle_scalar_properties(self):
        """Per-cycle properties (entry pressure, contact angles, invasion method).

        get_values_in_step already resolves these through _sample_cycle_element, but they
        live on the cycle loader rather than the step loader so nothing listed them and they
        were unselectable in every combo.
        """
        sample_cycle = self._sample_cycle()
        if sample_cycle is None:
            return []

        cycle = self._diagnostics.cycle[sample_cycle]
        properties = []
        for prefix, element in (("pore", cycle.pore), ("throat", cycle.throat)):
            properties.extend(self._scalar_attributes(prefix, element))
        return sorted(properties)

    def _network_properties(self):
        info = self._diagnostics.info
        counts = {"pore": info.n_pores, "throat": info.n_throats}
        properties = []
        for key, value in self._diagnostics.network.items():
            prefix, _, name = key.partition(".")
            if prefix not in counts or not name:
                continue
            try:
                array = np.asarray(value)
            except ValueError:
                continue
            is_number = np.issubdtype(array.dtype, np.number) or array.dtype == bool
            if array.ndim == 1 and array.shape[0] == counts[prefix] and is_number:
                properties.append(key)
        return properties

    def _sample_step_element(self, prefix, prop, cycle=0, step=0):
        step = self._diagnostics.cycle[cycle][step]
        element = step.pore if prefix == "pore" else step.throat
        if hasattr(element, prop):
            return getattr(element, prop)
        else:
            return None

    def _sample_cycle_element(self, prefix, prop, cycle=0):
        step = self._diagnostics.cycle[cycle]
        element = step.pore if prefix == "pore" else step.throat
        if hasattr(element, prop):
            return getattr(element, prop)
        else:
            return None

    def _get_conversor(self, prop):
        try:
            return (*self.CONVERSORS[prop],)
        except KeyError:
            return prop, lambda x: x
