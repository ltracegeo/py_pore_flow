import numpy as np

# attributes a step/cycle loader sets on itself that are not per-element property data
_LOADER_ATTRS = {"path", "step", "n_elements"}

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

    def __init__(self, diagnostics):
        self._diagnostics = diagnostics

    def get_property_names(self):
        return sorted(set(self._step_scalar_properties()) | set(self._network_properties()))

    def get_pore_property_names(self):
        return [name[len(PORE_PREFIX) :] for name in self.get_property_names() if name.startswith(PORE_PREFIX)]

    def get_throat_property_names(self):
        return [name[len(THROAT_PREFIX) :] for name in self.get_property_names() if name.startswith(THROAT_PREFIX)]

    def get_element_property_names(self):
        pore_names = set(self.get_pore_property_names())
        throat_names = set(self.get_throat_property_names())
        return sorted(pore_names & throat_names)

    def get_property_values(self, property_name, cycle_id=None, step=None):
        return self._read_values(property_name, cycle_id, step)

    def get_pore_property_values(self, property_name, cycle_id=None, step=None):
        return self._read_values(PORE_PREFIX + property_name, cycle_id, step)

    def get_throat_property_values(self, property_name, cycle_id=None, step=None):
        return self._read_values(THROAT_PREFIX + property_name, cycle_id, step)

    def get_element_property_values(self, property_name, cycle_id=None, step=None):
        pore_key = PORE_PREFIX + property_name
        throat_key = THROAT_PREFIX + property_name

        pore_values = self._read_values(pore_key, cycle_id, step) if self._has_values(pore_key) else None
        throat_values = self._read_values(throat_key, cycle_id, step) if self._has_values(throat_key) else None

        if pore_values is not None and throat_values is not None:
            return self._concatenate_element_values(pore_values, throat_values)
        if pore_values is not None:
            return pore_values
        if throat_values is not None:
            return throat_values
        raise KeyError(property_name)

    @staticmethod
    def _concatenate_element_values(pore_values, throat_values):
        if isinstance(pore_values, list) and isinstance(throat_values, list):
            return [np.concatenate([pore, throat]) for pore, throat in zip(pore_values, throat_values)]
        return np.concatenate([pore_values, throat_values])

    def _step_scalar_properties(self):
        info = self._diagnostics.info
        sample_cycle = next((c for c in range(info.n_cycles) if info.cycle_n_steps.get(c, 0) > 0), None)
        if sample_cycle is None:
            return []

        sample_step = self._diagnostics.cycle[sample_cycle][0]
        properties = []
        for prefix, element in (("pore", sample_step.pore), ("throat", sample_step.throat)):
            for attr, value in vars(element).items():
                if attr in _LOADER_ATTRS:
                    continue
                array = np.asarray(value)
                if array.ndim == 1 and (np.issubdtype(array.dtype, np.number) or array.dtype == bool):
                    properties.append(f"{prefix}.{attr}")
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

    def _sample_element(self, prefix):
        info = self._diagnostics.info
        sample_cycle = next((c for c in range(info.n_cycles) if info.cycle_n_steps.get(c, 0) > 0), None)
        if sample_cycle is None:
            return None
        step = self._diagnostics.cycle[sample_cycle][0]
        return step.pore if prefix == "pore" else step.throat

    def _has_values(self, full_name):
        if full_name in self._diagnostics.network:
            return True
        prefix, _, prop = full_name.partition(".")
        element = self._sample_element(prefix)
        return element is not None and hasattr(element, prop)

    def _read_values(self, full_name, cycle_id=None, step=None):
        if cycle_id is None:
            if full_name in self._diagnostics.network:
                return np.asarray(self._diagnostics.network[full_name])
            prefix, _, prop = full_name.partition(".")
            element = self._sample_element(prefix)
            if element is not None and hasattr(element, prop):
                return np.asarray(getattr(element, prop))
            raise KeyError(full_name)

        step_values = self._read_cycle_step_values(full_name, cycle_id)
        if step is not None and step_values is not None:
            return step_values[step]
        return step_values

    def _read_cycle_step_values(self, full_name, cycle_id):
        n_steps = self._diagnostics.info.cycle_n_steps.get(cycle_id, 0)
        if n_steps <= 0:
            return None

        prefix, _, prop = full_name.partition(".")
        sample = self._diagnostics.cycle[cycle_id][0]
        sample_element = sample.pore if prefix == "pore" else sample.throat
        if hasattr(sample_element, prop):
            step_values = []
            for step in range(n_steps):
                cycle_step = self._diagnostics.cycle[cycle_id][step]
                element = cycle_step.pore if prefix == "pore" else cycle_step.throat
                step_values.append(np.asarray(getattr(element, prop), dtype=float).ravel())
            return step_values

        # not a per-step property: use the static value from the network dump for every step
        if full_name not in self._diagnostics.network:
            return None
        values = np.asarray(self._diagnostics.network[full_name], dtype=float).ravel()
        return [values] * n_steps
