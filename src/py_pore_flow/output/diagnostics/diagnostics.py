import numpy as np

from py_pore_flow.output.diagnostics.data.diagnostics_info import (
    DiagnosticsInfo,
    DiagnosticsInfoStorer,
)
from py_pore_flow.output.diagnostics.data.diagnostics_network import export_network, load_network
from py_pore_flow.output.diagnostics.data.diagnostics_pores_cycle import (
    DiagnosticsPoresCycle,
    DiagnosticsPoresCycleFloats,
    DiagnosticsPoresCycleInts,
)
from py_pore_flow.output.diagnostics.data.diagnostics_pores_step import (
    DiagnosticsPoresBools,
    DiagnosticsPoresCornersBools,
    DiagnosticsPoresCornersFloats,
    DiagnosticsPoresFloats,
    DiagnosticsPoresStep,
)
from py_pore_flow.output.diagnostics.data.diagnostics_scalars import (
    DiagnosticsScalars,
    DiagnosticsScalarsCycle,
    DiagnosticsScalarsStep,
)
from py_pore_flow.output.diagnostics.data.diagnostics_throats_cycle import (
    DiagnosticsThroatsCycle,
    DiagnosticsThroatsCycleFloats,
    DiagnosticsThroatsCycleInts,
)
from py_pore_flow.output.diagnostics.data.diagnostics_throats_step import (
    DiagnosticsThroatsBools,
    DiagnosticsThroatsCornersBools,
    DiagnosticsThroatsCornersFloats,
    DiagnosticsThroatsFloats,
    DiagnosticsThroatsStep,
)
from py_pore_flow.output.diagnostics.utils.diagnostics_file_cache import DiagnosticsFileCache


class DiagnosticsWriter:
    def __init__(self, network, path="."):
        self.pn = network.pn_core

        self.cycle_id = 0
        self.n_saved_sates = 0

        n_throats = self.pn.throat.water_saturation.size
        n_pores = self.pn.pore.water_saturation.size

        self.info_storer = DiagnosticsInfoStorer(path)
        self.info_storer.save_number_of_elements(n_throats, n_pores)
        self.step_scalars_storer = DiagnosticsScalars(path)

        self.storer_throats_floats = DiagnosticsThroatsFloats(n_throats, path)
        self.storer_throats_bools = DiagnosticsThroatsBools(n_throats, path)
        self.storer_throats_corners_bool = DiagnosticsThroatsCornersBools(n_throats, path)
        self.storer_throats_corners_float = DiagnosticsThroatsCornersFloats(n_throats, path)
        self.storer_throats_cycle_floats = DiagnosticsThroatsCycleFloats(n_throats, path)
        self.storer_throats_cycle_ints = DiagnosticsThroatsCycleInts(n_throats, path)

        self.storer_pores_floats = DiagnosticsPoresFloats(n_pores, path)
        self.storer_pores_bools = DiagnosticsPoresBools(n_pores, path)
        self.storer_pores_corners_bools = DiagnosticsPoresCornersBools(n_pores, path)
        self.storer_pores_corners_floats = DiagnosticsPoresCornersFloats(n_pores, path)
        self.storer_pores_cycle_floats = DiagnosticsPoresCycleFloats(n_pores, path)
        self.storer_pores_cycle_ints = DiagnosticsPoresCycleInts(n_pores, path)

        export_network(network.pn, path)

    # stores elements properties of a single step
    def save_state(self, pc=np.nan, sw=np.nan, krw=np.nan, kro=np.nan):
        self.n_saved_sates += 1
        self.step_scalars_storer.save_state(pc, sw, krw, kro)

        throat = self.pn.throat
        pore = self.pn.pore

        self.storer_throats_floats.save_state(throat)
        self.storer_throats_bools.save_state(throat)
        self.storer_throats_corners_bool.save_state(throat)
        self.storer_throats_corners_float.save_state(throat)
        self.storer_pores_floats.save_state(pore)
        self.storer_pores_bools.save_state(pore)
        self.storer_pores_corners_bools.save_state(pore)
        self.storer_pores_corners_floats.save_state(pore)

    # stores some elements properties at the end of a cycle
    def save_cycle_data(self):
        self.info_storer.save_number_of_steps(self.cycle_id, self.n_saved_sates)
        self.cycle_id += 1
        self.n_saved_sates = 0

        throat = self.pn.throat
        pore = self.pn.pore

        self.storer_throats_cycle_floats.save_state(throat)
        self.storer_throats_cycle_ints.save_state(throat)
        self.storer_pores_cycle_floats.save_state(pore)
        self.storer_pores_cycle_ints.save_state(pore)

    def close(self):
        self.step_scalars_storer.close()
        self.storer_throats_floats.close()
        self.storer_throats_bools.close()
        self.storer_throats_corners_bool.close()
        self.storer_throats_corners_float.close()
        self.storer_throats_cycle_floats.close()
        self.storer_throats_cycle_ints.close()
        self.storer_pores_floats.close()
        self.storer_pores_bools.close()
        self.storer_pores_corners_bools.close()
        self.storer_pores_corners_floats.close()
        self.storer_pores_cycle_floats.close()
        self.storer_pores_cycle_ints.close()


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
        self.throat = DiagnosticsThroatsCycle(self.path, self.cycle, self.info.n_throats, cache)
        self.pore = DiagnosticsPoresCycle(self.path, self.cycle, self.info.n_pores, cache)
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
            self._network = load_network(self.path)
        return self._network

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
