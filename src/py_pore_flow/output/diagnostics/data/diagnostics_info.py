import os

import numpy as np

from py_pore_flow.output.diagnostics.utils.diagnostics_file_cache import FILE_EXTENSION

INFO_FILE_NAME = "info"


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
