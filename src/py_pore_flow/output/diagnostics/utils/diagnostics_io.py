import os

import numpy as np

from py_pore_flow.equations.geometry import MAX_NUMBER_OF_CORNERS
from py_pore_flow.output.diagnostics.utils.diagnostics_file_cache import FILE_EXTENSION, DiagnosticsFileCache

# trailing per-element shape of the corner layer arrays (one value per corner)
CORNER_SHAPE = (MAX_NUMBER_OF_CORNERS,)


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

    def _save_state(self, arrays):
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
