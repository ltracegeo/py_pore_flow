import os
import re

import numpy as np

from py_pore_flow.output.diagnostics.utils.concatenated_memmaps import ConcatenatedMemMaps

FILE_EXTENSION = ".dat"


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
