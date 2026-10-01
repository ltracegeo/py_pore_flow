import numpy as np

from py_pore_flow import log


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
