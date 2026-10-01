import json
import os

import numpy as np

NETWORK_JSON_NAME = "network.json"
NETWORK_ARRAYS_NAME = "network.npz"
NETWORK_EXTRAS_NAME = "network_extras.json"

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


def export_network(pn, path):
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


def load_network(path):
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
