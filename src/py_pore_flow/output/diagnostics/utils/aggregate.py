import numpy as np

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


def throat_axis_angles(cosines):
    """Angle in degrees between each throat and each axis, in [0, 90]."""
    return np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0)))


def throat_axis_cosines(network):
    """Unsigned direction cosine of every throat against each axis, or None.

    Unsigned because which of its two pores a throat is measured from is arbitrary, so only
    the axis it lies along is meaningful, not the direction along it.
    """
    unit = _throat_unit_vectors(network)
    return None if unit is None else np.abs(unit)


def _throat_unit_vectors(network):
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

    conns = _zero_based_conns(conns, coords.shape[0])
    if conns is None:
        return None

    vectors = coords[conns[:, 1], :3] - coords[conns[:, 0], :3]
    lengths = np.linalg.norm(vectors, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = vectors / lengths[:, None]
    unit[~np.isfinite(unit)] = np.nan
    return unit


def _zero_based_conns(conns, n_pores):
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
