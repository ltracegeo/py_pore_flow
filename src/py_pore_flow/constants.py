import numpy as np
g = 9.80665  # Gravitational acceleration

pw = 1000  # water_density
po = 900  # oil_density

sigma = 0.03  # interfacial_tension

miw = 0.00105  # water viscosity
mio = 0.00139  # oil viscosity

pore_fill_alg = "oren2"
pore_fill_weights = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)

TRIANGLE_THRESHOLD = np.sqrt(3) / 36 # Triangle if lower than
SQUARE_THRESHOLD = 1/16 # Square if lower than, and not triangle
# Circle otherswise

TRIANGLE = 0
SQUARE = 1
CIRCLE = 2

HALF_RIGHT_ANGLE = np.deg2rad(45)
