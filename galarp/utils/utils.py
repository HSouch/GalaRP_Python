import numpy as np
from astropy import units as u

def to_value(x, unit):
    """Strip astropy units (if any) into the simulation's unit system."""
    if isinstance(x, u.Quantity):
        return x.to(unit).value
    if isinstance(x, (list, tuple)) and len(x) and isinstance(x[0], u.Quantity):
        return np.array([xi.to(unit).value for xi in x])
    return np.asarray(x, dtype=float)