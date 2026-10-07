
import galarp
from astropy import units as u

import numpy as np


class Density:
    """Parent class to represent time-variable densities."""

    def __init__(self, units=galarp.units.galactic):
        self.units = units

    def evaluate(self, t):
        raise NotImplementedError("Must be implemented in a subclass")
    
    def evaluate_arr(self, ts):
        return np.array([self.evaluate(t) for t in ts])


class ConstantDensity(Density):
    
    def __init__(self, rho: u.Quantity, units = galarp.units.galactic):
        super().__init__(units=units)
        self.rho = rho.to(self.units.density).value
    
    def evaluate(self, t):
        return self.rho
