import galarp

from astropy import units as u

import numpy as np


class RPWind:

    def __init__(self, units=galarp.units.galactic, **kwargs):
        self.units=units


    def evaluate(self, t):
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def evaluate_arr(self, ts):
        return np.array([self.evaluate(t) for t in ts])
    

    def speed(self, t):
        return np.linalg.norm(self.evaluate(t))

    
    def unit_vector(self, t):
        vec = self.evaluate(t)
        return vec / np.linalg.norm(vec)
    

class ConstantWind(RPWind):
    # A constant strength wind in the y-z direction, similar to typical Enzo 
    # wind-tunnel simulations

    def __init__(self, dwa_deg, strength: u.Quantity, units=galarp.units.galactic, **kwargs):
        super().__init__(units=units)

        self.dwa = np.deg2rad(dwa_deg)  # Disk-wind angle
        self.wind_vector = np.array([0, np.sin(self.dwa), np.cos(self.dwa)])
        self.strength = strength.to(self.units.velocity).value

    
    def evaluate(self, t=0):
        return self.wind_vector * self.strength


class LorentzianWind(ConstantWind):

    def __init__(self, strength = 500 * u.km / u.s, dwa_deg=90,
        t0: u.Quantity = 200 * u.Myr, width: u.Quantity = 200 * u.Myr, 
        units=galarp.units.galactic):

        super().__init__(dwa_deg=dwa_deg, strength=strength, units=units)

        self.t0 = t0.to(self.units.time).value
        self.width = width.to(self.units.time).value
    

    def evaluate(self, t):
        return super().evaluate(t) / ((2 * (t - self.t0) / self.width) ** 2 + 1)
