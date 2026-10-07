import galarp

from astropy import units as u
import numpy as np

from scipy.interpolate import interp1d


class SigmaGas:
    def __init__(self, units=galarp.units.galactic, **kwargs):
        self.units = units
    
    def evaluate(self, t):
        raise NotImplementedError("Must be implemented in a subclass.")

    def evaluate_arr(self, ts):
        try:
            return np.array([self.evaluate(t) for t in ts])
        except Exception:
            return np.array([self.evaluate(t) for t in ts.value])


class ConstantSigmaGas(SigmaGas):
    def __init__(self, sigma: u.Quantity, units=galarp.units.galactic, **kwargs):
        super().__init__(units=units)

        self.sigma_original = sigma
        self.sigma = self.sigma_original.to(units.surface_density).value
    

    def evaluate(self, t=0):
        return self.sigma


class InterpolatedSigmaGas(SigmaGas):

    def __init__(self, interp, units=galarp.units.galactic, **kwargs):
        super().__init__(units=units)
        self.interp = interp

    def evaluate(self, t):
        val = self.interp(t) * u.Msun / u.kpc ** 2
        return val


class CoolingClouds(InterpolatedSigmaGas):

    @classmethod
    def cooling_period(tmin=0, tmax=500, 
                       density_min=50 * u.Msun / u.pc ** 2, 
                       density_max=200 * u.Msun / u.pc ** 2,
                         **kwargs):

        density_min = density_min.to(u.Msun / u.kpc ** 2).value
        density_max = density_max.to(u.Msun / u.kpc ** 2).value

        growth_time = np.arange(tmin, tmax, 10)
        interp = interp1d(growth_time, 
                  np.linspace(0, 1, len(growth_time)) * (density_max - density_min) + density_min, 
                  bounds_error=False, fill_value=density_max)
        
        return InterpolatedSigmaGas(interp, **kwargs)
