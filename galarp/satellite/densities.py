from astropy import units as u
import numpy as np

from scipy.interpolate import interp1d

from gala.units import galactic


class SigmaGas:

    def evaluate(self, t):
        return self.sigma

    def evaluate_arr(self, ts):
        try:
            return np.array([self.evaluate(t) for t in ts])
        except Exception:
            return np.array([self.evaluate(t) for t in ts.value])
        
    def __init__(self, sigma, nparticles = 1, units=galactic, **kwargs):
        if isinstance(sigma, SigmaGas):
            self.sigma = sigma.sigma
            self.nparticles = sigma.nparticles
        else:
            self.sigma = sigma
            self.nparticles = nparticles
        self.units = units["mass"] / units["length"]**2

        if not isinstance(self.sigma, u.Quantity):
            self.sigma *= self.units
        if self.sigma.isscalar:
            self.sigma = [self.sigma.to(self.units).value for i in range(nparticles)] * self.units

        self.sigma = self.sigma.to(self.units)
        self.nparticles = len(self.sigma)

    @staticmethod
    def load_spherical(mass, radius, n_particles=1):
        return SigmaGas(sigma = mass / (np.pi * radius**2), nparticles = n_particles)


class InterpolatedSigmaGas(SigmaGas):

    def __init__(self, interp, units="galactic", **kwargs):
        super().__init__(sigma=1e-27, nparticles=kwargs.get("nparticles", 1))

        self.interp = interp

    def evaluate(self, t):
        val = self.interp(t) * u.Msun / u.kpc ** 2
        return [val for i in range(self.nparticles)] * self.units
    
    def evaluate_arr(self, ts):
        try:
            return np.array([self.evaluate(t) for t in ts])
        except Exception:
            return np.array([self.evaluate(t) for t in ts.value])


class CoolingClouds(InterpolatedSigmaGas):

    @staticmethod
    def cooling_period(tmin=0, tmax=500, 
                       density_min=50 * u.Msun / u.pc ** 2, 
                       density_max=200 * u.Msun / u.pc ** 2,
                       units=galactic,
                         **kwargs):

        density_min = density_min.to(u.Msun / u.kpc ** 2).value
        density_max = density_max.to(u.Msun / u.kpc ** 2).value

        growth_time = np.arange(tmin, tmax, 10)
        interp = interp1d(growth_time, 
                  np.linspace(0, 1, len(growth_time)) * (density_max - density_min) + density_min, 
                  bounds_error=False, fill_value=density_max)
        
        return InterpolatedSigmaGas(interp, **kwargs)
