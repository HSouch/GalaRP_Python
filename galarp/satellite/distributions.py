
import os

from astropy import units as u

import numpy as np

from gala.units import galactic

from .. import utils


class ParticleDistribution:

    def __init__(self, units=galactic, **kwargs):
        self.x, self.y, self.z = [], [], []
        self.n_particles = 0
        self.units = units

    def generate(self, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass.")
    
    # def plot_positions(self, **kwargs):
    #     general_plots.plot_density([self.x, self.y, self.z], **kwargs)

    def save(self, outname):
        outdir = os.path.dirname(outname)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        utils.pickle_obj(self, outname)
    
    def save_positions(self, outname="positions.npy"):
        outdir = os.path.dirname(outname)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        np.save(outname, np.array([self.x, self.y, self.z]))
    
    @staticmethod
    def from_file(filename, **kwargs):
        x, y, z = np.load(filename)

        pd = ParticleDistribution(gen_positions=False, **kwargs)
        pd.x, pd.y, pd.z = x * pd.units["length"], y * pd.units["length"], z * pd.units["length"]
        pd.n_particles = len(x)

        return pd


class PlaneDistribution(ParticleDistribution):
    # Class to generate uniformly spaced particles in a plane

    def __init__(self, Rmax=10, zmax=10, spacing=1, gen_positions=True, **kwargs):
        super().__init__(**kwargs)

        self.Rmax = Rmax
        self.zmax = zmax
        self.spacing = spacing
        
        if gen_positions:
            self.generate()
    
    def generate(self, positions=None, velocity_dispersion=0):
        particle_range = np.arange(-self.Rmax, self.Rmax + self.spacing, self.spacing) * u.kpc
        for xi in particle_range:
            for yi in particle_range:
                R = np.sqrt(xi**2 + yi**2)
                if R > self.Rmax * u.kpc:
                    continue
                self.x.append(xi.to(self.units["length"]).value)
                self.y.append(yi.to(self.units["length"]).value)
                self.z.append(0.)




