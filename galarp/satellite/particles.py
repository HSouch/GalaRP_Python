
import galarp


import numpy as np


class ParticleDistribution:

    def __init__(self, units=galarp.galactic, **kwargs):
        self.x, self.y, self.z = [], [], []
        self.n_particles = 0
        self.units = units 
    

    def generate(self, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass.")

    
    def init_circular_motions(self, potential):
        pass


class PlaneDistribution(ParticleDistribution):

    def __init__(self, Rmax=10, zmax=10, spacing=1, **kwargs):
        super().__init__(**kwargs)

        self.Rmax = Rmax
        self.zmax = zmax
        self.spacing = spacing
    

    def generate(self, positions=None, velocity_dispersion=0):
        particle_range = np.arange(-self.Rmax, self.Rmax + self.spacing, self.spacing)
        for xi in particle_range:
            for yi in particle_range:
                R = np.sqrt(xi**2 + yi**2)
                if R > self.Rmax:
                    continue
                self.x.append(xi)
                self.y.append(yi)
                self.z.append(0.)
    
