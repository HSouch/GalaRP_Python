
import galarp as grp

import gala.potential as gp

class Satellite:

    def __init__(self, particles, densities, potential):
        
        self.particles = particles
        self.densities = densities
        self.potential = potential
        