
from matplotlib import pyplot as plt


class ParticlePlotter:
    def __init__(self, particles):
        self.particles = particles
    

    def plot_xy(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        ax.scatter(
            self.particles.x, self.particles.y,
            **kwargs
        )
    

    def plot_xz(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        ax.scatter(
            self.particles.x, self.particles.z,
            **kwargs
        )

    def plot_yz(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        ax.scatter(
            self.particles.y, self.particles.z,
            **kwargs
        )


