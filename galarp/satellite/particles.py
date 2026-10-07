
import galarp

import gala.dynamics as gd

import numpy as np


def default_velocity_init(q, potential, units=galarp.units.galactic, rotation=1, rmin=1e-6):
    m_enc = potential.mass_enclosed(q)

    r = np.sqrt(np.sum(q ** 2))

    if r < rmin:
        return [0, 0, 0]

    inc = np.arctan2(q[2], np.sqrt(q[0]**2 + q[1]**2))

    theta = np.arctan2(q[1], q[0])
    m_enc = potential.mass_enclosed(q).to(units.mass).value

    v_circ = np.sqrt(units.G.value * m_enc / r)[0]
    
    vx = -np.sin(theta) * v_circ * rotation
    vy = np.cos(theta) * v_circ * rotation
    vz = np.sqrt(vx ** 2 + vy ** 2) * np.sin(inc)

    

    return [vx, vy, vz]
    

class ParticleDistribution:
    def __init__(self, units=galarp.units.galactic, **kwargs):
        self.x, self.y, self.z = [], [], []
        self.vx, self.vy, self.vz = [], [], []

        self.container = []

        self.n_particles = 0
        self.units = units


    def generate(self, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass")
    

    def q(self):
        return np.array([self.x, self.y, self.z])
    

    def p(self):
        return np.array([self.vx, self.vy, self.vz])

    
    def R_sph(self):
        return np.sqrt(self.x **2 + self.y**2 + self.z **2)


    def R_cyl(self):
        return np.sqrt(self.x **2 + self.y**2)


    def vtot(self):
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)


    def init_velocities(self, potential, method=default_velocity_init):
    
        qs = self.q().T

        vels = []
        for i in range(len(qs)):
            q = qs[i]
            vels.append(method(q, potential))

        vels = np.array(vels).T
        self.vx, self.vy, self.vz = vels[:]


    def gen_container(self):
        self.container = []

        if len(self.x) != len(self.vx):
            print("Velocities not initialized! Gen before making container")
            return None

        qs, ps = self.q().T, self.p().T

        for i in range(len(self.x)):
            
            w0 = gd.PhaseSpacePosition(
                pos=qs[i] * self.units.length, 
                vel=ps[i] * self.units.velocity)

            self.container.append(w0)

    def plot_xy(self, **kwargs):
        pp = galarp.plotting.ParticlePlotter(self)
        pp.plot_xy(**kwargs)
    

class PlaneDistribution(ParticleDistribution):
    # Class to generate uniformly spaced particles in a plane

    def __init__(self, Rmax=10, zmax=10, spacing=1, **kwargs):
        super().__init__(**kwargs)

        self.Rmax = Rmax
        self.zmax = zmax
        self.spacing = spacing

    def generate(self, positions=None, velocity_dispersion=0):

        self.x, self.y, self.z = [], [], []

        particle_range = np.arange(-self.Rmax, self.Rmax + self.spacing, self.spacing)
        for xi in particle_range:
            for yi in particle_range:
                R = np.sqrt(xi**2 + yi**2)
                if R > self.Rmax:
                    continue
                self.x.append(xi)
                self.y.append(yi)
                self.z.append(0.)
        
        self.n_particles = len(self.x)
        self.x, self.y, self.z = self.q()   # This initializes them properly as numpy arrays

