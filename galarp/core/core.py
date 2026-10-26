"""Classes for implementing orbits under ram pressure acceleration."""

import galarp

# Third Party
import astropy.units as u
import numpy as np

import gala
import gala.dynamics as gd
import gala.integrate as gi



class RPSim:
    def __init__(
        self,
        wind,
        satellite,
        density,
        shadow=None,
        acceleration=None

    ):
        self.wind = wind
        self.satellite = satellite
        self.density = density

        if acceleration is None:
            self.acceleration = galarp.Acceleration(
                satellite=satellite,
                potential=satellite.potential,
                shadow=shadow,
                wind=wind,
                rho=density
            )

    def run(self, dt=5 * u.Myr, t_int=500 * u.Myr):
        self.satellite.particles.gen_container()


        integrator = gi.RK5Integrator(
            self.acceleration.apply_force,
            
            func_units=gala.units.galactic,
            progress=False,
        )

        orbits = integrator(
            self.satellite.particles.container, dt=dt, t1=0, t2=t_int
        )

        metadata = {
        }

        out = OrbitContainer(orbits, units=galarp.units.galactic, metadata=metadata)

        return out



class OrbitContainer:
    """ Container for storing orbits and metadata from a completed GalaRP run. """
    def __init__(self, data, units=None, metadata={}):
        self.data = data
        self.units = units

        self.metadata = metadata
    
    def get_orbit_data(self, length_units = u.kpc, vel_units = u.km/u.s, transposed=True):
        """ Return the position and velocity data as explicit arrays. """
        pos, vel = self.data.pos, self.data.vel

        x, y, z = pos.xyz.to(length_units).value
        vx, vy, vz = vel.d_xyz.to(vel_units).value
        
        if transposed:
            x, y, z = x.T, y.T, z.T
            vx, vy, vz = vx.T, vy.T, vz.T

        return x, y, z, vx, vy, vz

