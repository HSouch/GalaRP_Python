"""Classes for implementing orbits under ram pressure acceleration."""

import galarp

# Third Party
import astropy.units as u
import numpy as np

import gala
import gala.dynamics as gd
import gala.integrate as gi


def F_RPS_Surface_Density(
    t, w, potential, shadow, wind, rho, sigma_gas, wind_on=True, debug=False):
    x, y, z, vx, vy, vz = w

    q = np.stack((x, y, z), axis=1)
    p = np.stack((vx, vy, vz), axis=1)


    # compute acceleration from potential:
    _t = np.array([0.0])
    acc = -potential._gradient(q, _t).T


    # Compute acceleration from ram pressure
    # a_ram = rho * v_perp^2 / Sigma_cloud
    if wind_on:
        v_rel = wind.evaluate(t) - p

        a_ram = (rho.evaluate(t) / sigma_gas.evaluate(t)) 
        a_ram *= np.linalg.norm(v_rel, axis=1, keepdims=True) * v_rel
    
       
       # a_ram = v_rel * np.abs(v_rel) * (rho.evaluate(t) / sigma_gas.evaluate(t))

        a_ram = a_ram.T

        # If wind is on and shadow exists, apply shadow to appropriate particles
        if shadow is not None:
            shadow = shadow.evaluate(q, p, t).T
            a_ram *= shadow

        acc += a_ram

    return np.vstack((p.T, acc))


class RPSim:
    def __init__(
        self,
        wind,
        satellite,
        density,
        method=F_RPS_Surface_Density
    ):
        self.wind = wind
        self.satellite = satellite
        self.density = density

        self.method=method

    def run(self, dt=5 * u.Myr, t_int=500 * u.Myr):
        self.satellite.particles.gen_container()

        func_args = (self.satellite.potential, None, self.wind, self.density, self.satellite.densities)


        integrator = gi.RK5Integrator(
            self.method,
            func_args=func_args,
            func_units=gala.units.galactic,
            progress=False,
        )

        orbits = integrator(
            gd.combine(self.satellite.particles.container), dt=dt, t1=0, t2=t_int
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

