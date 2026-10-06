"""Classes for implementing orbits under ram pressure acceleration."""

# Third Party
import os
import pickle
import astropy.units as u
import numpy as np

import gala.dynamics as gd
import gala.integrate as gi
from gala.units import galactic


# import integrate as gi            # If debugging
import gala.integrate as gi


def F_RPS_Surface_Density(
    t, w, potential, shadow, wind, rho, sigma_gas, wind_on=True, debug=False):
    # By this point, the system should now be in a harmonized unit set

    # position units are in kpc
    # velocity units here are in kpc/Myr
    x, y, z, vx, vy, vz = w
    q = np.stack((x, y, z), axis=1)
    p = np.stack((vx, vy, vz), axis=1)

    # compute acceleration from potential:
    _t = np.array([0.0])
    acc = -potential._gradient(q, _t).T

    # Compute acceleration from ram pressure
    # a_ram = rho * v_perp^2 / Sigma_cloud
    # TODO remove units altogether
    if wind_on:
        v_perp = wind.evaluate(t) - p
        a_ram = (v_perp**2) * (rho.evaluate(t) / sigma_gas.evaluate(t)).to(1 / u.kpc).value[:, np.newaxis] * np.sign(v_perp)

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
        satpot,
        shadow=None,
        method=F_RPS_Surface_Density 
    ):
        self.wind = wind
        self.satpot = satpot

        self.shadow = shadow


    def run():
        pass    


