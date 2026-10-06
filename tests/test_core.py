import numpy as np
import astropy.units as u
import gala.potential as gp
from gala.units import galactic

import galarp as grp
from galarp.core import F_RPS_Surface_Density


class ConstWind:
    """Stub wind: uniform velocity (kpc/Myr) for every particle."""
    def __init__(self, v):
        self.v = np.asarray(v, float)

    def evaluate(self, t):
        return self.v


class Const:
    def __init__(self, val):
        self.val = val

    def evaluate(self, t):
        return self.val


def _w(n, v=(0, 0, 0)):
    x = np.linspace(1, 5, n)
    z = np.zeros(n)
    return np.array([x, z, z, z + v[0], z + v[1], z + v[2]])


def _pot():
    return gp.NullPotential(units=galactic)


def test_no_wind_is_pure_gravity():
    pot = gp.HernquistPotential(1e10 * u.Msun, 1 * u.kpc, units=galactic)
    w = _w(3)
    out = F_RPS_Surface_Density(0., w, pot, None, None, None, None, wind_on=False)
    q = np.ascontiguousarray(w[:3].T)
    expected = -pot._gradient(q, np.array([0.])).T
    np.testing.assert_allclose(out[3:], expected)
    np.testing.assert_allclose(out[:3], w[3:])


def test_ram_acceleration_magnitude_and_direction():
    n = 4
    rho = 1e3 * u.Msun / u.kpc**3
    sigma = grp.particles.SigmaGas(1e7 * u.Msun / u.kpc**2, nparticles=n)
    vw = 0.5  # kpc/Myr
    out = F_RPS_Surface_Density(
        0., _w(n), _pot(), None, ConstWind([vw, 0, 0]),
        Const(np.full(n, 1e3) * u.Msun / u.kpc**3), sigma)
    expected = vw**2 * 1e3 / 1e7
    np.testing.assert_allclose(out[3], expected)       # along wind
    np.testing.assert_allclose(out[4:], 0, atol=1e-30)


def test_ram_acceleration_sign_follows_relative_velocity():
    n = 2
    sigma = grp.particles.SigmaGas(1e7 * u.Msun / u.kpc**2, nparticles=n)
    rho = Const(np.full(n, 1e3) * u.Msun / u.kpc**3)
    # particle moving faster than the wind -> drag decelerates it
    out = F_RPS_Surface_Density(0., _w(n, v=(1, 0, 0)), _pot(), None,
                                ConstWind([0.2, 0, 0]), rho, sigma)
    assert np.all(out[3] < 0)


def test_shadow_masks_acceleration():
    n = 3
    sigma = grp.particles.SigmaGas(1e7 * u.Msun / u.kpc**2, nparticles=n)
    rho = Const(np.full(n, 1e3) * u.Msun / u.kpc**3)

    class Shadow:
        def evaluate(self, q, p, t):
            return np.array([[1, 0, 1]] * 3, float).T  # (n,3); particle 1 shadowed

    out = F_RPS_Surface_Density(0., _w(n), _pot(), Shadow(),
                                ConstWind([0.5, 0, 0]), rho, sigma)
    assert out[3, 1] == 0 and out[3, 0] > 0 and out[3, 2] > 0


def test_sigmagas_scalar_broadcast_and_units():
    s = grp.particles.SigmaGas(1e7 * u.Msun / u.kpc**2, nparticles=5)
    assert s.nparticles == 5
    assert s.sigma.unit == u.Msun / u.kpc**2


def test_sigmagas_load_spherical():
    s = grp.particles.SigmaGas.load_spherical(1e9 * u.Msun, 2 * u.kpc)
    np.testing.assert_allclose(s.sigma.value[0], 1e9 / (np.pi * 4))


def test_plane_distribution():
    d = grp.particles.PlaneDistribution(Rmax=3, spacing=1)
    assert len(d.x) == len(d.y) == len(d.z)
    assert max(np.hypot(d.x, d.y)) <= 3
    assert d.n_particles == len(d.x)          # currently fails: never set


def test_save_positions_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = grp.particles.PlaneDistribution(Rmax=2, spacing=1)
    d.save_positions()                         # default "positions.npy"


def test_pickle_obj_exported_and_save(tmp_path):
    d = grp.particles.PlaneDistribution(Rmax=2, spacing=1)
    d.save(str(tmp_path / "sub" / "d.pkl"))


def test_cooling_clouds_returns_per_particle_sigma():
    c = grp.particles.CoolingClouds.cooling_period(nparticles=3)
    v = c.evaluate(100.)
    assert len(v) == 3
