"""Solid bodies (planets, moons, ...) that particles cannot pass through.

"""

import galarp

import astropy.units as u
import numpy as np

from galarp.utils import to_value

class SolidSphere:


    def __init__(self, center, radius, velocity=(0, 0, 0), particle_radius=0.0,
                 units=galarp.units.galactic):
        self.units = units
        self.center0 = to_value(center, units.length)
        self.radius = float(to_value(radius, units.length))
        self.velocity = to_value(velocity, units.velocity)
        self.particle_radius = float(to_value(particle_radius, units.length))

    @property
    def min_dist(self):
        return self.radius + self.particle_radius

    def center(self, t):
        return self.center0 + self.velocity * t

    def resolve(self, t, w_old, w_new):
        """Enforce the constraint on a state of shape (6, N); returns a new array."""
        w = w_new.copy()
        q, p = w[:3].T, w[3:].T          # views, shape (N, 3)
        q_old = w_old[:3].T
        c = self.center(t)
        R = self.min_dist

        # --- 1. Swept test: particles that started outside but whose straight
        #        path this step crosses the sphere (catches tunnelling straight
        #        through when |v| dt is comparable to the radius).
        d = q - q_old
        f = q_old - c
        a = np.einsum("ij,ij->i", d, d)
        b = 2.0 * np.einsum("ij,ij->i", f, d)
        cc = np.einsum("ij,ij->i", f, f) - R**2
        disc = b**2 - 4.0 * a * cc

        swept = (cc > 0) & (disc >= 0) & (a > 0)
        s = np.full(len(a), np.inf)
        s[swept] = (-b[swept] - np.sqrt(disc[swept])) / (2.0 * a[swept])
        swept &= (s >= 0) & (s <= 1)
        if swept.any():
            q[swept] = q_old[swept] + s[swept, None] * d[swept]

        # --- 2. Overlap test: anything inside (or now exactly on) the surface.
        offset = q - c
        dist = np.linalg.norm(offset, axis=1)
        hit = swept | (dist < R)
        if not hit.any():
            return w

        n = np.tile([0.0, 0.0, 1.0], (hit.sum(), 1))   # fallback at the center
        ok = dist[hit] > 1e-12 * max(R, 1.0)
        n[ok] = offset[hit][ok] / dist[hit][ok, None]

        q[hit] = c + n * R

        # --- 3. Velocity, measured relative to the moving surface.
        v_rel = p[hit] - self.velocity
        vn = np.einsum("ij,ij->i", v_rel, n)
        v_normal = vn[:, None] * n
        v_tangent = v_rel - v_normal

        inward = (vn < 0)[:, None]
        v_rel_new = np.where(
            inward,
            (1.0 - self.friction) * v_tangent - self.restitution * v_normal,
            v_rel,                       # already separating: leave it alone
        )
        p[hit] = v_rel_new + self.velocity
        return w

    def inside(self, q, t=0):
        """Boolean mask of particles (q shape (3, N)) inside the sphere."""
        return np.linalg.norm(q.T - self.center(t), axis=1) < self.min_dist
