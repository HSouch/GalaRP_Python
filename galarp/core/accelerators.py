
import numpy as np


class Acceleration:
    def __init__(self, satellite, potential=None, shadow=None, wind=None, rho=None, debug=False):
        self.satellite = satellite
        self.potential = self.satellite.potential
        self.shadow = shadow
        self.wind = wind
        self.rho = rho
        self.debug = debug
    
    def apply_force(self, t, w):
        x, y, z, vx, vy, vz = w
        
        q = np.stack((x, y, z), axis=1)
        p = np.stack((vx, vy, vz), axis=1)
    
    
        # compute acceleration from potential:
        _t = np.array([0.0])
        acc = -self.potential._gradient(q, _t).T
    
    
        # Compute acceleration from ram pressure
        # a_ram = rho * v_perp^2 / Sigma_cloud
        if self.wind is not None:
            v_rel = self.wind.evaluate(t) - p
    
            a_ram = (self.rho.evaluate(t) / self.satellite.densities.evaluate(t)) 
            a_ram *= np.linalg.norm(v_rel, axis=1, keepdims=True) * v_rel

            a_ram = a_ram.T
    
            # If wind is on and shadow exists, apply shadow to appropriate particles
            if self.shadow is not None:
                shadowing = self.shadow.evaluate(q, p, t).T
                a_ram *= shadowing
    
            acc += a_ram
    
        return np.vstack((p.T, acc))
