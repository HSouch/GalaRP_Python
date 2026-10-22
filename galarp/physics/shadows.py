import galarp
import numpy as np


class Shadow:
    def __init__(self, 
        wind:None | galarp.RPWind, 
        density: None | galarp.Density = None,
        units=galarp.units.galactic,
        line_of_sight = np.array([0, 0, 1]),
        north_vector = np.array([0, 0, 1]),
        **kwargs
        ):

        self.wind = wind
        self.density = density        
        self.north_vector = north_vector

        self.units = units


        self.update_unit_vectors(0)

    def update_unit_vectors(self, t=0):
        self.line_of_sight = self.wind.unit_vector(t)

        if np.all(self.line_of_sight == self.north_vector):
            self.north_vector = np.array([0, 1, 0])

        self.xcam = np.cross(self.north_vector, self.line_of_sight)
        self.xcam /= np.linalg.norm(self.xcam)
        
        self.ycam = np.cross(self.line_of_sight, self.xcam)
        self.ycam /= np.linalg.norm(self.ycam)

        self.basis = np.array([self.xcam, self.ycam, self.line_of_sight])

        

        
    def evaluate(self, q, p, t=0):
        raise NotImplementedError("Must be implemented by a subclass.")




class DynamicShadow(Shadow):

    def __init__(self, wind, density, C_D=0.5, units=galarp.units.galactic, bin_width=2, **kwargs):
        super().__init__(wind=wind, density=density, units=units, **kwargs)

        self.bin_width = bin_width

        self.psi_0 = self.momentum_flux(0)
        self.C_D = C_D


    def momentum_flux(self, t=0):
        return self.density.evaluate(t) * self.wind.speed(t) ** 2


    def evaluate(self, q, p, areas, t=0):
        self.update_unit_vectors(t)
        

        xproj = np.dot(q.T, self.basis[0])
        yproj = np.dot(q.T, self.basis[1])
        d_los = np.dot(q.T, self.basis[2])

        v_los = np.dot(p.T, self.line_of_sight)


        self.psi_0 = self.momentum_flux(t)

        shadowing = np.ones(len(xproj))

        column_id = self._assign_columns(xproj, yproj)
        columns = self._group_into_columns(column_id, d_los)

        shadowing = np.ones(len(xproj))
        for indices in columns:
            if len(indices) < 2:
                continue
            
            shadowing[indices], _ = self._march_column(indices, areas)
        
        return shadowing
        # process_column(self, indices)
    
    def _assign_columns(self, xproj, yproj):
        ix = np.floor(xproj / self.bin_width).astype(int)
        iy = np.floor(yproj / self.bin_width).astype(int)

        # Each distinct (ix, iy) pair is one column; relabel them 0, 1, 2, ...
        cells = np.column_stack([ix, iy])
        _, column_id = np.unique(cells, axis=0, return_inverse=True)
        return column_id.ravel()

    def _group_into_columns(self, column_id, d_los):
        order = np.lexsort((d_los, column_id))
        
        sorted_ids = column_id[order]
        boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1

        return np.split(order, boundaries)
    
    def _march_column(self, indices, areas):
        cell_area = self.bin_width ** 2
        strengths = np.empty(len(indices))
        phi = 1.0

        for k, i in enumerate(indices):
            tau = self.C_D * areas[i] / cell_area
            absorbed = -np.expm1(-tau)                  # 1 - e^{-tau}, accurate for small tau
            self_shielding = absorbed / tau if tau > 0 else 1.0

            strengths[k] = phi * self_shielding         # upstream shadowing x self-shielding
            phi = phi * (1.0 - absorbed)    

        return strengths, phi