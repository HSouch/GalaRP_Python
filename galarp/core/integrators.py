from gala import integrate as gi

DEFAULT_INTEGRATOR = gi.RK5Integrator

class ConstrainedRK5Integrator(DEFAULT_INTEGRATOR):
    """Gala's RK5 with a hook ``constraint(t, w_old, w_new) -> w_new`` that runs
    once after every full step (not at the intermediate RK stages)."""

    def __init__(self, func, constraint=None, **kwargs):
        super().__init__(func, **kwargs)
        self.constraint = constraint

    def step(self, t, w, dt):
        w_new = super().step(t, w, dt)
        if self.constraint is not None:
            w_new = self.constraint(t + dt, w, w_new)
        return w_new
