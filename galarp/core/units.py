from astropy import units as u
from astropy import constants as c


class UnitSet:
    def __init__(self, length:u.Unit, time:u.Unit, mass:u.Unit):
        self.length = length
        self.time = time
        self.mass = mass

        self.surface_area = self.length ** 2
        self.volume = self.length ** 3

        self.velocity = self.length / self.time
        self.density = self.mass / self.length ** 3
        self.surface_density = self.mass / self.length ** 2

        self.G = c.G.to(self.length ** 3 / self.mass / self.time ** 2)


galactic = UnitSet(length=u.kpc, time=u.Myr, mass=u.solMass)
solarsystem = UnitSet(length=u.au, time=u.yr, mass=u.solMass)
