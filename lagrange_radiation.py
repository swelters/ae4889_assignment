import numpy as np

# Particle
d = 10e-2  # particle diameter in meters, 10cm
rho = 1200  # density in kg/m^3, 1200g/cm^3
reflectivity = 0.5  # reflectivity of the surface, 50%

# Constants
c = 3e8  # speed of light in m/s
solar_constant = 1361  # solar constant in W/m^2
AU = 1.496e11  # astronomical unit in meters

area = np.pi * (d / 2) ** 2  # cross-sectional area of the particle
mass = (4 / 3) * np.pi * (d / 2) ** 3 * rho  # mass of the particle

def solar_pressure_acc(x: float, y: float, mu: float):
  r = np.sqrt(x**2 + y**2)  # distance from the Sun
  solar_constant = 1361* (1**2/(x**2 + y**2))  # solar constant at distance r=sqrt(x**2 + y**2)
  
  b = (1 + reflectivity) * solar_constant * area/mass * / mu  # dimensionless parameter
  return b * np.array([x,y])/np.sqrt(x**2 + y**2)  # acceleration due to solar radiation pressure
