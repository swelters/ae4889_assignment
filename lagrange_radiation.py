from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt


class LagrangeSystem:
    def __init__(
        self, m1, m2, particle_d=0.0, particle_rho=0.0, reflectivity=0.0
    ):
        self.m1 = m1
        self.m2 = m2
        self.mu = m2 / (m1 + m2)
        self.particle_d = particle_d
        self.particle_rho = particle_rho
        self.reflectivity = reflectivity
        self.c = 3e8  # speed of light in m/s
        self.solar_constant = 1361  # W/m^2
        self.AU = 1.496e11  # m
        self.area = np.pi * (particle_d / 2) ** 2
        self.mass = (4 / 3) * np.pi * (particle_d / 2) ** 3 * particle_rho

    def r_norm(self, x, y):
        r1 = np.sqrt((x + self.mu) ** 2 + y**2)
        r2 = np.sqrt((x - (1 - self.mu)) ** 2 + y**2)
        return r1, r2

    def colinear_lagrange(self, x):
        r1, r2 = self.r_norm(x, 0)
        f_mux = x - (1 - self.mu) / r1**3 * (x + self.mu) - self.mu / r2**3 * (x - (1 - self.mu)) + self.solar_pressure_acc(x, 0)
        return f_mux

    def triangular_lagrange(self, x, y):
        r1, r2 = self.r_norm(x, y)
        # Placeholder for triangular points, can be expanded as needed
        return None

    def get_lagrange_points(self):
        l1 = newton(self.colinear_lagrange, 0.1)
        l2 = newton(self.colinear_lagrange, 1)
        l3 = newton(self.colinear_lagrange, -1)
        return l1, l2, l3

    def solar_pressure_acc(self, x, y):
        r = np.sqrt(x**2 + y**2)
        S = self.solar_constant * (1 / r) ** 2
        if self.particle_d == 0:
            return 0.0
        else:
          b = (
              (1 + self.reflectivity)
              * S
              * self.area
              / (self.mass * self.c)
          )
          return b * x / r


if __name__ == "__main__":
    # Example usage for Sun-Earth system
    m1 = 1.989e30  # Sun mass in kg
    m2 = 5.972e24  # Earth mass in kg

    system_em = LagrangeSystem(m1, m2)
    print("\nLagrange points for the Earth-Moon system (no radiation pressure):")
    l1, l2, l3 = system_em.get_lagrange_points()
    print(f"L1 = {l1}, L2 = {l2}, L3 = {l3}")
    
    system = LagrangeSystem(
        m1, m2, particle_d=0.1, particle_rho=1200, reflectivity=0.5
    )
    print("\nLagrange points for the Sun-Earth system (with radiation pressure):")
    l1, l2, l3 = system.get_lagrange_points()
    print(f"L1 = {l1}, L2 = {l2}, L3 = {l3}")


# L2:x=1.156 (FOR EARTH MOON)https://www.sciencedirect.com/topics/engineering/lagrange-point#chapters-articles
# me = 5.974×1024kg mm=7.348×1022kg
