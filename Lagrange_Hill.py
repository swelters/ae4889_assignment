import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# from Lagrange_CR3BP import LagrangeSystem_CR3BP

class LagrangeSystem_HillProblem:
    def __init__(self, mu1, mu2, particle_d=0.0, particle_rho=1200, reflectivity=0.0):
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu = mu2 / (mu1 + mu2)
        if mu1 <= mu2:
            raise ValueError("m1 must be greater than m2")

        self.c = 2.998e8  # speed of light in m/s
        self.luminosity_sun = 382.8e24  # W
        self.particle_d = particle_d  # Default, can be set externally
        self.particle_rho = particle_rho  # Default, can be set externally
        self.reflectivity = reflectivity  # Default, can be set externally
        self.particle_area = np.pi * (particle_d / 2) ** 2
        self.mass_particle = (4 / 3) * np.pi * (particle_d / 2) ** 3 * particle_rho

    def get_particle_loading(self):
        if self.particle_d == 0:
            return 0
        return self.mass_particle / self.particle_area

    def get_k(self):
        if self.particle_d == 0:
            return 0
        return (
            (1 + self.reflectivity)
            * (self.luminosity_sun / (4 * np.pi * self.c))
            / (self.mu1 ** (2 / 3) * self.mu2 ** (1 / 3))
            / self.get_particle_loading()
        )

    def get_r(self, xi, eta, zeta):
        r = np.sqrt(xi**2 + eta**2 + zeta**2)
        return r

    def get_Nabla_U(self, xi, eta, zeta):
        mu = self.mu
        r = self.get_r(xi, eta, zeta)
        k = self.get_k()
        Uxi = 3 * xi - xi / r**3 + k
        Ueta = -eta / r**3
        Uzeta = -zeta - zeta / r**3
        return Uxi, Ueta, Uzeta

    def system_equations(self, t, s):
        xi, eta, zeta, vxi, veta, vzeta = s

        axi = 2*veta + self.get_Nabla_U(xi, eta, zeta)[0]
        aeta = -2*vxi + self.get_Nabla_U(xi, eta, zeta)[1]
        azeta = self.get_Nabla_U(xi, eta, zeta)[2]

        return [vxi, veta, vzeta, axi, aeta, azeta]

    def compute_eigenvalues(self, xi, eta, zeta):
        r = self.get_r(xi, eta, zeta)
        Uxixi = 3 + (2 * xi**2 - eta**2 - zeta**2) / r**5
        Uetaeta = (2 * eta**2 - xi**2 - zeta**2) / r**5
        Uxieta = 3 * eta * xi / r**5
        Uzetazeta = -1 + (2 * zeta**2 - xi**2 - eta**2) / r**5
        A = [[0, 0, 1, 0], [0, 0, 0, 1], [Uxixi, Uxieta, 0, 2], [Uxieta, Uetaeta, -2, 0]]
        eigvals, eigvecs = np.linalg.eig(A)
        return eigvals, eigvecs


    def get_lagrange_point(self, xi0, eta0, zeta0):
        def equilibrium_func(pos):
            xi, eta, zeta = pos
            s = [xi, eta, zeta, 0, 0, 0]
            return self.system_equations(0, s)[3:]  # Return accelerations only

        sol = root(equilibrium_func, [xi0, eta0, zeta0])
        if not sol.success:
            raise RuntimeError("Root solver failed to converge: " + sol.message)
        return sol.x


def wp3():
    G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant
    mu1 = 1.9891e+30 * G
    mu2 = 4.4820e+11 * G
    system3 = LagrangeSystem_CR3BP(
        mu1=mu1,
        mu2=mu2,
        particle_d=0.1,  # No particle for this simple system
        particle_rho=1200,  # Density of the particle in kg/m^3
        reflectivity=0.5,  # No radiation pressure for this simple system
    )
    l2_3 = system3.get_lagrange_point(0.01, 0, 0)
    print("Lagrange points for the Sun-Asteroid system (no radiation pressure):")
    print(
        "L2 Simple Lagrange Point: [{:.12f}, {:.12f}, {:.12f}]".format(
            l2_3[0], l2_3[1], l2_3[2]
        )
    )
    system_simple = LagrangeSystem_HillProblem(mu1, mu2)
    l2_simple = system_simple.get_lagrange_point(0.693, 0, 0)
    print("\nLagrange points for the Sun-Asteroid system (no radiation pressure):")
    print(
        "L2 Simple Lagrange Point: [{:.12f}, {:.12f}, {:.12f}]".format(
            l2_simple[0], l2_simple[1], l2_simple[2]
        )
    )
    system_rad = LagrangeSystem_HillProblem(
        mu1, mu2, particle_d=0.1, particle_rho=1200, reflectivity=0.5
    )
    # L1, L2, L3 for system with radiation pressure (reflectivity 0.5)
    l1_rad = system_rad.get_lagrange_point(0.01, 0, 0)
    l2_rad = system_rad.get_lagrange_point(0.695, 0, 0)
    l3_rad = system_rad.get_lagrange_point(-1, 0, 0)
    print("\nLagrange points for the Sun-Asteroid-Dust system (with radiation pressure, 0.5):")
    print("L1: [{:.12f}, {:.12f}, {:.12f}]".format(l1_rad[0], l1_rad[1], l1_rad[2]))
    print("L2: [{:.12f}, {:.12f}, {:.12f}]".format(l2_rad[0], l2_rad[1], l2_rad[2]))
    print("L3: [{:.12f}, {:.12f}, {:.12f}]".format(l3_rad[0], l3_rad[1], l3_rad[2]))
    print(
        "\nLagrange points for the Sun-Asteroid-Dust system (with radiation pressure, 0.5):"
    )
    print(
        "L2 Simple Lagrange Point: [{:.12f}, {:.12f}, {:.12f}]".format(
            l2_rad[0], l2_rad[1], l2_rad[2]
        )
    )
    system_personal = LagrangeSystem_HillProblem(
        mu1, mu2, particle_d=0.1, particle_rho=1200, reflectivity=0.2303
    )
    l2_personal = system_personal.get_lagrange_point(15, 0, 0)
    print(
        "\nLagrange points for the Sun-Asteroid-Dust system (with radiation pressure, 0.2303):"
    )
    print(
        "L2 Simple Lagrange Point: [{:.12f}, {:.12f}, {:.12f}]".format(
            l2_personal[0], l2_personal[1], l2_personal[2]
        )
    )
    # difference between two reflectivities
    diff = l2_rad[0] - l2_personal[0]
    print("\nDifference in L2 x-coordinate between 0.5 and 0.2303 reflectivity:", diff)
    print("Percentage difference:", (diff / l2_rad[0]) * 100, "%")

    # Stability analysis at L2 without radiation pressure
    print("\nStability analysis at L2 (without radiation pressure):")
    eigenvalues, _ = system_simple.compute_eigenvalues(l2_simple[0], l2_simple[1], l2_simple[2])
    print("Eigenvalues:", eigenvalues)
    # Stability analysis at L2 with radiation pressure
    print("\nStability analysis at L2 (with radiation pressure, 0.5):")
    eigenvalues_rad, _ = system_rad.compute_eigenvalues(l2_rad[0], l2_rad[1], l2_rad[2])
    print("Eigenvalues:", eigenvalues_rad)
    print("\nStability analysis at L2 (with radiation pressure, 0.2303):")
    eigenvalues_personal, _ = system_personal.compute_eigenvalues(l2_personal[0], l2_personal[1], l2_personal[2])
    print("Eigenvalues:", eigenvalues_personal)


if __name__ == "__main__":
    wp3()
