import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt


class LagrangeSystem_CR3BP:
    def __init__(
        self,
        mu1,
        mu2,
        particle_d=0.0,
        particle_rho=1200,
        reflectivity=0.5,
    ):
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu = mu2 / (mu1 + mu2)
        if mu1 <= mu2:
            raise ValueError("mu1 must be greater than mu2")

        # Add constants for get_beta
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

    def get_beta(self):
        if self.particle_d == 0:
            return 0
        return (
            (1 + self.reflectivity)
            * (self.luminosity_sun / (4 * np.pi * self.c))
            / (self.mu1)
            / self.get_particle_loading()
        )

    def get_r(self, x, y, z):
        mu = self.mu
        r1 = np.array([x + mu, y, z])
        r2 = np.array([x - (1 - mu), y, z])
        r1n = np.linalg.norm(r1)
        r2n = np.linalg.norm(r2)
        return r1, r2, r1n, r2n

    def get_Nabla_U(self, x, y, z):
        mu = self.mu
        r1, r2, r1n, r2n = self.get_r(x, y, z)
        Ux = x - (1 - mu / r1n**3) * (mu + x) - mu / r2n**3 * (x - (1 - mu))
        Uy = y - (1 - mu) / r1n**3 * y - mu / r2n**3 * y
        Uz = -(1 - mu) / r1n**3 * z - mu / r2n**3 * z
        return Ux, Uy, Uz

    def radiation_pressure_acceleration(self, x, y, z):
        mu = self.mu
        r1, r2, r1n, r2n = self.get_r(x, y, z)

        if self.get_beta() == 0:
            return np.zeros(3)

        a_srp = self.get_beta() * (1 - mu) / r1n**2

        # n = nabla U/ abs(Nabla U)
        n = np.array(self.get_Nabla_U(x, y, z))
        n /= np.linalg.norm(n)
        a_srp_v = a_srp * (r1 @ n) ** 2 * n

        return a_srp_v

    def system_equations(self, t, s):
        x, y, z, vx, vy, vz = s
        mu = self.mu
        r1, r2, r1n, r2n = self.get_r(x, y, z)

        ax = x + 2 * vy - (1 - mu) * (x + mu) / r1n**3 - mu * (x - (1 - mu)) / r2n**3
        ay = y - 2 * vx - (1 - mu) * y / r1n**3 - mu * y / r2n**3
        az = -(1 - mu) * z / r1n**3 - mu * z / r2n**3

        a_srp = self.radiation_pressure_acceleration(x, y, z)
        ax += a_srp[0]
        ay += a_srp[1]
        az += a_srp[2]

        return [vx, vy, vz, ax, ay, az]

    def compute_eigenvalues(self, x, y, z):
        _, _, r1n, r2n = self.get_r(x, y, z)
        mu = self.mu
        Uxx = (
            1
            - (1 - mu) / r1n**3
            - mu / r2n**3
            + 3
            * ((x + mu) ** 2 * (1 - mu) / r1n**5 + (x - (1 - mu)) ** 2 * mu / r2n**5)
        )
        Uxy = 3 * y * ((x + mu) * (1 - mu) / r1n**5 + (x - (1 - mu)) * mu / r2n**5)
        Uyy = (
            1
            - (1 - mu) / r1n**3
            - mu / r2n**3
            + 3 * y**2 * ((1 - mu) / r1n**5 + mu / r2n**5)
        )
        A = [[0, 0, 1, 0], [0, 0, 0, 1], [Uxx, Uxy, 0, 2], [Uxy, Uyy, -2, 0]]
        eigvals, eigvecs = np.linalg.eig(A)
        return eigvals, eigvecs

    def propagate(self, X0, t_span, t_eval=None, rtol=1e-12, atol=1e-12):
        sol = solve_ivp(
            self.system_equations,
            t_span,
            X0,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
        return sol

    def get_lagrange_point(self, x0, y0, z0):
        def equilibrium_func(pos):
            x, y, z = pos
            s = [x, y, z, 0, 0, 0]
            return self.system_equations(0, s)[3:]  # Return accelerations only

        sol = root(equilibrium_func, [x0, y0, z0])
        if not sol.success:
            raise RuntimeError("Root solver failed to converge: " + sol.message)
        return sol.x


def wp1():
    G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant
    mu1 = 1.9891e+30 * G  # Mass of the Sun in kg * G
    mu2 = 6.0477e+24 * G  # Mass of the Earth in kg * G
    
    system_simple = LagrangeSystem_CR3BP(
        mu1=mu1,
        mu2=mu2,
        particle_d=0.0,  # No particle for this simple system
    )
    l2_simple = system_simple.get_lagrange_point(1, 0.0, 0.0)
    print(
        "L2 Simple Lagrange Point: [{:.12f}, {:.12f}, {:.12f}]".format(
            l2_simple[0], l2_simple[1], l2_simple[2]
        )
    )

    system_rad = LagrangeSystem_CR3BP(
        mu1=mu1,
        mu2=mu2,
        particle_d=0.1,  # No particle for this simple system
        particle_rho=1200,  # kg/m^3
        reflectivity=0.5,  # Reflectivity of the particle
    )
    l2_rad = system_rad.get_lagrange_point(1.01, 0.0, 0.0)
    print(
        "L2 Radiation Pressure Lagrange Point: [{:.12f}, {:.12f}, {:.12f}]".format(
            l2_rad[0], l2_rad[1], l2_rad[2]
        )
    )
    # Sensitivity analysis for different particle diameters
    particle_diameters = np.arange(0.01, 0.1, 0.001)  # in meters
    l2_positions = []
    for d in particle_diameters:
        system_sens = LagrangeSystem_CR3BP(
            mu1, mu2, particle_d=d, particle_rho=1200, reflectivity=0.5
        )
        l2_sens = system_sens.get_lagrange_point(1.01, 0, 0)
        l2_positions.append(l2_sens[0])
    plt.figure(figsize=(10, 6))
    plt.plot(particle_diameters, l2_positions, label="L2 x-position")
    plt.axhline(
        y=l2_simple[0],
        color="r",
        linestyle="--",
        label=f"L2 x-position without radiation pressure ({l2_simple[0]:.12f} AU)",
    )
    ax = plt.gca()
    yticks = np.linspace(min(l2_positions), max(l2_positions), num=6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(["{:,.07f}".format(x) for x in yticks])
    plt.title("Sensitivity of L2 x-position to Particle Diameter")
    plt.xlabel("Particle Diameter (m)")
    plt.ylabel("L2 x-position (AU)")
    plt.legend()
    plt.savefig("sensitivity_analysis_l2_position.png")
    plt.show()
    plt.close()


def wp2():
    G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant
    mu1 = 1.9891e+30 * G
    mu2 = 6.0477e+24 * G

    particle_d = 0.1
    particle_rho = 1200  # kg/m^3
    reflectivity = 0.2303  # Reflectivity of the particle

    system_simple = LagrangeSystem_CR3BP(
        mu1=mu1,
        mu2=mu2,
    )
    l2_simple = system_simple.get_lagrange_point(1.01, 0.0, 0.0)

    eigvals_simple, eigvecs_simple = system_simple.compute_eigenvalues(
        l2_simple[0], l2_simple[1], l2_simple[2]
    )
    print("Eigenvalues (Simple System):", eigvals_simple)

    system_rad = LagrangeSystem_CR3BP(
        mu1=mu1,
        mu2=mu2,
        particle_d=particle_d,
        particle_rho=particle_rho,
        reflectivity=reflectivity,
    )
    l2_rad = system_rad.get_lagrange_point(1.01, 0.0, 0.0)

    eigvals_rad, eigvecs_rad = system_rad.compute_eigenvalues(
        l2_rad[0], l2_rad[1], l2_rad[2]
    )
    print("Eigenvalues (with radiation pressure):", eigvals_rad)


if __name__ == "__main__":
    print("WP1:")
    wp1()
    print("WP2:")
    wp2()
