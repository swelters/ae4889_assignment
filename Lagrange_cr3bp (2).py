import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt


class LagrangeSystem_CR3BP:
    def __init__(
        self,
        mu1,
        mu2,
        cone=0.0,
        clock=0.0,
        particle_d=0.0,
        particle_rho=1200,
        reflectivity=1.0,
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

    def r_(self, x, y, z):
        mu = self.mu
        r1 = np.array([x + mu, y, z])
        r2 = np.array([x - (1 - mu), y, z])
        r1n = np.linalg.norm(r1)
        r2n = np.linalg.norm(r2)
        return r1, r2, r1n, r2n

    def get_Nabla_U(self, x, y, z):
        mu = self.mu
        r1, r2, r1n, r2n = self.r_(x, y, z)
        Ux = (1 - mu) * (x + mu) / r1n**3 + mu * (x - (1 - mu)) / r2n**3
        Uy = (1 - mu) * y / r1n**3 + mu * y / r2n**3
        Uz = (1 - mu) * z / r1n**3 + mu * z / r2n**3
        return Ux, Uy, Uz

    def radiation_pressure_acceleration(self, x, y, z):
        mu = self.mu
        r1, r2, r1n, r2n = self.r_(x, y, z)

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
        r1, r2, r1n, r2n = self.r_(x, y, z)

        ax = x + 2 * vy - (1 - mu) * (x + mu) / r1n**3 - mu * (x - (1 - mu)) / r2n**3
        ay = y - 2 * vx - (1 - mu) * y / r1n**3 - mu * y / r2n**3
        az = -(1 - mu) * z / r1n**3 - mu * z / r2n**3

        a_srp = self.radiation_pressure_acceleration(x, y, z)
        ax += a_srp[0]
        ay += a_srp[1]
        az += a_srp[2]

        return [vx, vy, vz, ax, ay, az]

    def compute_eigenvalues(self, x, y, z):
        r1, r2, r1n, r2n = self.r_(x, y, z)
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


if __name__ == "__main__":
    mu1 = (
        132712.0 * 1e6 * 1e9
    )  # Sun GM in m^3/s^2 (https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html)
    mu2 = (
        0.39860 * 1e6 * 1e9
    )  # Earth GM in m^3/s^2 (https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)
    system_simple = LagrangeSystem_CR3BP(
        mu1=mu1,
        mu2=mu2,
        particle_d=0.0,  # No particle for this simple system
    )
    l2_simple = system_simple.get_lagrange_point(1.01, 0.0, 0.0)
    print("L2 Simple Lagrange Point: [{:.12f}, {:.12f}, {:.12f}]".format(l2_simple[0], l2_simple[1], l2_simple[2]))

    system_rad = LagrangeSystem_CR3BP(
        mu1=mu1,
        mu2=mu2,
        particle_d=0.1,  # No particle for this simple system
        particle_rho=1200,  # kg/m^3
        reflectivity=0.5,  # Reflectivity of the particle
    )
    l2_rad = system_rad.get_lagrange_point(1.01, 0.0, 0.0)

    particle_d = 475
    particle_rho = 1200  # kg/m^3
    reflectivity = 0.2303  # Reflectivity of the particle
    
    

    epsilon = 1e-4
    rtol = 1e-12
    atol = 1e-12

    system = LagrangeSystem_CR3BP(
        mu1=mu1,
        mu2=mu2,
        particle_d=particle_d,
        particle_rho=particle_rho,
        reflectivity=reflectivity,
    )
    l2 = system.get_lagrange_point(1.01, 0.0, 0.0)
    print("L2 Lagrange Point: [{:.12f}, {:.12f}, {:.12f}]".format(l2[0], l2[1], l2[2]))

    eigvals, eigvecs = system.compute_eigenvalues(l2[0], l2[1], l2[2])
    print("Eigenvalues:", eigvals)

    idx_unstable = np.argmax(np.real(eigvals))
    idx_stable = np.argmin(np.real(eigvals))

    eigvec_unstable = np.real(eigvecs[:, idx_unstable])
    eigvec_stable = np.real(eigvecs[:, idx_stable])
    # Initial state
    X0 = np.array([l2[0], l2[1], l2[2], 0, 0, 0])

    # Perturbed initial conditions
    # Pad eigvecs to match X0: [dx, dy, 0, dx_dot, dy_dot, 0]
    eigvec_unstable_full = np.array(
        [
            eigvec_unstable[0],
            eigvec_unstable[1],
            0,
            eigvec_unstable[2],
            eigvec_unstable[3],
            0,
        ]
    )
    eigvec_stable_full = np.array(
        [eigvec_stable[0], eigvec_stable[1], 0, eigvec_stable[2], eigvec_stable[3], 0]
    )

    X0_unstable_p = X0 + epsilon * eigvec_unstable_full
    X0_unstable_m = X0 - epsilon * eigvec_unstable_full
    X0_stable_p = X0 + epsilon * eigvec_stable_full
    X0_stable_m = X0 - epsilon * eigvec_stable_full

    # Time arrays
    t_eval_fwd = np.linspace(0, 2 * np.pi, 100000)
    t_eval_bwd = np.linspace(0, -2 * np.pi, 100000)
    # Integrate manifolds
    sol_unstable_p = system.propagate(
        X0_unstable_p, (0, 2 * np.pi), t_eval=t_eval_fwd, rtol=rtol, atol=atol
    )
    sol_unstable_m = system.propagate(
        X0_unstable_m, (0, 2 * np.pi), t_eval=t_eval_fwd, rtol=rtol, atol=atol
    )
    sol_stable_p = system.propagate(
        X0_stable_p, (0, -2 * np.pi), t_eval=t_eval_bwd, rtol=rtol, atol=atol
    )
    sol_stable_m = system.propagate(
        X0_stable_m, (0, -2 * np.pi), t_eval=t_eval_bwd, rtol=rtol, atol=atol
    )

    # --- Plotting section ---
    # System-wide view
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # (x, y) projection
    axs[0].plot(sol_unstable_p.y[0], sol_unstable_p.y[1], "r", label="Unstable +")
    axs[0].plot(sol_unstable_m.y[0], sol_unstable_m.y[1], "r--", label="Unstable -")
    axs[0].plot(sol_stable_p.y[0], sol_stable_p.y[1], "b", label="Stable +")
    axs[0].plot(sol_stable_m.y[0], sol_stable_m.y[1], "b--", label="Stable -")
    axs[0].plot(l2[0], l2[1], "ko", label="sub-L2")
    axs[0].plot(0, 0, "go", label="Sun")
    axs[0].plot(1 - system.mu, 0, "mo", label="Earth")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("(x, y) projection (system-wide)")
    axs[0].set_xlim(-0.1, 1.3)
    axs[0].set_ylim(-0.4, 0.4)
    axs[0].legend()
    axs[0].set_aspect("equal", "box")

    # (x, z) projection (z=0 always)
    axs[1].plot(sol_unstable_p.y[0], sol_unstable_p.y[2], "r", label="Unstable +")
    axs[1].plot(sol_unstable_m.y[0], sol_unstable_m.y[2], "r--", label="Unstable -")
    axs[1].plot(sol_stable_p.y[0], sol_stable_p.y[2], "b", label="Stable +")
    axs[1].plot(sol_stable_m.y[0], sol_stable_m.y[2], "b--", label="Stable -")
    axs[1].plot(l2[0], l2[2], "ko", label="sub-L2")
    axs[1].plot(0, 0, "go", label="Sun")
    axs[1].plot(1 - system.mu, 0, "mo", label="Earth")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].set_title("(x, z) projection (system-wide)")
    axs[1].set_xlim(-0.1, 1.3)
    axs[1].legend()
    axs[1].set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig("figures/cr3bp_manifolds_systemwide.png")
    plt.show()
    plt.close()

    # Close-up view (zoom near sub-L2)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # (x, y) projection close-up
    axs[0].plot(sol_unstable_p.y[0], sol_unstable_p.y[1], "r", label="Unstable +")
    axs[0].plot(sol_unstable_m.y[0], sol_unstable_m.y[1], "r--", label="Unstable -")
    axs[0].plot(sol_stable_p.y[0], sol_stable_p.y[1], "b", label="Stable +")
    axs[0].plot(sol_stable_m.y[0], sol_stable_m.y[1], "b--", label="Stable -")
    axs[0].plot(l2[0], l2[1], "ko", label="sub-L2")
    axs[0].plot(1 - system.mu, 0, "mo", label="Earth")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("(x, y) projection (close-up)")
    axs[0].legend()
    axs[0].set_aspect("equal", "box")
    axs[0].set_xlim(l2[0] - 0.05, l2[0] + 0.05)
    axs[0].set_ylim(l2[1] - 0.05, l2[1] + 0.05)
    # (x, z) projection close-up
    axs[1].plot(sol_unstable_p.y[0], sol_unstable_p.y[2], "r", label="Unstable +")
    axs[1].plot(sol_unstable_m.y[0], sol_unstable_m.y[2], "r--", label="Unstable -")
    axs[1].plot(sol_stable_p.y[0], sol_stable_p.y[2], "b", label="Stable +")
    axs[1].plot(sol_stable_m.y[0], sol_stable_m.y[2], "b--", label="Stable -")
    axs[1].plot(l2[0], l2[2], "ko", label="sub-L2")
    axs[1].plot(1 - system.mu, 0, "mo", label="Earth")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].set_title("(x, z) projection (close-up)")
    axs[1].legend()
    axs[1].set_aspect("equal", "box")
    axs[1].set_xlim(l2[0] - 0.05, l2[0] + 0.05)
    axs[1].set_ylim(-0.01, 0.01)
    plt.tight_layout()
    plt.savefig("figures/cr3bp_manifolds_closeup.png")
    plt.show()
    plt.close()
