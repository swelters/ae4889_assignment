import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class LagrangeSystem_CR3BP:
    def __init__(self, mu1, mu2, particle_d=0.0, particle_rho=0.0, reflectivity=1.0):
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu = mu2 / (mu1 + mu2)
        if mu1 <= mu2:
            raise ValueError("m1 must be greater than m2")
        self.particle_d = particle_d
        self.particle_rho = particle_rho
        self.reflectivity = reflectivity
        self.c = 3e8  # speed of light in m/s
        self.luminosity_sun = 382.8e24  # W
        self.AU = 1.496e11  # m
        self.particle_area = np.pi * (particle_d / 2) ** 2
        self.mass_particle = (4 / 3) * np.pi * (particle_d / 2) ** 3 * particle_rho

    def r_norm(self, x, y):
        mu = self.mu
        r1 = ((x + mu) ** 2 + y**2) ** (1 / 2)
        r2 = ((x - (1 - mu)) ** 2 + y**2) ** (1 / 2)
        return r1, r2

    def get_particle_loading(self):
        if self.particle_d == 0:
            return 0
        return self.mass_particle / self.particle_area

    def get_b(self):
        if self.particle_d == 0:
            return 0
        return (
            (1 + self.reflectivity)
            * (self.luminosity_sun / (4 * np.pi * self.c))
            / (self.mu1)
            / self.get_particle_loading()
        )

    def potential_equations(self, vars):
        x, y = vars
        r1, r2 = self.r_norm(x, y)
        b = self.get_b()
        mu = self.mu
        A = (1 - b) * (1 - mu) / r1**3 + mu / r2**3
        O_x = (1 - A) * x - mu * (1 - mu) * ((1 - b) / r1**3 - 1 / r2**3)
        O_y = (1 - A) * y
        return [O_x, O_y]

    def get_lagrange_point(self, x_init, y_init):
        sol = root(self.potential_equations, [x_init, y_init])
        return sol.x

    def get_l2_analytical(self):
        """
        Analytical L2 location using the equation from the reference (Eq. 28).
        Returns the x position of L2 (in units of AU).
        """
        from scipy.optimize import newton

        mu = self.mu
        beta = self.get_b()

        def l2_eqn(rho):
            lhs = mu / (3 * (1 - mu))
            if rho == 0:
                return 1e6
            rhs = (rho**2 * (1 + rho + (rho**2) / 3 + beta / (3 * rho))) / (
                (1 + rho) ** 2 * (1 - rho**3)
            )
            return lhs - rhs

        rho_guess = 0.01
        sol = newton(l2_eqn, rho_guess)
        x_tilde = 1 + sol
        return x_tilde

    def compute_U_derivatives_colinear(self, x, y):
        mu = self.mu
        b = self.get_b()
        r1, r2 = self.r_norm(x, y)
        Uxx = 1 - (1-b)*(1-mu) * (r1**2 - 3*(x+mu)**2) / r1**5 - mu * (r2**2 - 3*(x-(1-mu))**2) / r2**5
        Uyy = 1 - ((1-b)*(1-mu) * (r1**2 - 3*(y)**2) / r1**5 - mu * (r2**2 - 3*(y)**2) / r2**5)
        Uxy = (1-b)*(1-mu) * (3*(x+mu)*y) / r1**5 + mu * (3*(x-(1-mu))*y) / r2**5
        return Uxx, Uyy, Uxy

    def get_A(self, x, y):
        Uxx, Uyy, Uxy = self.compute_U_derivatives_colinear(x, y)
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [Uxx, Uxy, 0, 2], [Uxy, Uyy, -2, 0]])
        return A

    def propagate(self, X0, t_span, t_eval=None, rtol=1e-12, atol=1e-12):
        """
        Integrate the full nonlinear equations of motion from initial state X0 = [x, y, x_dot, y_dot].
        """

        def system_equations(t, X):
            x, y, x_dot, y_dot = X
            A = self.get_A(x, y)
            return A @ X

        sol = solve_ivp(
            system_equations, t_span, X0, t_eval=t_eval, rtol=rtol, atol=atol, max_step=1e-4
        )

        return sol


def main_12():
    mu1 = (
        132712.0 * 1e6 * 1e9
    )  # Sun GM in m^3/s^2 (https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html)
    mu2 = (
        0.39860 * 1e6 * 1e9
    )  # Earth GM in m^3/s^2 (https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)
    print("\nLagrange points for the Sun-Earth system (no radiation pressure):")
    system = LagrangeSystem_CR3BP(mu1, mu2)
    l2 = system.get_lagrange_point(1, 0)
    print(f"L2 = {l2[0]}, {l2[1]}")

    print("\nLagrange points for the Sun-Earth system (with radiation pressure):")
    system_rad = LagrangeSystem_CR3BP(
        mu1, mu2, particle_d=0.1, particle_rho=1200, reflectivity=0.5
    )
    l2_rad = system_rad.get_lagrange_point(1, 0)
    print(f"L2 = {l2_rad[0]}, {l2_rad[1]}")

    # Sensitivity analysis for different particle diameters
    particle_diameters = np.arange(0.01, 0.1, 0.001)  # in meters
    l2_positions = []
    for d in particle_diameters:
        system_rad = LagrangeSystem_CR3BP(
            mu1, mu2, particle_d=d, particle_rho=1200, reflectivity=0.5
        )
        l2_rad = system_rad.get_lagrange_point(1, 0)
        l2_positions.append(l2_rad[0])
    plt.figure(figsize=(10, 6))
    plt.plot(particle_diameters, l2_positions, label="L2 x-position")
    plt.axhline(
        y=l2[0],
        color="r",
        linestyle="--",
        label=f"L2 x-position without radiation pressure ({l2[0]:.7f} AU)",
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
    plt.close()

    # Stability analysis at L2 without radiation pressure
    print("\nStability analysis at L2 (without radiation pressure):")
    A = system.get_A(l2[0], l2[1])
    print("Eigenvalues:", np.linalg.eigvals(A))
    # Stability analysis at L2 with radiation pressure
    print("\nStability analysis at L2 (with radiation pressure):")
    A_rad = system_rad.get_A(l2_rad[0], l2_rad[1])
    print("Eigenvalues:", np.linalg.eigvals(A_rad))

    # Plotting the particle loading (m/A) vs lightness number (Beta)
    particle_diameters = np.linspace(0.01, 0.1, 100)  # in meters
    b_values = []
    particle_loading_values = []
    for d in particle_diameters:
        system = LagrangeSystem_CR3BP(
            mu1, mu2, particle_d=d, particle_rho=1200, reflectivity=0.5
        )
        b = system.get_b()
        particle_loading = system.get_particle_loading() * 1000
        b_values.append(b)
        particle_loading_values.append(particle_loading)
    plt.figure(figsize=(10, 6))
    plt.plot(b_values, particle_loading_values)
    plt.title("Particle Loading vs Lightness Number (Beta)")
    plt.xlabel("Lightness Number (Beta)")
    plt.ylabel("Particle Loading (g/m^2)")
    plt.savefig("particle_loading_vs_lightness_number.png")
    plt.close()


def plot_CR3BP_manifolds_full():
    """
    Compute and plot the stable and unstable manifolds from the sub-L2 point in the Hill problem
    with projections (x, y) and (x, z), both system-wide and close-up.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters
    mu1 = 132712.0 * 1e6 * 1e9
    mu2 = 0.39860 * 1e6 * 1e9
    # Eath-Moon Test
    mu1 = 0.39860 * 1e6 * 1e9  # Earth GM in m^3/s^2
    mu2 = 0.00490 * 1e6 * 1e9  # Moon GM in m^3/s^2

    reflectivity = 0.2303  # Assigned value
    epsilon = 1e-4
    rtol = 1e-12
    atol = 1e-12
    system = LagrangeSystem_CR3BP(
        mu1, mu2, particle_d=0, particle_rho=1200, reflectivity=reflectivity
    )
    x0, y0 = system.get_lagrange_point(1.01, 0)  # sub-L2 point
    print(f"Sub-L2 point: x = {x0}, y = {y0}")
    # Linearized system matrix and eigenvectors
    A = system.get_A(x0, y0)
    eigvals, eigvecs = np.linalg.eig(A)
    # Find real eigenvectors (unstable: positive real part, stable: negative real part)
    eigvec_unstable = np.real(eigvecs[:, 0])
    eigvec_stable = np.real(eigvecs[:, 1])

    # Initial state
    X0 = np.array([x0, y0, 0, 0])
    # Perturbed initial conditions
    X0_unstable_p = X0 + epsilon * eigvec_unstable
    X0_unstable_m = X0 - epsilon * eigvec_unstable
    X0_stable_p = X0 + epsilon * eigvec_stable
    X0_stable_m = X0 - epsilon * eigvec_stable
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

    # System-wide plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # (x, y) projection (system-wide)
    axs[0].plot(sol_unstable_p.y[0], sol_unstable_p.y[1], "r", label="Unstable +")
    axs[0].plot(sol_unstable_m.y[0], sol_unstable_m.y[1], "r--", label="Unstable -")
    axs[0].plot(sol_stable_p.y[0], sol_stable_p.y[1], "b", label="Stable +")
    axs[0].plot(sol_stable_m.y[0], sol_stable_m.y[1], "b--", label="Stable -")
    axs[0].plot(x0, y0, "ko", label="sub-L2")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("(x, y) projection (system-wide)")
    axs[0].legend()
    axs[0].set_aspect("equal", "box")
    axs[0].set_xlim(-2, 2)
    axs[0].set_ylim(-2, 2)
    axs[0].set_xticks(np.arange(-2, 2.1, 0.4))
    axs[0].set_yticks(np.arange(-2, 2.1, 0.4))
    # # Since z is always 0, plot only x as a single axis plot
    # axs[1].plot(sol_unstable_p.y[0], np.zeros_like(sol_unstable_p.y[0]), "r", label="Unstable +")
    # axs[1].plot(sol_unstable_m.y[0], np.zeros_like(sol_unstable_m.y[0]), "r--", label="Unstable -")
    # axs[1].plot(sol_stable_p.y[0], np.zeros_like(sol_stable_p.y[0]), "b", label="Stable +")
    # axs[1].plot(sol_stable_m.y[0], np.zeros_like(sol_stable_m.y[0]), "b--", label="Stable -")
    # axs[1].plot(x0, 0, "ko", label="sub-L2")
    # axs[1].set_xlabel("x")
    # axs[1].set_yticks([0])
    # axs[1].set_ylabel("z = 0")
    # axs[1].set_title("x-axis (system-wide)")
    # axs[1].legend()
    # axs[1].set_aspect("equal", "box")
    # plt.tight_layout()
    # plt.savefig('cr3bp_manifolds_systemwide.png')
    plt.show()
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].plot(sol_unstable_p.y[0], sol_unstable_p.y[1], "r", label="Unstable +")
    axs[0].plot(sol_unstable_m.y[0], sol_unstable_m.y[1], "r--", label="Unstable -")
    axs[0].plot(sol_stable_p.y[0], sol_stable_p.y[1], "b", label="Stable +")
    axs[0].plot(sol_stable_m.y[0], sol_stable_m.y[1], "b--", label="Stable -")
    axs[0].plot(x0, y0, "ko", label="sub-L2")
    # Plot eigenvectors at the sub-L2 point
    scale = 0.05  # scaling factor for visualization
    # Unstable eigenvector (red)
    axs[0].arrow(
        x0, y0,
        scale * eigvec_unstable[0], scale * eigvec_unstable[1],
        head_width=0.01, head_length=0.02, fc='r', ec='r', label='Unstable eigvec'
    )
    # Stable eigenvector (blue)
    axs[0].arrow(
        x0, y0,
        scale * eigvec_stable[0], scale * eigvec_stable[1],
        head_width=0.01, head_length=0.02, fc='b', ec='b', label='Stable eigvec'
    )
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("(x, y) projection (close-up)")
    axs[0].legend()
    axs[0].set_aspect("equal", "box")
    axs[0].set_xlim(0.9, 1.5)
    axs[0].set_ylim(-0.1, 0.1)
    # axs[1].plot(sol_unstable_p.y[0], sol_unstable_p.y[2], "r", label="Unstable +")
    # axs[1].plot(sol_unstable_m.y[0], sol_unstable_m.y[2], "r--", label="Unstable -")
    # axs[1].plot(sol_stable_p.y[0], sol_stable_p.y[2], "b", label="Stable +")
    # axs[1].plot(sol_stable_m.y[0], sol_stable_m.y[2], "b--", label="Stable -")
    # axs[1].plot(x0, 0, "ko", label="sub-L2")
    # axs[1].set_xlim(x0 - zoom, x0 + zoom)
    # axs[1].set_xlabel("x")
    # axs[1].set_ylabel("z")
    # axs[1].set_title("(x, z) projection (close-up)")
    # axs[1].legend()
    # axs[1].set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig('cr3bp_manifolds_closeup.png')
    plt.close()
    print("Plots saved: cr3bp_manifolds_systemwide.png, cr3bp_manifolds_closeup.png")


if __name__ == "__main__":
    # Personal Assignment values:
    # Student-no    Reflectivity  |  ----------------------  Initial Conditions ---------------------------------------------------------------
    #                             |  x              y               z              x_dot           y_dot           z_dot           Reflectivity
    # 5204968       0.2303        |  0.213947317195 0               0.071828480574 0               -0.698557918092 0               0.2050
    # print("WP1 & 2:")
    # main_12()

    plot_CR3BP_manifolds_full()

# L2:x=1.156 (FOR EARTH MOON)https://www.sciencedirect.com/topics/engineering/lagrange-point#chapters-articles
# me = 5.974e24 mm=7.348e22
