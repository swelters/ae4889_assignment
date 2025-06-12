from Lagrange_CR3BP import LagrangeSystem_CR3BP
import numpy as np
import matplotlib.pyplot as plt


def manifold():
    G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant
    mu1 = 1.9891e30 * G
    mu2 = 4.4820e11 * G

    particle_d = 0.1
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
    
    print(f"t {sol_unstable_p.t} s")
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
    axs[1].set_aspect("equal", "box")
    fig.suptitle(
        "CR3BP Manifolds in (x, y) and (x, z) Projections around the Sun-Earth sub-L2 Point propagated 1 year backwards and 1 year forwards in dimensionless units."
    )
    plt.tight_layout()
    plt.savefig("figures/cr3bp_manifolds_systemwide.png")
    # plt.show()
    plt.close()

    # Close-up view (zoom near sub-L2)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # (x, y) projection close-up
    axs[0].plot(sol_unstable_p.y[0], sol_unstable_p.y[1], "r", label="Unstable +")
    axs[0].plot(sol_unstable_m.y[0], sol_unstable_m.y[1], "r--", label="Unstable -")
    axs[0].plot(sol_stable_p.y[0], sol_stable_p.y[1], "b", label="Stable +")
    axs[0].plot(sol_stable_m.y[0], sol_stable_m.y[1], "b--", label="Stable -")
    axs[0].plot(l2[0], l2[1], "ko", label="sub-L2")
    # Set Ryugu dot size proportional to its actual scale in the plot
    # Assume Ryugu's radius in dimensionless units is much smaller than the plot window
    ryugu_radius_m = 475  # Ryugu diameter ~475 m
    orbit_r_ryugu = 149597870.7e3  # m
    plot_width_units = 4 * 0.000000421465  # xlim width in units
    plot_width_m = plot_width_units * orbit_r_ryugu
    # Set marker size so that the dot's diameter is proportional to Ryugu's diameter on the plot
    # Matplotlib marker size is in points^2, so we scale accordingly
    ryugu_marker_size = (ryugu_radius_m / (plot_width_m / 72)) ** 2 * 4  # 72 points per inch, fudge factor 4

    axs[0].plot(1 - system.mu, 0, "mo", label="Ryugu")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("(x, y) projection (close-up)")
    axs[0].legend()
    axs[0].set_aspect("equal", "box")
    axs[0].set_xlim(l2[0] - plot_width_units/2, l2[0] + plot_width_units/2)
    axs[0].set_ylim(l2[1] - plot_width_units/2, l2[1] + plot_width_units/2)
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
    axs[1].set_aspect("equal", "box")
    axs[1].set_xlim(l2[0] - 0.05, l2[0] + 0.05)
    axs[1].set_ylim(-0.01, 0.01)
    fig.suptitle(
        "Close-up view of the CR3BP Manifolds in (x, y) and (x, z) Projections around the Sun-Earth sub-L2 Point \n"
        "propagated 1 year backwards and 1 year forwards in dimensionless units."
    )
    plt.tight_layout()
    plt.savefig("figures/cr3bp_manifolds_closeup.png")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    manifold()
