from Lagrange_Hill import LagrangeSystem_HillProblem


if __name__ == "__main__":
    mu1 = 132712.0 * 1e6 * 1e9  # Sun GM in m^3/s^2
    mu2 = 0.39860 * 1e6 * 1e9  # Earth GM in m^3/s^2
    # Student-no    Reflectivity  |  ----------------------  Initial Conditions ---------------------------------------------------------------
    #                             |  x              y               z              x_dot           y_dot           z_dot           Reflectivity
    # 5204968       0.2303        |  0.213947317195 0               0.071828480574 0               -0.698557918092 0               0.2050

    initial_conditions = [0.213947317195, 0, 0.071828480574, 0, -0.698557918092, 0]  # Initial conditions for the orbit
    reflectivity = 0.2050  # Reflectivity value

fig = plt.figure(figsize=(12, 10))

# --- Subplot 1: 3D View ---
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(xi, zeta, eta, label='Orbit')
ax1.scatter(0, 0, 0, color='red', marker='s', s=50, label='Ryugu')
ax1.scatter(xi0, 0, 0, color='blue', marker='o', s=40, label='Sub-L2')
ax1.set_xlabel('ξ')
ax1.set_ylabel('ζ')
ax1.set_zlabel('η')
ax1.set_title('3D View')
ax1.set_box_aspect([1, 1, 1])
ax1.legend()

# --- Subplot 2: (ξ, η) Projection ---
ax2 = fig.add_subplot(222)
ax2.plot(xi, eta, label='Orbit')
ax2.scatter(0, 0, color='red', marker='s', s=50, label='Ryugu')
ax2.scatter(xi0, 0, color='blue', marker='o', s=40, label='Sub-L2')
ax2.set_aspect('equal', 'box')
ax2.set_xlabel('ξ')
ax2.set_ylabel('η')
ax2.set_title('(ξ, η)-Projection')
ax2.grid(True)
ax2.legend()

# --- Subplot 3: (ξ, ζ) Projection ---
ax3 = fig.add_subplot(223)
ax3.plot(xi, zeta, label='Orbit')
ax3.scatter(0, 0, color='red', marker='s', s=50, label='Ryugu')
ax3.scatter(xi0, 0, color='blue', marker='o', s=40, label='Sub-L2')
ax3.set_aspect('equal', 'box')
ax3.set_xlabel('ξ')
ax3.set_ylabel('ζ')
ax3.set_title('(ξ, ζ)-Projection')
ax3.grid(True)
ax3.legend()

# --- Subplot 4: (η, ζ) Projection ---
ax4 = fig.add_subplot(224)
ax4.plot(eta, zeta, label='Orbit')
ax4.scatter(0, 0, color='red', marker='s', s=50, label='Ryugu')
ax4.scatter(0, 0, color='blue', marker='o', s=40, label='Sub-L2')
ax4.set_aspect('equal', 'box')
ax4.set_xlabel('η')
ax4.set_ylabel('ζ')
ax4.set_title('(η, ζ)-Projection')
ax4.grid(True)
ax4.legend(loc='upper right')

plt.tight_layout(rect=[0, 0.06, 1, 0.97], h_pad=2.5)
plt.savefig('figures/orbit_subplots.png', dpi=300, bbox_inches='tight')
plt.show()