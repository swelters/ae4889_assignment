from scipy.optimize import newton

def lagrange_colinear_function(x, mu):
    r1 = abs(x + mu)
    r2 = abs(x - (1 - mu))
    
    return x - (1 - mu) / (r1**3) * (x + mu) - mu / (r2**3) * (x - (1 - mu))

def lagrange_colinear_solar_function(x, mu, d, rho, r):
    r1 = abs(x + mu)
    r2 = abs(x - (1 - mu))

    
    return 



# Test cases
if __name__ == "__main__":
    G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant
    mu1 = 1.9891e+30 * G  # Mass of the Sun in kg * G
    mu2 = 6.0477e+24 * G  # Mass of the Earth in kg * G

    mu = mu2 / (mu1 + mu2)  # Non-dimensional mass parameter for Earth-Sun system

    x_l2 = newton(lagrange_colinear_function, x0=1, args=(mu,))
    
    print(f"L2 Lagrange point x-coordinate: {x_l2:.8g}")

    mu1_test = 5.972e+24 * G  # Mass of the Earth in kg * G
    mu2_test = 7.342e+22 * G  # Mass of the Moon in kg * G
    mu_test = mu2_test / (mu1_test + mu2_test)  # Non-dimensional mass parameter for Earth-Moon system
    x_l2_test = newton(lagrange_colinear_function, x0=1, args=(mu_test,))
    print(f"L2 Lagrange point x-coordinate (Earth-Moon): {x_l2_test:.8g}")