from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt

def r_norm(mu:float, x:float, y:float):
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    return r1, r2
  
def colinear_lagrange(x:float, mu:float):
  r1, r2 = r_norm(mu, x, 0)
  f_mux = x - (1-mu) / r1 ** 3 * (x + mu) - mu / r2 ** 3 * (x - (1 - mu))
  return f_mux
  
def triangular_lagrange(x:float, y:float, mu:float):
  r1, r2 = r_norm(mu, x, y)

  return ...

def get_lagrange_points(mu:float):
    # Lagrange points L1, L2, and L3
    l1 = newton(colinear_lagrange, 0, args=(mu,))
    l2 = newton(colinear_lagrange, 1, args=(mu,))
    l3 = newton(colinear_lagrange, -1, args=(mu,))

    return l1, l2, l3
  
if __name__ == "__main__":
    # Gravitational parameter for the Sun-Earth system
    m_earth = 5.972e24  # Earth mass in kg
    m_moon = 7.348e22  # Moon mass in kg
    m_sun = 1.989e30  # Sun mass in kg
    mu = m_earth / (m_earth + m_sun)  # Dimensionless (Earth mass / (Sun mass + Earth mass))
    mu_test = m_moon / (m_earth + m_moon)  # Dimensionless (Moon mass / (Earth mass + Moon mass))
    
    print("\nLagrange points for the Sun-Earth system:")
    print(f"mu: {mu}")
    l1, l2, l3 = get_lagrange_points(mu)
    print(f"Lagrange points SE: L1 = {l1}, L2 = {l2}, L3 = {l3}")
    
    print("\nLagrange points for the Earth-Moon system for test:")
    print(f"mu_test: {mu_test}")
    l1, l2, l3 = get_lagrange_points(mu_test)
    print(f"Lagrange points EM: L1 = {l1}, L2 = {l2}, L3 = {l3}")


  #L2:x=1.156 (FOR EARTH MOON)https://www.sciencedirect.com/topics/engineering/lagrange-point#chapters-articles
  # me = 5.974×1024kg mm=7.348×1022kg