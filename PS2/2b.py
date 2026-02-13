import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# -------------------------------------------------
# Orthonormal basis from Problem 2(a)
# -------------------------------------------------
def phi0(x):
    return 1.0 / np.sqrt(2)

def phi1(x):
    return np.sqrt(3/2) * x

def phi2(x):
    return np.sqrt(45/8) * (x**2 - 1/3)

def phi3(x):
    return np.sqrt(175/8) * (x**3 - 3*x/5)

basis = [phi0, phi1, phi2, phi3]

# -------------------------------------------------
# Functions to approximate
# -------------------------------------------------
def f_sin(x):
    return np.sin(np.pi * x)

def f_cos(x):
    return np.cos(np.pi * x)

# -------------------------------------------------
# Compute coefficients c_n = <f, phi_n>
# -------------------------------------------------
def coefficient(f, phi):
    integrand = lambda x: f(x) * phi(x)
    return quad(integrand, -1, 1)[0]

coeffs_sin = [coefficient(f_sin, phi) for phi in basis]
coeffs_cos = [coefficient(f_cos, phi) for phi in basis]

print("Sine coefficients:")
for i, c in enumerate(coeffs_sin):
    print(f"c_{i} = {c:.6f}")

print("\nCosine coefficients:")
for i, c in enumerate(coeffs_cos):
    print(f"c_{i} = {c:.6f}")

# -------------------------------------------------
# Build approximations
# -------------------------------------------------
def approx(x, coeffs):
    return sum(coeffs[i] * basis[i](x) for i in range(len(basis)))

# -------------------------------------------------
# Plot
# -------------------------------------------------
x_plot = np.linspace(-1, 1, 500)

sin_exact = np.sin(np.pi * x_plot)
sin_approx = np.array([approx(x, coeffs_sin) for x in x_plot])

cos_exact = np.cos(np.pi * x_plot)
cos_approx = np.array([approx(x, coeffs_cos) for x in x_plot])

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(x_plot, sin_exact, label="sin(pi x)", linewidth=2)
plt.plot(x_plot, sin_approx, "--", label="4-term approx")
plt.title("Sin approximation")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(x_plot, cos_exact, label="cos(pi x)", linewidth=2)
plt.plot(x_plot, cos_approx, "--", label="4-term approx")
plt.title("Cos approximation")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

