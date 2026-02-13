import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Symbolic variable
# -------------------------------------------------
x = sp.symbols('x', real=True)

# -------------------------------------------------
# Orthonormal basis from Problem 2(a)
# -------------------------------------------------
phi0 = 1/sp.sqrt(2)
phi1 = sp.sqrt(sp.Rational(3,2)) * x
phi2 = sp.sqrt(sp.Rational(45,8)) * (x**2 - sp.Rational(1,3))
phi3 = sp.sqrt(sp.Rational(175,8)) * (x**3 - sp.Rational(3,5)*x)

basis = [phi0, phi1, phi2, phi3]

# -------------------------------------------------
# Functions to expand
# -------------------------------------------------
f_sin = sp.sin(sp.pi * x)
f_cos = sp.cos(sp.pi * x)

# -------------------------------------------------
# Inner product on [-1,1]
# -------------------------------------------------
def inner_product(f, g):
    return sp.integrate(f * g, (x, -1, 1))

# -------------------------------------------------
# Compute coefficients
# -------------------------------------------------
coeffs_sin = [sp.simplify(inner_product(f_sin, phi)) for phi in basis]
coeffs_cos = [sp.simplify(inner_product(f_cos, phi)) for phi in basis]

print("Sine coefficients:")
for i, c in enumerate(coeffs_sin):
    print(f"c_{i} =", c)

print("\nCosine coefficients:")
for i, c in enumerate(coeffs_cos):
    print(f"c_{i} =", c)

# -------------------------------------------------
# Symbolic approximations
# -------------------------------------------------
sin_approx_sym = sum(coeffs_sin[i] * basis[i] for i in range(4))
cos_approx_sym = sum(coeffs_cos[i] * basis[i] for i in range(4))

sin_approx_sym = sp.simplify(sin_approx_sym)
cos_approx_sym = sp.simplify(cos_approx_sym)

# -------------------------------------------------
# Convert to numerical functions
# -------------------------------------------------
f_sin_num = sp.lambdify(x, f_sin, "numpy")
f_cos_num = sp.lambdify(x, f_cos, "numpy")
sin_approx_num = sp.lambdify(x, sin_approx_sym, "numpy")
cos_approx_num = sp.lambdify(x, cos_approx_sym, "numpy")

# -------------------------------------------------
# Plot
# -------------------------------------------------
x_plot = np.linspace(-1, 1, 600)

plt.figure(figsize=(10,4))

# ---- Sine ----
plt.subplot(1,2,1)
plt.plot(x_plot, f_sin_num(x_plot), label=r"$\sin(\pi x)$", linewidth=2)
plt.plot(x_plot, sin_approx_num(x_plot), "--", label="4-term approximation")
plt.title("Sin approximation")
plt.xlabel("x")
plt.ylabel("value")
plt.legend()
plt.grid(True)

# ---- Cosine ----
plt.subplot(1,2,2)
plt.plot(x_plot, f_cos_num(x_plot), label=r"$\cos(\pi x)$", linewidth=2)
plt.plot(x_plot, cos_approx_num(x_plot), "--", label="4-term approximation")
plt.title("Cos approximation")
plt.xlabel("x")
plt.ylabel("value")
plt.legend()
plt.grid(True)

plt.tight_layout()

# -------------------------------------------------
# Save figure
# -------------------------------------------------
plt.savefig("problem_2b_legendre_projection.pdf", dpi=300)
plt.show()