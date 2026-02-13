import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Symbolic variable
# -------------------------------------------------
x = sp.symbols('x', real=True)

# -------------------------------------------------
# Inner product on C[0,1]
# <u,v> = ∫_0^1 x u(x) v(x) dx
# -------------------------------------------------
def inner_product(u, v):
    return sp.integrate(sp.expand(x * u * v), (x, 0, 1))

# -------------------------------------------------
# Raw polynomial basis (USE SYMBOLIC OBJECTS ONLY)
# -------------------------------------------------
raw_basis = [sp.Integer(1), x, x**2, x**3]

# -------------------------------------------------
# Gram–Schmidt (STABLE + SYMBOLIC SAFE)
# -------------------------------------------------
orthogonal_basis = []
orthonormal_basis = []
norms = []

for v in raw_basis:
    w = v
    for u in orthogonal_basis:
        w -= inner_product(w, u) / inner_product(u, u) * u
        w = sp.expand(w)

    norm_sq = inner_product(w, w)
    norm = sp.sqrt(norm_sq)

    orthogonal_basis.append(w)
    orthonormal_basis.append(sp.simplify(w / norm))
    norms.append(norm)

# -------------------------------------------------
# Print orthonormal basis
# -------------------------------------------------
print("Orthonormal basis on C[0,1] (polynomial subspace):\n")
for i, phi in enumerate(orthonormal_basis):
    print(f"phi_{i}(x) =")
    sp.pprint(phi)
    print()

# -------------------------------------------------
# Functions to project
# -------------------------------------------------
f_sin = sp.sin(sp.pi * x)
f_cos = sp.cos(sp.pi * x)

# -------------------------------------------------
# SAFE projection coefficients
# (project using unnormalized basis)
# -------------------------------------------------
coeffs_sin = []
coeffs_cos = []

for w, norm in zip(orthogonal_basis, norms):
    coeffs_sin.append(sp.simplify(inner_product(f_sin, w) / norm))
    coeffs_cos.append(sp.simplify(inner_product(f_cos, w) / norm))

print("Projection coefficients for sin(pi x):")
for i, c in enumerate(coeffs_sin):
    print(f"c_{i} =", c)

print("\nProjection coefficients for cos(pi x):")
for i, c in enumerate(coeffs_cos):
    print(f"c_{i} =", c)

# -------------------------------------------------
# Symbolic projections
# -------------------------------------------------
sin_proj = sp.simplify(sum(c * phi for c, phi in zip(coeffs_sin, orthonormal_basis)))
cos_proj = sp.simplify(sum(c * phi for c, phi in zip(coeffs_cos, orthonormal_basis)))

print("\nProjection of sin(pi x):")
sp.pprint(sin_proj)

print("\nProjection of cos(pi x):")
sp.pprint(cos_proj)

# -------------------------------------------------
# Convert to numerical functions for plotting
# -------------------------------------------------
sin_exact = sp.lambdify(x, f_sin, "numpy")
cos_exact = sp.lambdify(x, f_cos, "numpy")
sin_proj_num = sp.lambdify(x, sin_proj, "numpy")
cos_proj_num = sp.lambdify(x, cos_proj, "numpy")

# -------------------------------------------------
# Plot and save
# -------------------------------------------------
x_plot = np.linspace(0, 1, 600)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(x_plot, sin_exact(x_plot), linewidth=2, label=r"$\sin(\pi x)$")
plt.plot(x_plot, sin_proj_num(x_plot), "--", label="Projection")
plt.title("Projection of sin($\pi x$)")
plt.xlabel("x")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(x_plot, cos_exact(x_plot), linewidth=2, label=r"$\cos(\pi x)$")
plt.plot(x_plot, cos_proj_num(x_plot), "--", label="Projection")
plt.title("Projection of cos($\pi x$)")
plt.xlabel("x")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("part_c_weighted_projection.pdf", dpi=300)
plt.show()