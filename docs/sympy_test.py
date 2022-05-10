# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: pyobjmap
#     language: python
#     name: pyobjmap
# ---

# %% [markdown]
# # sympy for symbolic mathematics
#
# This might be useful for coding the derivatives of the covariance functions.

# %%
import sympy as sym
from sympy.printing.pycode import PythonCodePrinter

x, y, z, r, l, h, A = sym.symbols("x y z r l h A")
rdef = sym.sqrt(x**2 + y**2)

# %% [markdown]
# Can sub `sym.sqrt(x**2 + y**2)` later?

# %% [markdown]
# ## Gaussian

# %%
C = A*sym.exp(-r**2/(2*l**2))
C

# %%
C = A*sym.exp(-r**2/(2*l**2))
print(PythonCodePrinter().doprint(C))

# %%
R = (-1/r)*sym.diff(C, r)
print(PythonCodePrinter().doprint(R))

# %%
S = sym.simplify(-sym.diff(C, r, 2))
print(PythonCodePrinter().doprint(S))

# %%
Cuu = sym.simplify(x**2 * (R - S) / r**2 + S)
print(PythonCodePrinter().doprint(Cuu))

# %%
Cvv = sym.simplify(y**2 * (R - S) / r**2 + S)
print(PythonCodePrinter().doprint(Cvv))

# %%
Cuv = sym.simplify(x * y * (R - S) / r**2)
print(PythonCodePrinter().doprint(Cuv))

# %%
Cpsiu = y * R
print(PythonCodePrinter().doprint(Cpsiu))

# %%
Cpsiv = - x * R
print(PythonCodePrinter().doprint(Cpsiv))

# %% [markdown]
# Manual gradients...

# %%
# Cuu
C.subs(r, rdef).diff(y).diff(y).simplify().subs(x**2 + y**2, r**2)

# %%
# Cvv
C.subs(r, rdef).diff(x).diff(x).simplify().subs(x**2 + y**2, r**2)

# %%
# Cuv
-C.subs(r, rdef).diff(x).diff(y).simplify().subs(x**2 + y**2, r**2)

# %%
# Cpsiu
-C.subs(r, rdef).diff(y).simplify().subs(x**2 + y**2, r**2)

# %%
# Cpsiv
C.subs(r, rdef).diff(x).simplify().subs(x**2 + y**2, r**2)

# %% [markdown]
# ## Letra

# %%
Cr_letra = sym.simplify(A*(1 + r/l + (r/l)**2/6 - (r/l)**3/6)*sym.exp(-r/l))
Cr_letra

# %%
Cuu_letra = Cr_letra.subs(r, rdef).diff(y).diff(y).simplify().subs(sym.sqrt(x**2 + y**2), r)
Cuu_letra

# %%
Cvv_letra = Cr_letra.subs(r, rdef).diff(x).diff(x).simplify().subs(sym.sqrt(x**2 + y**2), r)
Cvv_letra

# %%
Cuv_letra = -Cr_letra.subs(r, rdef).diff(y).diff(x).simplify().subs(sym.sqrt(x**2 + y**2), r).subs(x**2 + y**2, r**2)
Cuv_letra

# %% [markdown]
# ## Simple Gaussian 2D

# %%
C = A*sym.exp(-x**2/(2*l**2) - z**2/(2*h**2))
C

# %%
C.integrate(x)
