# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
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
from sympy.printing.pycode import NumPyPrinter

x, y, r, l, A = sym.symbols("x y r l A")
rdef = sym.sqrt(x**2 + y**2)

# %% [markdown]
# Can sub `sym.sqrt(x**2 + y**2)` later?

# %% [markdown]
# ## Gaussian

# %%
C = A*sym.exp(-r**2/(2*l**2))
print(NumPyPrinter().doprint(C))

# %%
R = (-1/r)*sym.diff(C, r)
print(NumPyPrinter().doprint(R))

# %%
S = sym.simplify(-sym.diff(C, r, 2))
print(NumPyPrinter().doprint(S))

# %%
Cuu = sym.simplify(x**2 * (R - S) / r**2 + S)
print(NumPyPrinter().doprint(Cuu))

# %%
Cvv = sym.simplify(y**2 * (R - S) / r**2 + S)
print(NumPyPrinter().doprint(Cvv))

# %%
Cuv = sym.simplify(x * y * (R - S) / r**2)
print(NumPyPrinter().doprint(Cuv))

# %%
Cpsiu = y * R
print(NumPyPrinter().doprint(Cpsiu))

# %%
Cpsiv = - x * R
print(NumPyPrinter().doprint(Cpsiv))

# %% [markdown]
# ## Letra

# %%
Cr_letra = sym.simplify(A*(1 + r/l + (r/l)**2/6 - (r/l)**3/6)*sym.exp(-r/l))
print(Cr_letra)

# %%
R_letra = sym.simplify((-1/r)*sym.diff(Cr_letra, r))
print(R_letra)

# %%
S_letra = sym.simplify(-sym.diff(Cr_letra, r, 2))
print(S_letra)
