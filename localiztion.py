"""
Maksim Stepnov 15.07.2026

Numerical verification + figures for:

Localization of the Intermediate Point at Every Order
 
Core object: mu = probability weight on [0,1].

Mean-value equation:  g(theta) = E_mu[g],  g = (n+1)st derivative in normalized coords.

Admissibility (main theorem):  1/rho <= A(theta)/B(theta) <= rho,  rho = Lambda/lambda,

where A(t) = E[(S-t)_+], B(t) = E[(t-S)_+].

"""
import numpy as np
from scipy.special import betainc, betaln
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# ---------------------------------------------------------------- weights
def beta_cdf(s, a, b):
    return betainc(a, b, np.clip(s, 0, 1))
 
def AB_beta(t, a, b):
    """A(t)=E[(S-t)_+], B(t)=E[(t-S)_+] for S~Beta(a,b)."""
    A = quad(lambda s: 1.0 - beta_cdf(s, a, b), t, 1.0, limit=200)[0]
    B = quad(lambda s: beta_cdf(s, a, b), 0.0, t, limit=200)[0]
    return A, B
 
def AB_lagrange(t, n):
    """Closed form for mu = Beta(1,n+1), density (n+1)(1-s)^n.  N=n+2."""
    N = n + 2.0
    u = 1.0 - t
    A = u**N / N
    B = t - (1.0 - u**N) / N
    return A, B
 
def Psi_lagrange(t, n):
    A, B = AB_lagrange(t, n)
    return A / B if B > 0 else np.inf
 
def band_lagrange(n, rho):
    """Exact admissible interval [t-,t+] for the Lagrange weight."""
    lo = brentq(lambda t: Psi_lagrange(t, n) - rho, 1e-14, 1.0 - 1e-14)
    hi = brentq(lambda t: Psi_lagrange(t, n) - 1.0 / rho, 1e-14, 1.0 - 1e-14)
    return lo, hi
 
def band_beta(a, b, rho):
    f = lambda t, r: (lambda AB: AB[0] - r * AB[1])(AB_beta(t, a, b))
    lo = brentq(lambda t: f(t, rho), 1e-9, 1 - 1e-9)
    hi = brentq(lambda t: f(t, 1.0 / rho), 1e-9, 1 - 1e-9)
    return lo, hi
 
# ---------------------------------------------------------------- checks
print("=" * 68)
print("CHECK 1  anchor = mean of weight;  Psi(anchor) = 1")
for n in [0, 1, 2, 5, 10, 40]:
    anchor = 1.0 / (n + 2)
    print(f"  n={n:3d}  1/(n+2)={anchor:.6f}   Psi={Psi_lagrange(anchor, n):.12f}")
 
print("=" * 68)
print("CHECK 2  n=0 closed form  theta_- = 1/(1+sqrt(rho))")
for rho in [1.2, 2.0, 5.0, 20.0]:
    lo, hi = band_lagrange(0, rho)
    print(f"  rho={rho:5.1f}  numeric=({lo:.8f},{hi:.8f})  "
          f"closed=({1/(1+np.sqrt(rho)):.8f},{np.sqrt(rho)/(1+np.sqrt(rho)):.8f})")
 
print("=" * 68)
print("CHECK 3  MAD of Beta(1,n+1) closed form vs quadrature")
for n in [0, 1, 3, 8]:
    mad_cf = 2.0 / (n + 2) * ((n + 1) / (n + 2)) ** (n + 2)
    mad_q = quad(lambda s: abs(s - 1/(n+2)) * (n+1)*(1-s)**n, 0, 1, limit=200)[0]
    print(f"  n={n}  closed={mad_cf:.10f}  quad={mad_q:.10f}")
 
print("=" * 68)
print("CHECK 4  Monte Carlo: random admissible g stays inside the band")
rng = np.random.default_rng(7)
grid = np.linspace(0, 1, 4001)
def theta_of(gvals, n):
    w = (n + 1) * (1 - grid) ** n
    mean = np.trapezoid(gvals * w, grid)
    idx = np.where(np.diff(np.sign(gvals - mean)))[0][0]
    x0, x1 = grid[idx], grid[idx + 1]
    y0, y1 = gvals[idx] - mean, gvals[idx + 1] - mean
    return x0 - y0 * (x1 - x0) / (y1 - y0)
