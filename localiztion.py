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
 
# weights
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
 
# checks
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
 
viol = 0
records = []
for trial in range(4000):
    n = int(rng.integers(0, 13))
    rho = float(np.exp(rng.uniform(np.log(1.05), np.log(8.0))))
    lam, Lam = 1.0, rho
    k = int(rng.integers(1, 6))
    c = rng.uniform(-1, 1, k)
    raw = sum(c[j] * np.sin((j + 1) * np.pi * grid + rng.uniform(0, np.pi)) for j in range(k))
    raw = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-12)   # slope profile in [0,1]
    gp = lam + (Lam - lam) * raw
    g = np.concatenate([[0.0], np.cumsum((gp[1:] + gp[:-1]) / 2 * np.diff(grid))])
    t = theta_of(g, n)
    lo, hi = band_lagrange(n, rho)
    records.append((n, rho, t, lo, hi))
    if not (lo - 1e-6 <= t <= hi + 1e-6):
        viol += 1
print(f"  4000 random functions, violations = {viol}")
 
print("=" * 68)
print("CHECK 5  two-slope hinge functions attain the band endpoints exactly")
for n in [0, 2, 6]:
    for rho in [1.5, 3.0]:
        lo, hi = band_lagrange(n, rho)
        for target, lab in [(lo, "lo"), (hi, "hi")]:
            # target crossing point
            if lab == "lo":
                gp = np.where(grid > target, 1.0, rho)
            else:
                gp = np.where(grid > target, rho, 1.0)
            g = np.concatenate([[0.0], np.cumsum((gp[1:] + gp[:-1]) / 2 * np.diff(grid))])
            t = theta_of(g, n)
            print(f"  n={n} rho={rho} {lab}: target={target:.7f} realized={t:.7f}"
                  f"  err={abs(t-target):.2e}")
 
print("=" * 68)
print("CHECK 6  rescaled band x=(n+2)theta converges to the universal shape")
Psi_inf = lambda x: np.exp(-x) / (x - 1 + np.exp(-x))
for rho in [1.5, 2.0, 4.0]:
    xlo = brentq(lambda x: Psi_inf(x) - rho, 1e-9, 1.0 - 1e-12)
    xhi = brentq(lambda x: Psi_inf(x) - 1 / rho, 1.0 + 1e-12, 60)
    print(f"  rho={rho}: limit x-band = [{xlo:.6f},{xhi:.6f}] width={xhi-xlo:.6f}")
    for n in [0, 2, 10, 50, 400, 5000]:
        lo, hi = band_lagrange(n, rho)
        print(f"     n={n:5d}  x-band=[{(n+2)*lo:.6f},{(n+2)*hi:.6f}]  "
              f"width={(n+2)*(hi-lo):.6f}")
 
print("=" * 68)
print("CHECK 7  weighted case: anchor for Beta(a,b) is a/(a+b)")
for (a, b) in [(1, 3), (2, 2), (3, 1.5), (0.7, 4)]:
    A, B = AB_beta(a / (a + b), a, b)
    print(f"  Beta({a},{b}) anchor={a/(a+b):.6f}  A/B={A/B:.10f}")
 
# figures
plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY = "#1f4e79", "#b03a2e", "#7f8c8d"
 
# Figure 1
ns = np.arange(0, 21)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for rho, alpha in [(4.0, 0.13), (2.0, 0.20), (1.4, 0.30)]:
    los = np.array([band_lagrange(n, rho)[0] for n in ns])
    his = np.array([band_lagrange(n, rho)[1] for n in ns])
    ax.fill_between(ns, los, his, color=BLUE, alpha=alpha,
                    lw=0, label=rf"$\rho={rho}$")
ax.plot(ns, 1 / (ns + 2), color=RED, lw=1.6, label=r"anchor $1/(n+2)$")
rng2 = np.random.default_rng(3)
for (n, rho, t, lo, hi) in records[:900]:
    if rho < 4.0 and n <= 20:
        ax.plot(n + rng2.uniform(-.22, .22), t, ".", ms=2.0, color="#222222", alpha=.35)
ax.set_xlabel("order $n$"); ax.set_ylabel(r"normalized position $\theta$")
ax.set_title("Admissible region for the intermediate point")
ax.set_ylim(0, 0.85); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig1_headline.pdf"); fig.savefig("fig1_headline.png"); plt.close(fig)
