"""
Reference physical models used for the cross-validation (Section 5) of the review.

All models are deliberately minimal, single-file and dependency-light (numpy/scipy)
so that every curve in the manuscript can be regenerated from the stated parameters.

  * EDFA  : Giles-Desurvire two-level model, forward pumped, ASE neglected
            (numerical ODE)  vs  Saleh et al. (1990) closed-form photon-balance solution.
  * SOA   : steady-state travelling-wave model with A/B/C recombination
            (numerical)      vs  homogeneous-saturation closed form G = G0 exp[-(G-1)Pin/Psat].
  * Raman : counter-pumped distributed amplifier with pump depletion (two-point BVP)
            vs  undepleted-pump closed form.
"""
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import brentq

h, c, q = 6.62607015e-34, 2.99792458e8, 1.602176634e-19
DB = np.log(10) / 10.0          # dB -> neper (per unit length)


def dbm2w(x):
    return 1e-3 * 10 ** (np.asarray(x) / 10)


def w2dbm(p):
    return 10 * np.log10(np.asarray(p) / 1e-3)


# ---------------------------------------------------------------------------
# 1) EDFA  (Giles & Desurvire, JLT 9(2):271, 1991 - parameters of their Table I)
# ---------------------------------------------------------------------------
import cards as _cards

# All parameters are read from the card files in ../cards (layer L1); see cards.py for the labels.
_E = _cards.values(_cards.load("EDFA-GD91-T1"))
EDFA = dict(
    xi=_E["xi"],                                                           # saturation parameter [1/(m s)]
    lam_p=_E["lambda_p"] * 1e-9, alpha_p=_E["alpha_p"] * DB, g_p=_E["g_p"] * DB,   # pump abs./gain coefficients [1/m]
    lam_s=_E["lambda_s"] * 1e-9, alpha_s=_E["alpha_s"] * DB, g_s=_E["g_s"] * DB,   # signal abs./gain coefficients [1/m]
)


def edfa_inversion(P, alpha, g, nu, xi):
    """Normalised upper-level population n2/nt (steady state, two-level)."""
    phi = P / (h * nu * xi)
    return np.sum(phi * alpha) / (1.0 + np.sum(phi * (alpha + g)))


def edfa_ode(Pp0, Ps0, L, p=EDFA, npts=301):
    nu = c / np.array([p["lam_p"], p["lam_s"]])
    alpha = np.array([p["alpha_p"], p["alpha_s"]])
    g = np.array([p["g_p"], p["g_s"]])

    def rhs(z, P):
        n2 = edfa_inversion(P, alpha, g, nu, p["xi"])
        return ((alpha + g) * n2 - alpha) * P

    z = np.linspace(0, L, npts)
    sol = solve_ivp(rhs, (0, L), [Pp0, Ps0], t_eval=z, rtol=1e-10, atol=1e-16, method="LSODA")
    return sol.t, sol.y[0], sol.y[1]


def edfa_saleh(Pp0, Ps0, L, p=EDFA):
    """Closed-form solution (Saleh, Jopson, Evankow, Aspell, IEEE PTL 2(10):714, 1990).
    G_k = exp[(alpha_k+g_k)(Q_in-Q_out)/xi - alpha_k L], Q = total photon flux."""
    nu = c / np.array([p["lam_p"], p["lam_s"]])
    alpha = np.array([p["alpha_p"], p["alpha_s"]])
    g = np.array([p["g_p"], p["g_s"]])
    Pin = np.array([Pp0, Ps0])
    Qin = np.sum(Pin / (h * nu))

    def f(Qout):
        with np.errstate(over="ignore"):
            G = np.exp(np.clip((alpha + g) * (Qin - Qout) / p["xi"] - alpha * L, -700, 700))
            return np.sum(Pin * G / (h * nu)) - Qout

    Qout = brentq(f, 1e-12 * Qin, 2 * Qin + 1e-9, xtol=1e-6)
    G = np.exp((alpha + g) * (Qin - Qout) / p["xi"] - alpha * L)
    return Pin * G


# ---------------------------------------------------------------------------
# 2) SOA  (bulk InGaAsP, representative parameters; ASE neglected)
# ---------------------------------------------------------------------------
_S = _cards.values(_cards.load("SOA-BULK-REF"))
SOA = dict(
    L=_S["L"], w=_S["w"], d=_S["d"],                       # active region length / width / thickness [m]
    Gamma=_S["Gamma"], a=_S["a"], N0=_S["N0"],             # confinement, differential gain [m^2], transparency [m^-3]
    alpha_int=_S["alpha_int"],                             # internal loss [1/m]
    A=_S["A"], B=_S["B"], C=_S["C"],                       # recombination coefficients
    lam=_S["lam"] * 1e-9,
)


def soa_R(N, p=SOA):
    return p["A"] * N + p["B"] * N ** 2 + p["C"] * N ** 3


def soa_carrier(I, P, p=SOA):
    """Local steady-state carrier density for current I [A] and local signal power P [W]."""
    V = p["L"] * p["w"] * p["d"]
    Aact = p["w"] * p["d"]
    hv = h * c / p["lam"]

    def f(N):
        return I / (q * V) - soa_R(N, p) - p["Gamma"] * p["a"] * (N - p["N0"]) * P / (hv * Aact)

    return brentq(f, 1e18, 1e26, xtol=1e14)


def soa_numerical(I, Pin, p=SOA, npts=400):
    def rhs(z, y):
        P = y[0]
        N = soa_carrier(I, P, p)
        return [(p["Gamma"] * p["a"] * (N - p["N0"]) - p["alpha_int"]) * P]

    sol = solve_ivp(rhs, (0, p["L"]), [Pin], rtol=1e-9, atol=1e-18, t_eval=np.linspace(0, p["L"], npts))
    return sol.t, sol.y[0]


def soa_analytic(I, Pin, p=SOA):
    """Homogeneous-saturation closed form (Agrawal & Olsson 1989 / Saleh):
    G = G0 exp[-(G-1) Pin/Psat], with G0 from the unsaturated net gain and
    Psat = h nu A_act / (Gamma a tau_d), tau_d = differential carrier lifetime."""
    N = soa_carrier(I, 0.0, p)
    G0 = np.exp((p["Gamma"] * p["a"] * (N - p["N0"]) - p["alpha_int"]) * p["L"])
    tau_d = 1.0 / (p["A"] + 2 * p["B"] * N + 3 * p["C"] * N ** 2)
    Psat = h * c / p["lam"] * p["w"] * p["d"] / (p["Gamma"] * p["a"] * tau_d)
    G = brentq(lambda G: G - G0 * np.exp(-(G - 1) * Pin / Psat), 1.0, G0)
    return G, G0, Psat


# ---------------------------------------------------------------------------
# 3) Distributed Raman, counter-pumped, standard SMF (G.652)
# ---------------------------------------------------------------------------
_R = _cards.values(_cards.load("RAMAN-G652"))
RAMAN = dict(
    L=_R["L"], lam_p=_R["lam_p"], lam_s=_R["lam_s"],
    gR=_R["gR"],                    # peak Raman gain coefficient for the card pump wavelength [m/W]
    Aeff=_R["Aeff"],                # effective area [m^2]
    K=_R["K"],                      # polarization factor
    alpha_s=_R["alpha_s"] * DB * 1e-3, alpha_p=_R["alpha_p"] * DB * 1e-3,   # [1/m]
)


def raman_CR(p=RAMAN):
    return p["gR"] / (p["K"] * p["Aeff"])      # [1/(W m)]


def raman_numerical(Pp0, Ps0, p=RAMAN, n=401):
    """Two-point BVP: Ps(0)=Ps0 (signal launched at z=0), Pp(L)=Pp0 (pump launched at z=L)."""
    CR, r = raman_CR(p), p["lam_s"] / p["lam_p"]      # r = nu_p/nu_s
    L = p["L"]

    def f(z, y):     # y = [ln Ps, ln Pp]
        Ps, Pp = np.exp(y[0]), np.exp(y[1])
        return np.vstack([CR * Pp - p["alpha_s"], p["alpha_p"] + r * CR * Ps])

    def bc(ya, yb):
        return np.array([ya[0] - np.log(Ps0), yb[1] - np.log(Pp0)])

    z = np.linspace(0, L, n)
    ys, yp = raman_analytic(Pp0, Ps0, z, p)
    sol = solve_bvp(f, bc, z, np.vstack([np.log(ys), np.log(yp)]), tol=1e-8, max_nodes=100000)
    return z, np.exp(sol.sol(z)[0]), np.exp(sol.sol(z)[1]), sol.status


def raman_analytic(Pp0, Ps0, z, p=RAMAN):
    """Undepleted-pump closed form."""
    CR, L = raman_CR(p), p["L"]
    Pp = Pp0 * np.exp(-p["alpha_p"] * (L - z))
    # integral of Pp from 0..z
    intPp = Pp0 * np.exp(-p["alpha_p"] * L) * (np.exp(p["alpha_p"] * z) - 1) / p["alpha_p"]
    Ps = Ps0 * np.exp(CR * intPp - p["alpha_s"] * z)
    return Ps, Pp


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))
