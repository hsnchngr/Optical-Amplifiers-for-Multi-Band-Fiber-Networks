"""Figure 11 - Code verification and model inter-comparison of the reference models (Section 5.4).

No measurement enters this figure. Two kinds of comparison are made and kept apart:
  code verification       a numerical solver against an exact solution of the SAME equations. The residual
                          measures the numerics (integrator, tolerance, discretization), not model accuracy.
  model inter-comparison  a simplified closed form against a more complete numerical model with the same card.
                          The deviation is the model-form difference of the simplification; it is not a validation.

Panels
  a) EDFA, card EDFA-GD91-T1: ODE solution (models.edfa_ode) and Saleh closed form (models.edfa_saleh)
  b) code-verification residual of (a) versus the relative tolerance of three integrators
  c) SOA, card SOA-BULK-REF: full numerical model, intermediate model, calibrated homogeneous-saturation closed form
  d) deviation of that closed form versus input power, split into an internal-loss and a recombination part
  e) Raman, card RAMAN-G652: depleted-pump BVP (models.raman_numerical) and undepleted closed form
  f) deviation of the undepleted closed form versus signal launch power

Sign convention: every deviation is (closed form or numerical solution under test) minus (reference), in dB.
Metrics: root-mean-square value (RMSE), maximum absolute value and bias (mean) over the grid stated with each number.

Outputs: ../figures/Fig11_model_intercomparison.png and .pdf, ../figures/Fig11_model_intercomparison_results.json
(output names kept from earlier versions; this script was called fig07_crossvalidation.py before revision 3)
Run from any directory:  python fig_intercomparison.py
"""
import json
import os
import platform
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import brentq

import style
from models import (EDFA, SOA, RAMAN, h, c, q, dbm2w, w2dbm, edfa_ode, edfa_saleh, edfa_inversion,
                    soa_carrier, soa_R, soa_numerical, soa_analytic, raman_CR, raman_numerical, raman_analytic)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
NAME = "Fig11_model_intercomparison"
T_START = time.perf_counter()

# Operating points (configuration of this figure; the material parameters come from the cards via models.py)
OP = {
    "EDFA": {"card": "EDFA-GD91-T1", "pump_W": 0.100, "pump_nm": 1480, "pump_direction": "forward",
             "signal_dBm": -20.0, "signal_nm": 1550, "lengths_m": list(np.arange(1.0, 21.0, 1.0)),
             "physics": "two-level Giles model; ASE and background loss excluded in both solutions"},
    "SOA": {"card": "SOA-BULK-REF", "currents_A": [0.12, 0.15], "signal_nm": 1550,
            "input_grid_dBm": "-40 to +5 dBm in 1 dB steps (46 points)",
            "physics": "steady-state traveling-wave model, chip gain (facet to facet), ASE neglected"},
    "RAMAN": {"card": "RAMAN-G652", "pump_W": 0.500, "pump_nm": 1450, "pump_direction": "counter-propagating",
              "span_km": RAMAN["L"] / 1e3, "signal_nm": 1550,
              "loads": {"single weak channel": -10.0, "lumped load": 16.0},
              "load_note": "the +16 dBm load is ONE lumped signal at 1550 nm standing for the total power of "
                           "40 channels at 0 dBm; inter-channel SRS, the spectral variation of the Raman gain "
                           "across the load, ASE and double Rayleigh backscattering are not modeled",
              "grid": "401 equally spaced positions from 0 to 50 km (125 m spacing)",
              "physics": "pump-signal coupled equations, one pump and one signal"},
}


def db(x):
    return 10.0 * np.log10(x)


def metrics(dev, grid):
    """RMSE, maximum absolute value and bias (mean) of a deviation array [dB], with its grid description."""
    dev = np.asarray(dev, float)
    return {"rmse_dB": float(np.sqrt(np.mean(dev ** 2))), "max_abs_dB": float(np.max(np.abs(dev))),
            "bias_dB": float(np.mean(dev)), "n_points": int(dev.size), "grid": grid}


R = {"figure": NAME, "script": "fig_intercomparison.py",
     "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
     "sign_convention": "deviation = (closed form or numerical solution under test) - (reference), in dB",
     "operating_points": OP}

# =====================================================================================================
# 1) EDFA: code verification of the ODE solver against the exact Saleh solution (Eq. 5)
# =====================================================================================================
E = OP["EDFA"]
Pp0, Ps0 = E["pump_W"], dbm2w(E["signal_dBm"])
Lk = np.array(E["lengths_m"])
LGRID = "20 fiber lengths, 1 to 20 m in 1 m steps (one integration per length)"
G_eq5 = np.array([db(edfa_saleh(Pp0, Ps0, L)[1] / Ps0) for L in Lk])
Pp_eq5 = np.array([edfa_saleh(Pp0, Ps0, L)[0] for L in Lk])


def edfa_ivp(L, method, rtol, atol, p=EDFA):
    """Right-hand side of models.edfa_ode (forward pump, ASE and background loss excluded) with the integrator
    and its tolerances as arguments. Returns the output signal power [W] and the number of RHS evaluations."""
    nu = c / np.array([p["lam_p"], p["lam_s"]])
    al = np.array([p["alpha_p"], p["alpha_s"]])
    g = np.array([p["g_p"], p["g_s"]])
    sol = solve_ivp(lambda z, P: ((al + g) * edfa_inversion(P, al, g, nu, p["xi"]) - al) * P, (0.0, L), [Pp0, Ps0],
                    t_eval=[0.0, L], method=method, rtol=rtol, atol=atol)
    return sol.y[1, -1], sol.nfev


# production function (models.edfa_ode: LSODA, rtol 1e-10, atol 1e-16 W)
G_prod = np.array([db(edfa_ode(Pp0, Ps0, L, npts=2)[2][-1] / Ps0) for L in Lk])
G_copy = np.array([db(edfa_ivp(L, "LSODA", 1e-10, 1e-16)[0] / Ps0) for L in Lk])
assert np.max(np.abs(G_copy - G_prod)) < 1e-12, "local copy of the EDFA right-hand side differs from models.edfa_ode"

RTOLS = 10.0 ** -np.arange(4, 13)              # 1e-4 ... 1e-12
ATOL_PER_RTOL = 1e-6                           # atol = 1e-6 W x rtol, i.e. rtol relative to 1 uW (input signal 10 uW)
tol_study = {}
for method in ("LSODA", "RK45", "DOP853", "Radau"):
    rows = []
    for rt in RTOLS:
        out = [edfa_ivp(L, method, rt, ATOL_PER_RTOL * rt) for L in Lk]
        dev = db(np.array([o[0] for o in out]) / Ps0) - G_eq5
        m = metrics(dev, LGRID)
        m.update({"rtol": float(rt), "atol_W": float(ATOL_PER_RTOL * rt), "rhs_evaluations_20m": int(out[-1][1])})
        rows.append(m)
    tol_study[method] = rows
lsoda_fixed_atol = []
for rt in RTOLS:                               # production atol (1e-16 W) kept fixed, for comparison with earlier checks
    dev = db(np.array([edfa_ivp(L, "LSODA", rt, 1e-16)[0] for L in Lk]) / Ps0) - G_eq5
    m = metrics(dev, LGRID)
    m.update({"rtol": float(rt), "atol_W": 1e-16})
    lsoda_fixed_atol.append(m)

# termination bound of the bracketing root finder used for Eq. (5) (brentq: xtol 1e-6 plus 4 eps |Q_out|)
nu_ps = c / np.array([EDFA["lam_p"], EDFA["lam_s"]])
Q_in = Pp0 / (h * nu_ps[0]) + Ps0 / (h * nu_ps[1])
eq5_floor_dB = float(10 / np.log(10) * (EDFA["alpha_s"] + EDFA["g_s"]) / EDFA["xi"]
                     * (1e-6 + 4 * np.finfo(float).eps * Q_in))


def eq5_cancellation_free(L, p=EDFA):
    """Eq. (5) solved for u = Q_in - Q_out instead of Q_out, which avoids the cancellation between two nearly equal
    photon fluxes; used only to locate the origin of the residual floor in panel b."""
    al = np.array([p["alpha_p"], p["alpha_s"]])
    g = np.array([p["g_p"], p["g_s"]])
    Qk = np.array([Pp0, Ps0]) / (h * nu_ps)
    with np.errstate(over="ignore"):
        u = brentq(lambda v: v - Qk.sum() + np.sum(Qk * np.exp((al + g) * v / p["xi"] - al * L)), 0.0, Qk.sum(),
                   xtol=1e-30, rtol=4 * np.finfo(float).eps, maxiter=500)
    return db(np.exp((al[1] + g[1]) * u / p["xi"] - al[1] * L))


G_eq5_cf = np.array([eq5_cancellation_free(L) for L in Lk])
G_radau_tight = db(np.array([edfa_ivp(L, "Radau", 1e-12, 1e-18)[0] for L in Lk]) / Ps0)
floor_origin = {
    "models.edfa_saleh_minus_cancellation_free_eq5": metrics(G_eq5 - G_eq5_cf, LGRID),
    "Radau_rtol1e-12_minus_cancellation_free_eq5": metrics(G_radau_tight - G_eq5_cf, LGRID),
    "conclusion": "the floor of about 2e-13 dB in panel b is the precision of models.edfa_saleh (root finding for "
                  "Q_out), not of the integrators",
}
R["EDFA"] = {
    "gain_eq5_dB": {f"{L:.0f} m": float(G) for L, G in zip(Lk, G_eq5)},
    "pump_out_eq5_dBm": {f"{L:.0f} m": float(w2dbm(P)) for L, P in zip(Lk, Pp_eq5)},
    "residual_default_setting": dict(metrics(G_prod - G_eq5, LGRID), method="LSODA", rtol=1e-10, atol_W=1e-16,
                                     where="models.edfa_ode"),
    "tolerance_study": {"atol_rule": "atol = 1e-6 W x rtol", "methods": tol_study},
    "tolerance_study_LSODA_fixed_atol_1e-16W": lsoda_fixed_atol,
    "eq5_root_finding_bound_dB": eq5_floor_dB,
    "residual_floor_origin": floor_origin,
    "interpretation": "residual between two solutions of identical ASE-free equations; it follows the integrator "
                      "tolerance and says nothing about the accuracy of either model for a real amplifier",
}

# Optional diagnostic, NOT plotted: ASE contribution at the same operating point with edfa_multipass.solve (owned by
# another script, read-only here). That solver lumps all ASE into ONE band at 1550 nm with the 1550 nm coefficients,
# because card EDFA-GD91-T1 gives alpha and g* only at 1480 and 1550 nm. Its result scales with the assumed band width
# and misses the 1530 nm emission peak, so it is reported as an indication only.
try:
    import edfa_multipass as em

    def em_gain(L, band_hz):
        saved = em.P["dnu_ase"]
        try:
            em.P["dnu_ase"] = band_hz           # band of the lumped ASE channel; 1e-30 Hz switches spontaneous emission off
            r = em.solve(Pp0, Ps0, L, double_pass=False, N=800, tol=1e-7, max_iter=400)
        finally:
            em.P["dnu_ase"] = saved
        return db(r["G"]), int(r["sweeps"])

    bands = {"ASE off": 1e-30, "10 nm (1545-1555 nm)": c / 1545e-9 - c / 1555e-9,
             "30 nm (1535-1565 nm, solver default)": c / 1535e-9 - c / 1565e-9,
             "50 nm (1525-1575 nm)": c / 1525e-9 - c / 1575e-9}
    G_em = {k: np.array([em_gain(L, b)[0] for L in Lk]) for k, b in bands.items()}
    ase = {"model": "edfa_multipass.solve, single pass, 800 nodes, forward/backward lumped ASE channel at 1550 nm",
           "status": "indicative only, not plotted (not clean: depends on the assumed lumped band, no spectral data "
                     "in the card)",
           "sweep_solver_ASE_off_minus_eq5": metrics(G_em["ASE off"] - G_eq5, LGRID + ", 800 nodes")}
    for k in list(bands)[1:]:
        dev = G_em["ASE off"] - G_em[k]        # gain lost to ASE, same solver and grid
        ase[k] = dict(metrics(dev, LGRID), at_10m_dB=float(dev[9]), at_15m_dB=float(dev[14]), at_20m_dB=float(dev[19]))
    R["EDFA"]["ase_indication_not_plotted"] = ase
except Exception as err:                       # the diagnostic must not stop the figure
    R["EDFA"]["ase_indication_not_plotted"] = {"status": f"not computed: {err!r}"}

# =====================================================================================================
# 2) SOA: calibrated closed form (Eq. 8) against the numerical model, with the intermediate model
# =====================================================================================================
S = OP["SOA"]
Pin_dbm = np.linspace(-40.0, 5.0, 46)
SGRID = "input power -40 to +5 dBm, 1 dB steps"


def soa_unsaturated(I, p=SOA):
    """Carrier density, differential lifetime, saturation power and modal material gain at zero signal power.
    These are the quantities that models.soa_analytic takes from the numerical model (the calibration)."""
    N = soa_carrier(I, 0.0, p)
    tau_d = 1.0 / (p["A"] + 2 * p["B"] * N + 3 * p["C"] * N ** 2)
    Psat = h * c / p["lam"] * p["w"] * p["d"] / (p["Gamma"] * p["a"] * tau_d)
    return N, tau_d, Psat, p["Gamma"] * p["a"] * (N - p["N0"])


def soa_linearized(I, Pin, loss=True, p=SOA):
    """Traveling-wave model of models.soa_numerical with R(N) replaced by its tangent at the unsaturated density
    (lifetime tau_d). loss=False also removes the internal loss. Same integrator settings as models.soa_numerical."""
    V, Aact, hv = p["L"] * p["w"] * p["d"], p["w"] * p["d"], h * c / p["lam"]
    Nus, tau_d, _, _ = soa_unsaturated(I, p)
    Rus = soa_R(Nus, p)
    aint = p["alpha_int"] if loss else 0.0

    def carrier(P):     # I/(qV) - [R(Nus) + (N - Nus)/tau_d] - Gamma a (N - N0) P/(h nu A) = 0, solved for N
        k = p["Gamma"] * p["a"] * P / (hv * Aact)
        return (I / (q * V) - Rus + Nus / tau_d + k * p["N0"]) / (1.0 / tau_d + k)

    sol = solve_ivp(lambda z, y: [(p["Gamma"] * p["a"] * (carrier(y[0]) - p["N0"]) - aint) * y[0]],
                    (0.0, p["L"]), [Pin], rtol=1e-9, atol=1e-18)
    return sol.y[0, -1] / Pin


def soa_full(I, Pin):
    return soa_numerical(I, Pin, npts=2)[1][-1] / Pin


def homogeneous(G0, Pin, Psat):
    return brentq(lambda G: G - G0 * np.exp(-(G - 1) * Pin / Psat), 1.0, G0)


def compression_3dB(gain_db):
    """Input and output power [dBm] at which the gain is 3 dB below its value at -60 dBm input."""
    target = gain_db(-60.0) - 3.0
    x = brentq(lambda v: gain_db(v) - target, -40.0, 10.0, xtol=1e-7)
    return float(x), float(x + gain_db(x))


SOA_RES, SOA_CURVES = {}, {}
for I in S["currents_A"]:
    key = f"{I * 1e3:.0f} mA"
    Nus, tau_d, Psat, g_mat = soa_unsaturated(I)
    G0_cf, Psat_cf = soa_analytic(I, 1e-12)[1:]
    Gf = np.array([db(soa_full(I, dbm2w(x))) for x in Pin_dbm])
    Gl = np.array([db(soa_linearized(I, dbm2w(x))) for x in Pin_dbm])
    Gc = np.array([db(soa_analytic(I, dbm2w(x))[0]) for x in Pin_dbm])
    # code verification: internal loss removed and recombination linearized -> must equal Eq. (8) with material G0
    G0_mat = np.exp(g_mat * SOA["L"])
    Gl0 = np.array([db(soa_linearized(I, dbm2w(x), loss=False)) for x in Pin_dbm])
    Gc0 = np.array([db(homogeneous(G0_mat, dbm2w(x), Psat)) for x in Pin_dbm])
    tot, loss, rec = Gc - Gf, Gc - Gl, Gl - Gf
    SOA_CURVES[key] = dict(full=Gf, lin=Gl, cf=Gc, tot=tot, loss=loss, rec=rec)
    windows = {"-40 to -20 dBm": Pin_dbm <= -20, "-40 to -10 dBm": Pin_dbm <= -10, "-40 to +5 dBm": Pin_dbm <= 5}
    dev = {}
    for name, d in (("total (closed form - numerical)", tot), ("internal-loss part (closed form - intermediate)", loss),
                    ("recombination part (intermediate - numerical)", rec)):
        dev[name] = {w: metrics(d[m], f"{SGRID}, window {w}") for w, m in windows.items()}
        dev[name]["at_input_dBm"] = {f"{x:+.0f}": float(d[Pin_dbm == x][0]) for x in (-40, -30, -20, -10, 0, 5)}
    i0 = int(np.where(Pin_dbm == 0)[0][0])
    comp = {}
    for name, fun in (("numerical", lambda v: db(soa_full(I, dbm2w(v)))),
                      ("intermediate", lambda v: db(soa_linearized(I, dbm2w(v)))),
                      ("closed form", lambda v: db(soa_analytic(I, dbm2w(v))[0]))):
        pin3, pout3 = compression_3dB(fun)
        comp[name] = {"input_dBm": pin3, "output_dBm": pout3}
    SOA_RES[key] = {
        "calibration_from_numerical_model": {
            "N_unsaturated_m-3": float(Nus), "tau_d_ns": float(tau_d * 1e9),
            "G0_dB_net_including_internal_loss": float(db(G0_cf)), "G0_dB_material_only": float(db(G0_mat)),
            "Psat_internal_dBm": float(w2dbm(Psat_cf)), "Esat_pJ": float(Psat_cf * tau_d * 1e12),
            "note": "G0 and tau_d are computed from the card through the carrier equation of the numerical model at zero "
                    "signal power; the closed form therefore agrees with the numerical model at small signal by construction"},
        "small_signal_gain_dB": {"numerical": float(Gf[0]), "intermediate": float(Gl[0]), "closed form": float(Gc[0]),
                                 "at": "-40 dBm input"},
        "deviation": dev,
        "share_of_total_at_0dBm": {"internal loss": float(loss[i0] / tot[i0]), "recombination": float(rec[i0] / tot[i0])},
        "share_of_total_rmse_-40_to_+5dBm": {"internal loss": float(np.sqrt(np.mean(loss ** 2)) / np.sqrt(np.mean(tot ** 2))),
                                             "recombination": float(np.sqrt(np.mean(rec ** 2)) / np.sqrt(np.mean(tot ** 2)))},
        "compression_3dB": comp,
        "code_verification_linear_R_no_loss_vs_eq8": metrics(Gl0 - Gc0, SGRID),
    }
    P_out0 = soa_numerical(I, dbm2w(0.0), npts=2)[1][-1]     # numerical model, 0 dBm input
    N_out0 = soa_carrier(I, P_out0)
    SOA_RES[key]["numerical_model_output_at_0dBm_input"] = {
        "P_out_dBm": float(w2dbm(P_out0)), "N_m-3": float(N_out0), "N_change_vs_unsaturated": float(N_out0 / Nus - 1),
        "tau_d_ns": float(1e9 / (SOA["A"] + 2 * SOA["B"] * N_out0 + 3 * SOA["C"] * N_out0 ** 2))}
R["SOA"] = SOA_RES

# =====================================================================================================
# 3) Raman: undepleted closed form (Eq. 12) against the depleted-pump BVP
# =====================================================================================================
Rm = OP["RAMAN"]
Pp_R = Rm["pump_W"]
ZGRID = Rm["grid"]
RAMAN_CASES = {}
for name, x in Rm["loads"].items():
    z, Ps, Pp, status = raman_numerical(Pp_R, dbm2w(x))
    Pa, Ppa = raman_analytic(Pp_R, dbm2w(x), z)
    d = w2dbm(Pa) - w2dbm(Ps)
    RAMAN_CASES[name] = dict(z=z, rel_bvp=w2dbm(Ps) - x, rel_cf=w2dbm(Pa) - x, dev=d,
                             res=dict(metrics(d, ZGRID), launch_dBm=x, bvp_status=int(status),
                                      span_output_error_dB=float(d[-1]),
                                      net_gain_bvp_dB=float(w2dbm(Ps[-1]) - x), net_gain_closed_form_dB=float(w2dbm(Pa[-1]) - x),
                                      residual_pump_at_z0_bvp_dBm=float(w2dbm(Pp[0])),
                                      residual_pump_at_z0_closed_form_dBm=float(w2dbm(Ppa[0])),
                                      position_of_max_abs_km=float(z[np.argmax(np.abs(d))] / 1e3)))
CR, a_p, Lspan = raman_CR(), RAMAN["alpha_p"], RAMAN["L"]
Leff = (1 - np.exp(-a_p * Lspan)) / a_p

# launch-power sweep of the deviation
P_sweep = np.arange(-40.0, 20.5, 1.0)
sweep = []
for x in P_sweep:
    z, Ps, Pp, status = raman_numerical(Pp_R, dbm2w(x))
    Pa, _ = raman_analytic(Pp_R, dbm2w(x), z)
    d = w2dbm(Pa) - w2dbm(Ps)
    sweep.append(dict(metrics(d, ZGRID), launch_dBm=float(x), span_output_error_dB=float(d[-1]), bvp_status=int(status)))
out_err = np.array([s["span_output_error_dB"] for s in sweep])
rms_err = np.array([s["rmse_dB"] for s in sweep])
# proportionality of the deviation to the launch power at low power (first-order depletion) bounds the BVP error
e40, e30 = out_err[P_sweep == -40][0], out_err[P_sweep == -30][0]


def raman_bvp_no_depletion(Pp0_, Ps0_, p=RAMAN, n=401):
    """Code verification: models.raman_numerical with the depletion term removed and a flat initial guess must
    reproduce the closed form; same tolerance (1e-8), initial mesh and log-power formulation."""
    CR_, L_ = raman_CR(p), p["L"]
    z_ = np.linspace(0, L_, n)
    sol = solve_bvp(lambda zz, y: np.vstack([CR_ * np.exp(y[1]) - p["alpha_s"], p["alpha_p"] + 0.0 * y[0]]),
                    lambda ya, yb: np.array([ya[0] - np.log(Ps0_), yb[1] - np.log(Pp0_)]),
                    z_, np.vstack([np.full(n, np.log(Ps0_)), np.full(n, np.log(Pp0_))]), tol=1e-8, max_nodes=100000)
    return z_, np.exp(sol.sol(z_)[0]), sol.status, sol.x.size


# how representative is the lumped load? Raman gain efficiency of 40 channels over 1530-1565 nm relative to 1550 nm,
# from the default SSMF profile of GNPy in the data folder; first-order statement about the total pump depletion only
import cards
off_THz, g0_prof, _ = cards.raman_profile_ssmf()
f_pump, f_1550 = c / RAMAN["lam_p"], c / RAMAN["lam_s"]
f_ch = np.linspace(c / 1565e-9, c / 1530e-9, 40)
eff = np.interp((f_pump - f_ch) / 1e12, off_THz, g0_prof) / np.interp((f_pump - f_1550) / 1e12, off_THz, g0_prof)
lumped_check = {"channels": "40, equally spaced in frequency over 1530-1565 nm", "profile": "data/raman_ssmf_gnpy.csv",
                "offset_of_1550nm_from_pump_THz": float((f_pump - f_1550) / 1e12),
                "efficiency_relative_to_1550nm_mean": float(eff.mean()), "min": float(eff.min()), "max": float(eff.max())}

zv, Psv, stv, nodes_v = raman_bvp_no_depletion(Pp_R, dbm2w(16.0))
Pav, _ = raman_analytic(Pp_R, dbm2w(16.0), zv)
R["RAMAN"] = {
    "C_R_per_W_km": float(CR * 1e3), "L_eff_km": float(Leff / 1e3),
    "on_off_gain_undepleted_dB": float(db(np.exp(CR * Pp_R * Leff))),
    "span_loss_signal_dB": float(RAMAN["alpha_s"] * Lspan * 10 / np.log(10)),
    "cases": {k: v["res"] for k, v in RAMAN_CASES.items()},
    "lumped_load_check": lumped_check,
    "launch_power_sweep": sweep,
    "code_verification_bvp_without_depletion": dict(metrics(w2dbm(Psv) - w2dbm(Pav), ZGRID), bvp_status=int(stv),
                                                    final_mesh_nodes=int(nodes_v), launch_dBm=16.0,
                                                    initial_guess="flat (constant powers)"),
    "linearity_check": {"output_error_-40dBm_dB": float(e40), "output_error_-30dBm_dB": float(e30),
                        "ratio": float(e30 / e40),
                        "implied_numerical_offset_dB": float((10 * e40 - e30) / 9),
                        "note": "the deviation is proportional to the launch power at low power (first-order pump "
                                "depletion), so it is model-form; any numerical offset of the BVP is bounded by the "
                                "implied offset"},
}

# =====================================================================================================
# Figure
# =====================================================================================================
style.apply()
style.math_in_text_font()          # math minus signs in Arial, like the rest of the text
C, INK, INK2 = style.C, style.INK, style.INK2
fig, axs = plt.subplots(3, 2, figsize=(7.1, 7.0))
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.065, top=0.965, wspace=0.30, hspace=0.50)

# ---- a) EDFA along the fiber -------------------------------------------------------------------------------
ax = axs[0, 0]
z, Pp_z, Ps_z = edfa_ode(Pp0, Ps0, Lk[-1])
ax.plot(z, w2dbm(Ps_z), color=C[0], lw=1.5)
ax.plot(z, w2dbm(Pp_z), color=C[1], lw=1.5)
ax.plot(Lk, G_eq5 + E["signal_dBm"], "o", ms=4.2, mfc="white", mec=C[0], mew=1.1, zorder=3)
ax.plot(Lk, w2dbm(Pp_eq5), "o", ms=4.2, mfc="white", mec=C[1], mew=1.1, zorder=3)
ax.text(0.8, 12.5, "Signal gain\n" + " / ".join(f"{G_eq5[i]:.1f}" for i in (9, 14, 19)) + " dB\nat 10 / 15 / 20 m",
        ha="left", va="top", fontsize=7, color=INK2, linespacing=1.2)
ax.set_xlabel("Er-doped fiber length [m]")
ax.set_ylabel("Power [dBm]")
ax.set_xlim(0, 20.5)
ax.set_ylim(-24, 24)
ax.legend(handles=[Line2D([], [], color=C[0], lw=1.5, label="Signal, 1550 nm"),
                   Line2D([], [], color=C[1], lw=1.5, label="Pump, 1480 nm"),
                   Line2D([], [], color=INK, lw=1.2, label="Numerical ODE"),
                   Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=INK, mew=1.1,
                          label="Saleh closed form")],
          loc="lower right", fontsize=7, handlelength=1.8, borderaxespad=0.2)
style.label(ax, "a")

# ---- b) residual versus tolerance --------------------------------------------------------------------------
ax = axs[0, 1]
for k, (method, lab, mk) in enumerate((("LSODA", "LSODA", "o"), ("RK45", "RK45", "s"), ("Radau", "Radau", "^"))):
    rows = tol_study[method]
    ax.loglog([r["rtol"] for r in rows], [r["rmse_dB"] for r in rows], marker=mk, ms=3.6, lw=1.3, color=C[k], label=lab)
dflt = R["EDFA"]["residual_default_setting"]["rmse_dB"]
ax.loglog([1e-10], [dflt], marker="*", ms=10, mfc="none", mec=INK, mew=1.0, ls="none", label="Setting used in this work")
ax.set_xlabel("Relative tolerance of the integrator")
ax.set_ylabel("Residual RMSE [dB]")
ax.set_xlim(3e-13, 3e-4)
ax.set_ylim(1e-14, 1e-2)
ax.set_yticks(10.0 ** np.arange(-14, -1, 2))
ax.set_xticks(10.0 ** np.arange(-12, -3, 2))
ax.minorticks_off()
ax.legend(loc="upper left", fontsize=7, handlelength=1.8, borderaxespad=0.2)
ax.annotate("precision of the closed-form evaluation", xy=(4e-12, 2.0e-13), xytext=(1.5e-11, 2.5e-14), fontsize=7,
            color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
style.label(ax, "b")

# ---- c) SOA gain -------------------------------------------------------------------------------------------
ax = axs[1, 0]
for k, key in enumerate(SOA_CURVES):
    cv = SOA_CURVES[key]
    ax.plot(Pin_dbm, cv["full"], color=C[k], lw=1.5)
    ax.plot(Pin_dbm, cv["lin"], color=C[k], lw=1.2, ls=(0, (5, 1.5, 1, 1.5)))
    ax.plot(Pin_dbm, cv["cf"], color=C[k], lw=1.3, ls=(0, (3.5, 2)))
ax.set_xlabel("Input power [dBm]")
ax.set_ylabel("Chip gain [dB]")
ax.set_xlim(-41, 6)
ax.set_ylim(0, 30)
ax.legend(handles=[Line2D([], [], color=C[0], lw=1.5, label="120 mA"),
                   Line2D([], [], color=C[1], lw=1.5, label="150 mA"),
                   Line2D([], [], color=INK, lw=1.5, label="Numerical model"),
                   Line2D([], [], color=INK, lw=1.2, ls=(0, (5, 1.5, 1, 1.5)), label="Intermediate model"),
                   Line2D([], [], color=INK, lw=1.3, ls=(0, (3.5, 2)), label="Closed form, calibrated")],
          loc="lower left", fontsize=7, handlelength=2.6, borderaxespad=0.2, ncol=1)
style.label(ax, "c")

# ---- d) SOA deviation versus input power -------------------------------------------------------------------
ax = axs[1, 1]
ax.axvspan(-41, -20, color="#efeeea", lw=0, zorder=0)
i0 = int(np.where(Pin_dbm == 0)[0][0])
for k, key in enumerate(SOA_CURVES):
    cv = SOA_CURVES[key]
    ax.plot(Pin_dbm, cv["tot"], color=C[k], lw=1.5)
    ax.plot(Pin_dbm, cv["loss"], color=C[k], lw=1.2, ls=(0, (3.5, 2)))
    ax.plot(Pin_dbm, cv["rec"], color=C[k], lw=1.2, ls=(0, (1, 1.5)))
    ax.plot([0], [cv["tot"][i0]], "o", ms=4, color=C[k], zorder=4)
t120, t150 = SOA_CURVES["120 mA"]["tot"][i0], SOA_CURVES["150 mA"]["tot"][i0]
ax.text(-1.3, t150 + 0.22, f"{t150:.2f}", ha="right", va="bottom", fontsize=7, color=INK)
ax.text(0.9, t120 - 0.42, f"{t120:.2f}", ha="left", va="top", fontsize=7, color=INK)
r120 = SOA_RES["120 mA"]["deviation"]["total (closed form - numerical)"]["-40 to -20 dBm"]["rmse_dB"]
r150 = SOA_RES["150 mA"]["deviation"]["total (closed form - numerical)"]["-40 to -20 dBm"]["rmse_dB"]
ax.text(-30.5, 0.95, f"RMSE up to $-$20 dBm:\n{r120:.2f} and {r150:.2f} dB", ha="center", va="center", fontsize=7,
        color=INK2, linespacing=1.2)
ax.set_xlabel("Input power [dBm]")
ax.set_ylabel("Closed form $-$ numerical [dB]")
ax.set_xlim(-41, 6)
ax.set_ylim(-0.1, 3.8)
ax.legend(handles=[Line2D([], [], color=C[0], lw=1.5, label="120 mA"),
                   Line2D([], [], color=C[1], lw=1.5, label="150 mA"),
                   Line2D([], [], color=INK, lw=1.5, label="Total"),
                   Line2D([], [], color=INK, lw=1.2, ls=(0, (3.5, 2)), label="Internal-loss part"),
                   Line2D([], [], color=INK, lw=1.2, ls=(0, (1, 1.5)), label="Recombination part")],
          loc="upper left", fontsize=7, handlelength=2.4, borderaxespad=0.2)
style.label(ax, "d")

# ---- e) Raman along the span -------------------------------------------------------------------------------
ax = axs[2, 0]
cw, cl = RAMAN_CASES["single weak channel"], RAMAN_CASES["lumped load"]
zkm = cw["z"] / 1e3
ax.plot(zkm, -RAMAN["alpha_s"] * cw["z"] * 10 / np.log(10), ":", color=INK2, lw=1.1, label="Unpumped fiber")
ax.plot(zkm, cw["rel_bvp"], color=C[0], lw=1.5, label="BVP, $-$10 dBm signal")
ax.plot(zkm, cl["rel_bvp"], color=C[1], lw=1.5, label="BVP, +16 dBm lumped load")
ax.plot(zkm, cw["rel_cf"], color=INK, lw=1.3, ls=(0, (3.5, 2)), label="Undepleted closed form")
ax.set_xlabel("Position along the span [km]")
ax.set_ylabel("Signal power relative\nto launch [dB]")
ax.set_xlim(0, 51)
ax.set_ylim(-10.5, 6.5)
ax.legend(loc="lower left", fontsize=7, handlelength=2.4, borderaxespad=0.2)
style.label(ax, "e")

# ---- f) Raman deviation versus launch power ----------------------------------------------------------------
ax = axs[2, 1]
ax.semilogy(P_sweep, out_err, color=INK, lw=1.5, label="Error at the span output")
ax.semilogy(P_sweep, rms_err, color=INK2, lw=1.3, ls=(0, (3.5, 2)), label="RMSE along the span")
for k, (name, x) in enumerate(Rm["loads"].items()):
    e = RAMAN_CASES[name]["res"]["span_output_error_dB"]
    ax.plot([x], [e], "o", ms=5, color=C[k], zorder=4)
    ax.text(x + (1.2 if k == 0 else -1.2), e * (0.55 if k == 0 else 1.0), f"{e:.3f} dB" if e < 0.1 else f"{e:.2f} dB",
            ha="left" if k == 0 else "right", va="top" if k == 0 else "bottom", fontsize=7, color=INK)
ax.set_xlabel("Signal launch power [dBm]")
ax.set_ylabel("Closed form $-$ BVP [dB]")
ax.set_xlim(-41, 21)
ax.set_ylim(1e-6, 30)
ax.set_yticks(10.0 ** np.arange(-6, 2))
ax.minorticks_off()
ax.legend(loc="upper left", fontsize=7, handlelength=2.4, borderaxespad=0.2)
style.label(ax, "f")

style.save(fig, NAME, outdir=FIGDIR)
R["runtime_s"] = round(time.perf_counter() - T_START, 1)
with open(os.path.join(FIGDIR, NAME + "_results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=True)

# short console summary
print(f"EDFA Eq.5 gain 10/15/20 m: {G_eq5[9]:.2f} / {G_eq5[14]:.2f} / {G_eq5[19]:.2f} dB; default-setting residual "
      f"RMSE {dflt:.2e} dB")
for key, v in SOA_RES.items():
    t = v["deviation"]["total (closed form - numerical)"]
    print(f"SOA {key}: RMSE <=-20 dBm {t['-40 to -20 dBm']['rmse_dB']:.3f}, all {t['-40 to +5 dBm']['rmse_dB']:.3f} dB, "
          f"at 0 dBm {t['at_input_dBm']['+0']:.2f} dB, loss share at 0 dBm {v['share_of_total_at_0dBm']['internal loss']:.2f}")
for k, v in RAMAN_CASES.items():
    print(f"Raman {k}: RMSE {v['res']['rmse_dB']:.4f}, max {v['res']['max_abs_dB']:.4f}, bias {v['res']['bias_dB']:+.4f}, "
          f"output {v['res']['span_output_error_dB']:+.4f} dB")
print(f"written: {os.path.join(FIGDIR, NAME)}.png/.pdf and _results.json  ({R['runtime_s']} s)")
