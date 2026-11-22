"""Hybrid distributed-Raman + lumped-EDFA span: design model (layer L3) and its full-model check (layer L2).

Revision 3. Replaces the illustrative model of the submitted version (analytic Raman line shape, illustrative Er
line shapes at a fixed inversion of 0.62, EDFA gain plus Raman on-off gain reported as 'net gain').

Reference planes (net gain as in ITU-T G.665): plane A = input of the transmission fiber, plane B = EDFA output.
    G_net(lambda) = G_on-off,Raman(lambda) - alpha_s L_span - L_passive + G_EDFA(lambda)          [dB]
Ripple = maximum multichannel gain variation (ITU-T G.661): max_i G_net,i - min_i G_net,i over the channels.

Parameters come only from the cards (cards.py):
    RAMAN-G652      C_R (1450 nm pump), alpha_s, alpha_p
    EDFA-PERIN2019  EDF length L, alpha(lambda), g*(lambda) (digitized Fig. 7 of ref. [124]), zeta, alpha(980 nm)
    HYBRID-C80      span length, passive loss, channel plan, pump grid, pump and on-off gain limits, inversion bounds,
                    Raman scaling rule, Rayleigh coefficient for the DRB estimate (cards/hybrid_c80.json)
Data: measured SSMF Raman gain-efficiency profile distributed with GNPy (cards.raman_profile_ssmf).

Models
  first-order design model (design_lp): Raman on-off gains of the pumps add in dB (undepleted pumps with attenuation,
      no pump-pump or signal-signal Raman interaction); EDFA gain G_EDFA = 4.343 L [(alpha + g*) n - alpha], which is
      linear in the length-averaged inversion n, so n is an LP variable. Objective: minimum true peak-to-peak ripple
      (variables u, l with l <= G_net,i <= u) at a fixed mean net gain (equality constraint).
  full Raman model (raman_full): all channels and pumps, counter-propagating pumps, pump depletion, pump-pump,
      pump-signal and signal-signal stimulated Raman scattering with the photon-number factor nu_i/nu_k, attenuation;
      two-point boundary-value problem solved with scipy.integrate.solve_bvp (evaluate_full gives the model-form
      error of a first-order design).
  full model in the loop: design_corrected (first-order LP with the full-model residual as a frozen offset; used as
      an initializer) and design_slp (sequential linear programming with a finite-difference Jacobian of the full
      model and a trust region; returns a local optimum of the full-model ripple at the fixed mean).
Not included anywhere: amplified spontaneous emission and noise figure, double Rayleigh scattering (estimated
separately with drb_crosstalk), pump RIN, temperature dependence, polarization effects, the spectral shape of a real
gain-flattening filter, EDFA spectral hole burning.
"""
import os
import numpy as np
from scipy.optimize import linprog
from scipy.integrate import solve_bvp, solve_ivp

import cards

C0 = 2.99792458e8            # m/s
H = 6.62607015e-34           # J s
DB = 10.0 * np.log10(np.e)   # dB per neper (4.3429)

# ---------------------------------------------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------------------------------------------
HYB_CARD = cards.load(os.path.join(cards.CARD_DIR, "hybrid_c80.json"))
HC = cards.values(HYB_CARD)
RC = cards.values(cards.load("RAMAN-G652"))
EC = cards.values(cards.load("EDFA-PERIN2019"))

L_SPAN = HC["L_span"]                                  # km
L_PASSIVE = HC["L_passive"]                            # dB
ALPHA_S_DB = RC["alpha_s"]                             # dB/km, signal attenuation (card value at 1550 nm)
ALPHA_P_DB = RC["alpha_p"]                             # dB/km, pump attenuation (card value at 1450 nm)
CR_CARD = RC["C_R"]                                    # 1/(W km), peak gain efficiency for a 1450 nm pump
F_CARD = C0 / RC["lam_p"] / 1e12                       # THz, pump frequency of the card value
L_EDF = EC["L"]                                        # m
ZETA = EC["zeta"]                                      # 1/(m s)
ALPHA_P980 = EC["alpha_p"]                             # 1/m
LAM_P980 = EC["lambda_p"]                              # nm
PUMP_GRID = np.array(HC["lam_pump_grid"], float)       # nm
P_PUMP_MAX = HC["P_pump_max"]                          # W
G_CAP = HC["G_onoff_cap"]                              # dB
NBAR_BOUNDS = (HC["nbar_min"], HC["nbar_max"])
R_RAYLEIGH = HC["r_rayleigh"]                          # 1/km


def channel_plan(n=None, lam_min=None, lam_max=None):
    """Channel wavelengths [nm], equally spaced in frequency, end points included (card HYBRID-C80)."""
    n = int(HC["n_ch"] if n is None else n)
    lam_min = HC["lam_ch_min"] if lam_min is None else lam_min
    lam_max = HC["lam_ch_max"] if lam_max is None else lam_max
    f = np.linspace(C0 / (lam_min * 1e-9), C0 / (lam_max * 1e-9), n)
    return C0 / f * 1e9


def thz(lam_nm):
    return C0 / (np.asarray(lam_nm, float) * 1e-9) / 1e12


# ---------------------------------------------------------------------------------------------------------------
# Raman gain efficiency from the measured SSMF profile
# ---------------------------------------------------------------------------------------------------------------
_OFF, _G0, _FREF = cards.raman_profile_ssmf()          # THz, 1/(W m), Hz
_G0MAX = _G0.max()
_SHAPE = _G0 / _G0MAX
F_REF_GNPY = _FREF / 1e12                              # THz (1454 nm)
A_REF_GNPY = 75.74659443542413                         # um^2, reference effective area of the GNPy profile (data file header)
A_CARD = RC["Aeff"] * 1e12                             # um^2


def raman_shape_measured(df_thz):
    """Measured SSMF Raman gain-efficiency profile normalized to its maximum (4.195e-4 1/(W m) at 12.75 THz)."""
    d = np.asarray(df_thz, float)
    return np.where(d > 0, np.interp(d, _OFF, _SHAPE, left=0.0, right=0.0), 0.0)


def cr(df_thz, fp_thz, scaling="card"):
    """Raman gain efficiency C_R [1/(W km)] for the Stokes wave at offset df = f_pump - f_Stokes > 0 (THz).

    scaling = 'card'       : C_R[RAMAN-G652] (f_p / f_card) g0(df)/max(g0)       (choice of revision 3)
              'gnpy_area'  : g0(df) (f_p / f_ref) (A_ref / A_card), A_ref = 75.75 um^2 (alternative, sensitivity only)
    The wavelength dependence of the effective area is neglected (card A_eff = 80 um^2 for 1450-1550 nm).
    """
    shape = raman_shape_measured(df_thz)
    fp = np.asarray(fp_thz, float)
    if scaling == "card":
        return CR_CARD * (fp / F_CARD) * shape
    if scaling == "gnpy_area":
        return _G0MAX * 1e3 * (fp / F_REF_GNPY) * (A_REF_GNPY / A_CARD) * shape
    raise ValueError(scaling)


def cr_scaling_check():
    """Peak efficiency at the GNPy reference pump (1454 nm) for the chosen and two alternative scalings."""
    gnpy = _G0MAX * 1e3
    chosen = float(cr(12.75, F_REF_GNPY))
    alt_area = float(cr(12.75, F_REF_GNPY, "gnpy_area"))
    # g_R,GNPy / A_card: GNPy stores g_R = g0 * A_overlap; A_overlap = 79.37 um^2 at 12.75 THz (gnpy/core/parameters.py)
    alt_overlap = gnpy * 79.37 / A_CARD
    return dict(gnpy_profile_peak=gnpy, chosen=chosen, chosen_vs_gnpy=chosen / gnpy - 1,
                alt_gnpy_area=alt_area, alt_gnpy_area_vs_chosen=alt_area / chosen - 1,
                alt_gR_over_Acard=alt_overlap, alt_gR_over_Acard_vs_chosen=alt_overlap / chosen - 1)


def coupling_matrix(f_thz, scaling="card"):
    """K[i, k] [1/(W km)]: d ln P_i / dz contribution per watt of wave k (along the direction of wave i).
    f_k > f_i: gain C_R(f_k - f_i; f_k);  f_k < f_i: depletion -(f_i/f_k) C_R(f_i - f_k; f_i) (photon-number factor)."""
    f = np.asarray(f_thz, float)
    df = f[None, :] - f[:, None]                       # f_k - f_i
    gain = cr(np.abs(df), np.maximum(f[None, :], f[:, None]), scaling)
    K = np.where(df > 0, gain, -(f[:, None] / f[None, :]) * gain)
    np.fill_diagonal(K, 0.0)
    return K


def l_eff(alpha_db_km=ALPHA_P_DB, L=L_SPAN):
    a = alpha_db_km / DB
    return (1.0 - np.exp(-a * L)) / a


def onoff_matrix(lam_ch, lam_p, scaling="card", L=L_SPAN):
    """First-order on-off gain matrix A [dB/W]: G_on-off,i = sum_j A_ij P_j (undepleted counter-propagating pumps)."""
    fs, fp = thz(lam_ch), thz(lam_p)
    return DB * cr(fp[None, :] - fs[:, None], fp[None, :], scaling) * l_eff(ALPHA_P_DB, L)


# ---------------------------------------------------------------------------------------------------------------
# EDFA: average-inversion (Giles) relation with the card of ref. [124]
# ---------------------------------------------------------------------------------------------------------------
_LAM_E, _ALPHA_E, _GSTAR_E = cards.perin2019_coefficients()


def edfa_coeffs(lam_nm):
    lam = np.asarray(lam_nm, float)
    if lam.min() < _LAM_E.min() or lam.max() > _LAM_E.max():
        raise ValueError("wavelength outside the digitized range of card EDFA-PERIN2019")
    return np.interp(lam, _LAM_E, _ALPHA_E), np.interp(lam, _LAM_E, _GSTAR_E)


def edfa_gain_db(lam_nm, nbar, L=L_EDF):
    """Internal EDF gain [dB] for a length-averaged inversion nbar: 4.343 L [(alpha + g*) nbar - alpha]."""
    a, g = edfa_coeffs(lam_nm)
    return DB * L * ((a + g) * nbar - a)


def edfa_pump_980(nbar, lam_ch, p_in_w, L=L_EDF):
    """980 nm pump power [W] that holds the mean inversion nbar for the given EDF input signal powers.
    ASE-free two-level model (photon-number balance of Saleh et al.): Q_p (1 - G_p) = zeta L nbar + sum Q_s (G_s - 1).
    ASE, excited-state absorption and background loss are neglected, so the value is a lower bound."""
    a, g = edfa_coeffs(lam_ch)
    gs = np.exp(L * ((a + g) * nbar - a))
    q_in = np.asarray(p_in_w) / (H * C0 / (np.asarray(lam_ch) * 1e-9))
    gp = np.exp(-ALPHA_P980 * L * (1.0 - nbar))
    qp = (ZETA * L * nbar + np.sum(q_in * (gs - 1.0))) / (1.0 - gp)
    return qp * H * C0 / (LAM_P980 * 1e-9)


# ---------------------------------------------------------------------------------------------------------------
# Linear-programming design (layer L3)
# ---------------------------------------------------------------------------------------------------------------
def fixed_loss_db(L=L_SPAN, l_passive=L_PASSIVE):
    return ALPHA_S_DB * L + l_passive


def nbar_for_raman_share(lam_ch, mean_onoff_db, target=None, L=L_SPAN, l_passive=L_PASSIVE):
    """Mean inversion at which the Raman section must supply a given mean on-off gain to reach the target."""
    target = HC["G_target"] if target is None else target
    a, g = edfa_coeffs(lam_ch)
    need = target + fixed_loss_db(L, l_passive) - mean_onoff_db           # mean EDFA gain [dB]
    return float((need / (DB * L_EDF) + a.mean()) / (a + g).mean())


def design_lp(lam_ch, budget, lam_p=None, target=None, nbar=None, nbar_bounds=None, p_max=None, g_cap=None,
              offset=None, L=L_SPAN, l_passive=L_PASSIVE, scaling="card", tiebreak=True, tol_ripple=1e-4,
              l_edf=None):
    """Minimum true peak-to-peak ripple of G_net at a fixed mean G_net.

    Variables x = [P_1..P_M (W), nbar, u, l].  min u - l  s.t.
        l <= G_net,i <= u  (all channels),   mean_i G_net,i = target,   sum P <= budget,
        0 <= P_j <= p_max,   sum_j A_ij P_j <= g_cap,   nbar in nbar_bounds (or fixed).
    offset: optional per-channel correction [dB] added to G_net (used by design_corrected).
    tiebreak: among solutions within tol_ripple of the minimum ripple, take the one with the least total pump power.
    l_edf: EDF length [m] (default: card value); scaling alpha and g* by s is equivalent to l_edf = s L_EDF.
    Returns None if infeasible (e.g. budget too small to reach the mean)."""
    lam_p = PUMP_GRID if lam_p is None else np.asarray(lam_p, float)
    target = HC["G_target"] if target is None else target
    p_max = P_PUMP_MAX if p_max is None else p_max
    g_cap = G_CAP if g_cap is None else g_cap
    nbar_bounds = NBAR_BOUNDS if nbar_bounds is None else nbar_bounds
    l_edf = L_EDF if l_edf is None else l_edf
    A = onoff_matrix(lam_ch, lam_p, scaling, L)
    a, g = edfa_coeffs(lam_ch)
    B, Cc = DB * l_edf * (a + g), DB * l_edf * a
    D = fixed_loss_db(L, l_passive)
    off = np.zeros(len(lam_ch)) if offset is None else np.asarray(offset, float)
    n, m = len(lam_ch), len(lam_p)
    # G_net,i = A_i P + B_i nbar - C_i - D + off_i
    nv = m + 3
    iu, il, inb = m + 1, m + 2, m
    rows, rhs = [], []
    for i in range(n):
        r = np.zeros(nv); r[:m] = A[i]; r[inb] = B[i]; r[iu] = -1.0
        rows.append(r); rhs.append(Cc[i] + D - off[i])                       # G_net,i - u <= 0
        r = np.zeros(nv); r[:m] = -A[i]; r[inb] = -B[i]; r[il] = 1.0
        rows.append(r); rhs.append(-(Cc[i] + D - off[i]))                    # l - G_net,i <= 0
        r = np.zeros(nv); r[:m] = A[i]
        rows.append(r); rhs.append(g_cap)                                    # on-off cap
    r = np.zeros(nv); r[:m] = 1.0
    rows.append(r); rhs.append(budget)                                       # total budget
    aeq = np.zeros((1, nv)); aeq[0, :m] = A.mean(0); aeq[0, inb] = B.mean()
    beq = [target + Cc.mean() + D - off.mean()]
    nb = (nbar, nbar) if nbar is not None else tuple(nbar_bounds)
    bounds = [(0.0, p_max)] * m + [nb, (None, None), (None, None)]
    c = np.zeros(nv); c[iu], c[il] = 1.0, -1.0
    A_ub, b_ub = np.array(rows), np.array(rhs)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=aeq, b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    ripple_min = res.fun
    x = res.x
    if tiebreak:
        c2 = np.zeros(nv); c2[:m] = 1.0
        r = np.zeros(nv); r[iu], r[il] = 1.0, -1.0
        res2 = linprog(c2, A_ub=np.vstack([A_ub, r]), b_ub=np.r_[b_ub, ripple_min + tol_ripple], A_eq=aeq, b_eq=beq,
                       bounds=bounds, method="highs")
        if res2.status == 0:
            x = res2.x
    P, nb_opt = x[:m], x[inb]
    g_on = A @ P
    g_ed = B * nb_opt - Cc
    g_net = g_on - D + g_ed + off
    return dict(lam_ch=np.asarray(lam_ch), lam_p=lam_p, P=P, P_total=float(P.sum()), nbar=float(nb_opt),
                G_onoff=g_on, G_edfa=g_ed, G_net=g_net, ripple=float(g_net.max() - g_net.min()),
                ripple_lp=float(ripple_min), mean_net=float(g_net.mean()), loss_db=D, A=A, offset=off,
                budget=budget, target=target)


# ---------------------------------------------------------------------------------------------------------------
# Full multi-wavelength counter-pumped Raman model (layer L2)
# ---------------------------------------------------------------------------------------------------------------
def raman_full(lam_ch, p_ch_w, lam_p, p_p_w, L=L_SPAN, terms=("dep", "pp", "ss"), scaling="card",
               n_nodes=321, tol=1e-7, p_min=1e-6):
    """Signals launched at z = 0 (forward), pumps launched at z = L (backward). Solves for ln P of every wave.
    terms: 'dep' signal-induced pump depletion, 'pp' pump-pump SRS, 'ss' signal-signal SRS. The pump-to-signal gain
    term is always present. Returns powers along z and the solver status; pumps below p_min are omitted."""
    lam_ch = np.asarray(lam_ch, float)
    p_ch_w = np.broadcast_to(np.asarray(p_ch_w, float), lam_ch.shape).copy()
    keep = np.asarray(p_p_w) > p_min
    lam_p, p_p_w = np.asarray(lam_p, float)[keep], np.asarray(p_p_w, float)[keep]
    ns, npump = len(lam_ch), len(lam_p)
    if npump == 0:                                      # no pump above p_min: forward initial-value problem
        zz = np.linspace(0.0, L, 801)
        K = coupling_matrix(thz(lam_ch), scaling) if "ss" in terms else np.zeros((ns, ns))
        a_s = ALPHA_S_DB / DB
        sol = solve_ivp(lambda z, y: -a_s + K @ np.exp(y), (0.0, L), np.log(p_ch_w), t_eval=zz, rtol=1e-10,
                        atol=1e-12)
        pw = np.exp(sol.y)
        return dict(z=zz, P=pw, ns=ns, lam_ch=lam_ch, lam_p=lam_p, p_p=p_p_w, status=int(sol.status),
                    message=sol.message, nodes=zz.size, Ps_out=pw[:, -1], Pp_in=np.zeros(0), photon_balance_rel=None)
    f = np.r_[thz(lam_ch), thz(lam_p)]
    K = coupling_matrix(f, scaling)
    S, P = slice(0, ns), slice(ns, ns + npump)
    if "dep" not in terms:
        K[P, S] = 0.0
    if "pp" not in terms:
        K[P, P] = 0.0
    if "ss" not in terms:
        K[S, S] = 0.0
    a_s, a_p = ALPHA_S_DB / DB, ALPHA_P_DB / DB                   # 1/km
    sign = np.r_[np.ones(ns), -np.ones(npump)]
    alpha = np.r_[np.full(ns, a_s), np.full(npump, a_p)]

    def fun(z, y):
        pw = np.exp(y)
        return sign[:, None] * (-alpha[:, None] + K @ pw)

    def jac(z, y):
        pw = np.exp(y)
        return sign[:, None, None] * K[:, :, None] * pw[None, :, :]

    def bc(ya, yb):
        return np.r_[ya[S] - np.log(p_ch_w), yb[P] - np.log(p_p_w)]

    z = np.linspace(0.0, L, n_nodes)
    # initial guess: undepleted pumps, no signal-signal interaction
    leff_z = np.exp(-a_p * L) * (np.exp(a_p * z) - 1.0) / a_p
    y0 = np.zeros((ns + npump, z.size))
    y0[P] = np.log(p_p_w)[:, None] - a_p * (L - z)[None, :]
    y0[S] = np.log(p_ch_w)[:, None] - a_s * z[None, :] + (K[S, P] @ p_p_w)[:, None] * leff_z[None, :]
    sol = solve_bvp(fun, bc, z, y0, fun_jac=jac, tol=tol, max_nodes=20000)
    zz = np.linspace(0.0, L, 801)
    pw = np.exp(sol.sol(zz))
    out = dict(z=zz, P=pw, ns=ns, lam_ch=lam_ch, lam_p=lam_p, p_p=p_p_w, status=int(sol.status),
               message=sol.message, nodes=int(sol.x.size), Ps_out=pw[S, -1], Pp_in=pw[P, 0])
    # photon-number balance (exact for the complete model): d/dz(sum_fwd N - sum_bwd N) = -sum alpha N
    if all(t in terms for t in ("dep", "pp", "ss")):
        hnu = H * f * 1e12
        N = pw / hnu[:, None]
        flux = N[S].sum(0) - N[P].sum(0)
        loss = (alpha[:, None] * N).sum(0)
        lhs = flux[-1] - flux[0]
        rhs = -np.trapezoid(loss, zz)
        out["photon_balance_rel"] = float(abs(lhs - rhs) / N[P, -1].sum())
    return out


def signals_only(lam_ch, p_ch_w, L=L_SPAN, ss=True, scaling="card"):
    """Pumps off: forward propagation of the WDM load with attenuation and (optionally) signal-signal SRS."""
    lam_ch = np.asarray(lam_ch, float)
    p0 = np.broadcast_to(np.asarray(p_ch_w, float), lam_ch.shape).copy()
    K = coupling_matrix(thz(lam_ch), scaling) if ss else np.zeros((lam_ch.size, lam_ch.size))
    a_s = ALPHA_S_DB / DB
    sol = solve_ivp(lambda z, y: -a_s + K @ np.exp(y), (0.0, L), np.log(p0), rtol=1e-10, atol=1e-12)
    return np.exp(sol.y[:, -1])


def evaluate_full(design, p_ch_dbm=None, terms=("dep", "pp", "ss"), scaling="card", l_passive=L_PASSIVE):
    """Net gain, on-off gain and model-form error of a first-order design evaluated with the full model."""
    p_ch_dbm = HC["P_ch_in"] if p_ch_dbm is None else p_ch_dbm
    lam = design["lam_ch"]
    p_in = 1e-3 * 10 ** (np.broadcast_to(np.asarray(p_ch_dbm, float), lam.shape) / 10)
    fo = raman_full(lam, p_in, design["lam_p"], design["P"], terms=terms, scaling=scaling)
    p_off = signals_only(lam, p_in, ss=("ss" in terms), scaling=scaling)
    g_span_raman = 10 * np.log10(fo["Ps_out"] / p_in)                   # fiber section only, pumps on
    g_onoff = 10 * np.log10(fo["Ps_out"] / p_off)
    g_net = g_span_raman - l_passive + design["G_edfa"]
    err = g_net - (design["G_net"] - design["offset"])                  # full model minus pure first-order model
    return dict(sol=fo, G_onoff=g_onoff, G_net=g_net, ripple=float(g_net.max() - g_net.min()),
                mean_net=float(g_net.mean()), err=err, err_rms=float(np.sqrt(np.mean(err ** 2))),
                err_max=float(np.max(np.abs(err))), err_mean=float(err.mean()),
                err_pp=float(err.max() - err.min()), p_in=p_in, p_off=p_off)


def design_corrected(lam_ch, budget, p_ch_dbm=None, relax=0.6, n_iter=30, tol_db=2e-3, **kw):
    """Successive linear programming (layer L2 feeding back into layer L3).

    The first-order LP is re-solved with a per-channel offset equal to the full-model residual
    (full minus first-order net gain) at the previous pump setting, under-relaxed with factor 'relax'.
    Converged when the residual at the current pumps differs from the offset used by less than tol_db; the LP
    prediction then equals the full-model net gain, and its ripple is an upper bound on the full-model minimum.
    Returns (design, full-model evaluation, history, converged); design is None if an LP becomes infeasible."""
    d = design_lp(lam_ch, budget, **kw)
    if d is None:
        return None, None, [], False
    off = np.zeros(len(lam_ch))
    hist = []
    for it in range(n_iter):
        ev = evaluate_full(d, p_ch_dbm)
        resid = ev["G_net"] - (d["G_net"] - d["offset"])
        change = float(np.max(np.abs(resid - off)))
        hist.append(dict(iter=it, P_mW=[round(float(x), 2) for x in d["P"] * 1e3], nbar=round(d["nbar"], 5),
                         ripple_full=round(ev["ripple"], 4), mean_full=round(ev["mean_net"], 4),
                         ripple_lp=round(d["ripple"], 4), change_dB=round(change, 5)))
        if change < tol_db:
            return d, ev, hist, True
        off = off + relax * (resid - off)
        d_new = design_lp(lam_ch, budget, offset=off, **kw)
        if d_new is None:
            return None, ev, hist, False
        d = d_new
    return d, evaluate_full(d, p_ch_dbm), hist, False


def first_order(lam_ch, lam_p, P, nbar, L=L_SPAN, l_passive=L_PASSIVE, scaling="card"):
    """First-order prediction (on-off gain, EDFA gain, span net gain) for any pump setting."""
    A = onoff_matrix(lam_ch, lam_p, scaling, L)
    g_on = A @ np.asarray(P, float)
    g_ed = edfa_gain_db(lam_ch, nbar)
    g_net = g_on - fixed_loss_db(L, l_passive) + g_ed
    return dict(A=A, G_onoff=g_on, G_edfa=g_ed, G_net=g_net, ripple=float(g_net.max() - g_net.min()),
                mean_net=float(g_net.mean()))


def design_slp(lam_ch, budget, starts, nbar=None, nbar_bounds=None, p_ch_dbm=None, target=None, p_max=None,
               g_cap=None, lam_p=None, mu=50.0, delta0=0.03, max_iter=40, h=2e-3, l_passive=L_PASSIVE, L=L_SPAN,
               tol_merit=1e-5, scaling="card", l_edf=None):
    """Minimum-ripple design with the full Raman model in the loop (sequential linear programming).

    Each iteration linearizes the full-model span gain G_span,i(P) around the current pumps with a forward-difference
    Jacobian (step h) and solves an LP for the step dP (trust region |dP_j| <= delta) and the EDFA inversion.
    Merit = peak-to-peak ripple of G_net + mu |mean G_net - target| (exact-penalty form of the mean constraint),
    always evaluated with the full model; steps that do not reduce the merit are rejected and delta is halved.
    starts: list of designs (dicts with 'P' and 'nbar'); the best one (after clipping to the budget) is the start.
    Returns a local optimum; a residual |mean - target| > 0.01 dB means the budget cannot reach the target."""
    lam_ch = np.asarray(lam_ch, float)
    lam_p = PUMP_GRID if lam_p is None else np.asarray(lam_p, float)
    target = HC["G_target"] if target is None else target
    p_max = P_PUMP_MAX if p_max is None else p_max
    g_cap = G_CAP if g_cap is None else g_cap
    nbar_bounds = NBAR_BOUNDS if nbar_bounds is None else nbar_bounds
    p_ch_dbm = HC["P_ch_in"] if p_ch_dbm is None else p_ch_dbm
    l_edf = L_EDF if l_edf is None else l_edf
    p_in = 1e-3 * 10 ** (np.broadcast_to(np.asarray(p_ch_dbm, float), lam_ch.shape) / 10)
    a, g = edfa_coeffs(lam_ch)
    B, Cc = DB * l_edf * (a + g), DB * l_edf * a
    g_off = 10 * np.log10(signals_only(lam_ch, p_in, L=L, scaling=scaling) / p_in)
    n, m = lam_ch.size, lam_p.size
    nfev = [0]

    def gspan(P):
        nfev[0] += 1
        return 10 * np.log10(raman_full(lam_ch, p_in, lam_p, P, L=L, scaling=scaling)["Ps_out"] / p_in)

    def merit(gs, nb):
        gn = gs - l_passive + B * nb - Cc
        return float(gn.max() - gn.min() + mu * abs(gn.mean() - target)), gn

    def best_nbar(gs):
        if nbar is not None:
            return nbar
        nb = (target - (gs - l_passive - Cc).mean()) / B.mean()
        return float(np.clip(nb, *nbar_bounds))

    best = None
    for s in starts:
        if s is None:
            continue
        P0 = np.clip(np.asarray(s["P"], float), 0.0, p_max)
        if P0.sum() > budget:
            P0 = P0 * budget / P0.sum()
        gs0 = gspan(P0)
        nb0 = best_nbar(gs0)
        m0, _ = merit(gs0, nb0)
        if best is None or m0 < best[3]:
            best = (P0, nb0, gs0, m0)
    P, nb, gs, mcur = best
    delta = delta0
    hist = []
    for it in range(max_iter):
        J = np.empty((n, m))
        for j in range(m):
            Pj = P.copy()
            Pj[j] += h
            J[:, j] = (gspan(Pj) - gs) / h
        # LP variables: [dP (m), nb, u, l, s]
        nv = m + 4
        inb, iu, il, isl = m, m + 1, m + 2, m + 3
        rows, rhs = [], []
        base = gs - l_passive - Cc
        for i in range(n):
            rr = np.zeros(nv); rr[:m] = J[i]; rr[inb] = B[i]; rr[iu] = -1.0
            rows.append(rr); rhs.append(-base[i])
            rr = np.zeros(nv); rr[:m] = -J[i]; rr[inb] = -B[i]; rr[il] = 1.0
            rows.append(rr); rhs.append(base[i])
            rr = np.zeros(nv); rr[:m] = J[i]
            rows.append(rr); rhs.append(g_cap - (gs[i] - g_off[i]))
        rr = np.zeros(nv); rr[:m] = J.mean(0); rr[inb] = B.mean(); rr[isl] = -1.0
        rows.append(rr); rhs.append(target - base.mean())
        rr = np.zeros(nv); rr[:m] = -J.mean(0); rr[inb] = -B.mean(); rr[isl] = -1.0
        rows.append(rr); rhs.append(base.mean() - target)
        rr = np.zeros(nv); rr[:m] = 1.0
        rows.append(rr); rhs.append(budget - P.sum())
        bnds = [(max(-delta, -P[j]), min(delta, p_max - P[j])) for j in range(m)]
        bnds += [(nbar, nbar) if nbar is not None else tuple(nbar_bounds), (None, None), (None, None), (0.0, None)]
        c = np.zeros(nv); c[iu], c[il], c[isl] = 1.0, -1.0, mu
        res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bnds, method="highs")
        if res.status != 0:
            delta *= 0.5
            hist.append(dict(iter=it, status="lp_failed", delta=delta))
            if delta < 2e-4:
                break
            continue
        dP, nb_new = res.x[:m], res.x[inb]
        m_pred = res.fun
        if mcur - m_pred < tol_merit:              # no improving direction within the trust region: stationary
            hist.append(dict(iter=it, merit=round(mcur, 5), merit_pred=round(float(m_pred), 5), stationary=True,
                             delta=delta))
            break
        P_new = np.clip(P + dP, 0.0, p_max)
        gs_new = gspan(P_new)
        m_new, _ = merit(gs_new, nb_new)
        accepted = m_new < mcur - 1e-7
        hist.append(dict(iter=it, merit=round(mcur, 5), merit_pred=round(float(m_pred), 5), merit_new=round(m_new, 5),
                         accepted=bool(accepted), delta=delta))
        if accepted:
            rho = (mcur - m_new) / max(mcur - m_pred, 1e-12)
            P, nb, gs, mcur = P_new, float(nb_new), gs_new, m_new
            if rho > 0.75 and np.max(np.abs(dP)) > 0.9 * delta:
                delta = min(2.0 * delta, 0.2)
            elif rho < 0.25:
                delta *= 0.5
        else:
            delta *= 0.5
        if delta < 2e-4:
            break
    _, gn = merit(gs, nb)
    return dict(lam_ch=lam_ch, lam_p=lam_p, P=P, P_total=float(P.sum()), nbar=float(nb), G_net=gn,
                G_onoff=gs - g_off, G_span=gs, G_edfa=B * nb - Cc, ripple=float(gn.max() - gn.min()),
                mean_net=float(gn.mean()), mean_dev=float(gn.mean() - target), feasible=bool(abs(gn.mean() - target) < 0.01),
                merit=mcur, iterations=len(hist), n_full_model_solves=nfev[0], history=hist, budget=budget, target=target)


def nsp_two_level(lam_nm, nbar):
    """Spontaneous-emission factor n_sp = n g* / (n g* - (1 - n) alpha) evaluated at the mean inversion
    (indicative only: the noise figure of a real EDFA depends on the inversion near its input)."""
    a, g = edfa_coeffs(lam_nm)
    den = nbar * g - (1.0 - nbar) * a
    return np.where(den > 0, nbar * g / np.where(den > 0, den, 1.0), np.inf)


# ---------------------------------------------------------------------------------------------------------------
# Double Rayleigh scattering estimate (not part of the design model)
# ---------------------------------------------------------------------------------------------------------------
def drb_crosstalk(z_km, g_lin, r=R_RAYLEIGH):
    """DRB crosstalk ratio R = r^2 int_0^L dz2 G(z2)^2 int_0^z2 G(z1)^-2 dz1, G(z) = net signal gain from 0 to z
    (same distributed gain assumed for the backscattered light). Returns R in dB."""
    z = np.asarray(z_km, float)
    g = np.asarray(g_lin, float)
    inner = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] ** -2 + g[:-1] ** -2) * np.diff(z))])
    R = r ** 2 * np.trapezoid(g ** 2 * inner, z)
    return float(10 * np.log10(R))
# Removed in revision 3: the analytic Raman line shape (raman_shape), the illustrative Er line shapes (er_xs) used at a
# fixed inversion of 0.62, and the minimax LP around a fixed 20 dB target (design) of the submitted version.
