"""Band-requirement and beyond-C-band tables of revision 3 (reviewer points R1-9, R2-B3, R2-A3 Raman part).

What this script computes (all inputs from card files in ../cards, loaded by path through cards.load):

 1. Edges and widths [THz] of the ITU-T O, E, S, C, L and U bands (card MULTIBAND-G652D-SPAN).
 2. Passive loss of an 80 km G.652.D span from measured attenuation anchors (stated interpolation).
 3. Inter-channel stimulated Raman scattering (ISRS) for a stated full-load scenario: the coupled power
    equations of all signal bins are integrated numerically with the measured SSMF Raman gain-efficiency
    profile distributed with GNPy (0-42 THz), scaled to each pump frequency and pump/Stokes overlap area.
    The triangular approximation of Semrau et al. (valid to about 15 THz) is used only as a check.
 4. The gain each band amplifier must supply (span loss minus ISRS power change) for band-alone and
    full loading, i.e. the gain range that follows the channel load.
 5. Raman fiber types (card RAMAN-FIBER-TYPES-REF): area-scaled efficiencies (labelled as derived) and
    undepleted on-off gains for a stated pump.
 6. Consistency test of the BDFA reference card (BDFA-FIRSTOV2020-REF): a single-center two-level reading
    of the fitted saturation data against the measured amplifier data.

Results: ../figures/Tables_bands_results.json; markdown tables are printed to stdout.
Run from this directory:  python tables_bands.py
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import cards

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

C0 = 299792458.0
H = 6.62607015e-34
DB = np.log(10.0) / 10.0            # dB -> neper
TEN_LOG_E = 10.0 / np.log(10.0)     # neper -> dB (4.343)
BANDS = ["O", "E", "S", "C", "L", "U"]

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
CARD_SPAN = os.path.join(cards.CARD_DIR, "multiband_g652d_span.json")
CARD_FIBERS = os.path.join(cards.CARD_DIR, "raman_fiber_types_reference.json")
CARD_BDFA = os.path.join(cards.CARD_DIR, "bdfa_firstov2020_reference.json")


def thz(lam_nm):
    return C0 / (np.asarray(lam_nm, float) * 1e-9) / 1e12


def nm(f_thz):
    return C0 / (np.asarray(f_thz, float) * 1e12) * 1e9


# ---------------------------------------------------------------------------------------------
# 1-2. Band geometry and attenuation
# ---------------------------------------------------------------------------------------------
def band_geometry(V):
    out = {}
    for b in BANDS:
        lo, hi = V["band_" + b]
        out[b] = {"lambda_nm": [lo, hi], "f_THz": [float(thz(hi)), float(thz(lo))],
                  "width_THz": float(thz(lo) - thz(hi))}
    return out


def attenuation(lam_nm, V, delta=0.0):
    """alpha(lambda) [dB/km]: piecewise linear through the card anchors, linear extrapolation outside."""
    x = np.asarray(V["att_lambda"], float)
    y = np.asarray(V["att_value"], float) + delta
    lam = np.atleast_1d(np.asarray(lam_nm, float))
    a = np.interp(lam, x, y)
    s_lo = (y[1] - y[0]) / (x[1] - x[0])
    s_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])
    a = np.where(lam < x[0], y[0] + s_lo * (lam - x[0]), a)
    a = np.where(lam > x[-1], y[-1] + s_hi * (lam - x[-1]), a)
    return a


# ---------------------------------------------------------------------------------------------
# 3. Raman gain efficiency between two frequencies
# ---------------------------------------------------------------------------------------------
class RamanEfficiency:
    """C(f_p -> f_s) [1/(W km)] for f_p > f_s (frequencies in THz).

    mode 'full'       : g0(df) (f_p/f_ref) A_ov(f_ref-df, f_ref)/A_ov(f_p, f_s)   (main case)
         'freq'       : g0(df) (f_p/f_ref)                       (no area scaling)
         'none'       : g0(df)                                   (profile as measured at 1454 nm)
         'trunc15'    : as 'full' but g0 = 0 beyond 15 THz       (role of the long-offset tail)
         'triangular' : C_r df for df <= 15 THz, 0 beyond        (Semrau et al. approximation)
         'tri_nocut'  : C_r df for any df                        (code check against the closed form)
    """

    def __init__(self, V, mode="full"):
        off, g0, fref = cards.raman_profile_ssmf()
        self.off = np.asarray(off, float)
        self.g0 = np.asarray(g0, float) * 1e3            # 1/(W m) -> 1/(W km)
        self.fref = fref / 1e12
        self.mode = mode
        self.Cr = V["Cr_slope_S"]
        a, n1 = V["core_radius"], V["n1"]
        self.pia2 = np.pi * a ** 2
        self.fA = V["f_ref_area"] / 1e12
        self.V_A = np.exp(self.pia2 / V["Aeff_1550"])     # normalized frequency at f_A (from A = pi a^2/ln V)
        self.n1 = n1

    def area(self, f_thz):
        """Effective area [m^2] of the Gaussian step-index approximation (GNPy FiberParams model)."""
        Vf = self.V_A * np.asarray(f_thz, float) / self.fA
        return self.pia2 / np.log(Vf)

    def a_ov(self, f1, f2):
        return 0.5 * (self.area(f1) + self.area(f2))

    def __call__(self, fp, fs):
        fp = np.asarray(fp, float)
        fs = np.asarray(fs, float)
        df = fp - fs
        if self.mode in ("triangular", "tri_nocut"):
            g = self.Cr * df
            if self.mode == "triangular":
                g = np.where(df <= 15.0, g, 0.0)
            return g
        g = np.interp(df, self.off, self.g0, left=0.0, right=0.0)
        if self.mode == "trunc15":
            g = np.where(df <= 15.0, g, 0.0)
        if self.mode in ("full", "trunc15"):
            return g * (fp / self.fref) * self.a_ov(self.fref - df, self.fref) / self.a_ov(fp, fs)
        if self.mode == "freq":
            return g * (fp / self.fref)
        return g


def coupling_matrix(f, eff, photon=True):
    """K[i, j] such that dP_i/dz = P_i (-alpha_i + sum_j K[i, j] P_j); f in THz, K in 1/(W km)."""
    Fi, Fj = np.meshgrid(f, f, indexing="ij")
    K = np.zeros_like(Fi)
    gain = Fj > Fi                                   # bin j pumps bin i
    K[gain] = eff(Fj[gain], Fi[gain])
    dep = Fj < Fi                                    # bin i pumps bin j and is depleted
    c = eff(Fi[dep], Fj[dep])
    K[dep] = -(Fi[dep] / Fj[dep]) * c if photon else -c
    return K


# ---------------------------------------------------------------------------------------------
# ISRS propagation
# ---------------------------------------------------------------------------------------------
def make_grid(V, load, df_bin):
    """load: {band: total power [dBm]} spread uniformly over each band, or
             {'contiguous': (lam_lo, lam_hi, P_tot_dBm)}."""
    f, P, tag = [], [], []
    if "contiguous" in load:
        lam_lo, lam_hi, ptot = load["contiguous"]
        segments = [("X", thz(lam_hi), thz(lam_lo), ptot)]
    else:
        segments = []
        for b in BANDS:
            if b in load:
                lo, hi = V["band_" + b]
                segments.append((b, thz(hi), thz(lo), load[b]))
    for b, f0, f1, pdbm in segments:
        n = max(1, int(round((f1 - f0) / df_bin)))
        fc = f0 + (np.arange(n) + 0.5) * (f1 - f0) / n
        f += list(fc)
        P += [1e-3 * 10 ** (pdbm / 10.0) / n] * n
        tag += [b] * n
    return np.array(f), np.array(P), np.array(tag)


def isrs_run(V, load, mode="full", L_km=None, df_bin=None, att_delta=0.0, photon=True, alpha_const=None):
    L_km = V["L_span"] if L_km is None else L_km
    df_bin = V["df_bin"] if df_bin is None else df_bin
    f, P0, tag = make_grid(V, load, df_bin)
    lam = nm(f)
    a_db = np.full_like(f, alpha_const) if alpha_const is not None else attenuation(lam, V, att_delta)
    alpha = a_db * DB                                                  # 1/km
    K = coupling_matrix(f, RamanEfficiency(V, mode), photon)
    sol = solve_ivp(lambda z, y: -alpha + K @ np.exp(y), (0.0, L_km), np.log(P0),
                    method="RK45", rtol=1e-10, atol=1e-10)
    if sol.status != 0:
        raise RuntimeError(sol.message)
    PL = np.exp(sol.y[:, -1])
    Ppass = P0 * np.exp(-alpha * L_km)
    d = 10 * np.log10(PL / Ppass)                                      # ISRS power change [dB]; + = gain
    res = {"f": f, "lam": lam, "tag": tag, "P0": P0, "PL": PL, "d_dB": d, "alpha_dB_km": a_db,
           "span_loss_dB": a_db * L_km, "L_km": L_km, "P_tot_dBm": float(10 * np.log10(P0.sum() / 1e-3))}
    res["bands"] = {}
    for b in np.unique(tag):
        m = tag == b
        res["bands"][str(b)] = {
            "mean_dB": float(10 * np.log10(PL[m].sum() / Ppass[m].sum())),
            "min_dB": float(d[m].min()), "max_dB": float(d[m].max()),
            "short_edge_dB": float(d[m][np.argmax(f[m])]), "long_edge_dB": float(d[m][np.argmin(f[m])]),
            "P_band_dBm": float(10 * np.log10(P0[m].sum() / 1e-3)), "n_bins": int(m.sum())}
    return res


def band_summary(res, digits=3):
    return {b: {k: (round(v, digits) if isinstance(v, float) else v) for k, v in s.items()}
            for b, s in res["bands"].items()}


# ---------------------------------------------------------------------------------------------
# 6. BDFA card: single-center two-level reading
# ---------------------------------------------------------------------------------------------
def _bdfa_gain(Pp_launch, Pin, g_s, xi, prm):
    """Backward-pumped two-level amplifier with unsaturable loss; returns (gain dB, residual pump W)."""
    L, a_p, a_s, l_p, l_s, nu_p, nu_s = (prm[k] for k in ("L", "a_p", "a_s", "l_p", "l_s", "nu_p", "nu_s"))
    a = np.array([a_p, a_s])
    g = np.array([0.0, g_s])
    nu = np.array([nu_p, nu_s])

    def rhs(z, y):
        P = np.exp(y)
        phi = P / (H * nu * xi)
        n2 = np.sum(phi * a) / (1.0 + np.sum(phi * (a + g)))
        # pump travels toward z = 0 (launched at z = L); g*(pump) = 0, so its net coefficient is a_p n2 - a_p - l_p
        return [-(a_p * n2 - a_p - l_p), (a_s + g_s) * n2 - a_s - l_s]

    def shoot(lnp0):
        s = solve_ivp(rhs, (0.0, L), [lnp0, np.log(Pin)], method="LSODA", rtol=1e-9, atol=1e-11)
        return s.y[0, -1] - np.log(Pp_launch), s.y[1, -1]

    x = brentq(lambda v: shoot(v)[0], np.log(Pp_launch) - 80.0, np.log(Pp_launch), xtol=1e-11)
    return 10 * np.log10(np.exp(shoot(x)[1]) / Pin), float(np.exp(x))


def bdfa_two_level_check(card):
    v = cards.values(card)
    lam_s = v["lambda_s_test"] * 1e-9
    prm = dict(L=v["L"], a_p=v["alpha0_1230"] * DB, a_s=v["alpha0_1310"] * DB,
               l_p=v["alphaBG_1230"] * DB, l_s=v["alphaBG_1310"] * DB,
               nu_p=C0 / (v["lambda_p"] * 1e-9), nu_s=C0 / lam_s)
    Psat_p, Psat_s = v["Psat_1230"] * 1e-3, v["Psat_1310"] * 1e-3
    Pin = 1e-3 * 10 ** (v["P_probe"] / 10)
    xi = Psat_p * prm["a_p"] / (H * prm["nu_p"])                 # g*(1230) = 0
    g_s = H * prm["nu_s"] * xi / Psat_s - prm["a_s"]
    out = {"assumptions": "single center, two levels, g*(1230 nm) = 0, backward pump, -25 dBm at 1310 nm, unsaturable loss explicit",
           "xi_per_m_s": float(xi), "gstar_1310_dB_m": float(g_s / DB),
           "full_inversion_bound_dB": float((g_s - prm["l_s"]) * prm["L"] / DB)}
    G250, _ = _bdfa_gain(v["P_p_launched"] * 1e-3, Pin, g_s, xi, prm)
    Pgrid = np.asarray(v["Pp_grid_test"], float) * 1e-3
    Gg = np.array([_bdfa_gain(p, Pin, g_s, xi, prm)[0] for p in Pgrid])
    out.update({"G_250mW_dB": float(G250), "max_gain_coeff_dB_per_mW": float(np.max(Gg / (Pgrid * 1e3))),
                "G_vs_Pp": {"Pp_mW": (Pgrid * 1e3).tolist(), "G_dB": np.round(Gg, 2).tolist()},
                "measured": {"G_250mW_dB": v["G_250mW"], "gain_coeff_dB_per_mW": v["gain_coefficient"],
                             "G_peak_dB_lower_bound": v["G_peak"]}})
    sens = {}
    for fct in (0.7, 1.3):
        gs = H * prm["nu_s"] * xi / (Psat_s * fct) - prm["a_s"]
        sens[f"Psat_1310_x{fct}"] = float(gs / DB)
    out["gstar_1310_sensitivity_dB_m"] = sens
    # pair chosen in the analysis phase to hit the two anchors (read from the card, tested here)
    tp = card["consistency_tests"]["tested_pair"]
    xi_c, g_c = tp["xi_per_m_s"], tp["gstar_dB_per_m"] * DB
    Gc = np.array([_bdfa_gain(p, Pin, g_c, xi_c, prm)[0] for p in Pgrid])
    out["tested_pair"] = {
        "xi_per_m_s": xi_c, "gstar_dB_m": tp["gstar_dB_per_m"],
        "G_250mW_dB": float(_bdfa_gain(v["P_p_launched"] * 1e-3, Pin, g_c, xi_c, prm)[0]),
        "max_gain_coeff_dB_per_mW": float(np.max(Gc / (Pgrid * 1e3))),
        "implied_Psat_1310_mW": float(H * prm["nu_s"] * xi_c / (prm["a_s"] + g_c) * 1e3),
        "implied_Psat_1230_mW": float(H * prm["nu_p"] * xi_c / prm["a_p"] * 1e3),
        "gain_cap_dB": float((g_c - prm["l_s"]) * prm["L"] / DB)}
    # V2: the residuals written into the card must match this computation
    rec = card["consistency_tests"]["recorded"]
    now = {"xi_per_m_s": out["xi_per_m_s"], "gstar_1310_dB_m": out["gstar_1310_dB_m"],
           "full_inversion_bound_dB": out["full_inversion_bound_dB"], "G_250mW_dB": out["G_250mW_dB"],
           "max_gain_coeff_dB_per_mW": out["max_gain_coeff_dB_per_mW"],
           "tested_pair_G_250mW_dB": out["tested_pair"]["G_250mW_dB"],
           "tested_pair_max_gain_coeff_dB_per_mW": out["tested_pair"]["max_gain_coeff_dB_per_mW"],
           "tested_pair_implied_Psat_1310_mW": out["tested_pair"]["implied_Psat_1310_mW"],
           "tested_pair_implied_Psat_1230_mW": out["tested_pair"]["implied_Psat_1230_mW"],
           "tested_pair_gain_cap_dB": out["tested_pair"]["gain_cap_dB"]}
    bad = {k: (rec[k], now[k]) for k in rec if abs(now[k] - rec[k]) > 0.01 * abs(rec[k]) + 1e-12}
    if bad:
        raise RuntimeError(f"BDFA card 'recorded' values disagree with the computation: {bad}")
    out["card_record_check"] = "recorded values of the card reproduced within 1 %"
    return out


# ---------------------------------------------------------------------------------------------
# 5. Raman fiber types
# ---------------------------------------------------------------------------------------------
def raman_fiber_types(Fc, V):
    F = cards.values(Fc)
    P_pump, L_km = F["onoff_pump_W"], F["onoff_span_km"]
    ref_cr, ref_A = F["ssmf_CR_peak"], F["ssmf_Aeff"]

    def g_onoff(cr, a_p_db):
        ap = a_p_db * DB
        leff = (1 - np.exp(-ap * L_km)) / ap
        return float(TEN_LOG_E * cr * P_pump * leff), float(leff)

    rows = {}
    g, le = g_onoff(ref_cr, F["ssmf_alpha_p"])
    rows["G.652.D SSMF"] = {"Aeff_um2": ref_A, "CR_traced": ref_cr, "CR_traced_note": "GNPy profile, 1454 nm pump",
                            "CR_typ_other": F["ssmf_CR_typ_OFS"], "CR_area_scaled": None,
                            "alpha_s": F["ssmf_alpha_s"], "alpha_p": F["ssmf_alpha_p"],
                            "G_onoff_dB": g, "Leff_p_km": le}
    cr_s = ref_cr * ref_A / F["nzdf_la_Aeff"]
    rows["G.655 large-area NZDF"] = {"Aeff_um2": F["nzdf_la_Aeff"], "CR_traced": F["nzdf_la_CR_typ"],
                                     "CR_traced_note": "typical value, OFS white paper Fig. 3",
                                     "CR_area_scaled": cr_s, "dev_scaled_vs_traced": cr_s / F["nzdf_la_CR_typ"] - 1,
                                     "G_onoff_dB": None}
    cr_s = ref_cr * ref_A / F["twr_Aeff"]
    g, le = g_onoff(F["twr_CR"], F["twr_alpha"]["1450"][1])
    rows["G.655 small-area NZDF (TrueWave REACH)"] = {
        "Aeff_um2": F["twr_Aeff"], "CR_traced": F["twr_CR"], "CR_traced_note": "production average, 1453 nm pump",
        "CR_area_scaled": cr_s, "dev_scaled_vs_traced": cr_s / F["twr_CR"] - 1,
        "alpha_s": F["twr_alpha"]["1550"][1], "alpha_p": F["twr_alpha"]["1450"][1], "G_onoff_dB": g, "Leff_p_km": le,
        "pump_ratio_check": {"P_SSMF_over_P_TWR_measured": F["twr_pump_ratio_OFS"][1] / F["twr_pump_ratio_OFS"][0],
                             "CR_ratio": F["twr_CR"] / ref_cr}}
    cr_s = ref_cr * ref_A / F["g654e_Aeff"]
    cr_lo = F["g654e_gR_ratio"] * ref_cr
    ap = F["g654e_alpha_p"]["1425"]
    g_lo, le = g_onoff(cr_lo, ap)
    g_hi, _ = g_onoff(cr_s, ap)
    rows["G.654.E (Vascade EX2500)"] = {
        "Aeff_um2": F["g654e_Aeff"], "CR_traced": None,
        "CR_traced_note": "about 0.5 x G.652.D reported (coefficient type not stated)",
        "CR_area_scaled": cr_s, "CR_from_ratio": cr_lo, "scaled_over_ratio": cr_s / cr_lo - 1,
        "alpha_s": F["g654e_alpha_s"], "alpha_p_1425": ap, "G_onoff_dB_range": [g_lo, g_hi], "Leff_p_km": le}
    # pump-wavelength dependence of the SSMF efficiency (GNPy-type scaling of the measured profile)
    eff = RamanEfficiency(V, "full")
    pk = float(eff.off[np.argmax(eff.g0)])
    cr_pump = {}
    for lp in F["cr_eval_pump_nm"]:
        fp = float(thz(lp))
        cr_pump[f"{lp:.0f}"] = float(eff(fp, fp - pk))
    return {"conditions": f"counter-pumped, undepleted single pump, {P_pump} W, {L_km} km; on-off gain = 4.343 C_R P L_eff(alpha_p)",
            "rows": rows, "ssmf_peak_CR_vs_pump_nm": cr_pump, "peak_offset_THz": pk}


# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------
def checks(V):
    out = {}
    eff = RamanEfficiency(V, "full")
    A1454 = float(eff.area(V["f_ref_raman"] / 1e12))
    out["area_model"] = {"A_1454nm_um2": A1454 * 1e12, "GNPy_reference_um2": V["A_ref_raman_gnpy"] * 1e12,
                         "rel_dev": A1454 / V["A_ref_raman_gnpy"] - 1,
                         "A_um2_at": {f"{l:.0f}": float(eff.area(thz(l)) * 1e12) for l in (1260, 1310, 1550, 1675)}}
    # (a) code check: triangular gain without cut-off, constant loss, energy-conserving depletion -> closed form
    a_db = V["check_alpha_const"]
    leff = (1 - np.exp(-a_db * DB * V["L_span"])) / (a_db * DB)
    cc = {}
    for name, (l0, l1, ptot) in V["check_loads"].items():
        r = isrs_run(V, {"contiguous": (l0, l1, ptot)}, mode="tri_nocut", photon=False, alpha_const=a_db)
        tilt_num = float(r["d_dB"][np.argmin(r["f"])] - r["d_dB"][np.argmax(r["f"])])
        B = float(r["f"].max() - r["f"].min())
        tilt_cf = TEN_LOG_E * V["Cr_slope_S"] * 1e-3 * 10 ** (ptot / 10) * leff * B
        cc[name] = {"tilt_numerical_dB": tilt_num, "tilt_closed_form_dB": tilt_cf, "abs_diff_dB": abs(tilt_num - tilt_cf),
                    "B_bin_centres_THz": B, "Leff_km": leff}
    out["code_check_triangular_closed_form"] = cc
    # (b) slope of a linear fit through the origin of the measured profile over 0-15 THz
    xs = np.linspace(0.0, 15.0, 1501)
    gs = np.interp(xs, eff.off, eff.g0)
    out["profile_linear_slope_0_15THz"] = {"slope_per_W_km_THz": float(np.sum(xs * gs) / np.sum(xs * xs)),
                                           "Semrau_Cr": V["Cr_slope_S"]}
    # (c) physical comparison: measured profile versus Eq. (10) of Semrau et al. for C and C+L loads
    phys = {}
    for name, (l0, l1, ptot) in V["check_loads"].items():
        r = isrs_run(V, {"contiguous": (l0, l1, ptot)}, mode="full")
        tilt = float(r["d_dB"][np.argmin(r["f"])] - r["d_dB"][np.argmax(r["f"])])
        am = float(np.mean(r["alpha_dB_km"]))
        le = (1 - np.exp(-am * DB * V["L_span"])) / (am * DB)
        B = float(thz(l0) - thz(l1))
        phys[name] = {"tilt_measured_profile_dB": tilt,
                      "tilt_Eq10_dB": TEN_LOG_E * V["Cr_slope_S"] * 1e-3 * 10 ** (ptot / 10) * le * B,
                      "mean_alpha_dB_km": am, "B_THz": B}
    out["measured_profile_vs_Eq10"] = phys
    return out


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def fmt(x, nd=1):
    return f"{x:+.{nd}f}".replace("-", "−")


def main():
    span_card = cards.load(CARD_SPAN)
    fib_card = cards.load(CARD_FIBERS)
    bdfa_card = cards.load(CARD_BDFA)
    V = cards.values(span_card)
    L = V["L_span"]
    ptot = 10 * np.log10(sum(10 ** (p / 10) for p in V["P_band_P"].values()))
    R = {"conditions": {
        "span": f"{L:.0f} km G.652.D, attenuation from card MULTIBAND-G652D-SPAN (anchors of Puttnam et al. 2025, piecewise-linear)",
        "load_main": "per-band total launch powers of Puttnam et al. 2025 ("
                     + ", ".join(f"{b} {p:g}" for b, p in V["P_band_P"].items())
                     + f" dBm), each spread uniformly over the full ITU-T band; total {ptot:.1f} dBm",
        "model": "coupled power equations of all signal bins (0.1 THz), measured SSMF Raman profile of GNPy (0-42 THz) scaled with pump frequency and overlap area, photon-number-conserving depletion; no Raman pumps, no ASE, no Kerr effects",
        "sign": "ISRS power change > 0: the band gains power from shorter-wavelength bands; < 0: it is depleted"}}

    # 1-2. geometry and passive loss
    geo = band_geometry(V)
    for b in BANDS:
        lo, hi = geo[b]["lambda_nm"]
        a_lo, a_hi = attenuation([lo, hi], V)
        geo[b]["alpha_edges_dB_km"] = [float(a_lo), float(a_hi)]
        lam = np.linspace(lo, hi, 201)
        a = attenuation(lam, V)
        geo[b]["span_loss_range_dB"] = [float(a.min() * L), float(a.max() * L)]
        geo[b]["extrapolated"] = bool(lo < V["att_lambda"][0] or hi > V["att_lambda"][-1])
    R["band_geometry"] = geo
    R["attenuation_checks"] = {"interp_1385": float(attenuation(1385, V)[0]), "interp_1425": float(attenuation(1425, V)[0]),
                               "measured_Y_1385_1425": [V["att_pump_lwp_Y"]["1385"], V["att_pump_lwp_Y"]["1425"]],
                               "extrap_1260": float(attenuation(1260, V)[0]), "F_O_band_max": V["att_O_F"],
                               "extrap_1675": float(attenuation(1675, V)[0]),
                               "spec_SMF28U_1310_1550_1625": [V["att_spec_smf28ultra"][k] for k in ("1310", "1550", "1625")]}

    # 3. ISRS: main scenario, band-alone runs, other loads
    load = dict(V["P_band_P"])
    main_run = isrs_run(V, load)
    R["isrs_main"] = band_summary(main_run)
    R["isrs_main"]["P_tot_dBm"] = main_run["P_tot_dBm"]
    alone = {b: isrs_run(V, {b: load[b]}) for b in BANDS}
    R["isrs_band_alone"] = {b: band_summary(alone[b])[b] for b in BANDS}
    scen = {"C_only": {"C": load["C"]},
            "C+L": {"C": load["C"], "L": load["L"]},
            "S+C+L": {"S": load["S"], "C": load["C"], "L": load["L"]},
            "O_to_L": {b: load[b] for b in "OESCL"}}
    Fp = {b: V["Pch_F_75km"][b] + 10 * np.log10(V["nch_F"][b]) for b in "OESCL"}
    scen["Ferrari75_O_to_L"] = Fp
    R["isrs_scenarios"] = {}
    for k, ld in scen.items():
        r = isrs_run(V, ld)
        R["isrs_scenarios"][k] = {"load_dBm": {b: round(float(p), 2) for b, p in ld.items()},
                                  "P_tot_dBm": r["P_tot_dBm"], "bands": band_summary(r)}

    # variants and sensitivities of the main scenario
    R["profile_variants"] = {}
    for mode in ("full", "freq", "none", "trunc15", "triangular"):
        r = isrs_run(V, load, mode=mode)
        R["profile_variants"][mode] = {b: round(s["mean_dB"], 3) for b, s in r["bands"].items()}
    r = isrs_run(V, load, photon=False)
    R["profile_variants"]["full_without_photon_factor"] = {b: round(s["mean_dB"], 3) for b, s in r["bands"].items()}
    R["sensitivity_attenuation"] = {}
    for dlt in (-0.03, 0.03):
        r = isrs_run(V, load, att_delta=dlt)
        R["sensitivity_attenuation"][f"{dlt:+.2f}"] = {b: round(s["mean_dB"] - main_run["bands"][b]["mean_dB"], 4)
                                                       for b, s in r["bands"].items()}
    R["sensitivity_bin_width"] = {}
    for dfb in (0.05, 0.2):
        r = isrs_run(V, load, df_bin=dfb)
        R["sensitivity_bin_width"][f"{dfb}"] = {
            b: round(max(abs(s[k] - main_run["bands"][b][k]) for k in ("mean_dB", "min_dB", "max_dB")), 4)
            for b, s in r["bands"].items()}
    R["sensitivity_span_length"] = {}
    for Lk in (50.0, 100.0):
        r = isrs_run(V, load, L_km=Lk)
        R["sensitivity_span_length"][f"{Lk:.0f} km"] = {b: round(s["mean_dB"], 3) for b, s in r["bands"].items()}

    # 4. gain requirement: alpha L - ISRS change, band alone and full load
    greq = {}
    for b in BANDS:
        m = main_run["tag"] == b
        g_full = main_run["span_loss_dB"][m] - main_run["d_dB"][m]
        g_alone = alone[b]["span_loss_dB"] - alone[b]["d_dB"]
        fb = main_run["f"][m]
        greq[b] = {"band_alone_dB": [float(g_alone.min()), float(g_alone.max())],
                   "full_load_dB": [float(g_full.min()), float(g_full.max())],
                   "mean_gain_change_full_minus_alone_dB": float(alone[b]["bands"][b]["mean_dB"]
                                                                 - main_run["bands"][b]["mean_dB"]),
                   "launch_psd_dBm_per_50GHz": float(load[b] - 10 * np.log10(geo[b]["width_THz"] / 0.05)),
                   "range_dB": [float(min(g_alone.min(), g_full.min())), float(max(g_alone.max(), g_full.max()))],
                   "tilt_full_dB": float(g_full[np.argmax(fb)] - g_full[np.argmin(fb)]),
                   "tilt_alone_dB": float(g_alone[np.argmax(fb)] - g_alone[np.argmin(fb)]),
                   "passive_dB": [float(main_run["span_loss_dB"][m].min()), float(main_run["span_loss_dB"][m].max())]}
    R["gain_requirement"] = greq
    R["gain_requirement_note"] = ("G_req = alpha(lambda) L - ISRS power change; tilt = G_req(short edge) - G_req(long edge); "
                                  "band combiner/splitter, gain-flattening filter, connector losses and control margin not included")

    # 5-6. fibers and BDFA
    R["checks"] = checks(V)
    R["raman_fiber_types"] = raman_fiber_types(fib_card, V)
    R["bdfa_consistency"] = bdfa_two_level_check(bdfa_card)

    # markdown band table (printed)
    lines = ["| Band | Range [nm] / width [THz] | Passive loss, 80 km [dB] | Launch power, band total [dBm] | "
             "ISRS power change, full load: band average (range) [dB] | Gain to supply [dB]: band alone / full load |",
             "|---|---|---|---|---|---|"]
    for b in BANDS:
        g = geo[b]
        s = main_run["bands"][b]
        q = greq[b]
        lines.append(f"| {b} | {g['lambda_nm'][0]:.0f}-{g['lambda_nm'][1]:.0f} / {g['width_THz']:.1f} | "
                     f"{g['span_loss_range_dB'][0]:.1f}-{g['span_loss_range_dB'][1]:.1f} | {load[b]:.1f} | "
                     f"{fmt(s['mean_dB'])} ({fmt(s['min_dB'])} to {fmt(s['max_dB'])}) | "
                     f"{q['band_alone_dB'][0]:.1f}-{q['band_alone_dB'][1]:.1f} / {q['full_load_dB'][0]:.1f}-{q['full_load_dB'][1]:.1f} |")
    R["markdown_band_table_core"] = "\n".join(lines)
    print(R["markdown_band_table_core"])
    print()
    for k in ("profile_variants", "sensitivity_attenuation", "sensitivity_bin_width", "sensitivity_span_length"):
        print(k, json.dumps(R[k]))
    print("scenarios:", json.dumps({k: {b: round(s['mean_dB'], 2) for b, s in v['bands'].items()} for k, v in R["isrs_scenarios"].items()}))
    print("checks:", json.dumps(R["checks"], indent=1))
    print("fibers:", json.dumps(R["raman_fiber_types"], indent=1))
    print("bdfa:", json.dumps(R["bdfa_consistency"], indent=1))

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "Tables_bands_results.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)
    return R, main_run, alone


if __name__ == "__main__":
    main()
