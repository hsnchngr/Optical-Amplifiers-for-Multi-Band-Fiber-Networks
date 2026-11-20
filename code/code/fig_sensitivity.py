"""Figure 10 (revision 3): parameter sensitivity of the reference models and the parameter-mixing example of
Section 5.2.

Panels
  a) EDFA, card EDFA-GD91-T1, ASE-free two-level model (models.edfa_ode): 10 m fiber, 30 mW forward pump at
     1480 nm, -5 dBm signal at 1550 nm (operating point of the architecture example, Figure 8, in saturated
     operation). Output: signal gain between the fiber ends.
  b) SOA, card SOA-BULK-REF, traveling-wave model without ASE (models.soa_numerical): 150 mA, small-signal
     chip gain (1 nW input).
  c) Distributed Raman, card RAMAN-G652, depleted-pump BVP (models.raman_numerical): 50 km, 500 mW
     counter-propagating pump at 1450 nm, +16 dBm total signal power lumped at 1550 nm. Output: net gain
     P_s(L)/P_s(0) between span input and span output (fiber only).
  d) Gain error caused by combining the saturation parameter of the Table II fiber of [41] (card EDFA-GD91-T2)
     with the Table I card (EDFA-GD91-T1), against pump power; 10 m, -20 dBm and -5 dBm signal at 1550 nm.

Method (panels a-c): every entry is changed one at a time by +-10 % (sensitivity) and by +-u (uncertainty-
weighted effect, 'importance'), where u is a typical relative standard uncertainty assigned from the provenance
category of the entry. The values of u are ASSUMPTIONS of this figure (table U below); they are written to the
results file and must be stated with the figure. Entries that enter a model only as a product are treated as one
entry: C_R = g_R/(K A_eff) (Raman), Gamma*a and w*d (SOA). Entries of category 'fitted' (SOA Gamma*a and N0, fitted
jointly to a 25 dB target) get no u: their uncertainty is not defined without a fit to device data.

Adequacy check of panel a: the gain and the +-10 % sensitivities are recomputed with forward and backward ASE,
(i) one lumped 30 nm ASE band at 1550 nm (as edfa_multipass.py) and (ii) 2 nm ASE bins over 1500-1568 nm whose
absorption shape is the digitized shape of the fiber of ref. [124] scaled to alpha(1550) of this card, with g*
from the McCumber relation (card entries lambda_x and T). Variant (ii) transfers a spectral SHAPE between fibers
and is used only to estimate the size of the ASE effect.

Outputs: ../figures/Fig10_sensitivity.png/.pdf and ../figures/Fig10_sensitivity_results.json.
Usage  : python fig_sensitivity.py [--plot-only]   (--plot-only redraws the figure from the results file)
"""
import copy
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

import style
import cards
import models
from models import EDFA, SOA, RAMAN, h, c, dbm2w, w2dbm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
kB = 1.380649e-23

# ---------------------------------------------------------------------------------------------------------------
# Operating points of this figure (configuration choices, category 'config'; powers in W, lengths in m)
# ---------------------------------------------------------------------------------------------------------------
OP_EDFA = dict(L=10.0, Pp=30e-3, Ps=dbm2w(-5.0))          # panel a (same fiber length and pump as Figure 8)
OP_EDFA_SMALL = dict(L=10.0, Pp=30e-3, Ps=dbm2w(-20.0))   # small-signal variant (results file only)
OP_SOA = dict(I=0.150, Pin=1e-9)                         # small-signal chip gain at the fit current of the card
OP_RAMAN = dict(Pp=0.500, Ps=dbm2w(16.0))                # span length L is a card entry (50 km)
MIX_L = 10.0                                             # panel d fiber length
MIX_PIN_DBM = (-20.0, -5.0)
MIX_PUMPS_MW = np.geomspace(3.0, 300.0, 49)
MIX_MARK_MW = (3.0, 5.0, 10.0, 30.0, 100.0)
MIX_TABLE_MW = (3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0)   # results file only (values quoted in the text)

# ---------------------------------------------------------------------------------------------------------------
# Typical relative standard uncertainties u (ASSUMPTIONS of this figure), by entry, with the reason
# ---------------------------------------------------------------------------------------------------------------
U = {
    # EDFA (card EDFA-GD91-T1 + operating point)
    ("edfa", "g_s"): (0.10, "measured gain coefficient; gain-coefficient measurements and McCumber-derived values carry about 5-10 %"),
    ("edfa", "alpha_s"): (0.05, "measured absorption coefficient (cut-back); about 5 %"),
    ("edfa", "g_p"): (0.10, "measured gain coefficient; quoted as 0.5 dB/m, so rounding alone is +-10 %"),
    ("edfa", "alpha_p"): (0.05, "measured absorption coefficient (cut-back); about 5 %"),
    ("edfa", "xi"): (0.10, "measured indirectly (saturation power) or derived from b, N_t and tau; about 10 %"),
    ("edfa", "L"): (0.01, "fiber length, measured to about 0.1 m"),
    ("edfa", "Pp"): (0.05, "launched pump power (WDM coupler and splice loss, power meter); about 0.2 dB"),
    ("edfa", "Ps"): (0.05, "launched signal power; about 0.2 dB"),
    # SOA (card SOA-BULK-REF + operating point)
    ("soa", "Gamma_a"): (None, "fitted jointly with N0 to a 25 dB target; no uncertainty without a fit to device data"),
    ("soa", "N0"): (None, "fitted jointly with Gamma*a to a 25 dB target; no uncertainty without a fit to device data"),
    ("soa", "wd"): (0.07, "design values; about 5 % fabrication tolerance each for w and d, combined in quadrature"),
    ("soa", "L"): (0.01, "cleaved length; about 1 %"),
    ("soa", "I"): (0.01, "bias current; about 1 %"),
    ("soa", "alpha_int"): (0.30, "assumed value; about 30 %"),
    ("soa", "A"): (0.30, "value of another device's model; about 30 % (literature spread can be larger)"),
    ("soa", "B"): (0.30, "value of another device's model; about 30 % (literature spread can be larger)"),
    ("soa", "C"): (0.30, "value of another device's model; about 30 % (literature spread can be larger)"),
    # Raman (card RAMAN-G652 + operating point)
    ("raman", "C_R"): (0.10, "derived from class-typical g_R and A_eff; fiber-to-fiber spread within the G.652 class about 10 %"),
    ("raman", "alpha_p"): (0.10, "assumed 0.25 dB/km between the 1383 nm and 1490 nm specifications; about +-0.025 dB/km"),
    ("raman", "alpha_s"): (0.05, "assumed cabled value 0.19 dB/km; about +-0.01 dB/km"),
    ("raman", "Pp"): (0.05, "launched pump power (WDM coupler, connectors); about 0.2 dB"),
    ("raman", "Ps"): (0.05, "launched total signal power; about 0.2 dB"),
    ("raman", "L"): (0.01, "span length (OTDR); about 1 %"),
}

# Entry definitions: key, plot label, where it is perturbed ('card' key list or 'op' key), category, level
_CE = cards.load("EDFA-GD91-T1")["entries"]
_CS = cards.load("SOA-BULK-REF")["entries"]
_CR = cards.load("RAMAN-G652")["entries"]


def _lvl(entries, *keys):
    return "/".join(sorted({entries[k]["verification"]["level"] for k in keys}))


EDFA_ENTRIES = [
    ("g_s", r"$g^*$(1550 nm)", ("card", "g_s"), _CE["g_s"]["category"], _lvl(_CE, "g_s")),
    ("alpha_s", r"$\alpha$(1550 nm)", ("card", "alpha_s"), _CE["alpha_s"]["category"], _lvl(_CE, "alpha_s")),
    ("g_p", r"$g^*$(1480 nm)", ("card", "g_p"), _CE["g_p"]["category"], _lvl(_CE, "g_p")),
    ("alpha_p", r"$\alpha$(1480 nm)", ("card", "alpha_p"), _CE["alpha_p"]["category"], _lvl(_CE, "alpha_p")),
    ("xi", r"$\zeta$", ("card", "xi"), _CE["xi"]["category"], _lvl(_CE, "xi")),
    ("L", r"$L$", ("op", "L"), "config", "op"),
    ("Pp", r"$P_\mathrm{p}$", ("op", "Pp"), "config", "op"),
    ("Ps", r"$P_\mathrm{s}$", ("op", "Ps"), "config", "op"),
]
SOA_ENTRIES = [
    ("Gamma_a", r"$\Gamma a$", ("card", "Gamma"), "fitted", _lvl(_CS, "Gamma", "a")),   # enters only as Gamma*a
    ("N0", r"$N_0$", ("card", "N0"), _CS["N0"]["category"], _lvl(_CS, "N0")),
    ("wd", r"$wd$", ("card", "w"), "config", _lvl(_CS, "w", "d")),                     # enters only as w*d
    ("L", r"$L$", ("card", "L"), _CS["L"]["category"], _lvl(_CS, "L")),
    ("I", r"$I$", ("op", "I"), "config", "op"),
    # internal loss: symbol l_int of the notation registry (card key alpha_int)
    ("alpha_int", r"$l_\mathrm{int}$", ("card", "alpha_int"), _CS["alpha_int"]["category"], _lvl(_CS, "alpha_int")),
    ("A", r"$A$", ("card", "A"), _CS["A"]["category"], _lvl(_CS, "A")),
    ("B", r"$B$", ("card", "B"), _CS["B"]["category"], _lvl(_CS, "B")),
    ("C", r"$C$", ("card", "C"), _CS["C"]["category"], _lvl(_CS, "C")),
]
RAMAN_ENTRIES = [
    ("C_R", r"$C_\mathrm{R}$", ("card", "gR"), _CR["C_R"]["category"], _lvl(_CR, "C_R")),   # C_R = g_R/(K A_eff)
    # fiber attenuation at pump and signal: symbols l_p, l_s of the notation registry (card keys alpha_p, alpha_s)
    ("alpha_p", r"$l_\mathrm{p}$", ("card", "alpha_p"), _CR["alpha_p"]["category"], _lvl(_CR, "alpha_p")),
    ("alpha_s", r"$l_\mathrm{s}$", ("card", "alpha_s"), _CR["alpha_s"]["category"], _lvl(_CR, "alpha_s")),
    ("Pp", r"$P_\mathrm{p}$", ("op", "Pp"), "config", "op"),
    ("Ps", r"$P_\mathrm{s}$", ("op", "Ps"), "config", "op"),
    ("L", r"$L$", ("card", "L"), _CR["L"]["category"], _lvl(_CR, "L")),
]


# ---------------------------------------------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------------------------------------------
def edfa_fn(p, op):
    z, Pp, Ps = models.edfa_ode(op["Pp"], op["Ps"], op["L"], p=p, npts=2)
    return 10 * np.log10(Ps[-1] / op["Ps"])


def soa_fn(p, op):
    z, P = models.soa_numerical(op["I"], op["Pin"], p=p, npts=2)
    return 10 * np.log10(P[-1] / op["Pin"])


def raman_fn(p, op):
    z, Ps, Pp, status = models.raman_numerical(op["Pp"], op["Ps"], p=p, n=201)
    if status != 0:
        raise RuntimeError(f"solve_bvp status {status}")
    return 10 * np.log10(Ps[-1] / op["Ps"])


def perturbed(base_p, base_op, where, f):
    p, op = copy.deepcopy(base_p), dict(base_op)
    kind, key = where
    (p if kind == "card" else op)[key] *= f
    return p, op


def run_entries(model, fn, base_p, base_op, entries):
    y0 = fn(base_p, base_op)
    rows = []
    for key, lab, where, cat, lvl in entries:
        u, why = U[(model, key)]
        r = dict(entry=key, label=lab, perturbed=f"{where[0]}:{where[1]}", category=cat, level=lvl, u_rel=u, u_basis=why)
        r["d_plus10"] = fn(*perturbed(base_p, base_op, where, 1.10)) - y0
        r["d_minus10"] = fn(*perturbed(base_p, base_op, where, 0.90)) - y0
        if u is None:
            r["d_plus_u"] = r["d_minus_u"] = r["importance"] = None
        else:
            r["d_plus_u"] = fn(*perturbed(base_p, base_op, where, 1 + u)) - y0
            r["d_minus_u"] = fn(*perturbed(base_p, base_op, where, 1 - u)) - y0
            r["importance"] = max(abs(r["d_plus_u"]), abs(r["d_minus_u"]))
        rows.append(r)
    weighted = [0.5 * (abs(r["d_plus_u"]) + abs(r["d_minus_u"])) for r in rows if r["u_rel"] is not None]
    rss = float(np.sqrt(np.sum(np.square(weighted))))
    by10 = sorted(rows, key=lambda r: -max(abs(r["d_plus10"]), abs(r["d_minus10"])))
    byu = sorted([r for r in rows if r["u_rel"] is not None], key=lambda r: -r["importance"])
    for i, r in enumerate(by10):
        r["rank_pm10"] = i + 1
    for i, r in enumerate(byu):
        r["rank_weighted"] = i + 1
    return y0, rows, rss


# ---------------------------------------------------------------------------------------------------------------
# ASE check model (adequacy of panel a and robustness of panel d)
# ---------------------------------------------------------------------------------------------------------------
_EC = cards.values(cards.load("EDFA-GD91-T1"))
LAM_X, T_K = _EC["lambda_x"] * 1e-9, _EC["T"]


def ase_bins(p, kind, dl_nm=2.0, lam_lo=1500.0, lam_hi=1568.0, lumped_bw_nm=30.0):
    """ASE channels built from the BASELINE card p (kept fixed when card entries are perturbed).
    'lumped'  : one 30 nm band at the signal wavelength with the 1550 nm coefficients (as edfa_multipass.py);
    'spectral': 2 nm bins over 1500-1568 nm; absorption shape of the EDF of ref. [124] (digitized data file)
                scaled to alpha(1550) of this card; g* from the McCumber relation with the card's lambda_x and T."""
    lam_s = p["lam_s"]
    if kind == "lumped":
        dnu = c / (lam_s - 0.5 * lumped_bw_nm * 1e-9) - c / (lam_s + 0.5 * lumped_bw_nm * 1e-9)
        return np.array([lam_s]), np.array([p["alpha_s"]]), np.array([p["g_s"]]), np.array([dnu])
    lam = np.arange(lam_lo + dl_nm / 2, lam_hi, dl_nm) * 1e-9
    lp, ap, _ = cards.perin2019_coefficients()
    a = p["alpha_s"] * np.interp(lam * 1e9, lp, ap) / np.interp(lam_s * 1e9, lp, ap)
    g = a * np.exp(h * c / (kB * T_K) * (1.0 / LAM_X - 1.0 / lam))
    return lam, a, g, c * dl_nm * 1e-9 / lam ** 2


def edfa_ase_gain(p, op, bins, N=400, tol=1e-6, max_iter=200, relax=0.5):
    """Two-level model with forward pump, forward signal and forward/backward ASE (2 polarizations), no background
    loss; forward-backward sweeps with RK2 steps. Returns (gain [dB], dict of diagnostics)."""
    lam_a, a_a, g_a, dnu = bins
    K = len(lam_a)
    al = np.r_[p["alpha_p"], p["alpha_s"], a_a, a_a]
    gg = np.r_[p["g_p"], p["g_s"], g_a, g_a]
    nu = np.r_[c / p["lam_p"], c / p["lam_s"], c / lam_a, c / lam_a]
    sp = np.r_[0.0, 0.0, 2 * h * c / lam_a * dnu, 2 * h * c / lam_a * dnu]
    fw, bw = np.arange(0, 2 + K), np.arange(2 + K, 2 + 2 * K)
    w = 1.0 / (h * nu * p["xi"])
    L, dz = op["L"], op["L"] / N
    P = np.zeros((2 + 2 * K, N + 1)); P[0, 0] = op["Pp"]; P[1, 0] = op["Ps"]

    def rhs(Pv, idx):
        x = Pv * w
        n2 = np.dot(x, al) / (1.0 + np.dot(x, al + gg))
        return ((al[idx] + gg[idx]) * n2 - al[idx]) * Pv[idx] + gg[idx] * n2 * sp[idx]

    last = None
    for it in range(max_iter):
        for i in range(N):
            Pv = P[:, i].copy()
            Pm = Pv.copy(); Pm[fw] += 0.5 * dz * rhs(Pv, fw); Pm[bw] = 0.5 * (P[bw, i] + P[bw, i + 1])
            P[fw, i + 1] = np.maximum(Pv[fw] + dz * rhs(Pm, fw), 0.0)
        nb = np.zeros((K, N + 1))
        for i in range(N, 0, -1):
            Pv = P[:, i].copy(); Pv[bw] = nb[:, i]
            Pm = Pv.copy(); Pm[bw] = Pv[bw] + 0.5 * dz * rhs(Pv, bw); Pm[fw] = 0.5 * (P[fw, i] + P[fw, i - 1])
            nb[:, i - 1] = np.maximum(Pv[bw] + dz * rhs(Pm, bw), 0.0)
        P[bw] = nb if it == 0 else relax * nb + (1 - relax) * P[bw]
        out = P[1, N]
        if last is not None and it > 3 and abs(10 * np.log10(out / last)) < tol:
            break
        last = out
    return 10 * np.log10(P[1, N] / op["Ps"]), dict(sweeps=it + 1, ase_fwd_dBm=float(w2dbm(P[2:2 + K, N].sum())),
                                                    ase_bwd_dBm=float(w2dbm(P[2 + K:, 0].sum())))


def ase_adequacy(op):
    """Gain and +-10 % sensitivities of the EDFA card entries with the ASE-free model and both ASE variants."""
    out = {}
    for kind in ("none", "lumped", "spectral"):
        bins = None if kind == "none" else ase_bins(EDFA, kind)
        fn = edfa_fn if kind == "none" else (lambda p, o, b=bins: edfa_ase_gain(p, o, b)[0])
        y0 = fn(EDFA, op)
        sens = {}
        for key, lab, where, cat, lvl in EDFA_ENTRIES:
            sens[key] = [fn(*perturbed(EDFA, op, where, f)) - y0 for f in (1.1, 0.9)]
        out[kind] = dict(gain_dB=y0, sens_pm10=sens)
        if kind != "none":
            out[kind]["diag"] = edfa_ase_gain(EDFA, op, bins)[1]
    for kind in ("lumped", "spectral"):
        out[kind]["gain_minus_noASE_dB"] = out[kind]["gain_dB"] - out["none"]["gain_dB"]
        out[kind]["max_abs_sens_diff_dB"] = max(abs(a - b) for k in out["none"]["sens_pm10"]
                                                for a, b in zip(out[kind]["sens_pm10"][k], out["none"]["sens_pm10"][k]))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------------------------------------------
def compute():
    t0 = time.time()
    res = dict(figure="Fig10_sensitivity", generated_by="fig_sensitivity.py", cards_used=[
        "EDFA-GD91-T1", "EDFA-GD91-T2 (cards/edfa_gd91_t2.json)", "SOA-BULK-REF", "RAMAN-G652",
        "data/perin2019_giles_coefficients.csv (absorption SHAPE only, ASE check)"])
    res["uncertainty_assumptions"] = [dict(model=m, entry=k, u_rel=v[0], basis=v[1]) for (m, k), v in U.items()]

    y, rows, rss = run_entries("edfa", edfa_fn, EDFA, OP_EDFA, EDFA_ENTRIES)
    res["edfa"] = dict(card="EDFA-GD91-T1", model="models.edfa_ode: two-level, forward 1480 nm pump, ASE and background loss neglected",
                       operating_point=dict(L_m=OP_EDFA["L"], Pp_mW=OP_EDFA["Pp"] * 1e3, Ps_dBm=float(w2dbm(OP_EDFA["Ps"])),
                                            lambda_p_nm=1480, lambda_s_nm=1550),
                       output="signal gain between the fiber ends [dB]", baseline_dB=y, rows=rows, rss_u_dB=rss)
    y, rows, rss = run_entries("edfa", edfa_fn, EDFA, OP_EDFA_SMALL, EDFA_ENTRIES)
    res["edfa_small_signal_variant"] = dict(operating_point=dict(L_m=10.0, Pp_mW=30.0, Ps_dBm=-20.0), baseline_dB=y,
                                            rows=rows, rss_u_dB=rss, note="results file only; not plotted")
    print(f"EDFA done ({time.time() - t0:.0f} s)")

    y, rows, rss = run_entries("soa", soa_fn, SOA, OP_SOA, SOA_ENTRIES)
    res["soa"] = dict(card="SOA-BULK-REF", model="models.soa_numerical: traveling-wave, A/B/C recombination, internal loss, ASE neglected",
                      operating_point=dict(I_mA=OP_SOA["I"] * 1e3, Pin_W=OP_SOA["Pin"], lambda_nm=1550),
                      output="small-signal chip gain [dB] (fit target of Gamma, a, N0: 25 dB)", baseline_dB=y, rows=rows,
                      rss_u_dB_excluding_fitted=rss)
    print(f"SOA done ({time.time() - t0:.0f} s)")

    y, rows, rss = run_entries("raman", raman_fn, RAMAN, OP_RAMAN, RAMAN_ENTRIES)
    res["raman"] = dict(card="RAMAN-G652", model="models.raman_numerical: counter-pumped, pump depletion, single signal wavelength",
                        operating_point=dict(L_km=RAMAN["L"] / 1e3, Pp_mW=OP_RAMAN["Pp"] * 1e3, Ps_dBm=float(w2dbm(OP_RAMAN["Ps"])),
                                             lambda_p_nm=1450, lambda_s_nm=1550,
                                             note="+16 dBm total signal power lumped at 1550 nm (single-wavelength representation of a WDM load)"),
                        output="net gain P_s(L)/P_s(0) between span input and output, fiber only [dB]", baseline_dB=y,
                        rows=rows, rss_u_dB=rss)
    print(f"Raman done ({time.time() - t0:.0f} s)")

    # identities used to merge entries and to explain panel d
    chk = {}
    p1 = copy.deepcopy(RAMAN); p1["gR"] *= 1.1
    p2 = copy.deepcopy(RAMAN); p2["Aeff"] /= 1.1
    chk["raman_gR_x1.1_vs_Aeff_div1.1_dB"] = [raman_fn(p1, OP_RAMAN) - res["raman"]["baseline_dB"],
                                              raman_fn(p2, OP_RAMAN) - res["raman"]["baseline_dB"]]
    s1 = copy.deepcopy(SOA); s1["Gamma"] *= 1.1
    s2 = copy.deepcopy(SOA); s2["a"] *= 1.1
    s3 = copy.deepcopy(SOA); s3["w"] *= 1.1
    s4 = copy.deepcopy(SOA); s4["d"] *= 1.1
    b = res["soa"]["baseline_dB"]
    chk["soa_Gamma_vs_a_x1.1_dB"] = [soa_fn(s1, OP_SOA) - b, soa_fn(s2, OP_SOA) - b]
    chk["soa_w_vs_d_x1.1_dB"] = [soa_fn(s3, OP_SOA) - b, soa_fn(s4, OP_SOA) - b]
    k = 4.2 / 1.5
    pm = copy.deepcopy(EDFA); pm["xi"] *= k
    chk["edfa_zeta_scaling_residual_dB"] = max(abs(edfa_fn(pm, dict(L=10.0, Pp=P, Ps=dbm2w(-20)))
                                                   - edfa_fn(EDFA, dict(L=10.0, Pp=P / k, Ps=dbm2w(-20) / k)))
                                               for P in (5e-3, 30e-3, 100e-3))
    try:   # cross-check of the lumped ASE variant against the Section 4.1 model, if its interface is unchanged
        import edfa_multipass
        ref = edfa_multipass.solve(OP_EDFA_SMALL["Pp"], OP_EDFA_SMALL["Ps"], OP_EDFA_SMALL["L"], N=400)
        chk["lumped_vs_edfa_multipass_dB"] = (edfa_ase_gain(EDFA, OP_EDFA_SMALL, ase_bins(EDFA, "lumped"))[0]
                                              - 10 * np.log10(ref["G"]))
    except Exception as e:  # pragma: no cover
        chk["lumped_vs_edfa_multipass_dB"] = f"not evaluated ({type(e).__name__})"
    res["checks"] = chk

    # adequacy of the EDFA model at the operating point of panel a and at the small-signal variant
    res["edfa"]["ase_check"] = ase_adequacy(OP_EDFA)
    res["edfa_small_signal_variant"]["ase_check"] = ase_adequacy(OP_EDFA_SMALL)
    print(f"ASE checks done ({time.time() - t0:.0f} s)")

    # panel d: parameter mixing
    zeta1 = EDFA["xi"]
    zeta2 = cards.values(cards.load(os.path.join(cards.CARD_DIR, "edfa_gd91_t2.json")))["xi"]
    pmix = copy.deepcopy(EDFA); pmix["xi"] = zeta2
    mix = dict(L_m=MIX_L, zeta_table1=zeta1, zeta_table2=zeta2, ratio=zeta2 / zeta1,
               equivalent_launched_power_error_dB=-10 * np.log10(zeta2 / zeta1), curves={}, markers={}, ase_robustness={})
    for pin in MIX_PIN_DBM:
        g1, g2 = [], []
        for P in MIX_PUMPS_MW:
            op = dict(L=MIX_L, Pp=P * 1e-3, Ps=dbm2w(pin))
            g1.append(edfa_fn(EDFA, op)); g2.append(edfa_fn(pmix, op))
        g1, g2 = np.array(g1), np.array(g2)
        mix["curves"][f"{pin:g} dBm"] = dict(Pp_mW=MIX_PUMPS_MW.tolist(), G_consistent_dB=g1.tolist(),
                                             G_mixed_dB=g2.tolist(), dG_dB=(g2 - g1).tolist())
        mk, rb = {}, {}
        for P in MIX_MARK_MW:
            op = dict(L=MIX_L, Pp=P * 1e-3, Ps=dbm2w(pin))
            G1, G2 = edfa_fn(EDFA, op), edfa_fn(pmix, op)
            mk[f"{P:g} mW"] = dict(G_consistent_dB=G1, G_mixed_dB=G2, dG_dB=G2 - G1)
            r = {}
            for kind in ("lumped", "spectral"):
                bins = ase_bins(EDFA, kind)
                a1, a2 = edfa_ase_gain(EDFA, op, bins)[0], edfa_ase_gain(pmix, op, bins)[0]
                r[kind] = dict(G_consistent_dB=a1, G_mixed_dB=a2, dG_dB=a2 - a1)
            rb[f"{P:g} mW"] = r
        mix["markers"][f"{pin:g} dBm"] = mk
        mix["ase_robustness"][f"{pin:g} dBm"] = rb
        tab = {}
        for P in MIX_TABLE_MW:
            op = dict(L=MIX_L, Pp=P * 1e-3, Ps=dbm2w(pin))
            G1, G2 = edfa_fn(EDFA, op), edfa_fn(pmix, op)
            tab[f"{P:g} mW"] = dict(G_consistent_dB=G1, G_mixed_dB=G2, dG_dB=G2 - G1)
        mix.setdefault("table_noASE", {})[f"{pin:g} dBm"] = tab
    orig = {}
    for P in (5.0, 10.0, 20.0, 30.0, 50.0, 100.0):
        op = dict(L=15.0, Pp=P * 1e-3, Ps=dbm2w(-20.0))
        G1, G2 = edfa_fn(EDFA, op), edfa_fn(pmix, op)
        orig[f"{P:g} mW"] = dict(G_consistent_dB=G1, G_mixed_dB=G2, dG_dB=G2 - G1)
    op = dict(L=15.0, Pp=0.1, Ps=dbm2w(-20.0))
    orig["note"] = ("configuration in which the slip was noticed (15 m, -20 dBm, ASE-free model); the consistent-card gain is "
                    "20-31 dB there, at or beyond the ~20 dB range in which [41] reports the ASE-free closed form to be accurate")
    orig["ase_spectral_minus_noASE_at_100mW_dB"] = (edfa_ase_gain(EDFA, op, ase_bins(EDFA, "spectral"))[0] - edfa_fn(EDFA, op))
    mix["original_15m"] = orig
    res["mixing"] = mix
    print(f"mixing done ({time.time() - t0:.0f} s)")
    return res


# ---------------------------------------------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------------------------------------------
LIGHT_BLUE, LIGHT_ORANGE = "#86b6ef", "#f2ab8c"      # +-10 % (tints of slots 1 and 2)
BLUE, ORANGE = style.C[0], style.C[1]                 # +-u
VIOLET = "#6250d6"                                    # ASE-included check in panel d (a hue not used elsewhere)
CAT_SHORT = {"measured": "measured", "derived": "derived", "lit-same-class": "lit-same-class", "lit-other": "lit-other",
             "spec": "spec", "assumed": "assumed", "fitted": "fitted", "config": "config"}


def tornado(ax, rows, xlim, xlabel):
    fitted = [r for r in rows if r["u_rel"] is None]
    rest = sorted([r for r in rows if r["u_rel"] is not None], key=lambda r: -r["importance"])
    order = fitted + rest
    n = len(order)
    hb, off = 0.34, 0.19
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, r in enumerate(order):
        yc = n - 1 - i
        hatch = "////" if r["u_rel"] is None else None
        for d, col in ((r["d_plus10"], LIGHT_BLUE), (r["d_minus10"], LIGHT_ORANGE)):
            ax.barh(yc + off, d, height=hb, color=col, lw=0, hatch=hatch, edgecolor=style.INK2 if hatch else None)
        if r["u_rel"] is None:
            ax.text(0.03 * xlim, yc - off, "u not assigned", fontsize=6.5, color=style.INK2, va="center", ha="left")
            tag = "fitted"
        else:
            for d, col in ((r["d_plus_u"], BLUE), (r["d_minus_u"], ORANGE)):
                ax.barh(yc - off, d, height=hb, color=col, lw=0)
            tag = f"{CAT_SHORT[r['category']]}, {100 * r['u_rel']:.0f} %"
        ax.text(1.02, yc, tag, transform=trans, fontsize=6.8, color=style.INK2, va="center", ha="left")
    if fitted:
        ax.axhline(n - len(fitted) - 0.5, color=style.GRID, lw=0.8)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([r["label"] for r in order], fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-xlim, xlim)
    ax.axvline(0, color=style.INK2, lw=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    return order


def make_figure(res):
    style.apply()
    style.math_in_text_font()      # entry symbols (g*, alpha, P_p, ...) in Arial, like the rest of the text
    fig = plt.figure(figsize=(7.1, 6.0))
    # axes rectangles [left, bottom, width, height] in figure fractions
    ax_a = fig.add_axes([0.115, 0.555, 0.265, 0.355])
    ax_b = fig.add_axes([0.605, 0.555, 0.265, 0.355])
    ax_c = fig.add_axes([0.115, 0.085, 0.265, 0.33])
    ax_d = fig.add_axes([0.595, 0.085, 0.385, 0.33])

    tornado(ax_a, res["edfa"]["rows"], 1.2, "Gain change [dB]")
    tornado(ax_b, res["soa"]["rows"], 10.0, "Chip-gain change [dB]")
    tornado(ax_c, res["raman"]["rows"], 1.0, "Net-gain change [dB]")
    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        style.label(ax, lab, x=-0.40, y=1.03)

    # panel d
    mix = res["mixing"]
    for pin, ls, lab in ((-20.0, "-", "−20 dBm signal"), (-5.0, "--", "−5 dBm signal")):
        cv = mix["curves"][f"{pin:g} dBm"]
        ax_d.semilogx(cv["Pp_mW"], cv["dG_dB"], ls, color=style.INK, lw=1.5, label=lab)
    mk = mix["markers"]["-20 dBm"]
    xs = [float(k.split()[0]) for k in mk]
    ys = [v["dG_dB"] for v in mk.values()]
    ax_d.plot(xs, ys, "o", ms=4.5, color=style.INK, zorder=4, clip_on=False)
    for x, y in zip(xs, ys):
        txt = f"{y:.2f}" if abs(y) < 1 else f"{y:.1f}"
        ax_d.annotate(txt.replace("-", "−") + " dB", (x, y), xytext=(6, -2), textcoords="offset points",
                      fontsize=7, color=style.INK, va="top", ha="left")
    rb = mix["ase_robustness"]["-20 dBm"]
    ax_d.plot([float(k.split()[0]) for k in rb], [v["spectral"]["dG_dB"] for v in rb.values()], "x", ms=5.5, mew=1.2,
              color=VIOLET, zorder=5, clip_on=False, label="−20 dBm, ASE included")
    opd = mix["markers"]["-5 dBm"]["30 mW"]["dG_dB"]
    ax_d.plot([30.0], [opd], "D", ms=5, mfc="white", mec=style.INK, mew=1.2, zorder=5, label="operating point of a)")
    ax_d.set_xlim(3, 300)
    ax_d.set_ylim(-13.5, 0.8)
    ax_d.set_xticks([3, 10, 30, 100, 300])
    ax_d.set_xticklabels(["3", "10", "30", "100", "300"])
    ax_d.set_yticks(np.arange(-12, 1, 2))
    ax_d.axhline(0, color=style.INK2, lw=0.7)
    ax_d.set_xlabel("Pump power [mW]")
    ax_d.set_ylabel("Gain error [dB]")
    ax_d.text(6.5, -10.8, "pump-limited", fontsize=7, style="italic", color=style.INK2, ha="left")
    ax_d.text(120, -2.6, "bleached", fontsize=7, style="italic", color=style.INK2, ha="center")
    ax_d.legend(loc="lower right", fontsize=7, handlelength=2.2, borderaxespad=0.3)
    style.label(ax_d, "d", x=-0.17, y=1.03)

    handles = [Patch(color=LIGHT_BLUE, label="+10 %"), Patch(color=LIGHT_ORANGE, label="−10 %"),
               Patch(color=BLUE, label="+u"), Patch(color=ORANGE, label="−u"),
               Patch(facecolor=LIGHT_BLUE, hatch="////", lw=0, label="fitted entry")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995), fontsize=7.5,
               handlelength=1.6, columnspacing=1.4)
    return fig


def main():
    import sys
    if "--plot-only" in sys.argv:          # redraw Figure 10 from the results file, without computation
        with open(os.path.join(FIG_DIR, "Fig10_sensitivity_results.json"), encoding="utf-8") as f:
            res = json.load(f)
        style.save(make_figure(res), "Fig10_sensitivity", outdir=FIG_DIR)
        print("Fig10_sensitivity redrawn from the results file")
        return
    res = compute()
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "Fig10_sensitivity_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=True, default=float)
    fig = make_figure(res)
    style.save(fig, "Fig10_sensitivity", outdir=FIG_DIR)
    # short ASCII summary
    for m in ("edfa", "soa", "raman"):
        print(f"{m}: baseline {res[m]['baseline_dB']:.2f} dB")
        for r in res[m]["rows"]:
            pu = "n/a" if r["u_rel"] is None else f"{r['d_plus_u']:+.2f}/{r['d_minus_u']:+.2f} (u={100 * r['u_rel']:.0f} %)"
            print(f"   {r['entry']:10s} +-10 %: {r['d_plus10']:+.2f}/{r['d_minus10']:+.2f}   +-u: {pu}")
    print("mixing -20 dBm:", {k: round(v["dG_dB"], 2) for k, v in res["mixing"]["markers"]["-20 dBm"].items()})
    print("mixing  -5 dBm:", {k: round(v["dG_dB"], 2) for k, v in res["mixing"]["markers"]["-5 dBm"].items()})


if __name__ == "__main__":
    main()
