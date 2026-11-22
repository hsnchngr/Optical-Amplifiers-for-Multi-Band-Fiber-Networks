"""Figure 8 (revision 3): quantified architecture trade-offs (model results, ASE only).

a, b) Gain and noise figure of a single-pass and a double-pass EDFA versus signal input power, for the test fiber of card
      EDFA-GD91-T1 at the operating point of card CFG-FIG08-ARCH (10 m, 30 mW forward pump at 1480 nm, Faraday rotator
      mirror reflectance -1 dB, residual pump not reflected; fiber-end reference plane). Lines: lumped ASE band of 30 nm;
      shaded areas: band width varied from 10 to 50 nm. Dashed: single pass through 20 m of the same fiber. Dotted line in a:
      pump-limited bound P_out <= P_in + (lambda_p/lambda_s) P_p. Dashed line in b: noise-figure limit 2 n_sp,min for
      pumping at 1480 nm (card coefficients); dotted: 3 dB (complete inversion).
c)    Equivalent noise figure of an 80 km counter-pumped distributed Raman span followed by a passive element of
      transmission eta and a lumped amplifier (SOA or EDFA), Equation (16):
          F_eq,tot = exp(-l_s L) [F_span + (F_amp/eta - 1)/G_net] = (F_amp/eta + 2 S_ASE/(h nu)) / G_on-off,
      with F_span = (1 + 2 S_ASE/(h nu))/G_net and G_net = G_on-off exp(-l_s L) from raman_physics.profile
      (card RAMAN-G652, whose key alpha_s holds the fiber attenuation l_s; undepleted pump, single signal at
      1550 nm). F_amp refers to the amplifier input port. The legend writes eta, as Equation (16); the result keys
      keep the card names T_base_db and T_variant_db of the two losses (eta = 10^(-T_.../10)).
      Excluded: pump-RIN transfer, double Rayleigh scattering, SOA nonlinear distortion.

Usage:  python fig_sec4.py            compute (about 2.5 min), write the results file, draw the figure
        python fig_sec4.py --replot   redraw the figure from ../figures/Fig08_architecture_tradeoffs_results.json
Outputs: ../figures/Fig08_architecture_tradeoffs.png/.pdf and ../figures/Fig08_architecture_tradeoffs_results.json.
"""
import os
import sys
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.constants import k as kB

import cards
import style
import edfa_multipass as em
from raman_physics import profile
from models import dbm2w, h, c, DB, RAMAN, raman_CR

NAME = "Fig08_architecture_tradeoffs"
CHECK_NUMERICS = True          # grid and tolerance check of the double-pass solver (adds about 20 s)
SHOW_SINGLE_PASS_2L = True     # dashed comparison curve 'single pass, 20 m' in panels a,b
FIGDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
RESULTS = os.path.join(FIGDIR, NAME + "_results.json")
CFG_CARD = cards.load(os.path.join(cards.CARD_DIR, "cfg_fig08_architectures.json"))
CFG = cards.values(CFG_CARD)


def db(x):
    return 10 * np.log10(x)


def nsp_min_mccumber(lp, ls, T):
    """With Equation (1) for pump and signal, n_sp,min = [1 - exp(-h(nu_p - nu_s)/(k T))]^-1, independent of the fiber."""
    return 1.0 / (1.0 - np.exp(-h * c * (1 / lp - 1 / ls) / (kB * T)))


def compute():
    t0 = time.time()
    EDF = cards.values(cards.load("EDFA-GD91-T1"))
    # ------------------------------------------------------------------ a, b) single pass vs double pass
    L1, L2 = CFG["L_edf"], CFG["L_edf_long"]
    PP = CFG["P_pump"] * 1e-3
    RDB = CFG["R_frm"]
    BWS = [CFG["ase_bw_low"], CFG["ase_bw_nominal"], CFG["ase_bw_high"]]
    pins = np.arange(CFG["pin_min"], CFG["pin_max"] + 0.5 * CFG["pin_step"], CFG["pin_step"])
    N1 = 800                                   # grid of the submitted version (dz = 12.5 mm)
    N2 = int(round(N1 * L2 / L1))              # same step size for the 20 m fiber
    cases = {"SP": dict(L=L1, dp=False, N=N1), "DP": dict(L=L1, dp=True, N=N1), "SP_2L": dict(L=L2, dp=False, N=N2)}
    ab = {k: {} for k in cases}
    for key, cs in cases.items():
        for bw in BWS:
            G, NF, SW = [], [], []
            for pin in pins:
                r = em.solve(PP, dbm2w(pin), cs["L"], double_pass=cs["dp"], R_db=RDB, N=cs["N"], ase_bw_nm=bw)
                G.append(float(db(r["G"]))); NF.append(float(db(r["NF"]))); SW.append(int(r["sweeps"]))
            ab[key][f"{bw:g}"] = dict(G_dB=G, NF_dB=NF, sweeps=SW)
        print(f"{key} done ({time.time() - t0:.0f} s)", flush=True)

    def at(key, bw, pin, q):
        return ab[key][f"{bw:g}"][q][int(np.argmin(np.abs(pins - pin)))]

    summary_ab = {}
    for bw in BWS:
        s = {}
        for pin in (-40.0, -30.0, -20.0, -10.0, -5.0, 0.0, 5.0):
            s[f"{pin:g}"] = dict(
                G_SP_dB=at("SP", bw, pin, "G_dB"), G_DP_dB=at("DP", bw, pin, "G_dB"), G_SP2L_dB=at("SP_2L", bw, pin, "G_dB"),
                NF_SP_dB=at("SP", bw, pin, "NF_dB"), NF_DP_dB=at("DP", bw, pin, "NF_dB"), NF_SP2L_dB=at("SP_2L", bw, pin, "NF_dB"),
                DP_advantage_dB=at("DP", bw, pin, "G_dB") - at("SP", bw, pin, "G_dB"),
                DP_NF_penalty_dB=at("DP", bw, pin, "NF_dB") - at("SP", bw, pin, "NF_dB"),
                DP_minus_SP2L_gain_dB=at("DP", bw, pin, "G_dB") - at("SP_2L", bw, pin, "G_dB"),
                DP_minus_SP2L_NF_dB=at("DP", bw, pin, "NF_dB") - at("SP_2L", bw, pin, "NF_dB"))
        summary_ab[f"{bw:g}"] = s

    # noise-figure limit for pumping at 1480 nm: card coefficients, McCumber (any fiber), measured spectra of ref. [124]
    fl = em.nf_floor()
    T_K = EDF["T"]
    lam_p, lam_s = em.P["lam_p"], em.P["lam_s"]
    lam_k, a_k, g_k = cards.perin2019_coefficients()

    def nsp_min_spectra(lp_nm, ls_nm):
        ap, gp = np.interp(lp_nm, lam_k, a_k), np.interp(lp_nm, lam_k, g_k)
        as_, gs = np.interp(ls_nm, lam_k, a_k), np.interp(ls_nm, lam_k, g_k)
        n2 = ap / (ap + gp)
        return gs * n2 / ((as_ + gs) * n2 - as_)

    floor = dict(
        card_EDFA_GD91_T1=dict(lambda_p_nm=1480.0, lambda_s_nm=1550.0, n2_max=float(fl["n2_max"]), nsp_min=float(fl["nsp_min"]),
                               NF_min_dB=float(fl["NF_min_dB"]), basis="alpha and g* of the card at 1480 and 1550 nm"),
        mccumber=dict(T_K=T_K, basis="Equation (1) at pump and signal; independent of the fiber",
                      nsp_min={f"{ls:g}": float(nsp_min_mccumber(1480e-9, ls * 1e-9, T_K)) for ls in (1532.0, 1550.0)},
                      NF_min_dB={f"{ls:g}": float(db(2 * nsp_min_mccumber(1480e-9, ls * 1e-9, T_K))) for ls in (1532.0, 1550.0)}),
        spectra_ref124=dict(basis="digitized alpha and g* of the fiber of ref. [124] (data/perin2019_giles_coefficients.csv)",
                            NF_min_dB={f"{ls:g}": float(db(2 * nsp_min_spectra(1480.0, ls))) for ls in (1532.0, 1550.0)}),
        note="F >= 2 n_sp,min (1 - 1/G) + 1/G for any single- or multi-pass amplifier pumped only at 1480 nm (no background loss)")

    # pump-limited bound: every signal photon added needs one absorbed pump photon
    bound_G_dB = db(1 + (lam_p / lam_s) * PP / dbm2w(pins))

    # mechanism: inversion and noise budget of the double pass (nominal ASE band)
    mech = {}
    nus = c / lam_s
    for pin in (-40.0, -20.0, 0.0):
        s = em.solve(PP, dbm2w(pin), L1, R_db=RDB, profiles=True)
        d = em.solve(PP, dbm2w(pin), L1, double_pass=True, R_db=RDB, profiles=True)
        G1 = d["Pz"][1, -1] / dbm2w(pin)
        F1 = (1 + d["Pz"][3, -1] / (h * nus * d["dnu_ase"])) / G1
        mech[f"{pin:g}"] = dict(
            n2_input_end_SP=float(s["n2"][0]), n2_input_end_DP=float(d["n2"][0]),
            n2_mean_SP=float(s["n2"].mean()), n2_mean_DP=float(d["n2"].mean()),
            residual_pump_fraction_SP=float(s["Pz"][0, -1] / PP), residual_pump_fraction_DP=float(d["Pz"][0, -1] / PP),
            Pout_dBm_SP=float(db(s["Pout"] / 1e-3)), Pout_dBm_DP=float(db(d["Pout"] / 1e-3)),
            Pout_bound_dBm=float(db((dbm2w(pin) + (lam_p / lam_s) * PP) / 1e-3)),
            ASE_out_mW_SP=float(s["Pz"][3, -1] * 1e3), ASE_out_mW_DP=float(d["Pz"][4, 0] * 1e3),
            backward_ASE_at_input_mW_SP=float(s["Pz"][4, 0] * 1e3),
            DP_first_pass_gain_dB=float(db(G1)), DP_first_pass_NF_dB=float(db(F1)),
            NF_SP_dB=float(db(s["NF"])), NF_DP_dB=float(db(d["NF"])),
            NF_penalty_first_pass_dB=float(db(F1) - db(s["NF"])),
            NF_penalty_mirror_and_second_pass_dB=float(db(d["NF"]) - db(F1)))

    numerics = {}
    if CHECK_NUMERICS:
        for pin in (-40.0, 0.0):
            a0 = em.solve(PP, dbm2w(pin), L1, double_pass=True, R_db=RDB)
            a1 = em.solve(PP, dbm2w(pin), L1, double_pass=True, R_db=RDB, N=4 * N1)
            a2 = em.solve(PP, dbm2w(pin), L1, double_pass=True, R_db=RDB, tol=1e-10, max_iter=3000)
            numerics[f"DP_{pin:g}dBm"] = dict(
                dG_grid_x4_dB=float(db(a1["G"]) - db(a0["G"])), dNF_grid_x4_dB=float(db(a1["NF"]) - db(a0["NF"])),
                dG_tol_1em10_dB=float(db(a2["G"]) - db(a0["G"])), dNF_tol_1em10_dB=float(db(a2["NF"]) - db(a0["NF"])),
                sweeps=int(a0["sweeps"]), sweeps_tol_1em10=int(a2["sweeps"]))
        print(f"numerics done ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------------ c) Raman span + passive element eta + amplifier, Eq. (16)
    LS = CFG["L_span"]
    Gon = np.arange(0.0, CFG["Gon_max"] + 0.5 * CFG["Gon_step"], CFG["Gon_step"])
    F_SOA, F_EDFA = 10 ** (CFG["NF_soa"] / 10), 10 ** (CFG["NF_edfa"] / 10)
    ETA0, ETA1 = 10 ** (-CFG["T_base_db"] / 10), 10 ** (-CFG["T_variant_db"] / 10)    # passive transmissions eta

    def span(G_db, L=LS, p=RAMAN):
        r = profile("counter", float(G_db), L=L, p=p)
        return dict(F_span=float(r["NF_span"]), G_net=float(r["Gnet"]), P_pump_W=float(r["Ptot"]),
                    x=float(r["NF_span"] * r["Gnet"] - 1.0), loss=float(np.exp(-p["alpha_s"] * L)))

    def f_eq_tot(F_amp, eta, sp):
        """Equation (16): F_eq,tot = exp(-l_s L) [F_span + (F_amp/eta - 1)/G_net] (sp["loss"] = exp(-l_s L))."""
        return sp["loss"] * (sp["F_span"] + (F_amp / eta - 1.0) / sp["G_net"])

    # result keys: '_T1' = base transmission ETA0 (card T_base_db), '_Tvar' = variant ETA1 (card T_variant_db)
    combos = (("SOA_T1", F_SOA, ETA0), ("EDFA_T1", F_EDFA, ETA0), ("ideal_T1", 1.0, ETA0),
              ("SOA_Tvar", F_SOA, ETA1), ("EDFA_Tvar", F_EDFA, ETA1))
    spans = [span(G) for G in Gon]
    curves = {name: [float(db(f_eq_tot(Fa, eta, sp))) for sp in spans] for name, Fa, eta in combos}
    # consistency of the two forms of Equation (16) and of the realized on-off gain
    eq16_check = max(abs(db(f_eq_tot(F_SOA, ETA1, sp)) - db((F_SOA / ETA1 + sp["x"]) * sp["loss"] / sp["G_net"]))
                     for sp in spans)
    gon_err = max(abs(db(sp["G_net"] / sp["loss"]) - G) for sp, G in zip(spans, Gon))

    summary_c = {}
    for G in (0.0, CFG["G_onoff_exp"], 10.0, 15.0, CFG["Gon_max"]):
        sp = span(G)
        e = {name: float(db(f_eq_tot(Fa, eta, sp))) for name, Fa, eta in combos}
        summary_c[f"{G:g}"] = dict(
            P_pump_W=sp["P_pump_W"], G_net_dB=float(db(sp["G_net"])), NFeq_dB=e,
            SOA_minus_EDFA_T1_dB=e["SOA_T1"] - e["EDFA_T1"], SOA_minus_EDFA_Tvar_dB=e["SOA_Tvar"] - e["EDFA_Tvar"],
            rise_by_Tvar_dB=dict(SOA=e["SOA_Tvar"] - e["SOA_T1"], EDFA=e["EDFA_Tvar"] - e["EDFA_T1"]),
            NF_SOA_ref21_minus_EDFA_T1_dB=float(db(f_eq_tot(10 ** (CFG["NF_soa_ref21"] / 10), ETA0, sp))
                                                - db(f_eq_tot(F_EDFA, ETA0, sp))))
    # span of the experiments of [21]: 100 km with 20 dB loss, other card values unchanged
    p21 = dict(RAMAN)
    p21["alpha_s"] = CFG["span_loss_ref21"] / (CFG["L_span_ref21"] * 1e-3) * DB * 1e-3
    sp21 = span(CFG["G_onoff_exp"], L=CFG["L_span_ref21"], p=p21)
    summary_c["ref21_span_100km_20dB"] = dict(
        G_onoff_dB=CFG["G_onoff_exp"], P_pump_W=sp21["P_pump_W"], G_net_dB=float(db(sp21["G_net"])),
        SOA_minus_EDFA_T1_dB=float(db(f_eq_tot(F_SOA, ETA0, sp21)) - db(f_eq_tot(F_EDFA, ETA0, sp21))))

    return dict(
        figure=NAME, script="fig_sec4.py", solver="edfa_multipass.solve (iterative forward-backward sweeps, RK2)",
        cards=dict(fiber_ab="EDFA-GD91-T1", fiber_c="RAMAN-G652", configuration="CFG-FIG08-ARCH"),
        configuration={k: dict(value=e["value"], unit=e["unit"], category=e["category"], level=e["verification"]["level"])
                       for k, e in CFG_CARD["entries"].items()},
        assumptions_ab=["two-level Giles model, no background loss",
                        "one lumped ASE band centered at 1550 nm with the 1550 nm coefficients",
                        "forward pump at 1480 nm, residual pump not reflected", "FRM reflectance -1 dB for signal and ASE",
                        "gain and noise figure at the fiber ends (circulator, isolator and coupler losses excluded)",
                        "F = (1 + P_ASE/(h nu dnu))/G with the ASE that leaves together with the signal"],
        assumptions_c=["undepleted pump, single signal at 1550 nm, one pump at 1450 nm (card RAMAN-G652, peak C_R)",
                       "Equation (16) with F_amp at the amplifier input port and a passive element of transmission eta "
                       "between span and amplifier",
                       "ASE only: pump-RIN transfer, double Rayleigh scattering and SOA nonlinear distortion excluded"],
        panel_ab=dict(pin_dBm=pins.tolist(), ase_bw_nm=BWS, N_grid=dict(SP=N1, DP=N1, SP_2L=N2), results=ab),
        panel_ab_summary=summary_ab,
        nf_floor_1480nm_pumping=floor,
        pump_bound=dict(formula="P_out <= P_in + (lambda_p/lambda_s) P_p", lambda_p_nm=lam_p * 1e9, lambda_s_nm=lam_s * 1e9,
                        P_p_mW=PP * 1e3, G_bound_dB=bound_G_dB.tolist()),
        double_pass_mechanism_30nm=mech,
        numerics=numerics,
        panel_c=dict(G_onoff_dB=Gon.tolist(), L_span_m=LS, C_R_per_W_km=float(raman_CR() * 1e3),
                     alpha_s_dB_per_km=float(RAMAN["alpha_s"] / DB * 1e3), alpha_p_dB_per_km=float(RAMAN["alpha_p"] / DB * 1e3),
                     P_pump_W=[sp["P_pump_W"] for sp in spans], G_net_dB=[float(db(sp["G_net"])) for sp in spans],
                     two_S_over_hnu=[sp["x"] for sp in spans], NFeq_dB=curves,
                     T_base_dB=CFG["T_base_db"], T_variant_dB=CFG["T_variant_db"],
                     check_eq16_forms_max_diff_dB=float(eq16_check), check_on_off_gain_realized_max_err_dB=float(gon_err)),
        panel_c_summary=summary_c,
        runtime_s=round(time.time() - t0, 1))


def plot(res):
    style.apply()
    style.math_in_text_font()      # math symbols (eta) in Arial, like the rest of the text
    C, INK, INK2 = style.C, style.INK, style.INK2
    COL_SP, COL_DP = C[0], C[1]
    COL_EDFA, COL_SOA = C[0], C[2]
    pins = np.array(res["panel_ab"]["pin_dBm"])
    ab = res["panel_ab"]["results"]
    bws = [f"{b:g}" for b in res["panel_ab"]["ase_bw_nm"]]
    bnom = f"{CFG['ase_bw_nominal']:g}"
    summ = res["panel_ab_summary"]
    bound = np.array(res["pump_bound"]["G_bound_dB"])
    nf_floor = res["nf_floor_1480nm_pumping"]["card_EDFA_GD91_T1"]["NF_min_dB"]

    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.75), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.06)

    def band(ax, key, q, col):
        arr = np.array([ab[key][b][q] for b in bws])
        ax.fill_between(pins, arr.min(0), arr.max(0), color=col, alpha=0.22, lw=0)
        ax.plot(pins, ab[key][bnom][q], color=col, lw=1.6)

    def at(key, pin, q):
        return ab[key][bnom][q][int(np.argmin(np.abs(pins - pin)))]

    # a) gain
    ax = axs[0]
    band(ax, "SP", "G_dB", COL_SP)
    band(ax, "DP", "G_dB", COL_DP)
    if SHOW_SINGLE_PASS_2L:
        ax.plot(pins, ab["SP_2L"][bnom]["G_dB"], color=COL_SP, lw=1.2, ls=(0, (4, 2)))
    ax.plot(pins, bound, color=INK2, lw=1.0, ls=(0, (1.5, 1.5)))
    ax.set_xlim(pins[0], pins[-1]); ax.set_ylim(5, 42)
    ax.set_xticks(np.arange(-40, 6, 10))
    ax.set_xlabel("Input signal power [dBm]"); ax.set_ylabel("Gain [dB]")
    lo = summ[f"{CFG['ase_bw_high']:g}"]["-40"]["DP_advantage_dB"]
    hi = summ[f"{CFG['ase_bw_low']:g}"]["-40"]["DP_advantage_dB"]
    mid = summ[bnom]["-40"]["DP_advantage_dB"]
    x0 = pins[0] + 2.5
    ax.annotate("", xy=(x0, at("DP", x0, "G_dB")), xytext=(x0, at("SP", x0, "G_dB")),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(x0 + 1.2, 25.0, f"+{mid:.1f} dB\n(+{lo:.1f} to +{hi:.1f})", fontsize=7.5, color=INK, va="center")
    xb = -16.0
    yb = float(np.interp(xb, pins, bound))
    bound_label = ax.text(xb, yb + 2.6, "pump bound", fontsize=7.5, color=INK2, rotation_mode="anchor",
                          ha="left", va="bottom")
    handles = [Line2D([], [], color=COL_SP, lw=1.6, label="single pass, 10 m"),
               Line2D([], [], color=COL_DP, lw=1.6, label="double pass, 10 m"),
               Line2D([], [], color=COL_SP, lw=1.2, ls=(0, (4, 2)), label="single pass, 20 m"),
               Patch(facecolor=INK2, alpha=0.25, lw=0, label="ASE band 10–50 nm")]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
    style.label(ax, "a")

    # b) noise figure
    ax = axs[1]
    band(ax, "SP", "NF_dB", COL_SP)
    band(ax, "DP", "NF_dB", COL_DP)
    if SHOW_SINGLE_PASS_2L:
        ax.plot(pins, ab["SP_2L"][bnom]["NF_dB"], color=COL_SP, lw=1.2, ls=(0, (4, 2)))
    ax.axhline(nf_floor, color=INK2, lw=1.0, ls=(0, (4, 2)))
    ax.axhline(db(2.0), color=INK2, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(pins[0] + 1, nf_floor - 0.12, f"limit for 1480 nm pump: {nf_floor:.1f} dB", fontsize=7.5, color=INK2, va="top")
    ax.text(pins[0] + 1, db(2.0) - 0.12, "3 dB: complete inversion", fontsize=7.5, color=INK2, va="top")
    ax.set_xlim(pins[0], pins[-1]); ax.set_ylim(2.3, 9.2)
    ax.set_xticks(np.arange(-40, 6, 10))
    ax.set_xlabel("Input signal power [dBm]"); ax.set_ylabel("Noise figure [dB]")
    style.label(ax, "b")

    # c) Raman span + amplifier
    ax = axs[2]
    pc, sc = res["panel_c"], res["panel_c_summary"]
    Gon, cv = np.array(pc["G_onoff_dB"]), pc["NFeq_dB"]
    G_exp, G_hi = CFG["G_onoff_exp"], CFG["Gon_max"]
    eta0, eta1 = 10 ** (-CFG["T_base_db"] / 10), 10 ** (-CFG["T_variant_db"] / 10)    # transmission eta, Eq. (16)
    ax.plot(Gon, cv["SOA_T1"], color=COL_SOA, lw=1.6)
    ax.plot(Gon, cv["EDFA_T1"], color=COL_EDFA, lw=1.6)
    ax.plot(Gon, cv["SOA_Tvar"], color=COL_SOA, lw=1.2, ls=(0, (4, 2)))
    ax.plot(Gon, cv["EDFA_Tvar"], color=COL_EDFA, lw=1.2, ls=(0, (4, 2)))
    ax.plot(Gon, cv["ideal_T1"], color=INK, lw=1.0, ls=(0, (1, 1.5)))
    ax.plot([G_exp, G_exp], [-2.6, 4.6], color=INK2, lw=0.8)      # stops below the legend
    e9, e20 = sc[f"{G_exp:g}"]["NFeq_dB"], sc[f"{G_hi:g}"]["NFeq_dB"]
    d_exp, d_hi = sc[f"{G_exp:g}"]["SOA_minus_EDFA_T1_dB"], sc[f"{G_hi:g}"]["SOA_minus_EDFA_T1_dB"]
    xa = G_exp + 0.35
    ax.annotate("", xy=(xa, e9["SOA_T1"]), xytext=(xa, e9["EDFA_T1"]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax.annotate(f"SOA − EDFA:\n{d_exp:.2f} dB", xy=(xa + 0.1, 0.5 * (e9["SOA_T1"] + e9["EDFA_T1"])),
                xytext=(G_exp + 1.6, 3.3), fontsize=7.5, color=INK, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=1, shrinkB=1))
    ax.text(G_exp - 0.3, -2.4, "≈9 dB in\nexperiments", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.annotate(f"{d_hi:.2f} dB", xy=(G_hi - 0.25, 0.5 * (e20["SOA_T1"] + e20["EDFA_T1"])), xytext=(G_hi - 3.4, 0.9),
                fontsize=7.5, color=INK, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=1, shrinkB=1))
    ax.set_xlim(0, G_hi); ax.set_ylim(-2.6, 8.4)
    ax.set_xticks(np.arange(0, G_hi + 1, 5))
    ax.set_xlabel("Raman on-off gain [dB]"); ax.set_ylabel("Equivalent noise figure [dB]")
    handles = [Line2D([], [], color=COL_SOA, lw=1.6, label=f"SOA, NF {CFG['NF_soa']:g} dB"),
               Line2D([], [], color=COL_EDFA, lw=1.6, label=f"EDFA, NF {CFG['NF_edfa']:g} dB"),
               Line2D([], [], color=INK, lw=1.0, ls=(0, (1, 1.5)), label="span alone"),
               Line2D([], [], color=INK2, lw=1.6, label=f"$\\eta$ = {eta0:g}"),
               Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label=f"$\\eta$ = {eta1:.2f}")]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, handlelength=1.3, handletextpad=0.5, borderaxespad=0.1,
              labelspacing=0.3)
    style.label(ax, "c")

    # align the 'pump bound' label with the bound line in screen coordinates (after the layout is fixed)
    fig.canvas.draw()
    p1 = axs[0].transData.transform((-20.0, float(np.interp(-20.0, pins, bound))))
    p2 = axs[0].transData.transform((-5.0, float(np.interp(-5.0, pins, bound))))
    bound_label.set_rotation(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))
    style.save(fig, NAME, outdir=FIGDIR)
    return fig


def report(res):
    for b in res["panel_ab"]["ase_bw_nm"]:
        s = res["panel_ab_summary"][f"{b:g}"]
        print(f"ASE band {b:g} nm: DP advantage {s['-40']['DP_advantage_dB']:.2f} / {s['-20']['DP_advantage_dB']:.2f} / "
              f"{s['0']['DP_advantage_dB']:.2f} dB, NF penalty {s['-40']['DP_NF_penalty_dB']:.2f} / "
              f"{s['-20']['DP_NF_penalty_dB']:.2f} / {s['0']['DP_NF_penalty_dB']:.2f} dB at -40 / -20 / 0 dBm")
    fl = res["nf_floor_1480nm_pumping"]
    print("NF limit (1480 nm pump):", round(fl["card_EDFA_GD91_T1"]["NF_min_dB"], 2), "dB (card, 1550 nm); McCumber",
          {k: round(v, 2) for k, v in fl["mccumber"]["NF_min_dB"].items()}, "; spectra [124]",
          {k: round(v, 2) for k, v in fl["spectra_ref124"]["NF_min_dB"].items()})
    sc = res["panel_c_summary"]
    for G in (f"{CFG['G_onoff_exp']:g}", "10", "15", f"{CFG['Gon_max']:g}"):
        s = sc[G]
        print(f"G_on-off {G} dB: pump {s['P_pump_W']:.3f} W, G_net {s['G_net_dB']:.2f} dB, SOA-EDFA {s['SOA_minus_EDFA_T1_dB']:.3f} dB"
              f" (eta = 1), {s['SOA_minus_EDFA_Tvar_dB']:.3f} dB ({CFG['T_variant_db']:g} dB loss)")
    print("100 km / 20 dB span at", CFG["G_onoff_exp"], "dB on-off gain:",
          round(sc["ref21_span_100km_20dB"]["SOA_minus_EDFA_T1_dB"], 3), "dB")


if __name__ == "__main__":
    if "--replot" in sys.argv and os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            res = json.load(f)
    else:
        res = compute()
        os.makedirs(FIGDIR, exist_ok=True)
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    plot(res)
    report(res)
