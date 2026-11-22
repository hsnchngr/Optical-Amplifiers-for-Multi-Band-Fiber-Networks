"""Figure 9 (Fig09_hybrid_design) and Figure 13 (Fig13_pump_budget) of revision 3.

Hybrid distributed-Raman + EDFA span: 80 km G.652 fiber, 40 C-band channels (1530-1565 nm, 0 dBm each at the fiber
input), transparent span (mean span net gain 0 dB). Model illustration, not a hardware result.
  Fig09: reference design (Raman pump budget 0.5 W, EDFA mean inversion free). (a) gain budget along wavelength,
         (b) span net gain: first-order LP design against the full multi-wavelength Raman model, and the design
         re-optimized with the full model in the loop, (c) pump allocation.
  Fig13: minimum peak-to-peak ripple at a fixed mean span net gain of 0 dB against the Raman pump budget for three
         families (EDFA inversion free; EDFA mean gain fixed at 10.7 dB or 6.7 dB), and the model-form error of the
         first-order design for the free family.
The file name is kept from revision 2 (it produced Figs. 10 and 15 of the submitted version).
Outputs: ../figures/Fig09_hybrid_design.{png,pdf}, ../figures/Fig13_pump_budget.{png,pdf} and their *_results.json.
Run time about 15 min (sequential linear programming with the full boundary-value model in the loop).
"""
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import style
import hybrid as hy

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
C, INK, INK2 = style.C, style.INK, style.INK2

LAM = hy.channel_plan()                       # 40 channels, 1530-1565 nm
P_CH_DBM = hy.HC["P_ch_in"]                   # 0 dBm per channel at plane A
BUDGET_REF = 0.5                              # W, reference design of Fig09
SHARES = (6.0, 10.0)                          # mean Raman on-off gains that the fixed-inversion families require [dB]
BUDGETS = np.round(np.arange(0.0, 1.0001, 0.025), 4)


def r(x, n=4):
    """Round floats/arrays for the JSON files."""
    if isinstance(x, dict):
        return {k: r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    if isinstance(x, np.ndarray):
        return [r(v, n) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        if x != 0.0 and abs(x) < 10.0 ** (1 - n):
            return float(f"{float(x):.3g}")          # keep small residuals visible
        return round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    return x


def stats(v):
    v = np.asarray(v, float)
    return dict(mean=v.mean(), min=v.min(), max=v.max())


def full_eval(lam_p, P, nbar, terms=("dep", "pp", "ss"), p_ch_dbm=P_CH_DBM):
    """Full-model span net gain, on-off gain and first-order comparison for one pump setting."""
    p_in = 1e-3 * 10 ** (p_ch_dbm / 10) * np.ones(LAM.size)
    sol = hy.raman_full(LAM, p_in, lam_p, P, terms=terms)
    p_off = hy.signals_only(LAM, p_in, ss=("ss" in terms))
    fo = hy.first_order(LAM, lam_p, P, nbar)
    g_net = 10 * np.log10(sol["Ps_out"] / p_in) - hy.L_PASSIVE + fo["G_edfa"]
    err = g_net - fo["G_net"]
    return dict(sol=sol, fo=fo, G_net=g_net, G_onoff=10 * np.log10(sol["Ps_out"] / p_off),
                ripple=float(g_net.max() - g_net.min()), mean_net=float(g_net.mean()),
                err=err, err_mean=float(err.mean()), err_rms=float(np.sqrt(np.mean(err ** 2))),
                err_max=float(np.max(np.abs(err))), err_pp=float(err.max() - err.min()))


def describe(lam_p, P, nbar, ev, budget):
    fo = ev["fo"]
    return dict(budget_W=budget, pump_wavelengths_nm=lam_p, pump_powers_mW=np.asarray(P) * 1e3,
                total_pump_mW=float(np.sum(P)) * 1e3, nbar=nbar, edfa_gain_dB=stats(fo["G_edfa"]),
                first_order=dict(net_mean=fo["mean_net"], net_ripple_pp=fo["ripple"], onoff=stats(fo["G_onoff"])),
                full_model=dict(net_mean=ev["mean_net"], net_ripple_pp=ev["ripple"], net=stats(ev["G_net"]),
                                onoff=stats(ev["G_onoff"]), residual_pump_power_at_z0_mW=ev["sol"]["Pp_in"] * 1e3,
                                bvp_status=ev["sol"]["status"], photon_balance_rel=ev["sol"]["photon_balance_rel"]),
                model_form_error_full_minus_first_order_dB=dict(mean=ev["err_mean"], rms=ev["err_rms"],
                                                               max_abs=ev["err_max"], peak_to_peak=ev["err_pp"]))


# =====================================================================================================================
# Figure 9: reference design
# =====================================================================================================================
def compute_fig09():
    t0 = time.time()
    res = {}
    d_fo = hy.design_lp(LAM, BUDGET_REF)                                      # layer L3, first-order model
    ev_fo = full_eval(d_fo["lam_p"], d_fo["P"], d_fo["nbar"])                 # layer L2 check
    d_c, ev_c, hist_c, conv_c = hy.design_corrected(LAM, BUDGET_REF, p_ch_dbm=P_CH_DBM)
    d_s = hy.design_slp(LAM, BUDGET_REF, [d_fo, d_c if conv_c else None], p_ch_dbm=P_CH_DBM)
    ev_s = full_eval(d_s["lam_p"], d_s["P"], d_s["nbar"])
    res["first_order_design"] = describe(d_fo["lam_p"], d_fo["P"], d_fo["nbar"], ev_fo, BUDGET_REF)
    res["residual_corrected_design_initializer"] = dict(
        converged=conv_c, iterations=len(hist_c), pump_powers_mW=d_c["P"] * 1e3, nbar=d_c["nbar"],
        full_model_ripple=ev_c["ripple"], full_model_mean=ev_c["mean_net"])
    res["full_model_optimized_design"] = describe(d_s["lam_p"], d_s["P"], d_s["nbar"], ev_s, BUDGET_REF)
    res["full_model_optimized_design"]["slp"] = dict(iterations=d_s["iterations"], full_model_solves=d_s["n_full_model_solves"],
                                                     mean_deviation_dB=d_s["mean_dev"], history=d_s["history"])
    # breakdown of the model-form error at the first-order pumps
    brk = {}
    for name, terms in (("pump_to_signal_gain_only", ()), ("plus_signal_induced_pump_depletion", ("dep",)),
                        ("plus_pump_pump_SRS", ("dep", "pp")), ("plus_signal_signal_SRS_full", ("dep", "pp", "ss"))):
        e = full_eval(d_fo["lam_p"], d_fo["P"], d_fo["nbar"], terms=terms)
        brk[name] = dict(err_mean=e["err_mean"], err_rms=e["err_rms"], err_max_abs=e["err_max"],
                         err_pp=e["err_pp"], net_ripple_pp=e["ripple"], net_mean=e["mean_net"])
    res["model_form_error_breakdown_at_first_order_pumps"] = brk
    # dependence of the model-form error on the WDM load (same first-order pumps)
    res["model_form_error_vs_load"] = {}
    for pdbm in (-3.0, 0.0, 3.0):
        e = full_eval(d_fo["lam_p"], d_fo["P"], d_fo["nbar"], p_ch_dbm=pdbm)
        key = f"{pdbm:+.0f} dBm per channel ({pdbm + 10 * np.log10(LAM.size):.1f} dBm total)"
        res["model_form_error_vs_load"][key] = dict(full_ripple=e["ripple"], full_mean=e["mean_net"],
                                                    err_mean=e["err_mean"], err_rms=e["err_rms"], err_max_abs=e["err_max"])
    # code verification of the boundary-value solver
    ver = {}
    for pp in (0.1, 0.3, 0.5):
        o = hy.raman_full([1550.0], [1e-7], [1450.0], [pp], terms=())
        g_num = 10 * np.log10(o["Ps_out"][0] / 1e-7) + hy.ALPHA_S_DB * hy.L_SPAN
        ver[f"single_pump_{pp:.1f}W_onoff_bvp_minus_closed_form_dB"] = g_num - hy.onoff_matrix([1550.0], [1450.0])[0, 0] * pp
    ver["multi_pump_gain_terms_only_max_abs_dev_from_first_order_dB"] = brk["pump_to_signal_gain_only"]["err_max_abs"]
    p_in = 1e-3 * 10 ** (P_CH_DBM / 10) * np.ones(LAM.size)
    a = hy.raman_full(LAM, p_in, d_fo["lam_p"], d_fo["P"], n_nodes=321, tol=1e-7)
    b = hy.raman_full(LAM, p_in, d_fo["lam_p"], d_fo["P"], n_nodes=1281, tol=1e-10)
    ver["refinement_tol1e-7_vs_tol1e-10_max_abs_dB"] = float(np.max(np.abs(10 * np.log10(a["Ps_out"] / b["Ps_out"]))))
    ver["photon_number_balance_rel_residual"] = a["photon_balance_rel"]
    res["solver_verification"] = ver
    # channel sampling: first-order ripple on a 0.05 nm grid against the 40-channel grid
    lf = np.linspace(1530, 1565, 701)
    gf = hy.onoff_matrix(lf, d_fo["lam_p"]) @ d_fo["P"] - d_fo["loss_db"] + hy.edfa_gain_db(lf, d_fo["nbar"])
    res["first_order_design_ripple_on_0p05nm_grid_dB"] = float(gf.max() - gf.min())
    # EDFA checks: 980 nm pump estimate and indicative spontaneous-emission factor
    p_edf_in = ev_s["sol"]["Ps_out"] * 10 ** (-hy.L_PASSIVE / 10)
    nb0 = hy.nbar_for_raman_share(LAM, 0.0)
    res["edfa_check"] = dict(
        note="ASE-free two-level estimate with card EDFA-PERIN2019, all passive loss placed before the EDF (lower bound)",
        edf_input_total_dBm=10 * np.log10(p_edf_in.sum() / 1e-3),
        edf_output_total_dBm=10 * np.log10(np.sum(p_edf_in * 10 ** (ev_s["fo"]["G_edfa"] / 10)) / 1e-3),
        pump_980_mW=hy.edfa_pump_980(d_s["nbar"], LAM, p_edf_in) * 1e3,
        nsp_design={f"{x:.0f} nm": float(hy.nsp_two_level(np.array([x]), d_s["nbar"])[0]) for x in (1530, 1545, 1565)},
        nbar_edfa_only=nb0,
        nsp_edfa_only={f"{x:.0f} nm": float(hy.nsp_two_level(np.array([x]), nb0)[0]) for x in (1530, 1545, 1565)})
    # double Rayleigh scattering estimate (not part of the design model)
    sol = ev_s["sol"]
    gz = sol["P"][:sol["ns"]] / sol["P"][:sol["ns"], :1]
    drb = np.array([hy.drb_crosstalk(sol["z"], gz[i]) for i in range(sol["ns"])])
    single = {}
    for pp in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        if pp == 0.0:
            z = np.linspace(0, hy.L_SPAN, 1601)
            g = np.exp(-hy.ALPHA_S_DB / hy.DB * z)
            gon = 0.0
        else:
            o = hy.raman_full([1550.0], [1e-6], [1450.0], [pp], terms=())
            z, g = o["z"], o["P"][0] / o["P"][0, 0]
            gon = 10 * np.log10(g[-1]) + hy.ALPHA_S_DB * hy.L_SPAN
        single[f"{gon:.1f} dB on-off"] = hy.drb_crosstalk(z, g)
    res["drb_estimate"] = dict(note="double-scattering integral, r_R = %.3g 1/km (card HYBRID-C80, UNVERIFIED)" % hy.R_RAYLEIGH,
                               design_channels_dB=stats(drb), single_pump_1450nm_signal_1550nm_dB=single)
    # sensitivity of the reference design
    sens = {}
    d_alt = hy.design_lp(LAM, BUDGET_REF, scaling="gnpy_area")
    s_alt = hy.design_slp(LAM, BUDGET_REF, [d_alt], scaling="gnpy_area")
    sens["raman_scaling_gnpy_area"] = dict(first_order_ripple=d_alt["ripple"], first_order_P_mW=d_alt["P"] * 1e3,
                                           full_optimized_ripple=s_alt["ripple"], full_optimized_P_mW=s_alt["P"] * 1e3,
                                           full_optimized_nbar=s_alt["nbar"])
    for lpas in (1.0, 2.0):
        dd = hy.design_lp(LAM, BUDGET_REF, l_passive=lpas)
        ss = hy.design_slp(LAM, BUDGET_REF, [dd], l_passive=lpas)
        sens[f"L_passive_{lpas:.1f}dB"] = dict(first_order_ripple=dd["ripple"], full_optimized_ripple=ss["ripple"],
                                               full_optimized_nbar=ss["nbar"], full_optimized_P_mW=ss["P"] * 1e3)
    # coefficient scale of card EDFA-PERIN2019 identified in the end-to-end case (Figure 12), if its results exist
    try:
        with open(os.path.join(FIG_DIR, "Fig12_end_to_end_results.json"), encoding="utf-8") as fh:
            cal = json.load(fh)["summary"]["calibrated"]
        s = float(dict(zip(cal["parameters"], cal["values"]))["coef_scale"])
        dd = hy.design_lp(LAM, BUDGET_REF, l_edf=s * hy.L_EDF)
        ss = hy.design_slp(LAM, BUDGET_REF, [dd], l_edf=s * hy.L_EDF)
        sens["edf_coefficient_scale_from_Fig12_calibration"] = dict(
            coef_scale=s, first_order_ripple=dd["ripple"], full_optimized_ripple=ss["ripple"],
            full_optimized_nbar=ss["nbar"], full_optimized_P_mW=ss["P"] * 1e3,
            note="alpha and g* scaled by coef_scale (equivalent to an EDF length of coef_scale x 8 m)")
    except (OSError, KeyError, ValueError) as exc:
        sens["edf_coefficient_scale_from_Fig12_calibration"] = f"not evaluated ({exc.__class__.__name__})"
    res["sensitivity_reference_design"] = sens
    # what sets the ripple floor (first-order LP only, EDFA inversion free)
    flo = {}
    grid5 = np.arange(1415.0, 1485.1, 5.0)
    lam_36 = hy.channel_plan(n=34, lam_min=1536.0, lam_max=1565.0)
    for bud in (0.5, 1.0):
        d5 = hy.design_lp(LAM, bud, lam_p=grid5)
        d36 = hy.design_lp(lam_36, bud)
        flo[f"{bud:.1f}W"] = dict(
            reference_7_pumps_1420_1480nm=hy.design_lp(LAM, bud)["ripple"],
            pumps_15_candidates_1415_1485nm_5nm_steps=d5["ripple"],
            pumps_15_candidates_used_mW={f"{lp:.0f}": p * 1e3 for lp, p in zip(grid5, d5["P"]) if p > 1e-4},
            band_1536_1565nm_34_channels=d36["ripple"])
    res["first_order_floor_checks"] = dict(note="first-order LP, EDFA inversion free, other conditions as the reference design", **flo)
    res["runtime_s"] = time.time() - t0
    return res, d_fo, ev_fo, d_s, ev_s


def plot_fig09(d_fo, ev_fo, d_s, ev_s):
    """d_fo, ev_fo, d_s, ev_s may be design/evaluation objects or plain dicts read back from the results JSON."""
    style.apply()
    style.math_in_text_font()      # math symbols in Arial, like the rest of the text
    fig = plt.figure(figsize=(7.1, 5.1))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.2, 1.0], hspace=0.42, wspace=1.3)
    ax = fig.add_subplot(gs[0, :])
    bx = fig.add_subplot(gs[1, :3])
    cx = fig.add_subplot(gs[1, 3:])
    lam = LAM
    fo = ev_s["fo"]
    A = hy.onoff_matrix(lam, d_s["lam_p"])
    # (a) gain budget of the full-model-optimized design
    for j, pw in enumerate(d_s["P"]):
        if pw > 1e-4:
            ax.plot(lam, A[:, j] * pw, color=INK2, lw=0.6, ls=":")
    ax.plot([], [], color=INK2, lw=0.6, ls=":", label="single pumps, first order")
    ax.plot(lam, fo["G_edfa"], color=C[0], label=r"EDFA gain ($\bar{n}_2$ = %.3f)" % d_s["nbar"])
    ax.plot(lam, fo["G_onoff"], color=C[2], ls="--", label="Raman on-off gain, first order")
    ax.plot(lam, ev_s["G_onoff"], color=C[2], label="Raman on-off gain, full model")
    loss = hy.fixed_loss_db()
    ax.axhline(-loss, color=INK2, lw=1.2, ls="-.", label="fiber + passive loss")
    ax.plot(lam, ev_s["G_net"], color=C[6], lw=2.0, label="span net gain, full model")
    ax.set_xlim(1529, 1566)
    ax.set_ylim(-19, 13)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Gain [dB]")
    ax.text(1565.5, -loss + 0.6, r"$-l_\mathrm{s} L - L_\mathrm{pas}$ = %s dB (%.1f + %.1f dB)" % (
        f"−{loss:.1f}", hy.ALPHA_S_DB * hy.L_SPAN, hy.L_PASSIVE), ha="right", va="bottom", fontsize=7, color=INK2)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.40), ncol=2, fontsize=7, handlelength=2.4, columnspacing=1.2)
    style.label(ax, "a", x=-0.075)
    # (b) span net gain: first-order design against the full model, and the full-model-optimized design
    bx.axhline(0.0, color=INK2, lw=0.6)
    bx.plot(lam, ev_s["G_net"], color=C[6], lw=1.3, marker="o", ms=2.6, zorder=2,
            label="full-model-optimized design, full model (%.2f dB)" % ev_s["ripple"])
    bx.plot(lam, ev_fo["G_net"], color=C[7], lw=1.4, zorder=3,
            label="first-order design, full model (%.2f dB)" % ev_fo["ripple"])
    bx.plot(lam, d_fo["G_net"], color=INK, ls=(0, (4, 2)), lw=1.1, zorder=4,
            label="first-order design, first-order model (%.2f dB)" % d_fo["ripple"])
    bx.set_xlim(1529, 1566)
    bx.set_ylim(-3.2, 4.9)                      # headroom: the legend stays above the curves (max 2.2 dB near 1558 nm)
    bx.set_xlabel("Wavelength [nm]")
    bx.set_ylabel("Span net gain [dB]")
    bx.legend(loc="upper left", fontsize=6.4, handlelength=2.2)
    style.label(bx, "b", x=-0.13)
    # (c) pump allocation
    x = np.arange(len(d_s["lam_p"]))
    w = 0.38
    b1 = cx.bar(x - w / 2, d_fo["P"] * 1e3, w, color="white", edgecolor=C[7], hatch="////", lw=0.8,
                label="first-order design")
    b2 = cx.bar(x + w / 2, d_s["P"] * 1e3, w, color=C[6], label="full-model-optimized")
    for bars in (b1, b2):
        for rect in bars:
            hgt = rect.get_height()
            if hgt > 1.0:
                cx.text(rect.get_x() + rect.get_width() / 2, hgt + 4, f"{hgt:.0f}", ha="center", va="bottom",
                        fontsize=6.4, color=INK, rotation=90)
    cx.set_xticks(x)
    cx.set_xticklabels([f"{v:.0f}" for v in d_s["lam_p"]], rotation=90)
    cx.set_xlabel("Pump wavelength [nm]")
    cx.set_ylabel("Pump power [mW]")
    cx.set_ylim(0, 300)
    cx.legend(loc="upper right", fontsize=6.4)
    style.label(cx, "c", x=-0.33)
    style.save(fig, "Fig09_hybrid_design", outdir=FIG_DIR)
    plt.close(fig)


# =====================================================================================================================
# Figure 13: minimum ripple against the Raman pump budget
# =====================================================================================================================
def threshold_fo(nbar, lo=0.0, hi=1.0):
    """Smallest budget for which the first-order LP reaches the mean (bisection to 1e-4 W)."""
    if hy.design_lp(LAM, hi, nbar=nbar) is None:
        return np.nan
    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        if hy.design_lp(LAM, mid, nbar=nbar, tiebreak=False) is None:
            lo = mid
        else:
            hi = mid
    return hi


def threshold_full(nbar, lo, hi, start):
    """Smallest budget for which the full-model optimization reaches the mean (bisection to 1e-3 W)."""
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        d0 = hy.design_lp(LAM, mid, nbar=nbar)
        ds = hy.design_slp(LAM, mid, [d0, start], nbar=nbar, p_ch_dbm=P_CH_DBM)
        if ds["feasible"]:
            hi = mid
        else:
            lo = mid
    return hi


def sweep(f):
    rows, prev = [], None
    for b in BUDGETS:
        d0 = hy.design_lp(LAM, b, nbar=f["nbar"])
        row = dict(budget=b, feasible_first_order=d0 is not None)
        if d0 is not None:
            e0 = full_eval(d0["lam_p"], d0["P"], d0["nbar"])
            row.update(fo_ripple=d0["ripple"], fo_pump_used_W=d0["P_total"], fo_nbar=d0["nbar"],
                       fo_onoff_mean=d0["G_onoff"].mean(), fo_onoff_max=d0["G_onoff"].max(),
                       fo_pumps_full_model_ripple=e0["ripple"], fo_pumps_full_model_mean=e0["mean_net"])
        if d0 is None and prev is None:
            row.update(full_feasible=False)
            rows.append(row)
            continue
        ds = hy.design_slp(LAM, b, [d0, prev], nbar=f["nbar"], p_ch_dbm=P_CH_DBM)
        row.update(full_feasible=ds["feasible"], full_mean_dev=ds["mean_dev"],
                   full_ripple=(ds["ripple"] if ds["feasible"] else np.nan), full_pump_used_W=ds["P_total"],
                   full_nbar=ds["nbar"], full_P_mW=ds["P"] * 1e3, full_onoff_mean=ds["G_onoff"].mean(),
                   full_onoff_max=ds["G_onoff"].max(), slp_iterations=ds["iterations"],
                   full_model_solves=ds["n_full_model_solves"])
        if ds["feasible"]:
            prev = ds
        rows.append(row)
    return rows


def compute_fig13():
    t0 = time.time()
    fams = [dict(key="free", nbar=None, label="EDFA inversion free",
                 description="EDFA mean inversion free, %.2f <= nbar <= %.2f" % hy.NBAR_BOUNDS)]
    for s in SHARES:
        nb = hy.nbar_for_raman_share(LAM, s)
        g_ed = hy.edfa_gain_db(LAM, nb).mean()
        fams.append(dict(key=f"edfa_{g_ed:.1f}dB", nbar=nb, share=s, edfa_mean=g_ed,
                         label="EDFA gain %.1f dB" % g_ed,
                         description="EDFA mean gain fixed at %.2f dB (nbar = %.4f); Raman mean on-off gain %.0f dB (first order)" % (g_ed, nb, s)))
    out = dict(budgets_W=BUDGETS, families={})
    for f in fams:
        rows = sweep(f)
        f_out = dict(label=f["label"], description=f["description"], nbar=f["nbar"], rows=rows)
        ok = [rw for rw in rows if rw.get("full_feasible")]
        f_out["full_model_floor_ripple_dB"] = min(rw["full_ripple"] for rw in ok)
        f_out["first_order_floor_ripple_dB"] = min(rw["fo_ripple"] for rw in rows if rw["feasible_first_order"])
        if f["nbar"] is not None:
            f_out["mean_raman_onoff_required_first_order_dB"] = f["share"]
            f_out["edfa_mean_gain_dB"] = f["edfa_mean"]
            f_out["min_budget_first_order_W"] = threshold_fo(f["nbar"])
            first = ok[0]
            lo = max([rw["budget"] for rw in rows if not rw.get("full_feasible") and rw["budget"] < first["budget"]], default=0.0)
            start = dict(P=np.asarray(first["full_P_mW"]) / 1e3, nbar=f["nbar"])
            f_out["min_budget_full_model_W"] = threshold_full(f["nbar"], lo, first["budget"], start)
        out["families"][f["key"]] = f_out
    # sensitivity of the floors at a 1 W budget
    sens = {}
    for name, kw in (("reference", {}), ("on_off_cap_12dB", dict(g_cap=12.0)), ("per_pump_max_0p25W", dict(p_max=0.25)),
                     ("four_pumps_1425_1440_1455_1465nm", dict(lam_p=np.array([1425.0, 1440.0, 1455.0, 1465.0])))):
        sens[name] = {}
        for f in fams:
            dd = hy.design_lp(LAM, 1.0, nbar=f["nbar"], **kw)
            if dd is None:
                sens[name][f["key"]] = "infeasible (first order)"
                continue
            ss = hy.design_slp(LAM, 1.0, [dd], nbar=f["nbar"], p_ch_dbm=P_CH_DBM, **kw)
            sens[name][f["key"]] = dict(first_order_ripple=dd["ripple"], first_order_pump_used_W=dd["P_total"],
                                        full_feasible=ss["feasible"], full_ripple=ss["ripple"],
                                        full_pump_used_W=ss["P_total"], full_onoff_max=ss["G_onoff"].max())
    out["sensitivity_at_1W"] = sens
    out["runtime_s"] = time.time() - t0
    return out, fams


def plot_fig13(res, fams):
    style.apply()
    style.math_in_text_font()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw=dict(wspace=0.32))
    cols = [C[0], C[1], C[2]]
    for k, f in enumerate(fams):
        fr = res["families"][f["key"]]
        rows = fr["rows"]
        b = np.array([rw["budget"] for rw in rows])
        y = np.array([rw.get("full_ripple", np.nan) if rw.get("full_feasible") else np.nan for rw in rows], float)
        yf = np.array([rw.get("fo_ripple", np.nan) for rw in rows], float)
        ax.plot(b, yf, color=cols[k], lw=0.9, ls=(0, (4, 2)))
        ax.plot(b, y, color=cols[k], marker="o", ms=2.3, lw=1.5)
        if f["nbar"] is not None:
            thr = fr["min_budget_full_model_W"]
            ax.axvspan(0, thr, color=cols[k], alpha=0.08, lw=0)
            ax.axvline(thr, color=cols[k], lw=0.8, ls=":")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 4.0)
    ax.set_xlabel("Raman pump budget [W]")
    ax.set_ylabel("Minimum ripple, peak-to-peak [dB]")
    handles = [Line2D([], [], color=cols[k], lw=1.5, marker="o", ms=2.3, label=f["label"]) for k, f in enumerate(fams)]
    handles += [Line2D([], [], color=INK2, lw=1.5, label="full model in the loop"),
                Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 2)), label="first-order LP")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.3, handlelength=2.0)
    style.label(ax, "a", x=-0.17)
    # (b) model-form error of the first-order design, free family
    rows = res["families"]["free"]["rows"]
    b = np.array([rw["budget"] for rw in rows])
    bx.plot(b, [rw["fo_pumps_full_model_ripple"] for rw in rows], color=C[7], lw=1.4,
            label="first-order design, full model")
    bx.plot(b, [rw["full_ripple"] for rw in rows], color=C[6], lw=1.3, marker="o", ms=2.3,
            label="full-model-optimized design")
    bx.plot(b, [rw["fo_ripple"] for rw in rows], color=INK, ls=(0, (4, 2)), lw=1.1,
            label="first-order design, first-order model")
    bx.set_xlim(0, 1.0)
    bx.set_ylim(0, 6.5)
    bx.set_xlabel("Raman pump budget [W]")
    bx.set_ylabel("Ripple, peak-to-peak [dB]")
    bx.legend(loc="lower right", bbox_to_anchor=(1.0, 0.17), fontsize=6.4)
    style.label(bx, "b", x=-0.17)
    style.save(fig, "Fig13_pump_budget", outdir=FIG_DIR)
    plt.close(fig)


CONDITIONS = dict(
    span="80 km G.652 fiber (card RAMAN-G652: alpha_s 0.19 dB/km, alpha_p 0.25 dB/km, C_R 0.406 1/(W km) at 1450 nm)",
    raman_profile="SSMF gain-efficiency profile distributed with GNPy, scaled as in card HYBRID-C80 (raman_scaling)",
    edfa="8 m EDF of ref. [124] (card EDFA-PERIN2019), average-inversion relation, internal EDF gain",
    passive_loss_dB=hy.L_PASSIVE,
    channels="40 channels 1530-1565 nm, equally spaced in frequency (112.4 GHz)",
    launch="0 dBm per channel at the fiber input (16.0 dBm total)", target="mean span net gain 0 dB (transparent span)",
    pump_grid_nm=hy.PUMP_GRID, per_pump_max_W=hy.P_PUMP_MAX, onoff_cap_dB=hy.G_CAP, nbar_bounds=hy.NBAR_BOUNDS,
    full_model="all channels and pumps, counter-propagating pumps, pump depletion, pump-pump, pump-signal and signal-signal SRS with photon-number factor, attenuation; solve_bvp",
    full_model_optimization="sequential LP with forward-difference Jacobian of the full model (step 2 mW), trust region, exact penalty mu = 50 on the mean; local optimum",
    excluded="ASE and noise figure, double Rayleigh scattering (estimated separately), pump RIN, temperature, polarization, spectral shape of a real GFF, spectral hole burning, EDFA dynamics")


def main(fig09_only=False):
    res9, d_fo, ev_fo, d_s, ev_s = compute_fig09()
    res9["conditions"] = dict(CONDITIONS, budget_W=BUDGET_REF)
    res9["raman_scaling_check"] = hy.cr_scaling_check()
    res9["channels_nm"] = LAM
    res9["curves"] = dict(G_edfa=ev_s["fo"]["G_edfa"], G_onoff_first_order=ev_s["fo"]["G_onoff"],
                          G_onoff_full=ev_s["G_onoff"], G_net_full=ev_s["G_net"],
                          G_net_first_order_design_first_order=d_fo["G_net"],
                          G_net_first_order_design_full=ev_fo["G_net"])
    with open(os.path.join(FIG_DIR, "Fig09_hybrid_design_results.json"), "w", encoding="utf-8") as fh:
        json.dump(r(res9), fh, indent=1, ensure_ascii=True)
    plot_fig09(d_fo, ev_fo, d_s, ev_s)
    print("Fig09: first-order design P [mW] =", np.round(d_fo["P"] * 1e3, 1), "nbar = %.4f" % d_fo["nbar"])
    print("  first-order ripple %.3f dB; full model at the same pumps: ripple %.3f dB, mean %.3f dB" % (
        d_fo["ripple"], ev_fo["ripple"], ev_fo["mean_net"]))
    print("  full-model-optimized P [mW] =", np.round(d_s["P"] * 1e3, 1), "nbar = %.4f; ripple %.3f dB, mean %.4f dB" % (
        d_s["nbar"], ev_s["ripple"], ev_s["mean_net"]))
    print("  sensitivity:", json.dumps(r(res9["sensitivity_reference_design"])))
    if fig09_only:
        return
    res13, fams = compute_fig13()
    res13["conditions"] = CONDITIONS
    res13["definitions"] = dict(
        ripple="maximum multichannel gain variation: max - min of the span net gain over the 40 channels (ITU-T G.661)",
        budget="upper limit of the total Raman pump power launched into the fiber; the optimizer may use less",
        full_model_in_the_loop=CONDITIONS["full_model_optimization"] + "; started from the first-order design and from the solution at the previous budget, so the curve cannot increase with budget",
        families="free: EDFA mean inversion is an LP variable; fixed: inversion set so that the EDFA mean gain is 10.7 or 6.7 dB and the Raman section supplies the rest",
        infeasible="budget below which no pump setting reaches the 0 dB mean span net gain (fixed families only)")
    with open(os.path.join(FIG_DIR, "Fig13_pump_budget_results.json"), "w", encoding="utf-8") as fh:
        json.dump(r(res13), fh, indent=1, ensure_ascii=True)
    plot_fig13(res13, fams)
    for k, f in res13["families"].items():
        at05 = next((rw for rw in f["rows"] if abs(rw["budget"] - 0.5) < 1e-9), {})
        print("Fig13 %-12s: min budget FO %s W, full %s W; ripple at 0.5 W FO %s / full %s dB; floors FO %.3f / full %.3f dB" % (
            k, f.get("min_budget_first_order_W", "-"), f.get("min_budget_full_model_W", "-"),
            at05.get("fo_ripple", "-"), at05.get("full_ripple", "-"), f["first_order_floor_ripple_dB"],
            f["full_model_floor_ripple_dB"]))
    print("runtime %.0f s + %.0f s" % (res9["runtime_s"], res13["runtime_s"]))


def plot_only(fig09_only=False):
    """Redraw both figures (or Figure 9 only) from the results JSON files (no computation)."""
    with open(os.path.join(FIG_DIR, "Fig09_hybrid_design_results.json"), encoding="utf-8") as fh:
        r9 = json.load(fh)
    cv = {k: np.array(v, float) for k, v in r9["curves"].items()}
    fd, sd = r9["first_order_design"], r9["full_model_optimized_design"]
    d_fo = dict(P=np.array(fd["pump_powers_mW"]) / 1e3, G_net=cv["G_net_first_order_design_first_order"],
                ripple=fd["first_order"]["net_ripple_pp"])
    ev_fo = dict(G_net=cv["G_net_first_order_design_full"], ripple=fd["full_model"]["net_ripple_pp"])
    d_s = dict(P=np.array(sd["pump_powers_mW"]) / 1e3, lam_p=np.array(sd["pump_wavelengths_nm"]), nbar=sd["nbar"])
    ev_s = dict(fo=dict(G_edfa=cv["G_edfa"], G_onoff=cv["G_onoff_first_order"]), G_onoff=cv["G_onoff_full"],
                G_net=cv["G_net_full"], ripple=sd["full_model"]["net_ripple_pp"])
    plot_fig09(d_fo, ev_fo, d_s, ev_s)
    if fig09_only:
        return
    with open(os.path.join(FIG_DIR, "Fig13_pump_budget_results.json"), encoding="utf-8") as fh:
        r13 = json.load(fh)
    fams = [dict(key=k, label=f["label"], nbar=f["nbar"]) for k, f in r13["families"].items()]
    plot_fig13(r13, fams)


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:              # with --fig09-only: redraw Figure 9 only
        plot_only(fig09_only="--fig09-only" in sys.argv)
    else:
        main(fig09_only="--fig09-only" in sys.argv)
