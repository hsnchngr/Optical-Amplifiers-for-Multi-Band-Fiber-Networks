"""Figure 4 (SOA physics) and Figure 5 (Raman physics) of the revised manuscript.

Figure 4: a) traveling-wave SOA; b) material gain spectra of the free-carrier band model (card SOA-BAND-ILL), normalized;
          c) gain compression and recovery after a short pulse for card SOA-BULK-REF at 150 mA, sectioned Equations
          (6)-(7) and the reservoir model Equation (9) with card-derived parameters; d) gain fluctuation seen by the data
          under PRBS NRZ-OOK, a WDM aggregate and (nearly) constant-envelope signals (soa_patterning.py).
Figure 5: a) default SSMF Raman gain-efficiency profile of GNPy with the card value (no inset; the three-pump example
          quoted in Section 2.5 is computed here for the results file only, key three_pump_example, and not drawn);
          b) signal power along an 80 km span for co-, counter- and bidirectional pumping at 15 dB on-off gain;
          c) equivalent noise figure (ASE only); d) OSNR improvement over the unpumped span at equal nonlinear phase
          for the three schemes and for the optimum split of the pump power (left axis), and that optimum forward
          share of the pump power (right axis).
Outputs: ../figures/Fig04_soa_physics.{png,pdf}, Fig05_raman_physics.{png,pdf} and their *_results.json.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.optimize import brentq

import style
import soa_physics as sp
import soa_patterning
import raman_physics as rp
from models import RAMAN, raman_CR, SOA

style.apply()
style.math_in_text_font()          # math symbols in Arial, like the rest of the text (one font family)
C, INK, INK2 = style.C, style.INK, style.INK2
c0 = 2.99792458e8


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=conv)


# =====================================================================================================================
# Figure 4
# =====================================================================================================================
out4 = {}
fig = plt.figure(figsize=(7.1, 5.5))
gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.30, left=0.08, right=0.97, top=0.90, bottom=0.08)
axA, axB, axC, axD = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))

# ---- a) schematic ----------------------------------------------------------------------------------------------------
ax = axA
ax.set_xlim(-1.6, 11.6); ax.set_ylim(-0.6, 7.2); ax.axis("off")
ax.add_patch(Rectangle((1, 1.6), 8, 1.2, fc="#e8e6e1", ec=INK2, lw=0.8))
ax.text(5, 2.2, "n-InP substrate / cladding", ha="center", va="center", fontsize=7)
ax.add_patch(Rectangle((1, 2.8), 8, 0.55, fc=C[1], ec=INK2, lw=0.8, alpha=0.9))
ax.text(5, 3.07, "active layer (bulk, MQW or QD)", ha="center", va="center", fontsize=6.6, color="white")
ax.add_patch(Rectangle((1, 3.35), 8, 1.2, fc="#f3f2ee", ec=INK2, lw=0.8))
ax.text(5, 3.95, "p-InP cladding", ha="center", va="center", fontsize=7)
ax.add_patch(Rectangle((3.4, 4.55), 3.2, 0.5, fc="#c9c7c0", ec=INK2, lw=0.8))
ax.text(5, 4.8, "p-contact", ha="center", va="center", fontsize=6.8)
ax.add_patch(FancyArrowPatch((5, 6.15), (5, 5.12), arrowstyle="-|>", mutation_scale=9, color=INK))
ax.text(5.25, 5.7, "bias current $I$", fontsize=7, va="center")
for x in (1, 9):
    ax.add_patch(Rectangle((x - 0.13, 1.6), 0.26, 2.95, fc=C[0], ec="none", alpha=0.6))
ax.text(-0.2, 5.0, "AR-coated or\ntilted facets", ha="center", va="center", fontsize=6.5, color=INK2)
ax.annotate("", xy=(0.95, 4.4), xytext=(0.15, 4.65), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.add_patch(FancyArrowPatch((-1.5, 3.07), (0.75, 3.07), arrowstyle="-|>", mutation_scale=9, color=C[0], lw=1.2))
ax.add_patch(FancyArrowPatch((9.25, 3.07), (11.5, 3.07), arrowstyle="-|>", mutation_scale=12, color=C[0], lw=2.4))
ax.text(-1.45, 3.45, r"$P_{\rm in}$", fontsize=8, color=C[0])
ax.text(9.55, 3.45, r"$G\,P_{\rm in}$", fontsize=8, color=C[0])
ax.annotate("", xy=(9, 1.05), xytext=(1, 1.05), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
# operators as text characters between math symbols, so that their spacing matches the rest of the line
ax.text(5, 0.62, r"length $L$ ≈ 0.5–2 mm", ha="center", va="center", fontsize=6.8, color=INK2)
ax.text(5, -0.25, r"card SOA-BULK-REF: $L$ = 0.8 mm, $w$ × $d$ = 1.9 × 0.12 µm$^2$, $\Gamma$ = 0.3",
        ha="center", va="center", fontsize=6.5, color=INK2)
style.label(ax, "a", x=0.0, y=1.0)

# ---- b) gain spectra -------------------------------------------------------------------------------------------------
lam = np.linspace(1350e-9, 1700e-9, 351)
op150 = sp.card_operating_point(0.15)
cons = sp.band_card_consistency(0.15)
lam_t = np.linspace(1350e-9, 1750e-9, 801)
N_tr_peak = brentq(lambda n: sp.material_gain(lam_t, n).max(), 0.5e24, 1.5e24, xtol=1e20)
Ns = [0.9e24, cons["N_tr_band"], 2.0e24, op150["N"]]
g_norm, gpk = sp.gain_spectra(Ns, lam, normalize=True)
g_abs = g_norm * gpk
ax = axB
cols_b = [C[6], C[2], C[0], C[1]]
peaks = []
for k, (N, row) in enumerate(zip(Ns, g_norm)):
    ax.plot(lam * 1e9, row, color=cols_b[k])
    i = int(np.argmax(row))
    peaks.append(dict(N=N, peak_nm=float(lam[i] * 1e9), peak_abs_per_cm=float(g_abs[k, i] / 100),
                      g1550_abs_per_cm=float(np.interp(1550e-9, lam, g_abs[k]) / 100)))
    ax.plot(lam[i] * 1e9, row[i], "o", ms=3, color=cols_b[k])
    if k == 3:
        ax.text(lam[i] * 1e9 + 6, row[i] + 0.035, f"{N/1e24:.2f} (150 mA)", color=cols_b[k], fontsize=6.8,
                va="bottom")
    else:
        ax.text(lam[i] * 1e9, row[i] - 0.06, f"{N/1e24:.2f}", color=cols_b[k], fontsize=6.8, va="top", ha="center")
ax.text(1356, 1.13, r"$N$ [10$^{24}$ m$^{-3}$]", fontsize=6.8, color=INK2, va="top")
ax.axhline(0, color=INK2, lw=0.6)
ax.axvline(1550, color=INK2, lw=0.6, ls=":")
ax.text(1546, 1.13, "1550 nm", fontsize=6.5, color=INK2, ha="right", va="top")
ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Normalized material gain")
ax.set_xlim(1350, 1700); ax.set_ylim(-0.5, 1.2)
style.label(ax, "b")
out4["b_gain_spectra"] = dict(card="SOA-BAND-ILL", normalization_peak_per_cm=gpk / 100, curves=peaks,
                              N_transparency_peak=N_tr_peak, N_transparency_1550nm=cons["N_tr_band"],
                              card_consistency_150mA_1550nm=cons)

# ---- c) pulse response -----------------------------------------------------------------------------------------------
ax = axC
energies = [0.003e-12, 0.03e-12, 0.3e-12]
cols_c = [C[2], C[0], C[1]]
pulse_rows = []
for E, col in zip(energies, cols_c):
    t, h_tw = sp.tw_pulse(E, 0.15, M=64, t1=1.2e-9, dt=0.05e-12)
    _, h_res = sp.reservoir_pulse(E, op150, t1=1.2e-9, dt=0.05e-12)
    G_tw, G_res = h_tw * sp.NP10, h_res * sp.NP10
    ax.plot(t * 1e12, G_tw, color=col, lw=1.6, label=f"{E*1e12:g} pJ")
    ax.plot(t * 1e12, G_res, color=col, lw=1.1, ls="--")
    i0 = int(np.argmin(G_tw))
    Ppk = float(sp.gaussian_pulse(0.0, E))
    pulse_rows.append(dict(E_pJ=E * 1e12, P_peak_mW=Ppk * 1e3, P_peak_over_Psat=Ppk / op150["Psat"],
                           E_over_Esat=E / op150["Esat"], G0E_over_Esat=op150["G0"] * E / op150["Esat"],
                           avg_power_as_10G_RZ_OOK_dBm=10 * np.log10(0.5 * E * 10e9 * 1e3),
                           compression_tw_dB=float(G_tw[0] - G_tw.min()),
                           compression_eq9_dB=float(G_res[0] - G_res.min()),
                           rec90_tw_ps=sp.recovery_time(t, G_tw) * 1e12, rec90_eq9_ps=sp.recovery_time(t, G_res) * 1e12,
                           gain_25ps_after_min_tw_dB=float(np.interp(t[i0] + 25e-12, t, G_tw)),
                           gain_100ps_after_min_tw_dB=float(np.interp(t[i0] + 100e-12, t, G_tw)),
                           G0_tw_dB=float(G_tw[0])))
for tb in (25, 100):                       # bit slots at 40 and 10 Gb/s; named in the legend (no label at the y-axis)
    ax.axvline(tb, color=INK2, lw=0.7, ls=":")
ax.plot([], [], color=INK2, lw=1.6, label="numerical model, sectioned")
ax.plot([], [], color=INK2, lw=1.1, ls="--", label="reservoir model, card-derived")
ax.plot([], [], color=INK2, lw=0.7, ls=":", label="$t$ = 25 and 100 ps")
ax.legend(loc="lower right", fontsize=6.5, borderaxespad=0.3, ncol=1)
ax.set_xlim(-30, 1000); ax.set_ylim(5, 28)
ax.set_xlabel("Time after pulse peak [ps]"); ax.set_ylabel("Gain [dB]")
style.label(ax, "c")
out4["c_pulse"] = dict(conditions=dict(card="SOA-BULK-REF", I_mA=150, G0_dB=op150["G0_dB"], tau_d_ps=op150["tau_d"] * 1e12,
                                       tau_c_ps=op150["tau_c"] * 1e12, Esat_pJ=op150["Esat"] * 1e12,
                                       Psat_mW=op150["Psat"] * 1e3, pulse="Gaussian, FWHM 8.3 ps",
                                       sectioned_model="64 sections, RK4, dt 0.05 ps"),
                       pulses=pulse_rows, eq9_rec90_equals_ln10_tau_d_ps=np.log(10) * op150["tau_d"] * 1e12)

# ---- d) patterning --------------------------------------------------------------------------------------------------
pat = soa_patterning.load_or_run()
m = pat["results"]["150mA"]
Pd = np.array(pat["config"]["P_tot_dBm"])
ax = axD
curves = [("ook10G_N1_total", "std_mark_dB", "10 Gb/s NRZ-OOK", C[0], "-"),
          ("ook40G_N1_total", "std_mark_dB", "40 Gb/s NRZ-OOK", C[1], "-"),
          ("ook10G_N40_total", "std_mark_dB", r"40 $\times$ 10 Gb/s NRZ-OOK, total power", C[0], "--"),
          ("dpsk10G", "std_sym_dB", "10 Gb/s NRZ-DPSK", C[2], "-.")]
for name, key, lab, col, ls in curves:
    ax.plot(Pd, m["tw"][name][key], color=col, ls=ls, marker="o", ms=2.2, label=lab)
ax.plot(Pd, np.zeros_like(Pd), color=INK, ls=":", lw=1.2, label="constant envelope")
ax.set_xlim(-30, 0); ax.set_ylim(-0.03, 1.55)
ax.set_xlabel("Average input power [dBm]"); ax.set_ylabel("Gain standard deviation [dB]")
ax.legend(loc="upper left", fontsize=6.3, borderaxespad=0.3, handlelength=3.2)
comp = np.array(m["cw_compression_dB"]["tw"])
fwd = lambda x: np.interp(x, Pd, comp)
inv = lambda y: np.interp(y, comp, Pd)
sec = ax.secondary_xaxis("top", functions=(fwd, inv))
sec.set_xticks([0.5, 1, 3, 6, 10])
sec.set_xticklabels(["0.5", "1", "3", "6", "10"])
sec.set_xlabel("CW gain compression [dB]", fontsize=8)
sec.tick_params(labelsize=7.5)
style.label(ax, "d", y=1.17)
sel = {}
for name, key, lab, col, ls in curves:
    sel[name] = {mod: {k: m[mod][name][k] for k in m[mod][name] if isinstance(m[mod][name][k], list)} for mod in ("tw", "res")}
out4["d_patterning"] = dict(source="soa_patterning.py (FigS2_soa_patterning_results.json)", P_avg_dBm=Pd,
                            operating_point=m["operating_point"], cw_compression_dB=m["cw_compression_dB"],
                            metric="standard deviation of the gain at the centers of the probe marks (OOK) or symbol "
                                   "centers (DPSK); WDM: 40 channels at the same total average power",
                            cases=sel)
style.save(fig, "Fig04_soa_physics")
plt.close(fig)
dump(out4, "../figures/Fig04_soa_physics_results.json")

# =====================================================================================================================
# Figure 5
# =====================================================================================================================
out5 = {}
fig = plt.figure(figsize=(7.1, 5.5))
gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.42, left=0.08, right=0.92, top=0.90, bottom=0.08)
axA, axB, axC, axD = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))

# ---- a) gain-efficiency profile (default SSMF profile of GNPy) and card value; no inset ------------------------------
ax = axA
off, cr, lref = rp.profile_reference()
ax.plot(off, cr, color=C[5], lw=1.6)
ax.fill_between(off, 0, cr, color=C[5], alpha=0.10, lw=0)
ipk = int(np.argmax(cr))
ax.annotate(f"peak {cr[ipk]:.2f} W$^{{-1}}$km$^{{-1}}$ at {off[ipk]:.2f} THz", xy=(off[ipk], cr[ipk]),
            xytext=(18.5, 0.452), fontsize=6.5, color=INK2, va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
df_card = (c0 / RAMAN["lam_p"] - c0 / RAMAN["lam_s"]) / 1e12
ax.plot(df_card, raman_CR() * 1e3, "x", color=INK, ms=5, mew=1.2)
ax.annotate("card RAMAN-G652", xy=(df_card, raman_CR() * 1e3), xytext=(18.5, 0.405), fontsize=6.5,
            color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlim(0, 42); ax.set_ylim(0, 0.48)
ax.set_xlabel("Frequency shift from pump [THz]"); ax.set_ylabel(r"Raman gain efficiency [W$^{-1}$ km$^{-1}$]")
f_ref = c0 / (lref * 1e-9) / 1e12


def _shift_to_lam(x):
    return c0 / ((f_ref - np.clip(np.asarray(x, float), -1e3, f_ref - 1.0)) * 1e12) * 1e9


def _lam_to_shift(lam_nm):
    return f_ref - c0 / (np.clip(np.asarray(lam_nm, float), 1.0, None) * 1e-9) / 1e12


sec = ax.secondary_xaxis("top", functions=(_shift_to_lam, _lam_to_shift))
sec.set_xticks([1500, 1600, 1700, 1800])
sec.set_xlabel(f"Stokes wavelength for a {lref:.0f} nm pump [nm]", fontsize=8)
sec.tick_params(labelsize=7.5)
style.label(ax, "a", y=1.17)
# Three-pump flat-gain example quoted in Section 2.5 and in the caption: computed for the results file only. The former
# inset of panel a was removed (no insets in the figures); Figure 9 shows a full multi-pump design.
lam_s = np.linspace(1500, 1620, 241)
band = (lam_s >= 1520) & (lam_s <= 1600)
pumps = [1425.0, 1455.0, 1485.0]
Pp, rip = rp.flat_pump_design(lam_s[band], pumps, 10.0, L=80e3)
tot, per = rp.onoff_gain_db(lam_s, pumps, Pp, L=80e3)
out5["a_profile"] = dict(source="cards.raman_profile_ssmf (GNPy default SSMF profile), pump-frequency scaling nu_p/nu_ref",
                         reference_pump_nm=lref, peak_CR_per_W_km=float(cr[ipk]), peak_shift_THz=float(off[ipk]),
                         card_CR_per_W_km=raman_CR() * 1e3, card_shift_THz=df_card,
                         gnpy_CR_at_card_shift_1450nm=float(rp.cr_measured(df_card, RAMAN["lam_p"] * 1e9)))
x = np.linspace(0, 42, 42001); y = np.interp(x, off, cr)
out5["a_profile"].update(half_max_range_THz=[float(x[y >= 0.5 * cr[ipk]].min()), float(x[y >= 0.5 * cr[ipk]].max())],
                         within_1p5pct_of_peak_THz=[float(x[y >= 0.985 * cr[ipk]].min()),
                                                    float(x[y >= 0.985 * cr[ipk]].max())],
                         rel_5THz=float(np.interp(5, off, cr) / cr[ipk]), rel_14p75THz=float(np.interp(14.75, off, cr) / cr[ipk]),
                         rel_16p5THz=float(np.interp(16.5, off, cr) / cr[ipk]))
out5["three_pump_example"] = dict(model="first order: undepleted, dB-additive on-off gains, no pump-pump SRS, "
                                        "80 km SSMF, l_p = 0.25 dB/km for all pumps",
                                  drawn="no (results file, Section 2.5 and caption only; former inset of panel a)",
                                  pumps_nm=pumps, powers_mW=Pp * 1e3, total_mW=float(Pp.sum() * 1e3),
                                  design_band_nm=[1520, 1600], target_dB=10.0, ripple_pp_dB=float(rip),
                                  mean_gain_dB=float(tot[band].mean()))

# ---- b) signal power along the span ----------------------------------------------------------------------------------
ax = axB
names = {"co": "co-pumped", "counter": "counter-pumped", "bi": "bidirectional (50:50)"}
cols_s = {"co": C[1], "counter": C[0], "bi": C[2]}
prof15 = {}
for sch in ("co", "counter", "bi"):
    r = rp.profile(sch, 15.0)
    prof15[sch] = r
    ax.plot(r["z"] / 1e3, 10 * np.log10(r["Prel"]), color=cols_s[sch])
z = np.linspace(0, 80e3, 200)
ax.plot(z / 1e3, -RAMAN["alpha_s"] / (np.log(10) / 10) * z, ":", color=INK2, lw=1.1)
ax.text(27, 7.6, "co-pumped", color=cols_s["co"], fontsize=7, ha="center", va="bottom")
ax.text(62, -8.4, "counter-pumped", color=cols_s["counter"], fontsize=7, ha="center", va="center")
ax.text(62, -3.1, "bidirectional", color=cols_s["bi"], fontsize=7, ha="center", va="center")
ax.text(38, -11.2, "unpumped", color=INK2, fontsize=7, ha="center", va="center")
ax.text(1.5, -15.6, f"15 dB on-off gain, {prof15['co']['Ptot']:.2f} W total pump", color=INK2, fontsize=6.5,
        va="bottom")
ax.set_xlim(0, 80); ax.set_ylim(-16.5, 9.8)
ax.set_xlabel("Position along the span [km]"); ax.set_ylabel("Signal power relative\nto launch [dB]")
style.label(ax, "b")
out5["b_profiles_15dB"] = {s: dict(pump_total_mW=prof15[s]["Ptot"] * 1e3,
                                   max_rel_dB=float(10 * np.log10(prof15[s]["Prel"].max())),
                                   min_rel_dB=float(10 * np.log10(prof15[s]["Prel"].min())),
                                   R_NL_dB=float(10 * np.log10(prof15[s]["R_NL"])),
                                   net_span_gain_dB=float(10 * np.log10(prof15[s]["Gnet"]))) for s in prof15}

# ---- c) equivalent noise figure --------------------------------------------------------------------------------------
ax = axC
Gs = np.arange(0.0, 22.01, 0.5)
nfeq = {}
for sch in ("co", "counter", "bi"):
    nfeq[sch] = np.array([0.0 if G == 0 else 10 * np.log10(rp.profile(sch, G)["NF_eq"]) for G in Gs])
    ax.plot(Gs, nfeq[sch], color=cols_s[sch], label=names[sch])
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlim(0, 22); ax.set_ylim(-11.5, 1.5)
ax.set_xlabel("On-off gain [dB]"); ax.set_ylabel("Equivalent noise figure [dB]")
ax.legend(loc="lower left", fontsize=6.6, borderaxespad=0.3)
style.label(ax, "c")
out5["c_NF_eq_dB"] = {s: {f"{G:g}": float(nfeq[s][np.argmin(abs(Gs - G))]) for G in (5, 10, 15, 20)} for s in nfeq}

# ---- d) OSNR improvement at equal nonlinear phase --------------------------------------------------------------------
ax = axD
NFA = 5.0
dO = {}
for sch, f in (("counter", 0.0), ("bi", 0.5), ("co", 1.0)):
    dO[sch] = np.array([0.0 if G == 0 else rp.d_osnr_nl(f, G, nf_amp_db=NFA) for G in Gs])
    ax.plot(Gs, dO[sch], color=cols_s[sch], label=names[sch])
opt = [(0.0, 0.0) if G == 0 else rp.optimum_share(G, nf_amp_db=NFA) for G in Gs]
fopt = np.array([o[0] for o in opt]); dopt = np.array([o[1] for o in opt])
# dashed: OSNR improvement with the optimum split of the pump power (left axis); dotted: that optimum forward share
# (right axis). The legend names both, so that 'optimum forward share' labels the share curve only.
ax.plot(Gs, dopt, color=INK, lw=1.1, ls="--", label="OSNR at optimum split")
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlim(0, 22); ax.set_ylim(-1.0, 7.2)
ax.set_yticks(np.arange(-1, 7.1, 1))
ax.set_xlabel("On-off gain [dB]"); ax.set_ylabel("OSNR improvement at\nequal nonlinear phase [dB]")
ax2 = ax.twinx()
ax2.plot(Gs[Gs > 0], 100 * fopt[Gs > 0], color=C[6], lw=1.0, ls=":")
ax2.set_ylim(-4, 100); ax2.set_ylabel("Optimum forward share [%]", color=C[6])
ax2.tick_params(axis="y", colors=C[6]); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(C[6])
ax2.grid(False)
h_osnr, _ = ax.get_legend_handles_labels()
h_share = Line2D([], [], color=C[6], lw=1.0, ls=":", label="optimum forward share")    # right axis, same color
ax.legend(handles=h_osnr + [h_share], loc="upper left", fontsize=6.4, borderaxespad=0.3)
style.label(ax, "d")
iopt = int(np.argmax(dopt))
tab = {}
for L in (80e3, 100e3, 120e3):
    for G in (10.0, 15.0, 20.0):
        fo, do = rp.optimum_share(G, L, NFA)
        tab[f"L{L/1e3:.0f}_G{G:.0f}"] = dict(counter=rp.d_osnr_nl(0.0, G, L, NFA), bi=rp.d_osnr_nl(0.5, G, L, NFA),
                                             co=rp.d_osnr_nl(1.0, G, L, NFA), f_opt=fo, opt=do,
                                             pump_mW=rp.profile("bi", G, L=L)["Ptot"] * 1e3)
glob = {}
for L in (80e3, 100e3, 120e3):
    best = max(((rp.optimum_share(G, L, NFA), G) for G in np.arange(1.0, 30.01, 0.5)), key=lambda t: t[0][1])
    bc = max(((rp.d_osnr_nl(0.0, G, L, NFA), G) for G in np.arange(1.0, 30.01, 0.5)))
    glob[f"L{L/1e3:.0f}"] = dict(opt_dOSNR_dB=best[0][1], opt_f=best[0][0], opt_Gon_dB=best[1],
                                 opt_pump_mW=rp.profile("bi", best[1], L=L)["Ptot"] * 1e3,
                                 counter_only_dOSNR_dB=bc[0], counter_only_Gon_dB=bc[1])
out5["d_osnr_equal_nl_phase"] = dict(
    definition="dOSNR_NL = NF_amp - NF_eq,tot - 10log10(R_NL); R_NL = int P(z)/P(0) dz / L_eff,passive; reference: "
               "unpumped 80 km span followed by the same amplifier (NF 5 dB); undepleted pump, single low-power "
               "channel, ASE only (no DRB, no pump RIN), equal nonlinear phase as first-order proxy",
    Gon_dB=Gs, counter=dO["counter"], bi=dO["bi"], co=dO["co"], optimum=dopt, f_opt=fopt,
    max_over_G_80km=dict(dOSNR_dB=float(dopt[iopt]), Gon_dB=float(Gs[iopt]), f_opt=float(fopt[iopt])),
    table=tab, unconstrained_optima=glob)
style.save(fig, "Fig05_raman_physics")
plt.close(fig)
dump(out5, "../figures/Fig05_raman_physics_results.json")
_tolist = lambda o: o.tolist() if hasattr(o, "tolist") else float(o)
print(json.dumps({k: out4[k] for k in ("b_gain_spectra", "c_pulse")}, indent=1, default=_tolist))
print(json.dumps({k: out5[k] for k in out5 if k != "d_osnr_equal_nl_phase"}, indent=1, default=_tolist))
print(json.dumps(out5["d_osnr_equal_nl_phase"]["table"], indent=1, default=_tolist))
print(json.dumps(out5["d_osnr_equal_nl_phase"]["unconstrained_optima"], indent=1, default=_tolist))
print("80 km max", out5["d_osnr_equal_nl_phase"]["max_over_G_80km"])
