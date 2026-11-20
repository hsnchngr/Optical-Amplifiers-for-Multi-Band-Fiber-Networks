"""Figure 3 (revision 3) - operating bands and spectroscopy of the amplifier gain media.

a) ITU-T bands and amplifier windows. Solid bars: documented single-device windows (C- and L-band EDFAs,
   TDFA, PDFA, O- and E-band BDFAs and U-band discrete Raman amplifier of one O- to U-band laboratory
   system, ref. P25; S-band BDFA of [69]). Pale box: range covered by BDFAs as a family; open bar: one
   double-pass Bi-doped fiber amplifier with gain > 20 dB [17]. Dashed: SOA design range. Dotted: Raman, gain
   band set by the pump wavelengths. Bottom row: pump band of a contiguous S+C+L Raman signal band (signal band shifted by
   the 13.2 THz Raman shift) and its overlap with the S-band signals (R2-M11). Top strip: bands used by
   commercial WDM line systems (C, L) and bands used so far in laboratory and field trials (R2-M1).
b) Er3+: Giles absorption and gain coefficients alpha(lambda), g*(lambda) of the C-band EDF of [124]
   (card EDFA-PERIN2019, digitized data), both divided by max(alpha); dotted line: alpha = g* (h nu = eps).
c) Pr3+: parameterized line shapes of the ground-state absorption 3H4->1G4 (PDFA pump band) and of the
   1G4->3H5 signal emission, each normalized to its own maximum; arrow: ESA from 1G4 near 1380 nm.
d) Tm3+: parameterized 3H6->3F4 absorption (single Gaussian in photon energy, peak 1640 nm) and its
   McCumber transform, Equation (1), which reproduces the emission peak (1865 nm) and the peak ratio
   (0.863) of [35]. The line shapes in (c) and (d) are NOT measured spectra.

Inputs (layer L1): ../cards/edfa_perin2019.json with ../data/perin2019_giles_coefficients.csv, and
../cards/ref_fig03_bands_spectroscopy.json (REF-FIG03-BANDS-SPECTRO). No other number enters the figure.
Outputs: ../figures/Fig03_bands_spectroscopy.png/.pdf and ../figures/Fig03_bands_spectroscopy_results.json.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cards
import style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
NAME = "Fig03_bands_spectroscopy"

NM_THZ = 299792.458            # lambda [nm] * nu [THz]
KB_CM = 0.6950348              # k_B / (h c) [cm^-1 K^-1]
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

REF_CARD = cards.load(os.path.join(cards.CARD_DIR, "ref_fig03_bands_spectroscopy.json"))
R = cards.values(REF_CARD)
PERIN_CARD = cards.load("EDFA-PERIN2019")


def thz(lam_nm):
    return NM_THZ / np.asarray(lam_nm, dtype=float)


def width_thz(win):
    lo, hi = win
    return float(thz(lo) - thz(hi))


def wavenumber(lam_nm):
    return 1e7 / np.asarray(lam_nm, dtype=float)


def gauss_energy(lam_nm, lam0_nm, s_cm):
    """Gaussian line shape in photon energy (wavenumber), evaluated on a wavelength grid; peak value 1."""
    return np.exp(-0.5 * ((wavenumber(lam_nm) - wavenumber(lam0_nm)) / s_cm) ** 2)


def s_from_fwhm_nm(lam0_nm, fwhm_nm):
    """Energy-space standard deviation [cm^-1] of a line whose FWHM is fwhm_nm near lam0_nm."""
    return 1e7 * fwhm_nm / lam0_nm ** 2 / FWHM_PER_SIGMA


def band_letter(ax, x, letter):
    """ITU-T band letter at the bottom of the shaded band of a spectroscopy panel."""
    ax.text(x, 0.07, letter, ha="center", va="center", fontsize=7.5, fontweight="bold", color=style.INK, zorder=1)


def spectro_axes(ax, ylabel):
    """Common axes of panels b-d: headroom above 1 for the legend, ticks only up to 1."""
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(0, 1.0)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", fontsize=6.6, handlelength=1.2, borderaxespad=0.15, labelspacing=0.25,
              handletextpad=0.5)


def fwhm_nm_numeric(lam, y):
    """Full width at half maximum of a sampled single-peaked curve [nm] (linear interpolation)."""
    half = y.max() / 2.0
    idx = np.where(y >= half)[0]
    i0, i1 = idx[0], idx[-1]
    lo = np.interp(half, [y[i0 - 1], y[i0]], [lam[i0 - 1], lam[i0]])
    hi = np.interp(half, [y[i1 + 1], y[i1]], [lam[i1 + 1], lam[i1]])
    return float(hi - lo)


# ----------------------------------------------------------------------------------------------------------------
# Numbers (all from the two cards); checks of the derived card entries (V2)
# ----------------------------------------------------------------------------------------------------------------
def tm_parameters():
    """Gaussian McCumber pair for Tm3+ 3H6<->3F4: three cited numbers fix (E_a, s, eps)."""
    T = R["T_spectro"]
    kT = KB_CM * T
    Ea, Ee = wavenumber(R["tm_abs_peak"]), wavenumber(R["tm_em_peak"])
    ratio = R["tm_sigma_e_peak"] / R["tm_sigma_a_peak"]
    s = np.sqrt(kT * (Ea - Ee))
    eps = 0.5 * (Ea + Ee) + kT * np.log(ratio)
    checks = {
        "tm_peak_ratio": (ratio, R["tm_peak_ratio"], 1e-3),
        "tm_width_s": (s, R["tm_width_s"], 0.1),
        "tm_eps": (eps, R["tm_eps"], 0.5),
    }
    for key, (calc, card, tol) in checks.items():
        if abs(calc - card) > tol:
            raise ValueError(f"card entry {key} = {card} disagrees with recomputed value {calc}")
    return dict(T=T, kT=kT, Ea=float(Ea), Ee=float(Ee), ratio=float(ratio), s=float(s), eps=float(eps))


def perin_mccumber():
    """McCumber consistency of the alpha/g* pair of [124]: fits of ln(g*/alpha) = (eps - E)/(k_B T)."""
    lam, a, g = cards.perin2019_coefficients()
    E = wavenumber(lam)
    y = np.log(g / a)
    out = {"range_nm": [float(lam.min()), float(lam.max())], "n_points": int(lam.size),
           "alpha_max_per_m": float(a.max()), "lambda_alpha_max_nm": float(lam[a.argmax()]),
           "gstar_max_per_m": float(g.max()), "lambda_gstar_max_nm": float(lam[g.argmax()]),
           "alpha_max_dB_per_m": float(10 * np.log10(np.e) * a.max())}
    k = np.where(np.diff(np.sign(y)) != 0)[0][0]
    out["lambda_alpha_eq_gstar_nm"] = float(lam[k] - y[k] * (lam[k + 1] - lam[k]) / (y[k + 1] - y[k]))
    # (i) T fixed at the card value (300 K)
    T0 = PERIN_CARD["entries"]["T"]["value"]
    kT0 = KB_CM * T0
    fits = {}
    for lo, hi in ((1470, 1569), (1500, 1565)):
        m = (lam >= lo) & (lam <= hi)
        eps = float(np.mean(y[m] * kT0 + E[m]))
        dev = g[m] / (a[m] * np.exp((eps - E[m]) / kT0)) - 1
        fits[f"T_fixed_{T0:.0f}K_{lo}_{hi}nm"] = {
            "eps_cm": eps, "lambda_eps_nm": 1e7 / eps,
            "max_abs_dev_pct": float(100 * np.abs(dev).max()), "rms_dev_pct": float(100 * np.sqrt(np.mean(dev ** 2)))}
    # (ii) T and eps free (linear fit of ln(g*/alpha) against wavenumber), whole digitized range
    slope, icpt = np.polyfit(E, y, 1)
    T_fit = -1.0 / (slope * KB_CM)
    eps_fit = -icpt / slope
    dev = g / (a * np.exp(icpt + slope * E)) - 1
    fits["T_free_1470_1569nm"] = {
        "T_K": float(T_fit), "eps_cm": float(eps_fit), "lambda_eps_nm": float(1e7 / eps_fit),
        "max_abs_dev_pct": float(100 * np.abs(dev).max()), "rms_dev_pct": float(100 * np.sqrt(np.mean(dev ** 2)))}
    out["mccumber_fits"] = fits
    return lam, a, g, out


def raman_overlap(shift_thz):
    lo, hi = R["raman_example_signal_band"]
    p_short = NM_THZ / (thz(lo) + shift_thz)        # pump for the shortest-wavelength signal
    p_long = NM_THZ / (thz(hi) + shift_thz)         # pump for the longest-wavelength signal
    return {"shift_THz": shift_thz, "signal_band_nm": [lo, hi], "signal_width_THz": width_thz((lo, hi)),
            "pump_band_nm": [float(p_short), float(p_long)],
            "overlap_nm": [float(lo), float(p_long)] if p_long > lo else None,
            "overlap_width_nm": float(max(0.0, p_long - lo))}


# ----------------------------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------------------------
def main():
    style.apply()
    C, INK, INK2 = style.C, style.INK, style.INK2
    COL = {"PDFA": C[0], "BDFA": C[0], "TDFA": C[0], "EDFA": C[0], "SOA": "#d62728", "Raman": INK}  # blue, red, black
    ABS, EMI = C[0], C[1]
    results = {"figure": NAME, "cards": [PERIN_CARD["card_id"], REF_CARD["card_id"]]}

    fig = plt.figure(figsize=(7.1, 5.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.42, wspace=0.42,
                          left=0.155, right=0.985, top=0.93, bottom=0.085)

    # ------------------------------------------------------------------------------------------------ (a)
    ax = fig.add_subplot(gs[0, :])
    bands = [(b, R[f"band_{b}"]) for b in "OESCLU"]
    shade = ["#efeeea", "#f8f7f4"]
    y_top_shade = 9.62
    for i, (b, (lo, hi)) in enumerate(bands):
        for xe in ((lo, hi) if i == 0 else (hi,)):      # white background; thin lines at the band edges
            ax.plot([xe, xe], [0.95, y_top_shade], color="#b9b7b1", lw=0.5, zorder=0)
        ax.text(0.5 * (lo + hi), 9.32, b, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(0.5 * (lo + hi), 8.93, f"{width_thz((lo, hi)):.1f} THz", ha="center", va="center",
                fontsize=6.6, color=INK2)
    results["itu_bands"] = {b: {"edges_nm": list(w), "width_THz": round(width_thz(w), 2)} for b, w in bands}

    # status strip (R2-M1), inside the axes below the frequency ticks
    y_st = 9.92
    ax.plot([1530, 1625], [y_st, y_st], color=INK, lw=2.6, solid_capstyle="butt")
    for lo, hi in ((1260, 1530), (1625, 1675)):
        ax.plot([lo + 0.8, hi - 0.8], [y_st, y_st], color="#b9b7b1", lw=2.6, solid_capstyle="butt")
    ax.text(1577.5, y_st + 0.22, "commercial WDM", ha="center", va="bottom", fontsize=6.8, color=INK)
    ax.text(1395, y_st + 0.22, "laboratory and field transmission experiments", ha="center", va="bottom",
            fontsize=6.8, color=INK2)
    ax.text(1650, y_st + 0.22, "laboratory", ha="center", va="bottom", fontsize=6.8, color=INK2)

    H = 0.34      # bar height (data units)

    def solid(lo, hi, y, col, h=H, z=3):
        ax.add_patch(Rectangle((lo, y - h / 2), hi - lo, h, facecolor=col, edgecolor="white", lw=0.6, zorder=z))

    def hollow(lo, hi, y, col, h=H, z=3, lw=1.1, fill="white"):
        ax.add_patch(Rectangle((lo, y - h / 2), hi - lo, h, facecolor=fill, edgecolor=col, lw=lw, zorder=z))

    def note(x, y, s, ha="left", color=None, size=6.7):
        ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color or INK2, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    Y = {"PDFA": 8.2, "BDFA": 7.12, "TDFA": 6.05, "EDFA_C": 5.25, "EDFA_L": 4.45, "SOA": 3.6, "Raman": 2.8,
         "RamanEx": 1.6}
    drawn = {}

    # PDFA
    w = R["win_PDFA"]; solid(*w, Y["PDFA"], COL["PDFA"]); note(w[1] + 4, Y["PDFA"], f"{width_thz(w):.1f} THz")
    drawn["PDFA (Pr:fluoride)"] = {"win_PDFA": w}

    # BDFA family: pale box, single-device segments, one double-pass fiber with G > 20 dB
    fam = R["range_BDFA_family"]
    ax.add_patch(Rectangle((fam[0], Y["BDFA"] - 0.5), fam[1] - fam[0], 1.0, facecolor="none",
                           edgecolor=COL["BDFA"], lw=0.7, ls=(0, (1, 1.5)), zorder=1))
    yb_dev, yb_17 = Y["BDFA"] + 0.2, Y["BDFA"] - 0.24
    for key in ("win_BDFA_O", "win_BDFA_E", "win_BDFA_S_69"):
        solid(*R[key], yb_dev, COL["BDFA"], h=0.3)
    w17 = R["win_BDFA_17"]
    hollow(*w17, yb_17, COL["BDFA"], h=0.22, lw=1.0)
    note(w17[1] + 4, yb_17, "one fiber, gain > 20 dB")
    drawn["BDFA* (Bi:silica)"] = {k: R[k] for k in ("range_BDFA_family", "win_BDFA_O", "win_BDFA_E",
                                                   "win_BDFA_S_69", "win_BDFA_17")}

    # TDFA, EDFA C, EDFA L
    for row, key, col in (("TDFA", "win_TDFA_fluoride", COL["TDFA"]), ("EDFA_C", "win_EDFA_C", COL["EDFA"]),
                          ("EDFA_L", "win_EDFA_L", COL["EDFA"])):
        w = R[key]; solid(*w, Y[row], col); note(w[1] + 4, Y[row], f"{width_thz(w):.1f} THz")
        drawn[row] = {key: w}

    # SOA: design range (dashed)
    w = R["range_SOA_design"]
    ax.plot(w, [Y["SOA"]] * 2, color=COL["SOA"], lw=2.0, ls=(0, (5, 3)), solid_capstyle="butt", zorder=3)
    drawn["SOA"] = {"range_SOA_design": w}

    # Raman: gain band set by the pump wavelengths (dotted; card key range_Raman_pump_defined) and the U-band
    # discrete Raman amplifier
    w = R["range_Raman_pump_defined"]
    ax.plot(w, [Y["Raman"]] * 2, color=COL["Raman"], lw=2.0, ls=(0, (1, 2.2)), dash_capstyle="round", zorder=3)
    wu = R["win_DRA_U"]; solid(*wu, Y["Raman"], COL["Raman"], z=4)
    drawn["Raman"] = {"range_Raman_pump_defined": w, "win_DRA_U": wu}

    # Raman S+C+L example (R2-M11): signal band and the pump band = signal band shifted by the Raman shift
    ex = raman_overlap(R["raman_shift"])
    f_off, g0, _ = cards.raman_profile_ssmf()           # default SSMF profile of GNPy, for the sensitivity check
    f_peak = float(f_off[g0.argmax()])
    ex_alt = raman_overlap(f_peak)
    g_rel = float(np.interp(R["raman_shift"], f_off, g0) / g0.max())
    ys, yp = Y["RamanEx"] + 0.36, Y["RamanEx"] - 0.36
    sig, pmp, ov = ex["signal_band_nm"], ex["pump_band_nm"], ex["overlap_nm"]
    ax.add_patch(Rectangle((ov[0], yp - 0.2), ov[1] - ov[0], (ys - yp) + 0.4, facecolor="none",
                           edgecolor="#d62728", lw=0.7, ls=(0, (1, 1.5)), zorder=1))
    solid(sig[0], sig[1], ys, COL["Raman"], h=0.26)
    hollow(pmp[0], pmp[1], yp, COL["Raman"], h=0.26, lw=1.1)
    note(sig[0] - 4, ys, f"signals ({ex['signal_width_THz']:.1f} THz)", ha="right")
    note(pmp[0] - 4, yp, "pumps", ha="right")
    note(pmp[1] + 4, yp, f"= signal band shifted up by {R['raman_shift']:.1f} THz", ha="left")
    note(0.5 * (ov[0] + ov[1]), Y["RamanEx"], "overlap", ha="center", color=INK)
    results["raman_example"] = {
        "with_raman_shift_card": ex, "with_gnpy_profile_peak": ex_alt, "gnpy_profile_peak_THz": f_peak,
        "gnpy_profile_at_13.2THz_rel_to_peak": round(g_rel, 4),
        "statement": "pump frequencies = signal frequencies + Raman shift (gain maximum); a contiguous band wider "
                     "than the shift (about 13 THz) contains its own pumps",
        "L_band_pumps_nm": [float(NM_THZ / (thz(R['band_L'][0]) + R['raman_shift'])),
                            float(NM_THZ / (thz(R['band_L'][1]) + R['raman_shift']))],
        "C_plus_L_pumps_nm_for_1530_1625": [float(NM_THZ / (thz(1530) + R['raman_shift'])),
                                            float(NM_THZ / (thz(1625) + R['raman_shift']))],
        "pump_band_shift_between_shift_values_nm": float(max(abs(ex["pump_band_nm"][0] - ex_alt["pump_band_nm"][0]),
                                                             abs(ex["pump_band_nm"][1] - ex_alt["pump_band_nm"][1])))}

    labels = {"PDFA": "PDFA (Pr:fluoride)", "BDFA": "BDFA* (Bi:silica)", "TDFA": "TDFA (Tm:fluoride)",
              "EDFA_C": "EDFA, C-band", "EDFA_L": "EDFA, L-band", "SOA": "SOA (InP-based)",
              "Raman": "Raman (set by\npump wavelengths)", "RamanEx": "Raman, S+C+L:\npumps vs. signals"}
    ax.set_yticks([Y[k] for k in labels]); ax.set_yticklabels(list(labels.values()), fontsize=7.6)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlim(1250, 1685); ax.set_ylim(0.85, 10.62)
    ax.grid(False); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Wavelength [nm]")
    top = ax.secondary_xaxis("top", functions=(lambda x: NM_THZ / np.maximum(x, 1.0),
                                               lambda f: NM_THZ / np.maximum(f, 1e-3)))
    top.set_xticks([240, 230, 220, 210, 200, 190, 180])
    top.set_xlabel("Frequency [THz]", labelpad=3)
    top.tick_params(axis="x", labelsize=7.5)
    style.label(ax, "a", x=-0.205, y=1.1)
    results["band_map"] = {
        "windows_nm_and_THz": {k: {"edges_nm": R[k], "width_THz": round(width_thz(R[k]), 2)}
                               for k in ("win_EDFA_C", "win_EDFA_L", "win_TDFA_fluoride", "win_PDFA", "win_BDFA_O",
                                         "win_BDFA_E", "win_BDFA_S_69", "win_BDFA_17", "win_DRA_U",
                                         "win_TDFA_fluorotellurite_19", "range_BDFA_family")},
        "drawn": drawn,
        "status_strip": {"commercial_WDM": ["C", "L"], "laboratory_and_field": ["O", "E", "S", "U"]}}

    # ------------------------------------------------------------------------------------------------ (b) Er
    lam, a, g, er = perin_mccumber()
    results["er_panel_perin2019"] = er
    ax = fig.add_subplot(gs[1, 0])
    c0, c1 = R["band_C"]
    ax.axvspan(c0, c1, color="#efeeea", zorder=0, lw=0)
    band_letter(ax, 0.5 * (c0 + c1), "C")
    amax = a.max()
    ax.plot(lam, a / amax, color=ABS, label=r"$\alpha$:  $^4\mathrm{I}_{15/2}\rightarrow{}^4\mathrm{I}_{13/2}$")
    ax.plot(lam, g / amax, color=EMI, label=r"$g^*$:  $^4\mathrm{I}_{13/2}\rightarrow{}^4\mathrm{I}_{15/2}$")
    lx = er["lambda_alpha_eq_gstar_nm"]
    ax.plot([lx, lx], [0, float(np.interp(lx, lam, a)) / amax], color=INK2, lw=0.8, ls=(0, (1, 1.5)), zorder=2)
    ax.text(lx - 1.5, 0.04, r"$h\nu=\varepsilon$", fontsize=6.6, color=INK2, ha="right", va="bottom")
    ax.set_xlim(1466, 1574); ax.set_xticks([1480, 1510, 1540, 1570])
    spectro_axes(ax, "Normalized coefficient")
    style.label(ax, "b", x=-0.36, y=1.02)

    # ------------------------------------------------------------------------------------------------ (c) Pr
    ax = fig.add_subplot(gs[1, 1])
    o0, o1 = R["band_O"]
    ax.axvspan(o0, o1, color="#efeeea", zorder=0, lw=0)
    band_letter(ax, 0.5 * (o0 + o1), "O")
    lam_c = np.linspace(940, 1460, 2081)
    s_gsa = s_from_fwhm_nm(R["pr_gsa_peak"], R["pr_gsa_fwhm"])
    s_em = s_from_fwhm_nm(R["pr_em_peak"], R["pr_em_fwhm"])
    gsa = gauss_energy(lam_c, R["pr_gsa_peak"], s_gsa)
    em = gauss_energy(lam_c, R["pr_em_peak"], s_em)
    ax.plot(lam_c, gsa, color=ABS, label=r"$^3\mathrm{H}_4\rightarrow{}^1\mathrm{G}_4$ (pump band)")
    ax.plot(lam_c, em, color=EMI, label=r"$^1\mathrm{G}_4\rightarrow{}^3\mathrm{H}_5$ (signal)")
    xe = R["pr_esa_peak"]
    ax.annotate("", xy=(xe, 0.13), xytext=(xe, 0.56),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=7))
    ax.text(xe + 22, 0.6, "ESA\nfrom $^1\\mathrm{G}_4$", ha="center", va="bottom", fontsize=6.6, color=INK)
    ax.set_xlim(950, 1450); ax.set_xticks([1000, 1100, 1200, 1300, 1400])
    spectro_axes(ax, "Normalized cross section")
    style.label(ax, "c", x=-0.36, y=1.02)
    results["pr_panel"] = {
        "status": "parameterized line shapes (Gaussian in photon energy), each normalized to its own maximum; "
                  "not measured spectra",
        "gsa_3H4_1G4": {"peak_nm": R["pr_gsa_peak"], "fwhm_nm_card": R["pr_gsa_fwhm"], "s_cm": round(s_gsa, 2),
                        "peak_nm_plotted": float(lam_c[gsa.argmax()]), "fwhm_nm_plotted": fwhm_nm_numeric(lam_c, gsa)},
        "em_1G4_3H5": {"peak_nm": R["pr_em_peak"], "fwhm_nm_card": R["pr_em_fwhm"], "s_cm": round(s_em, 2),
                       "peak_nm_plotted": float(lam_c[em.argmax()]), "fwhm_nm_plotted": fwhm_nm_numeric(lam_c, em)},
        "esa_from_1G4_nm": xe}

    # ------------------------------------------------------------------------------------------------ (d) Tm
    tm = tm_parameters()
    ax = fig.add_subplot(gs[1, 2])
    u0, u1 = R["band_U"]
    ax.axvspan(u0, u1, color="#efeeea", zorder=0, lw=0)
    band_letter(ax, 0.5 * (u0 + u1), "U")
    lam_d = np.linspace(1440, 2160, 2881)
    sa = gauss_energy(lam_d, R["tm_abs_peak"], tm["s"])
    se = sa * np.exp((tm["eps"] - wavenumber(lam_d)) / tm["kT"])          # Equation (1)
    ax.plot(lam_d, sa, color=ABS, label=r"$^3\mathrm{H}_6\rightarrow{}^3\mathrm{F}_4$ absorption")
    ax.plot(lam_d, se, color=EMI, label=r"$^3\mathrm{F}_4\rightarrow{}^3\mathrm{H}_6$ emission")
    ax.set_xlim(1450, 2150); ax.set_xticks([1500, 1700, 1900, 2100])
    spectro_axes(ax, "Normalized cross section")
    style.label(ax, "d", x=-0.36, y=1.02)
    i_a, i_e = sa.argmax(), se.argmax()
    k = np.where(np.diff(np.sign(se - sa)) != 0)[0][0]
    lam_cross = float(lam_d[k] - (se - sa)[k] * (lam_d[k + 1] - lam_d[k]) / ((se - sa)[k + 1] - (se - sa)[k]))
    results["tm_panel"] = {
        "status": "parameterized line shapes; absorption = single Gaussian in photon energy, emission = its "
                  "McCumber transform (Equation 1) at T = 300 K; not measured spectra",
        "inputs_from_card": {"abs_peak_nm": R["tm_abs_peak"], "em_peak_nm": R["tm_em_peak"],
                             "sigma_a_peak_m2": R["tm_sigma_a_peak"], "sigma_e_peak_m2": R["tm_sigma_e_peak"]},
        "derived": {"peak_ratio": round(tm["ratio"], 4), "s_cm": round(tm["s"], 2),
                    "fwhm_cm": round(FWHM_PER_SIGMA * tm["s"], 1), "eps_cm": round(tm["eps"], 2),
                    "lambda_eps_nm": round(1e7 / tm["eps"], 2)},
        "plotted_checks": {"abs_peak_nm": float(lam_d[i_a]), "em_peak_nm": float(lam_d[i_e]),
                           "em_to_abs_peak_ratio": round(float(se.max() / sa.max()), 4),
                           "crossing_nm": round(lam_cross, 2),
                           "abs_fwhm_nm": round(fwhm_nm_numeric(lam_d, sa), 1),
                           "em_fwhm_nm": round(fwhm_nm_numeric(lam_d, se), 1)}}

    # ------------------------------------------------------------------------------------------------ checks, save
    pc = results["tm_panel"]["plotted_checks"]
    assert abs(pc["abs_peak_nm"] - R["tm_abs_peak"]) < 0.5 and abs(pc["em_peak_nm"] - R["tm_em_peak"]) < 0.5
    assert abs(pc["em_to_abs_peak_ratio"] - tm["ratio"]) < 1e-3
    assert abs(results["pr_panel"]["gsa_3H4_1G4"]["peak_nm_plotted"] - R["pr_gsa_peak"]) < 0.5
    assert abs(results["pr_panel"]["em_1G4_3H5"]["peak_nm_plotted"] - R["pr_em_peak"]) < 0.5

    os.makedirs(FIGDIR, exist_ok=True)
    style.save(fig, NAME, outdir=FIGDIR)
    with open(os.path.join(FIGDIR, NAME + "_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({"saved": NAME, "er_mccumber": er["mccumber_fits"], "tm": results["tm_panel"]["plotted_checks"],
                      "raman_example": ex}, indent=1))


if __name__ == "__main__":
    main()
