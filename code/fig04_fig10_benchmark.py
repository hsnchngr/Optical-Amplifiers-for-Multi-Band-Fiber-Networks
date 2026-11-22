"""Figure 6 (revision 3): recent experimental amplifier results on one bandwidth convention.

(a) Minimum in-band gain against continuous bandwidth [THz] for the results of Table 2.
    Data: ../data/table2_records.csv (one row per result, with operating point, criterion and provenance).
    Points are labeled with the first author and year of the source ('Zhai 2024'; letters a, b separate two sources
    with the same first author and year, see source_labels()), never with reference numbers, because the reference
    numbers change when the manuscript is assembled. Table 2 lists the same labels with the reference.
(b) One amplifier, several bandwidths: the S-band BDFA of Donodin 2025b (ref. [69] of the submitted version; gain
    spectra digitized from its Figure 3e,f, CC BY 4.0; ../data/donodin2025_fig3ef_digitized.csv, produced by
    digitize_donodin2025.py).

Convention (Section 3): the continuous bandwidth is the width of the widest gap-free interval in which a stated criterion
is met at a stated operating point (x-dB width, or gain above a stated level G0); it is given in THz because a width in
nm is not band-neutral. The minimum in-band gain is the lower bound of the gain inside that interval implied by the
criterion: peak - x for an x-dB width, G0 for a threshold criterion, average - variation for a flatness criterion.

The semi-quantitative radar chart of the submitted version (hand-assigned scores) has been deleted.

Table 1, row "Raman, distributed": the single-pump 3-dB band (4.9 THz at 15 dB peak on-off gain) is computed here from
the default SSMF Raman gain-efficiency profile distributed with GNPy (../data/raman_ssmf_gnpy.csv), see
raman_single_pump_band(); the values are written to the results file (key table1_raman_single_pump_band) and checked
against ../data/table1_envelope.csv.

Usage:  python fig04_fig10_benchmark.py            -> ../figures/Fig06_literature_records.png/.pdf and _results.json
        python fig04_fig10_benchmark.py --tables   -> also prints Markdown versions of Tables 1 and 2
"""
import csv
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

import cards
import style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
FIGS = os.path.normpath(os.path.join(HERE, "..", "figures"))
C_NM_THZ = 299792.458

# technology order, color and marker are fixed (color follows the technology; marker is the secondary encoding)
TECH = ["EDFA (Er, Er:Yb)", "TDFA", "BDFA", "Doped-fiber hybrid (Bi/Er, Bi/Nd)", "SOA", "Discrete Raman",
        "Band-split BDFA + discrete Raman", "Parallel hybrid EDFA/Raman"]
TECH_LABEL = {"EDFA (Er, Er:Yb)": "Er or Er:Yb fiber (L-band)", "TDFA": "Tm fiber (TDFA)", "BDFA": "Bi fiber (BDFA)",
              "Doped-fiber hybrid (Bi/Er, Bi/Nd)": "Bi/Er or Bi/Nd fiber hybrid", "SOA": "SOA",
              "Discrete Raman": "Discrete Raman", "Band-split BDFA + discrete Raman": "Band split: BDFA | Raman",
              "Parallel hybrid EDFA/Raman": "Parallel EDFA/Raman"}
COLOR = {t: style.C[i] for i, t in enumerate(TECH)}
MARKER = {"EDFA (Er, Er:Yb)": "o", "TDFA": "D", "BDFA": "s", "Doped-fiber hybrid (Bi/Er, Bi/Nd)": "P", "SOA": "^",
          "Discrete Raman": "v", "Band-split BDFA + discrete Raman": "X", "Parallel hybrid EDFA/Raman": "p"}


def thz(lo, hi):
    return C_NM_THZ / lo - C_NM_THZ / hi


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_table(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def read_digitized():
    with open(os.path.join(DATA, "donodin2025_fig3ef_digitized.csv"), encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    rows = list(csv.DictReader(lines))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def band_above(lam, g, level):
    """Edges of the contiguous interval containing the gain maximum in which the (piecewise-linear) gain >= level."""
    i0 = int(np.argmax(g))
    lo = lam[0]
    for i in range(i0, 0, -1):
        if g[i - 1] < level:
            lo = lam[i - 1] + (level - g[i - 1]) * (lam[i] - lam[i - 1]) / (g[i] - g[i - 1])
            break
    hi = lam[-1]
    for i in range(i0, len(lam) - 1):
        if g[i + 1] < level:
            hi = lam[i] + (g[i] - level) * (lam[i + 1] - lam[i]) / (g[i] - g[i + 1])
            break
    return float(lo), float(hi)


DIACRITICS = {"Gurkaynak": "G\u00fcrkaynak"}   # the data file spells names in ASCII


def source_labels(rows):
    """Author-year label of every source, e.g. {'[17]': 'Zhai 2024'}. 'Wang F.' becomes 'F. Wang'. Sources with the same
    first author and year get letters a, b, ... in the order of their first row in table2_records.csv, which is the
    order of Table 2 (for example Donodin 2024a and Donodin 2024b)."""
    first = {}
    for r in rows:
        if r["ref"] in first:
            continue
        name = r["first_author"].strip()
        m = re.match(r"^(\S+)\s+([A-Z])\.$", name)
        if m:
            name = f"{m.group(2)}. {m.group(1)}"
        for k, v in DIACRITICS.items():
            name = name.replace(k, v)
        first[r["ref"]] = f"{name} {r['year']}"
    n_same = {}
    for lab in first.values():
        n_same[lab] = n_same.get(lab, 0) + 1
    seen, out = {}, {}
    for ref, lab in first.items():
        if n_same[lab] > 1:
            seen[lab] = seen.get(lab, 0) + 1
            lab += "abcdefgh"[seen[lab] - 1]
        out[ref] = lab
    return out


# ------------------------------------------------------------------------------------------------------------------
rows = read_table("table2_records.csv")
dig = read_digitized()
SRC = source_labels(rows)
# Which rows are plotted is decided only by the column 'plot' of table2_records.csv (row 17d: plot = yes since the
# decision of 18 Nov 2026, recorded in the notes of that row; the former override in this script was removed).

# consistency check: THz values in the file agree with the band edges
thz_check = []
for r in rows:
    lo, hi, b = fnum(r["band_lo_nm"]), fnum(r["band_hi_nm"]), fnum(r["bw_THz"])
    if lo and hi and b is not None:
        thz_check.append(abs(thz(lo, hi) - b))
assert max(thz_check) < 0.006, "bw_THz column inconsistent with band edges"

# ---------------------------------------------------- worked example of ref. [69] (re-derived from the digitized data)
lam = dig["lambda_nm"]
g20, g0 = dig["gain_m20dBm_dB"], dig["gain_0dBm_dB"]
PEAK_STATED_M20 = 35.6  # dB, text and Table 1 of [69]
ex = {}
for key, (g, level, crit) in {
    "3dB_m20_stated_peak": (g20, PEAK_STATED_M20 - 3.0, "3-dB width, -20 dBm (stated peak 35.6 dB)"),
    "3dB_m20_digitized_peak": (g20, g20.max() - 3.0, "3-dB width, -20 dBm (digitized peak)"),
    "G_ge_20_m20": (g20, 20.0, "G >= 20 dB, -20 dBm"),
    "G_ge_10_m20": (g20, 10.0, "G >= 10 dB, -20 dBm"),
    "3dB_0dBm": (g0, g0.max() - 3.0, "3-dB width, 0 dBm"),
}.items():
    lo, hi = band_above(lam, g, level)
    ex[key] = {"criterion": crit, "level_dB": round(level, 2), "edges_nm": [round(lo, 1), round(hi, 1)],
               "width_nm": round(hi - lo, 1), "width_THz": round(thz(lo, hi), 2),
               "limited_by_measured_range": [bool(lo <= lam[0] + 1e-6), bool(hi >= lam[-1] - 1e-6)]}

# ---------------------------------------- Table 1, row "Raman, distributed": single-pump 3-dB band (model, reproducible)
def raman_single_pump_band(peaks_dB=(10.0, 15.0, 20.0), lam_nm=1550.0):
    """Width of the band in which the on-off gain of ONE Raman pump stays within 3 dB of its peak, computed from the
    default SSMF Raman gain-efficiency profile distributed with GNPy (data/raman_ssmf_gnpy.csv).
    Undepleted pump: the on-off gain in dB is proportional to the gain efficiency, G(df) = G_peak g(df)/g_peak, so
    the 3-dB band is the contiguous interval around the profile peak in which g(df) >= (1 - 3/G_peak) g_peak; its
    edges are found by linear interpolation of the tabulated profile (band_above). The width does not depend on span
    length, attenuation, pump power or pump wavelength (the profile is scaled with the pump frequency only)."""
    off, g0, _ = cards.raman_profile_ssmf()
    out = {"source": "data/raman_ssmf_gnpy.csv (default SSMF Raman gain-efficiency profile distributed with GNPy)",
           "model": "single undepleted pump; on-off gain in dB proportional to the gain efficiency; band = contiguous "
                    "interval around the profile peak in which the on-off gain is within 3 dB of its peak; edges by "
                    "linear interpolation of the tabulated profile",
           "peak_on_off_gain": {}}
    for G in peaks_dB:
        lo, hi = band_above(off, g0, (1.0 - 3.0 / G) * g0.max())
        out["peak_on_off_gain"][f"{G:g} dB"] = {
            "level_rel_to_peak_efficiency": round(1.0 - 3.0 / G, 4),
            "edges_THz_below_pump": [round(lo, 3), round(hi, 3)], "width_THz": round(hi - lo, 3),
            f"width_nm_at_{lam_nm:g}nm": round((hi - lo) * lam_nm ** 2 / C_NM_THZ, 1)}
    lo, hi = band_above(off, g0, 0.5 * g0.max())
    out["half_maximum_edges_THz"] = [round(lo, 3), round(hi, 3)]
    out["half_maximum_width_THz"] = round(hi - lo, 3)
    # cross-check with the value written in Table 1 (data/table1_envelope.csv): 15 dB peak on-off gain
    row = next((r for r in read_table("table1_envelope.csv") if r["technology"] == "Raman, distributed"), None)
    m = re.match(r"\s*([0-9.]+)", row["band_THz"]) if row else None
    w15 = out["peak_on_off_gain"]["15 dB"]["width_THz"]
    out["table1_check"] = {"row": "Raman, distributed", "table1_band_THz": row["band_THz"] if row else None,
                           "computed_15dB_THz_rounded": round(w15, 1),
                           "agrees": bool(m) and abs(float(m.group(1)) - round(w15, 1)) < 1e-9}
    if not out["table1_check"]["agrees"]:
        print("WARNING: Table 1 Raman single-pump band differs from the computed value", out["table1_check"])
    return out


raman_band = raman_single_pump_band()

# ------------------------------------------------------------------------------------------------------------------ plot
style.apply()
LABEL_FS = 6.3   # point labels of panel a
fig = plt.figure(figsize=(7.1, 3.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.38, 1.0], height_ratios=[2.15, 1.0], wspace=0.30, hspace=0.10,
                      left=0.075, right=0.985, top=0.94, bottom=0.12)
ax = fig.add_subplot(gs[:, 0])
axb = fig.add_subplot(gs[0, 1])
axs = fig.add_subplot(gs[1, 1], sharex=axb)

plotted, not_plotted = [], []
for r in rows:
    x, y = fnum(r["bw_THz"]), fnum(r["g_min_in_band_dB"])
    if r["plot"] != "yes" or x is None or y is None:
        not_plotted.append({"row_id": r["row_id"], "ref": r["ref"], "reason": r["notes"] or r["bw_criterion"]})
        continue
    filled = r["conditions_complete"] == "yes"
    p = {"row_id": r["row_id"], "ref": r["ref"], "label": SRC[r["ref"]], "technology": r["technology"], "bw_THz": x,
         "bw_THz_unc": fnum(r["bw_THz_unc"]) or 0.0, "g_min_dB": y, "filled": filled,
         "span_THz": fnum(r["span_THz"]), "criterion": r["bw_criterion"], "input_power": r["input_power"],
         "source_access": r["source_access"], "group": r["plot_group"]}
    plotted.append(p)

# points of the same amplifier at different input powers are not joined (a line would cross other records);
# they carry the input power in their labels instead
# spans including gaps (band-split or pump guard bands)
for p in plotted:
    if p["span_THz"] and p["span_THz"] > p["bw_THz"] + 0.5:
        ax.plot([p["bw_THz"], p["span_THz"]], [p["g_min_dB"]] * 2, ls=(0, (1, 1.6)), lw=1.0,
                color=COLOR[p["technology"]], zorder=2)
        ax.plot([p["span_THz"]], [p["g_min_dB"]], marker="|", ms=7, mew=1.0, color=COLOR[p["technology"]], zorder=2)
# uncertainty of the THz value where band edges are not stated
for p in plotted:
    if p["bw_THz_unc"] >= 0.1:
        ax.errorbar(p["bw_THz"], p["g_min_dB"], xerr=p["bw_THz_unc"], fmt="none", ecolor=style.INK2, elinewidth=0.7,
                    capsize=2.0, zorder=3)
for p in plotted:
    c = COLOR[p["technology"]]
    ax.scatter(p["bw_THz"], p["g_min_dB"], s=34, marker=MARKER[p["technology"]], facecolor=c if p["filled"] else "white",
               edgecolor=c, linewidth=1.2, zorder=4)

# point labels: author-year label of the source (SRC) + operating point where needed. Mode "pt": (dx, dy) is an offset
# in points from the marker; mode "data": (dx, dy) is the label position in data units, joined to the marker by a thin
# leader line (used for the crowded group near 10 THz and 20 dB, and for the two adjacent markers near 10-12 THz and
# 10 dB: Donodin 2024b to the left, Hazarika 2022b to the right, each with its own leader). Positions chosen by eye to
# avoid data and labels.
U = "\u2212"
LABELS = {  # row_id: (suffix, dx, dy, ha, mode)
    "34": (", tunable,\n< 1 nm", 2, -11, "left", "pt"),
    "82": ("", -6, 0, "right", "pt"),
    "36a": (f", {U}25 dBm", 7, 0, "left", "pt"),
    "36b": (", +5 dBm", 6, -4, "left", "pt"),
    "66": ("", -6, 0, "right", "pt"),
    "83": ("", 6, -2, "left", "pt"),
    "73": ("", 0, -8, "center", "pt"),
    "19a": ("", 7, -2, "left", "pt"),
    "17d": (f", {U}23 dBm, 6-dB width", 7, 3, "left", "pt"),
    "79": ("", 7.9, 25.0, "right", "data"),
    "80a": ("", 7.9, 26.8, "right", "data"),
    "77a": ("", 7.9, 28.6, "right", "data"),
    "81a": ("", 7, -2, "left", "pt"),
    "69a": ("", -6, -6, "right", "pt"),
    "71a": ("", 13.4, 7.4, "left", "data"),
    "76a": ("", 9.0, 7.4, "right", "data"),
    "22b": (", +6 dBm", 7, 0, "left", "pt"),
    "17b": (f", {U}10 dBm", 0, 8, "center", "pt"),
    "17a": (f",\n{U}23 dBm, G > 20 dB", -7, 0, "right", "pt"),
    "17c": (", 0 dBm", 7, 0, "left", "pt"),
    "78a": ("", 3, -9, "center", "pt"),
    "75": ("", 0, 8, "center", "pt"),
}
pid = {p["row_id"]: p for p in plotted}
missing_labels = sorted(set(pid) - set(LABELS))
assert not missing_labels, f"plotted points without a label: {missing_labels}"
for rid, (suffix, dx, dy, ha, mode) in LABELS.items():
    if rid not in pid:
        continue
    p = pid[rid]
    leader = None
    if mode == "data":   # leader starts at the end of the label nearest to the marker
        leader = dict(arrowstyle="-", lw=0.45, color=style.INK2, shrinkA=1.5, shrinkB=3.8,
                      relpos={"right": (1.0, 0.5), "left": (0.0, 0.5)}.get(ha, (0.5, 0.5)))
    ax.annotate(p["label"] + suffix, (p["bw_THz"], p["g_min_dB"]), xytext=(dx, dy),
                textcoords="data" if mode == "data" else "offset points", fontsize=LABEL_FS, color=style.INK,
                ha=ha, va="center", zorder=5, linespacing=1.0, arrowprops=leader)
ax.set_xlim(-0.6, 28.8)
ax.set_ylim(5, 48)
ax.set_yticks(np.arange(5, 41, 5))
ax.set_xlabel("Continuous bandwidth [THz]")
ax.set_ylabel("Minimum in-band gain [dB]")
style.label(ax, "a", x=-0.12)

# one legend in the empty band above the data: technology (color + marker) and encodings (fill, lines)
from matplotlib.lines import Line2D
tech_present = [t for t in TECH if any(p["technology"] == t for p in plotted)]
h_tech = [Line2D([], [], ls="none", marker=MARKER[t], ms=5.0, mfc=COLOR[t], mec=COLOR[t], label=TECH_LABEL[t])
          for t in tech_present]
h_enc = [Line2D([], [], ls="none", marker="o", ms=5.0, mfc=style.INK2, mec=style.INK2, label="conditions complete"),
         Line2D([], [], ls="none", marker="o", ms=5.0, mfc="white", mec=style.INK2, label="conditions incomplete"),
         Line2D([], [], color=style.INK2, lw=1.0, ls=(0, (1, 1.6)), label="span including gaps")]
ax.legend(handles=h_tech + h_enc, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.3,
          handletextpad=0.3, columnspacing=0.9, labelspacing=0.28, borderpad=0.2, handlelength=1.5, frameon=False)

# ---------------------------------------------------------------------------------------------- (b) worked example [69]
green = COLOR["BDFA"]
for a in (axb, axs):
    a.axvspan(1455, lam[0], color=style.GRID, alpha=0.55, lw=0, zorder=0)
    a.axvspan(lam[-1], 1540, color=style.GRID, alpha=0.55, lw=0, zorder=0)
axb.plot(lam, g20, color=green, lw=1.4, marker="s", ms=3.2, mfc=green, mec=green, zorder=3, label="\u221220 dBm input")
axb.plot(lam, g0, color=green, lw=1.2, ls="--", marker="s", ms=3.2, mfc="white", mec=green, zorder=3,
         label="0 dBm input")
# the four bandwidth statements: edges as stated in [69] where stated, otherwise re-derived here (dagger)
STATED = {"G_ge_10_m20": (1465.0, 1530.0), "G_ge_20_m20": (1465.0, 1502.5), "3dB_0dBm": (1465.0, 1505.0)}
stmt = []
for key, txt, solid, lvl in (("3dB_m20_stated_peak", "3-dB width, \u221220 dBm\u2020", True, PEAK_STATED_M20 - 3.0),
                             ("G_ge_20_m20", "G > 20 dB, \u221220 dBm", True, 20.0),
                             ("3dB_0dBm", "3-dB width, 0 dBm", False, 21.2 - 3.0),
                             ("G_ge_10_m20", "G > 10 dB, \u221220 dBm", True, 10.0)):
    lo, hi = STATED.get(key, tuple(ex[key]["edges_nm"]))
    stmt.append({"key": key, "label": txt, "solid": solid, "level_dB": lvl, "edges_nm": [lo, hi],
                 "width_nm": round(hi - lo, 1), "width_THz": round(thz(lo, hi), 2),
                 "basis": "stated in [69]" if key in STATED else "re-derived from the digitized curve"})
# threshold levels drawn over their intervals
for s in stmt:
    axb.plot(s["edges_nm"], [s["level_dB"]] * 2, color=style.INK2, lw=0.8, ls=(0, (2, 1.5)), zorder=2)
axb.annotate("\u22123 dB", (stmt[0]["edges_nm"][1], stmt[0]["level_dB"]), textcoords="offset points", xytext=(3, 3),
             fontsize=6.2, color=style.INK2, ha="left", va="bottom")
axb.annotate("\u22123 dB", (stmt[2]["edges_nm"][0], stmt[2]["level_dB"]), textcoords="offset points", xytext=(2, -2),
             fontsize=6.2, color=style.INK2, ha="left", va="top")
axb.text(1497.5, 37.2, "measured range 1465\u20131530 nm", fontsize=6.2, color=style.INK2, ha="center", va="top")
axb.set_xlim(1458, 1537)
axb.set_ylim(8, 38)
axb.set_ylabel("Gain [dB]")
axb.tick_params(labelbottom=False)
axb.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6.4, handlelength=2.2, frameon=False)
style.label(axb, "b", x=-0.20)

# interval strip: one bar per bandwidth statement
for i, s in enumerate(stmt):
    yb = len(stmt) - 1 - i
    w = s["edges_nm"][1] - s["edges_nm"][0]
    axs.barh(yb, w, left=s["edges_nm"][0], height=0.36, color=green if s["solid"] else "white", edgecolor=green,
             lw=1.0, zorder=3)
    axs.text(s["edges_nm"][0] + 0.8, yb + 0.25, f"{s['label']}: {s['width_nm']:g} nm ({s['width_THz']:.1f} THz)",
             fontsize=6.3, color=style.INK, ha="left", va="bottom", zorder=4)
bars = stmt
axs.set_ylim(-0.45, len(bars) - 0.05)
axs.set_yticks([])
axs.grid(False, axis="y")
axs.spines["left"].set_visible(False)
axs.set_xlabel("Wavelength [nm]")

os.makedirs(FIGS, exist_ok=True)
style.save(fig, "Fig06_literature_records", outdir=FIGS)

# ------------------------------------------------------------------------------------------------------------ results
gmin_wide = [p for p in plotted if p["bw_THz"] > 9.0]
results = {
    "figure": "Fig06_literature_records",
    "script": "fig04_fig10_benchmark.py",
    "data_files": ["data/table2_records.csv", "data/donodin2025_fig3ef_digitized.csv"],
    "convention": "x = width of the widest gap-free interval meeting the stated criterion at the stated operating point "
                  "[THz]; y = lower bound of the gain inside that interval implied by the criterion [dB]",
    "marker_fill": "filled = operating conditions complete in an accessible full text, preprint or accepted manuscript; "
                   "open = conditions incomplete or only an abstract accessible",
    "point_labels": "first author and year of the source (key source_labels); letters a, b separate two sources with "
                    "the same first author and year, in the order of table2_records.csv (= order of Table 2)",
    "source_labels": SRC,
    "n_rows": len(rows), "n_sources": len(SRC), "n_plotted": len(plotted), "n_filled": sum(p["filled"] for p in plotted),
    "n_open": sum(not p["filled"] for p in plotted),
    "rows_by_source_access": {k: sum(r["source_access"] == k for r in rows) for k in sorted({r["source_access"] for r in rows})},
    "panel_a_points": plotted,
    "not_plotted": not_plotted,
    "observations": {
        "max_gmin_with_bw_above_9THz": max(gmin_wide, key=lambda p: p["g_min_dB"])["row_id"] if gmin_wide else None,
        "max_gmin_with_bw_above_9THz_dB": max(p["g_min_dB"] for p in gmin_wide) if gmin_wide else None,
        "points_with_gmin_above_24dB_whose_error_bar_reaches_above_9THz": [
            (p["row_id"], p["bw_THz"], p["bw_THz_unc"], round(p["bw_THz"] + p["bw_THz_unc"], 2), p["g_min_dB"])
            for p in plotted if p["g_min_dB"] > 24 and p["bw_THz"] + p["bw_THz_unc"] > 9.0],
        "widest_bw_with_gmin_ge_20dB": max((p for p in plotted if p["g_min_dB"] >= 20), key=lambda p: p["bw_THz"])["row_id"],
        "points_with_gmin_ge_30dB": [(p["row_id"], p["bw_THz"], p["g_min_dB"]) for p in plotted if p["g_min_dB"] >= 30],
        "open_points": [(p["row_id"], p["label"]) for p in plotted if not p["filled"]],
    },
    "worked_example_ref69": {
        "device": "400 m Bi:germanosilicate BDFA, single stage and pass, 950 mW at 1425 nm (bidirectional) [69]",
        "re_derived_from": "vector data of Figure 3e,f of [69] (5 nm measurement steps, linear interpolation)",
        "statements_plotted": stmt,
        "widths_re_derived": ex,
        "source_stated": {"G_gt_10_m20": "1465-1530 nm (65 nm)", "G_gt_20_m20": "1465-1502.5 nm (37.5 nm)",
                          "3dB_0dBm": "1465-1505 nm (40 nm)", "3dB_m20": "not stated"},
        "digitized_extremes_vs_source_table1": {
            "gain_m20_dB": [round(float(g20.min()), 2), round(float(g20.max()), 2), "source 11.3-35.6"],
            "gain_0_dB": [round(float(g0.min()), 2), round(float(g0.max()), 2), "source 10.6-21.2"],
            "nf_m20_dB": [round(float(dig["nf_m20dBm_dB"].min()), 2), round(float(dig["nf_m20dBm_dB"].max()), 2), "source 5.6-7.7"],
            "nf_0_dB": [round(float(dig["nf_0dBm_dB"].min()), 2), round(float(dig["nf_0dBm_dB"].max()), 2), "source 5.9-7.8"]},
    },
    "checks": {"max_abs_THz_column_vs_edges": round(max(thz_check), 4)},
    "table1_raman_single_pump_band": raman_band,
}
with open(os.path.join(FIGS, "Fig06_literature_records_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, ensure_ascii=True)
print("Fig06 written:", len(plotted), "points plotted,", len(not_plotted), "rows not plotted")
for k, e in ex.items():
    print(f"  [69] {e['criterion']:45s} {e['edges_nm'][0]:.1f}-{e['edges_nm'][1]:.1f} nm  {e['width_nm']:5.1f} nm  {e['width_THz']:.2f} THz")


# ------------------------------------------------------------------------------------------------- Markdown tables
def md(s):
    """ASCII data text -> manuscript typography (ranges, minus signs, +-, degrees, micro)."""
    s = s.replace("+-", "\u00b1").replace("degC", "\u00b0C").replace(" um", " \u00b5m").replace("||", "\u2016")
    s = s.replace("MHz-", "MHz\u2013")
    s = re.sub(r"(?<=\d) x (?=\d)", " \u00d7 ", s)
    s = re.sub(r"(?<=\d)-(?=\d)", "\u2013", s)
    s = re.sub(r"(^|[\s(/;,])-(?=\d)", "\\1\u2212", s)
    s = s.replace(">=", "\u2265").replace("<=", "\u2264")
    return s.replace("|", "/")


def table2_md(rows):
    out = ["| Ref. (year) | Amplifier | Operating point: input; pump or bias | Gain [dB] (type; statistic) | "
           "Continuous band: edges [nm]; width [nm] ([THz]); criterion | Min. in-band gain [dB] | NF [dB] (statistic; conditions) | "
           "Same op. point | Access |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        edges = f"{r['band_lo_nm']}-{r['band_hi_nm']}; " if r["band_lo_nm"] else ""
        width = f"{r['bw_nm']} ({r['bw_THz']})" if r["bw_nm"] and r["bw_THz"] else (r["bw_nm"] or "")
        band = f"{edges}{width}; {r['bw_criterion']}" if width else r["bw_criterion"]
        if r["span_THz"]:
            band += f"; span {r['span_lo_nm']}-{r['span_hi_nm']} nm ({r['span_THz']} THz) including gaps"
        def paren(*parts):
            parts = [p for p in parts if p]
            return f" ({'; '.join(parts)})" if parts else ""
        gain = f"{r['gain_dB']}{paren(r['gain_type'], r['gain_stat'])}" if r["gain_dB"] else "n.s."
        nf = r["nf_dB"] + paren(r["nf_statistic"], r["nf_conditions"])
        cells = [f"{r['ref']} {r['row_id'][len(r['ref']) - 2:]} ({r['year']})".replace("  ", " "), r["configuration"],
                 f"{r['input_power']}; {r['pump_or_bias']}", gain, band, r["g_min_in_band_dB"] or "n.d.", nf,
                 r["same_operating_point"], r["source_access"]]
        out.append("| " + " | ".join(md(c) for c in cells) + " |")
    return "\n".join(out)


def table1_md(rows):
    out = ["| Technology | Host or structure | Maturity | Continuous single-device band [nm] ([THz]); criterion | "
           "Gain [dB] (conditions) | NF [dB] (conditions; reference plane) | Output power | Pump | Sources |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        band = (f"{r['band_lo_nm']}-{r['band_hi_nm']} ({r['band_THz']}); " if r["band_lo_nm"] else
                (f"({r['band_THz']}); " if r["band_THz"] else "")) + r["band_criterion"]
        if r["band_other"]:
            band += "; " + r["band_other"]
        cells = [r["technology"], r["host_or_structure"], f"{r['maturity']} ({r['maturity_basis']})", band, r["gain"],
                 r["nf"], r["output_power"], r["pump"], r["sources"]]
        out.append("| " + " | ".join(md(c) for c in cells) + " |")
    return "\n".join(out)


if "--tables" in sys.argv:
    print("\n### Table 1\n")
    print(table1_md(read_table("table1_envelope.csv")))
    rb = raman_band["peak_on_off_gain"]
    print("\nTable 1, distributed Raman, single pump (model, " + raman_band["source"] + "): on-off gain within 3 dB of "
          "its peak over " + " / ".join(f"{v['width_THz']:.2f}" for v in rb.values()) + " THz ("
          + " / ".join(f"{v['width_nm_at_1550nm']:.1f}" for v in rb.values()) + " nm at 1550 nm) for "
          + " / ".join(k.split()[0] for k in rb) + " dB peak on-off gain; half-maximum width of the profile "
          f"{raman_band['half_maximum_width_THz']:.2f} THz; Table 1 value "
          f"'{raman_band['table1_check']['table1_band_THz']}' reproduced: {raman_band['table1_check']['agrees']}")
    print("\n### Table 2 (full)\n")
    print(table2_md(rows))
