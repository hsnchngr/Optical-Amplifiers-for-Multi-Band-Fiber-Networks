"""Figure 12 (Fig12_end_to_end): end-to-end case study on the open EDFA of ref. [124] (Section 5.7).

Reads ../figures/Fig12_end_to_end_results.json (written by e2e_perin.py, which is run if the file is missing),
../figures/E2E_simulator_results.json (written by e2e_simulator_compare.py from the saved OptiSystem run files;
the script stops if that file is missing or incomplete), the card spectra through cards.py and the digitized
measurement ../data/perin2019_fig8_gain_ase.json.
Writes ../figures/Fig12_end_to_end.png and .pdf.

Panels
  a) L1 card: digitized alpha(lambda) and g*(lambda) of [124], with the McCumber transform of alpha (300 K).
  b) L1 V2 test: deviation of the McCumber transform from g* at 300 K and for a free-temperature fit. The printed
     values are the decision values of Section 5.7, computed here from the card data with e2e_perin.mccumber_test:
     at 300 K the crossing wavelength is the median of the pointwise values over 1500-1565 nm (shaded) and the
     maximum deviation is taken over that band; the free fit (temperature and crossing wavelength) uses the whole
     table. They are checked against the L1 block of the results file.
  c) L2/L4 gain: measured (markers; filled = calibration pump powers, open = held-out pump powers), nominal card
     (dashed) and calibrated card (solid), ASE-resolved Giles model.
  d) ASE in 0.1 nm: measured and model; the ASE was never used in the identification (blind).
  e) Error budget (dB, logarithmic): numerical, code-check, independent-simulator, model-form, parameter-uncertainty
     and validation terms. The two simulator rows are the RMSE of OptiSystem against the open ASE-resolved model
     (nominal card, 40 channels x 6 pump powers): gain and ASE with each simulator noise bin read at its center (the
     headline values), and the ASE with each bin read at its short-wavelength edge, where the bins carry the
     spectrum (key comparisons.ase_bin_placement of E2E_simulator_results.json). Row labels name sources by author,
     not by reference number.
  f) L3 design map with the calibrated card: gain ripple over EDF length and pump power, 10 dB mean-gain contours
     of both cards and the ripple-optimal designs.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

import style
import e2e_perin as E

MARKERS = ["o", "s", "^", "D", "v", "P"]
MC_BAND_NM = (1500.0, 1565.0)        # band of the 300 K McCumber test (decision values of Section 5.7)
SIM_PATH = os.path.join(E.FIG_DIR, "E2E_simulator_results.json")   # written by e2e_simulator_compare.py


def simulator_rows():
    """Error-budget rows of the independent simulator (layer L2, source iii): RMSE of OptiSystem minus the open
    ASE-resolved model, nominal card, 40 channels x 6 pump powers, as computed by e2e_simulator_compare.py from the
    saved run files. Returns (gain at bin centers, ASE at bin centers, ASE at the short-wavelength bin edges)."""
    if not os.path.exists(SIM_PATH):
        raise FileNotFoundError(f"{SIM_PATH} is missing; run e2e_simulator_compare.py first (it needs only the "
                                "saved run files in ../notes/optisystem_perin2019/runs, not OptiSystem)")
    with open(SIM_PATH, encoding="utf-8") as f:
        S = json.load(f)
    if S.get("status") != "complete":
        raise RuntimeError(f"{SIM_PATH}: status '{S.get('status')}', not 'complete'; the simulator rows of panel e "
                           "need all run sets")
    c = S["comparisons"]
    edge = c["ase_bin_placement"]["short_wavelength_edge"]["N"]["ase_vs_open_model"]["overall"]["rmse"]
    center = c["ase_bin_placement"]["centre"]["N"]["ase_vs_open_model"]["overall"]["rmse"]   # key spelled 'centre'
    ase = c["N_vs_open_model"]["ase"]["overall"]["rmse"]
    if abs(center - ase) > 1e-6:                    # the headline statistic is the center reading
        raise RuntimeError("E2E_simulator_results.json: headline ASE statistic is not the bin-center reading")
    return c["N_vs_open_model"]["gain"]["overall"]["rmse"], ase, edge


def mccumber_gstar(lam, alpha, lam_x, T):
    return alpha * E.mccumber_ratio(lam, lam_x, T)


def mccumber_decision_values(L1):
    """McCumber test of the card data (layer L1), computed here with e2e_perin.mccumber_test: at 300 K the median of
    the pointwise crossing wavelengths over MC_BAND_NM and the maximum deviation of the McCumber transform of alpha
    from g* over that band; the free fit of temperature and crossing wavelength over the whole table. Checked against
    the L1 block of the results file (written by e2e_perin.py from the same card), which must not be stale."""
    dev, _ = E.load_device()
    mc = E.mccumber_test(dev, 300.0, band=MC_BAND_NM)
    stored = L1["mccumber_300K"]
    checks = ((mc["band"]["lambda_x_median_nm"], stored["band"]["lambda_x_median_nm"]),
              (mc["band"]["max_dev_gstar_percent"], stored["band"]["max_dev_gstar_percent"]),
              (mc["free_T_fit"]["T_K"], stored["free_T_fit"]["T_K"]),
              (mc["free_T_fit"]["lambda_x_nm"], stored["free_T_fit"]["lambda_x_nm"]),
              (mc["free_T_fit"]["max_dev_percent"], stored["free_T_fit"]["max_dev_percent"]))
    for new, old in checks:
        if abs(new - old) > 1e-5 * max(abs(old), 1.0):
            raise RuntimeError(f"McCumber test of the card ({new}) differs from the results file ({old}); "
                               "rerun e2e_perin.py")
    if max(abs(float(a) - float(b)) for a, b in zip(stored["band"]["range_nm"], mc["band"]["range_nm"])) > 1e-6:
        raise RuntimeError("band of the McCumber test differs from the results file; rerun e2e_perin.py")
    return dict(lam=dev.lam_tab, alpha=dev.alpha_tab, gstar=dev.gstar_tab,
                lx300=mc["band"]["lambda_x_median_nm"], maxdev300_band=mc["band"]["max_dev_gstar_percent"],
                band_nm=mc["band"]["range_nm"], free=mc["free_T_fit"])


def clear_label_positions(ax, cs, avoid, ymax_frac=0.70, margin=0.06):
    """One label position per contour level: the vertex with the largest distance (axes fraction) from the
    points in `avoid` (data coordinates), inside the axes and below the legend (y < ymax_frac)."""
    to_ax = ax.transData + ax.transAxes.inverted()
    A = to_ax.transform(np.vstack(avoid))
    pos = []
    for path in cs.get_paths():
        v = path.vertices
        if len(v) < 2:
            continue
        F = to_ax.transform(v)
        ok = (F[:, 0] > margin) & (F[:, 0] < 1 - margin) & (F[:, 1] > margin) & (F[:, 1] < ymax_frac)
        if not ok.any():
            continue
        d = np.min(np.hypot(F[ok, None, 0] - A[None, :, 0], F[ok, None, 1] - A[None, :, 1]), axis=1)
        pos.append(tuple(v[ok][np.argmax(d)]))
    return pos


def main():
    if not os.path.exists(E.RESULTS_PATH):
        E.main()
    with open(E.RESULTS_PATH, encoding="utf-8") as f:
        R = json.load(f)
    meas, _ = E.load_measurement()
    pumps = R["device"]["pumps_mW"]
    lam_s = np.array(R["device"]["lambda_ch_nm"])
    cal_set = set(R["L4"]["splits"]["interleaved"]["cal"])
    chosen = tuple(R["L4"]["chosen"])
    cal_entry = next(e for e in R["L4"]["interleaved"] if tuple(e["fit"]["names"]) == chosen)

    style.apply()
    W_IN = 7.1                                      # double-column width; every artist stays inside the canvas
    fig = plt.figure(figsize=(W_IN, 8.3))
    gs = fig.add_gridspec(3, 2, left=0.085, right=0.98, top=0.975, bottom=0.06, hspace=0.42, wspace=0.30,
                          height_ratios=[1.0, 1.15, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    # bottom row: explicit rectangles aligned with the columns above. Panel e leaves 1.6 in for its row labels (the
    # longest is 1.4 in at 7 pt); panel f leaves room for its colorbar, tick labels and colorbar label.
    pe, pf = gs[2, 0].get_position(fig), gs[2, 1].get_position(fig)
    e_left = 1.62 / W_IN
    ax_e = fig.add_axes([e_left, pe.y0, pe.x1 - e_left, pe.height])
    f_right = pf.x1 - 0.60 / W_IN
    ax_f = fig.add_axes([pf.x0, pf.y0, f_right - pf.x0, pf.height])
    cax = fig.add_axes([f_right + 0.06 / W_IN, pf.y0, 0.12 / W_IN, pf.height])

    # ---------------- a) card spectra ----------------
    L1 = R["L1"]
    lam = np.array(L1["lam_tab"])
    al = np.array(L1["alpha_tab"])
    gs_ = np.array(L1["gstar_tab"])
    ax_a.axvspan(lam_s.min(), lam_s.max(), color=style.GRID, alpha=0.45, lw=0)
    ax_a.plot(lam, al, color=style.INK, lw=1.5)
    ax_a.plot(lam, gs_, color=style.C[6], lw=1.5)
    ax_a.text(1478, 0.80, r"$\alpha(\lambda)$", color=style.INK, fontsize=8)
    ax_a.text(1505, 0.30, r"$g^*(\lambda)$", color=style.C[6], fontsize=8)
    ax_a.text(0.5 * (lam_s.min() + lam_s.max()), 0.04, "channels", ha="center", fontsize=6.5, color=style.INK2)
    ax_a.set_xlim(1470, 1569)
    ax_a.set_ylim(0, 1.95)
    ax_a.set_xlabel("Wavelength [nm]")
    ax_a.set_ylabel("Coefficient [1/m]")
    style.label(ax_a, "a")

    # ---------------- b) McCumber test (decision values computed from the card data) ----------------
    mcv = mccumber_decision_values(L1)
    lam_b, al_b, gs_b = mcv["lam"], mcv["alpha"], mcv["gstar"]
    lx300, ft = mcv["lx300"], mcv["free"]
    d300 = 100 * (mccumber_gstar(lam_b, al_b, lx300, 300.0) / gs_b - 1)
    dfit = 100 * (mccumber_gstar(lam_b, al_b, ft["lambda_x_nm"], ft["T_K"]) / gs_b - 1)
    in_band = (lam_b >= MC_BAND_NM[0]) & (lam_b <= MC_BAND_NM[1])
    assert abs(np.max(np.abs(d300[in_band])) - mcv["maxdev300_band"]) < 1e-9     # same curve as the printed value
    ax_b.axvspan(*MC_BAND_NM, color=style.GRID, alpha=0.45, lw=0)
    ax_b.axhline(0, color=style.INK2, lw=0.6)
    ax_b.plot(lam_b, d300, color=style.INK, lw=1.4)
    ax_b.plot(lam_b, dfit, color=style.C[6], lw=1.4)
    y_lo = float(np.floor(min(d300.min(), dfit.min()) - 0.5))
    y_hi = float(np.ceil(max(d300.max(), dfit.max()) + 0.5))
    ax_b.text(1566, y_hi - 0.2, f"300 K, $\\lambda_x$ = {lx300:.1f} nm\n"
              f"(median over {MC_BAND_NM[0]:.0f}–{MC_BAND_NM[1]:.0f} nm, shaded)\n"
              f"max |deviation| there {mcv['maxdev300_band']:.1f} %", color=style.INK, fontsize=7, ha="right",
              va="top")
    ax_b.text(1474, -0.45, f"free fit: {ft['T_K']:.1f} K, $\\lambda_x$ = {ft['lambda_x_nm']:.1f} nm\n"
              f"(max |deviation| {ft['max_dev_percent']:.3f} %)", color=style.C[6], fontsize=7, va="top")
    ax_b.set_xlim(1470, 1569)
    ax_b.set_ylim(y_lo, y_hi)
    ax_b.set_xlabel("Wavelength [nm]")
    ax_b.set_ylabel("McCumber $g^*$ deviation [%]")
    style.label(ax_b, "b")

    # ---------------- c) gain ----------------
    Gnom = R["L2"]["spectra"]["giles_nominal_G"]
    Gcal = cal_entry["spectra"]["G"]
    for i, p in enumerate(pumps):
        col = style.C[i]
        key = f"{p:g}"
        filled = p in cal_set
        ax_c.plot(lam_s, Gnom[key], color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.8)
        ax_c.plot(lam_s, Gcal[key], color=col, lw=1.4)
        ax_c.plot(meas[p]["lam"][::2], meas[p]["G"][::2], ls="none", marker=MARKERS[i], ms=3.6, mew=0.9,
                  mfc=col if filled else "white", mec=col)
    ax_c.set_xlim(1529.5, 1563)
    ax_c.set_ylim(0, 22.5)
    ax_c.set_xlabel("Wavelength [nm]")
    ax_c.set_ylabel("Gain [dB]")
    h_p = [Line2D([], [], color=style.C[i], marker=MARKERS[i], ms=3.6, lw=1.2, label=f"{p:g} mW")
           for i, p in enumerate(pumps)]
    ax_c.legend(handles=h_p, ncol=3, loc="upper left", fontsize=6.8, handlelength=1.6, columnspacing=0.9,
                handletextpad=0.4, borderaxespad=0.2, title="Pump power", title_fontsize=6.8)
    style.label(ax_c, "c")

    # ---------------- d) ASE ----------------
    Anom = R["L2"]["spectra"]["giles_nominal_ASE"]
    Acal = cal_entry["spectra"]["ASE"]
    for i, p in enumerate(pumps):
        col = style.C[i]
        key = f"{p:g}"
        filled = p in cal_set
        ax_d.plot(lam_s, Anom[key], color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.8)
        ax_d.plot(lam_s, Acal[key], color=col, lw=1.4)
        ax_d.plot(meas[p]["lam"][::2], meas[p]["ASE"][::2], ls="none", marker=MARKERS[i], ms=3.6, mew=0.9,
                  mfc=col if filled else "white", mec=col)
    ax_d.set_xlim(1529.5, 1563)
    ax_d.set_ylim(-50.5, -33.0)
    ax_d.set_xlabel("Wavelength [nm]")
    ax_d.set_ylabel("ASE in 0.1 nm [dBm]")
    h_s = [Line2D([], [], color=style.INK2, lw=1.4, label="calibrated card"),
           Line2D([], [], color=style.INK2, lw=0.9, ls=(0, (3, 2)), label="nominal card"),
           Line2D([], [], color=style.INK2, marker="o", ms=3.6, ls="none", label="measured, fit pumps"),
           Line2D([], [], color=style.INK2, marker="o", ms=3.6, mfc="white", ls="none", label="measured, held out")]
    ax_d.legend(handles=h_s, ncol=2, loc="upper left", fontsize=6.8, handlelength=1.8, columnspacing=0.9,
                handletextpad=0.4, borderaxespad=0.2)
    style.label(ax_d, "d")

    # ---------------- e) error budget ----------------
    num = R["numerics"]
    mf = {m["load"]: m for m in R["model_form"]}
    unc = R["uncertainty"]["linear"]
    L2 = R["L2"]
    sim_gain, sim_ase, sim_ase_edge = simulator_rows()
    rows = [
        ("Numerical (grid, bins, sweeps)", num["production"]["gain_max_abs_dB"], num["production"]["ase_max_abs_dB"]),
        ("Code check vs closed form", max(v["max_abs_dB"] for v in num["verification_vs_eq5"] if v["nz"] == 200),
         None),
        ("Simulator vs open model", sim_gain, sim_ase),                 # noise bins read at their centers
        ("Simulator, ASE at bin edge", None, sim_ase_edge),             # read at the short-wavelength edge
        ("Model form, 40 × $-$13 dBm", mf["40 x -13 dBm (measured load)"]["overall"]["rmse"], None),
        ("Model form, 1 × $-$40 dBm", mf["1 x -40 dBm at 1550 nm"]["overall"]["rmse"], None),
        ("Parameters, 1$\\sigma$", unc["gain_sd_rms_dB"], unc["ase_sd_rms_dB"]),
        ("Nominal card vs data", L2["giles_nominal_vs_measured"]["gain"]["overall"]["rmse"],
         L2["giles_nominal_vs_measured"]["ase"]["overall"]["rmse"]),
        ("Model of Perin et al. vs data", L2["source_model_vs_measured"]["gain"]["overall"]["rmse"],
         L2["source_model_vs_measured"]["ase"]["overall"]["rmse"]),
        ("Calibrated, fit pumps", cal_entry["gain_cal"]["overall"]["rmse"], cal_entry["ase_cal"]["overall"]["rmse"]),
        ("Calibrated, held-out pumps", cal_entry["gain_val"]["overall"]["rmse"],
         cal_entry["ase_val"]["overall"]["rmse"]),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (name, g, a) in zip(y, rows):
        vals = [v for v in (g, a) if v is not None]
        ax_e.plot([1e-8, max(vals)], [yi, yi], color=style.GRID, lw=0.8, zorder=1)
        if g is not None:
            ax_e.plot(g, yi + 0.14, ls="none", marker="o", ms=4.6, color=style.INK, zorder=3)
        if a is not None:
            ax_e.plot(a, yi - 0.14, ls="none", marker="s", ms=4.4, mfc="white", mec=style.INK, mew=1.0, zorder=3)
    ax_e.axhspan(y[-1] - 0.5, y[-4] + 0.5, color=style.GRID, alpha=0.35, lw=0, zorder=0)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax_e.set_xscale("log")
    ax_e.set_xlim(1e-7, 20)
    ax_e.set_ylim(-0.6, len(rows) + 0.35)          # empty strip above the first row holds the legend
    ax_e.set_xlabel("RMSE or max. |error| [dB]")
    ax_e.grid(axis="y", visible=False)
    # legend in that strip at the upper right, above every row and away from every guide line
    ax_e.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.6, color=style.INK, label="gain"),
                         Line2D([], [], ls="none", marker="s", ms=4.4, mfc="white", mec=style.INK, label="ASE")],
                loc="upper right", fontsize=6.8, handletextpad=0.3, borderaxespad=0.2, ncol=2, columnspacing=0.9)
    pc, pe_ = ax_c.get_position(), ax_e.get_position()     # panel label e) aligned with a) and c)
    style.label(ax_e, "e", x=(pc.x0 - 0.16 * pc.width - pe_.x0) / pe_.width)

    # ---------------- f) design map ----------------
    mcal = R["L3"]["map_calibrated"]
    mnom = R["L3"]["map_nominal"]
    Lm = np.array(mcal["lengths_m"])
    Pm = np.array(mcal["pumps_mW"])
    rip = np.array(mcal["ripple_dB"])
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9cc3ee", "#2a78d6", "#123c70"])
    levels = [0, 1, 2, 3, 4, 6, 8, 12]
    cf = ax_f.contourf(Lm, Pm, np.clip(rip, 0, 12), levels=levels, cmap=cmap, extend="neither")
    cs = ax_f.contour(Lm, Pm, rip, levels=[1, 2, 4, 8], colors=[style.INK2], linewidths=0.5)
    c_cal = ax_f.contour(Lm, Pm, np.array(mcal["mean_gain_dB"]), levels=[10.0], colors=[style.INK], linewidths=1.5)
    c_nom = ax_f.contour(Lm, Pm, np.array(mnom["mean_gain_dB"]), levels=[10.0], colors=[style.INK], linewidths=1.2,
                         linestyles=[(0, (3, 2))])
    on = R["L3"]["results"]["nominal"]["optimum"]
    oc = R["L3"]["results"]["calibrated"]["optimum"]
    ax_f.plot(on["L_m"], on["pump_mW"], ls="none", marker="o", ms=6, mfc="white", mec=style.INK, mew=1.2, zorder=5)
    ax_f.plot(oc["L_m"], oc["pump_mW"], ls="none", marker="o", ms=6, color=style.INK, zorder=5)
    ax_f.plot(np.full(len(pumps), 8.0), pumps, ls="none", marker="x", ms=4, color=style.C[1], mew=1.0, zorder=4)
    ax_f.set_yscale("log")
    ax_f.set_xlim(Lm.min(), Lm.max())
    ax_f.set_ylim(Pm.min(), Pm.max())
    # one label per ripple contour, at the point farthest from the 10 dB lines and the markers
    avoid = [p.vertices for p in c_cal.get_paths() + c_nom.get_paths() if len(p.vertices)]
    avoid += [np.array([[on["L_m"], on["pump_mW"]], [oc["L_m"], oc["pump_mW"]]]),
              np.column_stack([np.full(len(pumps), 8.0), pumps])]
    for t in ax_f.clabel(cs, fmt="%g dB", fontsize=6, inline_spacing=2,
                         manual=clear_label_positions(ax_f, cs, avoid)):
        if float(t.get_text().split()[0]) >= 6:          # label on the dark end of the color scale
            t.set_color("white")
    ax_f.set_yticks([10, 20, 50, 100, 200])
    ax_f.set_yticklabels(["10", "20", "50", "100", "200"])
    ax_f.set_xlabel("EDF length [m]")
    ax_f.set_ylabel("Pump power [mW]")
    ax_f.grid(False)
    cb = fig.colorbar(cf, cax=cax)                  # own axes, so that the colorbar label stays inside the canvas
    cb.set_label("Gain ripple, peak to peak [dB]", labelpad=3)
    cb.ax.tick_params(labelsize=7)
    ax_f.legend(handles=[Line2D([], [], color=style.INK, lw=1.5, label="10 dB, calibrated"),
                         Line2D([], [], color=style.INK, lw=1.2, ls=(0, (3, 2)), label="10 dB, nominal"),
                         Line2D([], [], ls="none", marker="x", ms=4, mew=1.0, color=style.C[1],
                                label="measured points"),
                         Line2D([], [], ls="none", marker="o", ms=5, color=style.INK, label="optimum, calibrated"),
                         Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=style.INK,
                                label="optimum, nominal")],
                loc="upper left", ncol=2, fontsize=6.2, handlelength=1.7, columnspacing=0.8, handletextpad=0.4,
                borderaxespad=0.25, frameon=True, framealpha=0.9, edgecolor="none")
    pd_, pf_ = ax_d.get_position(), ax_f.get_position()    # panel label f) aligned with b) and d)
    style.label(ax_f, "f", x=(pd_.x0 - 0.16 * pd_.width - pf_.x0) / pf_.width)

    style.save(fig, "Fig12_end_to_end", outdir=E.FIG_DIR)
    print("saved", os.path.join(E.FIG_DIR, "Fig12_end_to_end.png"))


if __name__ == "__main__":
    main()
