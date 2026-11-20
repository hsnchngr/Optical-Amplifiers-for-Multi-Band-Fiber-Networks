"""Supporting Information Figure S2: SOA patterning and cross-gain modulation versus operating conditions.

Data: soa_patterning.py (card SOA-BULK-REF; numerical model = sectioned Equations (6)-(7), reservoir model =
Equation (9)). Legends name the models ('numerical model', 'reservoir model'), as in Figure 4, and write the channel
count as '1 channel', '8 channels', '40 channels' (N is the carrier density in the notation registry).
  a) single channel, NRZ-OOK: peak-to-peak gain at the mark centers (de Bruijn order 10) at 2.5/10/40 Gb/s;
     numerical model (solid) and reservoir model with card-derived parameters (dashed), 150 mA.
  b) single channel: eye-opening penalty at the best sampling phase, 150 mA (solid) and 120 mA (dashed).
  c) signal formats at 150 mA: standard deviation of the gain at the mark or symbol centers.
  d-f) 1, 8 and 40 channels at fixed TOTAL average power; g-i) at fixed PER-CHANNEL average power;
     2.5, 10 and 40 Gb/s NRZ-OOK, standard deviation of the gain at the mark centers of the probe channel.
Output: ../figures/FigS2_soa_patterning.{png,pdf} (numbers in FigS2_soa_patterning_results.json).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style
import soa_patterning

style.apply()
style.math_in_text_font()          # one font family (no math symbols at present; kept for consistency)
C, INK, INK2 = style.C, style.INK, style.INK2

r = soa_patterning.load_or_run()
m150, m120 = r["results"]["150mA"], r["results"]["120mA"]
Pt = np.array(r["config"]["P_tot_dBm"])
Pc = np.array(r["config"]["P_ch_dBm"])
rates = ["2.5G", "10G", "40G"]
rate_lab = {"2.5G": "2.5 Gb/s", "10G": "10 Gb/s", "40G": "40 Gb/s"}
rcol = {"2.5G": C[2], "10G": C[0], "40G": C[1]}
ncol = {1: C[6], 8: C[3], 40: C[4]}

fig, axs = plt.subplots(3, 3, figsize=(7.1, 7.4))
fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.07, hspace=0.52, wspace=0.36)

# a) peak-to-peak mark gain, single channel
ax = axs[0, 0]
for rt in rates:
    ax.plot(Pt, m150["tw"][f"ook{rt}_N1_total"]["pp_mark_dB"], color=rcol[rt], label=rate_lab[rt])
    ax.plot(Pt, m150["res"][f"ook{rt}_N1_total"]["pp_mark_dB"], color=rcol[rt], ls="--", lw=1.1)
ax.plot([], [], color=INK2, label="numerical model")
ax.plot([], [], color=INK2, ls="--", lw=1.1, label="reservoir model")
ax.set_xlabel("Average input power [dBm]"); ax.set_ylabel("Peak-to-peak mark gain [dB]")
ax.set_xlim(-30, 0); ax.set_ylim(0, 7.8)
ax.legend(loc="upper left", fontsize=6.2, borderaxespad=0.2, handlelength=2.0)
style.label(ax, "a", x=-0.24)

# b) eye-opening penalty, single channel, two bias currents
ax = axs[0, 1]
for rt in rates:
    ax.plot(Pt, m150["tw"][f"ook{rt}_N1_total"]["eop_opt_dB"], color=rcol[rt], label=rate_lab[rt])
    ax.plot(Pt, m120["tw"][f"ook{rt}_N1_total"]["eop_opt_dB"], color=rcol[rt], ls="--", lw=1.1)
ax.plot([], [], color=INK2, label="150 mA")
ax.plot([], [], color=INK2, ls="--", lw=1.1, label="120 mA")
ax.set_xlabel("Average input power [dBm]"); ax.set_ylabel("Eye-opening penalty [dB]")
ax.set_xlim(-30, 0); ax.set_ylim(0, 2.6)
ax.legend(loc="upper left", fontsize=6.2, borderaxespad=0.2, handlelength=2.0)
style.label(ax, "b", x=-0.24)

# c) formats
ax = axs[0, 2]
ax.plot(Pt, m150["tw"]["ook10G_N1_total"]["std_mark_dB"], color=C[0], label="10 Gb/s NRZ-OOK")
ax.plot(Pt, m150["tw"]["qam16_32GBd"]["std_sym_dB"], color=C[5], ls="-.", label="32 GBd Nyquist PDM-16QAM")
ax.plot(Pt, m150["tw"]["dpsk10G"]["std_sym_dB"], color=C[2], ls="--", label="10 Gb/s NRZ-DPSK")
ax.plot(Pt, np.zeros_like(Pt), color=INK, ls=":", lw=1.2, label="constant envelope")
ax.set_xlabel("Average input power [dBm]"); ax.set_ylabel("Gain standard deviation [dB]")
ax.set_xlim(-30, 0); ax.set_ylim(-0.03, 1.8)
ax.legend(loc="upper left", fontsize=6.2, borderaxespad=0.2, handlelength=2.4)
style.label(ax, "c", x=-0.24)

# d-i) WDM aggregates
for row, grid, P, xl in ((1, "total", Pt, "Total average input power [dBm]"),
                         (2, "perch", Pc, "Per-channel input power [dBm]")):
    for col, rt in enumerate(rates):
        ax = axs[row, col]
        for N in (1, 8, 40):
            ax.semilogy(P, m150["tw"][f"ook{rt}_N{N}_{grid}"]["std_mark_dB"], color=ncol[N], marker="o", ms=2.0,
                        label=f"{N} channel" + ("s" if N > 1 else ""))
        ax.set_xlim(P[0], P[-1]); ax.set_ylim(3e-3, 3)
        ax.set_xlabel(xl)
        if col == 0:
            ax.set_ylabel("Gain standard deviation [dB]")
        ax.text(0.97, 0.05, rate_lab[rt] + (", fixed total power" if grid == "total" else ", fixed power per channel"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color=INK2)
        if col == 0:
            ax.legend(loc="upper left", fontsize=6.2, borderaxespad=0.2)
        style.label(ax, "defghi"[(row - 1) * 3 + col], x=-0.24)

style.save(fig, "FigS2_soa_patterning")
print("saved FigS2_soa_patterning")
