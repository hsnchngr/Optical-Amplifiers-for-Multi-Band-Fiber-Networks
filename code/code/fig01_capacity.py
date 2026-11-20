"""Figure 1 - Traffic growth as an average rate, and single-fiber capacity decomposed into bandwidth x spectral efficiency.

Revision 3 (reviewer points R2-M7 and R2-M4; the numbers also feed the R2-M1 and R2-M5 text changes).

Data   ../data/fig01_traffic_capacity_data.json  (every value with source, locator and verification level)
Output ../figures/Fig01_traffic_capacity.png/.pdf and ../figures/Fig01_traffic_capacity_results.json

Panels
  a) average traffic rate [b/s]: Cisco VNI Table 1 (1992-2017, the 2022 forecast is not plotted) and the ITU 2023
     end-user traffic of fixed + mobile broadband networks, converted from volumes to average rates;
  b) capacity [Tb/s] of selected laboratory records over one single-core single-mode fiber and of commercial
     long-haul WDM systems versus year;
  c) the same systems in the plane occupied optical bandwidth [THz] x spectral efficiency [b/s/Hz]; gray lines are
     lines of constant capacity, so a vertical step is growth by spectral efficiency and a horizontal step growth by
     bandwidth.

Plotted rate basis: net rates (payload after FEC overhead, or after decoding) wherever the source gives one; the
2020 record is only available as a GMI-based achievable rate and is drawn with an open marker.
"""
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "fig01_traffic_capacity_data.json"))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
NAME = "Fig01_traffic_capacity"
C_NM_THZ = 299792.458          # speed of light [nm THz]
SECONDS = {"GB/day": 86400.0, "GB/h": 3600.0, "GB/s": 1.0}


# ----------------------------------------------------------------------------------------------- data handling
def load():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


def si_rate(bps):
    """Format a rate in b/s with an SI prefix and two significant digits (9.3 Mb/s, 0.37 Pb/s)."""
    for pref, f in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if bps >= 0.1 * f:
            v = bps / f
            return f"{v:.2g} {pref}b/s" if v < 10 else f"{v:.0f} {pref}b/s"
    return f"{bps:.2g} b/s"


def traffic_rates(d):
    """Convert every traffic value of the data file into an average rate [b/s]."""
    out = []
    for p in d["traffic"]["cisco_vni_table1"]["points"]:
        bps = p["value"] * 1e9 * 8.0 / SECONDS[p["unit"]]
        out.append(dict(year=p["year"], source="Cisco VNI [1]", quantity="global Internet traffic",
                        value=p["value"], unit=p["unit"], kind=p["kind"], plotted=p["plotted"],
                        rate_bps=bps, rate=si_rate(bps)))
    for p in d["traffic"]["itu_ff2024"]["points"]:
        secs = (366 if p["year"] % 4 == 0 else 365) * 86400.0
        for q, zb in (("fixed broadband", p["fixed_ZB"]), ("mobile broadband", p["mobile_ZB"]),
                      ("fixed + mobile broadband", p["fixed_ZB"] + p["mobile_ZB"])):
            bps = zb * 1e21 * 8.0 / secs
            out.append(dict(year=p["year"], source="ITU [2]", quantity=q, value=round(zb, 3), unit="ZB/year",
                            kind=p["kind"], plotted=bool(p["plotted"] and q == "fixed + mobile broadband"),
                            rate_bps=bps, rate=si_rate(bps)))
    return out


def bandwidth(r):
    """Occupied optical bandwidth [THz] and how it was obtained (stated, channels x spacing, or capacity/SE)."""
    if r.get("bandwidth_THz"):
        return r["bandwidth_THz"], "stated in the source"
    if r.get("channels") and r.get("channel_spacing_GHz"):
        return r["channels"] * r["channel_spacing_GHz"] / 1e3, f"{r['channels']} x {r['channel_spacing_GHz']:g} GHz"
    if r.get("channels") and r.get("wavelength_range_nm"):
        lo, hi = r["wavelength_range_nm"]
        spacing = (C_NM_THZ / lo - C_NM_THZ / hi) / (r["channels"] - 1)
        return r["channels"] * spacing, (f"{r['channels']} x {spacing * 1e3:.1f} GHz mean spacing "
                                         f"({lo:.2f}-{hi:.2f} nm)")
    if r.get("se_bps_per_Hz_stated"):
        return r["capacity_net_Tbps"] / r["se_bps_per_Hz_stated"], "capacity / stated SE"
    raise ValueError(f"no bandwidth information for {r.get('id')}")


def systems(d):
    """Laboratory records and commercial systems with plotted rate, bandwidth and SE."""
    rows = []
    for kind, lst in (("lab", d["lab_records"]), ("commercial", d["commercial_systems"])):
        for r in lst:
            bw, how = bandwidth(r)
            net, gmi = r.get("capacity_net_Tbps"), r.get("capacity_gmi_Tbps")
            cap, basis = (net, "net") if net else (gmi, "GMI")
            rows.append(dict(
                id=r["id"], kind=kind, year=r["year"], ref=r["ref"], bands=r["bands"], band_label=r["band_label"],
                capacity_plotted_Tbps=cap, rate_basis_plotted=basis,
                capacity_net_Tbps=net, capacity_gmi_Tbps=gmi,
                gmi_is_lower_bound=bool(r.get("capacity_gmi_is_lower_bound", False)),
                bandwidth_THz=bw, bandwidth_basis=how,
                se_plotted=cap / bw, se_gmi=(gmi / bw if gmi else None),
                se_stated=r.get("se_bps_per_Hz_stated"),
                distance_km=r.get("distance_km"), spans=r.get("spans"), fiber=r.get("fiber"),
                doi=r.get("doi"), verification=r["verification"]["level"]))
    return rows


def step(a, b, cap_key="capacity_plotted_Tbps", se_key="se_plotted"):
    """Split the capacity growth between two systems into bandwidth and SE factors (log shares)."""
    fc = b[cap_key] / a[cap_key]
    fb = b["bandwidth_THz"] / a["bandwidth_THz"]
    fs = b[se_key] / a[se_key]
    return dict(capacity_factor=fc, bandwidth_factor=fb, se_factor=fs,
                product_check=fb * fs / fc - 1.0,
                share_of_log_growth_bandwidth=math.log(fb) / math.log(fc),
                share_of_log_growth_se=math.log(fs) / math.log(fc))


def cagr(v0, v1, y0, y1):
    return (v1 / v0) ** (1.0 / (y1 - y0)) - 1.0


# ----------------------------------------------------------------------------------------------- figure
def iso_label(ax, cap, B, text):
    """Write a capacity label on the iso-line SE = cap/B, rotated along the line."""
    p1 = ax.transData.transform((B, cap / B))
    p2 = ax.transData.transform((B * 1.2, cap / (B * 1.2)))
    ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    ax.text(B, cap / B, text, rotation=ang, rotation_mode="anchor", ha="left", va="bottom",
            fontsize=7.5, color=style.INK2, zorder=1,
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))


def make_figure(tr, rows, dec):
    style.apply()
    TRF, LAB, COM, ISO = style.C[0], style.C[1], style.INK2, "#b9b7b1"
    FS = 7.5   # annotation size = legend size of style.py

    fig = plt.figure(figsize=(7.1, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], left=0.085, right=0.985, bottom=0.095, top=0.955,
                          hspace=0.10, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[:, 1])

    # ---- a) traffic as an average rate
    cis = [t for t in tr if t["source"].startswith("Cisco") and t["plotted"]]
    itu = [t for t in tr if t["source"].startswith("ITU") and t["plotted"]]
    ax_a.plot([t["year"] for t in cis], [t["rate_bps"] for t in cis], "-o", color=TRF, ms=4.5, lw=1.4,
              label="Internet traffic, Cisco VNI")
    ax_a.plot([t["year"] for t in itu], [t["rate_bps"] for t in itu], "s", color=TRF, ms=5.0, mec="white",
              mew=0.6, label="Fixed + mobile broadband, ITU")
    ax_a.set_yscale("log")
    ax_a.set_ylim(1e6, 1e16)
    ax_a.yaxis.set_major_locator(FixedLocator([1e6, 1e8, 1e10, 1e12, 1e14, 1e16]))
    ax_a.yaxis.set_minor_formatter(NullFormatter())
    ax_a.set_ylabel("Average traffic rate [b/s]")
    first, c17, i23 = cis[0], [t for t in cis if t["year"] == 2017][0], itu[0]
    ax_a.annotate(first["rate"], (first["year"], first["rate_bps"]), xytext=(5, -1), textcoords="offset points",
                  ha="left", va="top", fontsize=FS, color=style.INK)
    ax_a.annotate(c17["rate"], (c17["year"], c17["rate_bps"]), xytext=(-5, 3), textcoords="offset points",
                  ha="right", va="bottom", fontsize=FS, color=style.INK)
    ax_a.annotate(i23["rate"], (i23["year"], i23["rate_bps"]), xytext=(4, -4), textcoords="offset points",
                  ha="left", va="top", fontsize=FS, color=style.INK)
    ax_a.legend(loc="lower right", handlelength=1.6, borderaxespad=0.3)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    style.label(ax_a, "a", x=-0.19)

    # ---- b) capacity versus year
    lab = [r for r in rows if r["kind"] == "lab"]
    com = [r for r in rows if r["kind"] == "commercial"]
    ax_b.plot([r["year"] for r in lab], [r["capacity_plotted_Tbps"] for r in lab], "-", color=LAB, lw=1.2, zorder=2)
    for r in lab:
        net = r["rate_basis_plotted"] == "net"
        ax_b.plot(r["year"], r["capacity_plotted_Tbps"], "o", color=LAB, ms=4.8, mfc=LAB if net else "white",
                  mew=1.2, zorder=3)
    ax_b.plot([r["year"] for r in com], [r["capacity_plotted_Tbps"] for r in com], "-s", color=COM, ms=4.3,
              lw=1.0, zorder=2)
    # band labels: (dx, dy) offsets in points, alignment
    off = {"onaka1996": (5, -3, "left", "top"), "bigo2001": (-5, 2, "right", "bottom"),
           "qian2011": (-5, 2, "right", "bottom"), "galdino2020": (-5, 3, "right", "bottom"),
           "puttnam2023": (3, -4, "left", "top"), "puttnam2024": (0, 6, "center", "bottom"),
           "luis2026": (5, 0, "left", "center"),
           "commercial1997": (5, -2, "left", "top"), "commercial2017": (5, -2, "left", "top")}
    for r in rows:
        dx, dy, ha, va = off[r["id"]]
        ax_b.annotate(r["band_label"], (r["year"], r["capacity_plotted_Tbps"]), xytext=(dx, dy),
                      textcoords="offset points", ha=ha, va=va, fontsize=FS, color=style.INK)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.01, 3000)
    ax_b.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100, 1000]))
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_b.yaxis.set_minor_formatter(NullFormatter())
    ax_b.set_ylabel("Capacity [Tb/s]")
    ax_b.set_xlabel("Year")
    ax_b.set_xlim(1990, 2030)
    ax_b.xaxis.set_major_locator(FixedLocator([1990, 2000, 2010, 2020, 2030]))
    h_net, = ax_b.plot([], [], "o", color=LAB, ms=4.8, label="Laboratory record, net rate")
    h_gmi, = ax_b.plot([], [], "o", color=LAB, ms=4.8, mfc="white", mew=1.2, label="Laboratory record, GMI-based")
    h_com, = ax_b.plot([], [], "-s", color=COM, ms=4.3, lw=1.0, label="Commercial system")
    ax_b.legend(handles=[h_net, h_gmi, h_com], loc="lower right", handlelength=1.6, borderaxespad=0.3)
    style.label(ax_b, "b", x=-0.19)

    # ---- c) spectral efficiency versus occupied bandwidth with lines of constant capacity
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(2, 70)
    ax_c.set_ylim(0.008, 40)
    ax_c.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
    ax_c.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_c.xaxis.set_minor_formatter(NullFormatter())
    ax_c.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_c.yaxis.set_minor_formatter(NullFormatter())
    ax_c.set_xlabel("Occupied optical bandwidth [THz]")
    ax_c.set_ylabel("Spectral efficiency [b/s/Hz]")
    ax_c.grid(False)
    Bl = np.logspace(np.log10(1.5), np.log10(100), 200)
    # label positions (bandwidth [THz] at the label anchor) along the lower-right ends of the lines, clear of data
    iso = {0.1: (5.6, "0.1 Tb/s"), 1: (19.0, "1 Tb/s"), 10: (27.0, "10 Tb/s"), 100: (38.0, "100 Tb/s"),
           1000: (38.0, "1 Pb/s")}
    for cap, (Bpos, txt) in iso.items():
        ax_c.plot(Bl, cap / Bl, color=ISO, lw=0.7, zorder=1)
    fig.canvas.draw()  # fixes the axes geometry before the label angles are computed
    for cap, (Bpos, txt) in iso.items():
        iso_label(ax_c, cap, Bpos, txt)

    ax_c.plot([r["bandwidth_THz"] for r in lab], [r["se_plotted"] for r in lab], "-", color=LAB, lw=1.2, zorder=2)
    for r in lab:
        net = r["rate_basis_plotted"] == "net"
        ax_c.plot(r["bandwidth_THz"], r["se_plotted"], "o", color=LAB, ms=4.8, mfc=LAB if net else "white",
                  mew=1.2, zorder=3)
    ax_c.plot([r["bandwidth_THz"] for r in com], [r["se_plotted"] for r in com], "-s", color=COM, ms=4.3, lw=1.0,
              zorder=2)
    yoff = {"onaka1996": (5, -2, "left", "top"), "bigo2001": (5, -1, "left", "top"),
            "qian2011": (-2, 5, "right", "bottom"), "galdino2020": (0, -6, "center", "top"),
            "puttnam2023": (0, 6, "center", "bottom"), "puttnam2024": (-1, -6, "center", "top"),
            "luis2026": (3, 6, "center", "bottom"),
            "commercial1997": (5, 0, "left", "center"), "commercial2017": (5, -2, "left", "top")}
    for r in rows:
        dx, dy, ha, va = yoff[r["id"]]
        ax_c.annotate(str(r["year"]), (r["bandwidth_THz"], r["se_plotted"]), xytext=(dx, dy),
                      textcoords="offset points", ha=ha, va=va, fontsize=FS, color=style.INK, zorder=4,
                      bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))

    # growth regimes (factors computed from the data, see results JSON)
    d1, d2, dcm = dec["2001->2011"], dec["2011->2026"], dec["commercial 1997->2017"]
    b01 = [r for r in lab if r["year"] == 2001][0]
    b11 = [r for r in lab if r["year"] == 2011][0]
    b26 = [r for r in lab if r["year"] == 2026][0]
    xa = b01["bandwidth_THz"] * 0.72
    ax_c.annotate("", xy=(xa, b11["se_plotted"] * 0.92), xytext=(xa, b01["se_plotted"] * 1.08),
                  arrowprops=dict(arrowstyle="-|>", color=style.INK2, lw=0.9, mutation_scale=8))
    ax_c.text(xa * 0.93, b11["se_plotted"] * 0.5, f"SE ×{d1['se_factor']:.1f}\n2001–2011", ha="right",
              va="center", fontsize=FS, color=style.INK)
    ya = b11["se_plotted"] * 1.6
    ax_c.annotate("", xy=(b26["bandwidth_THz"], ya), xytext=(b11["bandwidth_THz"] * 1.05, ya),
                  arrowprops=dict(arrowstyle="-|>", color=style.INK2, lw=0.9, mutation_scale=8))
    ax_c.text(math.sqrt(b11["bandwidth_THz"] * b26["bandwidth_THz"]), ya * 1.12,
              f"bandwidth ×{d2['bandwidth_factor']:.1f}\n2011–2026", ha="center", va="bottom", fontsize=FS,
              color=style.INK)
    c97 = [r for r in com if r["year"] == 1997][0]
    ax_c.text(c97["bandwidth_THz"] * 1.45, c97["se_plotted"] * 3.4,
              f"commercial\n1997–2017:\nSE ×{dcm['se_factor']:.0f},\nbandwidth ×{dcm['bandwidth_factor']:.0f}",
              ha="left", va="center", fontsize=FS, color=style.INK2, linespacing=1.0,
              bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    style.label(ax_c, "c", x=-0.17, y=1.008)
    return fig


# ----------------------------------------------------------------------------------------------- main
def main():
    d = load()
    tr = traffic_rates(d)
    rows = systems(d)
    by = {r["id"]: r for r in rows}
    dec = {
        "1996->2001": step(by["onaka1996"], by["bigo2001"]),
        "2001->2011": step(by["bigo2001"], by["qian2011"]),
        "1996->2011": step(by["onaka1996"], by["qian2011"]),
        "2011->2026": step(by["qian2011"], by["luis2026"]),
        "2011->2026 (2026 GMI-based)": step(by["qian2011"], dict(by["luis2026"],
                                            capacity_plotted_Tbps=by["luis2026"]["capacity_gmi_Tbps"],
                                            se_plotted=by["luis2026"]["se_gmi"])),
        "2011->2020": step(by["qian2011"], by["galdino2020"]),
        "1996->2026": step(by["onaka1996"], by["luis2026"]),
        "commercial 1997->2017": step(by["commercial1997"], by["commercial2017"]),
    }
    cis = {t["year"]: t["rate_bps"] for t in tr if t["source"].startswith("Cisco")}
    itu_tot = {t["year"]: t["rate_bps"] for t in tr if t["source"].startswith("ITU")
               and t["quantity"] == "fixed + mobile broadband"}
    growth = {
        "cisco_1992_2017_factor": cis[2017] / cis[1992],
        "cisco_1992_2017_orders_of_magnitude": math.log10(cis[2017] / cis[1992]),
        "cagr_cisco": {f"{a}-{b}": cagr(cis[a], cis[b], a, b) for a, b in
                       ((1992, 1997), (1997, 2002), (2002, 2007), (2007, 2017), (1992, 2017))},
        "cagr_cisco2017_to_itu2023_total_mixed_sources": cagr(cis[2017], itu_tot[2023], 2017, 2023),
        "lab_record_factor_1996_2026_net": by["luis2026"]["capacity_plotted_Tbps"] / by["onaka1996"]["capacity_plotted_Tbps"],
        "cagr_lab_record_1996_2026_net": cagr(by["onaka1996"]["capacity_plotted_Tbps"],
                                             by["luis2026"]["capacity_plotted_Tbps"], 1996, 2026),
        "commercial_factor_1997_2017": by["commercial2017"]["capacity_plotted_Tbps"] / by["commercial1997"]["capacity_plotted_Tbps"],
        "cagr_commercial_1997_2017": cagr(by["commercial1997"]["capacity_plotted_Tbps"],
                                          by["commercial2017"]["capacity_plotted_Tbps"], 1997, 2017),
    }
    checks = {
        "cisco2017_equals_IP_total_122EB_per_month_GBps": 122e18 / (365.25 / 12 * 86400) / 1e9,
        "cisco2017_table1_GBps": 46600,
        "qian2011_bandwidth_370x25GHz_vs_capacity_over_SE": [by["qian2011"]["bandwidth_THz"], 101.7 / 11.0],
        "galdino2020_bandwidth_stated_vs_660x25.5GHz": [16.83, 660 * 25.5e-3],
        "puttnam2024_bandwidth_stated_vs_1505x25GHz": [37.6, 1505 * 25e-3],
        "puttnam2023_span_1410.8_1623.1nm_THz": C_NM_THZ / 1410.8 - C_NM_THZ / 1623.1,
        "luis2026_span_1264.0_1617.8nm_THz": C_NM_THZ / 1264.0 - C_NM_THZ / 1617.8,
        "luis2026_mean_slot_GHz": 42.4e3 / 1273,
        "max_abs_product_residual": max(abs(v["product_check"]) for v in dec.values()),
    }
    fig = make_figure(tr, rows, dec)
    style.save(fig, NAME, outdir=FIGDIR)
    res = dict(figure=NAME, data_file=os.path.relpath(DATA, HERE), traffic_rates=tr, systems=rows,
               decomposition=dec, growth=growth, checks=checks,
               notes=["rates: 1 GB = 1e9 B, 1 ZB = 1e21 B, 365-day years (366 for 2024)",
                      "occupied bandwidth = channels x spacing, or as stated by the source, or capacity/stated SE (2001, commercial 1997)",
                      "plotted rate basis: net (payload after FEC overhead, or decoded) where available; GMI-based for 2020"])
    with open(os.path.join(FIGDIR, NAME + "_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    for t in tr:
        print(f"{t['year']} {t['source']:14s} {t['quantity']:26s} {t['value']:>9g} {t['unit']:8s} -> {t['rate_bps']:.4g} b/s ({t['rate']})"
              + ("" if t["plotted"] else "  [not plotted]"))
    for r in rows:
        print(f"{r['year']} {r['kind']:10s} {r['bands']:12s} C={r['capacity_plotted_Tbps']:8.4g} Tb/s ({r['rate_basis_plotted']})"
              f"  B={r['bandwidth_THz']:7.4g} THz  SE={r['se_plotted']:7.4g} b/s/Hz  [{r['bandwidth_basis']}]")
    for k, v in dec.items():
        print(f"{k:30s} C x{v['capacity_factor']:8.3f}  B x{v['bandwidth_factor']:6.3f}  SE x{v['se_factor']:7.3f}"
              f"  log share B {100 * v['share_of_log_growth_bandwidth']:5.1f} %  SE {100 * v['share_of_log_growth_se']:5.1f} %")
    print(json.dumps(growth, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
