"""Table of Contents graphic (revision 3): the gain media of the O- to U-bands and the four-layer framework.

Format (journal requirements for the ToC graphic): 55 mm wide x 50 mm high at 600 dpi, sans-serif labels of 10 pt
at this final size (no smaller text), no inset, no panel label.

Content, drawn by this script only (no third-party image):
  top    the ITU-T bands O, E, S, C, L and U (1260-1675 nm) and the gain media that cover them, from the card
         REF-FIG03-BANDS-SPECTRO (the same entries as Figure 3a): PDFA window; BDFA family range (pale) with the
         single-device windows of the O-, E- and S-band BDFAs; TDFA window; EDFA C- and L-band windows; SOA range in
         which InP-based active regions can place gain by design (dashed, no single device); Raman gain band set by
         the pump wavelengths (dotted), with the window of the U-band discrete Raman amplifier (solid);
  bottom the four layers of the framework of Figure 2 as a loop: cards -> comparison of the models at the same card
         and operating point -> design -> identification, which updates the card.

Outputs: ../figures/ToC_graphic.png and ../figures/ToC_graphic.tif (RGB, LZW-compressed), both 55 mm x 50 mm at
         600 dpi, and ../figures/ToC_graphic_results.json (card entries drawn, size, smallest font size, checks).
Usage  : python fig_toc.py
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import cards
import style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
NAME = "ToC_graphic"

W_MM, H_MM, DPI = 55.0, 50.0, 600
FS = 10.0                       # every label, at the final size
MARGIN = 0.8                    # [mm] free border on all sides

REF = cards.load("REF-FIG03-BANDS-SPECTRO")
R = cards.values(REF)
BANDS = [(b, R[f"band_{b}"]) for b in "OESCLU"]
LAM0, LAM1 = BANDS[0][1][0], BANDS[-1][1][1]          # 1260 and 1675 nm


def tint(hexc, a):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(round(255 - a * (255 - v))) for v in (r, g, b))


def main():
    style.apply()
    mpl.rcParams.update({"savefig.bbox": "standard", "savefig.pad_inches": 0.0})   # exact 55 mm x 50 mm canvas
    C, INK, INK2 = style.C, style.INK, style.INK2
    COL = {"PDFA": C[2], "BDFA": C[4], "TDFA": C[3], "EDFA": C[0], "SOA": C[1], "Raman": C[5]}   # as in Figure 3
    ACC = C[6]                                                                                  # framework loop

    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_MM)
    ax.set_ylim(0, H_MM)
    ax.set_aspect("equal")
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    def text_mm(s, weight="normal"):
        t = ax.text(0, 0, s, fontsize=FS, fontweight=weight)
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        return bb.width / DPI * 25.4, bb.height / DPI * 25.4

    rows = ["PDFA", "BDFA", "TDFA", "EDFA", "SOA", "Raman"]
    x_lab = MARGIN + max(text_mm(r)[0] for r in rows)          # right edge of the row labels
    X0, X1 = x_lab + 0.9, W_MM - MARGIN

    def X(lam):
        return X0 + (lam - LAM0) / (LAM1 - LAM0) * (X1 - X0)

    # ------------------------------------------------------------------ vertical layout [mm, from the bottom]
    y_head1 = H_MM - MARGIN
    y_head0 = y_head1 - 4.4
    pitch = 4.3
    y_rows_top = y_head0 - 0.5
    yc = {r: y_rows_top - (i + 0.5) * pitch for i, r in enumerate(rows)}
    y_rows_bot = y_rows_top - len(rows) * pitch
    hb = 2.5                                                    # bar height

    # band header and band stripes behind the rows
    head_fill = ["#dcd9d1", "#e9e7e1"]
    stripe_fill = ["#efeeea", "#f8f7f4"]
    for i, (b, (lo, hi)) in enumerate(BANDS):
        ax.add_patch(Rectangle((X(lo), y_head0), X(hi) - X(lo), y_head1 - y_head0, fc=head_fill[i % 2], ec="none",
                               zorder=0))
        ax.add_patch(Rectangle((X(lo), y_rows_bot - 0.2), X(hi) - X(lo), y_head0 - y_rows_bot + 0.2,
                               fc=stripe_fill[i % 2], ec="none", zorder=0))
        ax.text(0.5 * (X(lo) + X(hi)), 0.5 * (y_head0 + y_head1), b, ha="center", va="center", fontsize=FS,
                fontweight="bold", color=INK, zorder=3)

    def bar(lo, hi, y, col, alpha=1.0, z=3):
        ax.add_patch(Rectangle((X(lo), y - hb / 2), X(hi) - X(lo), hb, fc=col, ec="white", lw=0.5, alpha=alpha,
                               zorder=z))

    drawn = {}
    # PDFA
    bar(*R["win_PDFA"], yc["PDFA"], COL["PDFA"])
    drawn["PDFA"] = {"win_PDFA": R["win_PDFA"]}
    # BDFA: family range (pale) and the single-device windows (solid)
    fam = R["range_BDFA_family"]
    ax.add_patch(Rectangle((X(fam[0]), yc["BDFA"] - hb / 2), X(fam[1]) - X(fam[0]), hb, fc=COL["BDFA"], alpha=0.25,
                           ec="none", zorder=2))
    for key in ("win_BDFA_O", "win_BDFA_E", "win_BDFA_S_69"):
        bar(*R[key], yc["BDFA"], COL["BDFA"])
    drawn["BDFA"] = {k: R[k] for k in ("range_BDFA_family", "win_BDFA_O", "win_BDFA_E", "win_BDFA_S_69")}
    # TDFA and EDFA (C- and L-band windows)
    bar(*R["win_TDFA_fluoride"], yc["TDFA"], COL["TDFA"])
    drawn["TDFA"] = {"win_TDFA_fluoride": R["win_TDFA_fluoride"]}
    for key in ("win_EDFA_C", "win_EDFA_L"):
        bar(*R[key], yc["EDFA"], COL["EDFA"])
    drawn["EDFA"] = {k: R[k] for k in ("win_EDFA_C", "win_EDFA_L")}
    # SOA: range reachable by design (dashed), no single device
    w = R["range_SOA_design"]
    ax.plot([X(w[0]), X(w[1])], [yc["SOA"]] * 2, color=COL["SOA"], lw=5.0, ls=(0, (1.75, 0.85)),
            solid_capstyle="butt", dash_capstyle="butt", zorder=3)
    drawn["SOA"] = {"range_SOA_design": w}
    # Raman: gain band set by the pump wavelengths (dotted) and the U-band discrete Raman window (solid)
    w = R["range_Raman_pump_defined"]
    d = 4.6 / 72 * 25.4                                         # dot diameter [mm] (line width 4.6 pt)
    ax.plot([X(w[0]) + d / 2, X(w[1]) - d / 2], [yc["Raman"]] * 2, color=COL["Raman"], lw=4.6, ls=(0, (0.01, 1.9)),
            dash_capstyle="round", zorder=3)
    bar(*R["win_DRA_U"], yc["Raman"], COL["Raman"], z=4)
    drawn["Raman"] = {"range_Raman_pump_defined": w, "win_DRA_U": R["win_DRA_U"]}
    for r in rows:
        ax.text(x_lab, yc[r], r, ha="right", va="center", fontsize=FS, color=INK, zorder=3)

    # ------------------------------------------------------------------ framework loop (2 x 2)
    labels = [["Cards", "Comparison"], ["Identification", "Design"]]   # clockwise: cards, comparison, design, id.
    hc, gy, gx = 5.0, 4.3, 6.0                                 # chip height, vertical and horizontal gap [mm]
    wc = (W_MM - 2 * MARGIN - gx) / 2
    y_fw_top = y_rows_bot - 0.2 - 2.2
    y_fw_bot = y_fw_top - 2 * hc - gy
    if y_fw_bot < MARGIN - 1e-9:
        raise SystemExit(f"layout error: framework loop leaves the canvas ({y_fw_bot:.2f} mm)")
    chip = {}
    for i, row in enumerate(labels):
        for j, s in enumerate(row):
            x0 = MARGIN + j * (wc + gx)
            y0 = y_fw_top - hc - i * (hc + gy)
            ax.add_patch(FancyBboxPatch((x0, y0), wc, hc, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc=tint(ACC, 0.12), ec=ACC, lw=0.9, zorder=2))
            ax.text(x0 + wc / 2, y0 + hc / 2, s, ha="center", va="center", fontsize=FS, color=INK, zorder=3)
            chip[s] = (x0, y0)

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=ACC, lw=1.0,
                                     shrinkA=0, shrinkB=0, zorder=4))

    xl_c, xr_c = MARGIN + wc / 2, MARGIN + wc + gx + wc / 2
    yt_c, yb_c = y_fw_top - hc / 2, y_fw_top - hc - gy - hc / 2
    s_ = 0.6                                                   # clearance between arrow ends and chips [mm]
    arrow((MARGIN + wc + s_, yt_c), (MARGIN + wc + gx - s_, yt_c))                  # cards -> comparison
    arrow((xr_c, y_fw_top - hc - s_ + 0.15), (xr_c, y_fw_top - hc - gy + s_ - 0.15))  # comparison -> design
    arrow((MARGIN + wc + gx - s_, yb_c), (MARGIN + wc + s_, yb_c))                  # design -> identification
    arrow((xl_c, y_fw_top - hc - gy + s_ - 0.15), (xl_c, y_fw_top - hc - s_ + 0.15))  # identification -> cards

    # ------------------------------------------------------------------ checks: font size, canvas, overlaps
    fig.canvas.draw()
    texts = [t for t in ax.texts if t.get_text().strip()]
    sizes = [t.get_fontsize() for t in texts]
    boxes = []
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        boxes.append((t.get_text(), bb.x0 / DPI * 25.4, bb.y0 / DPI * 25.4, bb.x1 / DPI * 25.4, bb.y1 / DPI * 25.4))
    problems = []
    if min(sizes) < 10.0:
        problems.append(f"font smaller than 10 pt: {min(sizes)}")
    for s, x0, y0, x1, y1 in boxes:
        if x0 < 0.3 or y0 < 0.3 or x1 > W_MM - 0.3 or y1 > H_MM - 0.3:
            problems.append(f"text '{s}' closer than 0.3 mm to the edge")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                problems.append(f"text '{a[0]}' overlaps '{b[0]}'")
    for b, (lo, hi) in BANDS:                                  # band letters inside their band
        t = next(bx for bx in boxes if bx[0] == b)
        if t[1] < X(lo) + 0.2 or t[3] > X(hi) - 0.2:
            problems.append(f"band letter {b} wider than its band")
    for p in problems:
        print("warning:", p)

    os.makedirs(FIGDIR, exist_ok=True)
    png, tif = os.path.join(FIGDIR, NAME + ".png"), os.path.join(FIGDIR, NAME + ".tif")
    fig.savefig(png, dpi=DPI)
    plt.close(fig)

    # RGB without alpha channel (the canvas is opaque white), 600 dpi, for both files
    from PIL import Image
    with Image.open(png) as im:
        rgb = im.convert("RGB")
    rgb.save(png, dpi=(DPI, DPI))
    rgb.save(tif, dpi=(DPI, DPI), compression="tiff_lzw")
    sizes_px = {}
    for p in (png, tif):
        with Image.open(p) as im:
            sizes_px[os.path.basename(p)] = dict(px=list(im.size), mode=im.mode,
                                                 dpi=[round(float(v)) for v in im.info.get("dpi", (0, 0))])
    res = {"figure": NAME, "script": "fig_toc.py", "card": REF["card_id"],
           "size_mm": [W_MM, H_MM], "dpi": DPI, "files": sizes_px,
           "font": {"family": mpl.rcParams["font.sans-serif"][0], "smallest_pt": min(sizes), "largest_pt": max(sizes)},
           "bands_nm": {b: w for b, w in BANDS}, "drawn_nm": drawn,
           "framework_loop": ["Cards", "Comparison", "Design", "Identification", "back to Cards (card update)"],
           "checks": {"problems": problems}}
    with open(os.path.join(FIGDIR, NAME + "_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"written {NAME}.png and {NAME}.tif:", sizes_px, "; smallest font", min(sizes), "pt;",
          "checks passed" if not problems else f"{len(problems)} problem(s)")


if __name__ == "__main__":
    main()
