"""Schematics of revision 3: Figure 2 (benchmarking framework) and Figure 7 (amplifier architectures).

Figure 2 (Fig02_framework)
    The four layers of the framework. Every box is marked by its outline (explained in the legend of the figure):
      solid       = demonstrated in this paper;
      dash-dotted = implemented, illustrated in the Supporting Information (boxes in ILLUSTRATED: the
                    physics-informed neural networks, status 'Implemented, illustrated only' of the status table of
                    the Supporting Information, Table S26);
      dashed      = proposed procedure or outlook, not demonstrated in this paper.
    L1 lists the parameter-card files found in ../cards at run time (each one loaded and validated through
    cards.py, so the list follows the card files that actually exist); L2 lists the model sources actually
    used, including the independent simulator run with the same card at the same operating points (Section 5.7;
    run_optisystem_perin2019.py, e2e_simulator_compare.py); L3 the benchmark tables and the design tasks (design
    with the calibrated card, which is validated only at the measured operating points: 8 m, 18-115 mW of pump);
    L4 the least-squares identification that stores the identified entries in a calibrated card (demonstrated,
    solid feedback arrow), the physics-informed neural networks illustrated on synthetic Raman problems in the
    Supporting Information (single-instance networks and one parameterized network) and the outlook items
    (digital twin, transient prediction, gain control).
    Box texts use no abbreviations other than those defined in the caption (ASE, SOA, JSON) and the layer labels;
    card file names are printed as they are.
    Switches that change the status of a box once the corresponding work exists:
      SIMULATOR_RUN      True after an independent simulator was run with the same card (Section 5.7; done on
                         18 November 2026 with OptiSystem 23.1); SIMULATOR_NAME is then printed in the box;
      REPOSITORY_PUBLIC  True once the public repository and its persistent identifier exist.

Figure 7 (Fig07_architectures)
    a) double-pass doped-fiber amplifier; b) distributed Raman + EDFA span;
    c) Raman pre-amplification in front of SOAs: i) one wideband SOA module, no band splitter;
       ii) band splitter, pumps routed around it (bypass); iii) band splitter, separate pump module in front;
    d) band-split parallel multi-band amplifier; e) double-pass parallel hybrid (drawn after [80, 81]);
    f) PON reach extender.
    All abbreviations are defined in the caption (see ../notes/fig02_fig07_schematics.md).

Outputs: ../figures/Fig02_framework.(png|pdf), ../figures/Fig07_architectures.(png|pdf) and the
corresponding *_results.json files (box status, abbreviations). Earlier outputs of this script under
old names (Fig05_architectures.*) are removed. No numerical model is evaluated here.
"""
import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

import cards
import style

# ----------------------------------------------------------------------------------------------
# Status switches (see module docstring)
# ----------------------------------------------------------------------------------------------
SIMULATOR_RUN = True         # independent simulator run with the same card (Section 5.7, E2E_simulator_results.json)
SIMULATOR_NAME = "OptiSystem (commercial)"     # printed in the box when SIMULATOR_RUN is True
REPOSITORY_PUBLIC = False    # set True once the repository and its persistent identifier are public
ILLUSTRATED = ("pinn",)      # boxes with status "implemented, illustrated in the Supporting Information" (status
                             # table of the Supporting Information, Table S26): dash-dotted outline, not solid

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
LEGACY_OUTPUTS = ("Fig05_architectures.png", "Fig05_architectures.pdf")   # old names of this script's outputs

style.apply()
C = style.C
INK, INK2 = style.INK, style.INK2
FS = 7.0             # one font size for all schematic text (panel labels use style.label)
LS = 1.25            # line spacing
DASH = (0, (3.6, 2.2))                  # proposed or outlook
DASHDOT = (0, (5.0, 1.6, 1.1, 1.6))     # implemented, illustrated in the Supporting Information (Figure 2)


# ----------------------------------------------------------------------------------------------
# Helpers (all coordinates in inches; each figure uses one axes whose data units are inches)
# ----------------------------------------------------------------------------------------------
def tint(hexc, a=0.13):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(round(255 - a * (255 - v))) for v in (r, g, b))


def shade(hexc, f=0.8):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(round(f * v)) for v in (r, g, b))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def text_width(ax, s, fs=FS, weight="normal"):
    """Width of a single-line string in inches."""
    fig = ax.figure
    t = ax.text(0, 0, s, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def rbox(ax, x0, y0, w, h, ec=INK2, fc="white", dashed=False, lw=0.9, r=0.035, z=2, linestyle=None):
    """Rounded box; outline solid, dashed (dashed=True) or any explicit 'linestyle' (e.g. DASHDOT)."""
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       ls=linestyle if linestyle is not None else (DASH if dashed else "-"), zorder=z)
    ax.add_patch(p)
    return p


def label_box(ax, x0, y0, w, h, title, body="", ec=INK2, fc="white", dashed=False, lw=0.9, z=2, color=None,
              linestyle=None):
    """Rounded box with a bold title and an optional regular body, centred as one block."""
    rbox(ax, x0, y0, w, h, ec=ec, fc=fc, dashed=dashed, lw=lw, z=z, linestyle=linestyle)
    col = color or (INK2 if dashed else INK)
    lh = FS * LS / 72.0
    nt = title.count("\n") + 1 if title else 0
    nb = body.count("\n") + 1 if body else 0
    top = y0 + h / 2 + (nt + nb) * lh / 2
    xc = x0 + w / 2
    if title:
        ax.text(xc, top, title, ha="center", va="top", fontsize=FS, fontweight="bold", color=col,
                linespacing=LS, zorder=z + 1)
    if body:
        ax.text(xc, top - nt * lh, body, ha="center", va="top", fontsize=FS, color=col, linespacing=LS,
                zorder=z + 1)


def tbox(ax, xc, yc, w, h, s, ec=INK2, fc="white", dashed=False, lw=0.9, z=2, bold=False, color=INK):
    """Centred box with one text block (used in Figure 7)."""
    rbox(ax, xc - w / 2, yc - h / 2, w, h, ec=ec, fc=fc, dashed=dashed, lw=lw, z=z)
    ax.text(xc, yc, s, ha="center", va="center", fontsize=FS, color=color, linespacing=LS, zorder=z + 1,
            fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, col=INK2, lw=0.9, both=False, dashed=False, ms=7, z=4):
    """Straight arrow; a dashed arrow keeps a solid head."""
    if dashed:
        x0, y0 = p0
        x1, y1 = p1
        d = np.hypot(x1 - x0, y1 - y0)
        hl = min(0.07, 0.4 * d)
        xm, ym = x1 - (x1 - x0) * hl / d, y1 - (y1 - y0) * hl / d
        ax.add_line(Line2D([x0, xm], [y0, ym], color=col, lw=lw, ls=DASH, zorder=z))
        ax.add_patch(FancyArrowPatch((xm, ym), p1, arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw,
                                     shrinkA=0, shrinkB=0, zorder=z))
        if both:
            xs, ys = x0 + (x1 - x0) * hl / d, y0 + (y1 - y0) * hl / d
            ax.add_patch(FancyArrowPatch((xs, ys), p0, arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw,
                                         shrinkA=0, shrinkB=0, zorder=z))
        return
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=ms, color=col,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=z))


def polyarrow(ax, pts, col=INK2, lw=0.9, dashed=False, ms=7, z=4):
    """Arrow along a polyline; the head sits on the last segment."""
    xs, ys = zip(*pts[:-1])
    if len(pts) > 2:
        ax.add_line(Line2D(xs, ys, color=col, lw=lw, ls=DASH if dashed else "-", zorder=z,
                           solid_joinstyle="miter", solid_capstyle="butt"))
    arrow(ax, pts[-2], pts[-1], col=col, lw=lw, dashed=dashed, ms=ms, z=z)


def line(ax, pts, col=INK2, lw=0.9, dashed=False, z=3):
    xs, ys = zip(*pts)
    ax.add_line(Line2D(xs, ys, color=col, lw=lw, ls=DASH if dashed else "-", zorder=z,
                       solid_capstyle="butt"))


def coil(ax, xa, xb, y, n=4, col=INK2, lw=1.0, z=3):
    """Fiber-coil symbol (prolate cycloid) between xa and xb, centred on the fiber line at height y."""
    t = np.linspace(-np.pi / 2, 2 * np.pi * n - np.pi / 2, 60 * n)
    a = (xb - xa) / (2 * np.pi * n)
    b = 2.0 * a
    x = a * t - b * np.sin(t)
    x = xa + (x - x[0])
    yy = y - b * np.cos(t)
    ax.add_line(Line2D(x, yy, color=col, lw=lw, zorder=z))
    return b


def amp(ax, xc, yc, w=0.44, h=0.3, s="", col=C[0], left=False, z=3):
    """Amplifier triangle pointing in the signal direction, with its label inside."""
    if left:
        pts = [(xc + w / 2, yc - h / 2), (xc + w / 2, yc + h / 2), (xc - w / 2, yc)]
        tx = xc + 0.1 * w
    else:
        pts = [(xc - w / 2, yc - h / 2), (xc - w / 2, yc + h / 2), (xc + w / 2, yc)]
        tx = xc - 0.1 * w
    ax.add_patch(Polygon(pts, closed=True, fc=tint(col, 0.16), ec=shade(col, 0.85), lw=0.9, zorder=z))
    if s:
        ax.text(tx, yc, s, ha="center", va="center", fontsize=FS, color=INK, zorder=z + 1)
    return (xc - w / 2, yc), (xc + w / 2, yc)


def circulator(ax, xc, yc, r=0.15, z=3):
    """Three-port circulator: port 1 left, port 2 right, port 3 bottom."""
    ax.add_patch(Circle((xc, yc), r, fc="white", ec=INK2, lw=0.9, zorder=z))
    ax.add_patch(FancyArrowPatch((xc - 0.55 * r, yc + 0.25 * r), (xc + 0.55 * r, yc + 0.25 * r),
                                 connectionstyle="arc3,rad=-0.75", arrowstyle="-|>", mutation_scale=5,
                                 color=INK2, lw=0.8, shrinkA=0, shrinkB=0, zorder=z + 1))
    ax.add_patch(FancyArrowPatch((xc + 0.6 * r, yc - 0.15 * r), (xc + 0.12 * r, yc - 0.62 * r),
                                 connectionstyle="arc3,rad=-0.45", arrowstyle="-|>", mutation_scale=5,
                                 color=INK2, lw=0.8, shrinkA=0, shrinkB=0, zorder=z + 1))
    ax.text(xc - r - 0.03, yc + 0.035, "1", ha="right", va="bottom", fontsize=FS, color=INK2)
    ax.text(xc + r + 0.03, yc + 0.035, "2", ha="left", va="bottom", fontsize=FS, color=INK2)
    ax.text(xc + 0.04, yc - r - 0.03, "3", ha="left", va="top", fontsize=FS, color=INK2)
    return (xc - r, yc), (xc + r, yc), (xc, yc - r)


def mirror(ax, x, y0, y1, z=3):
    """Mirror symbol: thick bar with hatching on its back side (right)."""
    ax.add_line(Line2D([x, x], [y0, y1], color=INK, lw=2.0, zorder=z, solid_capstyle="butt"))
    n = max(3, int((y1 - y0) / 0.06))
    for yy in np.linspace(y0 + 0.02, y1 - 0.02, n):
        ax.add_line(Line2D([x + 0.012, x + 0.055], [yy, yy + 0.04], color=INK2, lw=0.6, zorder=z))


def save(fig, name):
    """style.save without extra padding: the canvas is exactly 7.1 in wide and carries its own margins."""
    with mpl.rc_context({"savefig.pad_inches": 0.0}):
        style.save(fig, name, outdir=FIGDIR)
    plt.close(fig)


def card_files():
    """All valid card files in ../cards (each one is loaded and validated through cards.py)."""
    names = []
    for fn in sorted(os.listdir(cards.CARD_DIR)):
        if not fn.lower().endswith(".json"):
            continue
        try:
            cards.load(os.path.join(cards.CARD_DIR, fn))
            names.append(fn)
        except Exception as exc:  # a malformed file is reported, not drawn
            print(f"warning: {fn} is not a valid card ({exc}); not listed in Figure 2")
    return names


def wrap_names(ax, names, width, sep="   ·   "):
    """Join names into as few lines as fit into 'width' inches, with line lengths balanced."""
    def greedy(wmax):
        lines, cur = [], ""
        for nm in names:
            trial = nm if not cur else cur + sep + nm
            if cur and text_width(ax, trial) > wmax:
                lines.append(cur)
                cur = nm
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    best = greedy(width)
    if len(best) > 1:                       # shrink the limit while the number of lines stays the same
        lo, hi = 0.3 * width, width
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            if len(greedy(mid)) <= len(best):
                hi = mid
            else:
                lo = mid
        best = greedy(hi)
    return best


# ----------------------------------------------------------------------------------------------
# Figure 2: framework
# ----------------------------------------------------------------------------------------------
def figure2():
    W = 7.1
    fig, ax = canvas(W, 6.0)        # provisional height; the final height follows from the layout pass
    lh = FS * LS / 72.0             # line height [in]
    PADV = 0.075                    # vertical padding inside a box [in]

    def hbox(n):
        return n * lh + 2 * PADV

    def nlines(*s):
        return sum(x.count("\n") + 1 for x in s if x)

    status = []         # (layer, box, status) for the results file
    col = {"L1": INK, "L2": INK, "L3": INK, "L4": INK}     # outlines black; white background
    FLOW, FEED = C[0], "#d62728"                        # data-flow arrows blue, card-update arrow red
    XL, XR = 0.05, 6.78             # band edges
    PX, GX = 0.10, 0.10             # inner padding, gap between boxes
    XF = 6.95                       # feedback path

    # ---------------- box texts (they set the box heights) ----------------
    eq = ("Governing equations",
          "doped fiber: rate and propagation equations\n"
          "SOA: carrier rate and propagation equations\n"
          "Raman: coupled pump-signal equations")
    w_eq = 2.45
    x_card = XL + PX + w_eq + GX
    w_card = XR - PX - x_card
    names = card_files()
    stems = [os.path.splitext(nm)[0] for nm in names]
    card = ("Parameter card files (.json)",
            "\n".join(wrap_names(ax, stems, w_card - 0.25)
                      + ["every entry: value, unit, source, category, verification level",
                         "sensitivity of the outputs to each entry (±10%)"]))
    h1 = hbox(max(nlines(*eq), nlines(*card)))

    n_src = 4
    w_src = (XR - XL - 2 * PX - (n_src - 1) * GX) / n_src
    x_src = [XL + PX + i * (w_src + GX) for i in range(n_src)]
    if SIMULATOR_RUN:
        sim_body = (SIMULATOR_NAME + ",\n" if SIMULATOR_NAME else "") + "same card and same\noperating points"
    else:
        sim_body = "same card and same\noperating point\n(proposed, not run here)"
    src = [("measured_data", "Measured data",
            "gain and ASE spectra of the\nend-to-end device at six pump\npowers (published, digitized)", False),
           ("closed_forms", "Closed-form models",
            "photon balance (doped fiber),\nhomogeneous saturation (SOA),\nundepleted pump (Raman)", False),
           ("numerical_solvers", "Numerical solvers",
            "initial-value, boundary-value\nand forward-backward sweep\nsolvers, with or without ASE", False),
           ("independent_simulator", "Independent simulator", sim_body, not SIMULATOR_RUN)]
    h_src = hbox(max(nlines(t, b) for _, t, b, _ in src))
    w_prot = XR - XL - 2 * PX
    prot_title = "Comparison at the same card and operating point"
    prot_body = "\n".join(wrap_names(ax, ["outputs: gain and ASE or noise-figure spectra",
                                          "metrics: root-mean-square error, maximum absolute error, bias",
                                          "error budget: numerical, model-form and parameter error"],
                                     w_prot - 0.3))
    h_prot = hbox(nlines(prot_title, prot_body))

    # L3: the design box names the calibrated card and its validated range (measured operating points of
    # Section 5.7: 8 m of fiber, 18-115 mW of pump); designs at other lengths extrapolate. Relative box widths
    # 0.97 : 1.03 in both rows, so that its title fits and the box edges of the two rows stay aligned.
    l3 = [[("literature_tables", "Literature benchmark tables",
            "published results with their\noperating conditions", False, 0.97),
           ("design_calibrated_card", "Design with the calibrated card",
            "(validated at 8 m, 18–115 mW):\nfiber length and pump power", False, 1.03)],
          [("hybrid_design", "Raman/erbium hybrid design",
            "pump allocation and budget\n(component demonstration)", False, 0.97),
           ("multiobjective", "Multi-objective design",
            "noise figure, wall-plug\nefficiency, cost (outlook)", True, 1.03)]]
    # L4: least squares demonstrated (Section 5.7); physics-informed neural networks implemented and illustrated on
    # synthetic Raman problems (Supporting Information: single-instance networks, Figure S3; one parameterized
    # network, Figure S4); outlook dashed. An optional fifth item is the relative box width within its row.
    l4 = [[("least_squares", "Least-squares identification",
            "fit on part of the measured data, blind prediction of the rest;\n"
            "identified entries stored in a calibrated card", False)],
          [("pinn", "Physics-informed neural networks",
            "single-instance and parameterized,\nillustrated on synthetic data\n(Supporting Information)", False, 1.5),
           ("twin_transient_control", "Outlook", "digital twin,\ntransient prediction,\ngain control", True, 1.0)]]
    h_r1 = hbox(max(nlines(it[1], it[2]) for row in (l3[0], l4[0]) for it in row))
    h_r2 = hbox(max(nlines(it[1], it[2]) for row in (l3[1], l4[1]) for it in row))

    # ---------------- vertical layout (inches from the top edge) ----------------
    t1 = 0.13
    y1 = t1 + 0.16
    b1 = y1 + h1 + 0.08
    t2 = b1 + 0.24
    y_src = t2 + 0.16
    y_prot = y_src + h_src + 0.18
    b2 = y_prot + h_prot + 0.08
    t3 = b2 + 0.24
    r1 = t3 + 0.16
    r2 = r1 + h_r1 + 0.08
    b3 = r2 + h_r2 + 0.08
    t_rel, h_rel = b3 + 0.14, 0.30
    y_leg = t_rel + h_rel + 0.18
    H = y_leg + 0.14
    fig.set_size_inches(W, H)
    ax.set_ylim(0, H)

    def Y(yd):          # y measured downwards from the top edge
        return H - yd

    def band(x0, x1, yt, yb, key, name):
        c = col[key]
        rbox(ax, x0, Y(yb), x1 - x0, yb - yt, ec=INK, fc="white", lw=0.6, r=0.05, z=1)
        s = f"{key}   {name}"
        w = text_width(ax, s, weight="bold") + 0.16
        rbox(ax, x0 + 0.1, Y(yt) - 0.085, w, 0.17, ec=INK, fc="white", lw=0.8, r=0.04, z=2)
        ax.text(x0 + 0.1 + w / 2, Y(yt), s, ha="center", va="center", fontsize=FS, fontweight="bold",
                color=INK, zorder=3)
        return w

    def sbox(key, layer, x0, yt, w, h, title, body, dashed=False):
        c = col[layer]
        illustrated = key in ILLUSTRATED and not dashed
        label_box(ax, x0, Y(yt + h), w, h, title, body, ec=c, dashed=dashed, lw=1.0 if not dashed else 0.95,
                  linestyle=DASHDOT if illustrated else None, color=INK if illustrated else None)
        widest = max([text_width(ax, s, weight="bold") for s in title.split("\n") if s]
                     + [text_width(ax, s) for s in body.split("\n") if s])
        if widest > w - 0.06:           # keep at least 0.03 in between the text and the box outline
            print(f"warning: Figure 2 box '{key}': text {widest:.2f} in wide in a box of {w:.2f} in")
        status.append({"layer": layer, "box": key, "title": title.replace("\n", " "),
                       "text": body.replace("\n", " "),
                       "status": "proposed or outlook" if dashed else (
                           "implemented, illustrated in the Supporting Information (dash-dotted outline; status "
                           "table of the Supporting Information: implemented, illustrated only)"
                           if illustrated else "demonstrated in this paper")})

    # ---------------- L1 ----------------
    band(XL, XR, t1, b1, "L1", "Physics and parameter cards")
    sbox("equations", "L1", XL + PX, y1, w_eq, h1, *eq)
    sbox("card_files", "L1", x_card, y1, w_card, h1, *card)
    card_mid_y = Y(y1 + h1 / 2)

    # ---------------- L2 ----------------
    band(XL, XR, t2, b2, "L2", "Common comparison protocol")
    for (key, title, body, dashed), x0 in zip(src, x_src):
        sbox(key, "L2", x0, y_src, w_src, h_src, title, body, dashed=dashed)
    sbox("protocol", "L2", XL + PX, y_prot, w_prot, h_prot, prot_title, prot_body)
    for i, x0 in enumerate(x_src):
        xm = x0 + w_src / 2
        arrow(ax, (xm, Y(y_src + h_src)), (xm, Y(y_prot)), col=FLOW,
              dashed=(src[i][3]))
    # L1 -> model sources (equations and cards feed every model; the measured data do not come from L1)
    for i, frac in ((1, 0.35), (2, 0.5), (3, 0.5)):
        xm = x_src[i] + frac * w_src
        arrow(ax, (xm, Y(y1 + h1)), (xm, Y(y_src)), col=FLOW, dashed=src[i][3])

    # ---------------- L3 and L4 ----------------
    xm3 = (XL + XR) / 2
    L3 = (XL, xm3 - 0.05)
    L4 = (xm3 + 0.05, XR)
    band(L3[0], L3[1], t3, b3, "L3", "Benchmark and design")
    band(L4[0], L4[1], t3, b3, "L4", "Identification and surrogates")
    for (x_a, x_b), layer, rows in ((L3, "L3", l3), (L4, "L4", l4)):
        for row, (yt, hh) in zip(rows, ((r1, h_r1), (r2, h_r2))):
            n = len(row)
            avail = x_b - x_a - 2 * PX - (n - 1) * GX
            wts = [it[4] if len(it) > 4 else 1.0 for it in row]
            x0 = x_a + PX
            for (key, title, body, dashed, *_), wt in zip(row, wts):
                wb = avail * wt / sum(wts)
                sbox(key, layer, x0, yt, wb, hh, title, body, dashed=dashed)
                x0 += wb + GX
    # protocol -> L3 and L4
    for (x_a, x_b), frac in ((L3, 0.72), (L4, 0.80)):
        xm = x_a + frac * (x_b - x_a)
        arrow(ax, (xm, Y(y_prot + h_prot)), (xm, Y(t3)), col=FLOW)
    # feedback: identified entries -> card files (demonstrated, solid)
    ls_mid = Y(r1 + h_r1 / 2)
    polyarrow(ax, [(XR - PX, ls_mid), (XF, ls_mid), (XF, card_mid_y), (XR - PX, card_mid_y)],
              col=FEED, lw=1.1)
    ax.text(XF + 0.04, (ls_mid + card_mid_y) / 2, "card update", rotation=90, ha="left", va="center",
            fontsize=FS, color=FEED)

    # ---------------- release ----------------
    wr1 = 4.05
    rbox(ax, XL, Y(t_rel + h_rel), wr1, h_rel, ec=INK, lw=1.0)
    ax.text(XL + wr1 / 2, Y(t_rel + h_rel / 2),
            "Release: figure scripts, card files, reference data and result files",
            ha="center", va="center", fontsize=FS, color=INK)
    status.append({"layer": "release", "box": "scripts_cards_data", "title": "figure scripts, card files, "
                   "reference data and result files", "status": "demonstrated in this paper"})
    xr2 = XL + wr1 + GX
    rbox(ax, xr2, Y(t_rel + h_rel), XR - xr2, h_rel, ec=INK, lw=1.0 if REPOSITORY_PUBLIC else 0.95,
         dashed=not REPOSITORY_PUBLIC)
    ax.text((xr2 + XR) / 2, Y(t_rel + h_rel / 2), "public repository with persistent identifier",
            ha="center", va="center", fontsize=FS, color=INK if REPOSITORY_PUBLIC else INK2)
    status.append({"layer": "release", "box": "public_repository", "title": "public repository with "
                   "persistent identifier",
                   "status": "demonstrated in this paper" if REPOSITORY_PUBLIC else "proposed or outlook"})

    # ---------------- legend (centered): the three outline styles ----------------
    items = [("demonstrated in this paper", "-"),
             ("implemented, illustrated in the Supporting Information", DASHDOT),
             ("proposed or outlook", DASH)]
    widths = [0.30 + 0.08 + text_width(ax, s) for s, _ in items]
    gap = 0.40
    if sum(widths) + gap * (len(items) - 1) > XR - XL:
        print("warning: Figure 2 legend wider than the figure")
    x = (XL + XR) / 2 - (sum(widths) + gap * (len(items) - 1)) / 2
    yl = Y(y_leg)
    for (s, ls_), wi in zip(items, widths):
        rbox(ax, x, yl - 0.065, 0.30, 0.13, ec=INK, lw=0.95 if ls_ is DASH else 1.0, linestyle=ls_)
        ax.text(x + 0.38, yl, s, ha="left", va="center", fontsize=FS, color=INK)
        x += wi + gap

    save(fig, "Fig02_framework")
    return {"card_files_listed": names, "boxes": status,
            "switches": {"SIMULATOR_RUN": SIMULATOR_RUN, "SIMULATOR_NAME": SIMULATOR_NAME,
                         "REPOSITORY_PUBLIC": REPOSITORY_PUBLIC},
            "figure_size_in": [W, round(H, 3)]}


# ----------------------------------------------------------------------------------------------
# Figure 7: architectures
# ----------------------------------------------------------------------------------------------
PUMP = C[1]          # pump sources and pump light
ER = C[0]            # erbium-doped fiber and EDFA
RAM = C[5]           # Raman gain fiber
SOAC = C[3]          # semiconductor optical amplifiers
MOD = "#efeee9"      # housing of a module


def panel_a(ax, ox, oy):
    def P(x, y):
        return ox + x, oy + y
    yl = 0.82
    p1, p2, p3 = circulator(ax, *P(0.68, yl), r=0.15)
    arrow(ax, P(0.05, yl), p1)
    ax.text(*P(0.05, yl - 0.06), "signal in", ha="left", va="top", fontsize=FS, color=INK)
    polyarrow(ax, [p3, P(0.68, 0.30), P(0.05, 0.30)])
    ax.text(*P(0.07, 0.34), "signal out", ha="left", va="bottom", fontsize=FS, color=INK)
    tbox(ax, *P(1.24, yl), 0.40, 0.24, "WDM")
    arrow(ax, p2, P(1.04, yl), both=True)
    tbox(ax, *P(1.24, 0.33), 0.84, 0.32, "pump\n980 or 1480 nm", ec=shade(PUMP, 0.9), fc=tint(PUMP, 0.14))
    arrow(ax, P(1.24, 0.49), P(1.24, yl - 0.12), col=shade(PUMP, 0.9))
    line(ax, [P(1.44, yl), P(1.58, yl)])
    b = coil(ax, ox + 1.58, ox + 2.40, oy + yl, n=5, col=shade(ER, 0.9), lw=1.1)
    ax.text(*P(1.99, yl - b - 0.05), "Er-doped fiber", ha="center", va="top", fontsize=FS, color=INK)
    line(ax, [P(2.40, yl), P(2.62, yl)])
    tbox(ax, *P(2.84, yl), 0.44, 0.30, "FRM", ec=C[6], fc=tint(C[6], 0.12))
    mirror(ax, ox + 3.09, oy + yl - 0.15, oy + yl + 0.15)
    arrow(ax, P(1.62, yl + 0.22), P(2.36, yl + 0.22), both=True, lw=0.8)
    ax.text(*P(1.99, yl + 0.27), "two passes", ha="center", va="bottom", fontsize=FS, color=INK2)


def panel_b(ax, ox, oy):
    def P(x, y):
        return ox + x, oy + y
    yl = 0.82
    arrow(ax, P(0.05, yl), P(0.32, yl))
    ax.text(*P(0.05, yl - 0.06), "signal in", ha="left", va="top", fontsize=FS, color=INK)
    line(ax, [P(0.32, yl), P(0.62, yl)], lw=1.2)
    b = coil(ax, ox + 0.62, ox + 1.42, oy + yl, n=5, col=INK2, lw=1.1)
    line(ax, [P(1.42, yl), P(1.76, yl)], lw=1.2)
    ax.text(*P(1.02, yl + b + 0.05), "transmission fiber", ha="center", va="bottom", fontsize=FS, color=INK)
    tbox(ax, *P(1.96, yl), 0.40, 0.24, "WDM")
    tbox(ax, *P(1.96, 0.30), 1.14, 0.32, "Raman pumps\n(several wavelengths)", ec=shade(PUMP, 0.9),
         fc=tint(PUMP, 0.14))
    arrow(ax, P(1.96, 0.46), P(1.96, yl - 0.12), col=shade(PUMP, 0.9))
    arrow(ax, P(1.70, yl - 0.20), P(0.40, yl - 0.20), col=shade(PUMP, 0.9), dashed=True)
    ax.text(*P(1.05, yl - 0.25), "counter-propagating pump light", ha="center", va="top", fontsize=FS,
            color=shade(PUMP, 0.9))
    (ai, ao) = amp(ax, *P(2.55, yl), w=0.46, h=0.32, s="EDFA", col=ER)
    line(ax, [P(2.16, yl), ai])
    tbox(ax, *P(3.10, yl), 0.40, 0.24, "GFF")
    line(ax, [ao, P(2.90, yl)])
    arrow(ax, P(3.30, yl), P(3.45, yl))


def panel_c(ax, ox, oy):
    def P(x, y):
        return ox + x, oy + y
    pc = shade(PUMP, 0.9)

    def span(y):
        line(ax, [P(0.30, y), P(0.40, y)], lw=1.2)
        coil(ax, ox + 0.40, ox + 0.78, oy + y, n=3, col=INK2, lw=1.0)
        line(ax, [P(0.78, y), P(0.86, y)], lw=1.2)

    def split_chain(y, xc):
        """Coupler at xc, band splitter, two band SOAs, band combiner; returns the coupler bottom point."""
        tbox(ax, *P(xc, y), 0.34, 0.22, "WDM")
        line(ax, [P(0.86, y), P(xc - 0.17, y)], lw=1.2)
        hsp = 0.60
        tbox(ax, *P(1.66, y), 0.52, hsp, "band\nsplitter")
        arrow(ax, P(xc + 0.17, y), P(1.40, y))
        for dy in (0.16, -0.16):
            ai, ao = amp(ax, *P(2.25, y + dy), w=0.44, h=0.28, s="SOA", col=SOAC)
            arrow(ax, (ox + 1.92, oy + y + dy), ai)
            arrow(ax, ao, (ox + 2.64, oy + y + dy))
        tbox(ax, *P(2.90, y), 0.52, hsp, "band\ncombiner")
        arrow(ax, P(3.16, y), P(3.44, y))
        return P(xc, y - 0.11)

    # i) one wideband SOA module, no band splitter
    y = 2.30
    ax.text(*P(0.02, y), "i)", ha="left", va="center", fontsize=FS, color=INK)
    span(y)
    ax.text(*P(0.59, y + 0.10), "span", ha="center", va="bottom", fontsize=FS, color=INK)
    tbox(ax, *P(1.12, y), 0.34, 0.22, "WDM")
    line(ax, [P(0.86, y), P(0.95, y)], lw=1.2)
    tbox(ax, *P(1.12, y - 0.37), 0.80, 0.22, "Raman pumps", ec=pc, fc=tint(PUMP, 0.14))
    arrow(ax, P(1.12, y - 0.26), P(1.12, y - 0.11), col=pc)
    ai, ao = amp(ax, *P(1.92, y), w=0.48, h=0.32, s="SOA", col=SOAC)
    arrow(ax, P(1.29, y), ai)
    arrow(ax, ao, P(2.55, y))
    ax.text(*P(1.92, y + 0.19), "wideband module", ha="center", va="bottom", fontsize=FS, color=INK)

    # ii) band splitter in front of band-specific SOAs; pumps behind it, routed around it
    y = 1.42
    ax.text(*P(0.02, y), "ii)", ha="left", va="center", fontsize=FS, color=INK)
    span(y)
    cb = split_chain(y, 1.07)
    yp = 0.92
    tbox(ax, *P(2.25, yp), 0.80, 0.22, "Raman pumps", ec=pc, fc=tint(PUMP, 0.14))
    polyarrow(ax, [P(1.85, yp), P(1.07, yp), cb], col=pc, lw=1.0)
    ax.text(*P(1.46, yp + 0.03), "pump bypass", ha="center", va="bottom", fontsize=FS, color=pc)

    # iii) band splitter; separate Raman pump module in front of it
    y = 0.42
    ax.text(*P(0.02, y), "iii)", ha="left", va="center", fontsize=FS, color=INK)
    rbox(ax, *P(0.84, -0.02), 0.52, 0.64, ec=INK2, fc=MOD, lw=0.7, r=0.04, z=1)
    span(y)
    cb = split_chain(y, 1.10)
    tbox(ax, *P(1.10, 0.13), 0.42, 0.16, "pumps", ec=pc, fc=tint(PUMP, 0.14))
    arrow(ax, P(1.10, 0.21), cb, col=pc)
    ax.text(*P(1.34, 0.65), "Raman pump module", ha="right", va="bottom", fontsize=FS, color=INK)


def panel_d(ax, ox, oy):
    def P(x, y):
        return ox + x, oy + y
    bands = [("O", "PDFA or BDFA", C[2]), ("E", "BDFA", C[4]), ("S", "TDFA, BDFA or Raman", C[6]),
             ("C", "EDFA", C[0]), ("L", "EDFA", C[3]), ("U", "discrete Raman", C[7])]
    ytop, ybot = 2.24, 0.26
    ymid = (ytop + ybot) / 2
    arrow(ax, P(0.02, ymid), P(0.30, ymid))
    tbox(ax, ox + 0.56, oy + ymid, 0.52, ytop - ybot + 0.30, "band\nsplitter")
    tbox(ax, ox + 2.92, oy + ymid, 0.52, ytop - ybot + 0.30, "band\ncombiner")
    arrow(ax, P(3.18, ymid), P(3.44, ymid))
    for k, (b, s, col) in enumerate(bands):
        y = ytop - k * (ytop - ybot) / (len(bands) - 1)
        tbox(ax, *P(1.74, y), 1.56, 0.26, f"{b}-band: {s}", ec=shade(col, 0.85), fc=tint(col, 0.13))
        arrow(ax, P(0.82, y), P(0.96, y))
        arrow(ax, P(2.52, y), P(2.66, y))


def panel_e(ax, ox, oy):
    def P(x, y):
        return ox + x, oy + y
    yl = 0.86
    p1, p2, p3 = circulator(ax, *P(0.60, yl), r=0.15)
    arrow(ax, P(0.05, yl), p1)
    ax.text(*P(0.05, yl - 0.06), "signal in", ha="left", va="top", fontsize=FS, color=INK)
    polyarrow(ax, [p3, P(0.60, 0.40), P(0.05, 0.40)])
    ax.text(*P(0.07, 0.44), "signal out", ha="left", va="bottom", fontsize=FS, color=INK)
    tbox(ax, *P(1.18, yl), 0.56, 0.86, "splitter/\ncombiner")
    arrow(ax, p2, P(0.90, yl), both=True)
    yb1, yb2 = 1.15, 0.57
    # branch 1: Er-doped fiber + Raman fiber
    arrow(ax, P(1.46, yb1), P(1.56, yb1), both=True, ms=5)
    b = coil(ax, ox + 1.56, ox + 2.04, oy + yb1, n=3, col=shade(ER, 0.9), lw=1.1)
    ax.text(*P(1.80, yb1 + b + 0.04), "EDF", ha="center", va="bottom", fontsize=FS, color=INK)
    line(ax, [P(2.04, yb1), P(2.16, yb1)])
    coil(ax, ox + 2.16, ox + 2.82, oy + yb1, n=4, col=shade(RAM, 0.95), lw=1.1)
    ax.text(*P(2.49, yb1 + b + 0.04), "Raman fiber", ha="center", va="bottom", fontsize=FS, color=INK)
    arrow(ax, P(2.82, yb1), P(3.07, yb1), both=True, ms=5)
    # branch 2: Raman fiber only
    arrow(ax, P(1.46, yb2), P(1.66, yb2), both=True, ms=5)
    coil(ax, ox + 1.66, ox + 2.66, oy + yb2, n=6, col=shade(RAM, 0.95), lw=1.1)
    ax.text(*P(2.16, yb2 + b + 0.04), "Raman fiber", ha="center", va="bottom", fontsize=FS, color=INK)
    arrow(ax, P(2.66, yb2), P(3.07, yb2), both=True, ms=5)
    mirror(ax, ox + 3.10, oy + yb2 - 0.17, oy + yb1 + 0.17)
    ax.text(*P(3.10, yb1 + 0.22), "mirror", ha="center", va="bottom", fontsize=FS, color=INK)
    pc = shade(PUMP, 0.9)
    tbox(ax, *P(2.16, 0.16), 1.62, 0.30, "Raman pumps 1410, 1480, 1495 nm\n(residual pump recycled)", ec=pc,
         fc=tint(PUMP, 0.14))
    arrow(ax, P(2.16, 0.31), P(2.16, yb2 - b - 0.01), col=pc)


def panel_f(ax, ox, oy):
    def P(x, y):
        return ox + x, oy + y
    yl = 0.82
    tbox(ax, *P(0.24, yl), 0.42, 0.40, "OLT")
    line(ax, [P(0.45, yl), P(0.56, yl)], lw=1.2)
    b = coil(ax, ox + 0.56, ox + 0.90, oy + yl, n=3, col=INK2, lw=1.0)
    line(ax, [P(0.90, yl), P(1.04, yl)], lw=1.2)
    ax.text(*P(0.73, yl + b + 0.05), "feeder\nfiber", ha="center", va="bottom", fontsize=FS, color=INK)
    # reach extender: diplexers (WDM filters) and one amplifier per direction
    rbox(ax, *P(1.04, 0.10), 1.24, 1.36, ec=INK2, fc=MOD, lw=0.7, r=0.04, z=1)
    ax.text(*P(1.66, 1.49), "reach extender", ha="center", va="bottom", fontsize=FS, color=INK)
    for xd in (1.17, 2.15):
        rbox(ax, ox + xd - 0.08, oy + yl - 0.31, 0.16, 0.62, ec=INK2, fc="white")
        ax.text(ox + xd, oy + yl, "WDM", rotation=90, ha="center", va="center", fontsize=FS, color=INK, zorder=4)
    yd, yu = yl + 0.20, yl - 0.20
    ai, ao = amp(ax, *P(1.66, yd), w=0.40, h=0.26, s="", col=INK2)
    line(ax, [P(1.25, yd), ai])
    arrow(ax, ao, P(2.07, yd))
    tip, base = amp(ax, *P(1.66, yu), w=0.40, h=0.26, s="", col=INK2, left=True)   # points left
    line(ax, [P(2.07, yu), base])
    arrow(ax, tip, P(1.25, yu))
    ax.text(*P(1.66, yd + 0.16), "downstream\n(S- or L-band)", ha="center", va="bottom", fontsize=FS, color=INK)
    ax.text(*P(1.66, yu - 0.16), "upstream (O-band,\nburst mode)", ha="center", va="top", fontsize=FS,
            color=INK)
    line(ax, [P(2.23, yl), P(2.28, yl)], lw=1.2)
    line(ax, [P(2.28, yl), P(2.56, yl)], lw=1.2)
    tbox(ax, *P(2.72, yl), 0.32, 0.36, "1:N")
    for yy in (1.30, 0.82, 0.34):
        line(ax, [P(2.88, yl), P(3.06, yy)], lw=0.9)
        tbox(ax, *P(3.24, yy), 0.40, 0.22, "ONU")


def figure7():
    W = 7.1
    cw, gx = 3.45, 0.17
    rows = [1.40, 2.62, 1.66]       # panel heights (a, b), (c, d), (e, f)
    gy = 0.26
    H = sum(rows) + 2 * gy + 0.06
    fig, ax = canvas(W, H)
    ytops = [H - 0.03, H - 0.03 - rows[0] - gy, H - 0.03 - rows[0] - rows[1] - 2 * gy]
    panels = [("a", panel_a, 0, 0), ("b", panel_b, 1, 0), ("c", panel_c, 0, 1), ("d", panel_d, 1, 1),
              ("e", panel_e, 0, 2), ("f", panel_f, 1, 2)]
    for letter, fn, col_, row in panels:
        ox = 0.02 + col_ * (cw + gx)
        oy = ytops[row] - rows[row]
        fn(ax, ox, oy)
        style.label(ax, letter, x=ox / W, y=(ytops[row] - 0.17) / H)      # bold "a)" at the top left
    save(fig, "Fig07_architectures")
    abbreviations = {
        "WDM": "wavelength-division-multiplexing (pump/signal) coupler or band filter",
        "FRM": "Faraday rotator mirror", "EDFA": "erbium-doped fiber amplifier",
        "EDF": "erbium-doped fiber", "GFF": "gain-flattening filter", "SOA": "semiconductor optical amplifier",
        "PDFA": "praseodymium-doped fiber amplifier", "BDFA": "bismuth-doped fiber amplifier",
        "TDFA": "thulium-doped fiber amplifier", "OLT": "optical line terminal", "ONU": "optical network unit",
        "PON": "passive optical network (caption only)"}
    return {"panels": {
        "a": "double-pass doped-fiber amplifier: circulator, WDM coupler, pump (980 or 1480 nm), Er-doped fiber, FRM",
        "b": "transmission fiber with counter-propagating Raman pumps (several wavelengths) via WDM coupler, EDFA, GFF",
        "c": "i) Raman pumps + one wideband SOA module, no band splitter; ii) band splitter in front of two band SOAs, "
             "pumps behind the splitter routed around it (bypass) to a coupler on the span side; iii) separate "
             "Raman pump module (coupler + pumps) in front of the band splitter",
        "d": "band splitter, O: PDFA or BDFA, E: BDFA, S: TDFA, BDFA or Raman, C: EDFA, L: EDFA, U: discrete Raman, "
             "band combiner",
        "e": "circulator, splitter/combiner, branch 1 EDF + Raman fiber, branch 2 Raman fiber, mirror, Raman pumps "
             "1410/1480/1495 nm with residual-pump recycling (drawn after [80, 81]; pump routing simplified)",
        "f": "OLT, feeder fiber, reach extender with two diplexers, downstream (S- or L-band) and upstream "
             "(O-band, burst mode) amplifiers, passive 1:N splitter, ONUs"},
        "abbreviations": abbreviations, "figure_size_in": [W, round(H, 3)],
        "removed_terms": ["REDFA", "UWB", "DS/US", "Tx", "LD", "Circ."]}


def remove_legacy():
    removed = []
    for fn in LEGACY_OUTPUTS:
        p = os.path.join(FIGDIR, fn)
        if os.path.exists(p):
            os.remove(p)
            removed.append(fn)
    return removed


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    removed = remove_legacy()
    r2 = figure2()
    r7 = figure7()
    meta = {"script": "fig02_fig05_schematics.py", "removed_legacy_outputs": removed,
            "note": "schematics only; no model is evaluated"}
    with open(os.path.join(FIGDIR, "Fig02_framework_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, **r2), f, indent=2, ensure_ascii=False)
    with open(os.path.join(FIGDIR, "Fig07_architectures_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, **r7), f, indent=2, ensure_ascii=False)
    print("Figure 2: card files listed:", ", ".join(r2["card_files_listed"]))
    print("Figure 2: boxes:", sum(b["status"].startswith("demonstrated") for b in r2["boxes"]), "demonstrated,",
          sum(b["status"].startswith("implemented") for b in r2["boxes"]), "implemented and illustrated,",
          sum(b["status"].startswith("proposed") for b in r2["boxes"]), "proposed or outlook")
    print("written: Fig02_framework, Fig07_architectures (png, pdf, results json)")
