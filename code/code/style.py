"""Shared plotting style for all figures (Wiley-VCH journal style, 300 dpi PNG and vector PDF).

Conventions used by every figure script:
  * sans-serif font of one family; panel labels 'a)', 'b)', ... in bold at the top left of each panel (style.label);
  * no in-figure titles; the caption explains every panel, inset and line style;
  * axis labels as 'Quantity [unit]', American spelling.
"""
import sys
import matplotlib as mpl

# Console output of the scripts contains Greek letters; make it safe on Windows code pages.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - older Python or redirected streams
    pass

# Categorical palette (fixed order, color-blind validated reference set)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def math_in_text_font():
    """Typeset math text ($\\alpha$, $P_\\mathrm{p}$, $-$, ...) in the text font, Arial, so that a figure uses one font
    family (with apply() alone, math text is set in DejaVu Sans; log-axis tick labels use the text font either way).
    Glyphs that Arial lacks fall back to STIX Sans. Call after apply(); no effect if Arial is not installed. Used by
    the scripts of Figures 4, 5, 8-11 and S1-S5."""
    from matplotlib import font_manager
    if "Arial" in {f.name for f in font_manager.fontManager.ttflist}:
        mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                             "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.fallback": "stixsans"})


def label(ax, letter, x=-0.16, y=1.02):
    """Bold panel label 'a)' at the top left of an axes (Wiley-VCH convention)."""
    ax.text(x, y, f"{letter})", transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def panel(ax, letter):  # backward-compatible alias
    label(ax, letter)


def save(fig, name, outdir="../figures"):
    """Save a figure as 300 dpi PNG and as vector PDF."""
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".png"))
    fig.savefig(os.path.join(outdir, name + ".pdf"))
