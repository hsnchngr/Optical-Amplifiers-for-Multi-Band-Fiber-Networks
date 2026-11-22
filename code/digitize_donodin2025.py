"""Digitize Figure 3e,f of ref. [69] (Donodin et al., Opt. Lett. 50(20), 6369-6372, 2025; CC BY 4.0) from the vector paths
of the published PDF and write ../data/donodin2025_fig3ef_digitized.csv.

The PDF is not distributed with this code. Obtain it from the publisher (doi 10.1364/OL.575125) or the Aston University
repository (publications.aston.ac.uk, eprint 48242) and run

    python -I digitize_donodin2025.py <path-to-pdf> [out.csv]

Method: the gain (blue, RGB 0/0.447/0.74) and NF (orange, RGB 0.852/0.325/0.098) curves of Figure 3 are vector polylines
through the measured points (5 nm steps). Each panel is calibrated with its own tick marks (gray vector segments matched
to the numeric tick labels), so the residual of the axis calibration is below 0.001 nm and 0.001 dB. Panels e (0 dBm)
and f (-20 dBm) belong to the 400 m amplifier with 950 mW of pump power.
"""
import os
import re
import sys

import numpy as np

BLUE = (0.0, 0.447, 0.74)
ORANGE = (0.852, 0.325, 0.098)


def _close(c1, c2, tol=0.01):
    return c1 is not None and all(abs(a - b) < tol for a, b in zip(c1, c2))


def _polyline(d):
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            p1, p2 = it[1], it[2]
            if not pts or abs(pts[-1][0] - p1.x) > 1e-3 or abs(pts[-1][1] - p1.y) > 1e-3:
                pts.append((p1.x, p1.y))
            pts.append((p2.x, p2.y))
    return pts


def digitize(pdf_path, page_index=2):
    import pymupdf  # only needed for digitizing
    page = pymupdf.open(pdf_path)[page_index]
    gain, nf, segs = [], [], []
    for d in page.get_drawings():
        col = d.get("color")
        if _close(col, BLUE) and len(_polyline(d)) >= 5:
            gain.append((d["rect"], _polyline(d)))
        elif _close(col, ORANGE) and len(_polyline(d)) >= 5:
            nf.append((d["rect"], _polyline(d)))
        else:
            for it in d["items"]:
                if it[0] == "l":
                    segs.append((it[1].x, it[1].y, it[2].x, it[2].y))
    labels = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].strip()
                if re.fullmatch(r"\d{1,4}", t):
                    x0, y0, x1, y1 = s["bbox"]
                    labels.append((int(t), (x0 + x1) / 2, (y0 + y1) / 2))
    panels = []
    for rect, pts in gain:
        vs = [s for s in segs if abs(s[0] - s[2]) < 0.05 and abs(s[3] - s[1]) > 30 and abs(s[0] - rect.x0) < 6
              and min(s[1], s[3]) <= rect.y0 + 1 and max(s[1], s[3]) >= rect.y1 - 1]
        vy0 = min(min(s[1], s[3]) for s in vs)
        vy1 = max(max(s[1], s[3]) for s in vs)
        yaxis_x, xaxis_y = vs[0][0], vy1
        xt = sorted({round(s[0], 3) for s in segs if abs(s[0] - s[2]) < 0.05 and 0.5 < abs(s[3] - s[1]) < 5
                     and min(abs(s[1] - xaxis_y), abs(s[3] - xaxis_y)) < 0.2 and rect.x0 - 2 <= s[0] <= rect.x1 + 2})
        yt = sorted({round(s[1], 3) for s in segs if abs(s[1] - s[3]) < 0.05 and 0.5 < abs(s[2] - s[0]) < 5
                     and min(abs(s[0] - yaxis_x), abs(s[2] - yaxis_x)) < 0.2 and vy0 - 2 <= s[1] <= vy1 + 2})
        xl = [l for l in labels if 1400 <= l[0] <= 1600 and xaxis_y < l[2] < xaxis_y + 15 and rect.x0 - 15 < l[1] < rect.x1 + 15]
        yl = [l for l in labels if l[0] <= 60 and yaxis_x - 25 < l[1] < yaxis_x and vy0 - 6 < l[2] < vy1 + 6]
        xm = [(min(xt, key=lambda v: abs(v - l[1])), l[0]) for l in xl]
        ym = [(min(yt, key=lambda v: abs(v - l[2])), l[0]) for l in yl]
        ax_, bx_ = np.polyfit([m[0] for m in xm], [m[1] for m in xm], 1)
        ay_, by_ = np.polyfit([m[0] for m in ym], [m[1] for m in ym], 1)
        lam = [ax_ * p[0] + bx_ for p in pts]
        g = [ay_ * p[1] + by_ for p in pts]
        nfc = None
        for r2, p2 in nf:
            if abs(r2.x0 - rect.x0) < 8 and vy0 - 2 <= r2.y0 and r2.y1 <= vy1 + 2:
                nfc = ([ax_ * p[0] + bx_ for p in p2], [ay_ * p[1] + by_ for p in p2])
        panels.append({"x0": rect.x0, "y0": rect.y0, "lambda_nm": lam, "gain_dB": g, "nf": nfc})
    # order: left-to-right columns (250, 700, 950 mW), top (0 dBm) then bottom (-20 dBm)
    panels.sort(key=lambda p: (round(p["x0"]), p["y0"]))
    return panels


def _dedup(lam, val):
    out = []
    for l, v in zip(lam, val):
        if out and abs(out[-1][0] - l) < 0.05:
            continue
        out.append((l, v))
    return out


if __name__ == "__main__":
    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                                                            "donodin2025_fig3ef_digitized.csv")
    p = digitize(pdf)
    e, f = p[4], p[5]  # 950 mW: e = 0 dBm (top), f = -20 dBm (bottom)
    ge, gf = _dedup(e["lambda_nm"], e["gain_dB"]), _dedup(f["lambda_nm"], f["gain_dB"])
    ne, nf = _dedup(*e["nf"]), _dedup(*f["nf"])
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Digitized with digitize_donodin2025.py from Figure 3e,f of ref. [69] (CC BY 4.0); see header of the released file.\n")
        fh.write("lambda_nm,gain_m20dBm_dB,nf_m20dBm_dB,gain_0dBm_dB,nf_0dBm_dB\n")
        for (l, g20), (_, n20), (_, g0), (_, n0) in zip(gf, nf, ge, ne):
            fh.write(f"{l:.1f},{g20:.2f},{n20:.2f},{g0:.2f},{n0:.2f}\n")
    print("written", out)
