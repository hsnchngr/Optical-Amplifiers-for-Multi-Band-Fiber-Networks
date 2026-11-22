"""Digitization of Figs. 7 and 8 of Perin et al., J. Lightwave Technol. 37(9), 2076 (2019), ref. [124], from the vector
graphics of the arXiv version (arXiv:1803.07905v3). Data preparation only: the figure scripts read the data files.

Stage 1, extract (needs PyMuPDF and the arXiv PDF, which is not distributed):
    python -I digitize_perin2019.py extract <arXiv PDF> [raw_dir]
  Writes the raw vector paths in PDF page coordinates (points; origin at the top left, y downward) to raw_dir
  (default ../data/raw_perin2019, where the files used for the data files are distributed):
    fig7_alpha_raw.npy, fig7_gstar_raw.npy   vertices (x, y) of the alpha(lambda) and g*(lambda) curves of Fig. 7
                                             (page 9): end points of the line and Bezier segments, consecutive
                                             duplicates removed
    fig8_raw.json                            for each pump power (18.1 ... 114.9 mW): centers of the measurement
                                             markers ('marks', gain and ASE panels) and vertices of the source's two
                                             model curves ('curves') of Fig. 8 (page 10)
  It also prints the gray axis and tick paths from which the axis calibrations of stage 2 were read.

Stage 2, calibrate (NumPy only):
    python digitize_perin2019.py calibrate [--raw DIR] [--out DIR] [--check]
  Converts the raw paths with the linear axis calibrations below and writes
    ../data/perin2019_giles_coefficients.csv   alpha and g* [1/m] on a 0.5 nm grid
    ../data/perin2019_fig8_gain_ase.json       measured gain and ASE, and the source's model curves
  --check compares the result byte by byte with the existing files and writes nothing. From the distributed raw files
  stage 2 reproduces both data files exactly (CRLF line ends, as written on the machine that produced them).

Axis calibrations (PDF points -> axis values, read from the tick marks printed by stage 1):
  Fig. 7       x: 114.85 -> 1480 nm, 284.59 -> 1560 nm;   y: 237.39 -> 0 1/m,   61.13 -> 2 1/m
  Fig. 8 gain  x:  89.75 -> 1530 nm, 274.89 -> 1560 nm;   y: 258.80 -> 0 dB,    89.61 -> 16 dB
  Fig. 8 ASE   x: 338.09 -> 1530 nm, 523.22 -> 1560 nm;   y: 258.80 -> -50 dBm, 89.61 -> -38 dBm
Resampling:
  Fig. 7: both curves sorted by wavelength and interpolated linearly on a 0.5 nm grid from the first to the last whole
          nanometer inside the overlap of the two curves (1470-1569 nm, 199 points); 5 decimals.
  Fig. 8: markers left of x = 310 pt belong to the gain panel, right of it to the ASE panel, each sorted by wavelength
          (40 channels); the model curves are taken vertex by vertex (40 vertices, wavelength increasing); 3 decimals.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw_perin2019")
CSV_NAME = "perin2019_giles_coefficients.csv"
JSON_NAME = "perin2019_fig8_gain_ase.json"
X_SPLIT_FIG8 = 310.0                      # page x [pt] between the gain panel (left) and the ASE panel (right) of Fig. 8


# =====================================================================================================================
# Stage 1: raw vector paths from the arXiv PDF
# =====================================================================================================================
def path_points(d):
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":
            pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]   # end points of the Bezier segment
        elif it[0] == "re":
            r = it[1]; pts += [(r.x0, r.y0), (r.x1, r.y1)]
    # remove consecutive duplicates
    out = []
    for p in pts:
        if not out or (abs(out[-1][0] - p[0]) > 1e-6 or abs(out[-1][1] - p[1]) > 1e-6):
            out.append(p)
    return np.array(out)


def col(d, key="color"):
    v = d.get(key)
    return tuple(round(x, 2) for x in v) if v else ()


def extract(pdf, raw_dir=RAW_DIR):
    """Stage 1: dump the raw vector paths of Figs. 7 and 8 into raw_dir and print the axis and tick paths."""
    import pymupdf
    os.makedirs(raw_dir, exist_ok=True)
    doc = pymupdf.open(pdf)

    # ---------------- page 9 : Fig. 7 ----------------
    page = doc[8]; dr = page.get_drawings()
    print("--- page 9 small drawings (axes/ticks) ---")
    for d in dr:
        if col(d) in [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.15, 0.15, 0.15)] or d.get("fill"):
            P = path_points(d)
            print(col(d), col(d, "fill"), len(d["items"]), [d["items"][0][0]], "bbox", [round(x, 2) for x in d["rect"]])
            if col(d) == (0.5, 0.5, 0.5):
                print("    ", np.round(P, 2).tolist())
    res = {}
    for name, c in (("alpha", (0.89, 0.29, 0.2)), ("gstar", (0.19, 0.51, 0.74))):
        for d in dr:
            if col(d) == c and len(d["items"]) > 50:
                res[name] = path_points(d)
                print(name, "npts", len(res[name]), "x range", res[name][:, 0].min(), res[name][:, 0].max(), "y range", res[name][:, 1].min(), res[name][:, 1].max())
                print("   item kinds:", set(it[0] for it in d["items"]))
    np.save(os.path.join(raw_dir, "fig7_alpha_raw.npy"), res["alpha"])
    np.save(os.path.join(raw_dir, "fig7_gstar_raw.npy"), res["gstar"])

    # ---------------- page 10 : Fig. 8 ----------------
    page = doc[9]; dr = page.get_drawings()
    print("--- page 10 small drawings (axes/ticks) ---")
    for d in dr:
        if col(d) in [(0.5, 0.5, 0.5), (0.15, 0.15, 0.15)] or (col(d) == (0.0, 0.0, 0.0) and len(d["items"]) == 1):
            P = path_points(d)
            print(col(d), col(d, "fill"), len(d["items"]), "bbox", [round(x, 2) for x in d["rect"]])
            if col(d) == (0.5, 0.5, 0.5):
                print("    ", np.round(P, 2).tolist())
    cols = {"18.1": (1.0, 0.5, 0.0), "31.9": (0.6, 0.31, 0.64), "45.8": (0.3, 0.69, 0.29), "59.6": (0.22, 0.49, 0.72), "87.2": (0.89, 0.1, 0.11), "114.9": (0.97, 0.51, 0.75)}
    out = {}
    for lab, c in cols.items():
        marks, curves = [], []
        for d in dr:
            if col(d) != c:
                continue
            if len(d["items"]) == 4:
                r = d["rect"]; marks.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
            elif len(d["items"]) > 10:
                curves.append(path_points(d))
        marks = np.array(marks)
        out[lab] = dict(marks=marks.tolist(), curves=[cv.tolist() for cv in curves])
        print(lab, "marks", marks.shape, "x range", marks[:, 0].min().round(1), marks[:, 0].max().round(1), "curves", [cv.shape for cv in curves],
              [(cv[:, 0].min().round(1), cv[:, 0].max().round(1)) for cv in curves])
    with open(os.path.join(raw_dir, "fig8_raw.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("raw vector paths written to", raw_dir)


# =====================================================================================================================
# Stage 2: axis calibration and resampling -> data files
# =====================================================================================================================
def axis(p0, v0, p1, v1):
    """Linear axis calibration: page coordinate p0 -> value v0, p1 -> v1."""
    return lambda p: v0 + (p - p0) / ((p1 - p0) / (v1 - v0))


FIG7_LAM = axis(114.85, 1480, 284.59, 1560)           # nm
FIG7_COEF = axis(237.39, 0, 61.13, 2)                 # 1/m
FIG8_GAIN_LAM = axis(89.75, 1530, 274.89, 1560)       # nm
FIG8_GAIN_DB = axis(258.80, 0, 89.61, 16)             # dB
FIG8_ASE_LAM = axis(338.09, 1530, 523.22, 1560)       # nm
FIG8_ASE_DBM = axis(258.80, -50, 89.61, -38)          # dBm in 0.1 nm

CSV_HEADER = [
    "# Giles absorption (alpha) and gain (g*) coefficients of the 8 m single-stage C-band EDF of",
    "# J.K. Perin, J.M. Kahn, J.D. Downie, J. Hurley, K. Bennett, J. Lightwave Technol. 37(9), 2076 (2019) [124], Fig. 7,",
    "# digitized from the vector graphics of the arXiv version (arXiv:1803.07905) with digitize_perin2019.py",
    "# (axis calibration: 1480-1560 nm, 0-2 1/m). Category: measured by the source (digitized here), level V1.",
    "lambda_nm,alpha_per_m,gstar_per_m"]
JSON_META = {
    "description": "Measured gain and ASE (0.1 nm resolution) of the EDFA of [124] (Perin et al., JLT 2019), Fig. 8, "
                   "digitized from the vector graphics of arXiv:1803.07905; 40 channels at -13 dBm each, 980 nm forward "
                   "pump, six pump powers. 'source_model' curves are the authors' own model predictions shown in the "
                   "same figure.",
    "category": "measured by the source (digitized here)",
    "level": "V1",
    "reference_plane": "as reported by the source (not stated in detail; to be confirmed)"}


def coefficients_csv(raw_dir=RAW_DIR):
    """Fig. 7: alpha and g* on the 0.5 nm grid (text of the CSV file, LF line ends)."""
    A = np.load(os.path.join(raw_dir, "fig7_alpha_raw.npy"))
    G = np.load(os.path.join(raw_dir, "fig7_gstar_raw.npy"))
    lamA, alA = FIG7_LAM(A[:, 0]), FIG7_COEF(A[:, 1])
    lamG, gsG = FIG7_LAM(G[:, 0]), FIG7_COEF(G[:, 1])
    iA, iG = np.argsort(lamA), np.argsort(lamG)
    lamA, alA, lamG, gsG = lamA[iA], alA[iA], lamG[iG], gsG[iG]
    lo, hi = max(lamA[0], lamG[0]), min(lamA[-1], lamG[-1])            # overlap of the two curves
    grid = np.arange(np.ceil(lo), np.floor(hi) + 0.25, 0.5)             # whole-nanometer ends, 0.5 nm steps
    alpha, gstar = np.interp(grid, lamA, alA), np.interp(grid, lamG, gsG)
    rows = [f"{lam:.2f},{a:.5f},{g:.5f}" for lam, a, g in zip(grid, alpha, gstar)]
    return "\n".join(CSV_HEADER + rows) + "\n"


def gain_ase_json(raw_dir=RAW_DIR):
    """Fig. 8: measured gain and ASE and the source's model curves per pump power (text of the JSON file)."""
    with open(os.path.join(raw_dir, "fig8_raw.json"), encoding="utf-8") as f:
        raw = json.load(f)
    r3 = lambda v: [round(float(x), 3) for x in v]
    data = {}
    for lab, d in raw.items():
        m = np.array(d["marks"])
        mg, ma = m[m[:, 0] < X_SPLIT_FIG8], m[m[:, 0] > X_SPLIT_FIG8]
        mg, ma = mg[np.argsort(mg[:, 0])], ma[np.argsort(ma[:, 0])]
        curves = [np.array(cv) for cv in d["curves"]]
        cg = [cv for cv in curves if cv[:, 0].max() < X_SPLIT_FIG8][0]
        ca = [cv for cv in curves if cv[:, 0].max() > X_SPLIT_FIG8][0]
        for cv in (cg, ca):
            if not np.all(np.diff(cv[:, 0]) > 0):
                raise ValueError(f"{lab} mW: model-curve vertices not in increasing wavelength order")
        data[lab] = {"pump_mW": float(lab),
                     "lambda_nm": r3(FIG8_GAIN_LAM(mg[:, 0])), "gain_dB_measured": r3(FIG8_GAIN_DB(mg[:, 1])),
                     "ase_lambda_nm": r3(FIG8_ASE_LAM(ma[:, 0])), "ase_dBm_per_0p1nm_measured": r3(FIG8_ASE_DBM(ma[:, 1])),
                     "theory_lambda_nm": r3(FIG8_GAIN_LAM(cg[:, 0])), "gain_dB_source_model": r3(FIG8_GAIN_DB(cg[:, 1])),
                     "theory_ase_lambda_nm": r3(FIG8_ASE_LAM(ca[:, 0])),
                     "ase_dBm_per_0p1nm_source_model": r3(FIG8_ASE_DBM(ca[:, 1]))}
    return json.dumps(dict(JSON_META, data=data), indent=1)


def calibrate(raw_dir=RAW_DIR, out_dir=DATA_DIR, check=False):
    """Stage 2: write (or, with check=True, compare) the two data files. Returns True if all files are identical."""
    files = {CSV_NAME: coefficients_csv(raw_dir), JSON_NAME: gain_ase_json(raw_dir)}
    same = True
    for name, text in files.items():
        blob = text.replace("\n", "\r\n").encode("utf-8")              # CRLF, as in the distributed data files
        path = os.path.join(out_dir, name)
        if check:
            old = open(path, "rb").read() if os.path.exists(path) else None
            ok = old == blob
            same &= ok
            print(f"{name}: {'identical to' if ok else 'DIFFERS from'} {path} ({len(blob)} bytes)")
        else:
            os.makedirs(out_dir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(blob)
            print("written", path)
    return same


def main(argv):
    if len(argv) >= 2 and (argv[1] == "extract" or argv[1].lower().endswith(".pdf")):
        args = argv[2:] if argv[1] == "extract" else argv[1:]       # 'digitize_perin2019.py <pdf>' still works
        if not args:
            sys.exit("usage: python -I digitize_perin2019.py extract <arXiv PDF> [raw_dir]")
        extract(args[0], args[1] if len(args) > 1 else RAW_DIR)
    elif len(argv) >= 2 and argv[1] == "calibrate":
        opts = argv[2:]
        get = lambda flag, default: opts[opts.index(flag) + 1] if flag in opts else default
        ok = calibrate(get("--raw", RAW_DIR), get("--out", DATA_DIR), check="--check" in opts)
        if "--check" in opts and not ok:
            sys.exit(1)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
