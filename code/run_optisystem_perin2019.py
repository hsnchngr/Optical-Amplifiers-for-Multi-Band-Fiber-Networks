"""Independent-simulator source (layer L2, source iii) of the end-to-end case study (Section 5.7).

The single-stage, 980-nm forward-pumped C-band EDFA of ref. [124] (Perin et al., J. Lightwave Technol. 37, 2076,
2019) is entered into OptiSystem with the SAME parameter card and the SAME operating points as the open model
(e2e_perin.py). The statistics against the open model and the measurement are computed afterwards by
e2e_simulator_compare.py.

REQUIREMENTS
  * Windows, OptiSystem 23 (installation used: 23.1.0.3744) with a VALID licence, and pywin32 + NumPy.
  * Run with a Python that has pywin32 and NumPy:   python run_optisystem_perin2019.py ...
    (for example a separate Python or conda environment; the other scripts of this folder do not need pywin32. This
    script imports only NumPy, pywin32, cards.py and the standard library, and works with NumPy 1.26).
  * OptiSystem must be closed. The script starts its own instance through COM ("OptiSystem.Application"); the
    OptiSystem window appears on the desktop while it runs. The script refuses to start if any OptiSystem process
    exists that it did not start, runs one instance at a time, and only ever terminates processes it started itself.
    It does not change OptiSystem preferences, global settings or licences.

STATUS (18 November 2026): run completed with OptiSystem 23.1.0.3744 (C1, C2, C3 passed; run sets V, N, C, W and the
numerical check Nconv); results in ../notes/optisystem_perin2019_results.md and ../figures/E2E_simulator_results.json.
  * A first start of OptiSystem through COM that day stopped with "License unavailable. Error #SH203: License not
    available [41] License expired" / "Could not obtain license. Terminating application."; the next start, 30 min
    later, obtained the licence. The watchdog below acknowledges such licence messages of its own instance only and
    the script then exits with code 3.
  * Parameter names are those the components report (checked at run time; the script stops with the list of
    available names if one is missing, no silent defaults). Several differ from the 23.1 Component Library, e.g.
    'Cross-section/Giles parameters file name', 'Include ion-ion interaction effects', 'Include Temperature
    Effects', 'Generate homogeneous cr'; the library's 'Calculation algorithm' (Saleh/Jopson/Giles) is not offered by
    this version of the steady-state 'Er Doped Fiber'. 'python run_optisystem_perin2019.py --explore' lists them.
  * Noise-bin powers: GetNoiseBinOpticalSignalPolarizationPower does NOT return the bin power. Its value is the PSD
    times the frequency resolution of the simulation window, S x 1/T_window (T_window = 102.4 ns in a new project,
    so S x 9.77 MHz); the first run read it and gave ASE 37 dB too low. The per-bin ASE is therefore read from the
    EDF graphs 'Forward/Backward ASE power' with one graph point per noise bin (method of the Yb recipe), and every
    run checks that the bins add up to the graphs 'Total forward/backward ASE power' (deviation ~1e-15).

METHOD (route A of ../notes/optisystem_input_perin2019.md)
  * 'Er Doped Fiber' reads Giles absorption and gain coefficients in dB/m from its cross-section file (parameter
    'Cross-section/Giles parameters file name'; the component recognizes the Giles format from the magnitude of the
    numbers, Component Library p. 1459) and the saturation parameter zeta = pi b^2 N_t / tau directly ('Radii values'
    = 'Saturation parameter', 'Saturation parameter' in 1/(s.m)).
  * Overlap factor read from a file with Gamma = 1 at all wavelengths, so that no core radius is needed (the source
    does not state one). For Giles input the overlap enters only the conversion sigma = alpha/(Gamma N_t) and back
    (Gamma sigma N_t = alpha), so the card is reproduced for any Gamma; C1 and C3 test this.
  * The coefficient file holds the card table (1470-1569 nm), the pump line (975-985 nm: alpha = 4.1692 dB/m,
    g* = 0) and guard rows of 1e-6 dB/m outside these ranges, so that noise bins outside the card range (if the
    simulator creates any) neither absorb nor emit, as in the open model (which has no ASE outside 1470-1569 nm).
  * Layout (ideal components; reference planes = EDF ends; no isolators, no reflections):
        n x 'Pump Laser' (parameterized CW signals: channels or probes) -> 'Ideal Mux' (0 dB)
          -> 'Pump Coupler Co-Propagating' Signal Input (0 dB)  <- 'Pump Laser' 980 nm, Pump Input (0 dB)
          -> 'Er Doped Fiber' Input 1 (z = 0);  'Optical Null' -> Input 2 (z = L)
        'Dual Port WDM Analyzer' on the coupler output (= EDF input) and on EDF Output 2 (z = L), via port monitors.
    A 'Pump Laser' emits only parameterized signals, and the EDF model treats pumps and signals alike (only the
    wavelength differs, Component Library p. 1457), so channels and probes are pump-laser components.
  * Run sets (pump power at the EDF input):
      C1  pump off: 980-nm probe of 1e-6 mW and 140 probes of -60 dBm (40 channel wavelengths and 1470...1569 nm
          in 1-nm steps); acceptance: -10 log10(T)/L equals alpha of the card within 1 % at every probe and
          4.169 dB/m at 980 nm. -60 dBm because 140 simultaneous probes of -40 dBm invert the fiber enough to bias
          alpha by up to 2.5 % (open model; at -60 dBm the bias is <= 0.03 %). Repeated with L = 1 m ("C1-short")
          to exclude a power floor at the most strongly absorbed probes (8 m: up to 62 dB of absorption).
      V   nominal card, 40 x -13 dBm, ASE suppressed (a single 1-GHz noise bin), six pumps -> C3: Equation (5)
      N   nominal card, 40 x -13 dBm, six pumps 18.1 ... 114.9 mW
      C   calibrated card: alpha, g*, alpha_p x 1.0664 and pump x 10^(-1.501/10) (12.811 ... 81.324 mW)
      W   nominal card, one channel at 1550 nm, -40 dBm, six pumps (ASE self-saturation)
      Nconv (only with --convergence) N at 18.1 and 114.9 mW with 1600 steps, 1e-8 and 25-GHz noise bins
  * Numerics: 'Longitudinal steps' 400, 'Relative error' 1e-6, 'Max. number of iterations' 1000, forward and
    backward ASE in noise bins over 1470.07-1568.93 nm (EDF Noise tab: centre 197.50638 THz, bandwidth 12.85 THz,
    spacing 50 GHz; OptiSystem creates 258 bins of equal width in wavelength, 0.384 nm = 46.8-53.3 GHz).
    'Include inhomogeneous broadening', ESA, ion-ion effects, temperature effects, Rayleigh, Raman, Brillouin, SPM
    and dispersion off; background loss 0 dB/km; 'Overlap factor' = 'Power independent'.
  * Exports per run: per-bin ASE at z = L (forward) and z = 0 (backward) from the EDF graphs (see STATUS); bin edges
    from the noise bins of EDF Output 2 / Output 1; channel powers from the parameterized signals at the coupler
    output (EDF input) and at Output 2. ASE in 0.1 nm at a channel = bin power / bin width (two-polarization PSD),
    interpolated log-linearly in wavelength between bin centres, times c * 0.1 nm / lambda^2 (interpolated, not
    integrated over 0.1 nm; the same rule as e2e_perin). NF = (1 + S/(h nu))/G (signal-spontaneous beat plus shot
    noise, internal planes; the open-model definition) and the Dual Port WDM Analyzer value ('Input noise for NF
    calculation' = 'EDFA with shot-noise limited input': NF = 1/G + (N_out - G N_in)/(h nu G BW), BW = 0.1 nm, the
    same definition for N_in = 0; 'Noise interpolation' = 'Off', i.e. the PSD of the bin that contains the channel).

OUTPUTS (../notes/optisystem_perin2019/runs/)
  inputs/                       files loaded by OptiSystem (coefficients nominal and calibrated, overlap) + plan.json
  C1_absorption.csv             probe table and the C1 result
  <set>_P<reported pump>mW.csv  per channel: wavelength, input/output power, gain, ASE in 0.1 nm, NF (two definitions)
  <set>_P<reported pump>mW.npz  noise bins (forward at z = L, backward at z = 0), all parameterized signals, the
                                analyzer curves and the raw port noise-bin values (diagnostic)
  run_log.json (+ timestamped copy)  OptiSystem version, every parameter as requested and as read back, all
                                component parameters, layout parameters (read only), per-run records with the ASE
                                checks, acceptance checks, dialogs seen, errors
  perin2019_C1_absorption.osd, perin2019_benchmark.osd   OptiSystem projects
  explore_parameters.json       (--explore) parameters and ports reported by every component used

USAGE (from this folder, with the Python that has pywin32)
  python run_optisystem_perin2019.py --prepare      no OptiSystem: write the input files and the plan only
  python run_optisystem_perin2019.py --explore      list the parameters/ports of every component; check the names
  python run_optisystem_perin2019.py --check        acceptance checks only: C1, C1-short and V (C3)
  python run_optisystem_perin2019.py                C1; only if C1 passes: V, N, C, W; saves the projects
  options: --sets V,N,C,W  --convergence  --no-analyzer  --no-save  --overlap calculate --core-radius 1.27
           --radii fiber  --steps 400  --rel-error 1e-6  --max-iter 1000  --bin-ghz 50  --skip-c1 (needs a passed C1)
  The production run of 18 Nov 2026 was:  python run_optisystem_perin2019.py --convergence   (about 1 min)
  Then (normal Python):  python e2e_simulator_compare.py   ->  ../figures/E2E_simulator_results.json
Exit codes: 0 done, 1 error, 2 OptiSystem already running, 3 licence unavailable, 4 parameter mismatch, 5 C1 failed.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import cards  # noqa: E402  (json/os only; NumPy for the CSV reader)

# ----------------------------------------------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------------------------------------------
ROOT = os.path.normpath(os.path.join(HERE, ".."))
NOTES = os.path.join(ROOT, "notes", "optisystem_perin2019")
RUNS = os.path.join(NOTES, "runs")
INPUTS = os.path.join(RUNS, "inputs")
OPS_CSV = os.path.join(NOTES, "operating_points.csv")
CARD_NOM = os.path.join(cards.CARD_DIR, "edfa_perin2019.json")
CARD_CAL = os.path.join(cards.CARD_DIR, "edfa_perin2019_calibrated.json")
OS_DIR = r"C:\Program Files\Optiwave Software\OptiSystem 23"
LOG_PATH = os.path.join(RUNS, "run_log.json")

C0 = 299792458.0
H_PLANCK = 6.62607015e-34
DB_PER_NEPER = 10.0 / np.log(10.0)
LAYOUT = "Layout 1"
LIB = {"Amplifiers": "{255EDC8F-37E4-11D4-93EC-0050DAB7C5D6}",      # CLSIDs: VBScripting Ref. Guide, p. 14
       "Transmitters": "{6DA31CEE-058F-11D4-93BD-0050DAB7C5D6}",
       "Passives": "{F11D0C07-3C7D-11D4-93F0-0050DAB7C5D6}",
       "Tools": "{E138711F-3E0D-11D4-93F3-0050DAB7C5D6}",
       "WDM": "{1747A24E-52A0-11D4-9403-0050DAB7C5D6}",
       "Visualizers": "{F11D0C25-3C7D-11D4-93F0-0050DAB7C5D6}"}
OS_IMAGES = ("optisystemx64.exe", "osysserver.exe")

SCAN_NM = np.arange(1470.0, 1569.0 + 1e-9, 1.0)     # C1 scan probes (nodes of the card table)
P_PROBE_MW = 1e-6                                    # -60 dBm per C1 probe
P_CH_MW = 10 ** (-13.0 / 10)                         # -13 dBm per channel
W_LAMBDA_NM, W_P_MW = 1550.0, 1e-4                   # run set W: one channel at 1550 nm, -40 dBm
ASE_BAND_NM = (1470.0, 1569.0)                       # card table range = open-model ASE band
GUARD_DB_PER_M = 1e-6                                # "no absorption, no emission" outside the card table
TOL_C1 = 0.01                                        # C1: 1 % (input sheet)
TOL_C3_DB = 0.01                                     # C3: 0.01 dB (input sheet)
CSV_COLUMNS = ["channel", "lambda_nm", "P_in_dBm", "P_out_dBm", "G_dB", "ASE_0p1nm_dBm", "NF_dB",
               "G_dpwdma_dB", "NF_dpwdma_dB"]
ASE_RULE = ("forward-ASE power of each noise bin at z = L (EDF graph 'Forward ASE power', one point per bin) divided "
            "by the bin width (two-polarization PSD), interpolated log-linearly in wavelength between bin centres to "
            "the channel wavelength, times c*0.1nm/lambda^2 (interpolated, not integrated over 0.1 nm)")
NF_RULE = ("NF_dB = (1 + S/(h nu))/G with S the two-polarization forward-ASE PSD at the channel (signal-spontaneous "
           "beat plus shot noise, internal planes); G_dpwdma_dB and NF_dpwdma_dB = Dual Port WDM Analyzer, 'EDFA with "
           "shot-noise limited input': NF = 1/G + (N_out - G N_in)/(h nu G BW), BW = 0.1 nm, 'Noise interpolation' Off "
           "(PSD of the noise bin containing the channel)")


class ForeignInstance(RuntimeError):
    pass


class LicenseUnavailable(RuntimeError):
    pass


class ParamError(RuntimeError):
    pass


class ExtractError(RuntimeError):
    pass


def stamp():
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def dbm(p_w):
    p = np.asarray(p_w, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(p * 1e3)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ----------------------------------------------------------------------------------------------------------------
# Card, operating points, input files and plan (no OptiSystem needed)
# ----------------------------------------------------------------------------------------------------------------
def read_operating_points(path=OPS_CSV):
    """Pump rows (run, reported, nominal and calibrated pump at the EDF input [mW]) and the 40 channels."""
    pumps, chans, section = [], [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("run,"):
                section = "pumps"
                continue
            if s.startswith("channel,"):
                section = "channels"
                continue
            p = s.split(",")
            if section == "pumps":
                pumps.append(dict(run=p[0], reported_mW=float(p[1]), nominal_mW=float(p[2]),
                                  calibrated_mW=float(p[3])))
            elif section == "channels":
                chans.append(dict(channel=int(p[0]), lambda_nm=float(p[1]), power_dBm=float(p[2])))
    if len(pumps) != 6 or len(chans) != 40:
        raise ValueError(f"{path}: expected 6 pump rows and 40 channels, found {len(pumps)} and {len(chans)}")
    return pumps, chans


def load_card_data():
    """Nominal and calibrated card values and the Giles table (lambda [nm], alpha [1/m], g* [1/m])."""
    nom = cards.values(cards.load(CARD_NOM))
    cal = cards.values(cards.load(CARD_CAL))
    lam, a, g = cards.perin2019_coefficients()
    zeta = np.pi * nom["r_er"] ** 2 * nom["n_t"] / nom["tau"]
    if abs(zeta / nom["zeta"] - 1) > 1e-3:
        raise ValueError("card EDFA-PERIN2019: zeta inconsistent with r_er, n_t, tau")
    return nom, cal, np.asarray(lam, float), np.asarray(a, float), np.asarray(g, float)


def coefficient_rows(lam, a_per_m, g_per_m, alpha_p_per_m, g_p_per_m=0.0, scale=1.0):
    """Rows (lambda nm, alpha dB/m, g* dB/m) of the Giles-format cross-section file (ascending wavelength)."""
    ap, gp, gd = alpha_p_per_m * DB_PER_NEPER * scale, g_p_per_m * DB_PER_NEPER * scale, GUARD_DB_PER_M
    rows = [(975.0, ap, gp), (980.0, ap, gp), (985.0, ap, gp), (990.0, gd, gd), (lam[0] - 1.0, gd, gd)]
    rows += [(float(l), float(x) * DB_PER_NEPER * scale, float(y) * DB_PER_NEPER * scale)
             for l, x, y in zip(lam, a_per_m, g_per_m)]
    rows += [(lam[-1] + 1.0, gd, gd), (1700.0, gd, gd)]
    return rows


def write_columns(path, rows):
    """Whitespace-separated columns in the format of the data files distributed with OptiSystem ('  %.7e')."""
    with open(path, "w", encoding="ascii", newline="\r\n") as f:
        for r in rows:
            f.write("".join("  %.7e" % float(v) for v in r) + "\n")


def noise_band(bin_ghz, band_nm=ASE_BAND_NM):
    """Noise centre frequency, bandwidth and spacing whose bins lie inside the card range."""
    f_hi, f_lo = C0 / (band_nm[0] * 1e-9), C0 / (band_nm[1] * 1e-9)
    n = int(np.floor((f_hi - f_lo) / (bin_ghz * 1e9) + 1e-9))
    bw = n * bin_ghz * 1e9
    fc = round(0.5 * (f_hi + f_lo) / 1e6) * 1e6             # 1 MHz grid
    return dict(center_THz=fc / 1e12, bandwidth_THz=bw / 1e12, spacing_GHz=float(bin_ghz), n_bins_nominal=n,
                edges_nm=[C0 / (fc + bw / 2) * 1e9, C0 / (fc - bw / 2) * 1e9])


def make_plan(args):
    nom, cal, lam, a, g = load_card_data()
    pumps, chans = read_operating_points()
    s_cal, loss_cal = float(cal["coef_scale"]), float(cal["pump_loss_dB"])
    pumps_cal = [p["reported_mW"] * 10 ** (-loss_cal / 10) for p in pumps]
    for p, pc in zip(pumps, pumps_cal):          # operating_points.csv rounds to 3 decimals
        if abs(p["calibrated_mW"] - pc) > 6e-4 or abs(p["nominal_mW"] - p["reported_mW"]) > 1e-9:
            raise ValueError("operating_points.csv does not match the calibrated card")
    files = dict(nominal=os.path.join(INPUTS, "perin2019_giles_nominal_dB_per_m.dat"),
                 calibrated=os.path.join(INPUTS, "perin2019_giles_calibrated_dB_per_m.dat"),
                 overlap=os.path.join(INPUTS, "overlap_unity.dat"))
    full = noise_band(args.bin_ghz)
    plan = dict(
        route="A (Giles coefficients in dB/m, saturation parameter; overlap fixed at 1 by file)"
        if args.overlap == "file" else
        f"A with calculated overlap (core radius {args.core_radius} um ASSUMED, not in the card)",
        radii=args.radii, overlap=args.overlap,
        card=dict(card_id="EDFA-PERIN2019", L_m=float(nom["L"]), lambda_p_nm=float(nom["lambda_p"]),
                  alpha_p_per_m=float(nom["alpha_p"]), alpha_p_dB_per_m=float(nom["alpha_p"]) * DB_PER_NEPER,
                  g_p_per_m=float(nom["g_p"]), r_er_um=float(nom["r_er"]) * 1e6, n_t_per_m3=float(nom["n_t"]),
                  tau_ms=float(nom["tau"]) * 1e3, zeta_per_m_s=float(nom["zeta"]), NA=float(nom["NA"]),
                  l_bg_dB_per_km=float(nom["l_bg"]) * DB_PER_NEPER * 1e3, table_nm=[float(lam[0]), float(lam[-1])],
                  n_table=int(lam.size)),
        calibrated=dict(card_id="EDFA-PERIN2019-CAL", coef_scale=s_cal, pump_loss_dB=loss_cal),
        channels_nm=[c["lambda_nm"] for c in chans], channel_power_mW=P_CH_MW,
        pumps_reported_mW=[p["reported_mW"] for p in pumps], pumps_nominal_mW=[p["nominal_mW"] for p in pumps],
        pumps_calibrated_mW=pumps_cal,
        W=dict(lambda_nm=W_LAMBDA_NM, power_mW=W_P_MW),
        c1=dict(scan_nm=SCAN_NM.tolist(), probe_power_mW=P_PROBE_MW, tolerance=TOL_C1,
                lengths_m=[float(nom["L"]), 1.0]),
        numerics=dict(longitudinal_steps=float(args.steps), relative_error=float(args.rel_error),
                      max_iterations=float(args.max_iter), algorithm="Giles"),
        noise_full=full,
        noise_suppressed=dict(center_THz=full["center_THz"], bandwidth_THz=0.001, spacing_GHz=1.0, n_bins_nominal=1),
        noise_convergence=noise_band(args.bin_ghz / 2),
        files=files, core_radius_um=float(args.core_radius))
    return plan, (lam, a, g)


def prepare(args):
    """Write the files that OptiSystem loads and the plan; returns the plan."""
    os.makedirs(INPUTS, exist_ok=True)
    plan, (lam, a, g) = make_plan(args)
    c = plan["card"]
    write_columns(plan["files"]["nominal"], coefficient_rows(lam, a, g, c["alpha_p_per_m"], c["g_p_per_m"]))
    write_columns(plan["files"]["calibrated"], coefficient_rows(lam, a, g, c["alpha_p_per_m"], c["g_p_per_m"],
                                                                scale=plan["calibrated"]["coef_scale"]))
    write_columns(plan["files"]["overlap"], [(x, 1.0) for x in np.arange(970.0, 1700.0 + 1e-9, 1.0)])
    # C2 (route A: automatic): the file reproduces the card table and g*/alpha exactly (rounding only)
    rows = np.loadtxt(plan["files"]["nominal"])
    sel = (rows[:, 0] >= lam[0] - 1e-9) & (rows[:, 0] <= lam[-1] + 1e-9)
    ra = rows[sel, 1] / (a * DB_PER_NEPER) - 1
    rg = rows[sel, 2] / (g * DB_PER_NEPER) - 1
    rr = (rows[sel, 2] / rows[sel, 1]) / (g / a) - 1
    plan["C2"] = dict(check="g*/alpha in the loaded file vs the card (route A: automatic)",
                      max_rel_dev_alpha=float(np.max(np.abs(ra))), max_rel_dev_gstar=float(np.max(np.abs(rg))),
                      max_rel_dev_ratio=float(np.max(np.abs(rr))),
                      passed=bool(np.max(np.abs(rr)) < TOL_C1))
    plan["file_sha256"] = {k: sha256(v) for k, v in plan["files"].items()}
    with open(os.path.join(INPUTS, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(plan), f, indent=1)
    return plan


# ----------------------------------------------------------------------------------------------------------------
# Post-processing (pure NumPy; also used by e2e_simulator_compare.py --selftest)
# ----------------------------------------------------------------------------------------------------------------
def postprocess(lam_ch_nm, pin_W, pout_W, bin_lo_Hz, bin_hi_Hz, bin_P_W):
    """Gain, ASE in 0.1 nm and NF at the channels from channel powers and forward-ASE noise bins at z = L."""
    lam_ch_nm = np.asarray(lam_ch_nm, float)
    G = np.asarray(pout_W, float) / np.asarray(pin_W, float)
    lo, hi, P = (np.asarray(x, float) for x in (bin_lo_Hz, bin_hi_Hz, bin_P_W))
    S_bins = P / (hi - lo)                                  # W/Hz, both polarizations
    lam_b = C0 / (0.5 * (lo + hi)) * 1e9
    o = np.argsort(lam_b)
    S = 10 ** np.interp(lam_ch_nm, lam_b[o], np.log10(np.maximum(S_bins[o], 1e-300)))
    ase_dBm = dbm(S * C0 * 0.1e-9 / (lam_ch_nm * 1e-9) ** 2)
    NF = (1.0 + S / (H_PLANCK * C0 / (lam_ch_nm * 1e-9))) / G
    return dict(G_dB=10 * np.log10(G), ASE_dBm=ase_dBm, NF_dB=10 * np.log10(NF), S_W_per_Hz=S)


def c1_evaluate(probe_nm, pin_W, pout_W, L_m, lam_tab, alpha_tab_per_m, pump_in_W, pump_out_W, alpha_p_per_m):
    """C1: -10 log10(T)/L against alpha of the card (dB/m) at every probe and at 980 nm."""
    a_sim = -10 * np.log10(np.asarray(pout_W, float) / np.asarray(pin_W, float)) / L_m
    a_card = np.interp(probe_nm, lam_tab, alpha_tab_per_m) * DB_PER_NEPER
    rel = a_sim / a_card - 1
    a980 = -10 * np.log10(pump_out_W / pump_in_W) / L_m
    rel980 = a980 / (alpha_p_per_m * DB_PER_NEPER) - 1
    return dict(alpha_sim_dB_per_m=a_sim, alpha_card_dB_per_m=a_card, rel_dev=rel,
                max_abs_rel_dev=float(np.nanmax(np.abs(rel))), alpha_980_dB_per_m=float(a980),
                rel_dev_980=float(rel980),
                passed=bool(np.nanmax(np.abs(rel)) <= TOL_C1 and abs(rel980) <= TOL_C1 and np.all(np.isfinite(rel))))


def write_run_csv(path, meta, lam_nm, pin_W, pout_W, pp, g_dp=None, nf_dp=None):
    n = len(lam_nm)
    g_dp = np.full(n, np.nan) if g_dp is None else np.asarray(g_dp, float)
    nf_dp = np.full(n, np.nan) if nf_dp is None else np.asarray(nf_dp, float)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write(",".join(CSV_COLUMNS) + "\n")
        for i in range(n):
            vals = [lam_nm[i], dbm(pin_W[i]), dbm(pout_W[i]), pp["G_dB"][i], pp["ASE_dBm"][i], pp["NF_dB"][i],
                    g_dp[i], nf_dp[i]]
            f.write(f"{i + 1}," + ",".join("nan" if not np.isfinite(v) else f"{float(v):.6f}" for v in vals) + "\n")


def read_run_csv(path):
    """(meta dict, column dict of arrays) of a file written by write_run_csv."""
    meta, rows, header = {}, [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                k, _, v = s[1:].partition(":")
                meta[k.strip()] = v.strip()
            elif header is None:
                header = s.split(",")
            else:
                rows.append([float(x) for x in s.split(",")])
    d = np.array(rows, float)
    return meta, {h: d[:, i] for i, h in enumerate(header)}


def run_file(set_name, reported_mW, ext="csv"):
    return os.path.join(RUNS, f"{set_name}_P{reported_mW:g}mW.{ext}")


# ----------------------------------------------------------------------------------------------------------------
# Process safety and dialogs (pywin32 imported lazily)
# ----------------------------------------------------------------------------------------------------------------
def os_processes():
    """{pid: image} of running OptiSystem processes."""
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace",
                             timeout=60).stdout
    except Exception:
        return {}
    res = {}
    for line in out.splitlines():
        parts = [p.strip().strip('"') for p in line.strip().split('","')]
        if len(parts) >= 2 and parts[0].lower() in OS_IMAGES:
            try:
                res[int(parts[1])] = parts[0]
            except ValueError:
                pass
    return res


def kill_own(pids, foreign):
    """taskkill only for PIDs this script started, after re-checking that each is still an OptiSystem process."""
    killed = []
    procs = os_processes()
    for pid in sorted(pids):
        if pid in foreign or pid not in procs:
            continue
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        killed.append(pid)
    return killed


class Watchdog(threading.Thread):
    """Watches the dialog boxes of the OptiSystem processes started by this script (never those of others).

    Every dialog is recorded (title, text, buttons). Licence-error dialogs (one button) are acknowledged, after which
    OptiSystem terminates itself ("Could not obtain license. Terminating application."). If any other dialog stays
    open longer than stuck_s, the script's own OptiSystem processes are terminated so that the run log is written."""

    def __init__(self, foreign, record, stuck_s=1800.0, startup_s=120.0):
        super().__init__(daemon=True)
        self.foreign, self.record = set(foreign), record
        self.stuck_s, self.startup_s = stuck_s, startup_s
        self.mine, self.license_error, self.killed = set(), None, []
        self._stop_evt, self._seen = threading.Event(), {}

    def stop(self):
        self._stop_evt.set()

    def run(self):
        import win32con
        import win32gui
        import win32process
        t0, last = time.time(), 0.0
        while not self._stop_evt.is_set():
            now = time.time()
            if now - last > 2.0:
                if now - t0 < self.startup_s:            # processes that appear right after the start are ours
                    self.mine |= {p for p in os_processes() if p not in self.foreign}
                last = now
            found = []

            def cb(h, _):
                try:
                    if win32gui.IsWindowVisible(h) and win32gui.GetClassName(h) == "#32770":
                        if win32process.GetWindowThreadProcessId(h)[1] in self.mine:
                            found.append(h)
                except Exception:
                    pass
                return True

            try:
                win32gui.EnumWindows(cb, None)
            except Exception:
                pass
            for h in found:
                texts, buttons = [], []

                def ccb(c, _):
                    try:
                        cls, t = win32gui.GetClassName(c), win32gui.GetWindowText(c)
                        if cls == "Static" and t:
                            texts.append(t)
                        elif cls == "Button":
                            buttons.append((c, t))
                    except Exception:
                        pass
                    return True

                try:
                    win32gui.EnumChildWindows(h, ccb, None)
                    title = win32gui.GetWindowText(h)
                except Exception:
                    continue
                if not texts and not buttons:               # splash or pane without content: ignored
                    continue
                if any("CIDF Scheduler" in t or "Calculating" in t for t in texts):
                    continue                                # calculation progress window: ignored

                key = (h, tuple(texts))
                if key not in self._seen:
                    self._seen[key] = now
                    self.record.append(dict(time=stamp(), title=title, text=texts, buttons=[b[1] for b in buttons]))
                    print(f"[OptiSystem dialog] {title}: {' | '.join(texts)}", flush=True)
                if not (texts and buttons):                 # only message boxes are acted upon
                    continue
                if any("licen" in t.lower() for t in texts):
                    msg = " | ".join(t.replace("\n", " | ") for t in texts)
                    self.license_error = msg if self.license_error is None else (
                        self.license_error if msg in self.license_error else self.license_error + " || " + msg)
                    if len(buttons) == 1:                  # acknowledge (OK) so that OptiSystem can terminate
                        try:
                            win32gui.SendMessage(buttons[0][0], win32con.BM_CLICK, 0, 0)
                        except Exception:
                            pass
                elif now - self._seen[key] > self.stuck_s and not self.killed:
                    self.killed = kill_own(self.mine, self.foreign)
                    self.record.append(dict(time=stamp(), action=f"dialog open > {self.stuck_s:.0f} s; "
                                                                 f"terminated own OptiSystem processes {self.killed}"))
            self._stop_evt.wait(0.5)


class Session:
    """One OptiSystem instance started by this script, with a new project and its first layout/canvas."""

    def __init__(self, log, stuck_s=1800.0):
        self.log, self.stuck_s = log, stuck_s
        self.app = self.doc = self.cv = self.wd = None

    def __enter__(self):
        from win32com.client import Dispatch
        foreign = os_processes()
        if foreign:
            raise ForeignInstance("OptiSystem is already running and was not started by this script "
                                  f"({foreign}); nothing was started or terminated")
        self.rec = dict(started=stamp())
        self.log.setdefault("sessions", []).append(self.rec)
        self.wd = Watchdog(set(foreign), self.log.setdefault("dialogs", []), self.stuck_s)
        self.wd.start()
        try:
            self.app = Dispatch("OptiSystem.Application")
            self._retry(lambda: self.app.OpenProject())
            self.doc = self._retry(lambda: self.app.GetActiveDocument())
            lay = self._retry(lambda: self.doc.GetLayoutMgr().GetCurrentLayout())
            lay.Name = LAYOUT
            self.cv = lay.GetCurrentCanvas()
            if self.wd.license_error:
                raise LicenseUnavailable(self.wd.license_error)
        except BaseException as e:
            self._shutdown()
            if self.wd.license_error and not isinstance(e, LicenseUnavailable):
                raise LicenseUnavailable(self.wd.license_error) from e
            raise
        self.rec["pids"] = sorted(self.wd.mine)
        return self

    def _retry(self, fn, n=120, wait=0.5):
        last = None
        for _ in range(n):
            if self.wd.license_error:
                time.sleep(3.0)                                    # let the second licence message arrive
                raise LicenseUnavailable(self.wd.license_error)
            try:
                return fn()
            except AttributeError as e:                            # COM server not ready yet (recipe)
                last = e
                time.sleep(wait)
        raise last

    def save(self, path):
        self.app.Save(os.path.abspath(path))
        self.rec.setdefault("saved", []).append(os.path.relpath(path, ROOT))

    def _shutdown(self):
        try:
            if self.app is not None:
                self.app.Quit()
        except Exception as e:
            self.rec["quit_error"] = str(e)[:200]
        t0 = time.time()
        while time.time() - t0 < 60 and (self.wd.mine & set(os_processes())):
            time.sleep(1.0)
        left = self.wd.mine & set(os_processes())
        if left:
            self.rec["terminated_own_pids"] = kill_own(left, self.wd.foreign)
        self.rec["own_pids"] = sorted(self.wd.mine)
        self.rec["licence_error"] = self.wd.license_error
        self.wd.stop()
        self.rec["ended"] = stamp()

    def __exit__(self, *exc):
        self._shutdown()
        return False


# ----------------------------------------------------------------------------------------------------------------
# Parameters: names from the OptiSystem 23.1 Component Library, verified against the component at run time
# ----------------------------------------------------------------------------------------------------------------
_NAMES = {}


def param_names(comp, key):
    if key not in _NAMES:
        names = []
        for i in range(int(comp.GetParametersCount())):
            try:
                names.append(str(comp.GetParameterObjectByIndex(i).Name))
            except Exception:
                pass
        _NAMES[key] = names
    return _NAMES[key]


def _norm_unit(u):
    return str(u).strip().lower().replace("\u00b5", "u").replace("\u03bc", "u").replace(" ", "").replace("^", "")


def _same(req, got):
    if isinstance(req, bool):
        return str(got).strip().lower() in (("true", "1", "-1") if req else ("false", "0"))
    if isinstance(req, (int, float)):
        try:
            g = float(got)
        except (TypeError, ValueError):
            return False
        return abs(g - req) <= 1e-9 * max(1.0, abs(req))
    return str(got).strip().lower() == str(req).strip().lower()


def setp(comp, key, aliases, value, log, units=None, required=True):
    """Set a parameter under the first alias the component reports; verify value and unit by reading back.

    key: component type (parameter-name cache); units: accepted unit strings, the first is requested."""
    aliases = [aliases] if isinstance(aliases, str) else list(aliases)
    names = param_names(comp, key)
    name = next((a for a in aliases if a in names), None)
    cname = str(comp.Name)
    if name is None:
        msg = f"{cname}: none of {aliases} offered by the component"
        if required:
            raise ParamError(msg + f"; available: {names}")
        log.setdefault("parameters_not_offered", []).append(msg)
        return None
    try:                                        # a scripted default (mode 3) would override the value
        mode = int(comp.GetParameterMode(name))
        if mode != 0:
            comp.SetParameterMode(name, 0)
            log.setdefault("parameter_modes_changed", []).append(f"{cname}.{name}: mode {mode} -> 0 (normal)")
    except Exception:
        pass
    if units:
        for u in units:
            if _norm_unit(comp.GetParameterUnit(name)) in [_norm_unit(x) for x in units]:
                break
            try:
                comp.SetParameterUnit(name, u)
            except Exception:
                pass
    comp.SetParameterValue(name, value)
    got, got_u = comp.GetParameterValue(name), str(comp.GetParameterUnit(name)).strip()
    log.setdefault("parameters_set", {}).setdefault(cname, {})[name] = dict(
        requested=value if not isinstance(value, float) else float(value), read_back=str(got), unit=got_u)
    unit_ok = (not units) or _norm_unit(got_u) in [_norm_unit(x) for x in units]
    if isinstance(value, str) and not _same(value, got):
        try:                                    # some choice parameters may read back as an index
            float(got)
            log.setdefault("warnings", []).append(f"{cname}.{name}: choice '{value}' read back as {got}")
            return name
        except (TypeError, ValueError):
            pass
    if not (_same(value, got) and unit_ok):
        raise ParamError(f"{cname}.{name}: requested {value} {units[0] if units else ''}, read back {got} {got_u}")
    return name


LAYOUT_PARAMETERS = ("Simulation window", "Bit rate", "Time window", "Sample rate", "Sequence length",
                     "Samples per bit", "Number of samples", "Symbol rate", "Iterations", "Initial delay",
                     "Parameterized", "Convert noise bins")


def layout_parameters(doc):
    """Global (layout) parameters, read only; this script never changes them."""
    out = {}
    for nm in LAYOUT_PARAMETERS:
        for it in (0, 1):
            try:
                v = doc.GetLayoutParameterValue(LAYOUT, it, nm)
                try:
                    u = doc.GetLayoutParameterUnit(LAYOUT, it, nm)
                except Exception:
                    u = ""
                out[nm] = f"{v} {u}".strip()
                break
            except Exception as e:
                out[nm] = f"(not readable: {str(e)[:60]})"
    return out


def all_parameters(comp):
    out = {}
    for i in range(int(comp.GetParametersCount())):
        try:
            nm = str(comp.GetParameterObjectByIndex(i).Name)
            out[nm] = f"{comp.GetParameterValue(nm)} {comp.GetParameterUnit(nm)}".strip()
        except Exception as e:
            out[f"#{i}"] = f"(not readable: {str(e)[:60]})"
    return out


def edf_spec(plan, args, coef_file, noise, numerics=None):
    """(aliases, value, units, required) for the 'Er Doped Fiber' (Component Library 23.1, pp. 1469-1477)."""
    c, num = plan["card"], dict(plan["numerics"], **(numerics or {}))
    S = [(["Length"], c["L_m"], ("m",), True)]
    if args.radii == "saturation":
        S += [(["Radii values"], "Saturation parameter", None, True),
              (["Saturation parameter"], c["zeta_per_m_s"], None, True),
              (["Er doping radius"], c["r_er_um"], ("um",), False)]      # consistent with zeta; may be inactive
    else:
        S += [(["Radii values"], "Fiber specification", None, True),
              (["Er doping radius"], c["r_er_um"], ("um",), True)]
    # names as reported by the component (OptiSystem 23.1.0.3744); the Component Library spelling is the 2nd alias
    S += [(["Er ion density"], c["n_t_per_m3"], ("m^-3", "m-3"), True),
          (["Numerical aperture"], c["NA"], None, True),
          (["Background loss data type"], "Constant", None, True),
          (["Background loss"], c["l_bg_dB_per_km"], ("dB/km", "dB/Km"), True),
          (["Er metastable lifetime"], c["tau_ms"], ("ms",), True),
          (["Include ion-ion interaction effects", "Include ion-ion interaction"], False, None, True),
          (["OptiAmplifier format"], False, None, True),
          (["File frequency unit"], "nm", None, True),
          (["Cross-section/Giles parameters file name", "Cross-section file name"], os.path.abspath(coef_file), None,
           True),
          (["Include Temperature Effects", "Include temperature effects"], False, None, True),
          (["Include ESA Effect", "Include ESA effect"], False, None, True),
          (["Include inhomogeneous broadening"], False, None, True),
          (["Double-clad fiber"], False, None, False),
          (["Include Rayleigh scattering", "Include Rayleigh backscattering"], False, None, False),
          (["Include Raman scattering"], False, None, False),
          (["Include dispersion", "Include Dispersion"], False, None, False),
          (["Include SPM"], False, None, False),
          (["Include Brillouin scattering"], False, None, False),
          (["Calculation algorithm"], num["algorithm"], None, False),     # listed in the library, not offered
          (["Relative error"], num["relative_error"], None, True),
          (["Max. number of iterations"], num["max_iterations"], None, True),
          (["Iterate for convergence"], True, None, False),
          (["Longitudinal steps"], num["longitudinal_steps"], None, True),
          (["Overlap factor"], "Power independent", None, True),
          (["Generate homogeneous cr", "Generate homogeneous cr."], False, None, False),
          (["Discretize sampled signal"], False, None, False),
          (["Calculate graphs"], True, None, True),          # per-bin ASE is read from the EDF graphs
          (["Number of distance steps"], 2.0, None, True),   # z = 0 and z = L
          (["Number of wavelength steps"], 2.0, None, True), # set to the number of noise bins after a calc
          (["Linear scale"], True, None, True),
          (["Enabled"], True, None, False),
          (["Enable reflections"], False, None, False),
          (["Noise center frequency"], noise["center_THz"], ("THz",), True),
          (["Noise bandwidth"], noise["bandwidth_THz"], ("THz",), True),
          (["Noise bins spacing"], noise["spacing_GHz"], ("GHz",), True),
          (["Convert noise bins"], False, None, False)]
    if args.overlap == "file":
        S += [(["Overlap factor data"], "From file", None, True),
              (["Overlap factor file name"], os.path.abspath(plan["files"]["overlap"]), None, True)]
    else:
        S += [(["Overlap factor data"], "Calculate", None, True),
              (["Core radius"], float(args.core_radius), ("um",), True)]
    return S


def configure_edf(edf, plan, args, log, coef_file, noise, numerics=None):
    for aliases, value, units, required in edf_spec(plan, args, coef_file, noise, numerics):
        setp(edf, "Er Doped Fiber", aliases, value, log, units, required)


# ----------------------------------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------------------------------
class Parts:
    pass


def build(cv, n_sources, log, analyzer=True):
    """Sources -> Ideal Mux -> co-propagating pump coupler (+ 980-nm pump) -> EDF (Input 2 <- Optical Null)."""
    def mk(typ, lib, x, y):
        return cv.CreateComponent(typ, LIB[lib], int(x), int(y), 34, 34, 0)

    P, ncol = Parts(), 10
    P.sources = []
    for i in range(n_sources):
        c = mk("Pump Laser", "Transmitters", 20 + 50 * (i % ncol), 20 + 50 * (i // ncol))
        c.Name = f"Src{i + 1:03d}"
        P.sources.append(c)
    y0 = 20 + 50 * ((n_sources - 1) // ncol + 1) + 60
    P.mux = mk("Ideal Mux", "WDM", 160, y0)
    P.mux.Name = "Mux"
    setp(P.mux, "Ideal Mux", ["Number of input ports"], float(n_sources), log)
    if int(P.mux.GetNumberOfInputPorts()) < n_sources:
        raise ParamError(f"Ideal Mux offers {P.mux.GetNumberOfInputPorts()} input ports, {n_sources} needed")
    setp(P.mux, "Ideal Mux", ["Loss"], 0.0, log, ("dB",))
    P.pump = mk("Pump Laser", "Transmitters", 260, y0 - 70)
    P.pump.Name = "Pump"
    P.cpl = mk("Pump Coupler Co-Propagating", "Passives", 280, y0)
    P.cpl.Name = "WDM"
    P.edf = mk("Er Doped Fiber", "Amplifiers", 420, y0)
    P.edf.Name = "EDF"
    P.null = mk("Optical Null", "Tools", 560, y0 + 70)
    P.null.Name = "Null"
    for i, c in enumerate(P.sources):
        c.GetOutputPort(1).Connect(P.mux.GetInputPort(i + 1))
    P.mux.GetOutputPort(1).Connect(P.cpl.GetInputPort(1))          # Signal Input
    P.pump.GetOutputPort(1).Connect(P.cpl.GetInputPort(2))         # Pump Input
    P.cpl.GetOutputPort(1).Connect(P.edf.GetInputPort(1))          # z = 0
    P.null.GetOutputPort(1).Connect(P.edf.GetInputPort(2))         # z = L: nothing enters
    for comp in (P.mux, P.pump, P.cpl, P.edf):
        for k in range(1, int(comp.GetNumberOfOutputPorts()) + 1):
            if not comp.GetOutputPort(k).HasMonitor():
                comp.GetOutputPort(k).CreateMonitor()
    setp(P.cpl, "Pump Coupler Co-Propagating", ["Signal attenuation"], 0.0, log, ("dB",))
    setp(P.cpl, "Pump Coupler Co-Propagating", ["Pump attenuation"], 0.0, log, ("dB",))
    P.dpw = None
    if analyzer:
        try:
            P.dpw = mk("Dual Port WDM Analyzer", "Visualizers", 560, y0 - 90)
            P.dpw.Name = "DPWDMA"
            P.cpl.GetOutputPort(1).ConnectVisualizer(P.dpw.GetInputPort(1))
            P.edf.GetOutputPort(2).ConnectVisualizer(P.dpw.GetInputPort(2))
            k = "Dual Port WDM Analyzer"
            setp(P.dpw, k, ["Lower frequency limit"], 191.4, log, ("THz",))       # 1566.4 nm
            setp(P.dpw, k, ["Upper frequency limit"], 196.2, log, ("THz",))       # 1528.0 nm
            setp(P.dpw, k, ["Resolution bandwidth"], 0.1, log, ("nm",))
            setp(P.dpw, k, ["Noise interpolation"], "Off", log)
            setp(P.dpw, k, ["Input noise for NF calculation"], "EDFA with shot-noise limited input", log)
        except Exception as e:
            log.setdefault("warnings", []).append(f"Dual Port WDM Analyzer not used: {str(e)[:300]}")
            P.dpw = None
    return P


def set_sources(P, lam_nm, p_mW, log):
    """Sources 1..len(lam_nm) on; the remaining ones disabled. Logged compactly (one row per source)."""
    scratch, rows = {}, []
    for i, c in enumerate(P.sources):
        if i < len(lam_nm):
            setp(c, "Pump Laser", ["Frequency"], float(lam_nm[i]), scratch, ("nm",))
            setp(c, "Pump Laser", ["Power"], float(p_mW[i]), scratch, ("mW",))
            setp(c, "Pump Laser", ["Enabled"], True, scratch)
        else:
            setp(c, "Pump Laser", ["Enabled"], False, scratch)
        s = scratch["parameters_set"][str(c.Name)]
        rows.append([str(c.Name), s.get("Frequency", {}).get("read_back"), s.get("Power", {}).get("read_back"),
                     s["Enabled"]["read_back"]])
    log.setdefault("sources_set", []).append(dict(time=stamp(), columns=["name", "frequency_nm", "power_mW",
                                                                         "enabled"], rows=rows))
    for k in ("warnings", "parameter_modes_changed"):
        if scratch.get(k):
            log.setdefault(k, []).extend(scratch[k])


def set_pump(P, lam_nm, p_mW, log):
    setp(P.pump, "Pump Laser", ["Frequency"], float(lam_nm), log, ("nm",))
    setp(P.pump, "Pump Laser", ["Power"], float(p_mW), log, ("mW",))
    setp(P.pump, "Pump Laser", ["Enabled"], True, log)


# ----------------------------------------------------------------------------------------------------------------
# Calculation and extraction
# ----------------------------------------------------------------------------------------------------------------
def calculate(doc):
    t = time.time()
    doc.CalculateAllSweepIterations(True, True)                    # as in the recipe
    try:
        msgs = [str(m) for m in doc.GetCalcErrorMsgs() if m]
    except Exception as e:
        msgs = [f"(GetCalcErrorMsgs not available: {str(e)[:80]})"]
    return round(time.time() - t, 2), msgs


def port_signal(comp, names):
    last = None
    for nm in names:
        try:
            s = comp.GetSignalByPortName(nm, 0)
            if s is not None:
                return nm, s
        except Exception as e:
            last = e
    raise ExtractError(f"{comp.Name}: no signal at {names} ({last})")


def read_parameterized(sig):
    """Centre frequencies [Hz] and powers [W] of all parameterized signals of a port signal."""
    n = int(sig.GetNumberOfParameterizedSignals())
    f, p = np.full(n, np.nan), np.zeros(n)
    for i in range(n):
        p[i] = float(sig.GetParameterizedSignalOpticalSignalPolarizationPower(i))
        try:
            lo, hi = sig.GetParameterizedSignalSignalBandwidth(i)
            f[i] = 0.5 * (float(lo) + float(hi))
        except Exception:
            pass
    if n and np.isnan(f).any():
        try:
            ch = np.array(sig.GetChannels(), float).ravel()
            if ch.size == n:
                f = ch
        except Exception:
            pass
    return f, p


def read_noise_bins(sig):
    n = int(sig.GetNumberOfNoiseBins())
    lo, hi, p = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        a, b = sig.GetNoiseBinSignalBandwidth(i)
        lo[i], hi[i] = float(a), float(b)
        p[i] = float(sig.GetNoiseBinOpticalSignalPolarizationPower(i))
    return lo, hi, p


def match(f_sig, targets_Hz, tol_Hz):
    """Index of the parameterized signal nearest to each target frequency (-1 if none within tol_Hz)."""
    idx = []
    for ft in targets_Hz:
        if f_sig.size == 0 or not np.isfinite(f_sig).any():
            idx.append(-1)
            continue
        j = int(np.nanargmin(np.abs(f_sig - ft)))
        idx.append(j if abs(f_sig[j] - ft) <= tol_Hz else -1)
    return np.array(idx, int)


def edf_graph_value(doc, names, index):
    for nm in names:
        try:
            g = doc.GetComponentGraph(LAYOUT, "EDF", nm)
            return nm, float(np.array(g.GetYData(1), float)[index])
        except Exception:
            continue
    return None, None


def _axis_to_nm(x):
    """Wavelength in nm from a graph axis given in nm, m, THz or Hz."""
    m = float(np.nanmedian(x))
    if m < 1e-3:
        return x * 1e9                      # m
    if m > 1e9:
        return C0 / x * 1e9                 # Hz
    if 100.0 < m < 450.0:
        return C0 / (x * 1e12) * 1e9        # THz
    return x                                # nm


def dpwdma_values(doc, lam_nm):
    """Per-channel gain and NF from the Dual Port WDM Analyzer graphs (NaN where not available); raw curves kept."""
    out = {"raw": {}}
    for key, names in (("G", ["Gain"]), ("NF", ["Noise figure", "Noise Figure", "NF", "Noise figure (dB)"])):
        out[key] = np.full(len(lam_nm), np.nan)
        for gname in names:
            try:
                g = doc.GetComponentGraph(LAYOUT, "DPWDMA", gname)
                x, y = np.array(g.GetXData(1), float), np.array(g.GetYData(1), float)
            except Exception as e:
                out[key + "_error"] = f"{gname}: {str(e)[:120]}"
                continue
            out.pop(key + "_error", None)
            out["raw"][key] = (gname, x, y)
            if x.size:
                xn = _axis_to_nm(x)
                j = [int(np.argmin(np.abs(xn - l))) for l in lam_nm]
                out[key] = np.array([y[k] if abs(xn[k] - l) < 0.1 else np.nan for k, l in zip(j, lam_nm)])
            break
    return out


def ase_graph_bins(doc):
    """Per-noise-bin ASE powers [W] at z = 0 (backward) and z = L (forward) from the EDF graphs.

    Method of the working recipe (Yb benchmark): 'Number of distance steps' = 2 and 'Number of wavelength steps' =
    number of noise bins, so that each graph point is the power of one bin; checked against the graphs 'Total
    forward/backward ASE power'. (The values returned by GetNoiseBinOpticalSignalPolarizationPower are NOT bin
    powers: they equal the PSD times 1/T_window, i.e. S x 9.77 MHz for the 102.4-ns default window; kept only as a
    diagnostic, see 'port_value_over_psd_median' in the run log.)"""
    out = {}
    for key, name, col in (("bwd", "Backward ASE power", 0), ("fwd", "Forward ASE power", -1)):
        g = doc.GetComponentGraph(LAYOUT, "EDF", name)
        nz, nw = int(g.GetSweepXDim(1)), int(g.GetSweepYDim(1))          # distance points, wavelength points
        v = np.array(g.GetSweepValues(1), float).reshape(nw, nz)
        out[key + "_W"] = v[:, col]
        # the wavelength step is returned by GetSweepXSpacing (recipe)
        out["wl_graph_nm"] = g.GetSweepYStartValue(1) + g.GetSweepXSpacing(1) * np.arange(nw)
        out[key + "_dims"] = [nz, nw]
    out["fwd_total_W"] = edf_graph_value(doc, ["Total forward ASE power"], -1)[1]
    out["bwd_total_W"] = edf_graph_value(doc, ["Total backward ASE power"], 0)[1]
    return out


def calculate_matched(sess, P, log):
    """Calculate; if the graph resolution differs from the number of noise bins, adjust it and calculate again."""
    dt, msgs = calculate(sess.doc)
    _, s_L = port_signal(P.edf, ["Output 2", "Output2"])
    nb = int(s_L.GetNumberOfNoiseBins())
    if nb > 0 and int(round(float(P.edf.GetParameterValue("Number of wavelength steps")))) != nb:
        setp(P.edf, "Er Doped Fiber", ["Number of wavelength steps"], float(nb), log)
        dt2, msgs2 = calculate(sess.doc)
        dt, msgs = dt + dt2, msgs + msgs2
    return dt, msgs


def edf_result(edf, names):
    for nm in names:
        try:
            r = edf.GetResult(nm)
            for arg in (1, 0):
                try:
                    return nm, float(r.GetValue(arg))
                except Exception:
                    continue
        except Exception:
            continue
    return None, None


def extract(sess, P, lam_nm, pump_nm=980.0, ase=True):
    """Channel powers at the EDF input and output, pump at both ends and (ase=True) per-bin ASE at both ends."""
    _, s_in = port_signal(P.cpl, ["Output"])
    _, s_L = port_signal(P.edf, ["Output 2", "Output2"])
    _, s_0 = port_signal(P.edf, ["Output 1", "Output1"])
    f_in, p_in = read_parameterized(s_in)
    f_L, p_L = read_parameterized(s_L)
    f_0, p_0 = read_parameterized(s_0)
    loL, hiL, vL = read_noise_bins(s_L)
    lo0, hi0, v0 = read_noise_bins(s_0)
    f_ch = C0 / (np.asarray(lam_nm, float) * 1e-9)
    i_in, i_out = match(f_in, f_ch, 5e9), match(f_L, f_ch, 5e9)
    j_in, j_out = match(f_in, [C0 / (pump_nm * 1e-9)], 100e9), match(f_L, [C0 / (pump_nm * 1e-9)], 100e9)
    if (i_in < 0).any() or (i_out < 0).any():
        raise ExtractError(f"channels not found by frequency: input {int((i_in < 0).sum())}, output "
                           f"{int((i_out < 0).sum())} of {len(lam_nm)}")
    oL = np.argsort(C0 / (0.5 * (loL + hiL)))                      # bins in ascending wavelength
    o0 = np.argsort(C0 / (0.5 * (lo0 + hi0)))
    x = dict(pin_W=p_in[i_in], pout_W=p_L[i_out],
             pump_in_W=float(p_in[j_in[0]]) if j_in[0] >= 0 else np.nan,
             pump_out_W=float(p_L[j_out[0]]) if j_out[0] >= 0 else np.nan,
             sig_in=(f_in, p_in), sig_L=(f_L, p_L), sig_0=(f_0, p_0),
             port_values_L=vL[oL], port_values_0=v0[o0],
             iterations=edf_result(P.edf, ["Number of iterations", "Iterations", "Number of iterations used"]))
    if not ase:
        return x
    g = ase_graph_bins(sess.doc)
    if g["fwd_W"].size != loL.size or g["bwd_W"].size != lo0.size:
        raise ExtractError(f"ASE graph points {g['fwd_W'].size}/{g['bwd_W'].size} differ from the noise bins "
                           f"{loL.size}/{lo0.size}")
    x["bins_L"] = (loL[oL], hiL[oL], g["fwd_W"])
    x["bins_0"] = (lo0[o0], hi0[o0], g["bwd_W"])
    wl_port = C0 / (0.5 * (loL[oL] + hiL[oL])) * 1e9
    x["ase_check"] = dict(
        method="EDF graphs 'Forward ASE power' (z = L) and 'Backward ASE power' (z = 0), one point per noise bin",
        forward_bin_sum_W=float(g["fwd_W"].sum()), forward_total_graph_W=g["fwd_total_W"],
        forward_rel_dev=float(g["fwd_W"].sum() / g["fwd_total_W"] - 1) if g["fwd_total_W"] else None,
        backward_bin_sum_W=float(g["bwd_W"].sum()), backward_total_graph_W=g["bwd_total_W"],
        backward_rel_dev=float(g["bwd_W"].sum() / g["bwd_total_W"] - 1) if g["bwd_total_W"] else None,
        graph_axis_max_dev_nm=float(np.max(np.abs(g["wl_graph_nm"] - wl_port))),
        graph_dims=[g["fwd_dims"], g["bwd_dims"]],
        port_value_over_psd_median=float(np.median(vL[oL] / (g["fwd_W"] / (hiL[oL] - loL[oL])))))
    return x


def bins_summary(lo, hi, P):
    if lo.size == 0:
        return dict(n=0)
    lam = C0 / (0.5 * (lo + hi)) * 1e9
    w = (hi - lo) / 1e9
    return dict(n=int(lo.size), lambda_min_nm=float(lam.min()), lambda_max_nm=float(lam.max()),
                width_min_GHz=float(w.min()), width_max_GHz=float(w.max()), total_mW=float(P.sum() * 1e3))


# ----------------------------------------------------------------------------------------------------------------
# Run sets
# ----------------------------------------------------------------------------------------------------------------
def run_c1(args, plan, log):
    """Acceptance check C1 in its own OptiSystem session (project perin2019_C1_absorption.osd)."""
    lam_tab, a_tab, _ = cards.perin2019_coefficients()
    probes = np.r_[plan["channels_nm"], plan["c1"]["scan_nm"]]
    rec = log.setdefault("C1", dict(probes=int(probes.size), probe_power_mW=P_PROBE_MW))
    res = {}
    with Session(log, args.dialog_timeout) as sess:
        P = build(sess.cv, probes.size, log, analyzer=False)
        set_sources(P, probes, [P_PROBE_MW] * probes.size, log)
        set_pump(P, plan["card"]["lambda_p_nm"], P_PROBE_MW, log)
        configure_edf(P.edf, plan, args, log, plan["files"]["nominal"], plan["noise_full"])
        for L_m in plan["c1"]["lengths_m"]:
            setp(P.edf, "Er Doped Fiber", ["Length"], float(L_m), log, ("m",))
            dt, msgs = calculate(sess.doc)
            x = extract(sess, P, probes, plan["card"]["lambda_p_nm"], ase=False)
            ev = c1_evaluate(probes, x["pin_W"], x["pout_W"], L_m, lam_tab, a_tab, x["pump_in_W"],
                             x["pump_out_W"], plan["card"]["alpha_p_per_m"])
            res[L_m] = (x, ev)
            rec[f"L_{L_m:g}m"] = dict(seconds=dt, messages=msgs, max_abs_rel_dev=ev["max_abs_rel_dev"],
                                      alpha_980_dB_per_m=ev["alpha_980_dB_per_m"], rel_dev_980=ev["rel_dev_980"],
                                      passed=ev["passed"], min_output_dBm=float(np.min(dbm(x["pout_W"]))))
            print(f"C1, L = {L_m:g} m: max |rel. dev.| {100 * ev['max_abs_rel_dev']:.3f} %, 980 nm "
                  f"{ev['alpha_980_dB_per_m']:.4f} dB/m -> {'PASS' if ev['passed'] else 'FAIL'}", flush=True)
        setp(P.edf, "Er Doped Fiber", ["Length"], plan["card"]["L_m"], log, ("m",))
        log.setdefault("all_parameters", {})["C1"] = {n: all_parameters(c) for n, c in
                                                      (("EDF", P.edf), ("Pump", P.pump), ("WDM", P.cpl),
                                                       ("Mux", P.mux), ("Src001", P.sources[0]))}
        if not args.no_save:
            sess.save(os.path.join(RUNS, "perin2019_C1_absorption.osd"))
    L8, L1 = plan["c1"]["lengths_m"]
    x8, e8 = res[L8]
    x1, e1 = res[L1]
    with open(os.path.join(RUNS, "C1_absorption.csv"), "w", encoding="utf-8") as f:
        f.write(f"# C1 (pump off): {probes.size} probes of {P_PROBE_MW:g} mW and a 980-nm probe of {P_PROBE_MW:g} mW; "
                f"route {plan['route']}\n# pass: |alpha_sim/alpha_card - 1| <= {TOL_C1:g} at every probe and at "
                f"980 nm (8 m: {e8['passed']}, 1 m: {e1['passed']})\n")
        f.write(f"# 980 nm: {e8['alpha_980_dB_per_m']:.5f} dB/m (8 m), {e1['alpha_980_dB_per_m']:.5f} dB/m (1 m), "
                f"card {plan['card']['alpha_p_dB_per_m']:.5f} dB/m\n")
        f.write("probe_nm,alpha_card_dB_per_m,alpha_sim_8m_dB_per_m,rel_dev_8m,P_out_8m_dBm,alpha_sim_1m_dB_per_m,"
                "rel_dev_1m\n")
        for k in range(probes.size):
            f.write(f"{probes[k]:.3f},{e8['alpha_card_dB_per_m'][k]:.6f},{e8['alpha_sim_dB_per_m'][k]:.6f},"
                    f"{e8['rel_dev'][k]:.3e},{dbm(x8['pout_W'][k]):.3f},{e1['alpha_sim_dB_per_m'][k]:.6f},"
                    f"{e1['rel_dev'][k]:.3e}\n")
    above = dbm(x8["pout_W"]) > -100.0
    passed = bool(e8["passed"]) or (bool(e1["passed"]) and abs(e8["rel_dev_980"]) <= TOL_C1 and
                                    bool(np.all(np.abs(e8["rel_dev"][above]) <= TOL_C1)))
    rec["passed"] = passed
    rec["rule"] = ("8 m passes at every probe and at 980 nm; or 1 m passes everywhere and the 8-m check passes at "
                   "980 nm and at every probe whose 8-m output is above -100 dBm (a failure confined to outputs "
                   "below -100 dBm indicates a power floor of the simulator, not an error in the card entry)")
    return passed


def runset_config(name, plan):
    full, supp = plan["noise_full"], plan["noise_suppressed"]
    rep = plan["pumps_reported_mW"]
    if name in ("N", "V", "W", "Nconv"):
        at_edf = plan["pumps_nominal_mW"]
        coef = "nominal"
    elif name == "C":
        at_edf = plan["pumps_calibrated_mW"]
        coef = "calibrated"
    else:
        raise ValueError(name)
    cfg = dict(coef=coef, noise=supp if name == "V" else full, reported=rep, at_edf=at_edf,
               numerics=None, load="40x-13dBm")
    if name == "W":
        cfg["load"] = "1x-40dBm@1550nm"
    if name == "Nconv":
        cfg["noise"] = plan["noise_convergence"]
        cfg["numerics"] = dict(longitudinal_steps=4 * plan["numerics"]["longitudinal_steps"],
                               relative_error=plan["numerics"]["relative_error"] / 100)
        keep = (0, len(rep) - 1)
        cfg["reported"] = [rep[i] for i in keep]
        cfg["at_edf"] = [at_edf[i] for i in keep]
    return cfg


DESCRIPTION = {"N": "nominal card", "C": "calibrated card (alpha, g*, alpha_p x coef_scale; pump x 10^(-loss/10))",
               "V": "nominal card, ASE suppressed (one 1-GHz noise bin): C3 against Equation (5)",
               "W": "nominal card, 1 channel at 1550 nm, -40 dBm",
               "Nconv": "N with 4x steps, 1/100 relative error, half bin spacing (numerical check)"}


def run_benchmark(args, plan, log, sets):
    chan = np.asarray(plan["channels_nm"], float)
    recs = log.setdefault("runs", [])
    with Session(log, args.dialog_timeout) as sess:
        P = build(sess.cv, chan.size, log, analyzer=not args.no_analyzer)
        set_sources(P, chan, [P_CH_MW] * chan.size, log)
        set_pump(P, plan["card"]["lambda_p_nm"], plan["pumps_nominal_mW"][0], log)
        configure_edf(P.edf, plan, args, log, plan["files"]["nominal"], plan["noise_full"])
        current_load = "40x-13dBm"
        for name in sets:
            cfg = runset_config(name, plan)
            configure_edf(P.edf, plan, args, log, plan["files"][cfg["coef"]], cfg["noise"], cfg["numerics"])
            if cfg["load"] != current_load:
                if cfg["load"] == "1x-40dBm@1550nm":
                    set_sources(P, [W_LAMBDA_NM], [W_P_MW], log)
                else:
                    set_sources(P, chan, [P_CH_MW] * chan.size, log)
                current_load = cfg["load"]
            lam = np.array([W_LAMBDA_NM]) if current_load.startswith("1x") else chan
            for rep_mW, edf_mW in zip(cfg["reported"], cfg["at_edf"]):
                setp(P.pump, "Pump Laser", ["Power"], float(edf_mW), log, ("mW",))
                dt, msgs = calculate_matched(sess, P, log)
                r = dict(set=name, description=DESCRIPTION[name], pump_reported_mW=rep_mW,
                         pump_at_EDF_mW=float(edf_mW), seconds=dt, messages=msgs, time=stamp())
                try:
                    x = extract(sess, P, lam, plan["card"]["lambda_p_nm"])
                    pp = postprocess(lam, x["pin_W"], x["pout_W"], *x["bins_L"])
                    dp = dpwdma_values(sess.doc, lam) if P.dpw is not None else {"raw": {}}
                    raw = {f"dpwdma_{k}_{a}": v[i + 1] for k, v in dp["raw"].items() for i, a in enumerate("xy")}
                    meta = {"run set": f"{name} ({DESCRIPTION[name]})",
                            "software": log.get("software", ""),
                            "route": plan["route"],
                            "pump at the EDF input [mW]": f"{edf_mW:.6g} (reported {rep_mW:g})",
                            "load": current_load,
                            "coefficient file": os.path.basename(plan["files"][cfg["coef"]]),
                            "noise bins": json.dumps(jsonable(bins_summary(*x["bins_L"]))),
                            "ASE_0p1nm_dBm": ASE_RULE, "NF": NF_RULE, "written": stamp()}
                    write_run_csv(run_file(name, rep_mW), meta, lam, x["pin_W"], x["pout_W"], pp,
                                  dp.get("G"), dp.get("NF"))
                    np.savez(run_file(name, rep_mW, "npz"), lambda_nm=lam, pin_W=x["pin_W"], pout_W=x["pout_W"],
                             fwd_lo_Hz=x["bins_L"][0], fwd_hi_Hz=x["bins_L"][1], fwd_P_W=x["bins_L"][2],
                             bwd_lo_Hz=x["bins_0"][0], bwd_hi_Hz=x["bins_0"][1], bwd_P_W=x["bins_0"][2],
                             sig_in_f_Hz=x["sig_in"][0], sig_in_P_W=x["sig_in"][1], sig_L_f_Hz=x["sig_L"][0],
                             sig_L_P_W=x["sig_L"][1], sig_0_f_Hz=x["sig_0"][0], sig_0_P_W=x["sig_0"][1],
                             pump_in_W=x["pump_in_W"], pump_out_W=x["pump_out_W"],
                             port_values_fwd=x["port_values_L"], port_values_bwd=x["port_values_0"], **raw)
                    r.update(mean_gain_dB=float(np.mean(pp["G_dB"])), mean_ase_dBm=float(np.mean(pp["ASE_dBm"])),
                             residual_pump_mW=x["pump_out_W"] * 1e3, pump_in_mW=x["pump_in_W"] * 1e3,
                             bins_forward=bins_summary(*x["bins_L"]), bins_backward=bins_summary(*x["bins_0"]),
                             ase_check=x["ase_check"], iterations=x["iterations"],
                             dpwdma=dict({k: v for k, v in dp.items() if k.endswith("_error")},
                                         graphs={k: v[0] for k, v in dp["raw"].items()},
                                         channels_matched={k: int(np.isfinite(dp[k]).sum()) for k in ("G", "NF")
                                                           if k in dp}),
                             csv=os.path.relpath(run_file(name, rep_mW), ROOT))
                    print(f"{name} {rep_mW:g} mW: mean gain {r['mean_gain_dB']:.3f} dB, mean ASE "
                          f"{r['mean_ase_dBm']:.2f} dBm, {dt:.1f} s", flush=True)
                except ExtractError as e:
                    r["error"] = str(e)
                    print(f"{name} {rep_mW:g} mW: extraction failed: {e}", flush=True)
                recs.append(r)
            if name == "N":
                prod = [("EDF", P.edf), ("Pump", P.pump), ("WDM", P.cpl), ("Mux", P.mux), ("Src001", P.sources[0])]
                if P.dpw is not None:
                    prod.append(("DPWDMA", P.dpw))
                log.setdefault("all_parameters", {})["benchmark after run set N (production settings; saved as "
                                                     "perin2019_benchmark.osd)"] = {n: all_parameters(c)
                                                                                   for n, c in prod}
                if not args.no_save:
                    sess.save(os.path.join(RUNS, "perin2019_benchmark.osd"))
        comps = [("EDF", P.edf), ("Pump", P.pump), ("WDM", P.cpl), ("Mux", P.mux), ("Src001", P.sources[0])]
        if P.dpw is not None:
            comps.append(("DPWDMA", P.dpw))
        log.setdefault("all_parameters", {})["benchmark (state at the end)"] = {n: all_parameters(c)
                                                                                for n, c in comps}
        log["layout_parameters (read only, defaults of a new project)"] = layout_parameters(sess.doc)


# ----------------------------------------------------------------------------------------------------------------
# Exploration of the components (names and ports as OptiSystem reports them)
# ----------------------------------------------------------------------------------------------------------------
COMPONENTS = [("Er Doped Fiber", "Amplifiers"), ("Pump Laser", "Transmitters"), ("Ideal Mux", "WDM"),
              ("Pump Coupler Co-Propagating", "Passives"), ("Optical Null", "Tools"),
              ("Dual Port WDM Analyzer", "Visualizers")]


def explore(args, plan, log):
    out = {}
    with Session(log, args.dialog_timeout) as sess:
        for k, (typ, lib) in enumerate(COMPONENTS):
            info = dict(library=lib)
            try:
                c = sess.cv.CreateComponent(typ, LIB[lib], 40 + 90 * k, 100, 34, 34, 0)
                info["parameters"] = [dict(name=n, value=v) for n, v in all_parameters(c).items()]
                info["n_inputs"], info["n_outputs"] = int(c.GetNumberOfInputPorts()), int(c.GetNumberOfOutputPorts())
                ports = []
                for kind, n, get in (("in", info["n_inputs"], c.GetInputPort), ("out", info["n_outputs"],
                                                                                 c.GetOutputPort)):
                    for i in range(1, n + 1):
                        try:
                            ports.append([kind, i, str(get(i).Name)])
                        except Exception as e:
                            ports.append([kind, i, f"(name not readable: {str(e)[:60]})"])
                info["ports"] = ports
                if typ == "Ideal Mux":
                    c.SetParameterValue("Number of input ports", 40.0)
                    info["inputs_after_setting_40"] = int(c.GetNumberOfInputPorts())
            except Exception as e:
                info["error"] = str(e)[:300]
            out[typ] = info
    used = {"Er Doped Fiber": [(s[0], s[3]) for s in edf_spec(plan, args, plan["files"]["nominal"],
                                                              plan["noise_full"])],
            "Pump Laser": [(["Frequency"], True), (["Power"], True), (["Enabled"], True)],
            "Ideal Mux": [(["Number of input ports"], True), (["Loss"], True)],
            "Pump Coupler Co-Propagating": [(["Signal attenuation"], True), (["Pump attenuation"], True)],
            "Dual Port WDM Analyzer": [([n], False) for n in ("Lower frequency limit", "Upper frequency limit",
                                                              "Resolution bandwidth", "Noise interpolation",
                                                              "Input noise for NF calculation")]}
    missing = {}
    for typ, entries in used.items():
        have = {p["name"] for p in out.get(typ, {}).get("parameters", [])}
        missing[typ] = [dict(aliases=al, required=req) for al, req in entries if not any(a in have for a in al)]
    out["names_used_by_this_script_but_not_reported"] = missing
    with open(os.path.join(RUNS, "explore_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, indent=1)
    print("explore_parameters.json written; names not reported:", missing, flush=True)


# ----------------------------------------------------------------------------------------------------------------
def software_version():
    try:
        with open(os.path.join(OS_DIR, "Version.txt"), encoding="utf-8", errors="replace") as f:
            return " ".join(f.read().split())
    except OSError:
        return "unknown"


def write_log(log):
    os.makedirs(RUNS, exist_ok=True)
    log["finished"] = stamp()
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(log), f, indent=1)
    stamped = os.path.join(RUNS, f"run_log_{log['started'].replace(':', '').replace('-', '')}.json")
    shutil.copyfile(LOG_PATH, stamped)
    return stamped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="write the input files and the plan (no OptiSystem)")
    mode.add_argument("--explore", action="store_true", help="list component parameters and ports")
    mode.add_argument("--check", action="store_true", help="acceptance checks only (C1, C1-short, V)")
    ap.add_argument("--sets", default="V,N,C,W", help="run sets after C1 (default V,N,C,W)")
    ap.add_argument("--convergence", action="store_true", help="add run set Nconv (numerical check)")
    ap.add_argument("--skip-c1", action="store_true", help="reuse a passed C1 from run_log.json")
    ap.add_argument("--no-analyzer", action="store_true", help="no Dual Port WDM Analyzer")
    ap.add_argument("--no-save", action="store_true", help="do not save the .osd projects")
    ap.add_argument("--overlap", choices=("file", "calculate"), default="file")
    ap.add_argument("--core-radius", type=float, default=1.27, help="um; only with --overlap calculate (ASSUMED)")
    ap.add_argument("--radii", choices=("saturation", "fiber"), default="saturation")
    ap.add_argument("--steps", type=float, default=400.0)
    ap.add_argument("--rel-error", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=float, default=1000.0)
    ap.add_argument("--bin-ghz", type=float, default=50.0)
    ap.add_argument("--dialog-timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    os.makedirs(RUNS, exist_ok=True)
    if args.prepare:                                   # no OptiSystem, no run log
        plan = prepare(args)
        print(json.dumps(jsonable({k: plan[k] for k in ("route", "noise_full", "noise_suppressed", "C2",
                                                         "file_sha256")}), indent=1))
        print("input files and plan written to", INPUTS)
        return 0
    log = dict(started=stamp(), script="run_optisystem_perin2019.py", argv=sys.argv[1:], software=software_version(),
               python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
               status="started")
    code = 0
    try:
        plan = prepare(args)
        log["plan"] = os.path.relpath(os.path.join(INPUTS, "plan.json"), ROOT)
        log["route"], log["C2"] = plan["route"], plan["C2"]
        log["input_files"] = {k: dict(path=os.path.relpath(v, ROOT), sha256=plan["file_sha256"][k])
                              for k, v in plan["files"].items()}
        if args.explore:
            explore(args, plan, log)
            log["status"] = "explored"
            return 0
        sets = [s for s in args.sets.split(",") if s]
        if args.check:
            sets = ["V"]
        if args.convergence:
            sets.append("Nconv")
        if args.skip_c1:
            prev = {}
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, encoding="utf-8") as f:
                    prev = json.load(f)
            if not prev.get("C1", {}).get("passed"):
                raise ParamError("--skip-c1 needs a passed C1 in run_log.json")
            log["C1"] = dict(prev["C1"], reused_from=prev.get("started"))
        else:
            if not run_c1(args, plan, log):
                log["status"] = "C1 failed: the card was not entered correctly; benchmark not run"
                print(log["status"], flush=True)
                return 5
            time.sleep(5.0)
        run_benchmark(args, plan, log, sets)
        v = [r for r in log.get("runs", []) if r["set"] == "V"]
        log["status"] = "completed" + ("" if v else " (no run set V)")
    except ForeignInstance as e:
        log["status"], code = f"not started: {e}", 2
    except LicenseUnavailable as e:
        log["status"], code = f"licence unavailable: {e}", 3
    except ParamError as e:
        log["status"], code = f"parameter mismatch: {e}", 4
    except Exception as e:
        log["status"], code = f"error: {type(e).__name__}: {e}", 1
        log["traceback"] = traceback.format_exc()
    finally:
        path = write_log(log)
        print(f"status: {log['status']}\nrun log: {LOG_PATH} (copy {os.path.basename(path)})", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
