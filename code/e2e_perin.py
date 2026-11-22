"""End-to-end case study of the benchmarking framework (Section 5.7) on ONE open device.

Device: the single-stage, 980-nm forward-pumped C-band EDFA of ref. [124]
(J.K. Perin, J.M. Kahn, J.D. Downie, J. Hurley, K. Bennett, J. Lightwave Technol. 37(9), 2076-2085, 2019;
arXiv:1803.07905v3, Appendix A, Figs. 7-8): 8 m of Er-doped fiber, 40 channels at -13 dBm (1531-1562 nm),
six pump powers 18.1-114.9 mW.

Layers exercised (Figure 2 of the review):
  L1  parameter card EDFA-PERIN2019 (../cards/edfa_perin2019.json), read through cards.py; McCumber consistency
      of the digitized g*/alpha recorded as a V2 test.
  L2  four sources at identical operating points
        (i)   multichannel closed form of Saleh et al. (Equation 5): no ASE, no background loss;
        (ii)  spectrally resolved Giles-Desurvire model with forward and backward ASE (this module), solved by
              forward-backward sweeps with RK4 steps;
        (iii) independent simulator: OptiSystem 23.1 ('Er Doped Fiber', steady state), run with the same card and
              the same operating points by run_optisystem_perin2019.py (input sheet:
              ../notes/optisystem_input_perin2019.md; run files: ../notes/optisystem_perin2019/runs/) and compared
              with (ii) and (iv) by e2e_simulator_compare.py (-> ../figures/E2E_simulator_results.json);
        (iv)  gain and ASE measured by the source (Fig. 8, digitized: ../data/perin2019_fig8_gain_ase.json),
              together with the source's own model curves.
      Metrics: RMSE, maximum absolute error and bias (model minus reference), in dB.
  L4  least-squares identification (scipy.optimize.least_squares) of at most two parameters on three pump powers;
      blind prediction of the other three pump powers and of the ASE; calibrated card written to
      ../cards/edfa_perin2019_calibrated.json.  (L4 here is least squares; the PINN of Section 7.3 is a separate
      illustration on a Raman problem and is not used in this chain.)
  L3  one design decision (EDF length and pump power for a target mean gain at minimum gain ripple) taken with the
      nominal and with the calibrated card.

Conventions
  * Powers in W inside the solvers; gains and ASE reported in dB / dBm.
  * ASE is resolved in bins of width dlam (nm) over the tabulated card range; each bin carries both polarizations
    (m = 2 in Equation 3). The ASE "in 0.1 nm" is S(lambda) * c * 0.1 nm / lambda^2, with S the two-polarization
    power spectral density of the forward ASE at the fiber output, interpolated to the channel wavelengths.
  * Internal noise figure: NF = (1 + S/(h nu)) / G (signal-spontaneous beat plus shot noise; S two-polarization PSD).
  * Measurement-plane quantities: the pump power and the channel powers given by the source are taken at a reference
    plane that is not stated in [124]. Optional lumped losses (pump_loss_dB, in_loss_dB, out_loss_dB) map the
    reported quantities to the fiber ends: P_pump,EDF = P_pump,reported * 10^(-pump_loss_dB/10), etc.; the reported
    gain is G_int - in_loss_dB - out_loss_dB and the reported ASE is ASE_int - out_loss_dB.

Run:  python e2e_perin.py            -> ../figures/Fig12_end_to_end_results.json, ../cards/edfa_perin2019_calibrated.json
                                        (about 6-7 min on one CPU core)
      python e2e_perin.py --summary  -> refresh the 'summary' block of an existing results file
      python e2e_perin.py --export   -> ../notes/optisystem_perin2019/*.csv (simulator input and reference values)
      python fig_e2e.py              -> ../figures/Fig12_end_to_end.png / .pdf
"""
import copy
import json
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq, least_squares

import cards
from models import h, c, dbm2w, w2dbm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
CARD_PATH = os.path.join(cards.CARD_DIR, "edfa_perin2019.json")
CAL_CARD_PATH = os.path.join(cards.CARD_DIR, "edfa_perin2019_calibrated.json")
DATA_PATH = os.path.join(cards.DATA_DIR, "perin2019_fig8_gain_ase.json")
RESULTS_PATH = os.path.join(FIG_DIR, "Fig12_end_to_end_results.json")

KB = 1.380649e-23          # Boltzmann constant [J/K]
HC_EV_NM = 1239.841984     # h c [eV nm]
M_POL = 2                  # polarization modes per ASE bin (Equation 3)
P_CH_DBM = -13.0           # per-channel input power of the source's experiment [dBm]
PUMPS_MW = (18.1, 31.9, 45.8, 59.6, 87.2, 114.9)   # pump powers of [124], Fig. 8 [mW]


# ----------------------------------------------------------------------------------------------------------------
# L1: device description built from the card
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Device:
    """Giles parameters of one EDF plus the lumped losses that map reported quantities to the fiber ends.

    Spectral coefficients are tabulated (lam_tab [nm], alpha_tab and gstar_tab [1/m]) and linearly interpolated.
    The identification parameters default to the nominal card (scales 1, losses 0)."""
    L: float                      # EDF length [m]
    zeta: float                   # saturation parameter pi b^2 N_t / tau [1/(m s)]
    lam_p: float                  # pump wavelength [nm]
    alpha_p: float                # absorption coefficient at the pump [1/m]
    g_p: float                    # gain coefficient at the pump [1/m]
    lam_tab: np.ndarray = field(repr=False)
    alpha_tab: np.ndarray = field(repr=False)
    gstar_tab: np.ndarray = field(repr=False)
    l_bg: float = 0.0             # background (excess) loss, same at all wavelengths [1/m]; the source sets 0
    coef_scale: float = 1.0       # common scale of alpha(lambda), g*(lambda) and alpha_p
    pump_scale: float = 1.0       # additional scale of alpha_p only
    pump_loss_dB: float = 0.0     # loss between the reported pump power and the EDF input [dB]
    in_loss_dB: float = 0.0       # loss between the reported signal input plane and the EDF input [dB]
    out_loss_dB: float = 0.0      # loss between the EDF output and the measurement plane [dB]
    pin_offset_dB: float = 0.0    # error of the absolute channel input level (moves the operating point only; the
                                  # measured gain is a ratio and the ASE is read at the output, so neither is shifted)

    def alpha(self, lam_nm):
        return self.coef_scale * np.interp(lam_nm, self.lam_tab, self.alpha_tab)

    def gstar(self, lam_nm):
        return self.coef_scale * np.interp(lam_nm, self.lam_tab, self.gstar_tab)

    def pump_coeffs(self):
        return self.coef_scale * self.pump_scale * self.alpha_p, self.coef_scale * self.g_p


def load_device(card_path=CARD_PATH):
    """Nominal Device from the card file (validated by cards.load). Returns (Device, card dict)."""
    card = cards.load(card_path)
    v = cards.values(card)
    lam, al, gs = cards.perin2019_coefficients()
    zeta = v["zeta"]
    # consistency of the derived entry with its inputs (zeta = pi b^2 N_t / tau)
    zeta_chk = np.pi * v["r_er"] ** 2 * v["n_t"] / v["tau"]
    assert abs(zeta_chk / zeta - 1) < 1e-3, "card entry zeta inconsistent with r_er, n_t, tau"
    dev = Device(L=v["L"], zeta=zeta, lam_p=v["lambda_p"], alpha_p=v["alpha_p"], g_p=v["g_p"],
                 lam_tab=lam, alpha_tab=al, gstar_tab=gs, l_bg=float(v.get("l_bg", 0.0)))
    return dev, card


def load_measurement(path=DATA_PATH):
    """Digitized Fig. 8 of [124]: per pump power the channel wavelengths, measured gain and ASE (0.1 nm),
    and the source's own model curves sampled at the same wavelengths."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for key, d in raw["data"].items():
        out[float(key)] = dict(
            lam=np.array(d["lambda_nm"]), G=np.array(d["gain_dB_measured"]),
            ASE=np.interp(d["lambda_nm"], d["ase_lambda_nm"], d["ase_dBm_per_0p1nm_measured"]),
            G_src=np.interp(d["lambda_nm"], d["theory_lambda_nm"], d["gain_dB_source_model"]),
            ASE_src=np.interp(d["lambda_nm"], d["theory_ase_lambda_nm"], d["ase_dBm_per_0p1nm_source_model"]))
    return out, {k: raw[k] for k in raw if k != "data"}


# ----------------------------------------------------------------------------------------------------------------
# L1 consistency test: McCumber relation (Equation 1) applied to the digitized g*/alpha
# ----------------------------------------------------------------------------------------------------------------
def mccumber_ratio(lam_nm, lam_x_nm, T):
    """g*/alpha = sigma_e/sigma_a = exp[(hc/kT)(1/lambda_x - 1/lambda)]."""
    kT_eV = KB * T / 1.602176634e-19
    return np.exp(HC_EV_NM / kT_eV * (1.0 / lam_x_nm - 1.0 / np.asarray(lam_nm)))


def mccumber_test(dev, T=300.0, band=(1500.0, 1565.0)):
    """V2 consistency test of the card spectra.

    Pointwise crossing wavelength lambda_x(lambda) = 1 / [1/lambda + (kT/hc) ln(g*/alpha)] at temperature T;
    its median and 5-95 % range over the full table and over `band`; maximum deviation of g* from the McCumber
    transform of alpha with the median lambda_x; and a two-parameter fit (lambda_x, T) of ln(g*/alpha)."""
    lam = dev.lam_tab
    r = dev.gstar_tab / dev.alpha_tab
    kT_eV = KB * T / 1.602176634e-19
    lam_x = 1.0 / (1.0 / lam + kT_eV / HC_EV_NM * np.log(r))
    res = dict(T_K=T)
    for name, sel in (("full_table", np.ones_like(lam, bool)), ("band", (lam >= band[0]) & (lam <= band[1]))):
        lx = lam_x[sel]
        med = float(np.median(lx))
        dev_g = dev.alpha_tab[sel] * mccumber_ratio(lam[sel], med, T) / dev.gstar_tab[sel] - 1.0
        res[name] = dict(range_nm=[float(lam[sel][0]), float(lam[sel][-1])], lambda_x_median_nm=med,
                         lambda_x_p05_nm=float(np.percentile(lx, 5)), lambda_x_p95_nm=float(np.percentile(lx, 95)),
                         max_dev_gstar_percent=float(100 * np.max(np.abs(dev_g))),
                         rms_dev_gstar_percent=float(100 * np.sqrt(np.mean(dev_g ** 2))))
    # free temperature: if g* was computed from alpha with Equation (1), a pair (lambda_x, T) fits almost exactly
    fit = least_squares(lambda p: np.log(r) - np.log(mccumber_ratio(lam, p[0], p[1])), [1533.0, 300.0])
    d = r / mccumber_ratio(lam, *fit.x) - 1.0
    res["free_T_fit"] = dict(lambda_x_nm=float(fit.x[0]), T_K=float(fit.x[1]),
                             rms_dev_percent=float(100 * np.sqrt(np.mean(d ** 2))),
                             max_dev_percent=float(100 * np.max(np.abs(d))))
    res["pointwise_lambda_x_nm"] = lam_x
    return res


# ----------------------------------------------------------------------------------------------------------------
# L2 source (i): multichannel closed form, Equation (5)
# ----------------------------------------------------------------------------------------------------------------
def saleh_gain(dev, Pp_W, lam_s_nm, Ps_W, L=None):
    """Internal gains (linear) of all channels from Equation (5) for a forward pump of Pp_W [W] at the EDF input.

    Exact for the two-level model without ASE and without background loss (photon balance, Saleh et al. 1990)."""
    if dev.l_bg != 0.0:
        raise ValueError("Equation (5) assumes zero background loss")
    L = dev.L if L is None else L
    a_p, g_p = dev.pump_coeffs()
    lam = np.r_[dev.lam_p, lam_s_nm] * 1e-9
    nu = c / lam
    a = np.r_[a_p, dev.alpha(lam_s_nm)]
    g = np.r_[g_p, dev.gstar(lam_s_nm)]
    Pin = np.r_[Pp_W, np.broadcast_to(Ps_W, np.shape(lam_s_nm))]
    q_in = Pin / (h * nu)
    Qin = q_in.sum()

    def f(Qout):
        with np.errstate(over="ignore"):
            return np.sum(q_in * np.exp(np.clip((a + g) * (Qin - Qout) / dev.zeta - a * L, -700, 700))) - Qout

    Qout = brentq(f, 1e-12 * Qin, Qin, xtol=1e-12 * Qin, rtol=1e-15, maxiter=500)
    G = np.exp((a + g) * (Qin - Qout) / dev.zeta - a * L)
    return G[1:], G[0]


# ----------------------------------------------------------------------------------------------------------------
# L2 source (ii): spectrally resolved Giles model with forward and backward ASE
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Numerics:
    nz: int = 200                 # spatial steps (uniform grid, RK4 per step)
    dlam: float = 0.5             # ASE bin width [nm]
    lam_min: float = 1470.0       # ASE band = card table range [nm]
    lam_max: float = 1569.0
    tol_dB: float = 1e-7          # stop when outputs change by less than this between sweeps [dB]
    max_sweeps: int = 300

    def ase_grid(self):
        n = int(np.floor((self.lam_max - self.lam_min) / self.dlam + 1e-9))
        return self.lam_min + self.dlam * (np.arange(n) + 0.5)


NUM = Numerics(dlam=0.25)                          # production setting (convergence: see numerics_study)
NUM_FIT = Numerics(nz=100, dlam=1.0, tol_dB=1e-6)  # faster setting used inside the least-squares loops


def giles_ase(dev, Pp_W, lam_s_nm, Ps_W, L=None, num=NUM, ase=True, history=False):
    """Steady-state Giles model (Equations 3-4) with spectrally resolved forward and backward ASE.

    Batch evaluation: Pp_W has shape (B,); Ps_W has shape (Ns,) or (B, Ns); L is None (card value) or (B,).
    The forward channels (pump, signals, forward ASE) are integrated from z = 0 to L with the backward ASE frozen,
    then the backward ASE from z = L to 0 with the forward channels frozen; frozen channels are interpolated
    linearly between grid nodes. The sweeps are repeated until the largest change of the output signals, the
    forward ASE at z = L and the backward ASE at z = 0 is below num.tol_dB.

    Returns a dict with internal gain G (B, Ns) [linear], forward-ASE two-polarization PSD at the channel
    wavelengths S (B, Ns) [W/Hz], internal NF (B, Ns) [linear], residual pump, total forward/backward ASE [W],
    mean inversion, number of sweeps and (optionally) the convergence history."""
    Pp_W = np.atleast_1d(np.asarray(Pp_W, float))
    B = Pp_W.size
    lam_s_nm = np.asarray(lam_s_nm, float)
    Ns = lam_s_nm.size
    Ps = np.broadcast_to(np.asarray(Ps_W, float), (B, Ns))
    Lv = np.full(B, dev.L) if L is None else np.broadcast_to(np.asarray(L, float), (B,)).copy()
    nz = num.nz
    dz = (Lv / nz)[:, None]

    lam_a = num.ase_grid() if ase else np.zeros(0)
    Na = lam_a.size
    a_p, g_p = dev.pump_coeffs()
    lam_f = np.r_[dev.lam_p, lam_s_nm, lam_a]          # forward channels: pump, signals, forward ASE
    nu_f = c / (lam_f * 1e-9)
    nu_b = c / (lam_a * 1e-9)
    a_f = np.r_[a_p, dev.alpha(lam_s_nm), dev.alpha(lam_a)]
    g_f = np.r_[g_p, dev.gstar(lam_s_nm), dev.gstar(lam_a)]
    a_b, g_b = dev.alpha(lam_a), dev.gstar(lam_a)
    l_f = np.full(lam_f.size, dev.l_bg)
    l_b = np.full(Na, dev.l_bg)
    dnu_a = c * num.dlam * 1e-9 / (lam_a * 1e-9) ** 2
    sp_f = np.r_[np.zeros(1 + Ns), M_POL * h * nu_b * dnu_a]   # spontaneous power per unit length / (g* n2)
    sp_b = M_POL * h * nu_b * dnu_a
    w_f, w_b = 1.0 / (h * nu_f * dev.zeta), 1.0 / (h * nu_b * dev.zeta)
    aw_f, agw_f = a_f * w_f, (a_f + g_f) * w_f
    aw_b, agw_b = a_b * w_b, (a_b + g_b) * w_b
    ag_f, ag_b = a_f + g_f, a_b + g_b
    gsp_f, gsp_b = g_f * sp_f, g_b * sp_b
    al_f, al_b = a_f + l_f, a_b + l_b

    nf = lam_f.size
    Pf = np.zeros((B, nf, nz + 1))
    Pb = np.zeros((B, Na, nz + 1))
    Pf[:, 0, 0] = Pp_W
    Pf[:, 1:1 + Ns, 0] = Ps

    def n2(pf, pb):
        num_ = pf @ aw_f + pb @ aw_b
        den_ = 1.0 + pf @ agw_f + pb @ agw_b
        return (num_ / den_)[:, None]

    def rhs_f(pf, pb):
        x = n2(pf, pb)
        return (ag_f * x - al_f) * pf + gsp_f * x

    def rhs_b(pb, pf):       # d/ds with s = L - z (growth along the propagation direction)
        x = n2(pf, pb)
        return (ag_b * x - al_b) * pb + gsp_b * x

    hist = []
    last = None
    floor = 1e-30
    sweeps = 0
    for sweeps in range(1, num.max_sweeps + 1):
        for i in range(nz):                                   # forward sweep
            p = Pf[:, :, i]
            b0, b1 = Pb[:, :, i], Pb[:, :, i + 1]
            bm = 0.5 * (b0 + b1)
            k1 = rhs_f(p, b0)
            k2 = rhs_f(p + 0.5 * dz * k1, bm)
            k3 = rhs_f(p + 0.5 * dz * k2, bm)
            k4 = rhs_f(p + dz * k3, b1)
            Pf[:, :, i + 1] = np.maximum(p + dz / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
        if Na == 0:
            break
        Pb[:, :, nz] = 0.0                                    # no reflection at z = L
        for i in range(nz, 0, -1):                            # backward sweep
            p = Pb[:, :, i]
            f0, f1 = Pf[:, :, i], Pf[:, :, i - 1]
            fm = 0.5 * (f0 + f1)
            k1 = rhs_b(p, f0)
            k2 = rhs_b(p + 0.5 * dz * k1, fm)
            k3 = rhs_b(p + 0.5 * dz * k2, fm)
            k4 = rhs_b(p + dz * k3, f1)
            Pb[:, :, i - 1] = np.maximum(p + dz / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
        out = np.concatenate([Pf[:, 1:, nz], Pb[:, :, 0]], axis=1)
        if last is not None:
            ch = np.max(np.abs(10 * np.log10(np.maximum(out, floor) / np.maximum(last, floor))))
            hist.append(float(ch))
            if ch < num.tol_dB:
                break
        last = out.copy()

    G = Pf[:, 1:1 + Ns, nz] / Ps
    res = dict(G=G, Pp_out=Pf[:, 0, nz], sweeps=sweeps, converged=(Na == 0 or (hist and hist[-1] < num.tol_dB)))
    if Na:
        S_bins = Pf[:, 1 + Ns:, nz] / dnu_a                  # two-polarization PSD of forward ASE [W/Hz]
        S = np.array([10 ** np.interp(lam_s_nm, lam_a, np.log10(np.maximum(sb, floor))) for sb in S_bins])
        res.update(S=S, NF=(1.0 + S / (h * c / (lam_s_nm * 1e-9))) / G,
                   Pase_f=Pf[:, 1 + Ns:, nz].sum(1), Pase_b=Pb[:, :, 0].sum(1),
                   S_bins=S_bins, lam_ase=lam_a)
    # mean inversion along the fiber (trapezoidal), for diagnostics
    x = np.empty((B, nz + 1))
    for i in range(nz + 1):
        x[:, i] = n2(Pf[:, :, i], Pb[:, :, i])[:, 0]
    res["n2_mean"] = np.trapezoid(x, dx=1.0, axis=1) / nz
    if history:
        res["history"] = hist
    return res


def ase_01nm_dBm(S, lam_nm):
    """ASE power in 0.1 nm [dBm] from the two-polarization PSD S [W/Hz]."""
    return w2dbm(S * c * 0.1e-9 / (np.asarray(lam_nm) * 1e-9) ** 2)


# ----------------------------------------------------------------------------------------------------------------
# Reported (measurement-plane) predictions
# ----------------------------------------------------------------------------------------------------------------
def predict(dev, pumps_mW, lam_s_nm, model="giles", num=NUM, p_ch_dBm=P_CH_DBM, L=None):
    """Gain [dB] and ASE in 0.1 nm [dBm] at the measurement plane, internal NF [dB], for the reported pump powers.

    model: 'giles' (spectrally resolved, with ASE), 'giles_noase' (same solver, ASE off) or 'saleh' (Equation 5)."""
    pumps_mW = np.atleast_1d(np.asarray(pumps_mW, float))
    Pp = pumps_mW * 1e-3 * 10 ** (-dev.pump_loss_dB / 10)
    Ps = np.full(np.size(lam_s_nm), dbm2w(p_ch_dBm + dev.pin_offset_dB - dev.in_loss_dB))
    loss_g = dev.in_loss_dB + dev.out_loss_dB
    if model == "saleh":
        Lv = np.full(pumps_mW.size, dev.L) if L is None else np.broadcast_to(L, pumps_mW.shape)
        G = np.array([saleh_gain(dev, p, lam_s_nm, Ps, L=l)[0] for p, l in zip(Pp, Lv)])
        return dict(G_dB=10 * np.log10(G) - loss_g)
    r = giles_ase(dev, Pp, lam_s_nm, Ps, L=L, num=num, ase=(model == "giles"))
    out = dict(G_dB=10 * np.log10(r["G"]) - loss_g, sweeps=r["sweeps"], n2_mean=r["n2_mean"], Pp_out=r["Pp_out"])
    if model == "giles":
        out.update(ASE_dBm=ase_01nm_dBm(r["S"], lam_s_nm) - dev.out_loss_dB, NF_dB=10 * np.log10(r["NF"]),
                   Pase_f=r["Pase_f"], Pase_b=r["Pase_b"])
    return out


def stats(err):
    """RMSE, maximum absolute error and bias (mean of model minus reference) [dB]."""
    e = np.asarray(err, float).ravel()
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), max_abs=float(np.max(np.abs(e))), bias=float(np.mean(e)),
                n=int(e.size))


def metrics_table(model_by_pump, ref_by_pump, pumps):
    """Per-pump and overall statistics of model minus reference (dicts keyed by pump power)."""
    per = {f"{p:g}": stats(model_by_pump[p] - ref_by_pump[p]) for p in pumps}
    allerr = np.concatenate([model_by_pump[p] - ref_by_pump[p] for p in pumps])
    return dict(per_pump=per, overall=stats(allerr))


# ----------------------------------------------------------------------------------------------------------------
# Numerical error: grid, ASE bin width, sweep tolerance; code verification against Equation (5)
# ----------------------------------------------------------------------------------------------------------------
def numerics_study(dev, lam_s, pumps=PUMPS_MW):
    """Numerical error of the ASE-resolved model at the measured operating points (all six pump powers)."""
    out = {}
    ref_num = Numerics(nz=1600, dlam=0.125, tol_dB=1e-10)
    t0 = time.perf_counter()
    ref = predict(dev, pumps, lam_s, num=ref_num)
    out["reference_setting"] = dict(nz=ref_num.nz, dlam_nm=ref_num.dlam, tol_dB=ref_num.tol_dB,
                                    runtime_s=time.perf_counter() - t0)

    def err(r):
        return dict(gain_max_abs_dB=float(np.max(np.abs(r["G_dB"] - ref["G_dB"]))),
                    ase_max_abs_dB=float(np.max(np.abs(r["ASE_dBm"] - ref["ASE_dBm"]))))

    grid = []
    for nz in (25, 50, 100, 200, 400, 800):
        r = predict(dev, pumps, lam_s, num=Numerics(nz=nz, dlam=ref_num.dlam, tol_dB=1e-10))
        grid.append(dict(nz=nz, **err(r)))
    out["grid"] = grid
    bins = []
    for dl in (4.0, 2.0, 1.0, 0.5, 0.25):
        r = predict(dev, pumps, lam_s, num=Numerics(nz=ref_num.nz, dlam=dl, tol_dB=1e-10))
        bins.append(dict(dlam_nm=dl, **err(r)))
    out["ase_bins"] = bins
    tols = []
    for tol in (1e-3, 1e-5, 1e-7):
        r = predict(dev, pumps, lam_s, num=Numerics(nz=ref_num.nz, dlam=ref_num.dlam, tol_dB=tol))
        tols.append(dict(tol_dB=tol, sweeps=int(r["sweeps"]), **err(r)))
    out["sweep_tolerance"] = tols
    t0 = time.perf_counter()
    prod = predict(dev, pumps, lam_s, num=NUM)
    out["production"] = dict(nz=NUM.nz, dlam_nm=NUM.dlam, tol_dB=NUM.tol_dB, sweeps=int(prod["sweeps"]),
                             runtime_s=time.perf_counter() - t0, **err(prod))
    fit = predict(dev, pumps, lam_s, num=NUM_FIT)
    out["fit_setting"] = dict(nz=NUM_FIT.nz, dlam_nm=NUM_FIT.dlam, tol_dB=NUM_FIT.tol_dB, **err(fit))
    # convergence history of the sweeps at the production grid (strongest pump)
    r = giles_ase(dev, [PUMPS_MW[-1] * 1e-3], lam_s, dbm2w(P_CH_DBM), num=Numerics(nz=NUM.nz, dlam=NUM.dlam,
                                                                                    tol_dB=1e-12), history=True)
    out["sweep_history_114p9mW"] = r["history"]
    # code verification: ASE switched off, the sweep solver must reproduce Equation (5)
    ver = []
    for nz in (25, 50, 100, 200):
        a = predict(dev, pumps, lam_s, model="giles_noase", num=Numerics(nz=nz))
        b = predict(dev, pumps, lam_s, model="saleh")
        ver.append(dict(nz=nz, max_abs_dB=float(np.max(np.abs(a["G_dB"] - b["G_dB"])))))
    out["verification_vs_eq5"] = ver
    return out


# ----------------------------------------------------------------------------------------------------------------
# Model-form error: closed form (no ASE) against the ASE-resolved model, by load
# ----------------------------------------------------------------------------------------------------------------
def model_form_study(dev, lam_s, pumps=PUMPS_MW):
    """Equation (5) minus the ASE-resolved model, same card, for the measured load and for weaker loads."""
    out = []
    loads = [("40 x -13 dBm (measured load)", lam_s, -13.0), ("40 x -20 dBm", lam_s, -20.0),
             ("40 x -30 dBm", lam_s, -30.0), ("1 x -20 dBm at 1550 nm", np.array([1550.0]), -20.0),
             ("1 x -40 dBm at 1550 nm", np.array([1550.0]), -40.0),
             ("1 x -40 dBm at 1532 nm", np.array([1532.0]), -40.0)]
    for name, ls, pin in loads:
        a = predict(dev, pumps, ls, p_ch_dBm=pin, model="saleh")["G_dB"]
        b = predict(dev, pumps, ls, p_ch_dBm=pin, model="giles")
        e = a - b["G_dB"]
        out.append(dict(load=name, p_ch_dBm=pin, n_ch=int(np.size(ls)),
                        per_pump={f"{p:g}": stats(e[i]) for i, p in enumerate(pumps)}, overall=stats(e),
                        giles_gain_mean_dB=[float(x) for x in b["G_dB"].mean(1)],
                        giles_NF_max_dB=[float(x) for x in b["NF_dB"].max(1)],
                        Pase_b_mW=[float(x) * 1e3 for x in b["Pase_b"]]))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Parameter-uncertainty propagation
# ----------------------------------------------------------------------------------------------------------------
# One-standard-deviation uncertainties assumed for the propagation (stated in the note and in the card):
UNC = {
    "coef_scale": (0.05, "relative", "common scale of alpha(lambda), g*(lambda), alpha_p; source: 'experimental error "
                                     "... about 5%' ([124], App. A), taken as one standard deviation"),
    "ratio": (0.02, "relative", "g*/alpha ratio; equivalent to +-1 nm in the McCumber crossing wavelength (assumed)"),
    "pump_scale": (0.05, "relative", "alpha_p alone (pump wavelength 'near 980 nm'; assumed)"),
    "zeta": (0.10, "relative", "saturation parameter (r_Er, N_t, tau 'approximately 10 ms'; assumed)"),
    "L": (0.01, "relative", "EDF length (assumed)"),
    "pump_dB": (0.10, "dB", "pump power at the fiber input (assumed)"),
    "pin_dB": (0.10, "dB", "channel input power (assumed)"),
}


def _perturbed(dev, name, x):
    """Device and input offsets for a perturbation x (relative or dB) of parameter `name`."""
    d = copy.copy(dev)
    dp = 0.0
    if name == "coef_scale":
        d.coef_scale = dev.coef_scale * (1 + x)
    elif name == "ratio":
        d.gstar_tab = dev.gstar_tab * (1 + x)
    elif name == "pump_scale":
        d.pump_scale = dev.pump_scale * (1 + x)
    elif name == "zeta":
        d.zeta = dev.zeta * (1 + x)
    elif name == "L":
        d.L = dev.L * (1 + x)
    elif name == "pump_dB":
        dp = x
    elif name == "pin_dB":
        d.pin_offset_dB = dev.pin_offset_dB + x       # operating point only
    d.pump_loss_dB = dev.pump_loss_dB - dp
    return d


def uncertainty_study(dev, lam_s, pumps=PUMPS_MW, n_mc=2000, seed=7):
    """Linearized propagation with the ASE-resolved model (gain and ASE) and a Monte Carlo cross-check of the gain
    with Equation (5), for the one-standard-deviation uncertainties in UNC."""
    base = predict(dev, pumps, lam_s, num=NUM_FIT)
    contrib = {}
    var_g = np.zeros_like(base["G_dB"])
    var_a = np.zeros_like(base["ASE_dBm"])
    for name, (sig, _, _) in UNC.items():
        up = predict(_perturbed(dev, name, +sig), pumps, lam_s, num=NUM_FIT)
        dn = predict(_perturbed(dev, name, -sig), pumps, lam_s, num=NUM_FIT)
        dg = 0.5 * (up["G_dB"] - dn["G_dB"])
        da = 0.5 * (up["ASE_dBm"] - dn["ASE_dBm"])
        var_g += dg ** 2
        var_a += da ** 2
        contrib[name] = dict(sigma=sig, gain_rms_dB=float(np.sqrt(np.mean(dg ** 2))),
                             gain_mean_shift_by_pump_dB=[float(x) for x in dg.mean(1)],
                             ase_rms_dB=float(np.sqrt(np.mean(da ** 2))))
    sd_g, sd_a = np.sqrt(var_g), np.sqrt(var_a)
    # Monte Carlo (gain only, Equation 5; ASE changes gain by about 0.01 dB at these points)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_mc):
        d = copy.copy(dev)
        d.coef_scale = dev.coef_scale * (1 + UNC["coef_scale"][0] * rng.standard_normal())
        d.gstar_tab = dev.gstar_tab * (1 + UNC["ratio"][0] * rng.standard_normal())
        d.pump_scale = dev.pump_scale * (1 + UNC["pump_scale"][0] * rng.standard_normal())
        d.zeta = dev.zeta * (1 + UNC["zeta"][0] * rng.standard_normal())
        d.L = dev.L * (1 + UNC["L"][0] * rng.standard_normal())
        d.pin_offset_dB = dev.pin_offset_dB + UNC["pin_dB"][0] * rng.standard_normal()
        d.pump_loss_dB = dev.pump_loss_dB - UNC["pump_dB"][0] * rng.standard_normal()
        samples.append(predict(d, pumps, lam_s, model="saleh")["G_dB"])
    samples = np.array(samples)
    sd_mc = samples.std(0)
    return dict(
        assumed_sigma={k: dict(value=v[0], kind=v[1], basis=v[2]) for k, v in UNC.items()},
        linear=dict(gain_sd_rms_dB=float(np.sqrt(np.mean(sd_g ** 2))), gain_sd_max_dB=float(sd_g.max()),
                    gain_sd_by_pump_dB=[float(np.sqrt(np.mean(s ** 2))) for s in sd_g],
                    ase_sd_rms_dB=float(np.sqrt(np.mean(sd_a ** 2))), ase_sd_max_dB=float(sd_a.max()),
                    ase_sd_by_pump_dB=[float(np.sqrt(np.mean(s ** 2))) for s in sd_a], contributions=contrib),
        monte_carlo_eq5=dict(n=n_mc, seed=seed, gain_sd_rms_dB=float(np.sqrt(np.mean(sd_mc ** 2))),
                             gain_sd_max_dB=float(sd_mc.max()),
                             gain_sd_by_pump_dB=[float(np.sqrt(np.mean(s ** 2))) for s in sd_mc]),
        gain_sd_dB=sd_g, ase_sd_dB=sd_a)


# ----------------------------------------------------------------------------------------------------------------
# L4: least-squares identification, fit-then-predict
# ----------------------------------------------------------------------------------------------------------------
# Candidate parameters: name -> (attribute, start value, lower bound, upper bound, unit, physical meaning)
PARAMS = {
    "pump_loss_dB": ("pump_loss_dB", 1.0, -3.0, 6.0, "dB", "loss between the reported pump power and the EDF input"),
    "out_loss_dB": ("out_loss_dB", 1.0, -3.0, 6.0, "dB", "loss between the EDF output and the measurement plane"),
    "in_loss_dB": ("in_loss_dB", 0.5, -3.0, 6.0, "dB", "loss between the reported input plane and the EDF input"),
    "coef_scale": ("coef_scale", 1.0, 0.5, 1.5, "1", "common scale of alpha(lambda), g*(lambda) and alpha_p"),
    "pump_scale": ("pump_scale", 0.8, 0.2, 1.5, "1", "scale of the pump absorption alpha_p only"),
    "l_bg_dB_per_m": ("l_bg", 0.1, 0.0, 1.0, "dB/m", "wavelength-independent background loss of the EDF"),
    "zeta_scale": ("zeta", 1.0, 0.1, 10.0, "1", "scale of the saturation parameter"),
    "L_scale": ("L", 1.0, 0.5, 1.5, "1", "scale of the EDF length"),
}
DB_PER_M = np.log(10) / 10


def _apply(dev, names, x):
    d = copy.copy(dev)
    for n, v in zip(names, x):
        attr = PARAMS[n][0]
        if n == "l_bg_dB_per_m":
            d.l_bg = v * DB_PER_M
        elif n in ("zeta_scale", "L_scale"):
            setattr(d, attr, getattr(dev, attr) * v)
        else:
            setattr(d, attr, v)
    return d


def identify(dev, names, meas, cal_pumps, lam_s, num=NUM_FIT):
    """Least-squares fit of the parameters `names` to the measured gain at the calibration pump powers.
    Returns the fitted values, a formal one-standard-deviation estimate (independent-residual assumption,
    which underestimates for spectrally correlated residuals) and the correlation matrix."""
    y = np.concatenate([meas[p]["G"] for p in cal_pumps])
    x0 = [PARAMS[n][1] for n in names]
    lo = [PARAMS[n][2] for n in names]
    hi = [PARAMS[n][3] for n in names]

    def resid(x):
        return predict(_apply(dev, names, x), cal_pumps, lam_s, num=num)["G_dB"].ravel() - y

    t0 = time.perf_counter()
    r = least_squares(resid, x0, bounds=(lo, hi), diff_step=1e-4, x_scale="jac", xtol=1e-10, ftol=1e-10)
    dof = max(len(y) - len(names), 1)
    s2 = float(np.sum(r.fun ** 2) / dof)
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * s2
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        sd = np.full(len(names), np.nan)
        corr = np.full((len(names), len(names)), np.nan)
    # residuals of neighboring channels are correlated (systematic spectral structure); an AR(1) correction with
    # the lag-1 autocorrelation rho inflates the formal standard deviation by sqrt((1 + rho)/(1 - rho))
    res_by_pump = r.fun.reshape(len(cal_pumps), -1)
    rho = float(np.mean([np.corrcoef(e[:-1], e[1:])[0, 1] for e in res_by_pump]))
    infl = float(np.sqrt((1 + rho) / (1 - rho))) if rho < 0.999 else np.inf
    return dict(names=list(names), x=[float(v) for v in r.x], sd_formal=[float(v) for v in sd],
                residual_lag1_autocorr=rho, sd_ar1=[float(v * infl) for v in sd],
                corr=corr.tolist(), cal_rmse_dB=float(np.sqrt(np.mean(r.fun ** 2))),
                cal_bias_dB=float(np.mean(r.fun)), cal_max_abs_dB=float(np.max(np.abs(r.fun))), nfev=int(r.nfev),
                runtime_s=time.perf_counter() - t0, at_bound=bool(np.any(np.isclose(r.x, lo) | np.isclose(r.x, hi))))


def evaluate(dev, meas, lam_s, cal_pumps, val_pumps, num=NUM):
    """Gain and ASE metrics of a device on the calibration and validation sets (full ASE model)."""
    pumps = sorted(set(cal_pumps) | set(val_pumps))
    r = predict(dev, pumps, lam_s, num=num)
    G = {p: r["G_dB"][i] for i, p in enumerate(pumps)}
    A = {p: r["ASE_dBm"][i] for i, p in enumerate(pumps)}
    NF = {p: r["NF_dB"][i] for i, p in enumerate(pumps)}
    mG = {p: meas[p]["G"] for p in pumps}
    mA = {p: meas[p]["ASE"] for p in pumps}
    return dict(
        gain_cal=metrics_table(G, mG, cal_pumps), gain_val=metrics_table(G, mG, val_pumps),
        ase_cal=metrics_table(A, mA, cal_pumps), ase_val=metrics_table(A, mA, val_pumps),
        gain_all=metrics_table(G, mG, pumps), ase_all=metrics_table(A, mA, pumps),
        spectra=dict(G={f"{p:g}": [round(float(v), 4) for v in G[p]] for p in pumps},
                     ASE={f"{p:g}": [round(float(v), 4) for v in A[p]] for p in pumps},
                     NF={f"{p:g}": [round(float(v), 4) for v in NF[p]] for p in pumps},
                     n2_mean={f"{p:g}": float(r["n2_mean"][i]) for i, p in enumerate(pumps)},
                     Pp_out_mW={f"{p:g}": float(r["Pp_out"][i] * 1e3) for i, p in enumerate(pumps)}))


CANDIDATES = [("pump_loss_dB",), ("out_loss_dB",), ("in_loss_dB",), ("coef_scale",), ("pump_scale",),
              ("l_bg_dB_per_m",), ("zeta_scale",), ("L_scale",),
              ("pump_loss_dB", "coef_scale"), ("pump_loss_dB", "out_loss_dB"), ("out_loss_dB", "coef_scale")]
# Model choice fixed before the held-out data are looked at:
#  * PRIMARY: pump-path loss. The source does not state the plane at which the pump power is given, and in this
#    pump-limited regime the nominal-card sensitivities (uncertainty_study) show that the gain responds to the coupled
#    pump power and hardly to a common coefficient scale.
#  * SECOND parameter: the common coefficient scale, because the source reports an 8 % scaling of its coefficients.
#    It is kept only if it is identifiable from the calibration fit alone (standard deviation below 0.02 after the
#    AR(1) correction for correlated residuals, and |correlation| with the first parameter below 0.9); held-out gain
#    and ASE are not used for this choice.
PRIMARY = ("pump_loss_dB",)
SECOND = ("pump_loss_dB", "coef_scale")


def second_parameter_kept(fit):
    """Identifiability rule for the second parameter (calibration fit only)."""
    return bool(len(fit["names"]) == 2 and fit["sd_ar1"][1] < 0.02 and abs(fit["corr"][0][1]) < 0.9)


def source_model_study(dev, meas, lam_s, pumps=PUMPS_MW):
    """Can the source's own model curves (Fig. 8 lines) be reproduced from its published parameters?
    Fits single parameters to the source's theory gain (all six pump powers); reports gain and ASE agreement."""
    y = np.concatenate([meas[p]["G_src"] for p in pumps])
    ya = np.concatenate([meas[p]["ASE_src"] for p in pumps])
    out = []
    for names in [(), ("coef_scale",), ("pump_loss_dB",), ("out_loss_dB",), ("pump_loss_dB", "coef_scale")]:
        if names:
            x0 = [PARAMS[n][1] for n in names]
            lo = [PARAMS[n][2] for n in names]
            hi = [PARAMS[n][3] for n in names]
            r = least_squares(lambda x: predict(_apply(dev, names, x), pumps, lam_s, num=NUM_FIT)["G_dB"].ravel() - y,
                              x0, bounds=(lo, hi), diff_step=1e-4, x_scale="jac")
            x = list(map(float, r.x))
        else:
            x = []
        p = predict(_apply(dev, names, x), pumps, lam_s, num=NUM)
        out.append(dict(parameters=list(names), values=x, gain=stats(p["G_dB"].ravel() - y),
                        ase=stats(p["ASE_dBm"].ravel() - ya)))
    # the source's statement: coefficients scaled up by 8 % (whether Fig. 7 shows nominal or scaled values is not
    # stated); effect of that scale on the nominal-card gain
    p0 = predict(dev, pumps, lam_s, num=NUM)["G_dB"]
    p8 = predict(_apply(dev, ("coef_scale",), [1.08]), pumps, lam_s, num=NUM)["G_dB"]
    return dict(fits=out, effect_of_8pct_scale_on_gain=stats(p8 - p0))


def identification_study(dev, meas, lam_s, cal_pumps, val_pumps, candidates=CANDIDATES):
    out = []
    for names in candidates:
        fit = identify(dev, names, meas, cal_pumps, lam_s)
        ev = evaluate(_apply(dev, names, fit["x"]), meas, lam_s, cal_pumps, val_pumps)
        out.append(dict(fit=fit, **{k: v for k, v in ev.items() if k != "spectra"}, spectra=ev["spectra"]))
        print(f"   {'+'.join(names):28s} x={np.round(fit['x'], 4)} sd={np.round(fit['sd_formal'], 4)} "
              f"cal {fit['cal_rmse_dB']:.3f} | val {ev['gain_val']['overall']['rmse']:.3f} "
              f"({ev['gain_val']['overall']['bias']:+.3f}) | ASE {ev['ase_all']['overall']['rmse']:.3f} "
              f"({ev['ase_all']['overall']['bias']:+.3f}) dB  [{fit['runtime_s']:.0f} s]", flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------
# L3: design decision with the nominal and the calibrated card
# ----------------------------------------------------------------------------------------------------------------
def design_study(devs, lam_s, target_dB=10.0, lengths=None, num=NUM_FIT, p_lo=1.0, p_hi=400.0, n_bisect=26):
    """For each EDF length, the reported pump power that gives the target mean gain (40 x -13 dBm) at the
    measurement plane, with the resulting peak-to-peak gain ripple, maximum internal NF and residual pump.
    Decision rule (stated before computing): the length with the smallest ripple; pump power at that length.
    devs: dict name -> Device."""
    lengths = np.round(np.arange(3.0, 10.001, 0.1), 2) if lengths is None else np.asarray(lengths, float)
    res = {}
    for name, dev in devs.items():
        lo = np.full(lengths.size, np.log(p_lo))
        hi = np.full(lengths.size, np.log(p_hi))
        for _ in range(n_bisect):               # batched bisection in log pump power
            mid = 0.5 * (lo + hi)
            g = predict(dev, np.exp(mid), lam_s, L=lengths, num=num)["G_dB"].mean(1)
            up = g > target_dB
            hi = np.where(up, mid, hi)
            lo = np.where(up, lo, mid)
        P = np.exp(0.5 * (lo + hi))
        r = predict(dev, P, lam_s, L=lengths, num=num)
        ok = np.abs(r["G_dB"].mean(1) - target_dB) < 0.01
        ripple = np.ptp(r["G_dB"], axis=1)
        k = int(np.argmin(np.where(ok, ripple, np.inf)))
        res[name] = dict(lengths_m=lengths.tolist(), pump_mW=P.tolist(), ripple_dB=ripple.tolist(),
                         nf_max_dB=r["NF_dB"].max(1).tolist(), nf_mean_dB=r["NF_dB"].mean(1).tolist(),
                         mean_gain_dB=r["G_dB"].mean(1).tolist(), reached=ok.tolist(),
                         pump_out_mW=(r["Pp_out"] * 1e3).tolist(),
                         optimum=dict(L_m=float(lengths[k]), pump_mW=float(P[k]), ripple_dB=float(ripple[k]),
                                      nf_max_dB=float(r["NF_dB"][k].max()),
                                      gain_spectrum_dB=[round(float(v), 4) for v in r["G_dB"][k]]))
    return res


def design_map(dev, lam_s, lengths, pumps_mW, num=NUM_FIT):
    """Mean gain and ripple on a grid of EDF length x reported pump power (batched)."""
    LL, PP = np.meshgrid(lengths, pumps_mW, indexing="xy")
    r = predict(dev, PP.ravel(), lam_s, L=LL.ravel(), num=num)
    return dict(lengths_m=list(map(float, lengths)), pumps_mW=list(map(float, pumps_mW)),
                mean_gain_dB=r["G_dB"].mean(1).reshape(PP.shape).tolist(),
                ripple_dB=np.ptp(r["G_dB"], axis=1).reshape(PP.shape).tolist(),
                nf_max_dB=r["NF_dB"].max(1).reshape(PP.shape).tolist())


# ----------------------------------------------------------------------------------------------------------------
# Calibrated card (L4 output)
# ----------------------------------------------------------------------------------------------------------------
def write_calibrated_card(card, fit, cal_eval, cal_pumps, val_pumps, alt_summary, path=CAL_CARD_PATH):
    """Copy of the nominal card plus the identified entries, each with its fit record and a history entry."""
    cal = copy.deepcopy(card)
    cal["card_id"] = "EDFA-PERIN2019-CAL"
    cal["title"] = card["title"] + " - calibrated by least-squares identification (layer L4)"
    cal["derived_from"] = dict(card_id=card["card_id"], file="cards/edfa_perin2019.json")
    today = "2026-11-17"
    data_cal = (f"[124] Fig. 8a (digitized), measured gain at {', '.join(f'{p:g}' for p in cal_pumps)} mW pump, "
                f"40 x -13 dBm; 120 points")
    nominal = {"pump_loss_dB": 0.0, "out_loss_dB": 0.0, "in_loss_dB": 0.0, "coef_scale": 1.0}
    meaning = {n: PARAMS[n][5] for n in fit["names"]}
    for n, x, sd, sd2 in zip(fit["names"], fit["x"], fit["sd_formal"], fit["sd_ar1"]):
        cal["entries"][n] = {
            "value": round(x, 4), "unit": PARAMS[n][4], "symbol": n, "category": "fitted",
            "source": {"ref": "this work", "locator": "e2e_perin.py, identify()"},
            "applies_to": meaning[n] + " (effective parameter of the measurement configuration of [124])",
            "fit": {"target": "measured gain of [124], Fig. 8a", "data": data_cal,
                    "method": "least squares (scipy.optimize.least_squares, trust-region reflective) on the "
                              "spectrally resolved Giles model with forward and backward ASE (e2e_perin.giles_ase, "
                              f"nz = {NUM_FIT.nz}, ASE bins {NUM_FIT.dlam} nm)",
                    "achieved": f"gain RMSE {fit['cal_rmse_dB']:.3f} dB on the calibration set",
                    "formal_sd": round(sd, 4), "sd_ar1_corrected": round(sd2, 4),
                    "residual_lag1_autocorrelation": round(fit["residual_lag1_autocorr"], 3),
                    "correlation": fit["corr"],
                    "not_unique": "the gain data are fitted almost equally well by other single parameters "
                                  "(see alternatives); the attribution is not identifiable from these data"},
            "verification": {"level": "V2",
                             "note": "fit residual recorded; the predictive test of the calibrated card on held-out "
                                     "data is recorded in 'validation' (card level)"},
            "history": [{"date": today, "old": nominal[n], "new": round(x, 4), "method": "least-squares identification",
                         "data": data_cal}]}
    cal["validation"] = {
        "type": "blind prediction with the calibrated card (layer L2 metrics), same experiment, digitized data",
        "held_out_gain": dict(pumps_mW=list(val_pumps), **cal_eval["gain_val"]["overall"]),
        "ase_not_fitted": dict(pumps_mW=sorted(set(cal_pumps) | set(val_pumps)), **cal_eval["ase_all"]["overall"]),
        "scope": "8 m EDF, 40 x -13 dBm, pump 18.1-114.9 mW, 1531-1562 nm; reference planes of [124] not stated",
        "level_note": "V3-type evidence for the card as a whole within this scope; it does not validate the "
                      "physical attribution of the identified parameter(s)"}
    cal["alternatives"] = alt_summary
    cal["history"] = card.get("history", []) + [{
        "date": today, "change": "added fitted entries " + ", ".join(fit["names"]),
        "old": {n: nominal[n] for n in fit["names"]}, "new": {n: round(x, 4) for n, x in zip(fit["names"], fit["x"])},
        "method": "least-squares identification (layer L4); the PINN of Section 7.3 is not used here",
        "data": data_cal}]
    cal["used_by"] = ["e2e_perin", "fig_e2e"]
    cards.validate(cal)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)
    return cal


# ----------------------------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------------------------
def _round(o, nd=7):
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, np.ndarray):
        return _round(o.tolist(), nd)
    if isinstance(o, (float, np.floating)):
        return float(f"{float(o):.{nd}g}")
    if isinstance(o, np.integer):
        return int(o)
    return o


def export_simulator_inputs(outdir=os.path.normpath(os.path.join(HERE, "..", "notes", "optisystem_perin2019"))):
    """Files for entering the same card into an independent simulator (see notes/optisystem_input_perin2019.md):
    Giles spectra in dB/m, effective cross sections for unit overlap, channel and pump table, and the open-model
    reference values (nominal and calibrated card) on the same wavelength grid."""
    dev, card = load_device()
    meas, _ = load_measurement()
    with open(RESULTS_PATH, encoding="utf-8") as f:
        R = json.load(f)
    os.makedirs(outdir, exist_ok=True)
    v = cards.values(card)
    db = 10 / np.log(10)
    hdr = "# Card EDFA-PERIN2019 (cards/edfa_perin2019.json); spectra digitized from [124] Fig. 7 (arXiv:1803.07905v3)\n"
    with open(os.path.join(outdir, "giles_coefficients_dB_per_m.csv"), "w", encoding="utf-8") as f:
        f.write(hdr + "# Route A: Giles absorption and gain coefficients; add the pump line separately: "
                      f"980 nm, alpha = {v['alpha_p'] * db:.4f} dB/m, g* = 0\nlambda_nm,alpha_dB_per_m,gstar_dB_per_m\n")
        for lam, a, g in zip(dev.lam_tab, dev.alpha_tab, dev.gstar_tab):
            f.write(f"{lam:.2f},{a * db:.5f},{g * db:.5f}\n")
    with open(os.path.join(outdir, "effective_cross_sections_unit_overlap.csv"), "w", encoding="utf-8") as f:
        f.write(hdr + f"# Route B: sigma_eff = coefficient / N_t with N_t = {v['n_t']:.4g} m^-3; valid ONLY if the "
                      "simulator applies an overlap of 1 (or the doped-area average intensity) at all wavelengths\n"
                      f"# pump: 980 nm, sigma_a,eff = {v['alpha_p'] / v['n_t']:.5e} m^2, sigma_e,eff = 0\n"
                      "lambda_nm,sigma_a_eff_m2,sigma_e_eff_m2\n")
        for lam, a, g in zip(dev.lam_tab, dev.alpha_tab, dev.gstar_tab):
            f.write(f"{lam:.2f},{a / v['n_t']:.5e},{g / v['n_t']:.5e}\n")
    lam_s = meas[PUMPS_MW[0]]["lam"]
    cal = next(e for e in R["L4"]["interleaved"] if e["fit"]["names"] == R["L4"]["chosen"])
    xcal = dict(zip(cal["fit"]["names"], cal["fit"]["x"]))
    with open(os.path.join(outdir, "operating_points.csv"), "w", encoding="utf-8") as f:
        f.write("# 40 channels at -13 dBm (EDF input, nominal case); wavelengths digitized from [124] Fig. 8\n"
                f"# calibrated card: pump at the EDF input = reported pump x 10^(-{xcal.get('pump_loss_dB', 0):.3f}/10); "
                f"alpha, g* and alpha_p x {xcal.get('coef_scale', 1.0):.4f}\n"
                "run,pump_reported_mW,pump_at_EDF_nominal_mW,pump_at_EDF_calibrated_mW\n")
        for i, p in enumerate(PUMPS_MW):
            f.write(f"P{i + 1},{p:g},{p:g},{p * 10 ** (-xcal.get('pump_loss_dB', 0) / 10):.3f}\n")
        f.write("channel,lambda_nm,power_dBm\n")
        for i, lam in enumerate(lam_s):
            f.write(f"{i + 1},{lam:.3f},{P_CH_DBM:.1f}\n")
    with open(os.path.join(outdir, "open_model_reference.csv"), "w", encoding="utf-8") as f:
        f.write("# Open ASE-resolved model (e2e_perin.giles_ase, nz = 200, ASE bins 0.25 nm), internal reference planes "
                "(EDF ends); ASE in 0.1 nm; NF internal\n"
                "pump_reported_mW,lambda_nm,G_nominal_dB,ASE_nominal_dBm,NF_nominal_dB,G_eq5_nominal_dB,"
                "G_calibrated_dB,ASE_calibrated_dBm,NF_calibrated_dB,G_measured_dB,ASE_measured_dBm\n")
        S = R["L2"]["spectra"]
        for p in PUMPS_MW:
            k = f"{p:g}"
            for j, lam in enumerate(lam_s):
                f.write(f"{p:g},{lam:.3f},{S['giles_nominal_G'][k][j]:.4f},{S['giles_nominal_ASE'][k][j]:.4f},"
                        f"{S['giles_nominal_NF'][k][j]:.4f},{S['saleh_nominal_G'][k][j]:.4f},"
                        f"{cal['spectra']['G'][k][j]:.4f},{cal['spectra']['ASE'][k][j]:.4f},"
                        f"{cal['spectra']['NF'][k][j]:.4f},{meas[p]['G'][j]:.3f},{meas[p]['ASE'][j]:.3f}\n")
    return outdir


def summarize(R):
    """Key numbers of a (rounded) results dictionary, each with its conditions."""
    L2, L4, L3 = R["L2"], R["L4"], R["L3"]
    ch = next(e for e in L4["interleaved"] if e["fit"]["names"] == L4["chosen"])
    one = next(e for e in L4["interleaved"] if e["fit"]["names"] == list(PRIMARY))
    mf = {m["load"]: m["overall"] for m in R["model_form"]}
    o_n, o_c = L3["results"]["nominal"]["optimum"], L3["results"]["calibrated"]["optimum"]
    alts = {k: v["optimum"] for k, v in L3["results"].items() if k.startswith("alt_")}
    num = R["numerics"]
    return {
        "conditions": "8 m EDF of [124], 980 nm forward pump 18.1-114.9 mW, 40 x -13 dBm 1531-1562 nm; measured data "
                      "digitized from arXiv:1803.07905v3 Fig. 8; model = spectrally resolved Giles model with forward "
                      "and backward ASE unless stated; metrics over 40 channels x stated pump powers, model minus data",
        "mccumber_300K": R["L1"]["mccumber_300K"]["full_table"],
        "mccumber_free_T": R["L1"]["mccumber_300K"]["free_T_fit"],
        "numerical_gain_max_dB": num.get("production", {}).get("gain_max_abs_dB"),
        "numerical_ase_max_dB": num.get("production", {}).get("ase_max_abs_dB"),
        "code_check_vs_eq5_max_dB": next((v["max_abs_dB"] for v in num.get("verification_vs_eq5", []) if v["nz"] == 200),
                                         None),
        "model_form_rmse_dB": {k: v["rmse"] for k, v in mf.items()},
        "parameter_sd_gain_dB": R["uncertainty"]["linear"]["gain_sd_rms_dB"],
        "parameter_sd_ase_dB": R["uncertainty"]["linear"]["ase_sd_rms_dB"],
        "nominal_vs_measured_gain": L2["giles_nominal_vs_measured"]["gain"]["overall"],
        "nominal_vs_measured_ase": L2["giles_nominal_vs_measured"]["ase"]["overall"],
        "saleh_vs_measured_gain": L2["saleh_nominal_vs_measured"]["gain"]["overall"],
        "source_model_vs_measured_gain": L2["source_model_vs_measured"]["gain"]["overall"],
        "source_model_vs_measured_ase": L2["source_model_vs_measured"]["ase"]["overall"],
        "nominal_vs_source_model_gain": L2["giles_nominal_vs_source_model"]["gain"]["overall"],
        "calibrated": {"parameters": ch["fit"]["names"], "values": ch["fit"]["x"], "sd_ar1": ch["fit"]["sd_ar1"],
                       "calibration_pumps_mW": L4["splits"]["interleaved"]["cal"],
                       "gain_cal": ch["gain_cal"]["overall"], "gain_heldout": ch["gain_val"]["overall"],
                       "ase_all_blind": ch["ase_all"]["overall"]},
        "one_parameter": {"parameters": one["fit"]["names"], "values": one["fit"]["x"], "sd_ar1": one["fit"]["sd_ar1"],
                          "gain_heldout": one["gain_val"]["overall"], "ase_all_blind": one["ase_all"]["overall"]},
        "design_target": f"{L3['target_mean_gain_dB']} dB mean gain, {L3['load']}, minimum peak-to-peak ripple",
        "design_nominal": o_n, "design_calibrated": o_c,
        "design_alternative_attributions": alts,
        "nominal_design_with_calibrated_card": L3["nominal_design_with_calibrated_card"]}


def main(quick=False):
    t_start = time.perf_counter()
    dev, card = load_device()
    meas, meas_meta = load_measurement()
    pumps = list(PUMPS_MW)
    lam_s = meas[pumps[0]]["lam"]
    res = dict(figure="Fig12_end_to_end", script="e2e_perin.py", date="2026-11-17",
               device=dict(card_id=card["card_id"], L_m=dev.L, zeta=dev.zeta, lambda_p_nm=dev.lam_p,
                           alpha_p_per_m=dev.alpha_p, n_channels=int(lam_s.size), p_ch_dBm=P_CH_DBM,
                           lambda_ch_nm=[round(float(x), 3) for x in lam_s], pumps_mW=pumps),
               reference_data=meas_meta)

    print("L1: McCumber test", flush=True)
    mc300 = mccumber_test(dev, 300.0)
    mc293 = mccumber_test(dev, 293.15)
    res["L1"] = dict(mccumber_300K={k: v for k, v in mc300.items() if k != "pointwise_lambda_x_nm"},
                     mccumber_293K={k: v for k, v in mc293.items() if k != "pointwise_lambda_x_nm"},
                     lam_tab=dev.lam_tab, alpha_tab=dev.alpha_tab, gstar_tab=dev.gstar_tab,
                     pointwise_lambda_x_300K=mc300["pointwise_lambda_x_nm"])

    print("Numerics study", flush=True)
    res["numerics"] = numerics_study(dev, lam_s) if not quick else {}

    print("L2: nominal card", flush=True)
    nom = predict(dev, pumps, lam_s, num=NUM)
    sal = predict(dev, pumps, lam_s, model="saleh")
    G_nom = {p: nom["G_dB"][i] for i, p in enumerate(pumps)}
    A_nom = {p: nom["ASE_dBm"][i] for i, p in enumerate(pumps)}
    G_sal = {p: sal["G_dB"][i] for i, p in enumerate(pumps)}
    mG = {p: meas[p]["G"] for p in pumps}
    mA = {p: meas[p]["ASE"] for p in pumps}
    sG = {p: meas[p]["G_src"] for p in pumps}
    sA = {p: meas[p]["ASE_src"] for p in pumps}
    res["L2"] = dict(
        giles_nominal_vs_measured=dict(gain=metrics_table(G_nom, mG, pumps), ase=metrics_table(A_nom, mA, pumps)),
        saleh_nominal_vs_measured=dict(gain=metrics_table(G_sal, mG, pumps)),
        saleh_vs_giles_nominal=dict(gain=metrics_table(G_sal, G_nom, pumps)),
        source_model_vs_measured=dict(gain=metrics_table(sG, mG, pumps), ase=metrics_table(sA, mA, pumps)),
        giles_nominal_vs_source_model=dict(gain=metrics_table(G_nom, sG, pumps), ase=metrics_table(A_nom, sA, pumps)),
        spectra=dict(giles_nominal_G={f"{p:g}": G_nom[p] for p in pumps},
                     giles_nominal_ASE={f"{p:g}": A_nom[p] for p in pumps},
                     giles_nominal_NF={f"{p:g}": nom["NF_dB"][i] for i, p in enumerate(pumps)},
                     saleh_nominal_G={f"{p:g}": G_sal[p] for p in pumps},
                     n2_mean={f"{p:g}": float(nom["n2_mean"][i]) for i, p in enumerate(pumps)},
                     Pp_out_mW={f"{p:g}": float(nom["Pp_out"][i] * 1e3) for i, p in enumerate(pumps)},
                     Pase_b_mW={f"{p:g}": float(nom["Pase_b"][i] * 1e3) for i, p in enumerate(pumps)}))

    print("Model-form study", flush=True)
    res["model_form"] = model_form_study(dev, lam_s)

    print("Uncertainty propagation", flush=True)
    unc = uncertainty_study(dev, lam_s, n_mc=300 if quick else 2000)
    res["uncertainty"] = {k: v for k, v in unc.items() if k not in ("gain_sd_dB", "ase_sd_dB")}
    res["uncertainty"]["gain_sd_dB"] = {f"{p:g}": unc["gain_sd_dB"][i] for i, p in enumerate(pumps)}
    res["uncertainty"]["ase_sd_dB"] = {f"{p:g}": unc["ase_sd_dB"][i] for i, p in enumerate(pumps)}

    print("Source-model reproduction", flush=True)
    res["source_model_reproduction"] = source_model_study(dev, meas, lam_s)

    print("L4: identification (fit on three pump powers, predict three + ASE)", flush=True)
    splits = {"interleaved": ((18.1, 45.8, 87.2), (31.9, 59.6, 114.9)),
              "extrapolation": ((18.1, 31.9, 45.8), (59.6, 87.2, 114.9))}
    res["L4"] = dict(method="least squares (scipy.optimize.least_squares); the PINN of Section 7.3 is not used",
                     primary=list(PRIMARY), second=list(SECOND),
                     second_rule="keep the second parameter only if formal sd < 0.02 and |corr| < 0.9 "
                                 "(calibration fit only)",
                     splits={k: dict(cal=list(v[0]), val=list(v[1])) for k, v in splits.items()})
    for sname, (cal_p, val_p) in splits.items():
        cands = CANDIDATES if sname == "interleaved" else [PRIMARY, ("out_loss_dB",), SECOND]
        if quick:
            cands = [PRIMARY, SECOND]
        print(f"  split {sname}: calibrate {cal_p}, predict {val_p}", flush=True)
        res["L4"][sname] = identification_study(dev, meas, lam_s, list(cal_p), list(val_p), cands)
    # nominal card on the same split (reference row)
    cal_p, val_p = splits["interleaved"]
    ev_nom = evaluate(dev, meas, lam_s, list(cal_p), list(val_p))
    res["L4"]["nominal_on_split"] = {k: v for k, v in ev_nom.items() if k != "spectra"}
    prim = next(e for e in res["L4"]["interleaved"] if tuple(e["fit"]["names"]) == PRIMARY)
    sec = next(e for e in res["L4"]["interleaved"] if tuple(e["fit"]["names"]) == SECOND)
    keep2 = second_parameter_kept(sec["fit"])
    res["L4"]["second_parameter_kept"] = keep2
    chosen = sec if keep2 else prim
    res["L4"]["chosen"] = chosen["fit"]["names"]
    # stability: refit of the chosen model on all six pump powers (not used for the card)
    res["L4"]["chosen_refit_all_six"] = identify(dev, tuple(chosen["fit"]["names"]), meas, pumps, lam_s)
    alt = [dict(parameters=e["fit"]["names"], values=e["fit"]["x"], cal_rmse_dB=e["fit"]["cal_rmse_dB"],
                heldout_gain_rmse_dB=e["gain_val"]["overall"]["rmse"], ase_rmse_dB=e["ase_all"]["overall"]["rmse"],
                ase_bias_dB=e["ase_all"]["overall"]["bias"]) for e in res["L4"]["interleaved"]]
    dev_cal = _apply(dev, tuple(chosen["fit"]["names"]), chosen["fit"]["x"])
    write_calibrated_card(card, chosen["fit"], chosen, list(cal_p), list(val_p), alt)
    res["L4"]["calibrated_card"] = "cards/edfa_perin2019_calibrated.json"

    print("L3: design decision", flush=True)
    devs = {"nominal": dev, "calibrated": dev_cal}
    for alt_names in (PRIMARY, SECOND, ("out_loss_dB",), ("l_bg_dB_per_m",)):
        e = next((e for e in res["L4"]["interleaved"] if tuple(e["fit"]["names"]) == alt_names), None)
        if e is not None and tuple(alt_names) != tuple(chosen["fit"]["names"]):
            devs["alt_" + "+".join(alt_names)] = _apply(dev, alt_names, e["fit"]["x"])
    design = design_study(devs, lam_s, num=NUM)
    # consequence of designing with the nominal card: the nominal optimum evaluated with the calibrated card
    o = design["nominal"]["optimum"]
    chk = predict(dev_cal, [o["pump_mW"]], lam_s, L=[o["L_m"]], num=NUM)
    res["L3"] = dict(target_mean_gain_dB=10.0, load="40 x -13 dBm", lengths_m=design["nominal"]["lengths_m"],
                     rule="minimize peak-to-peak gain ripple over the 40 channels at 10 dB mean gain "
                          "(measurement plane); report pump power and maximum internal NF at that length",
                     results=design,
                     nominal_design_with_calibrated_card=dict(
                         L_m=o["L_m"], pump_mW=o["pump_mW"], mean_gain_dB=float(chk["G_dB"].mean()),
                         ripple_dB=float(np.ptp(chk["G_dB"])), nf_max_dB=float(chk["NF_dB"].max())))
    lengths_map = np.arange(3.0, 12.01, 0.25)
    pumps_map = np.geomspace(10.0, 300.0, 36)
    res["L3"]["map_calibrated"] = design_map(dev_cal, lam_s, lengths_map, pumps_map)
    res["L3"]["map_nominal"] = design_map(dev, lam_s, lengths_map, pumps_map)

    res["runtime_s"] = time.perf_counter() - t_start
    out = _round(res)
    out = {"summary": summarize(out), **out}
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {RESULTS_PATH} ({res['runtime_s']:.0f} s)")
    return res


if __name__ == "__main__":
    import sys
    if "--summary" in sys.argv:           # add or refresh the summary block of an existing results file
        with open(RESULTS_PATH, encoding="utf-8") as f:
            R = json.load(f)
        R.pop("summary", None)
        R = {"summary": summarize(R), **R}
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(R, f, indent=1, ensure_ascii=False)
        print(json.dumps(R["summary"], indent=1))
    elif "--export" in sys.argv:          # simulator input files from an existing results file
        print("wrote", export_simulator_inputs())
    else:
        main(quick="--quick" in sys.argv)
