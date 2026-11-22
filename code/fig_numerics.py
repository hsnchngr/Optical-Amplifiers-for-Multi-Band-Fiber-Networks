"""Figure S1 (Supporting Information): numerical procedure, convergence and code verification; run-time table.

  a) flowchart of the solution procedure of the steady-state reference models;
  b) convergence of the forward-backward sweep iteration of the double-pass EDFA with ASE (edfa_multipass.solve);
  c) grid convergence of the sweep scheme (RK2) for the single- and double-pass EDFA: spontaneous emission switched
     off and compared with the exact photon-balance solution, and with ASE compared with a Richardson-extrapolated
     fine-grid reference.
The former panel d (code-verification residual of the EDFA ODE against the Saleh closed form, Equation (5), versus
the integrator tolerance) was removed because Figure 11b shows the same study (fig_intercomparison.py); its numbers
stay in the result file (key edfa_ode_tolerance), next to the solve_bvp tolerance check (key raman_bvp_tolerance).
Run-time table: median wall-clock time per evaluation of every reference model on this machine (one thread), plus
the least-squares identification of C_R with the collocation solver (conventional baseline for the PINN inverse
problem; run time measured here, accuracy statistics over the 2000 noise realizations of fig09_pinn.py read from
../figures/FigS3_pinn_results.json, the values quoted in the text) and the PINN rows (training and inference times
measured by fig09_pinn.py, read from the same file).

Outputs : ../figures/FigS1_numerics.png/.pdf, ../figures/FigS1_numerics_results.json,
          ../figures/FigS1_numerics_runtime_table.md
Usage   : python fig_numerics.py [--no-runtime] [--no-pin] [--quick] [--table-only] [--plot-only]
          (about 10 min on the machine of Table S1; --table-only rewrites Table S1 from the stored result file;
          --plot-only redraws Figure S1 from the stored result file and writes nothing else)
All parameters come from the card files through models.py (cards EDFA-GD91-T1, SOA-BULK-REF, RAMAN-G652).
"""
import os
import sys

# One thread for every numerical library; must be set before NumPy is imported.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
for _v in THREAD_VARS:
    os.environ[_v] = "1"

import argparse
import copy
import datetime
import gc
import hashlib
import inspect
import json
import platform
import re
import time

import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import brentq, least_squares

import style
import models
import edfa_multipass as em
from models import (EDFA, RAMAN, h, c, dbm2w, w2dbm, edfa_ode, edfa_saleh, edfa_inversion, soa_numerical,
                    soa_analytic, raman_numerical, raman_analytic, raman_CR)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
NAME = "FigS1_numerics"

# PINN rows of Table S1 (Section 7.3, Figure S3): training and inference times measured by fig09_pinn.py (PyTorch) and
# read from ../figures/FigS3_pinn_results.json (pinn_measured below). Fallback, used only if no such result file exists:
# the training time of the submitted code archive (Table 5 of the submitted manuscript; one run, forward and inverse
# problems with 6000 Adam iterations each, thread count not set, hardware not recorded) and no inference time.
PINN_TRAIN_S_PREVIOUS = 32.227
PINN_INFERENCE_S = None          # fallback: no inference time without a result file of fig09_pinn.py
PINN_KEYS = ("train_seconds_forward", "train_seconds_inverse", "inference_seconds_1", "inference_seconds_401")
PINN_RESULTS = "FigS3_pinn_results.json"


def ls_reference():
    """Accuracy of the least-squares identification of C_R over the noise realizations of fig09_pinn.py (keys
    least_squares/CR_err and cramer_rao/CR_only of FigS3_pinn_results.json): the values quoted in the text and in
    Table S1 (2000 realizations). None if the file is missing; Table S1 then reports the realizations of this script."""
    try:
        with open(os.path.join(FIGDIR, PINN_RESULTS), encoding="utf-8") as f:
            d = json.load(f)
        e = d["least_squares"]["CR_err"]
        return dict(error_sd_pct=float(e["std"]), error_mean_pct=float(e["mean"]), realizations=int(e["n"]),
                    cramer_rao_pct=float(d["cramer_rao"]["CR_only"]), source=f"{PINN_RESULTS} (fig09_pinn.py)")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pinn_measured():
    """Measured PINN costs from an instrumented re-run of fig09_pinn.py, if available; else None."""
    # FigS3_pinn_results.json: output name of the revision-3 fig09_pinn.py (integration fix; the older names are kept)
    for name in ("FigS3_pinn_results.json", "Fig14_pinn_results.json", "Fig09_results.json"):
        path = os.path.join(FIGDIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if all(k in d for k in PINN_KEYS):
            return dict({k: float(d[k]) for k in PINN_KEYS}, source=name,
                        **{k: d[k] for k in ("cpu", "torch_version", "torch_threads") if k in d})
    return None


# ----------------------------------------------------------------------------------------------------------------
# Machine description and timing utilities
# ----------------------------------------------------------------------------------------------------------------
def _cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return platform.processor()


def _blas():
    try:
        cfg = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{cfg.get('name')} {cfg.get('version')}"
    except Exception:
        return "unknown"


def _pin_performance_cores():
    """Windows only: restrict this process to the logical processors of the highest efficiency class, i.e. the
    performance cores of hybrid CPUs, so that the single-threaded timings do not depend on whether the scheduler
    places the process on a performance or an efficiency core. Affects this process only. Returns a description."""
    if sys.platform != "win32":
        return dict(pinned=False, reason="pinning implemented for Windows only")
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        n = wintypes.ULONG(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
        buf = (ctypes.c_ubyte * n.value)()
        if not k32.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0):
            return dict(pinned=False, reason="GetSystemCpuSetInformation failed")
        raw, off, cls = bytes(buf), 0, {}
        while off + 20 <= len(raw):
            size = int.from_bytes(raw[off:off + 4], "little")
            if size == 0:
                break
            if int.from_bytes(raw[off + 4:off + 8], "little") == 0:          # CpuSetInformation record
                group, lp, eff = int.from_bytes(raw[off + 12:off + 14], "little"), raw[off + 14], raw[off + 18]
                if group == 0:
                    cls[lp] = eff
            off += size
        counts = {int(e): sum(1 for v in cls.values() if v == e) for e in sorted(set(cls.values()))}
        if len(counts) < 2:
            return dict(pinned=False, reason="all logical processors have the same efficiency class",
                        efficiency_classes=counts)
        top = max(cls.values())
        lps = sorted(lp for lp, e in cls.items() if e == top)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        ok = bool(k32.SetProcessAffinityMask(k32.GetCurrentProcess(), sum(1 << lp for lp in lps)))
        return dict(pinned=ok, logical_processors=lps, efficiency_class=int(top), efficiency_classes=counts)
    except Exception as exc:                                                    # pragma: no cover
        return dict(pinned=False, reason=f"{type(exc).__name__}: {exc}")


def _system_times():
    """(idle, total) CPU time of the whole system, for the average utilization during the benchmark."""
    try:
        if sys.platform == "win32":
            import ctypes
            idle, kern, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
            return idle.value, kern.value + user.value            # kernel time includes idle time
        if os.path.exists("/proc/stat"):
            f = [float(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
    except Exception:
        pass
    return None


def _os_label(s=None):
    """Readable operating-system name; Python 3.11 reports Windows 11 as 'Windows-10-10.0.<build>-...'."""
    s = platform.platform() if s is None else s
    m = re.match(r"Windows-10-10\.0\.(\d+)", s)
    if m and int(m.group(1)) >= 22000:
        return f"Windows 11 (build {m.group(1)})"
    return s


def machine_info():
    return dict(cpu=_cpu_name(), logical_processors=os.cpu_count(), os=_os_label(),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                matplotlib=mpl.__version__, blas=_blas(), thread_env={v: os.environ.get(v) for v in THREAD_VARS})


def _sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


def bench(fn, min_sample_s=0.05, repeats=15, max_total_s=20.0):
    """timeit-style benchmark: one warm-up call, then `repeats` samples of `number` calls each (number chosen so that a
    sample lasts at least min_sample_s), garbage collector disabled while timing. Times are per call."""
    t0 = time.perf_counter(); fn(); t1 = time.perf_counter() - t0               # warm-up call (not counted)
    number = max(1, int(np.ceil(min_sample_s / max(t1, 1e-7))))
    repeats = int(max(3, min(repeats, max_total_s / max(t1 * number, 1e-9))))
    wall, cpu = [], []
    was_enabled = gc.isenabled(); gc.disable()
    try:
        for _ in range(repeats):
            c0, w0 = time.process_time(), time.perf_counter()
            for _ in range(number):
                fn()
            wall.append((time.perf_counter() - w0) / number); cpu.append((time.process_time() - c0) / number)
    finally:
        if was_enabled:
            gc.enable()
    w = np.array(wall)
    return dict(median_s=float(np.median(w)), min_s=float(w.min()), q25_s=float(np.percentile(w, 25)),
                q75_s=float(np.percentile(w, 75)), cpu_over_wall=float(np.sum(cpu) / np.sum(wall)),
                samples=repeats, calls_per_sample=number)


def _utilization(s0, s1):
    if s0 and s1 and (s1[1] - s0[1]) > 0:
        return float(1.0 - (s1[0] - s0[0]) / (s1[1] - s0[1]))
    return None


def bench_checked(fn, attempts=3, min_ratio=0.9, **kw):
    """bench() repeated (at most `attempts` times) while the process received less than min_ratio of a CPU during the
    measurement (process CPU time / wall-clock time), i.e. while it was descheduled by other load; the attempt with the
    highest ratio is kept. Also records the average system CPU utilization during the kept attempt."""
    best = None
    for k in range(attempts):
        s0 = _system_times(); r = bench(fn, **kw); s1 = _system_times()
        r["system_cpu_utilization"] = _utilization(s0, s1); r["attempt"] = k + 1
        if best is None or r["cpu_over_wall"] > best["cpu_over_wall"]:
            best = r
        if r["cpu_over_wall"] >= min_ratio:
            break
    return best


# ----------------------------------------------------------------------------------------------------------------
# Exact references
# ----------------------------------------------------------------------------------------------------------------
def photon_balance_gain_db(Pp, Pin, L, double_pass, R_db=-1.0, p=EDFA):
    """Exact signal gain [dB] of the ASE-free, loss-free two-level model (Equations (2)-(5)), forward pump.
    For every beam k, ln G_k = (alpha_k + g_k) X - alpha_k L with X = (Q_in - Q_out)/xi, independently of the
    propagation direction, where Q_in and Q_out are the photon fluxes entering and leaving the fiber through both
    ends. Double pass: the mirror at z = L returns R G_s Pin as backward signal, which leaves at z = 0 with R G_s^2 Pin,
    so xi X = Q_p (1 - G_p) + Q_s (1 - G_s)(1 + R G_s); single pass: R = 0."""
    nup, nus = c / p["lam_p"], c / p["lam_s"]
    Qp, Qs = Pp / (h * nup), Pin / (h * nus)
    ap, gp, as_, gs, xi = p["alpha_p"], p["g_p"], p["alpha_s"], p["g_s"], p["xi"]
    R = 10 ** (R_db / 10) if double_pass else 0.0

    def f(X):                                   # monotonically increasing in X, f(0) < 0 < f(L)
        Gp, Gs = np.exp((ap + gp) * X - ap * L), np.exp((as_ + gs) * X - as_ * L)
        return xi * X - (Qp * (1 - Gp) + Qs * (1 - Gs) * (1 + R * Gs))

    X = brentq(f, 0.0, L, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
    Gs = np.exp((as_ + gs) * X - as_ * L)
    return float(10 * np.log10(R * Gs ** 2 if double_pass else Gs))


def sweep_solve(Pp, Pin, L, double_pass, N=800, tol=1e-7, max_iter=400, spontaneous=True):
    """edfa_multipass.solve, optionally with the spontaneous-emission source term switched off (ASE band width 0)."""
    kw = dict(double_pass=double_pass, N=N, tol=tol, max_iter=max_iter)
    if spontaneous:
        return em.solve(Pp, Pin, L, **kw)
    with np.errstate(divide="ignore", invalid="ignore"):          # noise factor is undefined without ASE (0/0)
        if "ase_bw_nm" in inspect.signature(em.solve).parameters:
            return em.solve(Pp, Pin, L, ase_bw_nm=0.0, **kw)
        old = em.P["dnu_ase"]
        try:
            em.P["dnu_ase"] = 0.0
            return em.solve(Pp, Pin, L, **kw)
        finally:
            em.P["dnu_ase"] = old


def edfa_ode_output(Pp0, Ps0, L, rtol, atol=1e-16, method="LSODA", p=EDFA):
    """Output powers of the forward-pumped, ASE-free EDFA: same right-hand side as models.edfa_ode, but with the
    integrator method and tolerances as arguments (models.edfa_ode fixes LSODA, rtol 1e-10, atol 1e-16 W)."""
    nu = c / np.array([p["lam_p"], p["lam_s"]])
    alpha = np.array([p["alpha_p"], p["alpha_s"]]); g = np.array([p["g_p"], p["g_s"]])

    def rhs(z, P):
        return ((alpha + g) * edfa_inversion(P, alpha, g, nu, p["xi"]) - alpha) * P

    s = solve_ivp(rhs, (0, L), [Pp0, Ps0], rtol=rtol, atol=atol, method=method)
    return s.y[:, -1], s.nfev


# ----------------------------------------------------------------------------------------------------------------
# Panels b and c: computations; tolerance studies (result file only)
# ----------------------------------------------------------------------------------------------------------------
OP_SWEEP = dict(Pp=0.03, Pin_dBm=-20.0, L=10.0)          # operating point of the double-pass example (Section 4.1)


def sweep_convergence(quick=False):
    Pp, Pin, L = OP_SWEEP["Pp"], dbm2w(OP_SWEEP["Pin_dBm"]), OP_SWEEP["L"]
    Kmax = 60 if quick else 80
    ref = sweep_solve(Pp, Pin, L, True, tol=0.0, max_iter=Kmax)                 # history down to round-off
    hist = ref["hist"]                                                          # hist[j]: |change| at sweep j + 2
    G_inf, NF_inf = 10 * np.log10(ref["G"]), 10 * np.log10(ref["NF"])
    ks = list(range(1, 13)) + list(range(15, 49 if not quick else 31, 3))
    dev, dev_nf, sign = {}, {}, {}
    for k in ks:                                                                # output after exactly k sweeps
        r = sweep_solve(Pp, Pin, L, True, tol=0.0, max_iter=k)
        dev[k] = float(abs(10 * np.log10(r["G"]) - G_inf)); sign[k] = int(np.sign(10 * np.log10(r["G"]) - G_inf))
        dev_nf[k] = float(abs(10 * np.log10(r["NF"]) - NF_inf))
    stop = sweep_solve(Pp, Pin, L, True)                                        # default criterion, 1e-7 dB
    ratios = hist[1:] / np.where(hist[:-1] > 0, hist[:-1], np.nan)
    asym = ratios[15:35]
    sp = sweep_solve(Pp, Pin, L, False, tol=0.0, max_iter=12)
    sp_stop = sweep_solve(Pp, Pin, L, False)
    return dict(
        config=dict(model="edfa_multipass.solve, double pass, FRM -1 dB, lumped 30 nm ASE band, N = 800, RK2",
                    pump_W=Pp, pump_nm=1480, input_dBm=OP_SWEEP["Pin_dBm"], signal_nm=1550, L_m=L,
                    card="EDFA-GD91-T1"),
        change_between_sweeps_dB=[float(x) for x in hist], change_at_sweep=list(range(2, len(hist) + 2)),
        deviation_from_converged_dB=dev, deviation_sign=sign, NF_deviation_from_converged_dB=dev_nf,
        converged_gain_dB=float(G_inf), converged_NF_dB=float(NF_inf), reference_sweeps=Kmax,
        sweeps_to_stop=int(stop["sweeps"]), gain_at_stop_dB=float(10 * np.log10(stop["G"])),
        NF_at_stop_dB=float(10 * np.log10(stop["NF"])),
        deviation_at_stop_dB=float(abs(10 * np.log10(stop["G"]) - G_inf)),
        NF_deviation_at_stop_dB=float(abs(10 * np.log10(stop["NF"]) - NF_inf)),
        asymptotic_contraction_ratio=float(np.nanmedian(asym)),
        convergence_type="oscillatory: successive outputs alternate around the converged value",
        single_pass=dict(change_between_sweeps_dB=[float(x) for x in sp["hist"]],
                         sweeps_to_stop=int(sp_stop["sweeps"]), gain_dB=float(10 * np.log10(sp_stop["G"])),
                         NF_dB=float(10 * np.log10(sp_stop["NF"]))))


def grid_convergence(quick=False):
    Pp, Pin, L = OP_SWEEP["Pp"], dbm2w(OP_SWEEP["Pin_dBm"]), OP_SWEEP["L"]
    Ns = [25, 50, 100, 200, 400, 800] + ([] if quick else [1600])
    Nfine = [1600, 3200] if quick else [3200, 6400]
    out = dict(config=dict(model="edfa_multipass.solve (RK2 forward-backward sweeps), N = number of steps",
                           pump_W=Pp, input_dBm=OP_SWEEP["Pin_dBm"], L_m=L, sweep_tol_dB=1e-11, card="EDFA-GD91-T1"),
               N=Ns)
    for dp, key in ((False, "SP"), (True, "DP")):
        exact = photon_balance_gain_db(Pp, Pin, L, dp)
        g0 = [10 * np.log10(sweep_solve(Pp, Pin, L, dp, N=n, tol=1e-11, spontaneous=False)["G"]) for n in Ns]
        err0 = [float(x - exact) for x in g0]
        runs = {n: sweep_solve(Pp, Pin, L, dp, N=n, tol=1e-11) for n in sorted(set(Ns + Nfine))}
        gA = {n: 10 * np.log10(r["G"]) for n, r in runs.items()}
        nfA = {n: 10 * np.log10(r["NF"]) for n, r in runs.items()}
        n1, n2 = Nfine
        rich = gA[n2] + (gA[n2] - gA[n1]) / 3.0
        rich_nf = nfA[n2] + (nfA[n2] - nfA[n1]) / 3.0
        nm = Ns[-1]
        order_fine = float(np.log2(abs((gA[nm] - gA[n1]) / (gA[n1] - gA[n2])))) if nm < n1 else None
        out[key] = dict(
            exact_noASE_gain_dB=exact, noASE_gain_dB=[float(x) for x in g0], noASE_error_dB=err0,
            noASE_observed_order=[float(np.log2(abs(err0[i] / err0[i + 1]))) for i in range(len(Ns) - 1)],
            ASE_gain_dB={int(n): float(v) for n, v in gA.items()},
            ASE_NF_dB={int(n): float(v) for n, v in nfA.items()},
            ASE_richardson_reference_gain_dB=float(rich), ASE_richardson_reference_NF_dB=float(rich_nf),
            ASE_richardson_from=list(Nfine), ASE_observed_order_fine=order_fine,
            ASE_error_dB=[float(gA[n] - rich) for n in Ns], ASE_NF_error_dB=[float(nfA[n] - rich_nf) for n in Ns],
            ASE_error_vs_finest_dB=[float(gA[n] - gA[n2]) for n in Ns],
            ASE_observed_order=[float(np.log2(abs((gA[Ns[i]] - rich) / (gA[Ns[i + 1]] - rich)))) for i in range(len(Ns) - 1)],
            sweeps={int(n): int(r["sweeps"]) for n, r in runs.items()},
            ASE_effect_on_gain_dB=float(rich - exact))
    # other input powers of the double-pass example (with ASE, Richardson reference from 1600 and 3200 steps)
    out["other_inputs"] = {}
    for pin_dbm in ([] if quick else [-40.0, 0.0]):
        for dp, key in ((False, "SP"), (True, "DP")):
            g = {n: sweep_solve(Pp, dbm2w(pin_dbm), L, dp, N=n, tol=1e-11) for n in (400, 800, 1600, 3200)}
            gd = {n: 10 * np.log10(r["G"]) for n, r in g.items()}
            nf = {n: 10 * np.log10(r["NF"]) for n, r in g.items()}
            rich = gd[3200] + (gd[3200] - gd[1600]) / 3.0
            rich_nf = nf[3200] + (nf[3200] - nf[1600]) / 3.0
            out["other_inputs"][f"{key}_{pin_dbm:+.0f}dBm"] = dict(
                gain_dB=float(rich), NF_dB=float(rich_nf),
                error_N400_dB=float(gd[400] - rich), error_N800_dB=float(gd[800] - rich),
                NF_error_N800_dB=float(nf[800] - rich_nf),
                observed_order_800_1600_3200=float(np.log2(abs((gd[800] - gd[1600]) / (gd[1600] - gd[3200])))))
    return out


RTOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
METHODS = ("LSODA", "RK45", "DOP853")


def tolerance_study():
    """Supplementary check (result file only; formerly panel d): code-verification residual of the EDFA ODE against the
    Saleh closed form versus the relative tolerance, atol fixed at 1e-16 W. Figure 11b shows the same study with atol
    proportional to rtol."""
    Pp0, Ps0 = 0.100, dbm2w(-20)
    Lk = np.arange(1, 21, 1.0)                                  # lengths of the model inter-comparison figure (1-20 m)
    an = np.array([edfa_saleh(Pp0, Ps0, l) for l in Lk])
    paper = np.array([edfa_ode(Pp0, Ps0, l, npts=2)[2][-1] for l in Lk])
    mine = np.array([edfa_ode_output(Pp0, Ps0, l, 1e-10)[0][1] for l in Lk])
    res = dict(note="result file only (formerly Figure S1d); Figure 11b shows the same study with atol proportional "
                    "to rtol",
               config=dict(model="forward-pumped two-level EDFA without ASE and background loss (models.edfa_ode "
                                 "right-hand side) vs Saleh closed form models.edfa_saleh", pump_W=Pp0, pump_nm=1480,
                           input_dBm=-20.0, signal_nm=1550, lengths_m=Lk.tolist(), atol_W=1e-16,
                           metric="RMS over the 20 lengths of the signal output difference [dB]", card="EDFA-GD91-T1"),
               rtol=RTOLS,
               setting_of_models_edfa_ode=dict(method="LSODA", rtol=1e-10, atol=1e-16,
                                               rms_dB=float(np.sqrt(np.mean((w2dbm(paper) - w2dbm(an[:, 1])) ** 2))),
                                               max_dB=float(np.max(np.abs(w2dbm(paper) - w2dbm(an[:, 1]))))),
               wrapper_vs_models_edfa_ode_max_dB=float(np.max(np.abs(w2dbm(mine) - w2dbm(paper)))))
    for m in METHODS:
        rms, mx_s, mx_p, nfev = [], [], [], []
        for rt in RTOLS:
            out = [edfa_ode_output(Pp0, Ps0, l, rt, method=m) for l in Lk]
            num = np.array([o[0] for o in out])
            es, ep = w2dbm(num[:, 1]) - w2dbm(an[:, 1]), w2dbm(num[:, 0]) - w2dbm(an[:, 0])
            rms.append(float(np.sqrt(np.mean(es ** 2)))); mx_s.append(float(np.max(np.abs(es))))
            mx_p.append(float(np.max(np.abs(ep)))); nfev.append(float(np.mean([o[1] for o in out])))
        res[m] = dict(rms_signal_dB=rms, max_signal_dB=mx_s, max_pump_dB=mx_p, mean_rhs_evaluations=nfev,
                      rms_over_rtol=[r / t for r, t in zip(rms, RTOLS)])
    return res


def raman_bvp_tolerance():
    """Supplementary check (result file only): error of the collocation solution of the counter-pumped Raman span
    (same equations, boundary conditions, initial mesh and initial guess as models.raman_numerical) versus the
    tolerance of solve_bvp, relative to a shooting solution (DOP853, rtol 1e-13, root finding on the pump power that
    leaves the span at z = 0). Cases of the model intercomparison: one -10 dBm signal and a +16 dBm lumped load."""
    p = RAMAN
    CR, r, L = raman_CR(p), p["lam_s"] / p["lam_p"], p["L"]
    z = np.linspace(0, L, 401)

    def f(zz, y):
        return np.vstack([CR * np.exp(y[1]) - p["alpha_s"], p["alpha_p"] + r * CR * np.exp(y[0])])

    def shoot(Pp0, Ps0):
        def end(lnpp0):
            s = solve_ivp(lambda zz, y: f(zz, y[:, None]).ravel(), (0, L), [np.log(Ps0), lnpp0], method="DOP853",
                          rtol=1e-13, atol=1e-13, dense_output=True)
            return s
        x = brentq(lambda v: end(v).y[1, -1] - np.log(Pp0), np.log(Pp0) - 10 * p["alpha_p"] * L, np.log(Pp0),
                   xtol=1e-14, rtol=4 * np.finfo(float).eps)
        return end(x).sol(z)

    out = dict(config=dict(span_km=L / 1e3, pump_W=0.5, pump_nm=1450, signal_nm=1550, nodes_initial=401,
                           reference="shooting, DOP853 rtol = atol = 1e-13 on ln P, brentq on ln Pp(0)",
                           card="RAMAN-G652"), tol=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    for lab, Ps0 in (("-10dBm", dbm2w(-10.0)), ("+16dBm", dbm2w(16.0))):
        ref = shoot(0.5, Ps0)
        ys, yp = raman_analytic(0.5, Ps0, z)
        res = dict(net_gain_dB_reference=float(10 / np.log(10) * (ref[0, -1] - np.log(Ps0))), net_gain_error_dB=[],
                   max_signal_error_dB=[], max_pump_error_dB=[], final_nodes=[], status=[])
        for tol in out["tol"]:
            sol = solve_bvp(f, lambda ya, yb: np.array([ya[0] - np.log(Ps0), yb[1] - np.log(0.5)]), z,
                            np.vstack([np.log(ys), np.log(yp)]), tol=tol, max_nodes=100000)
            y = sol.sol(z)
            res["net_gain_error_dB"].append(float(10 / np.log(10) * (y[0, -1] - ref[0, -1])))
            res["max_signal_error_dB"].append(float(10 / np.log(10) * np.max(np.abs(y[0] - ref[0]))))
            res["max_pump_error_dB"].append(float(10 / np.log(10) * np.max(np.abs(y[1] - ref[1]))))
            res["final_nodes"].append(int(sol.x.size)); res["status"].append(int(sol.status))
        zz, Ps_m, Pp_m, st = raman_numerical(0.5, Ps0)                 # production setting (tol 1e-8)
        res["models_raman_numerical_net_gain_error_dB"] = float(w2dbm(Ps_m[-1]) - w2dbm(np.exp(ref[0, -1])))
        out[lab] = res
    return out


# ----------------------------------------------------------------------------------------------------------------
# Raman C_R identification: conventional least-squares baseline for the inverse problem of the PINN (Section 7.3)
# ----------------------------------------------------------------------------------------------------------------
def raman_identification(n_real=200, quick=False):
    """Same synthetic data as fig09_pinn.py: 50 km span, 500 mW counter-propagating pump at 1450 nm, +16 dBm lumped
    signal at 1550 nm, 8 samples of the signal power at the grid indices np.linspace(20, 400, 8) of the 401-point
    solution, Gaussian noise of 0.05 dB, attenuation coefficients and launched powers known exactly, initial guess
    0.5 x the true C_R. Estimator: scipy.optimize.least_squares (trust-region reflective, 2-point Jacobian) on the
    dB residuals, with models.raman_numerical (solve_bvp) as the forward model."""
    if quick:
        n_real = 30
    Pp0, Ps0, sig = 0.5, dbm2w(16.0), 0.05
    z, Ps, _, status = raman_numerical(Pp0, Ps0)
    idx = np.linspace(20, len(z) - 1, 8).astype(int)
    clean = w2dbm(Ps[idx])
    ncall = [0]

    def model_db(theta):
        ncall[0] += 1
        p = copy.deepcopy(RAMAN); p["gR"] = RAMAN["gR"] * float(theta[0])
        return w2dbm(raman_numerical(Pp0, Ps0, p)[1][idx])

    hstep = 1e-4                                                     # Cramer-Rao bound (linearized, unbiased)
    J = (model_db([1 + hstep]) - model_db([1 - hstep])) / (2 * hstep)        # dB per unit change of ln C_R
    crb_pct = float(100 / np.sqrt(np.sum(J ** 2) / sig ** 2))
    crb_out_pct = float(100 * sig / abs(J[-1]))

    def fit(y, x0=0.5):
        ncall[0] = 0
        t0 = time.perf_counter()
        r = least_squares(lambda th: model_db(th) - y, x0=[x0], bounds=([0.05], [5.0]), xtol=1e-10, ftol=1e-10)
        return 100 * (r.x[0] - 1), time.perf_counter() - t0, ncall[0], r.nfev, r.njev

    rng = np.random.default_rng(0)
    gc.disable()
    try:
        fit(clean + rng.normal(0, sig, len(idx)))                    # warm-up (not counted)
        rows = [fit(clean + rng.normal(0, sig, len(idx))) for _ in range(n_real)]
    finally:
        gc.enable()
    err = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows]); calls = np.array([r[2] for r in rows])
    np.random.seed(0)                                                 # noise realization of fig09_pinn.py
    legacy = fit(clean + np.random.normal(0, sig, len(idx)))
    return dict(
        config=dict(span_km=RAMAN["L"] / 1e3, pump_W=Pp0, pump_nm=1450, signal_dBm=16.0, signal_nm=1550,
                    samples_km=[float(x) for x in z[idx] / 1e3], noise_dB=sig, initial_guess=0.5,
                    estimator="scipy.optimize.least_squares (trf, 2-point Jacobian), xtol = ftol = 1e-10",
                    forward_model="models.raman_numerical (solve_bvp, tol 1e-8, 401 nodes)",
                    known_exactly="alpha_s, alpha_p, launched pump and signal powers", card="RAMAN-G652",
                    bvp_status=int(status)),
        CR_true_per_W_km=float(raman_CR() * 1e3), cramer_rao_pct=crb_pct, cramer_rao_output_sample_only_pct=crb_out_pct,
        sensitivity_dB_per_lnCR=[float(x) for x in J], realizations=int(n_real),
        error_mean_pct=float(err.mean()), error_sd_pct=float(err.std(ddof=1)), error_max_abs_pct=float(np.abs(err).max()),
        fraction_within_0p5pct=float(np.mean(np.abs(err) < 0.5)),
        time_median_s=float(np.median(tt)), time_q25_s=float(np.percentile(tt, 25)), time_q75_s=float(np.percentile(tt, 75)),
        bvp_solves_median=int(np.median(calls)), bvp_solves_range=[int(calls.min()), int(calls.max())],
        legacy_noise_realization=dict(error_pct=float(legacy[0]), time_s=float(legacy[1]), bvp_solves=int(legacy[2]),
                                      note="np.random.seed(0); np.random.normal(0, 0.05, 8), as in fig09_pinn.py; "
                                           "the PINN estimate for this realization was -0.49 % (previous result file)"))


def numpy_mlp_proxy():
    """For orientation only (not a PINN measurement): forward pass of a NumPy network with the architecture of the PINN
    (input z/L, three hidden layers of 32 tanh units, 2 outputs, 2242 parameters, random weights) including the
    hard-boundary ansatz and the conversion to power."""
    rng = np.random.default_rng(1)
    W = [rng.normal(size=(1, 32)), rng.normal(size=(32, 32)), rng.normal(size=(32, 32)), rng.normal(size=(32, 2))]
    b = [rng.normal(size=32), rng.normal(size=32), rng.normal(size=32), rng.normal(size=2)]
    nparam = int(sum(w.size for w in W) + sum(x.size for x in b))

    def net(zeta):
        a = zeta
        for i in range(3):
            a = np.tanh(a @ W[i] + b[i])
        o = a @ W[3] + b[3]
        return 0.04 * np.exp(zeta[:, 0] * o[:, 0]), 0.5 * np.exp((zeta[:, 0] - 1) * o[:, 1])

    res = dict(parameters=nparam)
    for n in (1, 401):
        x = np.linspace(0, 1, n).reshape(-1, 1)
        res[f"median_s_{n}_positions"] = bench(lambda: net(x), min_sample_s=0.02, repeats=25, max_total_s=3)["median_s"]
    return res


# ----------------------------------------------------------------------------------------------------------------
# Run-time table
# ----------------------------------------------------------------------------------------------------------------
def _row_specs(sweeps):
    """Models of the code base and one evaluation of each: (key, technology, model, method, operating point,
    factory). The factory does the set-up (not timed) and returns the callable that is timed; a missing module or a
    changed interface skips only that row (the error is recorded in the result file)."""
    Pin20, Pin30 = dbm2w(-20), dbm2w(-30)
    specs = []

    def add(key, tech, model, method, op, factory, callable_name, **opt):
        specs.append(dict(key=key, tech=tech, model=model, method=method, op=op, factory=factory,
                          callable=callable_name, opt=opt))

    # --- EDFA, card EDFA-GD91-T1 -----------------------------------------------------------------------------------
    add("edfa_saleh", "EDFA", "EDFA, closed form, Equation (5)", "root finding of the photon balance (brentq); "
        "card EDFA-GD91-T1", "15 m, 100 mW at 1480 nm, −20 dBm at 1550 nm",
        lambda: (lambda: edfa_saleh(0.1, Pin20, 15.0)), "models.edfa_saleh")
    add("edfa_ode", "EDFA", "EDFA, forward pump, no ASE", "ODE, LSODA, rtol 1e-10, atol 1e-16 W", "as above",
        lambda: (lambda: edfa_ode(0.1, Pin20, 15.0, npts=2)), "models.edfa_ode")
    add("edfa_sp_sweeps", "EDFA", "EDFA, single pass with forward and backward ASE",
        f"forward-backward sweeps, RK2, N = 800, stop at 1e-7 dB ({sweeps['SP']} sweeps); one lumped 30 nm ASE band",
        "10 m, 30 mW at 1480 nm, −20 dBm at 1550 nm", lambda: (lambda: em.solve(0.03, Pin20, 10.0)),
        "edfa_multipass.solve", max_total_s=15)
    add("edfa_dp_sweeps", "EDFA", "EDFA, double pass with ASE and mirror (−1 dB)",
        f"as above ({sweeps['DP']} sweeps)", "as above",
        lambda: (lambda: em.solve(0.03, Pin20, 10.0, double_pass=True)), "edfa_multipass.solve", repeats=7,
        max_total_s=40)

    def perin(kind):
        import e2e_perin as ep
        dev, _ = ep.load_device()
        meas, _ = ep.load_measurement()
        lam_s = meas[sorted(meas)[0]]["lam"]
        Ps = np.full(lam_s.size, dbm2w(ep.P_CH_DBM))
        if kind == "saleh":
            return lambda: ep.saleh_gain(dev, 0.0596, lam_s, Ps)
        if kind == "giles":
            return lambda: ep.giles_ase(dev, [0.0596], lam_s, Ps, num=ep.NUM)
        return lambda: ep.identify(dev, ep.PRIMARY, meas, [18.1, 45.8, 87.2], lam_s)

    try:
        import e2e_perin as _ep
        nb, nbf = _ep.NUM.ase_grid().size, _ep.NUM_FIT.ase_grid().size
        num_txt = (f"forward-backward sweeps, RK4, {_ep.NUM.nz} steps, {nb} ASE bins of {_ep.NUM.dlam:g} nm, stop at "
                   + f"{_ep.NUM.tol_dB:.0e}".replace("e-0", "e-") + " dB")
        fit_txt = (f"least squares (trust-region reflective) around the model of the previous row with {_ep.NUM_FIT.nz} "
                   f"steps and {nbf} bins of {_ep.NUM_FIT.dlam:g} nm")
    except Exception:
        num_txt = fit_txt = "see e2e_perin.py"
    add("perin_saleh", "EDFA", "EDFA of ref. [124], multichannel closed form, Equation (5)",
        "root finding of the photon balance (brentq); card EDFA-PERIN2019",
        "8 m, 59.6 mW at 980 nm, 40 channels at −13 dBm", lambda: perin("saleh"), "e2e_perin.saleh_gain")
    add("perin_giles", "EDFA", "EDFA of ref. [124], spectrally resolved model with forward and backward ASE", num_txt,
        "as above (one pump power)", lambda: perin("giles"), "e2e_perin.giles_ase")
    add("perin_identify", "EDFA", "EDFA of ref. [124], identification of the pump-path loss (end-to-end case, L4)",
        fit_txt, "gain at 3 pump powers × 40 channels", lambda: perin("identify"), "e2e_perin.identify",
        repeats=9, max_total_s=15)

    # --- SOA, card SOA-BULK-REF ------------------------------------------------------------------------------------
    add("soa_closed", "SOA", "SOA, homogeneous saturation, Equation (8)",
        "root finding (brentq); G0 and P_sat from the A/B/C carrier root; card SOA-BULK-REF",
        "800 µm, 150 mA, −30 dBm", lambda: (lambda: soa_analytic(0.15, Pin30)), "models.soa_analytic")
    add("soa_tw", "SOA", "SOA, traveling-wave steady state with A/B/C recombination",
        "ODE, RK45, rtol 1e-9, carrier density by root finding at every step", "as above",
        lambda: (lambda: soa_numerical(0.15, Pin30, npts=2)), "models.soa_numerical")

    def soa(kind):
        import soa_physics as sp
        if kind == "steady":
            return lambda: sp.tw_steady(Pin30, 0.15)
        if kind == "gain":
            lam = np.linspace(1350e-9, 1750e-9, 401)
            return lambda: sp.material_gain(lam, 2.0e24)
        op = sp.card_operating_point(0.15)
        if kind == "reservoir":
            return lambda: sp.reservoir_pulse(0.3e-12, op)
        return lambda: sp.tw_pulse(0.3e-12, 0.15)

    add("soa_tw_sections", "SOA", "SOA, sectioned traveling-wave steady state",
        "64 sections, carrier density by root finding in each section", "as above", lambda: soa("steady"),
        "soa_physics.tw_steady")
    add("soa_reservoir_pulse", "SOA", "SOA, reservoir model, Equation (9), one pulse",
        "RK4, 0.05 ps step, 1.55 ns window (31 000 steps)", "150 mA, 0.3 pJ Gaussian pulse",
        lambda: soa("reservoir"), "soa_physics.reservoir_pulse", max_total_s=10)
    add("soa_tw_pulse", "SOA", "SOA, sectioned traveling-wave time-domain model, one pulse",
        "RK4, 64 sections, 0.05 ps step, 1.55 ns window", "as above", lambda: soa("tw_pulse"),
        "soa_physics.tw_pulse", repeats=5, max_total_s=25)
    add("soa_gain_spectrum", "SOA", "SOA, material gain spectrum (illustrative band model)",
        "parabolic bands, Fermi-Dirac occupation, hyperbolic-secant broadening, 401 wavelengths", "N = 2.0 × 10^24 m^-3",
        lambda: soa("gain"), "soa_physics.material_gain")

    # --- Raman, card RAMAN-G652 ------------------------------------------------------------------------------------
    zr = np.linspace(0, RAMAN["L"], 401)
    add("raman_closed", "Raman", "Raman, undepleted closed form, Equation (12)",
        "analytic, 401 positions; card RAMAN-G652",
        "50 km, 500 mW at 1450 nm (counter-propagating), +16 dBm lumped at 1550 nm",
        lambda: (lambda: raman_analytic(0.5, dbm2w(16), zr)), "models.raman_analytic")
    add("raman_bvp", "Raman", "Raman, depleted counter-pumped span",
        "BVP, collocation (solve_bvp), tol 1e-8, 401 nodes, closed-form initial guess", "as above",
        lambda: (lambda: raman_numerical(0.5, dbm2w(16))), "models.raman_numerical")

    def raman_nf():
        import raman_physics as rp
        return lambda: rp.profile("counter", 15.0)

    add("raman_nf_profile", "Raman", "Raman, equivalent noise figure of a pumping scheme",
        "undepleted analytic pump profile, trapezoidal ASE integral, 2001 points",
        "80 km, counter-pumped, 15 dB on-off gain", raman_nf, "raman_physics.profile")

    # --- Hybrid Raman/EDFA span ----------------------------------------------------------------------------------------
    def hybrid(kind):
        import hybrid as hy
        lam_ch = hy.channel_plan()
        d = hy.design_lp(lam_ch, 1.0)
        if kind == "lp":
            return lambda: hy.design_lp(lam_ch, 1.0)
        if d is None:
            raise RuntimeError("design_lp infeasible at 1 W")
        p_in = 1e-3 * 10 ** (hy.HC["P_ch_in"] / 10)
        return lambda: hy.raman_full(d["lam_ch"], p_in, d["lam_p"], d["P"])

    try:
        import hybrid as _hy
        _d = _hy.design_lp(_hy.channel_plan(), 1.0)
        n_used = int(np.sum(np.asarray(_d["P"]) > 1e-6)) if _d is not None else None
        hyb_op = (f"{_hy.L_SPAN:g} km span, {_hy.channel_plan().size} channels, {len(_hy.PUMP_GRID)} candidate pump "
                  f"wavelengths, 1 W budget")
        full_txt = (f"BVP, collocation (solve_bvp), {_hy.channel_plan().size} signals + {n_used} pumps; pump depletion, "
                    f"pump-pump and signal-signal SRS")
    except Exception:
        hyb_op, full_txt = "see hybrid.py", "BVP, collocation (solve_bvp)"
    add("hybrid_lp", "Hybrid", "Hybrid Raman/EDFA span, first-order design", "linear program (HiGHS)", hyb_op,
        lambda: hybrid("lp"), "hybrid.design_lp")
    add("hybrid_full", "Hybrid", "Hybrid Raman/EDFA span, full multi-wavelength Raman model", full_txt,
        "design of the previous row", lambda: hybrid("full"), "hybrid.raman_full", max_total_s=10)
    return specs


def runtime_table(sweeps, quick=False):
    S = 0.3 if quick else 1.0
    rows = []
    t_sys0 = _system_times(); t0 = time.perf_counter()
    for sp in _row_specs(sweeps):
        row = {k: v for k, v in sp.items() if k not in ("factory", "opt")}
        try:
            fn = sp["factory"]()
            row["timing"] = bench_checked(fn, repeats=sp["opt"].get("repeats", 15),
                                          max_total_s=S * sp["opt"].get("max_total_s", 6))
            print(f"  {row['model'][:70]:70s} {fmt_time(row['timing']['median_s']):>9s}  "
                  f"(cpu/wall {row['timing']['cpu_over_wall']:.2f})", flush=True)
        except Exception as exc:
            row["timing"] = None; row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {row['model'][:70]:70s}  NOT TIMED: {row['error']}", flush=True)
        rows.append(row)
    util = _utilization(t_sys0, _system_times())
    return rows, dict(system_cpu_utilization=util, duration_s=time.perf_counter() - t0)


def fmt_time(s):
    """Two significant digits (three for values of 100 or more in the chosen unit)."""
    if s is None:
        return "n/a"
    for scale, unit in ((1e-6, "µs"), (1e-3, "ms"), (1.0, "s")):
        x = s / scale
        if x < 1000 or unit == "s":
            return f"{x:.1f} {unit}" if x < 10 else f"{x:.0f} {unit}"
    return f"{s:.3g} s"


def table_markdown(rows, ident, pinn_proxy, mach, pin, load, pm=None, lsref=None):
    lines = ["| Model | Method and numerical settings | Operating point | Median time per evaluation^b |",
             "|---|---|---|---|"]
    if pm:
        train_txt = (f"{pm['train_seconds_forward']:.1f} s (forward problem) and {pm['train_seconds_inverse']:.1f} s "
                     f"(inverse problem)^c")
        inf_txt = (f"{fmt_time(pm['inference_seconds_1'])} (1 position), {fmt_time(pm['inference_seconds_401'])} "
                   f"(401 positions)^c")
    else:
        train_txt = f"{PINN_TRAIN_S_PREVIOUS:.0f} s for the forward and inverse problems together^c"
        inf_txt = ("not available without the result file of fig09_pinn.py^d" if PINN_INFERENCE_S is None
                   else fmt_time(PINN_INFERENCE_S))
    if lsref:      # accuracy over the 2000 noise realizations of fig09_pinn.py (the values quoted in the text)
        same = round(lsref["error_sd_pct"], 2) == round(lsref["cramer_rao_pct"], 2)
        acc_txt = (f"standard deviation of the estimate {lsref['error_sd_pct']:.2f} % over {lsref['realizations']} "
                   f"noise realizations of fig09_pinn.py"
                   + (f", equal to the Cramér–Rao bound ({lsref['cramer_rao_pct']:.3f} %)" if same else
                      f" (Cramér–Rao bound {lsref['cramer_rao_pct']:.3f} %)"))
    elif ident:    # fallback: the realizations of this script
        acc_txt = (f"standard deviation of the estimate {ident['error_sd_pct']:.2f} % over {ident['realizations']} "
                   f"noise realizations (Cramér–Rao bound {ident['cramer_rao_pct']:.2f} %)")

    def emit(r):
        t = fmt_time(r["timing"]["median_s"]) if r.get("timing") else "not timed (" + r.get("error", "") + ")"
        lines.append(f"| {r['model']} | {r['method']} | {r['op']} | {t} |")

    for r in rows:
        emit(r)
        if r["key"] == "raman_bvp" and ident:
            lines.append(
                f"| Raman, identification of C_R from signal-power samples (conventional baseline for the PINN "
                f"inverse problem) | nonlinear least squares (trust-region reflective) around the BVP of the previous "
                f"row; median {ident['bvp_solves_median']} BVP solutions | 8 synthetic samples with 0.05 dB noise, "
                f"start at 0.5 C_R; {acc_txt} | {fmt_time(ident['time_median_s'])} |")
            lines.append(
                f"| Raman, PINN, training | Adam, 6000 iterations per problem, 256 collocation points per iteration "
                f"| as for the BVP; one fixed boundary-value problem, retraining needed for any other pump power, "
                f"signal load, length or fiber parameter | {train_txt} |")
            lines.append(f"| Raman, PINN, inference | forward pass of the trained network | as above | {inf_txt} |")
    pin_txt = (f"process restricted to the {len(pin['logical_processors'])} logical processors of the performance "
               f"cores" if pin.get("pinned") else "no processor affinity set")
    util = load.get("system_cpu_utilization")
    util_txt = (f"; average system CPU utilization during the benchmark {100 * util:.0f} % (other processes running)"
                if util is not None else "")
    lines += ["",
              f"^a {mach['cpu']} ({mach['logical_processors']} logical processors), {mach['os']}; Python "
              f"{mach['python']}, NumPy {mach['numpy']}, SciPy {mach['scipy']} ({mach['blas']}); OMP, OpenBLAS and MKL "
              f"threads set to 1; {pin_txt}{util_txt}. Interpreted Python, not optimized: the sweep and time-domain "
              f"models loop over the grid or the time steps in Python and would be much faster in compiled code.",
              "^b Median over repeated calls after one warm-up call (timeit-style batches, garbage collector disabled); "
              "one evaluation = one operating point (one complete identification for the identification rows). The "
              "set-up (reading card and data files, building the device description) is not timed; everything inside "
              "the model call, including the initial guess, is timed.",
              (f"^c Measured with an instrumented fig09_pinn.py ({pm['source']}). " if pm else
               f"^c Fallback (no result file of fig09_pinn.py found): result file of the submitted code archive "
               f"(Table 5 of the submitted manuscript), one run of {PINN_TRAIN_S_PREVIOUS:.1f} s, thread count not "
               f"set, hardware not recorded; not measured on the machine of footnote a. ") +
              "The network input is the normalized position only, so training is a one-off cost for one "
              "boundary-value problem and cannot be compared with the per-evaluation times of the solvers."]
    if not pm and pinn_proxy:      # footnote d belongs to the fallback inference entry only
        lines.append(f"^d For orientation only, not a PINN measurement: a NumPy network of the same architecture "
                     f"({pinn_proxy['parameters']} parameters, random weights) evaluates in "
                     f"{fmt_time(pinn_proxy['median_s_1_positions'])} for one position and "
                     f"{fmt_time(pinn_proxy['median_s_401_positions'])} for 401 positions on the machine of footnote "
                     f"a. Run fig09_pinn.py for the measured PINN times.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------------------------
def draw_flowchart(ax):
    C, INK, INK2 = style.C, style.INK, style.INK2
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs = 7.0

    def box(x, y, w, hgt, t, col, fc="white"):
        # clip_on=False: the top box reaches 0.001 above the axes (pad included) and was clipped at its top edge
        ax.add_patch(FancyBboxPatch((x - w / 2, y - hgt / 2), w, hgt, boxstyle="round,pad=0.006,rounding_size=0.018",
                                    fc=fc, ec=col, lw=1.1, clip_on=False))
        ax.text(x, y, t, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)

    def arr(p0, p1, txt=None, dx=0.015, col=None, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=col or INK2, lw=0.9, ls=ls,
                                     shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
        if txt:
            ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2, txt, fontsize=fs - 0.5, color=INK2,
                    va="center", ha="left" if dx >= 0 else "right")

    box(0.5, 0.955, 0.84, 0.08, "Parameter card (JSON file) and operating point", C[6])
    box(0.5, 0.825, 0.84, 0.08, "Do all beams propagate in the same direction?", C[3])
    box(0.225, 0.59, 0.43, 0.20, "Initial-value problem:\nsolve_ivp, LSODA (EDFA)\nor RK45 (SOA)", C[0])
    box(0.73, 0.59, 0.50, 0.25, "Boundary-value problem:\ncollocation (solve_bvp) or\nforward-backward sweeps\n"
                                "(RK2 on N steps)", C[0])
    box(0.5, 0.31, 0.98, 0.14, "Convergence checks: change between sweeps,\ngrid refinement, integrator tolerance",
        C[2])
    box(0.5, 0.085, 0.98, 0.14, "Verification against exact solutions; comparison with\n"
                                "closed forms and measured data: RMSE, max |error|, bias", C[4])
    arr((0.5, 0.915), (0.5, 0.865))
    arr((0.31, 0.785), (0.225, 0.69), "yes", dx=-0.025)
    arr((0.69, 0.785), (0.73, 0.715), "no", dx=0.025)
    arr((0.225, 0.49), (0.225, 0.38))
    arr((0.60, 0.465), (0.60, 0.38))
    arr((0.5, 0.24), (0.5, 0.155))
    arr((0.93, 0.38), (0.93, 0.465), col=C[7], ls="--")
    ax.text(0.905, 0.4225, "not converged", fontsize=fs - 0.5, color=C[7], va="center", ha="right")


def make_figure(sc, gr):
    style.apply()
    style.math_in_text_font()                           # mathtext in the same sans-serif family as the text
    C, INK, INK2 = style.C, style.INK, style.INK2
    W_IN = 7.1                                          # double-column width; every artist stays inside the canvas
    fig = plt.figure(figsize=(W_IN, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.15, 0.85], hspace=0.36, wspace=0.34,
                          left=0.035, right=0.98, top=0.96, bottom=0.085)

    ax = fig.add_subplot(gs[0, 0]); draw_flowchart(ax); style.label(ax, "a", x=-0.03, y=1.0)
    pa = ax.get_position()

    # b) sweep convergence (double pass)
    ax = fig.add_subplot(gs[0, 1])
    ch = np.array(sc["change_between_sweeps_dB"]); kk = np.array(sc["change_at_sweep"])
    m = ch > 0
    ax.semilogy(kk[m], ch[m], "o-", ms=2.6, lw=1.0, color=C[0], label="change between successive sweeps")
    ks = np.array(sorted(int(k) for k in sc["deviation_from_converged_dB"]))
    dv = np.array([sc["deviation_from_converged_dB"][k] for k in ks])
    ax.semilogy(ks[dv > 0], dv[dv > 0], "s", ms=3.4, mfc="white", mec=C[1], mew=1.0,
                label="deviation from converged output")
    ax.axhline(1e-7, color=INK2, ls=":", lw=0.9)
    ns = sc["sweeps_to_stop"]
    ax.plot([ns, ns], [1e-15, 3e-3], color=INK2, ls="--", lw=0.7)
    ax.text(ns + 1.2, 1e-4, f"stop after\n{ns} sweeps", fontsize=7, color=INK2, va="center")
    ax.text(kk[m][-1], 2.2e-7, "stopping criterion", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xlabel("Sweep number"); ax.set_ylabel("Output signal difference [dB]")
    ax.set_ylim(1e-15, 30); ax.set_xlim(0, kk[m][-1] + 2)
    ax.set_yticks([1e-14, 1e-11, 1e-8, 1e-5, 1e-2, 10])
    ax.legend(loc="upper right", fontsize=7, handlelength=1.6, borderaxespad=0.3)
    style.label(ax, "b")
    pb = ax.get_position()

    # c) grid convergence: whole bottom row; left edge moved in so that its tick and axis labels stay on the canvas
    pc = gs[1, 0].get_position(fig)
    c_left = 0.64 / W_IN
    ax = fig.add_axes([c_left, pc.y0, pb.x1 - c_left, pc.height])
    Ns = np.array(gr["N"], float)
    for key, col in (("SP", C[0]), ("DP", C[1])):
        e0 = np.abs(gr[key]["noASE_error_dB"]); eA = np.abs(gr[key]["ASE_error_dB"])
        ax.loglog(Ns, e0, "o-", ms=3.4, lw=1.1, color=col)
        ax.loglog(Ns, eA, "s--", ms=3.6, lw=1.0, color=col, mfc="white", mew=1.0)
    guide = 2.2 * abs(gr["SP"]["noASE_error_dB"][0]) * (Ns / Ns[0]) ** -2
    ax.loglog(Ns, guide, ":", color=INK2, lw=1.0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C[0], lw=1.4, label="single pass"),
               Line2D([], [], color=C[1], lw=1.4, label="double pass"),
               Line2D([], [], color=INK2, marker="o", ms=3.4, lw=1.0, label="no ASE, exact reference"),
               Line2D([], [], color=INK2, marker="s", ms=3.6, mfc="white", ls="--", lw=1.0,
                      label="with ASE, extrapolated reference"),
               Line2D([], [], color=INK2, ls=":", lw=1.0, label=r"slope $-2$")]
    ax.set_xticks(Ns); ax.set_xticklabels([f"{int(n)}" for n in Ns])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Number of spatial steps N"); ax.set_ylabel("Gain error [dB]")
    ax.set_ylim(2e-6, 0.6)
    ax.legend(handles=handles, loc="lower left", fontsize=6.8, handlelength=2.2, borderaxespad=0.3)
    pc_ = ax.get_position()                             # panel label c) aligned with a)
    style.label(ax, "c", x=(pa.x0 - 0.03 * pa.width - pc_.x0) / pc_.width)
    return fig


TABLE_TITLE = ("Table S1. Computational cost of the models of this review: median wall-clock time per evaluation on one "
               "thread.^a,b")


def write_table(res):
    """Write Table S1 (markdown) from a result dictionary. The PINN rows use the times measured by fig09_pinn.py and
    the least-squares row the accuracy over its 2000 noise realizations (FigS3_pinn_results.json) if available."""
    rt = res["runtime"]
    pm = pinn_measured()
    rt["pinn"]["measured_rerun"] = pm
    rt["pinn"]["training_note"] = (f"measured by fig09_pinn.py, see measured_rerun ({pm['source']}); "
                                   "training_s_previous_version is the fallback of the submitted code archive"
                                   if pm else "fallback: submitted code archive, forward + inverse problems, 6000 "
                                   "Adam iterations each, one run, thread count not set, hardware not recorded")
    rt["pinn"]["inference_note"] = (f"measured by fig09_pinn.py, see measured_rerun ({pm['source']})" if pm
                                    else "not available without a result file of fig09_pinn.py")
    ident = res["raman_identification_baseline"]
    lsref = ls_reference()
    ident["accuracy_in_table"] = (dict(lsref, note="accuracy statistics quoted in Table S1 and in the text; the "
                                                   "realizations of this script are used only for the run time")
                                  if lsref else dict(source="this script (no result file of fig09_pinn.py)",
                                                     error_sd_pct=ident["error_sd_pct"],
                                                     realizations=ident["realizations"],
                                                     cramer_rao_pct=ident["cramer_rao_pct"]))
    md = table_markdown(rt["rows"], ident, rt["pinn"]["numpy_mlp_proxy"], res["machine"], res["affinity"],
                        rt["load"], pm, lsref)
    with open(os.path.join(FIGDIR, NAME + "_runtime_table.md"), "w", encoding="utf-8") as f:
        f.write(TABLE_TITLE + "\n\n" + md + "\n")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-runtime", action="store_true", help="skip the run-time benchmark")
    ap.add_argument("--no-pin", action="store_true", help="do not restrict the process to the performance cores")
    ap.add_argument("--quick", action="store_true", help="coarser studies for testing")
    ap.add_argument("--table-only", action="store_true",
                    help="rewrite Table S1 from the stored result file without new computations or timings "
                         "(e.g. after an instrumented PINN re-run has written its result file)")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw Figure S1 from the stored result file; no computation, timing or file other than "
                         "the figure is written")
    args = ap.parse_args()

    if args.plot_only:
        with open(os.path.join(FIGDIR, NAME + "_results.json"), encoding="utf-8") as f:
            res = json.load(f)
        sc = res["b_sweep_convergence"]
        sc["deviation_from_converged_dB"] = {int(k): v for k, v in sc["deviation_from_converged_dB"].items()}
        style.save(make_figure(sc, res["c_grid_convergence"]), NAME, outdir=FIGDIR)
        print("figure redrawn from", NAME + "_results.json")
        return

    if args.table_only:
        path = os.path.join(FIGDIR, NAME + "_results.json")
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        res["machine"]["os"] = _os_label(res["machine"]["os"])
        print(write_table(res))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        return

    pin = dict(pinned=False, reason="--no-pin") if args.no_pin else _pin_performance_cores()
    mach = machine_info()
    print("machine:", json.dumps(mach), "\naffinity:", json.dumps(pin))
    res = dict(figure=NAME, generated=datetime.datetime.now().isoformat(timespec="seconds"), machine=mach,
               affinity=pin, sources_sha256_16={f: _sha256(os.path.join(HERE, f)) for f in
                                                ("models.py", "edfa_multipass.py", "e2e_perin.py", "soa_physics.py",
                                                 "raman_physics.py", "hybrid.py", "cards.py", "fig_numerics.py")})
    t0 = time.perf_counter()
    print("b) sweep convergence ..."); res["b_sweep_convergence"] = sweep_convergence(args.quick)
    print("c) grid convergence ..."); res["c_grid_convergence"] = grid_convergence(args.quick)
    print("supplementary: EDFA ODE tolerance (shown in Figure 11b) ..."); res["edfa_ode_tolerance"] = tolerance_study()
    print("supplementary: Raman collocation tolerance ..."); res["raman_bvp_tolerance"] = raman_bvp_tolerance()
    res["figure_computation_s"] = time.perf_counter() - t0

    fig = make_figure(res["b_sweep_convergence"], res["c_grid_convergence"])
    style.save(fig, NAME, outdir=FIGDIR)
    print("figure written")

    if not args.no_runtime:
        print("run-time benchmark ...")
        sweeps = {"SP": res["b_sweep_convergence"]["single_pass"]["sweeps_to_stop"],
                  "DP": res["b_sweep_convergence"]["sweeps_to_stop"]}
        rows, load = runtime_table(sweeps, args.quick)
        print("C_R identification baseline ...")
        s0 = _system_times()
        ident = raman_identification(quick=args.quick)
        ident["system_cpu_utilization"] = _utilization(s0, _system_times())
        proxy = numpy_mlp_proxy()
        pm = pinn_measured()
        res["runtime"] = dict(
            method="median wall-clock time per evaluation (time.perf_counter), timeit-style batches after one warm-up "
                   "call, garbage collector disabled; cpu_over_wall = process CPU time / wall-clock time (a value well "
                   "below 1 means that the process was descheduled; such measurements are repeated up to 3 times)",
            load=load, rows=rows,
            pinn=dict(training_s_previous_version=PINN_TRAIN_S_PREVIOUS,
                      training_note="set by write_table",
                      inference_s=PINN_INFERENCE_S,
                      inference_note="set by write_table",
                      measured_rerun=pm, numpy_mlp_proxy=proxy))
        res["raman_identification_baseline"] = ident
        print(write_table(res))
    with open(os.path.join(FIGDIR, NAME + "_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    b, cg, d = res["b_sweep_convergence"], res["c_grid_convergence"], res["edfa_ode_tolerance"]
    print(f"sweeps to 1e-7 dB: {b['sweeps_to_stop']}, deviation at stop {b['deviation_at_stop_dB']:.2e} dB, "
          f"contraction ratio {b['asymptotic_contraction_ratio']:.4f}")
    for k in ("SP", "DP"):
        print(f"{k}: no-ASE error {cg[k]['noASE_error_dB'][0]:.3e} (N={cg['N'][0]}) -> {cg[k]['noASE_error_dB'][-1]:.3e} dB "
              f"(N={cg['N'][-1]}); with ASE {cg[k]['ASE_error_dB'][0]:.3e} -> {cg[k]['ASE_error_dB'][-1]:.3e} dB")
    print(f"ODE vs Saleh at the models.edfa_ode setting: {d['setting_of_models_edfa_ode']['rms_dB']:.2e} dB RMS")


if __name__ == "__main__":
    main()
