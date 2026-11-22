"""Shared helpers for the physics-informed neural network (PINN) examples of Section 7.3 and Section S5.

Used by fig09_pinn.py (Figure S3: one fixed boundary-value problem, identification of C_R) and pinn_parametric.py
(Figure S4: parameterized PINN). All physical parameters come from the card RAMAN-G652 through models.py.

Scaled problem (normalized position s = z/L, u = ln[P_s(z)/P_s(0)], v = ln[P_p(z)/P_p(L)]):
    du/ds = kappa_p exp(v) - delta_s,      dv/ds = delta_p + kappa_s exp(u),      u(0) = 0,  v(1) = 0,
with kappa_p = C_R P_p(L) L, kappa_s = (nu_p/nu_s) C_R P_s(0) L, delta_s = l_s L, delta_p = l_p L (attenuations in
neper per unit length). These four dimensionless groups specify the counter-pumped single-signal problem completely.

The machine description, the processor pinning and the benchmark routine follow the protocol of fig_numerics.py
(Table S1) so that the PINN timings can be read next to the solver timings of that table.
"""
import gc
import os
import platform
import re
import sys
import time

import numpy as np
from scipy.integrate import solve_bvp

from models import RAMAN, raman_CR, raman_analytic, dbm2w, w2dbm, DB

NP2DB = 10.0 / np.log(10.0)            # neper -> dB
CARD_ID = "RAMAN-G652"
L_M = RAMAN["L"]                       # span length [m]
CR = raman_CR(RAMAN)                   # Raman gain efficiency [1/(W m)]
R_NU = RAMAN["lam_s"] / RAMAN["lam_p"]  # nu_p / nu_s
LS_DBKM = RAMAN["alpha_s"] / (DB * 1e-3)   # card attenuation [dB/km] (0.19)
LP_DBKM = RAMAN["alpha_p"] / (DB * 1e-3)   # card attenuation [dB/km] (0.25)

# Operating point of the single-instance example (Section 5.4 and the submitted Figure 16)
PP_CARD_W = 0.5           # counter-propagating pump launched at z = L [W]
PS_CARD_DBM = 16.0        # lumped signal launched at z = 0 [dBm]


# ----------------------------------------------------------------------------------------------------------------
# Dimensionless groups
# ----------------------------------------------------------------------------------------------------------------
def groups(Pp_W, Ps_dBm, ls_dBkm=LS_DBKM, lp_dBkm=LP_DBKM, CR_=CR, L=L_M):
    """(kappa_p, kappa_s, delta_s, delta_p) for pump power [W], signal power [dBm] and attenuations [dB/km]."""
    kp = CR_ * np.asarray(Pp_W) * L
    ks = R_NU * CR_ * dbm2w(Ps_dBm) * L
    ds = np.asarray(ls_dBkm) * DB * 1e-3 * L
    dp = np.asarray(lp_dBkm) * DB * 1e-3 * L
    return kp, ks, ds, dp


def physical(kp, ks, ds, dp, CR_=CR, L=L_M):
    """Inverse of groups(): pump power [W], signal power [dBm], attenuations [dB/km] at the card C_R and length."""
    return (kp / (CR_ * L), w2dbm(ks / (R_NU * CR_ * L)), ds / (DB * 1e-3 * L), dp / (DB * 1e-3 * L))


CARD_GROUPS = tuple(float(x) for x in groups(PP_CARD_W, PS_CARD_DBM))


def undepleted(s, kp, ds, dp):
    """Undepleted-pump closed form, Equation (12), in scaled form: u0(s), v0(s)."""
    s = np.asarray(s)
    v0 = -dp * (1.0 - s)
    u0 = kp * np.exp(-dp) * np.expm1(dp * s) / dp - ds * s
    return u0, v0


# ----------------------------------------------------------------------------------------------------------------
# Collocation solvers (scipy.integrate.solve_bvp)
# ----------------------------------------------------------------------------------------------------------------
def bvp_scaled(kp, ks, ds, dp, n=401, tol=1e-8, guess=None, s_eval=None):
    """Scaled BVP by collocation. Initial guess: undepleted closed form on n nodes (cold start) or a supplied
    (s_nodes, y_nodes) pair (warm start). Returns dict(s, u, v, status, nodes, niter, sol) with u, v on s_eval
    (default: n equidistant points)."""
    if guess is None:
        s0 = np.linspace(0.0, 1.0, n)
        y0 = np.vstack(undepleted(s0, kp, ds, dp))
    else:
        s0, y0 = guess

    def f(s, y):
        return np.vstack([kp * np.exp(y[1]) - ds, dp + ks * np.exp(y[0])])

    def bc(ya, yb):
        return np.array([ya[0], yb[1]])

    sol = solve_bvp(f, bc, s0, y0, tol=tol, max_nodes=200000)
    se = np.linspace(0.0, 1.0, n) if s_eval is None else s_eval
    y = sol.sol(se)
    return dict(s=se, u=y[0], v=y[1], status=int(sol.status), nodes=int(sol.x.size), niter=int(sol.niter), sol=sol)


def raman_bvp(Pp0, Ps0, p=RAMAN, n=401, guess=None):
    """Same computation as models.raman_numerical (physical units, tol 1e-8, n nodes, undepleted initial guess), with an
    optional warm-start guess (z_nodes, [ln Ps, ln Pp]) and the solution object returned. For guess=None the result is
    identical to models.raman_numerical (checked in fig09_pinn.py)."""
    CR_, r = raman_CR(p), p["lam_s"] / p["lam_p"]
    L = p["L"]

    def f(z, y):
        Ps, Pp = np.exp(y[0]), np.exp(y[1])
        return np.vstack([CR_ * Pp - p["alpha_s"], p["alpha_p"] + r * CR_ * Ps])

    def bc(ya, yb):
        return np.array([ya[0] - np.log(Ps0), yb[1] - np.log(Pp0)])

    z = np.linspace(0, L, n)
    if guess is None:
        ys, yp = raman_analytic(Pp0, Ps0, z, p)
        z0, y0 = z, np.vstack([np.log(ys), np.log(yp)])
    else:
        z0, y0 = guess
    sol = solve_bvp(f, bc, z0, y0, tol=1e-8, max_nodes=100000)
    return z, np.exp(sol.sol(z)[0]), np.exp(sol.sol(z)[1]), sol.status, sol


def card_with(CR_factor=1.0, ls_dBkm=None, lp_dBkm=None, p=RAMAN):
    """Copy of the card dictionary with C_R scaled (through g_R) and/or the attenuations replaced [dB/km]."""
    q = dict(p)
    q["gR"] = p["gR"] * CR_factor
    if ls_dBkm is not None:
        q["alpha_s"] = ls_dBkm * DB * 1e-3
    if lp_dBkm is not None:
        q["alpha_p"] = lp_dBkm * DB * 1e-3
    return q


# ----------------------------------------------------------------------------------------------------------------
# Machine description, processor pinning and timing (protocol of fig_numerics.py / Table S1)
# ----------------------------------------------------------------------------------------------------------------
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def os_label():
    s = platform.platform()
    m = re.match(r"Windows-10-10\.0\.(\d+)", s)
    if m and int(m.group(1)) >= 22000:
        return f"Windows 11 (build {m.group(1)})"
    return s


def machine_info():
    import scipy
    import matplotlib
    d = dict(cpu=cpu_name(), logical_processors=os.cpu_count(), os=os_label(), python=platform.python_version(),
             numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
             thread_env={v: os.environ.get(v) for v in THREAD_VARS})
    try:
        import torch
        d["torch"] = torch.__version__
        d["torch_parallel_backend"] = "OpenMP" if "OpenMP" in torch.__config__.parallel_info() else "other"
        mkl = re.search(r"Intel\(R\) oneAPI Math Kernel Library Version ([0-9.]+)", torch.__config__.show())
        d["torch_mkl"] = mkl.group(1) if mkl else None
    except Exception:
        d["torch"] = None
    return d


def pin_performance_cores():
    """Windows only: restrict this process to the logical processors of the highest efficiency class (performance
    cores of hybrid CPUs), as in fig_numerics.py. Affects this process only. Returns a description."""
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
            if int.from_bytes(raw[off + 4:off + 8], "little") == 0:
                group, lp, eff = int.from_bytes(raw[off + 12:off + 14], "little"), raw[off + 14], raw[off + 18]
                if group == 0:
                    cls[lp] = eff
            off += size
        counts = {int(e): sum(1 for v in cls.values() if v == e) for e in sorted(set(cls.values()))}
        if len(counts) < 2:
            return dict(pinned=False, reason="all logical processors have the same efficiency class")
        top = max(cls.values())
        lps = sorted(lp for lp, e in cls.items() if e == top)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        ok = bool(k32.SetProcessAffinityMask(k32.GetCurrentProcess(), sum(1 << lp for lp in lps)))
        return dict(pinned=ok, logical_processors=lps, efficiency_class=int(top), efficiency_classes=counts)
    except Exception as exc:  # pragma: no cover
        return dict(pinned=False, reason=f"{type(exc).__name__}: {exc}")


def system_times():
    """(idle, total) CPU time of the whole system (for the average utilization during a measurement)."""
    try:
        if sys.platform == "win32":
            import ctypes
            idle, kern, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
            return idle.value, kern.value + user.value
        if os.path.exists("/proc/stat"):
            f = [float(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
    except Exception:
        pass
    return None


def utilization(s0, s1):
    if s0 and s1 and (s1[1] - s0[1]) > 0:
        return float(1.0 - (s1[0] - s0[0]) / (s1[1] - s0[1]))
    return None


def bench(fn, min_sample_s=0.05, repeats=15, max_total_s=20.0):
    """timeit-style benchmark as in fig_numerics.py: one warm-up call, then `repeats` samples of `number` calls each
    (each sample lasts at least min_sample_s), garbage collector disabled. Returns per-call statistics."""
    t0 = time.perf_counter(); fn(); t1 = time.perf_counter() - t0
    number = max(1, int(np.ceil(min_sample_s / max(t1, 1e-7))))
    repeats = int(max(3, min(repeats, max_total_s / max(t1 * number, 1e-9))))
    wall, cpu = [], []
    s0 = system_times()
    was = gc.isenabled(); gc.disable()
    try:
        for _ in range(repeats):
            c0, w0 = time.process_time(), time.perf_counter()
            for _ in range(number):
                fn()
            wall.append((time.perf_counter() - w0) / number); cpu.append((time.process_time() - c0) / number)
    finally:
        if was:
            gc.enable()
    s1 = system_times()
    w = np.array(wall)
    return dict(median_s=float(np.median(w)), min_s=float(w.min()), q25_s=float(np.percentile(w, 25)),
                q75_s=float(np.percentile(w, 75)), process_cpu_over_wall=float(np.sum(cpu) / np.sum(wall)),
                samples=repeats, calls_per_sample=number, system_cpu_utilization=utilization(s0, s1))


def repeat_timed(fn, repeats=3):
    """Run a long task `repeats` times; return the median wall time, all wall times, process-CPU/wall ratios and the
    average system utilization; fn() must return its own result (kept from the last run)."""
    walls, ratios, out = [], [], None
    s0 = system_times()
    for _ in range(repeats):
        c0, w0 = time.process_time(), time.perf_counter()
        out = fn()
        w = time.perf_counter() - w0
        walls.append(w); ratios.append((time.process_time() - c0) / w)
    s1 = system_times()
    return dict(median_s=float(np.median(walls)), all_s=[float(x) for x in walls],
                process_cpu_over_wall=[float(x) for x in ratios], system_cpu_utilization=utilization(s0, s1)), out


def percentiles(x, q=(50, 95, 100)):
    x = np.asarray(x, dtype=float)
    return {f"p{int(k)}": float(np.percentile(x, k)) for k in q}


def jsonable(o):
    """Recursively convert numpy scalars/arrays to plain Python for json.dump."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o
