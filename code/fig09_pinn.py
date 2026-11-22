"""Figure S3 (Supporting Information, Section S5): physics-informed neural network (PINN) for ONE fixed counter-pumped
distributed Raman boundary-value problem (BVP), and identification of the Raman gain efficiency C_R.

  a) training loss of the forward and inverse problems;
  b) forward solution (signal and pump power) against the collocation solver (solve_bvp), and the deviation;
  c) identified C_R over network seeds and noise realizations, against nonlinear least squares with the collocation
     solver on the same data sets and the Cramer-Rao bound.

Problem: card RAMAN-G652 (50 km, C_R = 0.406 1/(W km), l_s = 0.19 dB/km at 1550 nm, l_p = 0.25 dB/km at 1450 nm),
0.5 W counter-propagating pump at 1450 nm, +16 dBm lumped signal at 1550 nm. Network (settings of the submitted
version): input s = z/L only; 3 hidden layers x 32 tanh units (2242 parameters); hard boundary conditions through the
ansatz u = s N_1(s), v = (s - 1) N_2(s), with u = ln[P_s/P_s(0)], v = ln[P_p/P_p(L)]; Adam, 6000 iterations,
learning rate 2e-3 with StepLR (x0.3 every 2000 iterations); 256 uniformly random collocation points per iteration;
float64. Because the only input is the position, the trained weights represent one solution: any change of pump power,
signal load, length or fiber parameter requires retraining (pinn_parametric.py treats the parameterized problem).

Inverse problem: 8 synthetic signal-power samples (2.5-50 km, Gaussian noise of 0.05 dB) generated with the collocation
solver; C_R unknown (start 0.5 x true); the residual loss is augmented by 10 x the mean squared misfit of ln P_s at the
samples. Unless stated otherwise the attenuations and launched powers are assumed exactly known. Conventional baseline:
nonlinear least squares (scipy.optimize.least_squares, trust-region reflective, 2-point Jacobian) around the collocation
solver of models.py on the same data sets. Cramer-Rao bound from the sensitivities of the collocation solution at the 8
sample positions. Identifiability: bias when the assumed signal attenuation is off by 0.01 dB/km or the launched pump
power by 5 %; joint identification of C_R and l_s.

Statistics: the trainings run in a pool of worker processes with one thread each (default 4 workers, i.e. at most four
cores in use); every training is seeded, so the results do not depend on the number of workers. The cost figures come
from a separate sequential benchmark with 1 and with 4 PyTorch threads.

Outputs: ../figures/FigS3_pinn.png/.pdf, ../figures/FigS3_pinn_results.json
Usage  : python fig09_pinn.py [--workers 4] [--seeds 5] [--realizations 30] [--quick] [--no-timing] [--plot-only]
         python fig09_pinn.py --weight-study   (data-weight check, merged into the existing result file)
"""
import os
import sys

# One BLAS/OpenMP thread for NumPy/SciPy, as in Table S1 (set before NumPy is imported; inherited by the worker
# processes). PyTorch threads are set explicitly with torch.set_num_threads().
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.optimize import least_squares

import models
from models import RAMAN, raman_numerical, dbm2w, w2dbm, DB
import pinn_common as pc

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
NAME = "FigS3_pinn"

PP0 = pc.PP_CARD_W                         # pump launched at z = L [W]
PS0 = float(dbm2w(pc.PS_CARD_DBM))         # signal launched at z = 0 [W]
KP, KS, DS, DPP = pc.CARD_GROUPS           # kappa_p, kappa_s, delta_s, delta_p of the card case
IDX = np.linspace(20, 400, 8).astype(int)  # sample indices on the 401-point grid (as in the submitted version)
SIGMA_DB = 0.05                            # noise standard deviation of the samples [dB]
ITERS = 6000
LS_OFFSET_DBKM = 0.01                      # mis-specified signal attenuation
PP_FACTOR = 0.05                           # mis-specified launched pump power (relative)


# ----------------------------------------------------------------------------------------------------------------
# Network and training (settings of the submitted version)
# ----------------------------------------------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        layers, d = [], 1
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers += [torch.nn.Linear(d, 2)]
        self.f = torch.nn.Sequential(*layers)

    def forward(self, s):
        o = self.f(s)
        return s * o[:, :1], (s - 1.0) * o[:, 1:]      # hard boundary conditions u(0) = 0, v(1) = 0


def n_parameters(net):
    return int(sum(p.numel() for p in net.parameters()))


def residual_loss(net, s, kp, ks, ds, dp):
    s.requires_grad_(True)
    u, v = net(s)
    du = torch.autograd.grad(u.sum(), s, create_graph=True)[0]
    dv = torch.autograd.grad(v.sum(), s, create_graph=True)[0]
    r1 = du - (kp * torch.exp(v) - ds)
    r2 = dv - (dp + ks * torch.exp(u))
    return (r1 ** 2).mean() + (r2 ** 2).mean()


def train(seed=0, data=None, unknown=(), assumed=None, iters=ITERS, hist_every=10, reseed=True, init_CR=0.5,
          init_ls=0.9, data_weight=10.0):
    """Train one PINN. data = (s_samples, u_samples) for the inverse problem; unknown subset of ('CR', 'ls') as
    log-multipliers of the assumed values (start init_CR x and init_ls x); assumed = dict(kp, ks, ds, dp) used in the
    residual (default: card values); data_weight multiplies the data misfit (10 in the submitted version).
    Returns (net, info)."""
    if reseed:
        torch.manual_seed(seed)
    a = dict(kp=KP, ks=KS, ds=DS, dp=DPP) if assumed is None else assumed
    net = Net()
    params = list(net.parameters())
    logm = {}
    if "CR" in unknown:
        logm["CR"] = torch.nn.Parameter(torch.tensor(np.log(init_CR)))
    if "ls" in unknown:
        logm["ls"] = torch.nn.Parameter(torch.tensor(np.log(init_ls)))
    params += list(logm.values())
    if data is not None:
        sd = torch.tensor(data[0]).reshape(-1, 1)
        ud = torch.tensor(data[1]).reshape(-1, 1)
    opt = torch.optim.Adam(params, lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 2000, 0.3)
    hist, cr_hist, ls_hist = [], [], []
    t0 = time.perf_counter()
    for it in range(iters):
        s = torch.rand(256, 1)
        mCR = torch.exp(logm["CR"]) if "CR" in logm else 1.0
        mls = torch.exp(logm["ls"]) if "ls" in logm else 1.0
        loss = residual_loss(net, s, a["kp"] * mCR, a["ks"] * mCR, a["ds"] * mls, a["dp"])
        if data is not None:
            u_pred, _ = net(sd)
            loss = loss + data_weight * ((u_pred - ud) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % hist_every == 0 or it == iters - 1:
            hist.append(float(loss.item()))
            if "CR" in logm:
                cr_hist.append(float(torch.exp(logm["CR"].detach())))
            if "ls" in logm:
                ls_hist.append(float(torch.exp(logm["ls"].detach())))
    wall = time.perf_counter() - t0
    info = dict(loss_hist=hist, final_loss=hist[-1], wall_s=wall,
                CR_factor=float(torch.exp(logm["CR"].detach())) if "CR" in logm else None,
                ls_factor=float(torch.exp(logm["ls"].detach())) if "ls" in logm else None,
                CR_hist=cr_hist, ls_hist=ls_hist)
    # diagnostics (not used in training): RMS residual of the two equations on 1001 equidistant points with the
    # identified parameters, and RMS misfit of the samples [dB]
    mCR = float(torch.exp(logm["CR"].detach())) if "CR" in logm else 1.0
    mls = float(torch.exp(logm["ls"].detach())) if "ls" in logm else 1.0
    se = torch.linspace(0, 1, 1001).reshape(-1, 1).requires_grad_(True)
    u, v = net(se)
    du = torch.autograd.grad(u.sum(), se, retain_graph=True)[0]
    dv = torch.autograd.grad(v.sum(), se)[0]
    r1 = du - (a["kp"] * mCR * torch.exp(v) - a["ds"] * mls)
    r2 = dv - (a["dp"] + a["ks"] * mCR * torch.exp(u))
    info["residual_rms"] = float(torch.sqrt(((r1 ** 2).mean() + (r2 ** 2).mean()) / 2).detach())
    if data is not None:
        with torch.no_grad():
            u_pred, _ = net(sd)
        info["data_misfit_rms_dB"] = float(torch.sqrt(((u_pred - ud) ** 2).mean()) * pc.NP2DB)
    return net, info


def infer(net, s_t):
    """Inference: signal and pump power [W] at the positions s_t (torch tensor of shape (n, 1))."""
    with torch.no_grad():
        u, v = net(s_t)
    return PS0 * np.exp(u.numpy().ravel()), PP0 * np.exp(v.numpy().ravel())


def assumed_groups(Pp_factor=1.0, ls_offset_dBkm=0.0):
    kp, ks, ds, dp = pc.groups(PP0 * Pp_factor, pc.PS_CARD_DBM, pc.LS_DBKM + ls_offset_dBkm, pc.LP_DBKM)
    return dict(kp=float(kp), ks=float(ks), ds=float(ds), dp=float(dp))


# ----------------------------------------------------------------------------------------------------------------
# Worker jobs (process pool, one thread per worker)
# ----------------------------------------------------------------------------------------------------------------
def _init_worker():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)


def run_job(job):
    torch.set_default_dtype(torch.float64)
    ref = job["ref"]
    s_ref = np.asarray(ref["s"])
    if job["kind"] == "forward":
        net, info = train(seed=job["seed"])
        Ps, Pp = infer(net, torch.tensor(s_ref).reshape(-1, 1))
        ds_ = w2dbm(Ps) - np.asarray(ref["Ps_dBm"])
        dp_ = w2dbm(Pp) - np.asarray(ref["Pp_dBm"])
        out = dict(rms_signal_dB=float(np.sqrt(np.mean(ds_ ** 2))), max_signal_dB=float(np.max(np.abs(ds_))),
                   rms_pump_dB=float(np.sqrt(np.mean(dp_ ** 2))), max_pump_dB=float(np.max(np.abs(dp_))),
                   net_gain_err_dB=float(ds_[-1]), dev_signal_dB=ds_.tolist(), dev_pump_dB=dp_.tolist(),
                   Ps_dBm=w2dbm(Ps).tolist(), Pp_dBm=w2dbm(Pp).tolist())
    elif job["kind"] == "replica":
        # sequence of the submitted script: torch.manual_seed(0), forward training, then inverse training with the
        # random-number stream continuing, on the submitted noise realization
        torch.manual_seed(0)
        net_f, info_f = train(reseed=False)
        net, info = train(data=job["data"], unknown=("CR",), reseed=False)
        out = dict(CR_factor=info["CR_factor"], forward_final_loss=info_f["final_loss"])
    else:
        assumed = assumed_groups(job.get("Pp_factor", 1.0), job.get("ls_offset", 0.0))
        net, info = train(seed=job["seed"], data=job["data"], unknown=tuple(job["unknown"]), assumed=assumed,
                          data_weight=job.get("data_weight", 10.0))
        out = dict(CR_factor=info["CR_factor"], ls_factor=info["ls_factor"],
                   data_misfit_rms_dB=info.get("data_misfit_rms_dB"))
    out.update(job={k: v for k, v in job.items() if k not in ("ref", "data")}, final_loss=info["final_loss"],
               wall_s=info["wall_s"], residual_rms=info.get("residual_rms"))
    if job.get("keep_history"):
        out.update(loss_hist=info["loss_hist"], CR_hist=info["CR_hist"], ls_hist=info["ls_hist"])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Data, conventional estimator and Cramer-Rao bound
# ----------------------------------------------------------------------------------------------------------------
def noise(j):
    """Noise realization j [dB]; j = 0 is the realization of the submitted script (np.random.seed(0))."""
    if j == 0:
        rs = np.random.RandomState(0)
        return rs.normal(0, SIGMA_DB, IDX.size)
    return np.random.default_rng(j).normal(0, SIGMA_DB, IDX.size)


class LSModel:
    """Signal power [dBm] at the sample positions from the collocation solver (models.raman_numerical) for a C_R factor
    and (optionally) an l_s factor, with assumed launched pump power and attenuation; counts the solver calls."""

    def __init__(self, Pp_factor=1.0, ls_offset_dBkm=0.0):
        self.Pp = PP0 * Pp_factor
        self.ls = pc.LS_DBKM + ls_offset_dBkm
        self.calls = 0

    def __call__(self, theta):
        self.calls += 1
        q = pc.card_with(CR_factor=theta[0], ls_dBkm=self.ls * (theta[1] if len(theta) > 1 else 1.0))
        z, Ps, Pp, st = raman_numerical(self.Pp, PS0, q)
        return w2dbm(Ps[IDX])


def ls_identify(y_db, joint=False, Pp_factor=1.0, ls_offset_dBkm=0.0):
    m = LSModel(Pp_factor, ls_offset_dBkm)
    x0 = [0.5, 0.9] if joint else [0.5]
    lb = [0.05, 0.2] if joint else [0.05]
    ub = [5.0, 3.0] if joint else [5.0]
    t0 = time.perf_counter()
    r = least_squares(lambda th: (m(th) - y_db) / SIGMA_DB, x0, method="trf", jac="2-point", bounds=(lb, ub),
                      xtol=1e-10, ftol=1e-10)
    dt = time.perf_counter() - t0
    return dict(CR_factor=float(r.x[0]), ls_factor=float(r.x[1]) if joint else None, seconds=dt,
                bvp_calls=m.calls, nfev=int(r.nfev), status=int(r.status))


def sensitivities(rel_step=1e-3):
    """d(signal power in dB)/d(ln theta) at the sample positions, for theta = C_R, l_s, l_p, P_p(L); accurate scaled
    collocation solution (tol 1e-10), central differences."""
    s_smp = IDX / 400.0

    def y(kp, ks, ds, dp):
        r = pc.bvp_scaled(kp, ks, ds, dp, tol=1e-10, s_eval=s_smp)
        assert r["status"] == 0
        return r["u"] * pc.NP2DB

    J = {}
    e = np.exp(rel_step)
    J["CR"] = (y(KP * e, KS * e, DS, DPP) - y(KP / e, KS / e, DS, DPP)) / (2 * rel_step)
    J["ls"] = (y(KP, KS, DS * e, DPP) - y(KP, KS, DS / e, DPP)) / (2 * rel_step)
    J["lp"] = (y(KP, KS, DS, DPP * e) - y(KP, KS, DS, DPP / e)) / (2 * rel_step)
    J["Pp"] = (y(KP * e, KS, DS, DPP) - y(KP / e, KS, DS, DPP)) / (2 * rel_step)
    return J


def crb(J, names):
    A = np.column_stack([J[n] for n in names])
    F = A.T @ A / SIGMA_DB ** 2
    C = np.linalg.inv(F)
    sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    return dict(names=list(names), std_rel_pct={n: float(100 * sd[i]) for i, n in enumerate(names)},
                correlation=corr.tolist())


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------
def stats_pct(x):
    x = 100 * (np.asarray(x, dtype=float) - 1.0)
    return dict(n=int(x.size), mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                min=float(np.min(x)), max=float(np.max(x)), rms=float(np.sqrt(np.mean(x ** 2))),
                max_abs=float(np.max(np.abs(x))), frac_within_0p5pct=float(np.mean(np.abs(x) <= 0.5)))


def compute(args):
    res = dict(script="fig09_pinn.py", date=time.strftime("%Y-%m-%d %H:%M"), card=pc.CARD_ID)
    res["machine"] = pc.machine_info()
    res["pinning"] = pc.pin_performance_cores()
    res["problem"] = dict(L_km=pc.L_M / 1e3, CR_per_W_km=pc.CR * 1e3, ls_dB_km=pc.LS_DBKM, lp_dB_km=pc.LP_DBKM,
                          Pp_L_W=PP0, Ps_0_dBm=pc.PS_CARD_DBM, kappa_p=KP, kappa_s=KS, delta_s=DS, delta_p=DPP,
                          sample_positions_km=(IDX * pc.L_M / 400 / 1e3).tolist(), noise_std_dB=SIGMA_DB)

    # reference solution (collocation solver of the paper) and its own accuracy
    z, Ps_ref, Pp_ref, st = raman_numerical(PP0, PS0)
    z2, Ps2, Pp2, st2, _ = pc.raman_bvp(PP0, PS0)
    acc = pc.bvp_scaled(KP, KS, DS, DPP, tol=1e-10)
    res["reference"] = dict(
        solver="models.raman_numerical (solve_bvp, tol 1e-8, 401 nodes, undepleted initial guess)", status=int(st),
        identical_to_pinn_common_raman_bvp=bool(np.array_equal(Ps_ref, Ps2) and np.array_equal(Pp_ref, Pp2)),
        max_dev_from_tight_scaled_solution_dB=dict(
            signal=float(np.max(np.abs(w2dbm(Ps_ref) - w2dbm(PS0 * np.exp(acc["u"]))))),
            pump=float(np.max(np.abs(w2dbm(Pp_ref) - w2dbm(PP0 * np.exp(acc["v"])))))),
        net_gain_dB=float(w2dbm(Ps_ref[-1]) - pc.PS_CARD_DBM),
        pump_at_z0_dBm=float(w2dbm(Pp_ref[0])))
    ref = dict(s=(z / pc.L_M).tolist(), Ps_dBm=w2dbm(Ps_ref).tolist(), Pp_dBm=w2dbm(Pp_ref).tolist())
    y_true = w2dbm(Ps_ref[IDX])
    s_smp = z[IDX] / pc.L_M

    def data_for(y_db):
        return (s_smp, np.log(dbm2w(y_db) / PS0))

    # Cramer-Rao bounds
    J = sensitivities()
    cr = dict(sensitivity_dB_per_unit_ln=dict((k, v.tolist()) for k, v in J.items()))
    cr["CR_only"] = float(100 * SIGMA_DB / np.sqrt(np.sum(J["CR"] ** 2)))
    cr["CR_only_output_sample"] = float(100 * SIGMA_DB / abs(J["CR"][-1]))
    cr["joint_CR_ls"] = crb(J, ("CR", "ls"))
    cr["joint_CR_ls_lp"] = crb(J, ("CR", "ls", "lp"))
    cr["joint_CR_Pp"] = crb(J, ("CR", "Pp"))
    jc = J["CR"]
    cr["linearized_bias_pct"] = {
        "ls_plus_0.01dB_per_km": float(-100 * (jc @ J["ls"]) / (jc @ jc) * np.log((pc.LS_DBKM + LS_OFFSET_DBKM) / pc.LS_DBKM)),
        "Pp_plus_5pct": float(-100 * (jc @ J["Pp"]) / (jc @ jc) * np.log(1 + PP_FACTOR))}
    res["cramer_rao"] = cr
    print(f"Cramer-Rao bound C_R: {cr['CR_only']:.3f} % (output sample alone {cr['CR_only_output_sample']:.3f} %)")

    # ------------------------------------------------------------------ PINN jobs (process pool)
    nf = 2 if args.quick else args.forward_seeds
    ns = 2 if args.quick else args.seeds
    nr = 3 if args.quick else args.realizations
    nj = 2 if args.quick else args.joint_realizations
    jobs = [dict(kind="forward", seed=k, keep_history=True) for k in range(nf)]
    jobs.append(dict(kind="replica", data=data_for(y_true + noise(0)), seed=0))
    for j in range(nr):
        for k in range(ns):
            jobs.append(dict(kind="inverse", seed=k, realization=j, unknown=("CR",), data=data_for(y_true + noise(j)),
                             keep_history=(j == 0)))
    for k in range(ns):
        jobs.append(dict(kind="inverse", seed=k, realization=None, unknown=("CR",), data=data_for(y_true)))
        for case, kw in (("ls+0.01", dict(ls_offset=+LS_OFFSET_DBKM)), ("ls-0.01", dict(ls_offset=-LS_OFFSET_DBKM)),
                         ("Pp+5%", dict(Pp_factor=1 + PP_FACTOR)), ("Pp-5%", dict(Pp_factor=1 - PP_FACTOR))):
            jobs.append(dict(kind="inverse", seed=k, realization=None, unknown=("CR",), data=data_for(y_true),
                             case=case, **kw))
        jobs.append(dict(kind="inverse", seed=k, realization=None, unknown=("CR", "ls"), data=data_for(y_true)))
        for j in range(nj):
            jobs.append(dict(kind="inverse", seed=k, realization=j, unknown=("CR", "ls"),
                             data=data_for(y_true + noise(j))))
    for jb in jobs:
        jb["ref"] = ref
    print(f"{len(jobs)} PINN trainings on {args.workers} worker processes (one thread each) ...", flush=True)
    t0 = time.perf_counter()
    out = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(run_job, jobs, chunksize=1)):
            out.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    res["pool"] = dict(trainings=len(jobs), workers=args.workers, threads_per_worker=1,
                       wall_s=time.perf_counter() - t0,
                       median_training_wall_s_in_pool=float(np.median([r["wall_s"] for r in out])))

    # forward statistics
    fw = [r for r in out if r["job"]["kind"] == "forward"]
    keys = ("rms_signal_dB", "max_signal_dB", "rms_pump_dB", "max_pump_dB")
    res["forward"] = dict(
        seeds=[r["job"]["seed"] for r in fw], per_seed={k: [r[k] for r in fw] for k in keys + ("net_gain_err_dB",)},
        summary={k: dict(mean=float(np.mean([r[k] for r in fw])), min=float(np.min([r[k] for r in fw])),
                         max=float(np.max([r[k] for r in fw]))) for k in keys},
        final_loss=[r["final_loss"] for r in fw],
        seed0=dict(Ps_dBm=fw[0]["Ps_dBm"], Pp_dBm=fw[0]["Pp_dBm"]),
        deviation_signal_dB=[r["dev_signal_dB"] for r in fw], deviation_pump_dB=[r["dev_pump_dB"] for r in fw],
        loss_hist=[r["loss_hist"] for r in fw], hist_every=10)
    rep = [r for r in out if r["job"]["kind"] == "replica"][0]
    res["replica_of_submitted_run"] = dict(CR_err_pct=100 * (rep["CR_factor"] - 1),
                                           submitted_value_pct=-0.49,
                                           note="forward then inverse training with the torch random stream continuing "
                                                "(torch.manual_seed(0)), noise realization of the submitted script")

    # ------------------------------------------------------------------ least squares (main process, one thread)
    print("least squares ...", flush=True)
    n_ls = 200 if args.quick else args.ls_realizations
    ls_runs = [ls_identify(y_true + noise(j)) for j in range(n_ls)]
    ls_cr = np.array([r["CR_factor"] for r in ls_runs])
    ls_free = ls_identify(y_true)
    res["least_squares"] = dict(
        method="scipy.optimize.least_squares, trust-region reflective, 2-point Jacobian, xtol = ftol = 1e-10, "
               "start 0.5 x C_R, forward model models.raman_numerical",
        realizations=n_ls, CR_err=stats_pct(ls_cr), CR_err_first=stats_pct(ls_cr[:nr]),
        CR_err_per_realization_pct=(100 * (ls_cr[:max(nr, nj)] - 1)).tolist(),
        noise_free_CR_err_pct=100 * (ls_free["CR_factor"] - 1),
        seconds=dict(median=float(np.median([r["seconds"] for r in ls_runs])),
                     q25=float(np.percentile([r["seconds"] for r in ls_runs], 25)),
                     q75=float(np.percentile([r["seconds"] for r in ls_runs], 75))),
        bvp_calls=dict(median=float(np.median([r["bvp_calls"] for r in ls_runs])),
                       min=int(np.min([r["bvp_calls"] for r in ls_runs])),
                       max=int(np.max([r["bvp_calls"] for r in ls_runs]))),
        submitted_realization_CR_err_pct=100 * (ls_cr[0] - 1))
    # mis-specified models: exact bias (noise-free) and mean over noisy realizations
    mis = {}
    for case, kw in (("ls+0.01", dict(ls_offset_dBkm=+LS_OFFSET_DBKM)), ("ls-0.01", dict(ls_offset_dBkm=-LS_OFFSET_DBKM)),
                     ("Pp+5%", dict(Pp_factor=1 + PP_FACTOR)), ("Pp-5%", dict(Pp_factor=1 - PP_FACTOR))):
        free = ls_identify(y_true, **kw)
        noisy = np.array([ls_identify(y_true + noise(j), **kw)["CR_factor"] for j in range(min(n_ls, 200))])
        mis[case] = dict(noise_free_CR_err_pct=100 * (free["CR_factor"] - 1), noisy=stats_pct(noisy))
    res["least_squares"]["misspecified"] = mis
    # joint C_R and l_s
    n_jls = min(n_ls, 500)
    jls = [ls_identify(y_true + noise(j), joint=True) for j in range(n_jls)]
    jcr = np.array([r["CR_factor"] for r in jls])
    jl = np.array([r["ls_factor"] for r in jls])
    jfree = ls_identify(y_true, joint=True)
    res["least_squares"]["joint_CR_ls"] = dict(
        realizations=n_jls, CR_err=stats_pct(jcr), ls_err=stats_pct(jl), correlation=float(np.corrcoef(jcr, jl)[0, 1]),
        noise_free=dict(CR_err_pct=100 * (jfree["CR_factor"] - 1), ls_err_pct=100 * (jfree["ls_factor"] - 1)),
        per_realization_CR_err_pct=(100 * (jcr[:nj] - 1)).tolist(), per_realization_ls_err_pct=(100 * (jl[:nj] - 1)).tolist(),
        seconds_median=float(np.median([r["seconds"] for r in jls])),
        bvp_calls_median=float(np.median([r["bvp_calls"] for r in jls])))

    # ------------------------------------------------------------------ PINN inverse statistics
    inv = [r for r in out if r["job"]["kind"] == "inverse" and tuple(r["job"]["unknown"]) == ("CR",)]
    noisy = [r for r in inv if r["job"].get("realization") is not None and "case" not in r["job"]]
    cr_p = np.array([r["CR_factor"] for r in noisy])
    real = np.array([r["job"]["realization"] for r in noisy])
    seeds = np.array([r["job"]["seed"] for r in noisy])
    diff = 100 * (cr_p - ls_cr[real])
    per_real_mean = np.array([np.mean(cr_p[real == j]) for j in range(nr)])
    per_seed_std = [float(np.std(100 * (cr_p[seeds == k] - 1), ddof=1)) for k in range(ns)]
    res["pinn_inverse"] = dict(
        seeds=ns, realizations=nr, CR_err=stats_pct(cr_p),
        CR_err_per_run=[dict(seed=int(a), realization=int(b), err_pct=float(100 * (c - 1)))
                        for a, b, c in zip(seeds, real, cr_p)],
        minus_ls_same_data_pct=dict(mean=float(np.mean(diff)), std=float(np.std(diff, ddof=1)),
                                    max_abs=float(np.max(np.abs(diff)))),
        seed_averaged_CR_err=stats_pct(per_real_mean), per_seed_std_pct=per_seed_std,
        submitted_realization_pct=[float(100 * (c - 1)) for c, j in zip(cr_p, real) if j == 0],
        noise_free_CR_err_pct=[100 * (r["CR_factor"] - 1) for r in inv
                               if r["job"].get("realization") is None and "case" not in r["job"]],
        final_loss=dict(median=float(np.median([r["final_loss"] for r in noisy]))),
        loss_hist_seed0=[r for r in noisy if r["job"]["seed"] == 0 and r["job"]["realization"] == 0][0]["loss_hist"],
        CR_hist_seed0=[r for r in noisy if r["job"]["seed"] == 0 and r["job"]["realization"] == 0][0]["CR_hist"])
    pm = {}
    for case in ("ls+0.01", "ls-0.01", "Pp+5%", "Pp-5%"):
        v = [100 * (r["CR_factor"] - 1) for r in inv if r["job"].get("case") == case]
        pm[case] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, values=v)
    res["pinn_inverse"]["misspecified_noise_free"] = pm
    jt = [r for r in out if r["job"]["kind"] == "inverse" and tuple(r["job"]["unknown"]) == ("CR", "ls")]
    jt_noisy = [r for r in jt if r["job"].get("realization") is not None]
    jt_free = [r for r in jt if r["job"].get("realization") is None]
    jcr_p = np.array([r["CR_factor"] for r in jt_noisy])
    jls_p = np.array([r["ls_factor"] for r in jt_noisy])
    jreal = np.array([r["job"]["realization"] for r in jt_noisy])
    res["pinn_joint_CR_ls"] = dict(
        seeds=ns, realizations=nj, CR_err=stats_pct(jcr_p), ls_err=stats_pct(jls_p),
        minus_ls_same_data_CR_pct=dict(mean=float(np.mean(100 * (jcr_p - jcr[jreal]))),
                                       std=float(np.std(100 * (jcr_p - jcr[jreal]), ddof=1))),
        minus_ls_same_data_ls_pct=dict(mean=float(np.mean(100 * (jls_p - jl[jreal]))),
                                       std=float(np.std(100 * (jls_p - jl[jreal]), ddof=1))),
        noise_free=dict(CR_err_pct=[100 * (r["CR_factor"] - 1) for r in jt_free],
                        ls_err_pct=[100 * (r["ls_factor"] - 1) for r in jt_free]),
        start=dict(CR=0.5, ls=0.9))

    # ------------------------------------------------------------------ timing (sequential benchmark)
    if not args.no_timing:
        res["timing"] = timing(args, ref)
    return res


def timing(args, ref):
    """Training and inference cost with 1 and 4 PyTorch threads (median of repeats), solver costs (one thread)."""
    print("timing ...", flush=True)
    torch.set_default_dtype(torch.float64)
    T = {}
    s401 = torch.linspace(0, 1, 401, dtype=torch.float64).reshape(-1, 1)
    s1 = torch.ones(1, 1, dtype=torch.float64)
    z, Ps_ref, Pp_ref, st = raman_numerical(PP0, PS0)
    y = w2dbm(Ps_ref[IDX]) + noise(0)
    data = (z[IDX] / pc.L_M, np.log(dbm2w(y) / PS0))
    reps = 1 if args.quick else 3
    iters = 600 if args.quick else ITERS
    for nthr in (1, 4):
        torch.set_num_threads(nthr)
        train(seed=0, iters=300)                                     # warm-up (not timed)
        train(seed=0, data=data, unknown=("CR",), iters=300)
        tf, (net_f, _) = pc.repeat_timed(lambda: train(seed=0, iters=iters), repeats=reps)
        ti, _ = pc.repeat_timed(lambda: train(seed=0, data=data, unknown=("CR",), iters=iters), repeats=reps)
        b401 = pc.bench(lambda: infer(net_f, s401))
        b1 = pc.bench(lambda: infer(net_f, s1))
        T[f"threads_{nthr}"] = dict(train_forward=tf, train_inverse=ti, inference_401_positions=b401,
                                    inference_1_position=b1, torch_threads=torch.get_num_threads(), iterations=iters)
        print(f"  {nthr} thread(s): training forward {tf['median_s']:.2f} s, inverse {ti['median_s']:.2f} s, "
              f"inference {b401['median_s'] * 1e6:.0f} us (401) / {b1['median_s'] * 1e6:.0f} us (1)", flush=True)
    torch.set_num_threads(1)
    # solver: cold start (= models.raman_numerical), warm starts, evaluation of the stored solution
    cold = pc.bench(lambda: raman_numerical(PP0, PS0))
    _, _, _, _, sol_near = pc.raman_bvp(PP0 * 0.95, PS0)
    _, _, _, _, sol_same = pc.raman_bvp(PP0, PS0)
    warm_near = pc.bench(lambda: pc.raman_bvp(PP0, PS0, guess=(sol_near.x, sol_near.y)))
    warm_same = pc.bench(lambda: pc.raman_bvp(PP0, PS0, guess=(sol_same.x, sol_same.y)))
    zz = np.linspace(0, pc.L_M, 401)
    stored = pc.bench(lambda: np.exp(sol_same.sol(zz)))
    T["solve_bvp"] = dict(cold_start=cold, warm_start_from_pump_minus_5pct=warm_near,
                          warm_start_from_converged_same_point=warm_same, stored_solution_401_positions=stored,
                          note="models.raman_numerical settings (tol 1e-8, 401 initial nodes); cold start = undepleted "
                               "closed-form initial guess; warm start = converged mesh and solution of the given point")
    # least squares, one identification (same data as the inverse PINN of realization 0)
    T["least_squares_one_identification"] = pc.bench(lambda: ls_identify(y), min_sample_s=0.2, repeats=15)
    # break-even numbers
    for nthr in (1, 4):
        d = T[f"threads_{nthr}"]
        t_train = d["train_forward"]["median_s"]
        t_inf = d["inference_401_positions"]["median_s"]
        t_sol = cold["median_s"]
        d["break_even_vs_cold_solver_same_point"] = (t_train / (t_sol - t_inf)) if t_sol > t_inf else None
        d["training_over_cold_solver"] = t_train / t_sol
        d["inverse_training_over_least_squares"] = d["train_inverse"]["median_s"] / \
            T["least_squares_one_identification"]["median_s"]
    return T


def weight_study(args, path):
    """Supplementary check, merged into the result file: dependence of the PINN estimate of C_R on the weight of the
    data misfit (10 in the submitted version) for noise-free data, for the submitted noise realization and for a signal
    attenuation mis-specified by +0.01 dB/km; 3 seeds per case. Diagnostics: RMS residual of the equations with the
    identified C_R (1001 points) and RMS misfit of the samples."""
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    z, Ps_ref, Pp_ref, st = raman_numerical(PP0, PS0)
    ref = dict(s=(z / pc.L_M).tolist(), Ps_dBm=w2dbm(Ps_ref).tolist(), Pp_dBm=w2dbm(Pp_ref).tolist())
    y_true = w2dbm(Ps_ref[IDX])
    s_smp = z[IDX] / pc.L_M

    def data_for(y_db):
        return (s_smp, np.log(dbm2w(y_db) / PS0))

    weights = (1.0, 10.0, 100.0, 1000.0)
    cases = (("noise-free", dict(), y_true), ("submitted realization", dict(), y_true + noise(0)),
             ("ls+0.01 dB/km, noise-free", dict(ls_offset=LS_OFFSET_DBKM), y_true))
    jobs = []
    for w in weights:
        for name, kw, y in cases:
            for k in range(3):
                jobs.append(dict(kind="inverse", seed=k, realization=None, unknown=("CR",), data=data_for(y),
                                 data_weight=w, study_case=name, ref=ref, **kw))
    print(f"data-weight study: {len(jobs)} trainings ...", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker) as ex:
        out = list(ex.map(run_job, jobs, chunksize=1))
    ls_ref = {"noise-free": res["least_squares"]["noise_free_CR_err_pct"],
              "submitted realization": res["least_squares"]["submitted_realization_CR_err_pct"],
              "ls+0.01 dB/km, noise-free": res["least_squares"]["misspecified"]["ls+0.01"]["noise_free_CR_err_pct"]}
    tab = []
    for w in weights:
        for name, _, _ in cases:
            rr = [r for r in out if r["job"]["data_weight"] == w and r["job"]["study_case"] == name]
            e = [100 * (r["CR_factor"] - 1) for r in rr]
            tab.append(dict(data_weight=w, case=name, CR_err_pct_mean=float(np.mean(e)),
                            CR_err_pct_values=e, least_squares_CR_err_pct=float(ls_ref[name]),
                            residual_rms_mean=float(np.mean([r["residual_rms"] for r in rr])),
                            data_misfit_rms_dB_mean=float(np.mean([r["data_misfit_rms_dB"] for r in rr]))))
            print(f"  weight {w:6.0f}  {name:28s} C_R error {np.mean(e):+.3f} % (LS {ls_ref[name]:+.3f} %), residual "
                  f"RMS {tab[-1]['residual_rms_mean']:.2e}, data misfit {tab[-1]['data_misfit_rms_dB_mean']:.4f} dB",
                  flush=True)
    res["pinn_data_weight_study"] = dict(seeds=3, rows=tab, wall_s=time.perf_counter() - t0, workers=args.workers)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pc.jsonable(res), f, indent=1)


# ----------------------------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------------------------
def plot(res, outdir=FIGDIR):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import style
    style.apply()
    style.math_in_text_font()          # C_R and the +- sign in Arial, like the rest of the text
    C = style.C
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.05, 1.15], height_ratios=[1.55, 1.0],
                  wspace=0.42, hspace=0.12)
    # a) training loss
    ax = fig.add_subplot(gs[:, 0])
    fw = res["forward"]
    it = np.arange(len(fw["loss_hist"][0])) * fw["hist_every"]
    for k, h in enumerate(fw["loss_hist"][1:]):
        ax.semilogy(it, h, color=C[0], lw=0.5, alpha=0.35)
    ax.semilogy(it, fw["loss_hist"][0], color=C[0], lw=1.0, label="Forward")
    hi = res["pinn_inverse"]["loss_hist_seed0"]
    ax.semilogy(np.arange(len(hi)) * fw["hist_every"], hi, color=C[1], lw=1.0, label="Inverse")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Training loss")
    ax.set_xlim(0, it[-1])
    ax.legend(loc="upper right")
    style.label(ax, "a", x=-0.3)
    # b) forward solution and deviation
    ax = fig.add_subplot(gs[0, 1])
    zkm = np.asarray(res["reference_profile"]["z_km"])
    ax.plot(zkm, res["reference_profile"]["Ps_dBm"], color=C[0], lw=1.3, label="Signal, solve_bvp")
    ax.plot(zkm, res["reference_profile"]["Pp_dBm"], color=C[5], lw=1.3, label="Pump, solve_bvp")
    ps, pp = np.asarray(fw["seed0"]["Ps_dBm"]), np.asarray(fw["seed0"]["Pp_dBm"])
    ax.plot(zkm[::25], ps[::25], "o", ms=3.2, mfc="white", mec=C[0], mew=0.9, label="PINN")
    ax.plot(zkm[::25], pp[::25], "s", ms=3.0, mfc="white", mec=C[5], mew=0.9)
    ax.set_ylabel("Power [dBm]")
    ax.tick_params(labelbottom=False)
    ax.set_xlim(0, 50)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.5)
    style.label(ax, "b", x=-0.3)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax)
    for k, (dsg, dpm) in enumerate(zip(fw["deviation_signal_dB"], fw["deviation_pump_dB"])):
        lw, al = (1.0, 1.0) if k == 0 else (0.5, 0.35)
        ax2.plot(zkm, dsg, color=C[0], lw=lw, alpha=al)
        ax2.plot(zkm, dpm, color=C[5], lw=lw, alpha=al)
    ax2.axhline(0, color=style.INK2, lw=0.6)
    ax2.set_xlabel("Position [km]")
    ax2.set_ylabel("Deviation [dB]")
    # c) identified C_R
    ax = fig.add_subplot(gs[:, 2])
    inv = res["pinn_inverse"]
    nr = inv["realizations"]
    ls_err = np.asarray(res["least_squares"]["CR_err_per_realization_pct"][:nr])
    order = np.argsort(ls_err)
    rank = np.empty(nr, int)
    rank[order] = np.arange(nr)
    crb = res["cramer_rao"]["CR_only"]
    ax.axhspan(-crb, crb, color=style.GRID, alpha=0.6, lw=0, label=r"$\pm$Cramér–Rao bound")
    runs = inv["CR_err_per_run"]
    for k in range(inv["seeds"]):
        xs = [rank[r["realization"]] + (k - (inv["seeds"] - 1) / 2) * 0.09 for r in runs if r["seed"] == k]
        ys = [r["err_pct"] for r in runs if r["seed"] == k]
        ax.plot(xs, ys, "o", ms=2.4, color=C[1], alpha=0.75, mew=0, label="PINN, 5 seeds" if k == 0 else None)
    ax.plot(np.arange(nr), ls_err[order], "_", ms=7, mew=1.4, color=style.INK, label="Least squares")
    r0 = rank[0]
    y0 = min([r["err_pct"] for r in runs if r["realization"] == 0] + [ls_err[0]])
    # realization 0 = the noise realization of the first version of this study (np.random.seed(0) in its script)
    ax.annotate("first-version\nrealization", xy=(r0, y0 - 0.05), xytext=(8, -28), textcoords="offset points",
                fontsize=6.3, ha="left", arrowprops=dict(arrowstyle="-", lw=0.6, color=style.INK2), color=style.INK2)
    ax.set_xlabel("Noise realization (sorted)")
    ax.set_ylabel(r"$C_\mathrm{R}$ error [%]")
    ax.set_xlim(-1, nr)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.2)
    style.label(ax, "c", x=-0.3)
    style.save(fig, NAME, outdir=outdir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--forward-seeds", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--realizations", type=int, default=30)
    ap.add_argument("--joint-realizations", type=int, default=10)
    ap.add_argument("--ls-realizations", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--outdir", default=FIGDIR, help="output folder (default ../figures)")
    ap.add_argument("--weight-study", action="store_true",
                    help="only run the data-weight check and merge it into an existing result file")
    args = ap.parse_args()
    path = os.path.join(args.outdir, NAME + "_results.json")
    if args.weight_study:
        torch.set_default_dtype(torch.float64)
        weight_study(args, path)
        return
    if args.plot_only:
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        plot(res, args.outdir)
        return
    torch.set_default_dtype(torch.float64)
    res = compute(args)
    z, Ps_ref, Pp_ref, _ = raman_numerical(PP0, PS0)
    res["reference_profile"] = dict(z_km=(z / 1e3).tolist(), Ps_dBm=w2dbm(Ps_ref).tolist(), Pp_dBm=w2dbm(Pp_ref).tolist())
    res["network"] = dict(inputs="s = z/L", hidden_layers=3, width=32, activation="tanh", parameters=n_parameters(Net()),
                          hard_bc="u = s N1(s), v = (s - 1) N2(s)", optimizer="Adam lr 2e-3, StepLR(2000, 0.3)",
                          iterations=ITERS, collocation_points_per_iteration=256, dtype="float64",
                          data_weight=10.0)
    if "timing" in res:   # keys read by fig_numerics.py for Table S1 (one thread)
        t1 = res["timing"]["threads_1"]
        res.update(train_seconds_forward=t1["train_forward"]["median_s"],
                   train_seconds_inverse=t1["train_inverse"]["median_s"],
                   inference_seconds_1=t1["inference_1_position"]["median_s"],
                   inference_seconds_401=t1["inference_401_positions"]["median_s"],
                   cpu=res["machine"]["cpu"], torch_version=res["machine"]["torch"], torch_threads=1)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pc.jsonable(res), f, indent=1)
    plot(res, args.outdir)
    f_ = res["forward"]["summary"]
    print(f"forward RMS signal {f_['rms_signal_dB']['mean']:.4f} dB (range {f_['rms_signal_dB']['min']:.4f}-"
          f"{f_['rms_signal_dB']['max']:.4f}), max signal up to {f_['max_signal_dB']['max']:.4f} dB; RMS pump "
          f"{f_['rms_pump_dB']['mean']:.4f} dB, max pump up to {f_['max_pump_dB']['max']:.4f} dB")
    pi, ls = res["pinn_inverse"]["CR_err"], res["least_squares"]["CR_err"]
    print(f"C_R error PINN {pi['mean']:+.2f} +- {pi['std']:.2f} % (range {pi['min']:+.2f} to {pi['max']:+.2f}); "
          f"least squares {ls['mean']:+.2f} +- {ls['std']:.2f} %; Cramer-Rao {res['cramer_rao']['CR_only']:.2f} %")


if __name__ == "__main__":
    main()
