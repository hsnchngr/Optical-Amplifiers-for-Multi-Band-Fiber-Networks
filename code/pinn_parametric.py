"""Figure S4 (Supporting Information, Section S5): parameterized physics-informed neural network (PINN) for the
counter-pumped distributed Raman span, tested across operating conditions.

  a) maximum deviation of the signal and pump profiles from the collocation solver over pump power and signal load
     (card attenuations), with the training box and the 8 x 8 test slice;
  b) distribution of the maximum profile deviation inside the training box and 10-20 % outside it (and for a supervised
     network of the same size trained on solver solutions);
  c) cost: training, inference per operating point (single call and batched), and the collocation solver.

Scaled problem (pinn_common.py): du/ds = kappa_p exp(v) - delta_s, dv/ds = delta_p + kappa_s exp(u), u(0) = 0,
v(1) = 0, with s = z/L, kappa_p = C_R P_p(L) L, kappa_s = (nu_p/nu_s) C_R P_s(0) L, delta_s = l_s L, delta_p = l_p L.
Network input (s, kappa_p, kappa_s, delta_s, delta_p) (each mapped to [-1, 1], kappa_s logarithmically); output
(N_1, N_2); hard boundary conditions and residual learning around the undepleted closed form (Equation (12)):
    u = u_0(s) + s kappa_s N_1,     v = v_0(s) + (s - 1) kappa_s N_2,
where u_0, v_0 is the undepleted solution; the factor kappa_s makes the correction vanish where pump depletion
vanishes. The derivative with respect to s is propagated through the network in the forward pass (exact, no
finite differences). Loss: mean squared residuals of the two equations at quasi-random (scrambled Sobol) collocation
points of the 5-dimensional domain; Adam (float32, fresh points every iteration, exponentially decaying learning rate),
then L-BFGS (float64, strong-Wolfe line search, collocation set redrawn between chunks). No solver data are used for
training the PINN.

Training box: derived from a 50 km G.652 span (card RAMAN-G652: C_R = 0.406 1/(W km), L = 50 km) with pump powers
0.1-0.9 W, signal loads -13 to +21 dBm, l_s = 0.165-0.25 dB/km and l_p = 0.22-0.32 dB/km.
Tests: (i) held-out parameter sets drawn uniformly inside the box (pseudo-random, independent of the training points);
(ii) physical slice of 8 pump powers x 8 signal loads at the card attenuations, including the single-instance point
(0.5 W, +16 dBm); (iii) parameter sets with one parameter (or all four) 10-20 % beyond the box bounds. Reference:
scipy.integrate.solve_bvp on the scaled problem (tolerance 1e-8, error below 1e-8 dB). A validation set (256 sets) is
used only to monitor training; the test sets are evaluated once with the final networks.
Baseline: a supervised network of the same architecture and ansatz trained on collocation solutions (mean squared
error of u and v), whose training cost includes generating the solutions.

Outputs: ../figures/FigS4_pinn_parametric.png/.pdf, ../figures/FigS4_pinn_parametric_results.json,
         ../figures/pinn_models/*.pt (trained weights; the evaluation and timing stages load them)
Usage  : python pinn_parametric.py [--stage all|train|eval|timing|plot] [--seeds 3] [--threads 4] [--no-supervised]
"""
import os
import sys

# One BLAS/OpenMP thread for NumPy/SciPy, as in Table S1 (set before NumPy is imported). PyTorch threads are set
# explicitly with torch.set_num_threads().
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import time

import numpy as np
import torch

import pinn_common as pc
from models import dbm2w, w2dbm

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
NAME = "FigS4_pinn_parametric"

# ----------------------------------------------------------------------------------------------------------------
# Training box from physical ranges (card C_R and length)
# ----------------------------------------------------------------------------------------------------------------
PHYS = dict(Pp_W=(0.1, 0.9), Ps_dBm=(-13.0, 21.0), ls_dBkm=(0.165, 0.25), lp_dBkm=(0.22, 0.32))
LO = np.array([float(x) for x in pc.groups(PHYS["Pp_W"][0], PHYS["Ps_dBm"][0], PHYS["ls_dBkm"][0], PHYS["lp_dBkm"][0])])
HI = np.array([float(x) for x in pc.groups(PHYS["Pp_W"][1], PHYS["Ps_dBm"][1], PHYS["ls_dBkm"][1], PHYS["lp_dBkm"][1])])
PNAMES = ("kappa_p", "kappa_s", "delta_s", "delta_p")
SLICE_PP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]               # W
SLICE_PS = [-13.0, -8.0, -3.0, 2.0, 7.0, 12.0, 16.0, 21.0]          # dBm
N401 = 401

DEFAULT = dict(width=64, depth=4, batch=4096, adam_iters=40000, lr0=2e-3, lr_end=2e-5, lbfgs_chunks=4,
               lbfgs_iters=500, lbfgs_points=32768, sup_sets=4096)


def to_unit(theta):
    """theta (n, 4) = (kappa_p, kappa_s, delta_s, delta_p) -> coordinates in the unit box (kappa_s logarithmic)."""
    th = np.atleast_2d(theta)
    x = (th - LO) / (HI - LO)
    x[:, 1] = (np.log(th[:, 1]) - np.log(LO[1])) / (np.log(HI[1]) - np.log(LO[1]))
    return x


def from_unit(x):
    x = np.atleast_2d(x)
    th = LO + x * (HI - LO)
    th[:, 1] = np.exp(np.log(LO[1]) + x[:, 1] * (np.log(HI[1]) - np.log(LO[1])))
    return th


def from_unit_t(x):
    lo = torch.as_tensor(LO, dtype=x.dtype)
    hi = torch.as_tensor(HI, dtype=x.dtype)
    th = lo + x * (hi - lo)
    ks = torch.exp(np.log(LO[1]) + x[:, 1:2] * float(np.log(HI[1]) - np.log(LO[1])))
    return torch.cat([th[:, :1], ks, th[:, 2:]], dim=1)


# ----------------------------------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------------------------------
class PNet(torch.nn.Module):
    """MLP (5 -> width x depth tanh -> 2) returning the outputs and their derivative with respect to s."""

    def __init__(self, width=64, depth=4):
        super().__init__()
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(5 if k == 0 else width, width) for k in range(depth)])
        self.out = torch.nn.Linear(width, 2)

    def forward(self, s, xn, with_ds=True):
        h = torch.cat([2.0 * s - 1.0, xn], dim=1)
        dh = None
        for k, lin in enumerate(self.hidden):
            z = lin(h)
            if with_ds:
                dz = 2.0 * lin.weight[:, 0] if k == 0 else dh @ lin.weight.t()
            h = torch.tanh(z)
            if with_ds:
                dh = (1.0 - h * h) * dz
        o = self.out(h)
        return o, (dh @ self.out.weight.t() if with_ds else None)


def n_parameters(net):
    return int(sum(p.numel() for p in net.parameters()))


def fields(net, s, theta, xn, with_residual=True):
    """u, v (and the residuals r1, r2 of the two equations) for column tensors s, theta (n, 4), xn = 2 x - 1."""
    kp, ks, ds, dp = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4]
    o, do = net(s, xn, with_residual)
    n1, n2 = o[:, :1], o[:, 1:]
    v0 = -dp * (1.0 - s)
    u0 = kp * torch.exp(-dp) * torch.expm1(dp * s) / dp - ds * s
    u = u0 + s * ks * n1
    v = v0 + (s - 1.0) * ks * n2
    if not with_residual:
        return u, v
    r1 = ks * (n1 + s * do[:, :1]) - kp * torch.exp(v0) * torch.expm1((s - 1.0) * ks * n2)
    r2 = ks * (n2 + (s - 1.0) * do[:, 1:] - torch.exp(u))
    return u, v, r1, r2


def sample(engine, n, dtype):
    x = engine.draw(n).to(dtype)
    xu = x[:, 1:]
    return x[:, :1], from_unit_t(xu), 2.0 * xu - 1.0


def predict(net, thetas, s_grid=None, dtype=torch.float64, chunk_rows=200000):
    """u, v [neper] of shape (m, n) for parameter sets thetas (m, 4) on s_grid (default 401 points)."""
    s_grid = np.linspace(0, 1, N401) if s_grid is None else np.asarray(s_grid)
    th = np.atleast_2d(thetas)
    m, n = th.shape[0], s_grid.size
    xn = 2.0 * to_unit(th) - 1.0
    U, V = np.empty((m, n)), np.empty((m, n))
    per = max(1, chunk_rows // n)
    with torch.no_grad():
        for a in range(0, m, per):
            b = min(m, a + per)
            k = b - a
            s_t = torch.as_tensor(np.tile(s_grid, k), dtype=dtype).reshape(-1, 1)
            th_t = torch.as_tensor(np.repeat(th[a:b], n, axis=0), dtype=dtype)
            xn_t = torch.as_tensor(np.repeat(xn[a:b], n, axis=0), dtype=dtype)
            u, v = fields(net, s_t, th_t, xn_t, with_residual=False)
            U[a:b] = u.numpy().reshape(k, n)
            V[a:b] = v.numpy().reshape(k, n)
    return U, V


# ----------------------------------------------------------------------------------------------------------------
# Reference solutions and error metrics
# ----------------------------------------------------------------------------------------------------------------
def references(thetas, tol=1e-8):
    th = np.atleast_2d(thetas)
    U, V, st, nodes, sec = [], [], [], [], []
    for row in th:
        t0 = time.perf_counter()
        r = pc.bvp_scaled(*row, n=N401, tol=tol)
        sec.append(time.perf_counter() - t0)
        U.append(r["u"]); V.append(r["v"]); st.append(r["status"]); nodes.append(r["nodes"])
    return dict(u=np.array(U), v=np.array(V), status=np.array(st), nodes=np.array(nodes), seconds=np.array(sec))


def metrics(U, V, ref):
    du = (U - ref["u"]) * pc.NP2DB
    dv = (V - ref["v"]) * pc.NP2DB
    return dict(rms_signal_dB=np.sqrt(np.mean(du ** 2, axis=1)), max_signal_dB=np.max(np.abs(du), axis=1),
                rms_pump_dB=np.sqrt(np.mean(dv ** 2, axis=1)), max_pump_dB=np.max(np.abs(dv), axis=1),
                net_gain_err_dB=du[:, -1],
                max_profile_dB=np.maximum(np.max(np.abs(du), axis=1), np.max(np.abs(dv), axis=1)))


def summarize(m):
    out = {}
    for k, v in m.items():
        a = np.abs(v) if k == "net_gain_err_dB" else v
        out[k] = pc.percentiles(a)
        out[k]["mean"] = float(np.mean(a))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Test sets
# ----------------------------------------------------------------------------------------------------------------
def test_sets(n_inside=2000, n_one_out=1000, n_all_out=200, seed=2026):
    rng = np.random.default_rng(seed)
    inside = from_unit(rng.random((n_inside, 4)))
    one_out, tags = [], []
    per = n_one_out // 8
    for p in range(4):
        for side in ("low", "high"):
            x = rng.random((per, 4))
            th = from_unit(x)
            f = rng.uniform(1.1, 1.2, per) if side == "high" else rng.uniform(0.8, 0.9, per)
            th[:, p] = (HI[p] if side == "high" else LO[p]) * f
            one_out.append(th)
            tags += [f"{PNAMES[p]}_{side}"] * per
    one_out = np.vstack(one_out)
    all_out = np.empty((n_all_out, 4))
    for p in range(4):
        high = rng.random(n_all_out) < 0.5
        f = np.where(high, rng.uniform(1.1, 1.2, n_all_out), rng.uniform(0.8, 0.9, n_all_out))
        all_out[:, p] = np.where(high, HI[p], LO[p]) * f
    pp, ps = np.meshgrid(SLICE_PP, SLICE_PS, indexing="ij")
    slice_th = np.column_stack(np.broadcast_arrays(*pc.groups(pp.ravel(), ps.ravel())))
    val = from_unit(torch.quasirandom.SobolEngine(4, scramble=True, seed=12345).draw(256).double().numpy())
    return dict(inside=inside, one_out=one_out, one_out_tags=np.array(tags), all_out=all_out, slice=slice_th,
                slice_Pp_W=pp.ravel(), slice_Ps_dBm=ps.ravel(), val=val)


def map_grid(n_pp=45, n_ps=41):
    pp = np.linspace(0.05, 1.1, n_pp)
    ps = np.linspace(-16.0, 24.0, n_ps)
    P, S = np.meshgrid(pp, ps, indexing="ij")
    return pp, ps, np.column_stack(np.broadcast_arrays(*pc.groups(P.ravel(), S.ravel())))


# ----------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------
def val_error(net, val, ref_val, dtype):
    sg = np.linspace(0, 1, 101)
    U, V = predict(net, val, sg, dtype=dtype)
    du = np.abs(U - ref_val["u"][:, ::4]) * pc.NP2DB
    dv = np.abs(V - ref_val["v"][:, ::4]) * pc.NP2DB
    mx = np.maximum(du.max(axis=1), dv.max(axis=1))
    return dict(p50=float(np.median(mx)), p95=float(np.percentile(mx, 95)), max=float(mx.max()))


def train_pinn(seed, cfg, val=None, ref_val=None, log_every=2000, verbose=True):
    torch.manual_seed(seed)
    net = PNet(cfg["width"], cfg["depth"]).to(torch.float32)
    eng = torch.quasirandom.SobolEngine(5, scramble=True, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr0"])
    gamma = (cfg["lr_end"] / cfg["lr0"]) ** (1.0 / cfg["adam_iters"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    hist = dict(adam_iter=[], adam_loss=[], val=[], lbfgs_loss=[], lbfgs_val=[])
    t0 = time.perf_counter()
    acc = 0.0
    for it in range(cfg["adam_iters"]):
        s, th, xn = sample(eng, cfg["batch"], torch.float32)
        u, v, r1, r2 = fields(net, s, th, xn)
        loss = (r1 ** 2).mean() + (r2 ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        acc += loss.item()
        if (it + 1) % log_every == 0:
            hist["adam_iter"].append(it + 1)
            hist["adam_loss"].append(acc / log_every)
            acc = 0.0
            if val is not None:
                hist["val"].append(val_error(net, val, ref_val, torch.float32))
            if verbose:
                print(f"    seed {seed} Adam {it + 1}: loss {hist['adam_loss'][-1]:.3e}"
                      + (f", val max dev p50/p95/max {hist['val'][-1]['p50']:.4f}/{hist['val'][-1]['p95']:.4f}/"
                         f"{hist['val'][-1]['max']:.4f} dB" if val is not None else "")
                      + f" ({time.perf_counter() - t0:.0f} s)", flush=True)
    t_adam = time.perf_counter() - t0
    net = net.double()
    eng2 = torch.quasirandom.SobolEngine(5, scramble=True, seed=seed + 1000)
    n_eval = 0
    for ch in range(cfg["lbfgs_chunks"]):
        s, th, xn = sample(eng2, cfg["lbfgs_points"], torch.float64)
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=cfg["lbfgs_iters"],
                                max_eval=int(1.25 * cfg["lbfgs_iters"]), history_size=50, tolerance_grad=1e-12,
                                tolerance_change=1e-16, line_search_fn="strong_wolfe")
        cnt = [0]

        def closure():
            cnt[0] += 1
            opt.zero_grad()
            u, v, r1, r2 = fields(net, s, th, xn)
            l_ = (r1 ** 2).mean() + (r2 ** 2).mean()
            l_.backward()
            return l_

        opt.step(closure)
        n_eval += cnt[0]
        lval = float(closure().item())
        hist["lbfgs_loss"].append(lval)
        if val is not None:
            hist["lbfgs_val"].append(val_error(net, val, ref_val, torch.float64))
        if verbose:
            print(f"    seed {seed} L-BFGS chunk {ch + 1}: loss {lval:.3e}, evaluations {cnt[0]}"
                  + (f", val max dev p50/p95/max {hist['lbfgs_val'][-1]['p50']:.4f}/{hist['lbfgs_val'][-1]['p95']:.4f}/"
                     f"{hist['lbfgs_val'][-1]['max']:.4f} dB" if val is not None else "")
                  + f" ({time.perf_counter() - t0:.0f} s)", flush=True)
    wall = time.perf_counter() - t0
    info = dict(seed=seed, wall_s=wall, adam_s=t_adam, lbfgs_s=wall - t_adam, lbfgs_evaluations=n_eval,
                threads=torch.get_num_threads(), hist=hist, cfg=cfg, parameters=n_parameters(net))
    return net, info


def train_supervised(seed, cfg, data, val=None, ref_val=None, log_every=2000, verbose=True):
    """Same network and ansatz trained on collocation solutions: data = dict(theta (m,4), u (m,401), v (m,401))."""
    torch.manual_seed(seed)
    net = PNet(cfg["width"], cfg["depth"]).to(torch.float32)
    th_all = torch.as_tensor(data["theta"])
    xn_all = torch.as_tensor(2.0 * to_unit(data["theta"]) - 1.0)
    U = torch.as_tensor(data["u"])
    V = torch.as_tensor(data["v"])
    m = th_all.shape[0]
    sg = torch.linspace(0, 1, N401, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr0"])
    gamma = (cfg["lr_end"] / cfg["lr0"]) ** (1.0 / cfg["adam_iters"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    hist = dict(adam_iter=[], adam_loss=[], val=[], lbfgs_loss=[], lbfgs_val=[])

    def batch(n, dtype):
        i = torch.randint(0, m, (n,), generator=gen)
        j = torch.randint(0, N401, (n,), generator=gen)
        return (sg[j].reshape(-1, 1).to(dtype), th_all[i].to(dtype), xn_all[i].to(dtype),
                U[i, j].reshape(-1, 1).to(dtype), V[i, j].reshape(-1, 1).to(dtype))

    t0 = time.perf_counter()
    acc = 0.0
    for it in range(cfg["adam_iters"]):
        s, th, xn, ut, vt = batch(cfg["batch"], torch.float32)
        u, v = fields(net, s, th, xn, with_residual=False)
        loss = ((u - ut) ** 2).mean() + ((v - vt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        acc += loss.item()
        if (it + 1) % log_every == 0:
            hist["adam_iter"].append(it + 1)
            hist["adam_loss"].append(acc / log_every)
            acc = 0.0
            if val is not None:
                hist["val"].append(val_error(net, val, ref_val, torch.float32))
            if verbose:
                print(f"    supervised seed {seed} Adam {it + 1}: loss {hist['adam_loss'][-1]:.3e}"
                      + (f", val max dev p50/p95/max {hist['val'][-1]['p50']:.4f}/{hist['val'][-1]['p95']:.4f}/"
                         f"{hist['val'][-1]['max']:.4f} dB" if val is not None else "")
                      + f" ({time.perf_counter() - t0:.0f} s)", flush=True)
    t_adam = time.perf_counter() - t0
    net = net.double()
    n_eval = 0
    for ch in range(cfg["lbfgs_chunks"]):
        s, th, xn, ut, vt = batch(cfg["lbfgs_points"], torch.float64)
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=cfg["lbfgs_iters"],
                                max_eval=int(1.25 * cfg["lbfgs_iters"]), history_size=50, tolerance_grad=1e-12,
                                tolerance_change=1e-16, line_search_fn="strong_wolfe")
        cnt = [0]

        def closure():
            cnt[0] += 1
            opt.zero_grad()
            u, v = fields(net, s, th, xn, with_residual=False)
            l_ = ((u - ut) ** 2).mean() + ((v - vt) ** 2).mean()
            l_.backward()
            return l_

        opt.step(closure)
        n_eval += cnt[0]
        lval = float(closure().item())
        hist["lbfgs_loss"].append(lval)
        if val is not None:
            hist["lbfgs_val"].append(val_error(net, val, ref_val, torch.float64))
        if verbose:
            print(f"    supervised seed {seed} L-BFGS chunk {ch + 1}: loss {lval:.3e}"
                  + (f", val max dev p50/p95/max {hist['lbfgs_val'][-1]['p50']:.4f}/"
                     f"{hist['lbfgs_val'][-1]['p95']:.4f}/{hist['lbfgs_val'][-1]['max']:.4f} dB" if val is not None else "")
                  + f" ({time.perf_counter() - t0:.0f} s)", flush=True)
    wall = time.perf_counter() - t0
    info = dict(seed=seed, wall_s=wall, adam_s=t_adam, lbfgs_s=wall - t_adam, lbfgs_evaluations=n_eval,
                threads=torch.get_num_threads(), hist=hist, cfg=cfg, parameters=n_parameters(net))
    return net, info


def model_path(outdir, kind, seed):
    return os.path.join(outdir, "pinn_models", f"{kind}_seed{seed}.pt")


def save_model(path, net, info):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(state_dict=net.state_dict(), info=pc.jsonable(info), box_lo=LO.tolist(), box_hi=HI.tolist()), path)


def load_model(path):
    d = torch.load(path, weights_only=False)
    cfg = d["info"]["cfg"]
    net = PNet(cfg["width"], cfg["depth"]).double()
    net.load_state_dict(d["state_dict"])
    net.eval()
    return net, d["info"]


# ----------------------------------------------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------------------------------------------
def stage_train(args, cfg, T):
    torch.set_num_threads(args.threads)
    ref_val = references(T["val"])
    log = {}
    for seed in range(args.seeds):
        p = model_path(args.outdir, "pinn", seed)
        if os.path.exists(p) and not args.retrain:
            print(f"  PINN seed {seed}: {p} exists (use --retrain to train again)")
            continue
        print(f"  training PINN seed {seed} ({args.threads} threads) ...", flush=True)
        net, info = train_pinn(seed, cfg, T["val"], ref_val)
        save_model(p, net, info)
        log[f"pinn_seed{seed}"] = info["wall_s"]
    if not args.no_supervised:
        for seed in range(args.sup_seeds):
            p = model_path(args.outdir, "supervised", seed)
            if os.path.exists(p) and not args.retrain:
                print(f"  supervised seed {seed}: {p} exists")
                continue
            # training data: collocation solutions on a scrambled Sobol set of parameter sets (independent of tests)
            x = torch.quasirandom.SobolEngine(4, scramble=True, seed=777 + seed).draw(cfg["sup_sets"]).double().numpy()
            th = from_unit(x)
            t0 = time.perf_counter()
            ref = references(th)
            t_data = time.perf_counter() - t0
            print(f"  supervised data: {th.shape[0]} solve_bvp solutions in {t_data:.1f} s", flush=True)
            scfg = dict(cfg, adam_iters=args.sup_adam_iters or cfg["adam_iters"])
            net, info = train_supervised(seed, scfg, dict(theta=th, u=ref["u"], v=ref["v"]), T["val"], ref_val)
            info["data_generation_s"] = t_data
            info["data_sets"] = int(th.shape[0])
            info["data_failed"] = int(np.sum(ref["status"] != 0))
            save_model(p, net, info)
    return log


def evaluate_all(args, T):
    res = {}
    t0 = time.perf_counter()
    R = {k: references(T[k]) for k in ("inside", "one_out", "all_out", "slice")}
    pp, ps, th_map = map_grid()
    R["map"] = references(th_map)
    print(f"  reference solutions: {time.perf_counter() - t0:.1f} s", flush=True)
    res["reference"] = {k: dict(sets=int(R[k]["u"].shape[0]), failed=int(np.sum(R[k]["status"] != 0)),
                                nodes_median=float(np.median(R[k]["nodes"])), nodes_max=int(np.max(R[k]["nodes"])),
                                seconds_median=float(np.median(R[k]["seconds"])))
                        for k in R}
    # closed-form (undepleted) deviation of the inside sets: measure of pump depletion
    s = np.linspace(0, 1, N401)
    dep = np.array([abs(pc.undepleted(1.0, th[0], th[2], th[3])[0] - R["inside"]["u"][i, -1]) * pc.NP2DB
                    for i, th in enumerate(T["inside"])])
    res["inside_depletion_net_gain_dB"] = pc.percentiles(dep, (0, 50, 95, 100))
    kinds = [("pinn", s_) for s_ in range(args.seeds)]
    if not args.no_supervised:
        kinds += [("supervised", s_) for s_ in range(args.sup_seeds)]
    res["models"] = {}
    for kind, seed in kinds:
        p = model_path(args.outdir, kind, seed)
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        net, info = load_model(p)
        out = dict(training=dict(wall_s=info["wall_s"], adam_s=info["adam_s"], lbfgs_s=info["lbfgs_s"],
                                 threads=info["threads"], parameters=info["parameters"],
                                 lbfgs_evaluations=info["lbfgs_evaluations"], hist=info["hist"],
                                 data_generation_s=info.get("data_generation_s"), data_sets=info.get("data_sets")))
        per = {}
        for k in ("inside", "one_out", "all_out", "slice", "map"):
            th = T[k] if k != "map" else th_map
            U, V = predict(net, th)
            per[k] = metrics(U, V, R[k])
            if k != "map":
                out[k] = summarize(per[k])
        # degradation with pump depletion (inside sets)
        bins = [0, 0.1, 1.0, 3.0, 10.0, 1e9]
        out["inside_by_depletion"] = []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (dep >= a) & (dep < b)
            if sel.any():
                out["inside_by_depletion"].append(dict(depletion_dB=[a, b if b < 1e8 else None], sets=int(sel.sum()),
                                                       **{q: v for q, v in pc.percentiles(per["inside"]["max_profile_dB"][sel]).items()}))
        # one parameter outside, per parameter and side
        tags = T["one_out_tags"]
        out["one_out_by_parameter"] = {t: pc.percentiles(per["one_out"]["max_profile_dB"][tags == t])
                                       for t in sorted(set(tags))}
        out["slice_table"] = dict(Pp_W=T["slice_Pp_W"].tolist(), Ps_dBm=T["slice_Ps_dBm"].tolist(),
                                  max_signal_dB=per["slice"]["max_signal_dB"].tolist(),
                                  max_pump_dB=per["slice"]["max_pump_dB"].tolist(),
                                  net_gain_err_dB=per["slice"]["net_gain_err_dB"].tolist())
        ci = int(np.argmin(np.abs(T["slice_Pp_W"] - 0.5) + np.abs(T["slice_Ps_dBm"] - 16.0)))
        out["single_instance_point"] = {k: float(per["slice"][k][ci]) for k in per["slice"]}
        for k in ("inside", "one_out", "all_out"):
            wi = np.argsort(per[k]["max_profile_dB"])[::-1][:5]
            out[f"{k}_worst5"] = [dict(groups=dict(zip(PNAMES, T[k][i].tolist())),
                                       physical=dict(zip(("Pp_W", "Ps_dBm", "ls_dBkm", "lp_dBkm"),
                                                         [float(x) for x in pc.physical(*T[k][i])])),
                                       max_signal_dB=float(per[k]["max_signal_dB"][i]),
                                       max_pump_dB=float(per[k]["max_pump_dB"][i]),
                                       net_gain_err_dB=float(per[k]["net_gain_err_dB"][i]),
                                       depletion_dB=float(dep[i]) if k == "inside" else None) for i in wi]
        out["map"] = dict(Pp_W=pp.tolist(), Ps_dBm=ps.tolist(),
                          max_signal_dB=per["map"]["max_signal_dB"].reshape(pp.size, ps.size).tolist(),
                          max_pump_dB=per["map"]["max_pump_dB"].reshape(pp.size, ps.size).tolist(),
                          max_profile_dB=per["map"]["max_profile_dB"].reshape(pp.size, ps.size).tolist())
        out["inside_values_max_profile_dB"] = per["inside"]["max_profile_dB"].tolist()
        out["one_out_values_max_profile_dB"] = per["one_out"]["max_profile_dB"].tolist()
        out["all_out_values_max_profile_dB"] = per["all_out"]["max_profile_dB"].tolist()
        res["models"][f"{kind}_seed{seed}"] = out
        i_ = out["inside"]
        print(f"  {kind} seed {seed}: inside max profile dev p50/p95/p100 {i_['max_profile_dB']['p50']:.4f}/"
              f"{i_['max_profile_dB']['p95']:.4f}/{i_['max_profile_dB']['p100']:.4f} dB; net gain |err| p95 "
              f"{i_['net_gain_err_dB']['p95']:.4f} dB; one-out p95 {out['one_out']['max_profile_dB']['p95']:.4f}; "
              f"all-out p95 {out['all_out']['max_profile_dB']['p95']:.4f}", flush=True)
    # pooled over PINN seeds
    pin = [v for k, v in res["models"].items() if k.startswith("pinn")]
    if pin:
        res["pinn_pooled"] = {k: pc.percentiles(np.concatenate([v[f"{k}_values_max_profile_dB"] for v in pin]))
                              for k in ("inside", "one_out", "all_out")}
    return res


def stage_timing(args, T, res):
    """Inference (1 and 4 threads, single call and batch of 1000 operating points) and solver costs (one thread)."""
    out = {}
    net, info = load_model(model_path(args.outdir, "pinn", 0))
    card = np.array([pc.CARD_GROUPS])
    batch = T["inside"][:1000]
    s401 = np.linspace(0, 1, N401)
    s1 = np.array([1.0])

    def infer(th, sg, dtype):
        U, V = predict(net if dtype == torch.float64 else net32, th, sg, dtype=dtype, chunk_rows=10 ** 7)
        return U * pc.NP2DB, V * pc.NP2DB

    # solver (models.raman_numerical settings, physical units) at the card point and over the batch; timed first,
    # before PyTorch has started its OpenMP threads (idle-spinning threads would raise the process-CPU/wall ratio)
    Pp, Psd, lsd, lpd = pc.physical(*card[0])
    q = pc.card_with(ls_dBkm=lsd, lp_dBkm=lpd)
    out["solve_bvp_cold_card"] = pc.bench(lambda: pc.raman_bvp(Pp, dbm2w(Psd), q))
    _, _, _, _, sol_near = pc.raman_bvp(Pp * 0.95, dbm2w(Psd), q)
    out["solve_bvp_warm_card_from_pump_minus_5pct"] = pc.bench(
        lambda: pc.raman_bvp(Pp, dbm2w(Psd), q, guess=(sol_near.x, sol_near.y)))
    t_cold, st_cold, err = [], [], []
    for th in batch:
        Pp_, Ps_, ls_, lp_ = pc.physical(*th)
        q_ = pc.card_with(ls_dBkm=ls_, lp_dBkm=lp_)
        t0 = time.perf_counter()
        z, Ps, Ppz, st, _ = pc.raman_bvp(Pp_, dbm2w(Ps_), q_)
        t_cold.append(time.perf_counter() - t0)
        st_cold.append(int(st))
    out["solve_bvp_cold_batch1000"] = dict(mean_s=float(np.mean(t_cold)), median_s=float(np.median(t_cold)),
                                           p95_s=float(np.percentile(t_cold, 95)), max_s=float(np.max(t_cold)),
                                           total_s=float(np.sum(t_cold)), failed=int(np.sum(np.array(st_cold) != 0)))
    # warm start along the 8 x 8 slice (continuation in signal load at each pump power)
    tw, tc = [], []
    for Pp_ in SLICE_PP:
        prev = None
        for Ps_ in SLICE_PS:
            t0 = time.perf_counter()
            z, Ps, Ppz, st, sol = pc.raman_bvp(Pp_, dbm2w(Ps_), guess=prev)
            (tw if prev is not None else tc).append(time.perf_counter() - t0)
            prev = (sol.x, sol.y)
    out["solve_bvp_warm_slice_continuation"] = dict(median_s=float(np.median(tw)), mean_s=float(np.mean(tw)),
                                                    first_point_cold_median_s=float(np.median(tc)))
    net32 = PNet(info["cfg"]["width"], info["cfg"]["depth"]).float()
    net32.load_state_dict({k: v.float() for k, v in net.state_dict().items()})
    for nthr in (1, 4):
        torch.set_num_threads(nthr)
        d = {}
        for dt_name, dt in (("float64", torch.float64), ("float32", torch.float32)):
            d[dt_name] = dict(
                single_401=pc.bench(lambda: infer(card, s401, dt)),
                single_1=pc.bench(lambda: infer(card, s1, dt)),
                batch1000_401=pc.bench(lambda: infer(batch, s401, dt), min_sample_s=0.2, repeats=7),
                batch1000_1=pc.bench(lambda: infer(batch, s1, dt)))
            for k in ("batch1000_401", "batch1000_1"):
                d[dt_name][k]["per_operating_point_s"] = d[dt_name][k]["median_s"] / batch.shape[0]
        out[f"inference_threads_{nthr}"] = d
        print(f"  inference ({nthr} thr, float64): single 401 {d['float64']['single_401']['median_s'] * 1e6:.0f} us, "
              f"single 1 {d['float64']['single_1']['median_s'] * 1e6:.0f} us, batch per point 401 "
              f"{d['float64']['batch1000_401']['per_operating_point_s'] * 1e6:.1f} us, 1 "
              f"{d['float64']['batch1000_1']['per_operating_point_s'] * 1e6:.2f} us", flush=True)
    torch.set_num_threads(1)
    # training cost per iteration with 1 thread (extrapolated training time) and with the training thread count
    cfg = info["cfg"]
    per_it = {}
    for nthr in (1, args.threads):
        torch.set_num_threads(nthr)
        c = dict(cfg, adam_iters=200, lbfgs_chunks=1, lbfgs_iters=20)
        train_pinn(0, dict(c, adam_iters=20, lbfgs_iters=2), verbose=False)            # warm-up
        _, inf = train_pinn(0, c, verbose=False)
        per_it[f"threads_{nthr}"] = dict(adam_s_per_iter=inf["adam_s"] / c["adam_iters"],
                                         lbfgs_s_per_eval=inf["lbfgs_s"] / max(1, inf["lbfgs_evaluations"]))
    torch.set_num_threads(1)
    out["training_per_iteration"] = per_it
    # PINN training time on one thread, extrapolated from the per-iteration cost and the iteration/evaluation counts
    # of the actual training runs (the trainings themselves ran with args.threads threads)
    p1 = per_it["threads_1"]
    out["pinn_training_1thread_extrapolated_s"] = {
        k: p1["adam_s_per_iter"] * v["training"].get("adam_iters", cfg["adam_iters"])
        + p1["lbfgs_s_per_eval"] * v["training"]["lbfgs_evaluations"]
        for k, v in res["models"].items() if k.startswith("pinn")}
    pT = per_it[f"threads_{args.threads}"]
    out["pinn_training_estimated_from_benchmark_s"] = {
        k: pT["adam_s_per_iter"] * v["training"].get("adam_iters", cfg["adam_iters"])
        + pT["lbfgs_s_per_eval"] * v["training"]["lbfgs_evaluations"]
        for k, v in res["models"].items() if k.startswith("pinn")}
    out["pinn_training_estimated_threads"] = args.threads
    # break-even numbers N* = T_train / (t_solver - t_inference)
    be = {}
    t_sol = out["solve_bvp_cold_batch1000"]["mean_s"]
    for kind in [k for k in res["models"] if k.startswith("pinn")] + \
                [k for k in res["models"] if k.startswith("supervised")]:
        tr = res["models"][kind]["training"]
        T_train = tr["wall_s"] + (tr.get("data_generation_s") or 0.0)
        for nthr in (1, 4):
            d = out[f"inference_threads_{nthr}"]["float64"]
            for lab, t_inf in (("single_call_401", d["single_401"]["median_s"]),
                               ("batched_401", d["batch1000_401"]["per_operating_point_s"]),
                               ("batched_net_gain_only", d["batch1000_1"]["per_operating_point_s"])):
                be[f"{kind}_threads{nthr}_{lab}"] = (T_train / (t_sol - t_inf)) if t_sol > t_inf else None
                if kind.startswith("pinn") and nthr == 1:
                    T1 = out["pinn_training_1thread_extrapolated_s"][kind]
                    be[f"{kind}_train1thread_threads1_{lab}"] = (T1 / (t_sol - t_inf)) if t_sol > t_inf else None
    out["break_even_evaluations"] = be
    out["break_even_note"] = ("N* = T_train / (t_solver - t_inference); T_train = measured training wall time of that "
                              "model with the training thread count (data generation included for the supervised "
                              "network), or, for keys '..._train1thread_...', the one-thread training time extrapolated "
                              "from the per-iteration cost; t_solver = mean cold-start time of the collocation solver "
                              "with the settings of models.raman_numerical over 1000 test sets (one thread); "
                              "t_inference with the thread count in the key")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------------------------
def fmt_time(t):
    if t >= 60:
        return f"{t / 60:.1f} min"
    if t >= 1:
        return f"{t:.1f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.1f} ms" if t < 1e-2 else f"{t * 1e3:.0f} ms"
    return f"{t * 1e6:.2g} µs" if t < 1e-5 else f"{t * 1e6:.0f} µs"


def plot(res, outdir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    import style
    style.apply()
    style.math_in_text_font()          # one font family (no math symbols at present; kept for consistency)
    C = style.C
    fig = plt.figure(figsize=(7.2, 2.75))
    ax_a = fig.add_axes([0.065, 0.17, 0.205, 0.74])
    ax_cb = fig.add_axes([0.278, 0.17, 0.010, 0.74])
    ax_b = fig.add_axes([0.425, 0.17, 0.215, 0.74])
    ax_c = fig.add_axes([0.695, 0.17, 0.295, 0.74])
    m0 = res["models"]["pinn_seed0"]
    # a) error map (seed 0)
    ax = ax_a
    mp = m0["map"]
    Z = np.asarray(mp["max_profile_dB"]).T
    pcm = ax.pcolormesh(mp["Pp_W"], mp["Ps_dBm"], Z, shading="nearest", cmap="viridis",
                        norm=LogNorm(vmin=1e-3, vmax=max(1.0, Z.max())), rasterized=True)
    cb = fig.colorbar(pcm, cax=ax_cb)
    cb.set_label("Maximum deviation [dB]", labelpad=2)
    cb.ax.tick_params(labelsize=7)
    ax.add_patch(Rectangle((PHYS["Pp_W"][0], PHYS["Ps_dBm"][0]), PHYS["Pp_W"][1] - PHYS["Pp_W"][0],
                           PHYS["Ps_dBm"][1] - PHYS["Ps_dBm"][0], fill=False, ec="white", lw=1.0, ls="--"))
    sl = m0["slice_table"]
    ax.plot(sl["Pp_W"], sl["Ps_dBm"], "o", ms=1.8, color="white", mew=0)
    ax.plot([0.5], [16.0], "*", ms=7, mfc=C[3], mec="black", mew=0.5)
    ax.set_xlabel("Pump power [W]")
    ax.set_ylabel("Signal power [dBm]")
    ax.set_xticks([0.2, 0.6, 1.0])
    ax.grid(False)
    style.label(ax, "a", x=-0.36)
    # b) distributions of the maximum profile deviation (all seeds pooled)
    ax = ax_b

    def ecdf(v, **kw):
        v = np.sort(np.asarray(v))
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", **kw)

    pin = [k for k in res["models"] if k.startswith("pinn")]
    ecdf(np.concatenate([res["models"][k]["inside_values_max_profile_dB"] for k in pin]), color=C[0], lw=1.3,
         label="PINN, inside")
    ecdf(np.concatenate([res["models"][k]["one_out_values_max_profile_dB"] for k in pin]), color=C[1], lw=1.3,
         label="PINN, one outside")
    ecdf(np.concatenate([res["models"][k]["all_out_values_max_profile_dB"] for k in pin]), color=C[7], lw=1.3,
         label="PINN, all outside")
    sup = [k for k in res["models"] if k.startswith("supervised")]
    if sup:
        ecdf(np.concatenate([res["models"][k]["inside_values_max_profile_dB"] for k in sup]), color=style.INK,
             lw=1.0, ls="--", label="Supervised, inside")
    ax.set_xscale("log")
    ax.set_xlabel("Maximum deviation [dB]")
    ax.set_ylabel("Cumulative fraction of sets")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=6.3, handlelength=1.6)
    style.label(ax, "b", x=-0.3)
    # c) cost
    ax = ax_c
    tm = res.get("timing", {})
    bars = []
    single = res.get("single_instance_training_s")
    if single:
        bars.append(("Training, single-instance PINN", single, C[4]))
    tr = [res["models"][k]["training"]["wall_s"] for k in pin]
    bars.append(("Training, parameterized PINN", float(np.median(tr)), C[0]))
    if sup:
        trs = [res["models"][k]["training"]["wall_s"] + res["models"][k]["training"]["data_generation_s"] for k in sup]
        bars.append(("Training, supervised network", float(np.median(trs)), C[2]))
    if tm:
        bars.append(("solve_bvp, cold start", tm["solve_bvp_cold_batch1000"]["mean_s"], style.INK2))
        bars.append(("solve_bvp, warm start", tm["solve_bvp_warm_card_from_pump_minus_5pct"]["median_s"], style.INK2))
        d1 = tm["inference_threads_1"]["float64"]
        bars.append(("Inference, 401 positions", d1["single_401"]["median_s"], C[0]))
        bars.append(("Inference, batched, 401 positions", d1["batch1000_401"]["per_operating_point_s"], C[0]))
        bars.append(("Inference, batched, net gain only", d1["batch1000_1"]["per_operating_point_s"], C[0]))
    n = len(bars)
    y = np.arange(n)[::-1] * 1.0
    x0 = 1e-7
    ax.barh(y, [b[1] for b in bars], left=x0, color=[b[2] for b in bars], height=0.42)
    for yi, (lab, val, col) in zip(y, bars):
        ax.text(x0 * 1.3, yi + 0.27, lab, fontsize=6.3, va="bottom", ha="left", color=style.INK)
        ax.text(val * 1.4, yi, fmt_time(val), fontsize=6.3, va="center", ha="left", color=style.INK2)
    ax.set_xscale("log")
    ax.set_xlim(x0, 3e4)
    ax.set_ylim(-0.5, n - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Time [s]")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    style.label(ax, "c", x=-0.04)
    style.save(fig, NAME, outdir=outdir)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "train", "eval", "timing", "plot"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sup-seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--adam-iters", type=int, default=DEFAULT["adam_iters"])
    ap.add_argument("--sup-adam-iters", type=int, default=None)
    ap.add_argument("--lbfgs-chunks", type=int, default=DEFAULT["lbfgs_chunks"])
    ap.add_argument("--lbfgs-iters", type=int, default=DEFAULT["lbfgs_iters"])
    ap.add_argument("--width", type=int, default=DEFAULT["width"])
    ap.add_argument("--depth", type=int, default=DEFAULT["depth"])
    ap.add_argument("--no-supervised", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--outdir", default=FIGDIR)
    args = ap.parse_args()
    cfg = dict(DEFAULT, adam_iters=args.adam_iters, lbfgs_chunks=args.lbfgs_chunks, lbfgs_iters=args.lbfgs_iters,
               width=args.width, depth=args.depth)
    torch.set_default_dtype(torch.float64)
    path = os.path.join(args.outdir, NAME + "_results.json")
    T = test_sets()
    if args.stage in ("all", "train"):
        stage_train(args, cfg, T)
    if args.stage in ("all", "eval"):
        res = dict(script="pinn_parametric.py", date=time.strftime("%Y-%m-%d %H:%M"), card=pc.CARD_ID,
                   machine=pc.machine_info(), pinning=pc.pin_performance_cores(),
                   box=dict(physical_ranges=PHYS, lo=dict(zip(PNAMES, LO.tolist())), hi=dict(zip(PNAMES, HI.tolist())),
                            kappa_s_sampling="log-uniform", card_groups=dict(zip(PNAMES, pc.CARD_GROUPS))),
                   tests=dict(inside=int(T["inside"].shape[0]), one_parameter_outside=int(T["one_out"].shape[0]),
                              all_parameters_outside=int(T["all_out"].shape[0]), slice=int(T["slice"].shape[0]),
                              outside_definition="parameter value 10-20 % beyond the box bound (x1.1-1.2 above the "
                                                 "upper or x0.8-0.9 below the lower bound); others uniform inside",
                              slice_Pp_W=SLICE_PP, slice_Ps_dBm=SLICE_PS),
                   config=cfg)
        res.update(evaluate_all(args, T))
        fs3 = os.path.join(args.outdir, "FigS3_pinn_results.json")
        if os.path.exists(fs3):
            with open(fs3, encoding="utf-8") as f:
                d3 = json.load(f)
            if "train_seconds_forward" in d3:
                res["single_instance_training_s"] = d3["train_seconds_forward"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pc.jsonable(res), f, indent=1)
    if args.stage in ("all", "timing"):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        res["pinning_timing"] = pc.pin_performance_cores()
        res["timing"] = stage_timing(args, T, res)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pc.jsonable(res), f, indent=1)
    if args.stage in ("all", "eval", "timing", "plot"):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        plot(res, args.outdir)


if __name__ == "__main__":
    main()
