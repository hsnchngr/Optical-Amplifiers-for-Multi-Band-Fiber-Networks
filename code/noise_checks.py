"""Numerical checks of the noise conventions of the revised Sections 2.3 and 2.5 (Equations (14)-(16); Reviewer 1,
point 5). Writes ../figures/noise_conventions_check_results.json.

(1) Equation (15) for a longitudinally non-uniform inversion: F = 1/G + 2 int_0^L gamma_e(z)/G(0, z) dz with
    gamma_e = g* N2/Nt, compared with the noise factor obtained from the propagated ASE, F = (1 + P_ASE/(h nu dnu))/G.
    (a) forward-pumped EDF of card EDFA-GD91-T1 (10 m, 30 mW at 1480 nm), one forward ASE band of 30 nm, no
        backward ASE, no background loss (stand-alone ODE, rtol 1e-11);
    (b) the same fiber in the shared bidirectional solver edfa_multipass.solve (forward and backward ASE), using its
        inversion profile.
    The length-averaged inversion ('2 n_sp' with mean N2) and the input-end inversion are given for comparison.
(2) Equation (13) for the counter-pumped Raman span (card RAMAN-G652): the integral form against a direct ODE
    integration of dS/dz = (C_R P_p - alpha_s) S + n_sp h nu C_R P_p.
(3) Equation (16): exp(-alpha_s L)[F_span + (F_amp - 1)/G_net] = (F_amp + 2 S_ASE/(h nu))/G_on-off, and the
    SOA (NF 7 dB) minus EDFA (NF 5 dB) difference after an 80 km counter-pumped span (ASE only).
(4) Noise-figure floor for 1480 nm pumping of card EDFA-GD91-T1 and the crude SOA chip estimate
    F = 2 n_sp Gamma g/(Gamma g - alpha_int) with n_sp = N/(N - N0) for card SOA-BULK-REF at 150 mA.
"""
import json
import os
import numpy as np
from scipy.integrate import solve_ivp

from models import EDFA, SOA, RAMAN, raman_CR, soa_carrier, h, c, dbm2w
import edfa_multipass
import raman_physics as rp

d = lambda x: 10 * np.log10(x)


def edfa_nonuniform(Pp0, Pin, L, bw_nm=30.0):
    p = EDFA
    nup, nus = c / p["lam_p"], c / p["lam_s"]
    dnu = c / (p["lam_s"] - 0.5 * bw_nm * 1e-9) - c / (p["lam_s"] + 0.5 * bw_nm * 1e-9)
    a = np.array([p["alpha_p"], p["alpha_s"], p["alpha_s"]]); g = np.array([p["g_p"], p["g_s"], p["g_s"]])
    nu = np.array([nup, nus, nus])

    def n2(P):
        phi = P / (h * nu * p["xi"])
        return np.sum(phi * a) / (1 + np.sum(phi * (a + g)))

    def rhs(z, P):
        n = n2(np.maximum(P, 0))
        dP = ((a + g) * n - a) * P
        dP[2] += g[2] * n * 2 * h * nus * dnu
        return dP

    z = np.linspace(0, L, 4001)
    sol = solve_ivp(rhs, (0, L), [Pp0, Pin, 0.0], t_eval=z, rtol=1e-11, atol=1e-20, method="LSODA")
    P = sol.y
    n = np.array([n2(P[:, i]) for i in range(z.size)])
    return _nf_from_profile(z, n, P[1, -1] / Pin, P[2, -1], dnu)


def _nf_from_profile(z, n, G_direct, Pase_out, dnu):
    p = EDFA
    nus = c / p["lam_s"]
    gl = (p["alpha_s"] + p["g_s"]) * n - p["alpha_s"]
    ge = p["g_s"] * n
    lnG = np.concatenate([[0], np.cumsum(0.5 * (gl[1:] + gl[:-1]) * np.diff(z))])
    G0z = np.exp(lnG); G = G0z[-1]
    F_direct = (1 + Pase_out / (h * nus * dnu)) / G_direct
    F_int = 1 / G + 2 * np.trapezoid(ge / G0z, z)
    nbar = n.mean()
    nsp_mean = p["g_s"] * nbar / ((p["alpha_s"] + p["g_s"]) * nbar - p["alpha_s"])
    nsp0 = p["g_s"] * n[0] / ((p["alpha_s"] + p["g_s"]) * n[0] - p["alpha_s"])
    return dict(G_dB=d(G_direct), NF_direct_dB=d(F_direct), NF_integral_dB=d(F_int),
                NF_uniform_mean_inversion_dB=d((1 + 2 * nsp_mean * (G - 1)) / G), NF_2nsp_input_end_dB=d(2 * nsp0),
                n2_input=float(n[0]), n2_output=float(n[-1]), n2_mean=float(nbar))


def raman_direct(Gon_db=15.0, L=80e3, scheme="counter"):
    r = rp.profile(scheme, Gon_db, L=L, n=20001)
    p = RAMAN; CR = raman_CR(p)
    nus = c / p["lam_s"]; ns_ = rp.nsp(c / p["lam_p"] - nus)
    zz, Pp = r["z"], r["Pp"]
    f = lambda z, S: (CR * np.interp(z, zz, Pp) - p["alpha_s"]) * S + ns_ * h * nus * CR * np.interp(z, zz, Pp)
    sol = solve_ivp(f, (0, L), [0.0], rtol=1e-10, atol=1e-30, method="LSODA", max_step=50.0)
    return dict(S_integral=float(r["S_ASE"]), S_ode=float(sol.y[0, -1]),
                rel_diff=float(sol.y[0, -1] / r["S_ASE"] - 1))


def main():
    out = {}
    rows = []
    for pin in (-40.0, -20.0, 0.0):
        a = edfa_nonuniform(0.03, dbm2w(pin), 10.0)
        s = edfa_multipass.solve(0.03, dbm2w(pin), 10.0, profiles=True)
        b = _nf_from_profile(s["z"], s["n2"], s["G"], s["Pz"][3, -1], s["dnu_ase"])
        rows.append(dict(Pin_dBm=pin, standalone_forward_ASE_only=a, shared_solver_fwd_bwd_ASE=b))
    out["eq15_nonuniform_inversion"] = dict(
        conditions="card EDFA-GD91-T1, L = 10 m, 30 mW forward pump at 1480 nm, signal 1550 nm, one 30 nm ASE band "
                   "(1535-1565 nm), no background loss; internal reference plane (fiber ends)", rows=rows)
    out["eq13_raman_integral_vs_ode"] = {f"{s}_{G}dB": raman_direct(G, scheme=s) for s in ("counter", "co")
                                         for G in (10.0, 20.0)}
    T = np.exp(-RAMAN["alpha_s"] * 80e3)
    casc = []
    for G in (0.0, 9.0, 10.0, 15.0, 20.0):
        row = dict(Gon_dB=G)
        for name, nfa in (("SOA", 7.0), ("EDFA", 5.0)):
            Fa = 10 ** (nfa / 10)
            if G == 0:
                F_tot = Fa; alt = Fa
            else:
                r = rp.profile("counter", G, nf_amp_db=nfa)
                F_tot = r["NF_eq_tot"]
                alt = (Fa + 2 * r["S_ASE"] / (h * c / RAMAN["lam_s"])) / 10 ** (G / 10)
            row[f"NF_eq_tot_{name}_dB"] = d(F_tot); row[f"identity_rel_err_{name}"] = float(alt / F_tot - 1)
        row["SOA_minus_EDFA_dB"] = row["NF_eq_tot_SOA_dB"] - row["NF_eq_tot_EDFA_dB"]
        if G > 0:
            row["pump_W"] = rp.profile("counter", G)["Ptot"]; row["net_span_gain_dB"] = G - d(1 / T)
        casc.append(row)
    out["eq16_cascade"] = dict(conditions="80 km SSMF, counter-pumped, card RAMAN-G652, undepleted pump, ASE only "
                                          "(no DRB, no pump RIN, no SOA nonlinearity); amplifier noise figures "
                                          "7 dB (SOA) and 5 dB (EDFA) at their input ports", rows=casc)
    out["raman_equivalent_NF_dB"] = {s: {str(G): d(rp.profile(s, G)["NF_eq"]) for G in (5, 10, 15, 20)}
                                     for s in ("counter", "bi", "co")}
    fl = edfa_multipass.nf_floor()
    out["edfa_1480_floor"] = dict(n2_max=fl["n2_max"], nsp_min=fl["nsp_min"], NF_min_dB=fl["NF_min_dB"])
    N = soa_carrier(0.15, 0.0)
    gmod = SOA["Gamma"] * SOA["a"] * (N - SOA["N0"])
    nsp_lin = N / (N - SOA["N0"]); loss_f = gmod / (gmod - SOA["alpha_int"])
    out["soa_chip_nf_crude"] = dict(I_A=0.15, N=N, nsp_linear_gain_estimate=nsp_lin, loss_factor=loss_f,
                                    NF_chip_dB=d(2 * nsp_lin * loss_f),
                                    note="crude estimate: n_sp = N/(N - N0) is not a band-structure value; "
                                         "chip reference plane, coupling loss excluded")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures",
                        "noise_conventions_check_results.json")
    with open(os.path.normpath(path), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
