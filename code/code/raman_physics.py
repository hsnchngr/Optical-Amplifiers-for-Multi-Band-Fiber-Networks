"""Distributed Raman amplification: pumping schemes, equivalent noise figure and OSNR at equal nonlinear phase
(Figure 5; profile() is also used by fig_sec4.py for the Raman/SOA cascade).

Model: undepleted pump(s), one low-power signal channel, spontaneous Raman ASE with the thermal phonon factor,
Equations (10)-(13); no double Rayleigh backscattering, no pump RIN transfer, no pump-pump interaction.
Card RAMAN-G652 (via models.RAMAN): C_R = g_R/(K A_eff) = 0.406 1/(W km) for a 1450 nm pump and a 1550 nm signal
(13.35 THz shift), alpha_s = 0.19 dB/km, alpha_p = 0.25 dB/km. The spectral shape of the gain efficiency is the
measured SSMF profile of GNPy (cards.raman_profile_ssmf), scaled with the pump frequency as in GNPy.

Definitions (noise conventions of the revised Sections 2.3 and 2.5):
  S_ASE      ASE power spectral density per polarization at the span output, Equation (13)
  F_span     (1 + 2 S_ASE/(h nu))/G_net with G_net = G_on-off exp(-alpha_s L)
  F_eq       F_span exp(-alpha_s L): equivalent (effective) noise factor of ITU-T G.665
  F_eq,tot   exp(-alpha_s L) [F_span + (F_amp - 1)/G_net]: span followed by a lumped amplifier of noise factor F_amp
  L_NL       int_0^L P(z)/P(0) dz, nonlinear effective length; R_NL = L_NL/L_eff,passive
  dOSNR_lin  NF_amp - NF_eq,tot [dB]: OSNR improvement over the unpumped span + same amplifier at equal launch power
  dOSNR_NL   dOSNR_lin - 10 log10(R_NL) [dB]: the same at equal nonlinear phase shift (launch power lowered by R_NL)
Noise factors F are linear; the dict keys keep the historical names NF_span and NF_eq for the linear values.
"""
import numpy as np
from scipy.optimize import minimize_scalar, linprog

import cards
from models import RAMAN, raman_CR, h, c, DB

kB = 1.380649e-23

_OFF_THZ, _G0_PER_WM, _F_REF = cards.raman_profile_ssmf()


def nsp(df_hz, T=300.0):
    """Thermal phonon factor of spontaneous Stokes scattering, 1 + n_th = [1 - exp(-h df/(kB T))]^-1."""
    return 1.0 / (1.0 - np.exp(-h * np.asarray(df_hz, float) / (kB * T)))


def _share(scheme, f_fwd):
    if f_fwd is not None:
        return float(f_fwd)
    return {"co": 1.0, "counter": 0.0}.get(scheme, 0.5)


def profile(scheme, Gon_db, L=80e3, p=RAMAN, n=2001, f_fwd=None, nf_amp_db=None):
    """Signal power along the span and noise figures for a given on-off gain.

    scheme: 'co', 'counter' or anything else for bidirectional 50:50 (overridden by f_fwd, the forward share of the
    total pump power, 0 = counter, 1 = co). Returns z [m], Prel (signal power relative to launch, linear), Ptot (total
    pump power [W]), NF_span, NF_eq (linear noise factors), Gnet (linear), and S_ASE [W/Hz], L_NL [m], R_NL,
    f_fwd, Pp [W]; with nf_amp_db also NF_eq_tot (linear), dOSNR_lin_dB and dOSNR_NL_dB."""
    f = _share(scheme, f_fwd)
    CR, ap, as_ = raman_CR(p), p["alpha_p"], p["alpha_s"]
    z = np.linspace(0, L, n)
    Leff_p = (1 - np.exp(-ap * L)) / ap
    Ptot = np.log(10 ** (Gon_db / 10)) / (CR * Leff_p)       # total pump power for the target on-off gain
    Pp = Ptot * (f * np.exp(-ap * z) + (1 - f) * np.exp(-ap * (L - z)))
    gl = CR * Pp - as_
    cum = np.concatenate([[0], np.cumsum(0.5 * (gl[1:] + gl[:-1]) * np.diff(z))])
    G0z = np.exp(cum)                       # gain from 0 to z (signal power relative to launch)
    GzL = np.exp(cum[-1] - cum)             # gain from z to L
    nus = c / p["lam_s"]
    S = nsp(c / p["lam_p"] - nus) * h * nus * np.trapezoid(CR * Pp * GzL, z)   # ASE PSD per polarization
    Gnet = G0z[-1]
    T = np.exp(-as_ * L)
    NF_span = (1 + 2 * S / (h * nus)) / Gnet
    NF_eq = NF_span * T                     # equivalent lumped noise factor (span loss removed)
    L_NL = np.trapezoid(G0z, z)
    Leff_s = (1 - T) / as_
    r = dict(z=z, Prel=G0z, Ptot=Ptot, NF_span=NF_span, NF_eq=NF_eq, Gnet=Gnet, S_ASE=S, L_NL=L_NL,
             R_NL=L_NL / Leff_s, Leff_s=Leff_s, f_fwd=f, Pp=Pp)
    if nf_amp_db is not None:
        Fa = 10 ** (nf_amp_db / 10)
        F_tot = T * (NF_span + (Fa - 1) / Gnet)
        r.update(NF_eq_tot=F_tot, dOSNR_lin_dB=nf_amp_db - 10 * np.log10(F_tot),
                 dOSNR_NL_dB=nf_amp_db - 10 * np.log10(F_tot) - 10 * np.log10(L_NL / Leff_s))
    return r


def d_osnr_nl(f_fwd, Gon_db, L=80e3, nf_amp_db=5.0, p=RAMAN, n=2001):
    """OSNR improvement [dB] at equal nonlinear phase over the unpumped span followed by the same amplifier."""
    return profile("bi", Gon_db, L=L, p=p, n=n, f_fwd=f_fwd, nf_amp_db=nf_amp_db)["dOSNR_NL_dB"]


def optimum_share(Gon_db, L=80e3, nf_amp_db=5.0, p=RAMAN, n=2001):
    """Forward pump share in [0, 1] that maximizes dOSNR_NL at the given on-off gain. Returns (f*, dOSNR_NL*)."""
    r = minimize_scalar(lambda f: -d_osnr_nl(f, Gon_db, L, nf_amp_db, p, n), bounds=(0.0, 1.0), method="bounded",
                        options=dict(xatol=1e-4))
    cand = [(float(r.x), -float(r.fun)), (0.0, d_osnr_nl(0.0, Gon_db, L, nf_amp_db, p, n)),
            (1.0, d_osnr_nl(1.0, Gon_db, L, nf_amp_db, p, n))]
    return max(cand, key=lambda t: t[1])


# ---------------------------------------------------------------------------------------------------------------------
# measured SSMF gain-efficiency profile (GNPy default, cards.raman_profile_ssmf)
# ---------------------------------------------------------------------------------------------------------------------
def cr_measured(df_thz, lam_p_nm=None):
    """Raman gain efficiency C_R [1/(W km)] of SSMF at frequency shift df [THz] for a pump at lam_p_nm.
    GNPy profile for its reference pump (1454 nm, effective-area overlap about 76 um^2), scaled with the pump frequency
    nu_p/nu_ref as GNPy does; the frequency dependence of the effective-area overlap is neglected here."""
    g = np.interp(np.asarray(df_thz, float), _OFF_THZ, _G0_PER_WM, left=0.0, right=0.0)
    scale = 1.0 if lam_p_nm is None else (c / (lam_p_nm * 1e-9)) / _F_REF
    return g * scale * 1e3


def profile_reference():
    """Offsets [THz], C_R [1/(W km)] of the measured profile at the reference pump, and the reference wavelength [nm]."""
    return _OFF_THZ, _G0_PER_WM * 1e3, c / _F_REF * 1e9


def onoff_gain_db(lam_s_nm, pumps_nm, P_W, L=80e3, alpha_p_dbkm=None, p=RAMAN):
    """First-order multi-pump on-off gain [dB] (undepleted, dB-additive, no pump-pump interaction):
    G_on-off,dB(lambda_s) = 4.343 sum_j C_R(nu_j - nu_s; lambda_j) P_j L_eff,j.  Returns (total, per-pump) [dB]."""
    ap = p["alpha_p"] if alpha_p_dbkm is None else alpha_p_dbkm * DB * 1e-3
    Leff = (1 - np.exp(-ap * L)) / ap
    fs = c / (np.asarray(lam_s_nm, float) * 1e-9) / 1e12
    per = []
    for lp, Pw in zip(pumps_nm, P_W):
        fp = c / (lp * 1e-9) / 1e12
        per.append(10 * np.log10(np.e) * cr_measured(fp - fs, lp) * 1e-3 * Pw * Leff)
    per = np.array(per)
    return per.sum(0), per


def flat_pump_design(lam_s_nm, pumps_nm, target_db, L=80e3, p=RAMAN):
    """Pump powers [W] that minimize the largest deviation from target_db over lam_s_nm (minimax LP, first-order)."""
    ap = p["alpha_p"]
    Leff = (1 - np.exp(-ap * L)) / ap
    fs = c / (np.asarray(lam_s_nm, float) * 1e-9) / 1e12
    Mx = np.array([10 * np.log10(np.e) * cr_measured(c / (lp * 1e-9) / 1e12 - fs, lp) * 1e-3 * Leff
                   for lp in pumps_nm]).T                                  # dB per W
    npump = len(pumps_nm)
    cost = np.r_[np.zeros(npump), 1.0]
    A = np.vstack([np.c_[Mx, -np.ones(len(fs))], np.c_[-Mx, -np.ones(len(fs))]])
    b = np.r_[np.full(len(fs), target_db), np.full(len(fs), -target_db)]
    r = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, None)] * npump + [(0, None)], method="highs")
    return r.x[:npump], 2 * r.x[-1]


if __name__ == "__main__":
    for s in ("co", "counter", "bi"):
        for G in (5, 10, 15, 20):
            r = profile(s, G, nf_amp_db=5.0)
            print(s, G, round(r["Ptot"] * 1e3), "mW", "NFeq dB", round(10 * np.log10(r["NF_eq"]), 2),
                  "NFspan dB", round(10 * np.log10(r["NF_span"]), 2), "R_NL dB", round(10 * np.log10(r["R_NL"]), 2),
                  "dOSNR_NL dB", round(r["dOSNR_NL_dB"], 2))
    for L in (80e3, 100e3, 120e3):
        for G in (15.0, 20.0):
            fo, do = optimum_share(G, L)
            print(f"L {L/1e3:.0f} km, {G:.0f} dB: optimum forward share {fo:.3f} -> {do:.2f} dB; counter "
                  f"{d_osnr_nl(0.0, G, L):.2f}, 50:50 {d_osnr_nl(0.5, G, L):.2f}, co {d_osnr_nl(1.0, G, L):.2f} dB")
    off, cr, lref = profile_reference()
    print("measured profile: peak %.4f 1/(W km) at %.2f THz (pump %.1f nm); card C_R %.4f; measured at card shift %.4f"
          % (cr.max(), off[np.argmax(cr)], lref, raman_CR() * 1e3,
             cr_measured((c / RAMAN["lam_p"] - c / RAMAN["lam_s"]) / 1e12, RAMAN["lam_p"] * 1e9)))
