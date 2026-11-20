"""Single-pass and double-pass EDFA with forward and backward ASE, solved by iterative forward-backward
sweeps (the standard relaxation scheme for bidirectional amplifier models).

Fiber: card EDFA-GD91-T1 (Giles and Desurvire 1991, Table I fiber), read through models.EDFA.
Model assumptions (state them with every result):
  * two-level Giles model, Equations (2)-(3), no background loss;
  * spontaneous emission is carried by ONE lumped ASE band centered at the signal wavelength (1550 nm) that uses
    the 1550 nm coefficients; its width is a modeling choice (default 30 nm, 1535-1565 nm). The band width sets
    the ASE power and hence the ASE self-saturation, so the small-signal results depend on it (argument ase_bw_nm);
  * forward pump only; in the double-pass case the residual pump is not reflected;
  * double pass: the signal and the forward ASE are reflected at z = L with the power reflectance R_db
    (Faraday rotator mirror); circulator, WDM-coupler and isolator losses are not included, so gain and noise
    figure refer to the fiber ends (internal reference plane);
  * noise factor F = (1 + P_ASE/(h nu dnu))/G = (1 + 2 S_ASE/(h nu))/G, with P_ASE the two-polarization ASE
    power in the band dnu that leaves together with the signal (S_ASE per polarization; Equation (14)).
"""
import numpy as np
from models import h, c, DB, dbm2w, w2dbm, EDFA

# Card EDFA-GD91-T1 (via models.EDFA); the lumped ASE band (1535-1565 nm) is a configuration choice of this model.
P = dict(xi=EDFA["xi"], lam_p=EDFA["lam_p"], a_p=EDFA["alpha_p"], g_p=EDFA["g_p"], lam_s=EDFA["lam_s"],
         a_s=EDFA["alpha_s"], g_s=EDFA["g_s"], dnu_ase=c / 1535e-9 - c / 1565e-9)


def ase_band_hz(width_nm, center=None):
    """Optical bandwidth [Hz] of a lumped ASE band of the given width [nm] centered at `center` [m] (default: signal)."""
    lc = P["lam_s"] if center is None else center
    w = width_nm * 1e-9
    return c / (lc - 0.5 * w) - c / (lc + 0.5 * w)


def nf_floor():
    """Lower bound of the noise figure set by the pump wavelength (card values, uniform maximum inversion).
    With pumping at lambda_p the inversion cannot exceed n2,max = alpha_p/(alpha_p + g*_p); the spontaneous-emission
    factor at the signal is then n_sp,min = g*_s n2,max / [(alpha_s + g*_s) n2,max - alpha_s] and F -> 2 n_sp,min for G >> 1.
    Returns dict(n2_max, nsp_min, F_min, NF_min_dB)."""
    n2max = P["a_p"] / (P["a_p"] + P["g_p"])
    nsp = P["g_s"] * n2max / ((P["a_s"] + P["g_s"]) * n2max - P["a_s"])
    return dict(n2_max=n2max, nsp_min=nsp, F_min=2 * nsp, NF_min_dB=10 * np.log10(2 * nsp))


def solve(Pp0, Pin, L, double_pass=False, R_db=-1.0, N=800, tol=1e-7, max_iter=400, history=False,
          ase_bw_nm=None, profiles=False):
    """Steady state of the single- or double-pass amplifier.

    Pp0 pump power [W] at z = 0 (forward), Pin signal input power [W], L fiber length [m], R_db reflectance of the mirror
    at z = L [dB] (double pass only), N grid intervals, tol convergence limit on the change of the output signal [dB].
    Optional: ase_bw_nm width of the lumped ASE band [nm] (None keeps the default 30 nm band of P['dnu_ase']);
    profiles=True adds z [m], Pz (powers of pump, forward signal, backward signal, forward ASE, backward ASE [W])
    and n2 (normalized upper-level population) along the fiber, and the ASE band dnu_ase [Hz].
    Returns output signal power (W), gain, noise factor, sweeps used and the convergence history (max change in dB)."""
    p = P
    dnu = p["dnu_ase"] if ase_bw_nm is None else ase_band_hz(ase_bw_nm)
    z = np.linspace(0, L, N + 1); dz = z[1] - z[0]
    nup, nus = c / p["lam_p"], c / p["lam_s"]
    a = np.array([p["a_p"], p["a_s"], p["a_s"], p["a_s"], p["a_s"]])      # pump, sig_f, sig_b, ase_f, ase_b
    g = np.array([p["g_p"], p["g_s"], p["g_s"], p["g_s"], p["g_s"]])
    nu = np.array([nup, nus, nus, nus, nus])
    u = np.array([1, 1, -1, 1, -1])
    spont = np.array([0, 0, 0, 2 * h * nus * dnu, 2 * h * nus * dnu])
    R = 10 ** (R_db / 10)
    Pz = np.zeros((5, N + 1)); Pz[0] = Pp0; Pz[1] = Pin
    if double_pass:
        Pz[2] = Pin * 0.0

    def n2(Pv):
        phi = Pv / (h * nu * p["xi"])
        return np.sum(phi * a) / (1 + np.sum(phi * (a + g)))

    def rates(Pv):
        """dP/dz of all five channels for the local powers Pv (one inversion evaluation per call)."""
        n = n2(Pv)
        return u * (((a + g) * n - a) * Pv + g * n * spont)

    fw, bw = [0, 1, 3], ([2, 4] if double_pass else [4])
    hist = []; last = None
    for it in range(max_iter):
        # forward sweep (RK2, backward channels frozen at current iterate)
        for i in range(N):
            Pv = Pz[:, i].copy()
            k1 = rates(Pv)
            Pm = Pv + 0.5 * dz * k1; Pm[[2, 4]] = 0.5 * (Pz[[2, 4], i] + Pz[[2, 4], i + 1])
            k2 = rates(Pm)
            Pz[fw, i + 1] = np.maximum(Pv[fw] + dz * k2[fw], 0)
        # boundary at z = L
        Pz[4, N] = R * Pz[3, N] if double_pass else 0.0
        if double_pass:
            Pz[2, N] = R * Pz[1, N]
        # backward sweep
        for i in range(N, 0, -1):
            Pv = Pz[:, i].copy()
            k1 = -rates(Pv)
            Pm = Pv + 0.5 * dz * k1; Pm[fw] = 0.5 * (Pz[fw, i] + Pz[fw, i - 1])
            k2 = -rates(Pm)
            Pz[bw, i - 1] = np.maximum(Pv[bw] + dz * k2[bw], 0)
        out = Pz[2, 0] if double_pass else Pz[1, N]
        ase_out = Pz[4, 0] if double_pass else Pz[3, N]
        if last is not None:
            ch = abs(w2dbm(out) - w2dbm(last)); hist.append(ch)
            if ch < tol:
                break
        last = out
    G = out / Pin
    nf = (1 + ase_out / (h * nus * dnu)) / G      # F = (1 + 2 S_ase/(h nu))/G, S_ase per polarization
    res = dict(Pout=out, G=G, NF=nf, sweeps=it + 1, hist=np.array(hist))
    if profiles:
        phi = Pz / (h * nu[:, None] * p["xi"])
        res.update(z=z, Pz=Pz.copy(), n2=np.sum(phi * a[:, None], 0) / (1 + np.sum(phi * (a + g)[:, None], 0)),
                   dnu_ase=dnu)
    return res


if __name__ == "__main__":
    fl = nf_floor()
    print("1480 nm pumping: n2,max = %.3f, n_sp,min = %.4f, NF floor = %.2f dB" % (fl["n2_max"], fl["nsp_min"], fl["NF_min_dB"]))
    for bw in (10, 30, 50):
        for pin in (-40, -20, 0):
            s = solve(0.03, dbm2w(pin), 10.0, ase_bw_nm=bw)
            d = solve(0.03, dbm2w(pin), 10.0, double_pass=True, ase_bw_nm=bw)
            gs, gd = 10 * np.log10(s["G"]), 10 * np.log10(d["G"])
            ns, nd = 10 * np.log10(s["NF"]), 10 * np.log10(d["NF"])
            print(f"ASE band {bw} nm, {pin:4d} dBm: SP {gs:6.2f} dB / NF {ns:4.2f} dB | DP {gd:6.2f} dB / NF {nd:4.2f} dB"
                  f" | advantage {gd - gs:5.2f} dB, NF penalty {nd - ns:4.2f} dB")
