"""SOA physics for Figure 4 and the patterning study of the Supporting Information (Figure S2).

(1) Material gain spectra (Figure 4b): free-carrier, parabolic-band, k-conserving model of bulk InGaAsP with heavy- and
    light-hole bands, exact Fermi-Dirac quasi-Fermi levels (charge neutrality) and hyperbolic-secant intraband
    broadening. Parameters: card SOA-BAND-ILL (illustrative; spectral shape and band-filling shift only).
(2) Reservoir (integrated-gain) model, Equation (9) (Agrawal & Olsson 1989):
        dh/dt = (h0 - h)/tau - (exp(h) - 1) P_in(t)/E_sat,
    with its parameters derived from the device card SOA-BULK-REF at a bias current I (card_operating_point):
    G0 = exp(h0) and the differential lifetime tau_d from the A/B/C steady-state model of models.py (a calibration
    against that model), and E_sat = h nu w d/(Gamma a).
(3) Time-domain travelling-wave model of the same card (tw_*): carrier density N(z, t) in M sections with A/B/C
    recombination, linear gain Gamma a (N - N0), internal loss alpha_int, retarded time, ASE neglected. It is the
    time-domain counterpart of models.soa_numerical and reduces exactly to Equation (9) when alpha_int = 0 and
    R(N) = N/tau (checked in verify()).
(4) Waveform generators (de Bruijn NRZ-OOK, NRZ-DPSK, Nyquist PDM-16QAM, WDM aggregates) and patterning metrics.

All powers are signal powers at the chip input (facet-to-facet reference plane of card SOA-BULK-REF); coupling losses,
ASE, gain dispersion, carrier heating, spectral hole burning, the alpha-factor phase and inter-channel four-wave
mixing are not included.
"""
import os
import numpy as np
from scipy.optimize import brentq

import cards
from models import SOA, soa_carrier, soa_analytic, soa_R, h, c, q

kB, m0, eps0 = 1.380649e-23, 9.1093837e-31, 8.8541878128e-12
hbar = h / (2 * np.pi)
NP10 = 10 / np.log(10)            # neper -> dB for log-gain h

BAND_CARD = os.path.join(cards.CARD_DIR, "soa_band_ill.json")
BAND = cards.values(cards.load(BAND_CARD))


# ---------------------------------------------------------------------------------------------------------------------
# (1) material gain spectra
# ---------------------------------------------------------------------------------------------------------------------
_FDX = np.linspace(0.0, 90.0, 9001)


def _f12(eta):
    """Fermi-Dirac integral F_1/2(eta) normalized so that n = N_c F_1/2(eta)."""
    return 2 / np.sqrt(np.pi) * np.trapezoid(np.sqrt(_FDX) / (1 + np.exp(np.clip(_FDX - eta, -700, 700))), _FDX)


def _nc(m, T):
    return 2 * (m * kB * T / (2 * np.pi * hbar ** 2)) ** 1.5


def quasi_fermi(N, b=BAND):
    """Quasi-Fermi energies [J] of electrons (measured upwards from the conduction-band edge) and of holes (measured
    downwards from the valence-band edge) for an undoped layer with P = N."""
    T = b["T"]; kT = kB * T
    Nc = _nc(b["m_e"] * m0, T)
    Nv = _nc(b["m_hh"] * m0, T) + _nc(b["m_lh"] * m0, T)
    ec = brentq(lambda e: Nc * _f12(e) - N, -40, 85)
    ev = brentq(lambda e: Nv * _f12(e) - N, -40, 85)
    return ec * kT, ev * kT


def material_gain(lam, N, b=BAND):
    """Material gain g(lambda, N) [1/m] of the free-carrier model (card SOA-BAND-ILL); lam in m (array).
    Intraband broadening: hyperbolic-secant line shape L(E) = sech(E/w)/(pi w), FWHM = 2 w ln(2 + sqrt 3)."""
    T = b["T"]; kT = kB * T
    Eg, Ep = b["E_g"] * q, b["E_p"] * q
    w = b["broadening_fwhm"] * q / (2 * np.log(2 + np.sqrt(3)))
    me = b["m_e"] * m0
    Fc, Fv = quasi_fermi(N, b)
    E = h * c / np.atleast_1d(np.asarray(lam, float))
    Et = np.linspace(Eg, Eg + 0.5 * q, 5001)            # transition energies
    ek = Et - Eg
    Lz = 1.0 / (np.pi * w * np.cosh(np.clip((E[:, None] - Et[None, :]) / w, -700, 700)))
    acc = 0.0
    for mh in (b["m_hh"] * m0, b["m_lh"] * m0):
        mr = me * mh / (me + mh)
        fc = 1 / (1 + np.exp((ek * mr / me - Fc) / kT))     # electron occupation of the conduction state
        fh = 1 / (1 + np.exp((ek * mr / mh - Fv) / kT))     # hole occupation of the valence state
        rho = (2 * mr / hbar ** 2) ** 1.5 * np.sqrt(ek) / (2 * np.pi ** 2)   # joint density of states incl. spin
        acc = acc + np.trapezoid((rho * (fc + fh - 1))[None, :] * Lz, Et, axis=1)
    C0 = np.pi * q ** 2 / (b["n_r"] * c * eps0 * m0 ** 2 * (E / hbar))
    return C0 * (m0 * Ep / 6) * acc


def gain_spectra(Ns, lam, normalize=True):
    """Gain spectra for the carrier densities Ns; returns (spectra, peak) with spectra normalized to the highest peak
    (normalize=True) or in 1/m, and peak = highest material gain [1/m]."""
    raw = np.array([material_gain(lam, N) for N in Ns])
    peak = float(raw.max())
    return (raw / peak if normalize else raw), peak


def band_card_consistency(I=0.15, lam=1550e-9, p=SOA, dN=0.1e24):
    """Compare the free-carrier model with the linear gain a(N - N0) of card SOA-BULK-REF at the operating density."""
    N = soa_carrier(I, 0.0, p)
    g = lambda n: float(material_gain([lam], n)[0])
    gN = g(N)
    a_diff = (g(N + dN) - g(N - dN)) / (2 * dN)
    Ntr = brentq(g, 0.8e24, N, xtol=1e19)
    return dict(I=I, lam_nm=lam * 1e9, N=N, g_band_per_cm=gN / 100, g_card_per_cm=p["a"] * (N - p["N0"]) / 100,
                ratio=gN / (p["a"] * (N - p["N0"])), a_band=a_diff, a_card=p["a"], N_tr_band=Ntr, N0_card=p["N0"],
                N0_tangent_band=N - gN / a_diff)


# ---------------------------------------------------------------------------------------------------------------------
# (2) reservoir model, Equation (9), with card-derived parameters
# ---------------------------------------------------------------------------------------------------------------------
def card_operating_point(I, p=SOA):
    """Parameters of Equations (8)-(9) derived from card SOA-BULK-REF at bias current I [A]."""
    N = soa_carrier(I, 0.0, p)
    tau_d = 1.0 / (p["A"] + 2 * p["B"] * N + 3 * p["C"] * N ** 2)     # differential lifetime
    tau_c = N / soa_R(N, p)                                            # total carrier lifetime N/R(N)
    _, G0, Psat = soa_analytic(I, 1e-12, p)
    hv = h * c / p["lam"]
    Esat = hv * p["w"] * p["d"] / (p["Gamma"] * p["a"])
    return dict(I=I, N=N, tau=tau_d, tau_d=tau_d, tau_c=tau_c, G0=G0, G0_dB=10 * np.log10(G0), h0=np.log(G0),
                Esat=Esat, Psat=Psat, Psat_dBm=10 * np.log10(Psat * 1e3))


def reservoir_cw(Pin, op):
    """Steady-state log gain h of Equation (8) for CW input power(s) Pin [W]."""
    Pin = np.atleast_1d(np.asarray(Pin, float))
    out = np.empty(Pin.shape)
    for i, P in np.ndenumerate(Pin):
        k = P * op["tau"] / op["Esat"]
        out[i] = brentq(lambda x: x - op["h0"] + np.expm1(x) * k, 0.0, op["h0"]) if P > 0 else op["h0"]
    return out


def saturation_powers(op):
    """3-dB output and input saturation powers of Equation (8) and the input power for 1 dB compression [W]."""
    G0 = op["G0"]
    Pout3 = np.log(2) * G0 / (G0 - 2) * op["Psat"]
    Pin3 = Pout3 / (G0 / 2)
    Pin1 = brentq(lambda P: op["h0"] - reservoir_cw(P, op)[0] - np.log(10 ** 0.1), 1e-12, 1.0)
    return dict(Pout_3dB=Pout3, Pin_3dB=Pin3, Pin_1dB=Pin1)


def gaussian_pulse(t, E, t_half_1e=5e-12):
    """Gaussian pulse power [W] of energy E with 1/e half-width t_half_1e (FWHM = 1.665 t_half_1e)."""
    return E / (t_half_1e * np.sqrt(np.pi)) * np.exp(-(t / t_half_1e) ** 2)


def reservoir_pulse(E_pulse, op, t_half_1e=5e-12, t0=-50e-12, t1=1.5e-9, dt=0.05e-12):
    """Equation (9) for a single Gaussian pulse (RK4, exact pulse shape at the half steps). Returns t, h(t)."""
    n = int(round((t1 - t0) / dt))
    t = t0 + dt * np.arange(n + 1)
    P = lambda tt: gaussian_pulse(tt, E_pulse, t_half_1e)
    f = lambda hh, PP: (op["h0"] - hh) / op["tau"] - np.expm1(hh) * PP / op["Esat"]
    hh = np.empty(n + 1); hh[0] = op["h0"]
    x = op["h0"]
    for i in range(n):
        p0, p1, p2 = P(t[i]), P(t[i] + 0.5 * dt), P(t[i] + dt)
        k1 = f(x, p0); k2 = f(x + 0.5 * dt * k1, p1); k3 = f(x + 0.5 * dt * k2, p1); k4 = f(x + dt * k3, p2)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        hh[i + 1] = x
    return t, hh


def reservoir_periodic(w2, dt, Pavg, op, warm_steps=None, h_init=None):
    """Equation (9) driven by periodic waveforms.

    w2    : (n_wave, 2 n_steps) waveforms with unit mean on a half-step grid (sample 2k at t_k, 2k+1 at t_k + dt/2)
    dt    : (n_wave,) time step of each waveform [s]
    Pavg  : (n_wave, n_pow) average input power [W] of each waveform/power combination
    Returns h at the full steps, shape (n_steps, n_wave, n_pow), in the periodic steady state."""
    w2 = np.asarray(w2, float); nw, n2 = w2.shape; ns = n2 // 2
    dt = np.asarray(dt, float)[:, None]
    P = np.asarray(Pavg, float)
    x = np.array([[reservoir_cw(p, op)[0] for p in row] for row in P]) if h_init is None else np.array(h_init, float)
    if warm_steps is None:
        warm_steps = ns
    out = np.empty((ns, nw, P.shape[1]))
    f = lambda hh, w: (op["h0"] - hh) / op["tau"] - np.expm1(hh) * (P * w[:, None]) / op["Esat"]
    for i in range(-warm_steps, ns):
        i0 = (2 * i) % n2
        if i >= 0:
            out[i] = x
        w0, w1, w2_ = w2[:, i0], w2[:, (i0 + 1) % n2], w2[:, (i0 + 2) % n2]
        k1 = f(x, w0); k2 = f(x + 0.5 * dt * k1, w1); k3 = f(x + 0.5 * dt * k2, w1); k4 = f(x + dt * k3, w2_)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# (3) time-domain travelling-wave model of card SOA-BULK-REF
# ---------------------------------------------------------------------------------------------------------------------
def _tw_const(p, M):
    hv = h * c / p["lam"]
    return dict(dz=p["L"] / M, V=p["L"] * p["w"] * p["d"], hvA=hv * p["w"] * p["d"])


def _tw_rates(N, Pin, I, p, k):
    """dN/dt for all sections and the net log gain h; N shape (..., M), Pin shape (...)."""
    gm = p["Gamma"] * p["a"] * (N - p["N0"])                 # modal gain without loss [1/m]
    x = (gm - p["alpha_int"]) * k["dz"]                      # net log gain of each section
    cum = np.cumsum(x, axis=-1)
    Pk = Pin[..., None] * np.exp(cum - x)                     # power entering each section
    small = np.abs(x) < 1e-8
    avg = Pk * np.where(small, 1.0 + 0.5 * x, np.expm1(x) / np.where(small, 1.0, x))   # exact section average
    dN = I / (q * k["V"]) - (p["A"] * N + p["B"] * N ** 2 + p["C"] * N ** 3) - gm * avg / k["hvA"]
    return dN, cum[..., -1]


def tw_steady(Pin, I, p=SOA, M=64):
    """CW steady state of the sectioned model: N per section solved section by section (brentq). Returns N (..., M)."""
    k = _tw_const(p, M)
    Pin = np.atleast_1d(np.asarray(Pin, float))
    Ns = np.empty(Pin.shape + (M,))
    for idx, P0 in np.ndenumerate(Pin):
        P = P0
        for m in range(M):
            def f(N):
                gm = p["Gamma"] * p["a"] * (N - p["N0"])
                x = (gm - p["alpha_int"]) * k["dz"]
                avg = P * (np.expm1(x) / x if abs(x) > 1e-12 else 1.0)
                return I / (q * k["V"]) - (p["A"] * N + p["B"] * N ** 2 + p["C"] * N ** 3) - gm * avg / k["hvA"]
            N = brentq(f, 1e20, 1e26, xtol=1e12)
            Ns[idx + (m,)] = N
            P = P * np.exp((p["Gamma"] * p["a"] * (N - p["N0"]) - p["alpha_int"]) * k["dz"])
    return Ns


def tw_log_gain(N, p=SOA, M=None):
    M = N.shape[-1] if M is None else M
    return np.sum((p["Gamma"] * p["a"] * (N - p["N0"]) - p["alpha_int"]) * (p["L"] / M), axis=-1)


def tw_pulse(E_pulse, I, p=SOA, M=64, t_half_1e=5e-12, t0=-50e-12, t1=1.5e-9, dt=0.05e-12):
    """Sectioned model for a single Gaussian pulse (RK4). Returns t, h(t) (net log gain seen by the input at time t)."""
    k = _tw_const(p, M)
    n = int(round((t1 - t0) / dt))
    t = t0 + dt * np.arange(n + 1)
    N = tw_steady(0.0, I, p, M)[0]
    hh = np.empty(n + 1)
    P = lambda tt: np.array(gaussian_pulse(tt, E_pulse, t_half_1e))
    for i in range(n):
        p0, p1, p2 = P(t[i]), P(t[i] + 0.5 * dt), P(t[i] + dt)
        k1, h_now = _tw_rates(N, p0, I, p, k)
        if i == 0:
            hh[0] = h_now
        k2, _ = _tw_rates(N + 0.5 * dt * k1, p1, I, p, k)
        k3, _ = _tw_rates(N + 0.5 * dt * k2, p1, I, p, k)
        k4, _ = _tw_rates(N + dt * k3, p2, I, p, k)
        N = N + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        hh[i + 1] = tw_log_gain(N, p, M)
    return t, hh


def tw_periodic(w2, dt, Pavg, I, p=SOA, M=32, warm_steps=None):
    """Sectioned model driven by periodic waveforms (same conventions as reservoir_periodic). Returns h (ns, nw, np)."""
    k = _tw_const(p, M)
    w2 = np.asarray(w2, float); nw, n2 = w2.shape; ns = n2 // 2
    dtv = np.asarray(dt, float)[:, None, None]
    P = np.asarray(Pavg, float)
    N = tw_steady(P, I, p, M)                                  # start from the CW steady state at the average power
    if warm_steps is None:
        warm_steps = ns
    out = np.empty((ns, nw, P.shape[1]))
    for i in range(-warm_steps, ns):
        i0 = (2 * i) % n2
        w0, w1, w2_ = w2[:, i0], w2[:, (i0 + 1) % n2], w2[:, (i0 + 2) % n2]
        k1, hcur = _tw_rates(N, P * w0[:, None], I, p, k)
        if i >= 0:
            out[i] = hcur
        k2, _ = _tw_rates(N + 0.5 * dtv * k1, P * w1[:, None], I, p, k)
        k3, _ = _tw_rates(N + 0.5 * dtv * k2, P * w1[:, None], I, p, k)
        k4, _ = _tw_rates(N + dtv * k3, P * w2_[:, None], I, p, k)
        N = N + dtv / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def recovery_time(t, G_db, frac=0.9):
    """Time from the gain minimum until the gain has recovered by `frac` of its compression (in dB)."""
    i0 = int(np.argmin(G_db)); g0 = G_db[0]; gmin = G_db[i0]
    idx = np.where((t > t[i0]) & (G_db >= gmin + frac * (g0 - gmin)))[0]
    return float(t[idx[0]] - t[i0]) if len(idx) else float("nan")


# ---------------------------------------------------------------------------------------------------------------------
# (4) waveforms and patterning metrics
# ---------------------------------------------------------------------------------------------------------------------
def de_bruijn(n=10):
    """Binary de Bruijn sequence of order n (length 2^n; every n-bit word occurs once per period)."""
    a = [0] * (2 * n); seq = []

    def db(t, pp):
        if t > n:
            if n % pp == 0:
                seq.extend(a[1:pp + 1])
        else:
            a[t] = a[t - pp]; db(t + 1, pp)
            for j in range(a[t - pp] + 1, 2):
                a[t] = j; db(t + 1, t)
    db(1, 1)
    return np.array(seq, dtype=float)


def _gauss_filter(x, spb, rise_frac=0.25, delay_frac=0.0):
    """Periodic Gaussian low-pass with 10-90 % rise time rise_frac*T_b; optional delay in bit periods."""
    n = x.size
    f = np.fft.rfftfreq(n, d=1.0 / spb)                      # in units of the bit rate
    sigma = rise_frac / 2.563
    H = np.exp(-2 * (np.pi * f * sigma) ** 2) * np.exp(-2j * np.pi * f * delay_frac)
    return np.fft.irfft(np.fft.rfft(x) * H, n)


def nrz(bits, spb, rise_frac=0.25, delay_frac=0.0, er_db=np.inf):
    """NRZ-OOK power waveform (unit mark power, zero space power for infinite extinction ratio)."""
    y = np.clip(_gauss_filter(np.repeat(bits, spb), spb, rise_frac, delay_frac), 0, None)
    if np.isfinite(er_db):
        r = 10 ** (-er_db / 10); y = r + (1 - r) * y
    return y


def dpsk_mzm(bits, spb, rise_frac=0.25):
    """NRZ-DPSK power waveform of a null-biased push-pull Mach-Zehnder modulator (intensity dips at phase changes)."""
    d = _gauss_filter(np.repeat(2 * bits - 1, spb), spb, rise_frac)
    return np.sin(np.pi / 2 * np.clip(d, -1, 1)) ** 2


def nyquist_qam_power(nsym, spb, rng, rolloff=0.1, M=16, delay_frac=0.0):
    """Power waveform |Ex|^2 + |Ey|^2 of a PDM M-QAM signal with a root-raised-cosine spectrum (periodic)."""
    lv = np.arange(-(np.sqrt(M) - 1), np.sqrt(M), 2)
    n = nsym * spb
    f = np.fft.fftfreq(n, d=1.0 / spb)
    af = np.abs(f); H = np.zeros(n)
    H[af <= (1 - rolloff) / 2] = 1.0
    m = (af > (1 - rolloff) / 2) & (af <= (1 + rolloff) / 2)
    H[m] = np.sqrt(0.5 * (1 + np.cos(np.pi / rolloff * (af[m] - (1 - rolloff) / 2))))
    P = 0
    for _ in range(2):
        s = rng.choice(lv, nsym) + 1j * rng.choice(lv, nsym)
        x = np.zeros(n, complex); x[::spb] = s
        e = np.fft.ifft(np.fft.fft(x) * H * np.exp(-2j * np.pi * f * delay_frac))
        P = P + np.abs(e) ** 2
    return P


def ook_metrics(hh, total, probe, bits, spb2, G0_dB):
    """Patterning metrics of the probe channel for log gains hh (n_steps, n_pow) sampled every second half-step.

    total, probe: waveforms on the full-step grid (unit mean); bits: probe bits; spb2 = full steps per bit.
    std_mark_dB / pp_mark_dB: standard deviation and peak-to-peak of the gain (dB) at the centers of the probe marks;
    eop_opt_dB: eye-opening penalty of the probe at the best sampling phase (12.5-87.5 % of the bit);
    rms_h_dB: standard deviation of the gain over all time; comp_dB: compression of the average power gain."""
    ns = hh.shape[0]; G = np.exp(hh); r = {}
    nb = bits.size
    r["comp_dB"] = G0_dB - 10 * np.log10((G * total[:, None]).mean(0) / total.mean())
    r["rms_h_dB"] = hh.std(0) * NP10
    out1 = G * probe[:, None]
    best = None
    for ph in range(spb2 // 8, spb2 - spb2 // 8 + 1, max(1, spb2 // 32)):
        idx = (np.arange(nb) * spb2 + ph) % ns
        mk, sp = idx[bits > 0.5], idx[bits < 0.5]
        eo = (out1[mk].min(0) - out1[sp].max(0)) / (out1[mk].mean(0) - out1[sp].mean(0))
        best = eo if best is None else np.maximum(best, eo)
    idx = (np.arange(nb) * spb2 + spb2 // 2) % ns
    hm = hh[idx[bits > 0.5]]
    r["std_mark_dB"] = hm.std(0) * NP10
    r["pp_mark_dB"] = (hm.max(0) - hm.min(0)) * NP10
    r["eop_opt_dB"] = -10 * np.log10(np.clip(best, 1e-6, None))
    return {key: np.asarray(v) for key, v in r.items()}


def generic_metrics(hh, total, spb2, G0_dB):
    """Metrics for formats without marks: all-time gain std and the std at the symbol centers."""
    ns = hh.shape[0]; G = np.exp(hh)
    idx = (np.arange(ns // spb2) * spb2 + spb2 // 2) % ns
    return dict(comp_dB=G0_dB - 10 * np.log10((G * total[:, None]).mean(0) / total.mean()),
                rms_h_dB=hh.std(0) * NP10, std_sym_dB=hh[idx].std(0) * NP10,
                pp_sym_dB=(hh[idx].max(0) - hh[idx].min(0)) * NP10)


def linearized_sigma(Pavg, m_rel, Tb, op):
    """Linearized Equation (9): standard deviation (over all time) of the log gain for a random NRZ total power with
    relative standard deviation m_rel; first-order low-pass with tau_eff = tau/(1 + P_out/P_sat)."""
    Pavg = np.atleast_1d(np.asarray(Pavg, float))
    G = np.exp(reservoir_cw(Pavg, op))
    K = (G - 1) * Pavg / op["Psat"] / (1 + G * Pavg / op["Psat"])
    te = op["tau"] / (1 + G * Pavg / op["Psat"])
    return K * m_rel * np.sqrt(1 - te / Tb * (1 - np.exp(-Tb / te))) * NP10


def verify(I=0.15, M=64):
    """Code checks: (i) the sectioned model reduces to Equation (9) for alpha_int = 0 and R = N/tau (pulse, 0.3 pJ);
    (ii) its CW steady state agrees with models.soa_numerical."""
    from models import soa_numerical
    op = card_operating_point(I)
    tau = op["tau_c"]                                # R = N/tau_c keeps the card's operating density N(I)
    p_lin = dict(SOA, A=1.0 / tau, B=0.0, C=0.0, alpha_int=0.0)
    op_lin = card_operating_point(I, p_lin)          # h0 = Gamma a (I tau/(qV) - N0) L
    t, h_tw = tw_pulse(0.3e-12, I, p_lin, M=M, t1=0.6e-9, dt=0.05e-12)
    _, h_res = reservoir_pulse(0.3e-12, op_lin, t1=0.6e-9, dt=0.05e-12)
    res = dict(lin_limit_max_abs_diff_dB=float(np.max(np.abs(h_tw - h_res)) * NP10),
               lin_limit_tau_ps=op_lin["tau_d"] * 1e12, lin_limit_G0_dB=op_lin["G0_dB"])
    rows = []
    for pd in (-40.0, -20.0, -10.0, 0.0):
        Pin = 1e-3 * 10 ** (pd / 10)
        Ns = tw_steady(Pin, I, SOA, M)[0]
        g_tw = tw_log_gain(Ns) * NP10
        z, P = soa_numerical(I, Pin)
        rows.append((pd, float(g_tw), float(10 * np.log10(P[-1] / Pin))))
    res["cw_check"] = [dict(Pin_dBm=a, G_sectioned_dB=b, G_soa_numerical_dB=cc, diff_dB=b - cc) for a, b, cc in rows]
    return res


if __name__ == "__main__":
    import json
    for I in (0.12, 0.15):
        op = card_operating_point(I)
        s = saturation_powers(op)
        print(f"I = {I*1e3:.0f} mA: N = {op['N']:.4e} 1/m^3, G0 = {op['G0_dB']:.2f} dB, tau_d = {op['tau_d']*1e12:.1f} ps, "
              f"tau_c = {op['tau_c']*1e12:.1f} ps, E_sat = {op['Esat']*1e12:.3f} pJ, P_sat = {op['Psat']*1e3:.2f} mW; "
              f"P_out,3dB = {10*np.log10(s['Pout_3dB']*1e3):.2f} dBm, P_in,3dB = {10*np.log10(s['Pin_3dB']*1e3):.2f} dBm, "
              f"P_in,1dB = {10*np.log10(s['Pin_1dB']*1e3):.2f} dBm")
    print(json.dumps(band_card_consistency(), indent=1))
    print(json.dumps(verify(), indent=1))
