"""SOA patterning study for Figure 4d and Supporting Information Figure S2 (Reviewer 1, point 3).

Device: card SOA-BULK-REF at 150 mA (G0 = 25.45 dB, tau_d = 261 ps, E_sat = 3.90 pJ, P_sat = 14.9 mW); the single-channel
cases are repeated at 120 mA (G0 = 18.04 dB, tau_d = 295 ps). Two models are run on identical waveforms:
  'tw'  time-domain travelling-wave form of Equations (6)-(7) (A/B/C recombination, internal loss, 32 sections);
  'res' reservoir model, Equation (9), with G0, tau_d and E_sat derived from the same card (calibration).
Signals (powers at the chip input, incoherent sum for WDM, all channels at 1550 nm, no gain dispersion):
  * NRZ-OOK at 2.5, 10 and 40 Gb/s, infinite extinction ratio, Gaussian edges with 10-90 % rise time 0.25 T_b.
    The probe channel carries a de Bruijn sequence of order 10 (1024 bits, all 10-bit words once per period);
    each further channel carries an independent random 1024-bit sequence with a random delay.
  * N = 1, 8, 40 channels at fixed TOTAL average power (-30 ... 0 dBm) and at fixed PER-CHANNEL average power
    (-40 ... -10 dBm).
  * 10 Gb/s NRZ-DPSK from a null-biased Mach-Zehnder modulator (nearly constant envelope), single-channel Nyquist
    PDM-16QAM at 32 GBd (root-raised-cosine roll-off 0.1), and an ideal constant envelope (CW).
Integration: RK4 with 32 steps per bit, periodic waveforms, 96 bits of warm-up from the CW steady state.
Metrics (soa_physics.ook_metrics / generic_metrics): standard deviation and peak-to-peak of the gain at the centers of
the probe marks, eye-opening penalty at the best sampling phase, all-time gain standard deviation, gain compression.
Results are cached in ../figures/FigS2_soa_patterning_results.json (run this file to recompute, about 2-4 min).
"""
import json
import os
import time
import numpy as np
import soa_physics as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "figures", "FigS2_soa_patterning_results.json"))
VERSION = "rev3-patterning-1"
SPB = 32
NBITS = 1024
WARM_BITS = 96
RATES = {"2.5G": 400e-12, "10G": 100e-12, "40G": 25e-12}
P_TOT_DBM = np.arange(-30.0, 0.01, 2.5)
P_CH_DBM = np.arange(-40.0, -9.99, 2.5)
NCH = (1, 8, 40)
M_SECTIONS = 32
I_MAIN, I_ALT = 0.15, 0.12
SEED = 20261117


def _w(dbm):
    return 1e-3 * 10 ** (np.asarray(dbm, float) / 10)


def build_cases(seed=SEED):
    """Waveforms (half-step grid, unit mean), time steps and power grids of all cases."""
    rng = np.random.default_rng(seed)
    db = sp.de_bruijn(10)
    S2 = 2 * SPB
    cases = []

    def add(name, kind, tot, Tsym, Pavg_w, probe=None, **meta):
        m = tot.mean()
        cases.append(dict(name=name, kind=kind, w2=tot / m, dt=Tsym / SPB, P=np.asarray(Pavg_w, float),
                          probe=None if probe is None else (probe / m)[::2], meta=meta))

    for rn, Tb in RATES.items():
        chans = [sp.nrz(db, S2)] + [sp.nrz((rng.random(NBITS) < 0.5).astype(float), S2, delay_frac=rng.random())
                                    for _ in range(max(NCH) - 1)]
        for N in NCH:
            tot = np.sum(chans[:N], 0)
            add(f"ook{rn}_N{N}_total", "ook", tot, Tb, _w(P_TOT_DBM), probe=chans[0], rate=rn, N=N, grid="total")
            add(f"ook{rn}_N{N}_perch", "ook", tot, Tb, N * _w(P_CH_DBM), probe=chans[0], rate=rn, N=N, grid="per-channel")
    add("dpsk10G", "gen", sp.dpsk_mzm(db, S2), 100e-12, _w(P_TOT_DBM), rate="10G", N=1, grid="total")
    add("qam16_32GBd", "gen", sp.nyquist_qam_power(NBITS, S2, rng), 31.25e-12, _w(P_TOT_DBM), rate="32GBd", N=1,
        grid="total")
    return cases, db


def _metrics(cases, db, hh, G0_dB):
    out = {}
    for i, cs in enumerate(cases):
        tot = cs["w2"][::2]
        if cs["kind"] == "ook":
            r = sp.ook_metrics(hh[:, i], tot, cs["probe"], db, SPB, G0_dB)
        else:
            r = sp.generic_metrics(hh[:, i], tot, SPB, G0_dB)
        r = {k: np.round(np.asarray(v, float), 5).tolist() for k, v in r.items()}
        r["rel_std_power"] = float(tot.std() / tot.mean())
        r["papr_dB"] = float(10 * np.log10(tot.max() / tot.mean()))
        out[cs["name"]] = r
    return out


def run(verbose=True):
    t_start = time.time()
    cases, db = build_cases()
    w2 = np.array([cs["w2"] for cs in cases])
    dts = np.array([cs["dt"] for cs in cases])
    P = np.array([cs["P"] for cs in cases])
    warm = WARM_BITS * SPB
    res = dict(version=VERSION, config=dict(
        card="SOA-BULK-REF", spb=SPB, nbits=NBITS, pattern="de Bruijn order 10 (probe); independent random 1024-bit "
        "sequences with random delays (other channels)", warm_bits=WARM_BITS, rise_10_90_frac_of_Tb=0.25,
        extinction_ratio="infinite", sections=M_SECTIONS, P_tot_dBm=P_TOT_DBM.tolist(), P_ch_dBm=P_CH_DBM.tolist(),
        rates=RATES, N=list(NCH), seed=SEED,
        exclusions="ASE, gain dispersion, carrier heating, spectral hole burning, alpha-factor phase, inter-channel "
                   "beating and four-wave mixing (incoherent power sum), coupling losses"),
        cases={c["name"]: dict(kind=c["kind"], **c["meta"]) for c in cases}, results={})
    for I in (I_MAIN, I_ALT):
        op = sp.card_operating_point(I)
        sat = sp.saturation_powers(op)
        key = f"{int(round(I * 1e3))}mA"
        sel = list(range(len(cases))) if I == I_MAIN else [i for i, c in enumerate(cases)
                                                          if c["kind"] == "ook" and c["meta"]["N"] == 1
                                                          and c["meta"]["grid"] == "total"]
        sub = [cases[i] for i in sel]
        t0 = time.time()
        h_res = sp.reservoir_periodic(w2[sel], dts[sel], P[sel], op, warm_steps=warm)
        t_res = time.time() - t0
        t0 = time.time()
        h_tw = sp.tw_periodic(w2[sel], dts[sel], P[sel], I, M=M_SECTIONS, warm_steps=warm)
        t_tw = time.time() - t0
        cw_res = 10 * np.log10(op["G0"]) - sp.reservoir_cw(_w(P_TOT_DBM), op) * sp.NP10
        Ns = sp.tw_steady(_w(P_TOT_DBM), I, M=M_SECTIONS)
        cw_tw = op["G0_dB"] - sp.tw_log_gain(Ns) * sp.NP10
        res["results"][key] = dict(
            operating_point=dict(I_A=I, N=op["N"], G0_dB=op["G0_dB"], tau_d_ps=op["tau_d"] * 1e12,
                                 tau_c_ps=op["tau_c"] * 1e12, Esat_pJ=op["Esat"] * 1e12, Psat_mW=op["Psat"] * 1e3,
                                 Pout_3dB_dBm=10 * np.log10(sat["Pout_3dB"] * 1e3),
                                 Pin_3dB_dBm=10 * np.log10(sat["Pin_3dB"] * 1e3),
                                 Pin_1dB_dBm=10 * np.log10(sat["Pin_1dB"] * 1e3)),
            cw_compression_dB=dict(res=np.round(cw_res, 4).tolist(), tw=np.round(cw_tw, 4).tolist()),
            res=_metrics(sub, db, h_res, op["G0_dB"]), tw=_metrics(sub, db, h_tw, op["G0_dB"]),
            runtime_s=dict(res=round(t_res, 1), tw=round(t_tw, 1)))
        if I == I_MAIN:
            # linearized closed form for the all-time gain fluctuation (fixed total power, reservoir parameters)
            lin = {}
            for i, cs in enumerate(cases):
                if cs["kind"] == "ook" and cs["meta"]["grid"] == "total":
                    m = cs["w2"][::2].std()
                    Tb = RATES[cs["meta"]["rate"]]
                    lin[cs["name"]] = np.round(sp.linearized_sigma(cs["P"], m, Tb, op), 5).tolist()
            res["results"][key]["linearized_sigma_all_time_dB"] = lin
        if verbose:
            print(f"{key}: reservoir {t_res:.1f} s, travelling-wave {t_tw:.1f} s")
    res["runtime_total_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def load_or_run():
    """Load the cached study if it matches this VERSION, otherwise run it."""
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("version") == VERSION:
            return r
    return run()


if __name__ == "__main__":
    r = run()
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    m = r["results"]["150mA"]
    print("P_tot (dBm)", P_TOT_DBM)
    print("CW compression tw ", np.array(m["cw_compression_dB"]["tw"]))
    print("CW compression res", np.array(m["cw_compression_dB"]["res"]))
    for name in ("ook10G_N1_total", "ook40G_N1_total", "ook2.5G_N1_total", "ook10G_N8_total", "ook10G_N40_total"):
        for mod in ("tw", "res"):
            print(f"{name:18s} {mod:3s} pp_mark", np.array(m[mod][name]["pp_mark_dB"]), " std_mark",
                  np.array(m[mod][name]["std_mark_dB"]))
    print("runtime", r["runtime_total_s"], "s")
