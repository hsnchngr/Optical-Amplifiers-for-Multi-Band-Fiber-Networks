# Code for "Optical Amplifiers for Multi-Band Fiber Networks: Physics, Architectures, and Reproducible Benchmarking" (review), revision 3

Every figure, table and model number of revision 3 is produced by the scripts in this folder.

- **Parameters** come only from the parameter cards in `../cards`: JSON files read and validated by `cards.py`. Every entry carries a value, unit, source, provenance category and verification level.
- **Measured or digitized reference data** are in `../data`.
- **Outputs** go to `../figures`: a 300 dpi PNG, a vector PDF and a `*_results.json` file holding the numbers quoted in the text.
- **Captions, replacement texts and provenance notes** are in `../notes`. The integration check of all scripts is `../notes/integration_report.md`.

## Requirements

- Python >= 3.10 (tested with 3.11.9).
- NumPy >= 2.0, because the code uses `np.trapezoid` (tested with 2.4.6).
- SciPy (tested with 1.17.1): `solve_ivp`, `solve_bvp`, `least_squares`, `linprog(method="highs")`.
- Matplotlib (tested with 3.11.2).
- **PyTorch only for the PINN scripts** `fig09_pinn.py` and `pinn_parametric.py` (Figures S3 and S4; tested with 2.15.0+cpu). No other script imports it.
- PyMuPDF only for `digitize_donodin2025.py` and stage 1 of `digitize_perin2019.py`, which are not needed to reproduce the figures.
- **OptiSystem only for** `run_optisystem_perin2019.py`, the independent-simulator run of the end-to-end example (Section 5.7, layer L2 source iii): Windows, OptiSystem 23 with a valid license (run with 23.1.0.3744 on 18 November 2026) and a separate Python with pywin32 and NumPy (tested with an Anaconda Python and NumPy 1.26). No other script needs it: `e2e_simulator_compare.py` computes all simulator statistics from the saved run files with the normal environment, and `fig_e2e.py` reads its results file.

    pip install numpy scipy matplotlib     # all figures and tables except Figures S3 and S4
    pip install torch                      # only for fig09_pinn.py and pinn_parametric.py

## How to run everything

Run the scripts from this folder. `fig_soa_raman.py` and `fig_soa_patterning.py` write to `../figures` relative to the working directory; all other scripts use paths relative to their own location. Use a non-interactive Matplotlib backend: `export MPLBACKEND=Agg` in bash, or `$env:MPLBACKEND = "Agg"` in PowerShell.

    python cards.py                         # validates EVERY card file in ../cards; exit status 1 on any problem
    python fig01_capacity.py
    python fig03_spectroscopy.py
    python fig04_fig10_benchmark.py --tables
    python fig_intercomparison.py
    python fig_sensitivity.py
    python noise_checks.py
    python tables_bands.py
    python soa_patterning.py                # computation for Figure 4d and Figure S2 (result cache)
    python fig_soa_raman.py
    python fig_soa_patterning.py
    python fig_sec4.py
    python e2e_perin.py                     # Figure 12 numbers and ../cards/edfa_perin2019_calibrated.json
    python e2e_perin.py --export            # simulator input files in ../notes/optisystem_perin2019
    # optional, needs OptiSystem and pywin32 (see Requirements); the saved run files are distributed:
    #   <python with pywin32> run_optisystem_perin2019.py --convergence
    python e2e_simulator_compare.py         # simulator vs open model and measurement, from the saved run files
    python fig_e2e.py
    python fig06_fig08_hybrid.py            # reads Fig12_end_to_end_results.json
    python fig02_fig05_schematics.py        # Figure 2 lists the card files present at run time
    python fig_toc.py                       # Table of Contents graphic (55 mm x 50 mm, 600 dpi, PNG and TIFF)
    python fig09_pinn.py                    # needs PyTorch
    python pinn_parametric.py               # needs PyTorch
    python fig_numerics.py                  # last, on an idle machine (run-time table S1)
    python fig_applications.py              # no-op: the applications figure was removed in revision 3
    python digitize_perin2019.py calibrate --check   # optional: data/perin2019_* follow exactly from data/raw_perin2019

The order matters only in these cases:

- `e2e_perin.py` must run first for four scripts:
  - `fig_e2e.py`, which runs `e2e_perin.py` itself if the results file is missing;
  - `e2e_simulator_compare.py`, which imports the open model and reads `../notes/optisystem_perin2019/open_model_reference.csv` (written by `e2e_perin.py --export`);
  - `fig06_fig08_hybrid.py`, which uses the identified coefficient scale for one sensitivity case;
  - `fig02_fig05_schematics.py`, which lists the calibrated card.
- `e2e_simulator_compare.py` must run before `fig_e2e.py`: panel e of Figure 12 reads the simulator rows from `../figures/E2E_simulator_results.json`, and `fig_e2e.py` stops if that file is missing or its status is not `complete`. `run_optisystem_perin2019.py` is needed only to repeat the simulator run itself.
- `soa_patterning.py` should run before `fig_soa_raman.py` and `fig_soa_patterning.py`. Otherwise those scripts compute the cache themselves.
- `fig09_pinn.py` should run before:
  - `fig_numerics.py`, whose Table S1 reads the PINN training and inference times and the accuracy of the least-squares identification of C_R (2000 noise realizations) from `../figures/FigS3_pinn_results.json`, and falls back to the submitted-version training time and to its own 200 realizations if that file is missing (`python fig_numerics.py --table-only` rewrites Table S1 without new timings);
  - `pinn_parametric.py`.
- `fig_numerics.py` measures run times, so run it alone on an idle machine.

**Run time.** The non-PINN scripts take about 40 min in total on an i9-14900HX, one core per script, with other jobs running at the same time. The longest steps are:

- `fig06_fig08_hybrid.py`: about 15 min (sequential LP with the full Raman model in the loop); `--plot-only` redraws Figures 9 and S5 from the result files in seconds, `--plot-only --fig09-only` Figure 9 alone;
- `e2e_perin.py`: about 6 min on an idle machine, 12 min measured under load;
- `fig_numerics.py`: 3 min measured (its owner gives up to 10 min);
- `fig_sec4.py`: about 2.5 to 5.5 min;
- `soa_patterning.py` and `fig_intercomparison.py`: about 1 min each.

All other scripts take seconds (`e2e_simulator_compare.py`: about 8 s). The PINN scripts take longer: `fig09_pinn.py` about 17 min, `pinn_parametric.py` about 1 h. The OptiSystem run (`run_optisystem_perin2019.py --convergence`) took about 1 min, including two OptiSystem starts. Rerunning reproduces the PNG files and the result numbers exactly; only run times and dates change. The OptiSystem run itself was made once (18 November 2026); from its saved run files, `e2e_simulator_compare.py` reproduces `E2E_simulator_results.json` exactly.

## Scripts: outputs and inputs

Figure numbers are those of revision 3. Several script names are kept from earlier versions and do not match the figure numbers they now produce. Cards are given by `card_id`; the files are listed under "Parameter cards" below.

| Script | Produces (in `../figures` unless stated) | Cards read | Other inputs |
|---|---|---|---|
| `fig01_capacity.py` | Figure 1, `Fig01_traffic_capacity` | none (no model) | `data/fig01_traffic_capacity_data.json` |
| `fig02_fig05_schematics.py` | Figure 2, `Fig02_framework`; Figure 7, `Fig07_architectures` | every card file in `../cards` (listed in Figure 2, each loaded and validated) | none |
| `fig03_spectroscopy.py` | Figure 3, `Fig03_bands_spectroscopy` | REF-FIG03-BANDS-SPECTRO, EDFA-PERIN2019 | `data/perin2019_giles_coefficients.csv`, `data/raman_ssmf_gnpy.csv` |
| `fig_soa_raman.py` | Figure 4, `Fig04_soa_physics`; Figure 5, `Fig05_raman_physics` | SOA-BULK-REF, SOA-BAND-ILL, RAMAN-G652 | `data/raman_ssmf_gnpy.csv`; cache `FigS2_soa_patterning_results.json` |
| `fig04_fig10_benchmark.py` | Figure 6, `Fig06_literature_records`. With `--tables` it also prints Tables 1 and 2 as Markdown. It computes the single-pump 3-dB band of the distributed-Raman row of Table 1 (4.9 THz at 15 dB peak on-off gain; key `table1_raman_single_pump_band` of the results file, checked against `table1_envelope.csv`) | none (literature data) | `data/table1_envelope.csv`, `data/table2_records.csv`, `data/donodin2025_fig3ef_digitized.csv`, `data/raman_ssmf_gnpy.csv` |
| `fig_sec4.py` | Figure 8, `Fig08_architecture_tradeoffs`. `--replot` redraws from the results file | EDFA-GD91-T1, RAMAN-G652, CFG-FIG08-ARCH | `data/perin2019_giles_coefficients.csv`, `data/raman_ssmf_gnpy.csv` |
| `fig06_fig08_hybrid.py` | Figure 9, `Fig09_hybrid_design`; Figure S5, `Fig13_pump_budget`. `--fig09-only` computes Figure 9 only; `--plot-only` redraws both figures from the result files (with `--fig09-only`: Figure 9 only) | HYBRID-C80, RAMAN-G652, EDFA-PERIN2019 | `data/raman_ssmf_gnpy.csv`, `data/perin2019_giles_coefficients.csv`, `Fig12_end_to_end_results.json` |
| `fig_sensitivity.py` | Figure 10, `Fig10_sensitivity`. `--plot-only` redraws it from the results file | EDFA-GD91-T1, EDFA-GD91-T2, SOA-BULK-REF, RAMAN-G652 | `data/perin2019_giles_coefficients.csv` (ASE-shape check) |
| `fig_intercomparison.py` (named `fig07_crossvalidation.py` before revision 3) | Figure 11, `Fig11_model_intercomparison` | EDFA-GD91-T1, SOA-BULK-REF, RAMAN-G652 | `data/raman_ssmf_gnpy.csv` |
| `e2e_perin.py` | Numbers of Figure 12, `Fig12_end_to_end_results.json`, and the calibrated card `../cards/edfa_perin2019_calibrated.json`. `--export` writes `../notes/optisystem_perin2019/*.csv`, `--summary` refreshes the summary block, `--quick` runs a short test | EDFA-PERIN2019 (writes EDFA-PERIN2019-CAL) | `data/perin2019_giles_coefficients.csv`, `data/perin2019_fig8_gain_ase.json` |
| `run_optisystem_perin2019.py` (**OptiSystem, pywin32**) | Independent-simulator run of the end-to-end example (layer L2, source iii): the card EDFA-PERIN2019 (and EDFA-PERIN2019-CAL) entered into the OptiSystem `Er Doped Fiber` component through COM (Giles coefficients and saturation parameter, overlap fixed at 1), acceptance checks C1 (absorption, pump off), C2 (card entry) and C3 (Equation (5) with ASE suppressed), run sets V, N, C, W and, with `--convergence`, Nconv. Writes `../notes/optisystem_perin2019/runs/`: input files, per-run CSV and NPZ files, `C1_absorption.csv`, `run_log.json` and the OptiSystem projects. Modes `--prepare` (input files only, no OptiSystem), `--explore`, `--check`; exit codes in its header. Refuses to start while an OptiSystem process it did not start exists | EDFA-PERIN2019, EDFA-PERIN2019-CAL | `data/perin2019_giles_coefficients.csv`, `../notes/optisystem_perin2019/operating_points.csv` (`e2e_perin.py --export`) |
| `e2e_simulator_compare.py` | `E2E_simulator_results.json`: simulator vs open ASE-resolved model and vs measurement (RMSE, bias, maximum; run sets N, C, W), acceptance checks, numerical check of the simulator settings (Nconv vs N), the simulator's analyzer noise figure, and the diagnostic `ase_bin_placement` (ASE and NF with each noise bin read at its center or at either edge; the bins match the spectrum at their short-wavelength edge). `--selftest` checks the comparison chain on synthetic input and writes nothing to `../figures`. Runs with the normal environment (no pywin32) | EDFA-PERIN2019 (through `e2e_perin.py`) | `../notes/optisystem_perin2019/runs/`, `../notes/optisystem_perin2019/open_model_reference.csv` (open-model and measured values, `e2e_perin.py --export`), `Fig12_end_to_end_results.json` |
| `fig_e2e.py` | Figure 12, `Fig12_end_to_end` (plot only; the McCumber values printed in panel b are recomputed from the card with `e2e_perin.mccumber_test` and checked against the results file; panel e takes the two independent-simulator rows from `E2E_simulator_results.json`) | EDFA-PERIN2019 (through `e2e_perin.py`) | `Fig12_end_to_end_results.json`, `E2E_simulator_results.json`, `data/perin2019_fig8_gain_ase.json`, `data/perin2019_giles_coefficients.csv` |
| `fig_numerics.py` | Figure S1, `FigS1_numerics` (panels a-c: solution procedure, sweep convergence, grid convergence), and Table S1, `FigS1_numerics_runtime_table.md`. The EDFA ODE tolerance study is kept in the results file only (key `edfa_ode_tolerance`), because Figure 11b shows it. Options: `--table-only`, `--no-runtime`, `--quick`, `--no-pin`, `--plot-only` (redraws Figure S1 from the results file and writes nothing else) | EDFA-GD91-T1, SOA-BULK-REF, RAMAN-G652; it also times the models of EDFA-PERIN2019, HYBRID-C80 and SOA-BAND-ILL | `FigS3_pinn_results.json` if present |
| `soa_patterning.py` | Computation for Figure 4d and Figure S2: `FigS2_soa_patterning_results.json`, reused while its `VERSION` is unchanged | SOA-BULK-REF (and SOA-BAND-ILL through `soa_physics.py`) | none |
| `fig_soa_patterning.py` | Figure S2, `FigS2_soa_patterning` (plot of the cache) | as `soa_patterning.py` | `FigS2_soa_patterning_results.json` |
| `fig09_pinn.py` (**PyTorch**) | Figure S3, `FigS3_pinn`: PINN for one fixed Raman boundary-value problem and C_R identification, with the least-squares baseline and the Cramér-Rao bound. About 17 min with 4 worker processes. Options: `--workers`, `--forward-seeds`, `--seeds`, `--realizations`, `--joint-realizations`, `--ls-realizations`, `--quick`, `--no-timing`, `--plot-only`, `--weight-study`, `--outdir` | RAMAN-G652 | none |
| `pinn_parametric.py` (**PyTorch**) | Figure S4, `FigS4_pinn_parametric`: parameterized PINN; trained weights in `../figures/pinn_models/`. Training takes about 50 min with 4 threads. Options: `--stage all\|train\|eval\|timing\|plot`, `--seeds`, `--sup-seeds`, `--threads`, `--no-supervised`, `--retrain`, `--outdir` and training settings | RAMAN-G652 | `FigS3_pinn_results.json` |
| `noise_checks.py` | `noise_conventions_check_results.json`: numerical checks of Equations (13) to (16), the 1480 nm noise-figure floor and the crude SOA chip noise figure | EDFA-GD91-T1, SOA-BULK-REF, RAMAN-G652 | `data/raman_ssmf_gnpy.csv` |
| `tables_bands.py` | `Tables_bands_results.json` and Markdown tables on stdout: band requirements with inter-channel SRS, modeling requirements beyond the C-band, Raman fiber types, BDFA card test | MULTIBAND-G652D-SPAN, RAMAN-FIBER-TYPES-REF, BDFA-FIRSTOV2020-REF | `data/raman_ssmf_gnpy.csv` |
| `fig_applications.py` | nothing (the applications figure was removed in revision 3; the script deletes outdated outputs) | none | none |
| `fig_toc.py` | Table of Contents graphic, `ToC_graphic.png` and `ToC_graphic.tif` (RGB, LZW; 55 mm x 50 mm at 600 dpi, all labels 10 pt, no inset, no panel label): O- to U-band gain media (same card entries as Figure 3a) and the four framework layers as a loop; `ToC_graphic_results.json` lists the drawn entries and the size and font checks | REF-FIG03-BANDS-SPECTRO | none |
| `digitize_donodin2025.py` | `data/donodin2025_fig3ef_digitized.csv` from the PDF of ref. [69] (CC BY; not distributed). Run as `python -I digitize_donodin2025.py <pdf> [out.csv]`. Needs PyMuPDF | none | publisher PDF |
| `digitize_perin2019.py` | `data/perin2019_giles_coefficients.csv` and `data/perin2019_fig8_gain_ase.json` from Figs. 7 and 8 of arXiv:1803.07905, in two stages. Stage 1, `python -I digitize_perin2019.py extract <pdf> [raw_dir]` (needs PyMuPDF and the PDF, not distributed): raw vector paths to `data/raw_perin2019/`. Stage 2, `python digitize_perin2019.py calibrate [--raw DIR] [--out DIR] [--check]` (NumPy only): axis calibration and resampling (values in the script docstring); it reproduces both data files byte by byte from the distributed raw paths, and `--check` confirms this without writing | none | arXiv PDF (stage 1); `data/raw_perin2019/` (stage 2) |

Library modules (imported, not run):

- `cards.py`: loads and validates cards; CSV readers for `../data`.
- `models.py`: reference EDFA, SOA and Raman models with cards EDFA-GD91-T1, SOA-BULK-REF and RAMAN-G652.
- `style.py`: shared plotting style: Wiley-VCH conventions, color-vision-deficiency-checked palette, `style.label` for the panel labels a), b), ...; `style.math_in_text_font` sets math symbols in Arial, the text font (used by the scripts of Figures 4, 5, 8 to 11 and S1 to S5)
- `edfa_multipass.py`: single- and double-pass EDFA with forward and backward ASE, solved by iterative sweeps.
- `raman_physics.py`: distributed-Raman pumping schemes, equivalent noise figure and OSNR at equal nonlinear phase.
- `soa_physics.py`: SOA band model (card SOA-BAND-ILL) and the time-domain reservoir and sectioned models.
- `hybrid.py`: hybrid Raman/EDFA span with the first-order LP design and the full multi-wavelength Raman model.
- `pinn_common.py`: shared PINN helpers; PyTorch is used only by the PINN scripts.

## Parameter cards (`../cards`)

All 15 cards are registered in `cards._FILES`; `python cards.py` validates each file.

| card_id | File | Used by |
|---|---|---|
| EDFA-GD91-T1 | `edfa_gd91_t1.json` | `models.py`, so every script that uses the reference EDFA (Figures 8, 10, 11, S1; `noise_checks.py`) |
| EDFA-GD91-T2 | `edfa_gd91_t2.json` | `fig_sensitivity.py` (parameter-mixing example, Figure 10d) |
| SOA-BULK-REF | `soa_bulk_ref.json` | `models.py`, `soa_physics.py`, `soa_patterning.py` (Figures 4, 10, 11, S2) |
| SOA-BAND-ILL | `soa_band_ill.json` | `soa_physics.py` (Figure 4b) |
| RAMAN-G652 | `raman_g652.json` | `models.py`, `raman_physics.py`, `hybrid.py`, PINN scripts (Figures 5, 8 to 11, S1, S3 to S5) |
| EDFA-PERIN2019 | `edfa_perin2019.json` | `e2e_perin.py`, `fig03_spectroscopy.py`, `hybrid.py` (Figures 3, 9, 12, S5); entered into OptiSystem by `run_optisystem_perin2019.py` |
| EDFA-PERIN2019-CAL | `edfa_perin2019_calibrated.json` | written by `e2e_perin.py` (layer L4 output); entered into OptiSystem by `run_optisystem_perin2019.py` (run set C) |
| HYBRID-C80 | `hybrid_c80.json` | `hybrid.py` (Figures 9, S5) |
| CFG-FIG08-ARCH | `cfg_fig08_architectures.json` | `fig_sec4.py` (Figure 8) |
| REF-FIG03-BANDS-SPECTRO | `ref_fig03_bands_spectroscopy.json` | `fig03_spectroscopy.py` (Figure 3), `fig_toc.py` (Table of Contents graphic) |
| MULTIBAND-G652D-SPAN | `multiband_g652d_span.json` | `tables_bands.py` |
| RAMAN-FIBER-TYPES-REF | `raman_fiber_types_reference.json` | `tables_bands.py` |
| BDFA-FIRSTOV2020-REF | `bdfa_firstov2020_reference.json` | `tables_bands.py` |
| TDFA-YUCEL2022 | `tdfa_yucel2022.json` | no script (listed in Figure 2 only): abstract-level card of the OptiSystem study of multi-pass S-band TDFAs, ref. [88]; configuration entries only, no material parameter |
| PDFA-CIHANGIR2025 | `pdfa_cihangir2025.json` | no script (listed in Figure 2 only): abstract-level card of the simulation study of triple-pass O-band PDFAs, ref. [89]; configuration entries only, no material parameter |

## Data (`../data`)

| File | Content and origin |
|---|---|
| `raman_ssmf_gnpy.csv` | Default SSMF Raman gain-efficiency profile distributed with GNPy (BSD-3-Clause), transcribed; the header records which elements were checked (all 90 against a downloaded copy of the repository file, 6 against the master branch). Record the GNPy commit hash before release |
| `perin2019_giles_coefficients.csv` | Giles alpha and g* of the EDF of ref. [124], digitized from arXiv:1803.07905 Fig. 7 (`digitize_perin2019.py`, stage 2 from `raw_perin2019/`) |
| `perin2019_fig8_gain_ase.json` | Measured gain and ASE of ref. [124], Fig. 8, digitized, with the source's own model curves (`digitize_perin2019.py`, stage 2 from `raw_perin2019/`) |
| `raw_perin2019/` | Raw vector paths of Figs. 7 and 8 of [124] in PDF page coordinates (stage 1 of `digitize_perin2019.py`): `fig7_alpha_raw.npy`, `fig7_gstar_raw.npy`, `fig8_raw.json`; described in its `README.md` |
| `donodin2025_fig3ef_digitized.csv` | Gain and noise figure of the S-band BDFA of ref. [69], Fig. 3e,f (`digitize_donodin2025.py`) |
| `fig01_traffic_capacity_data.json` | Traffic and capacity data of Figure 1, with source, locator and verification level for each value |
| `table1_envelope.csv`, `table2_records.csv` | Tables 1 and 2: literature values with operating conditions and the re-check record. The single-pump Raman band of Table 1 (4.9 THz) is computed by `fig04_fig10_benchmark.py` |

Simulator files of the end-to-end example (`../notes/optisystem_perin2019`):

| File | Content and origin |
|---|---|
| `operating_points.csv`, `open_model_reference.csv`, `giles_coefficients_dB_per_m.csv`, `effective_cross_sections_unit_overlap.csv` | Simulator inputs and open-model reference values with the measured values, written by `e2e_perin.py --export` |
| `runs/inputs/` | Files loaded by OptiSystem (Giles coefficients of the nominal and calibrated card in dB/m, overlap file) and `plan.json`, written by `run_optisystem_perin2019.py` |
| `runs/*_P<pump>mW.csv`, `runs/*_P<pump>mW.npz`, `runs/C1_absorption.csv` | OptiSystem outputs per run set and pump power (channel powers, gain, ASE in 0.1 nm, noise figure; noise bins and analyzer data) and the absorption check |
| `runs/run_log*.json`, `runs/attempt_*.json`, `runs/explore_parameters.json`, `runs/*.osd` | Run records (software version, every parameter as set and read back, acceptance checks), the record of a start that failed for lack of a license, the component parameters, and the OptiSystem projects |
| `../optisystem_perin2019_results.md`, `../optisystem_input_perin2019.md` | Results note with the independent check, and the input sheet of the run |

## Notes

- The ASE-free closed forms and the numerical solvers are compared in Figure 11 and Figure S1. These are code-verification residuals: they measure the integrator tolerance, not model accuracy.
- Material parameters are read only from the cards. A few operating-point and assumed component values are still also written in scripts, for example the amplifier noise figures in `noise_checks.py` and `fig_soa_raman.py`; they are listed in the integration report. To change a parameter, edit the card file and rerun the scripts that use it (table above), then `fig02_fig05_schematics.py` if card files were added or removed.
- Do not edit the generated card `edfa_perin2019_calibrated.json`; rerun `e2e_perin.py` instead.
