# Optical Amplifiers for Multi-Band Fiber Networks: code and parameter cards

Code, parameter cards and data for the review article

> H. Cihangir, M. Yücel, H. Çiğ, "Optical Amplifiers for Multi-Band Fiber Networks: Physics, Architectures, and
> Reproducible Benchmarking" (submitted to Laser & Photonics Reviews).

Every figure, every computed table entry and every model number of the article is produced by the scripts in `code/`
from the parameter cards in `cards/`. Model parameters are not typed into the scripts: they are read from the cards.

Repository: https://github.com/hsnchngr/Optical-Amplifiers-for-Multi-Band-Fiber-Networks

## Contents

| Folder | Content |
|---|---|
| `cards/` | Parameter cards (JSON), one file per amplifier, fiber or configuration |
| `code/` | Python models, figure scripts and table scripts; `code/README.md` lists every script with its inputs and outputs |
| `data/` | Reference data used by the scripts (literature values with their sources, digitized curves, search log) |
| `figures/` | Outputs: 300 dpi PNG, vector PDF and a `*_results.json` file with the numbers quoted in the article |
| `notes/` | Provenance notes and the integration report of all scripts |

## Parameter cards (`cards/`)

A card is a JSON file that holds every parameter of one model case. Each entry carries

- `value` and `unit`;
- `source`: the reference, table or figure the value comes from;
- `category`: provenance of the value (`measured`, `derived`, `lit-same-class`, `lit-other`, `spec`, `assumed`,
  `fitted` or `config`);
- `verification`: level `V0` to `V3` (from unchecked to checked against the full text of the source).

`code/cards.py` loads and validates the cards. The card files are

| File | Case |
|---|---|
| `edfa_gd91_t1.json`, `edfa_gd91_t2.json` | EDFA fibers of Giles and Desurvire (1991), Tables I and II |
| `edfa_perin2019.json` | C-band EDFA of Perin et al. (2019), nominal card of the end-to-end example |
| `edfa_perin2019_calibrated.json` | The same amplifier with the identified entries (least-squares fit) |
| `soa_bulk_ref.json`, `soa_band_ill.json` | Bulk SOA reference case and illustrative band cases |
| `raman_g652.json`, `raman_fiber_types_reference.json` | Raman amplification in G.652 fiber; fiber-type reference values |
| `hybrid_c80.json` | Hybrid Raman/EDFA design on an 80 km span |
| `multiband_g652d_span.json` | 80 km G.652.D span loaded in all six ITU-T bands |
| `pdfa_cihangir2025.json`, `tdfa_yucel2022.json`, `bdfa_firstov2020_reference.json` | PDFA, TDFA and BDFA cases |
| `cfg_fig08_architectures.json`, `ref_fig03_bands_spectroscopy.json` | Configuration and reference values of Figures 8 and 3 |

## Requirements

- Python 3.10 or later (tested with 3.11), NumPy 2.0 or later, SciPy, Matplotlib.
- PyTorch only for the two PINN scripts (`fig09_pinn.py`, `pinn_parametric.py`).
- PyMuPDF only for the digitizing scripts.
- OptiSystem (commercial, Windows) only for `run_optisystem_perin2019.py`. Its exported outputs are included, so the
  simulator comparison can be recomputed without a license.

```bash
pip install numpy scipy matplotlib
```

## How to run

All scripts are run from the `code/` folder and write to `figures/`:

```bash
cd code
python fig03_spectroscopy.py
```

`code/README.md` gives the complete list (script, figure or table, inputs) and the order in which to run everything.

## Data of Perin et al. (2019)

The end-to-end example (Section 5.7, Figure 12) uses absorption and gain coefficients and measured gain and ASE spectra
digitized from the figures of the preprint arXiv:1803.07905v3. These digitized files are **not redistributed** here.
To regenerate them, download the preprint PDF from arXiv and run `digitize_perin2019.py` in `code/` (the header of
that script describes its two stages and arguments). The scripts of the end-to-end example then run as described in
`code/README.md`.

## Third-party data

| Data | Source | License |
|---|---|---|
| `data/donodin2025_fig3ef_digitized.csv` | Digitized from Figure 3e,f of Donodin et al. (2025) | CC BY 4.0 |
| `data/raman_ssmf_gnpy.csv` | Default SSMF Raman gain-efficiency profile of GNPy (https://github.com/Telecominfraproject/oopt-gnpy) | BSD-3-Clause |

Literature values in `data/` are listed with the reference of each value.

## License

Code: MIT License (see `LICENSE`). Parameter cards and the authors' own data files: CC BY 4.0. Third-party data keep
the licenses listed above.

## Citation

If you use this code or the cards, please cite the article above.

## Contact

Hasan Cihangir, Harran University, cihangir@harran.edu.tr
