# Raw vector paths of ref. [124], Figs. 7 and 8

Output of stage 1 of `code/digitize_perin2019.py` (PyMuPDF; arXiv:1803.07905v3, pages 9 and 10). The files are kept so that the two Perin data files can be regenerated without the PDF. Coordinates are PDF page coordinates in points (origin at the top left, y downward).

| File | Content |
|---|---|
| `fig7_alpha_raw.npy`, `fig7_gstar_raw.npy` | (100, 2) arrays: vertices (x, y) of the alpha(lambda) and g*(lambda) curves of Fig. 7 |
| `fig8_raw.json` | For each pump power (18.1, 31.9, 45.8, 59.6, 87.2 and 114.9 mW): `marks`, the centers of the 80 measurement markers (gain panel left of x = 310 pt, ASE panel right of it); `curves`, the vertices of the source's two model curves (40 each) |

Stage 2, `python digitize_perin2019.py calibrate` (NumPy only), applies the axis calibrations listed in the script and writes `../perin2019_giles_coefficients.csv` and `../perin2019_fig8_gain_ase.json`. With `--check` it compares its result with both files byte by byte instead of writing; both are reproduced exactly (checked 18 Nov 2026). Stage 1 run on the arXiv PDF reproduces these three files byte by byte.

The files were produced on 17 Nov 2026 during the analysis phase and copied here unchanged. Same terms as the data files: digitized from the arXiv figures of [124]; redistribution to be agreed with the source authors.
