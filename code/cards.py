"""Parameter cards (layer L1 of the framework).

Each card is a JSON file in ../cards. Every entry stores its value and unit together with two independent labels:

  category  where the number comes from
            measured        reported for the device that the card describes
            derived         computed from other entries of the same card by a stated formula
            lit-same-class  literature value for the same host or fiber class, different sample
            lit-other       literature value for a different host, device or transition
            spec            manufacturer specification (limit or typical value)
            assumed         chosen by the authors
            fitted          adjusted to reach a stated target
            config          configuration choice (length, wavelength, operating point), not a material parameter
  level     what has been checked
            V0  transcribed only
            V1  value, unit and applicability confirmed against the primary source
            V2  consistency with other card entries tested and the residual recorded
            V3  validated against an independent measurement on the same device, with the error metric recorded

The scripts read model parameters only through this module, so a figure can always be traced to a card file.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CARD_DIR = os.path.normpath(os.path.join(HERE, "..", "cards"))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

CATEGORIES = ("measured", "derived", "lit-same-class", "lit-other", "spec", "assumed", "fitted", "config")
LEVELS = ("V0", "V1", "V2", "V3")
REQUIRED = ("value", "unit", "category", "source", "applies_to", "verification")

_FILES = {
    "EDFA-GD91-T1": "edfa_gd91_t1.json",
    "EDFA-GD91-T2": "edfa_gd91_t2.json",
    "SOA-BULK-REF": "soa_bulk_ref.json",
    "SOA-BAND-ILL": "soa_band_ill.json",
    "RAMAN-G652": "raman_g652.json",
    "EDFA-PERIN2019": "edfa_perin2019.json",
    "EDFA-PERIN2019-CAL": "edfa_perin2019_calibrated.json",      # written by e2e_perin.py (layer L4 output)
    "HYBRID-C80": "hybrid_c80.json",
    "CFG-FIG08-ARCH": "cfg_fig08_architectures.json",
    "REF-FIG03-BANDS-SPECTRO": "ref_fig03_bands_spectroscopy.json",
    "MULTIBAND-G652D-SPAN": "multiband_g652d_span.json",
    "RAMAN-FIBER-TYPES-REF": "raman_fiber_types_reference.json",
    "BDFA-FIRSTOV2020-REF": "bdfa_firstov2020_reference.json",
    "TDFA-YUCEL2022": "tdfa_yucel2022.json",                    # abstract-level card of a simulation study; no model
    "PDFA-CIHANGIR2025": "pdfa_cihangir2025.json",              # abstract-level card of a simulation study; no model
}
GENERATED = ("EDFA-PERIN2019-CAL",)      # cards written by a script; absent until that script has run


class CardError(ValueError):
    pass


def validate(card):
    """Raise CardError if a required field is missing or a label is not one of the defined codes."""
    for key in ("card_id", "schema_version", "entries"):
        if key not in card:
            raise CardError(f"card lacks '{key}'")
    for name, e in card["entries"].items():
        missing = [f for f in REQUIRED if f not in e]
        if missing:
            raise CardError(f"{card['card_id']}.{name}: missing {missing}")
        if e["category"] not in CATEGORIES:
            raise CardError(f"{card['card_id']}.{name}: unknown category '{e['category']}'")
        if e["verification"].get("level") not in LEVELS:
            raise CardError(f"{card['card_id']}.{name}: unknown verification level")
        if e["category"] == "derived" and "derivation" not in e:
            raise CardError(f"{card['card_id']}.{name}: derived entry without 'derivation'")
        if e["category"] == "fitted" and "fit" not in e:
            raise CardError(f"{card['card_id']}.{name}: fitted entry without 'fit'")
    return True


def load(card_id_or_path):
    """Load and validate a card by its id (see _FILES) or by a path to a JSON file."""
    path = card_id_or_path
    if card_id_or_path in _FILES:
        path = os.path.join(CARD_DIR, _FILES[card_id_or_path])
    with open(path, encoding="utf-8") as f:
        card = json.load(f)
    validate(card)
    return card


def values(card):
    """Plain {name: value} dictionary of a card."""
    return {k: e["value"] for k, e in card["entries"].items()}


def table_rows(card):
    """Rows (name, value, unit, category, level, source) for the manuscript and Supporting Information tables."""
    rows = []
    for k, e in card["entries"].items():
        src = e["source"].get("ref", "")
        if e["source"].get("locator"):
            src += ", " + e["source"]["locator"]
        rows.append((e.get("symbol", k), e["value"], e["unit"], e["category"], e["verification"]["level"], src))
    return rows


def read_csv(name):
    """Read a numeric CSV from ../data with '#' comment lines and one header line; returns (header, array)."""
    import numpy as np
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    header = lines[0].split(",")
    data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
    return header, data


def raman_profile_ssmf():
    """Measured SSMF Raman gain-efficiency profile (GNPy default): returns (offset_THz, g0 [1/(W m)], f_ref [Hz])."""
    _, d = read_csv("raman_ssmf_gnpy.csv")
    return d[:, 0], d[:, 1], 206.184634112792e12


def perin2019_coefficients():
    """Giles coefficients of the EDF of ref. [124], digitized: returns (lambda_nm, alpha [1/m], g* [1/m])."""
    _, d = read_csv("perin2019_giles_coefficients.csv")
    return d[:, 0], d[:, 1], d[:, 2]


if __name__ == "__main__":
    # Self-test: validate EVERY card file in ../cards (registered or not) and check the registry against the files.
    import glob
    import sys
    problems = 0
    ids = {}
    for path in sorted(glob.glob(os.path.join(CARD_DIR, "*.json"))):
        fn = os.path.basename(path)
        try:
            c = load(path)
        except (CardError, ValueError, OSError) as exc:
            print(f"{fn}: INVALID ({exc})")
            problems += 1
            continue
        cats = {}
        for e in c["entries"].values():
            cats[e["category"]] = cats.get(e["category"], 0) + 1
        reg = [k for k, v in _FILES.items() if v == fn]
        note = "" if reg == [c["card_id"]] else f"  REGISTRY MISMATCH (registered as {reg or 'nothing'})"
        problems += bool(note)
        if c["card_id"] in ids:
            note += f"  DUPLICATE card_id (also in {ids[c['card_id']]})"
            problems += 1
        ids[c["card_id"]] = fn
        print(f"{c['card_id']} ({fn}): {len(c['entries'])} entries, valid; categories {cats}{note}")
    for cid, fn in _FILES.items():
        if not os.path.exists(os.path.join(CARD_DIR, fn)):
            if cid in GENERATED:
                print(f"{cid}: {fn} not present yet (generated card; run e2e_perin.py)")
            else:
                print(f"{cid}: registered file {fn} is MISSING")
                problems += 1
    print(f"{len(ids)} card files checked, {problems} problem(s)")
    sys.exit(1 if problems else 0)
