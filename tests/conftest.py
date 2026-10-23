"""Shared setup: run tests from the repo folder with `python -m pytest tests`.

Data tests need the built outputs (python build_all.py <data_dir>) and skip
themselves if those are missing. FHIR_DATA can point at the bundle folder."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / os.environ.get("OUT_DIR", "out")
DATA = ROOT / os.environ.get("FHIR_DATA", "data/synthea_sample_data_fhir_latest")


def need(path):
    if not path.exists() or not any(path.iterdir()):
        pytest.skip(f"{path} not built; run python build_all.py first")
    return path
