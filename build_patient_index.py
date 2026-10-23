"""Build a patient index (one row per patient) from Synthea FHIR bundles.

Usage:
    python build_patient_index.py [DATA_DIR] [--out patients.csv]
"""
import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_DATA_DIR = r"C:\Users\ibish\repo\hwai-4ee7d800\data\synthea_sample_data_fhir_latest"


def clean(name_part: str) -> str:
    """Synthea appends digits to names ('John539'); strip them."""
    return re.sub(r"\d+", "", name_part).strip()


def name_by_use(patient: dict, use: str) -> dict | None:
    return next((n for n in patient.get("name", []) if n.get("use") == use), None)


def format_name(name: dict | None) -> str:
    if not name:
        return ""
    given = " ".join(clean(g) for g in name.get("given", []))
    return f"{given} {clean(name.get('family', ''))}".strip()


def mrn(patient: dict) -> str:
    """Medical record number: the identifier whose type code is 'MR'."""
    for ident in patient.get("identifier", []):
        codings = ident.get("type", {}).get("coding", [])
        if any(c.get("code") == "MR" for c in codings):
            return ident.get("value", "")
    return ""


def find_patient(bundle: dict) -> dict | None:
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") == "Patient":
            return res
    return None


def build_index(data_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(data_dir.glob("*.json")):
        # utf-8 explicitly: Windows defaults to cp1252, which breaks names like 'Germán'
        bundle = json.loads(path.read_text(encoding="utf-8"))
        patient = find_patient(bundle)
        if patient is None:  # hospitalInformation / practitionerInformation files
            continue
        rows.append({
            "patient_id": patient["id"],
            "mrn": mrn(patient),
            "display_name": format_name(name_by_use(patient, "official")),
            "maiden_name": format_name(name_by_use(patient, "maiden")),
            "birth_date": patient.get("birthDate", ""),
            "sex": patient.get("gender", ""),
            "deceased_date": patient.get("deceasedDateTime", "")[:10],
            "file": path.name,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default="patients.csv")
    args = ap.parse_args()

    rows = build_index(Path(args.data_dir))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} patients to {args.out}")
    for r in rows[:5]:
        print(f"  {r['display_name']:30s} {r['sex']:6s} {r['birth_date']}  MRN {r['mrn'][:8]}"
              + (f"  deceased {r['deceased_date']}" if r["deceased_date"] else ""))


if __name__ == "__main__":
    main()
