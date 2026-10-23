"""
Build every patient's context in one command.

  python build_all.py data\\synthea_sample_data_fhir_latest
  python build_all.py data\\synthea_sample_data_fhir_latest --only Merlene Cole   # some patients

Runs the three steps in order, stopping if one fails:
  1. parse_fhir.py      FHIR bundle -> one SQLite database per patient (out/db),
                        with everything removed logged to out/removed
  2. group_record.py    database -> grouped record, one entry per code and unit (out/grouped)
  3. render_context.py  grouped record -> the context text with line IDs (out/context),
                        checking that every reading survives the compression
Then build_patient_index.py writes patients.csv for the name lookup, if present.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(args):
    print(f"\n$ python {' '.join(args)}", flush=True)
    if subprocess.run([sys.executable, *args]).returncode != 0:
        raise SystemExit(f"Stopped: {args[0]} failed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="folder of FHIR bundle .json files")
    ap.add_argument("--out", default="out")
    ap.add_argument("--only", nargs="*", help="name prefixes, e.g. Merlene Cole")
    args = ap.parse_args()
    only = ["--only", *args.only] if args.only else []
    run(["parse_fhir.py", args.data_dir, "--out", args.out, *only])
    run(["group_record.py", "--out", args.out, *only])
    run(["render_context.py", "--out", args.out, *only])
    if Path("build_patient_index.py").exists():
        run(["build_patient_index.py", args.data_dir])
    print("\nDone. Try:  python ask.py \"Merlene Thompson\" \"What is she allergic to?\"   or   python app.py")


if __name__ == "__main__":
    main()
