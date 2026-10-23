"""
Print the "What is currently wrong with this patient?" answers from several
eval runs side by side, one patient at a time, for a human read.

  python compare_general.py out\\evals\\generic_questions_context_<time>.json out\\evals\\generic_questions_full_<time>.json out\\evals\\generic_questions_sql_<time>.json

Add --kind trend (or any other kind) to compare a different question.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="results files from eval_generic.py")
    ap.add_argument("--kind", default="general")
    args = ap.parse_args()

    runs = []
    for f in args.files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        runs.append((d["system"], {r["patient"]: r for r in d["results"] if r["kind"] == args.kind}))

    wrap = lambda s, ind: textwrap.fill(s, 100, initial_indent=ind, subsequent_indent=ind)
    for patient in dict.fromkeys(p for _, rows in runs for p in rows):
        first = next(rows[patient] for _, rows in runs if patient in rows)
        print("=" * 100)
        print(f"{patient}\nQ: {first['q']}\nKey: {first['gold']}")
        for system, rows in runs:
            r = rows.get(patient)
            if not r:
                continue
            print(f"\n--- {system}  ({'verified' if r['verified'] else 'NOT VERIFIED'}, {r['calls']} calls, "
                  f"{r['fixes']} fixes, ${r['cost']:.3f}, {r['seconds']}s, "
                  f"tools: {', '.join(t['tool'] for t in r['tool_calls']) or 'none'})")
            for a in (r["result"] or {}).get("answers", []):
                print(wrap(f"[{a.get('status')}] {a.get('answer', '')}", "  "))
                for inf in a.get("inferences") or []:
                    print(wrap(f"INFERRED: {inf.get('claim')}", "    "))
                    print(wrap(f"assumption: {inf.get('assumption')}", "      "))
        print()


if __name__ == "__main__":
    main()
