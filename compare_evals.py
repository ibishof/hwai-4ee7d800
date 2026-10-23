"""
Compare two or more eval runs on the same questions, side by side.
Every run is rescored with the current scorer in eval_generic.py, so a scoring
fix applies to old runs too.

  python compare_evals.py out/evals/generic_questions_full_<time>.json out/evals/generic_questions_sql_<time>.json

Prints a per-question table, totals per system, and totals per question kind,
and writes the per-question table to out/evals/comparison.csv for slides.
"""
import csv
import json
import sys
from pathlib import Path

from eval_generic import score

MARK = {True: "pass", False: "FAIL", None: "review"}


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for r in d["results"]:
        s = score(r, {"answer_text": r["answer"], "result": r["result"]})
        rows[(r["patient"], r["kind"])] = {**r, **{"pass": s["pass"], "missing": s["missing"],
                                                     "verdict_leaks": s["verdict_leaks"]}}
    return d["system"], rows


def totals(rows):
    rs = list(rows.values())
    graded = [r for r in rs if r["pass"] is not None]
    return {"correct": f"{sum(bool(r['pass']) for r in graded)}/{len(graded)}",
            "verified": f"{sum(r['verified'] for r in rs)}/{len(rs)}",
            "fixes": sum(r["fixes"] for r in rs),
            "calls": sum(r["calls"] for r in rs),
            "input tokens": f"{sum(r['input_tokens'] for r in rs):,}",
            "output tokens": f"{sum(r['output_tokens'] for r in rs):,}",
            "cost": f"${sum(r['cost'] for r in rs):.2f}",
            "seconds": round(sum(r["seconds"] for r in rs)),
            "verdict leaks": sum(bool(r["verdict_leaks"]) for r in rs)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runs = [load(p) for p in sys.argv[1:]]
    if len(runs) < 2:
        raise SystemExit("Give two or more results files.")
    names = [n for n, _ in runs]
    keys = [k for k in runs[0][1] if all(k in rows for _, rows in runs)]

    print(f"{'question':<28}" + "".join(f"{n:>34}" for n in names))
    table = []
    for k in keys:
        line = f"{k[0].split()[0] + ' ' + k[1]:<28}"
        rec = {"patient": k[0], "kind": k[1]}
        for n, rows in runs:
            r = rows[k]
            line += f"{MARK[r['pass']]:>8} {r['calls']:>2} calls {r['fixes']} fix ${r['cost']:.3f} {r['seconds']:>4.0f}s"
            rec.update({f"{n} result": MARK[r["pass"]], f"{n} calls": r["calls"], f"{n} fixes": r["fixes"],
                        f"{n} cost": r["cost"], f"{n} seconds": r["seconds"],
                        f"{n} problems": "; ".join(r["missing"] + [f"unlabeled: {w}" for w in r["verdict_leaks"]])})
        print(line)
        table.append(rec)

    print("\nTotals")
    tots = {n: totals({k: rows[k] for k in keys}) for n, rows in runs}
    for field in tots[names[0]]:
        print(f"  {field:<15}" + "".join(f"{str(tots[n][field]):>16}" for n in names))

    print("\nBy question kind (correct, calls, cost)")
    for kind in dict.fromkeys(k[1] for k in keys):
        line = f"  {kind:<15}"
        for n, rows in runs:
            sub = [rows[k] for k in keys if k[1] == kind]
            graded = [r for r in sub if r["pass"] is not None]
            c = f"{sum(bool(r['pass']) for r in graded)}/{len(graded)}" if graded else "review"
            line += f"{c:>8} {sum(r['calls'] for r in sub):>3}c ${sum(r['cost'] for r in sub):.2f}"
        print(line)

    for n, rows in runs:
        bad = [(k, rows[k]) for k in keys if rows[k]["pass"] is False]
        for k, r in bad:
            print(f"\n{n} FAIL {k[0]} {k[1]}: missing {r['missing']}, unlabeled {r['verdict_leaks']}")

    out = Path(sys.argv[1]).parent / "comparison.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"\ntable: {out}")


if __name__ == "__main__":
    main()
