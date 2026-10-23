"""
Eval that works on any patient: questions are generated from each patient's
own database, and the answer key is computed from that database with SQL,
never from the context text (so the context does not grade itself).

Five questions per patient:
  simple   latest value of one lab or vital, with its date
  complex  which medications were ordered for a condition, and when each was last ordered
  multi    latest value of a second item + last immunization date + a common test
           this patient never had (that part must come back not recorded)
  trend    how one item changed over time (first and latest readings must be
           reported; any verdict such as "rising" must be a labeled inference)
  general  "What is currently wrong with this patient?" (scored for coverage of
           active disorders only; the rest needs a human read)

What code can grade, it grades. What it cannot (is a verdict right, is the
general summary sensible), it prints in full for you to judge.

Usage:
  python eval_generic.py --generate                 # write the questions, show them, spend nothing
  python eval_generic.py                            # run them with the current system (record + tools)
  python eval_generic.py --system sql               # SQL-first: index only, every value from run_sql
  python eval_generic.py --system context           # record only, no tools
  python eval_generic.py --patients "Byron Emard" "Pat Rutherford"
  python eval_generic.py --only simple trend        # some question kinds only

Lab names are written the way a clinician says them where PLAIN_NAMES has a
rule ("BUN", "A1c"), so the model has to find the record item itself.

Questions are saved to out/evals/generic_questions.json and reused, so every
system answers the same questions. Delete that file (or pass --generate) to
make new ones.
"""
import argparse
import json
import os
import random
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from ask import ask, describe, find_patient, format_trace, load_env, load_patients

DEFAULT_PATIENTS = ["Byron Emard", "Rogelio Monahan", "Pat Rutherford"]  # small, medium, big
KINDS = ["simple", "complex", "multi", "trend", "general"]
SYSTEMS = {"full": {"use_tools": True},                      # full record + tools (the current system)
           "sql": {"use_tools": True, "sql_first": True},     # index only; every value from run_sql
           "context": {"use_tools": False}}                   # full record, no tools

# Words that state a verdict about a value or trend. Outside an (inferred: ...)
# span they count as an unlabeled judgment. Eval-only: used to measure, not to
# fix answers.
VERDICT = re.compile(r"\b(worse|worsen\w*|improv\w*|stable|steady|steadily|ris(e|es|ing)|rose|increas\w*|"
                     r"decreas\w*|declin\w*|fell|fall(s|ing)?|drop(s|ped)?|trend\w*|controlled|uncontrolled|"
                     r"narrow|flat|unchanged|fluctuat\w*|normal|abnormal|elevated)\b", re.I)
# Clinician wording for record names: the model must map "BUN" to the right
# record item itself. Applied to every question, at generation and at run time.
# Add a line here to reword another lab.
PLAIN_NAMES = [
    (r"Urea nitrogen \[[^\]]*\] in [\w ]+?(?=[,?])", "BUN"),
    (r"Hemoglobin A1c/Hemoglobin\.total in Blood", "A1c"),
]
# "The record does not label these readings as normal or abnormal" describes the
# record, not the patient, so a verdict word right after a negated labeling verb
# is not a leak. "No clear worsening" is still a leak (no labeling verb).
ABOUT_THE_RECORD = re.compile(r"\b(not|no|without|never)\b[^.,;]*\b(label\w*|flag\w*|mark\w*|interpret\w*|"
                              r"reference|range|describ\w*|call\w*|record\w*)\b[^.,;]*$")
MED_STOPWORDS = {"oral", "tablet", "injection", "extended", "release", "hydrochloride", "sodium", "topical"}


# ------------------------------------------------------------------ helpers

def plain_names(q):
    for pattern, plain in PLAIN_NAMES:
        q = re.sub(pattern, plain, q)
    return q


def connect(out, stem):
    return sqlite3.connect(f"file:{out / 'db' / (stem + '.sqlite')}?mode=ro", uri=True)


def num_forms(v):
    """Acceptable ways to write a recorded number: as stored, and rounded."""
    forms = {f"{v:g}" if abs(v) < 1e6 else str(v), str(round(v, 2))}
    if abs(v) >= 10:  # 145.03 may be written 145.0 or 145; 9.98 must not become 10
        forms |= {str(round(v, 1)), str(round(v))}
    forms = {f.rstrip("0").rstrip(".") if "." in f else f for f in forms}
    forms.add(str(float(v)))  # SQL returns 85.0 for 85
    return sorted(forms)


def has(text, phrase):
    """Numbers match whole (104 does not match 104.55); text matches anywhere."""
    if re.fullmatch(r"[\d./ -]+", phrase):
        return re.search(rf"(?<![\d.]){re.escape(phrase)}(?!\.?\d)", text) is not None
    return phrase.lower() in text


def norm_answer(t):
    """Lowercase, plain dashes, and numbers without trailing zeros, so the
    database's 85.0 or 9.980 match a key of 85 or 9.98."""
    t = t.lower().replace("‑", "-").replace("‐", "-").replace("–", "-")
    t = re.sub(r"(?<![\d.])(\d+)\.0+(?![\d])", r"\1", t)        # 85.0 -> 85
    return re.sub(r"(?<![\d.])(\d+\.\d*?[1-9])0+(?![\d])", r"\1", t)  # 9.980 -> 9.98


def recorded_part(answer):
    return re.sub(r"\(inferred:[^)]*\)", " ", answer, flags=re.I)


def med_key(display):
    """The drug's name word: 'Warfarin Sodium 5 MG Oral Tablet' -> 'warfarin'."""
    for w in re.findall(r"[A-Za-z]+", display):
        if len(w) >= 4 and w.lower() not in MED_STOPWORDS:
            return w.lower()
    return display.lower()


def numeric_items(con):
    """Labs and vitals with at least 4 numeric readings: [(code, display, n)]."""
    return con.execute("""
        SELECT code, display, COUNT(*) n FROM observations
        WHERE value_num IS NOT NULL AND category IN ('laboratory', 'vital-signs') AND display NOT LIKE 'Pain severity%'
        GROUP BY code HAVING COUNT(DISTINCT date) >= 4 ORDER BY category = 'laboratory' DESC, code""").fetchall()


def analyte(display):
    """'Urea nitrogen [Mass/volume] in Blood' -> 'urea nitrogen', so a test counts as
    missing only if the patient has no version of it (blood or serum, any unit)."""
    return display.split("[")[0].strip().lower()


def readings(con, code):
    return con.execute("SELECT date, value_num FROM observations WHERE code = ? AND value_num IS NOT NULL "
                       "ORDER BY date", (code,)).fetchall()


def common_tests(out, stems):
    """Lab codes found in at least 10% of all patients: [(code, display, share)], most common first."""
    seen = {}
    dbs = sorted((out / "db").glob("*.sqlite"))
    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for code, disp in con.execute("SELECT DISTINCT code, display FROM observations "
                                      "WHERE category = 'laboratory' AND value_num IS NOT NULL"):
            seen.setdefault(code, [disp, 0])[1] += 1
        con.close()
    return sorted(((c, d, n / len(dbs)) for c, (d, n) in seen.items() if n >= 0.1 * len(dbs)),
                  key=lambda t: (-t[2], t[0]))


# --------------------------------------------------------- question makers

def q_latest(con, item, pron):
    code, disp, _ = item
    rows = readings(con, code)
    last = rows[-1][0]
    vals = [v for d, v in rows if d == last]
    return {"q": f"What was {pron['his']} most recent {disp}?",
            "expect": [[last]] + [num_forms(v) for v in vals],
            "gold": f"{disp}: {', '.join(f'{v:g}' for v in vals)} on {last}"}


def make_questions(out, patient, tests):
    con = connect(out, patient["stem"])
    rng = random.Random(patient["patient_id"])  # same questions every time for this patient
    pron = {"he": "he", "his": "his"} if patient.get("sex", "").startswith("m") else {"he": "she", "his": "her"}
    items = numeric_items(con)
    if len(items) < 3:
        raise SystemExit(f"{patient['display_name']} has fewer than 3 labs/vitals with 4+ readings; pick another patient.")
    labs = [i for i in items if con.execute("SELECT 1 FROM observations WHERE code = ? AND category = 'laboratory'",
                                            (i[0],)).fetchone()]
    trend_item = rng.choice(labs or items)  # a lab trend when the patient has one
    simple_item, multi_item = rng.sample([i for i in items if i != trend_item], 2)
    qs = []

    # simple
    s = q_latest(con, simple_item, pron)
    qs.append({"kind": "simple", **s})

    # complex: medications ordered for one condition, with the latest order of each
    # prefer a diagnosis (not a referral) with the most different drugs, then the most recent
    reasons = con.execute("""SELECT reason, MAX(date) FROM med_requests WHERE reason IS NOT NULL
                             GROUP BY reason ORDER BY reason IN (SELECT display FROM conditions WHERE status = 'active') DESC,
                             reason LIKE '%(disorder)' DESC,
                             COUNT(DISTINCT display) DESC, MAX(date) DESC""").fetchall()
    if reasons:
        reason = reasons[0][0]
        meds = con.execute("SELECT display, MAX(date) FROM med_requests WHERE reason = ? GROUP BY display",
                           (reason,)).fetchall()
        cond = re.sub(r"\s*\((disorder|finding|procedure|situation)\)$", "", reason)
        qs.append({"kind": "complex",
                   "q": f"Which medications have been ordered for {pron['his']} {cond.lower()}, "
                        f"and when was each most recently ordered?",
                   "expect": [[med_key(d)] for d, _ in meds] + [[last] for _, last in meds],
                   "gold": "; ".join(f"{d} (last {last})" for d, last in meds)})
    else:
        qs.append({"kind": "complex", "skip": "no medication with a recorded reason"})

    # multi: second latest value + last immunization + a common test never done
    m = q_latest(con, multi_item, pron)
    imm = con.execute("SELECT display, MAX(date) FROM immunizations GROUP BY display ORDER BY MAX(date) DESC").fetchone()
    have = {analyte(r[0]) for r in con.execute("SELECT DISTINCT display FROM observations")}
    missing = [(c, d) for c, d, share in tests if analyte(d) not in have]
    missing = missing[:10]  # the most common tests this patient lacks
    parts, expect, gold = [m["q"].rstrip("?")], list(m["expect"]), [m["gold"]]
    if imm:
        vaccine = re.sub(r"\s+", " ", imm[0]).split(",")[0]
        noun = "" if "vaccine" in vaccine.lower() else " vaccine"
        parts.append(f"when did {pron['he']} last get the {vaccine}{noun}")
        expect.append([imm[1]])
        gold.append(f"{imm[0]} last {imm[1]}")
    absent = None
    if missing:
        absent = rng.choice(missing)[1]
        parts.append(f"what was {pron['his']} most recent {absent}")
        gold.append(f"{absent}: never recorded (status should be partial, or not_recorded for that part)")
    qs.append({"kind": "multi", "q": ", ".join(parts[:-1]) + ", and " + parts[-1] + "?",
               "expect": expect, "absent": absent, "gold": "; ".join(gold)})

    # trend
    code, disp, _ = trend_item
    rows = readings(con, code)
    (d0, v0), (d1, v1) = rows[0], rows[-1]
    qs.append({"kind": "trend", "q": f"How has {pron['his']} {disp} changed over time?",
               "expect": [[d0], num_forms(v0), [d1], num_forms(v1)],
               "gold": f"{len(rows)} readings, first {v0:g} on {d0}, latest {v1:g} on {d1}"})

    # general
    active = [r[0] for r in con.execute(
        "SELECT DISTINCT display FROM conditions WHERE status = 'active' AND display LIKE '%(disorder)%' ORDER BY display")]
    qs.append({"kind": "general", "q": "What is currently wrong with this patient?",
               "active_disorders": active,
               "gold": "Human review. Active disorders on file: " + ("; ".join(active) or "none")})
    con.close()
    for q in qs:
        if "q" in q:
            q["q"] = plain_names(q["q"])
        q["patient"] = patient["display_name"]
        q["patient_id"] = patient["patient_id"]
    return qs


# ------------------------------------------------------------------ scoring

def statuses(trace):
    return [a.get("status") for a in (trace["result"] or {}).get("answers", []) if isinstance(a, dict)]


def disorder_term(d):
    return re.sub(r"\s*\(disorder\)$", "", d).lower()


def score(item, trace):
    """Code-graded result for one question: {'pass': bool|None, 'missing': [...], ...}.
    pass is None when the question needs a human (general)."""
    answer = norm_answer(trace["answer_text"])
    # a verdict word that is in the record itself ("Abnormal findings ... (finding)") is not a leak
    quoted = " ".join(str(e.get("quote", "")) for x in (trace["result"] or {}).get("answers", []) if isinstance(x, dict)
                      for e in (x.get("evidence") or []) if isinstance(e, dict)).lower()
    text = recorded_part(answer)
    out = {"missing": [], "verdict_leaks": sorted({m.group(0).lower() for m in VERDICT.finditer(text)
                                                   if m.group(0).lower() not in quoted
                                                   and not ABOUT_THE_RECORD.search(text[max(0, m.start() - 70):m.start()])})}
    for group in item.get("expect", []):
        if not any(has(answer, alt) for alt in group):
            out["missing"].append(group[0])
    st = statuses(trace)
    if item["kind"] == "multi" and item.get("absent"):
        out["abstained"] = "not_recorded" in st or "partial" in st
        if not out["abstained"]:
            out["missing"].append(f"status partial/not_recorded for {item['absent']}")
    if item["kind"] == "general":
        terms = [disorder_term(d) for d in item["active_disorders"]]
        found = [t for t in terms if t in answer or " ".join(t.split()[:3]) in answer]
        out["disorder_recall"] = f"{len(found)}/{len(terms)}"
        out["pass"] = None
        return out
    out["pass"] = not out["missing"] and (item["kind"] != "trend" or not out["verdict_leaks"])
    return out


def usage_totals(trace):
    u = trace["usage"]
    return {"calls": len(u),
            "input_tokens": sum(x.get("prompt_tokens", 0) for x in u),
            "output_tokens": sum(x.get("completion_tokens", 0) for x in u),
            "cached_tokens": sum((x.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for x in u),
            "cost": round(sum(x.get("cost") or 0 for x in u), 4)}


# --------------------------------------------------------------------- main

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--patients-csv", default="patients.csv")
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="full")
    ap.add_argument("--patients", nargs="*", default=DEFAULT_PATIENTS)
    ap.add_argument("--only", nargs="*", choices=KINDS, help="question kinds to run")
    ap.add_argument("--generate", action="store_true", help="(re)write the questions and stop")
    ap.add_argument("--questions", help="questions file to run (default out/evals/generic_questions.json)")
    args = ap.parse_args()

    out = Path(args.out)
    qpath = out / "evals" / "generic_questions.json"
    qpath.parent.mkdir(parents=True, exist_ok=True)
    everyone = load_patients(Path(args.patients_csv), out)

    if args.generate or not qpath.exists():
        tests = common_tests(out, [p["stem"] for p in everyone])
        questions = []
        for name in args.patients:
            found = find_patient(name, everyone)
            if len(found) != 1:
                raise SystemExit(f'"{name}" matches {len(found)} patients; be more specific.')
            questions += make_questions(out, found[0], tests)
        qpath.write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(questions)} questions written to {qpath}\n")
        for q in questions:
            print(f"[{q['patient']}] {q['kind']}: {q.get('q', '(skipped: ' + q.get('skip', '') + ')')}")
            print(f"    key: {q.get('gold', '')}")
        if args.generate:
            return

    if args.questions:
        qpath = Path(args.questions)
    questions = json.loads(qpath.read_text(encoding="utf-8"))
    questions = [{**q, "q": plain_names(q["q"])} for q in questions
                 if "q" in q and (not args.only or q["kind"] in args.only)]
    key = os.environ.get("OPENROUTER_API_KEY") or load_env().get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not found in environment or .env")
    by_id = {p["patient_id"]: p for p in everyone}

    results = []
    for item in questions:
        patient = by_id[item["patient_id"]]
        t = ask(patient, item["q"], out, args.model, key, **SYSTEMS[args.system])
        s = score(item, t)
        u = usage_totals(t)
        results.append({**item, **s, **u, "verified": t["verified"], "fixes": len(t["repairs"]),
                        "statuses": statuses(t), "answer": t["answer_text"], "result": t["result"],
                        "tool_calls": t["steps"], "problems": t["problems"], "seconds": t["seconds"]})
        mark = {True: "PASS", False: "FAIL", None: "REVIEW"}[s["pass"]]
        print(f"\n{'=' * 70}\n[{mark}] {describe(patient)} | {item['kind']}")
        print(f"Q: {item['q']}\nKey: {item['gold']}")
        print("  " + format_trace(t).replace("\n", "\n  "))
        if s["missing"]:
            print(f"  MISSING: {s['missing']}")
        if s["verdict_leaks"]:
            print(f"  Verdict words outside (inferred: ...): {s['verdict_leaks']}")
        if "disorder_recall" in s:
            print(f"  Active disorders mentioned: {s['disorder_recall']}")
        print(f"  {u['calls']} calls, {u['input_tokens']:,} in / {u['output_tokens']:,} out, ${u['cost']}, {t['seconds']}s")

    print(f"\n{'=' * 70}\nSystem: {args.system}   Model: {args.model}   Questions: {qpath.name}\n")
    print(f"{'patient':<26}{'kind':<9}{'result':<8}{'verified':<10}{'fixes':<7}{'calls':<7}{'cost':<9}{'sec':<6}")
    for r in results:
        res = {True: "pass", False: "FAIL", None: "review"}[r["pass"]]
        print(f"{r['patient'][:25]:<26}{r['kind']:<9}{res:<8}{str(r['verified']):<10}{r['fixes']:<7}"
              f"{r['calls']:<7}{'$' + str(r['cost']):<9}{r['seconds']:<6}")
    graded = [r for r in results if r["pass"] is not None]
    print(f"\nCorrect (code-graded): {sum(r['pass'] for r in graded)}/{len(graded)}   "
          f"Verified: {sum(r['verified'] for r in results)}/{len(results)}   "
          f"Fixes: {sum(r['fixes'] for r in results)}   "
          f"Answers with verdict words outside inferences: {sum(bool(r['verdict_leaks']) for r in results)}")
    print(f"Total: {sum(r['calls'] for r in results)} calls, {sum(r['input_tokens'] for r in results):,} input tokens, "
          f"${sum(r['cost'] for r in results):.2f}, {sum(r['seconds'] for r in results):.0f}s")
    path = out / "evals" / f"{qpath.stem}_{args.system}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"system": args.system, "model": args.model, "questions": str(qpath),
                                "results": results},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
