"""
Clinician-style questions about Cole Corwin, the largest record (~184K tokens).

Single questions and multi-part questions, the way a doctor might ask them:
current meds, anticoagulation, kidney function, cardiac history, allergies,
screening, and a few traps (a test he never had, a date with no reading,
stale "active" statuses, a treatment that quietly stopped).

Every expected answer was computed from Cole's database with SQL, not read
off the text. A question is "correct" only if every required fact is in the answer, it states
no unmarked detail that is absent from Cole's data (see UNGROUNDED), and, for
abstention questions, the status says so. Separately, the run reports how many
answers passed the evidence checks in verify_answer.py, how many fixes that
took, and how many inferences were labeled.
Each group in "expect" lists acceptable ways to say one fact (for example a
value with or without rounding); all groups must match. "reject" fails the
answer if any phrase appears. Answers print in full so you can judge them.

Cost: each question sends Cole's full context, about $1 per question uncached
on Opus. Use --only to run a few.

Usage:
  python eval_cole.py                       # all questions, with tools
  python eval_cole.py --no-tools            # context only, same questions
  python eval_cole.py --only allergies meds_current
  python eval_cole.py --list                # show question ids without running
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from ask import ask, find_patient, format_trace, load_env, load_patients

QUESTIONS = [
    # ---------------------------------------------------------- single questions
    {"id": "allergies", "kind": "single", "tests": "absence: no allergies on record",
     "q": "Does he have any allergies?",
     "expect": [["no known allerg", "no allergies", "none recorded", "not recorded", "no recorded allerg"]],
     "expect_status": ["answered"],
     "reject": ["allergic to "]},
    {"id": "anticoag", "kind": "single", "tests": "current med linked to its reason",
     "q": "Is he anticoagulated? If so, with what and why?",
     "expect": [["warfarin"], ["atrial fibrillation", "afib", "a-fib"]]},
    {"id": "bp_latest", "kind": "single", "tests": "latest vital from a split blood pressure panel",
     "q": "What was his blood pressure at his most recent measurement?",
     "expect": [["107/88", "107 / 88", "systolic 107"], ["2026-08-05"]]},
    {"id": "dialysis_status", "kind": "single", "tests": "reasoning from dates: dialysis stopped appearing",
     "q": "Is he still on dialysis?",
     "expect": [["2026-03-24"]]},
    {"id": "tsh_never", "kind": "single", "tests": "absence: a test he never had",
     "q": "What was his most recent TSH?",
     "expect": [["not recorded", "no tsh", "no thyroid", "never", "no record", "none"]],
     "expect_status": ["not_recorded"],
     "reject": ["tsh was", "tsh of", "tsh: "]},
    {"id": "cardiac_history", "kind": "single", "tests": "history scattered across conditions",
     "q": "Does he have a history of heart attack or heart surgery?",
     "expect": [["2014-09-04", "2014"], ["bypass", "cabg"]]},
    {"id": "meds_current", "kind": "single", "tests": "current list vs stale 'active' orders from 2005",
     "q": "What medications is he currently taking?",
     "expect": [["warfarin"], ["digoxin"], ["verapamil"], ["metformin"], ["insulin", "humulin"],
                ["lisinopril"], ["hydrochlorothiazide", "hctz"]]},
    {"id": "chloride_mid", "kind": "single", "tests": "one value from the middle of a 174-reading line",
     "q": "What was his serum chloride on 2021-10-06?",
     "expect": [["104"]]},
    {"id": "chloride_no_reading", "kind": "single", "tests": "absence: a date with no reading",
     "q": "What was his serum chloride on 2021-10-07?",
     "expect": [["no reading", "no chloride", "not recorded", "no serum chloride", "no measurement",
                 "no value", "2021-10-06"]],
     "expect_status": ["not_recorded"],
     "reject": ["chloride on 2021-10-07 was", "2021-10-07 was 1", "2021-10-07: 1"]},
    # ------------------------------------------------------- multi-part questions
    {"id": "kidney", "kind": "multi", "tests": "two latest labs (one with a same-day conflict a week earlier) + stage",
     "q": "What are his latest eGFR and serum creatinine, and what stage is his kidney disease?",
     "expect": [["24.154", "24.15", "24.2"], ["2.3249", "2.32", "2.33"], ["stage 4"]]},
    {"id": "screening", "kind": "multi", "tests": "two dates from different sections",
     "q": "When was his last colonoscopy, and when did he last get a flu shot?",
     "expect": [["2019-10-11"], ["2026-04-15"]]},
    {"id": "dementia", "kind": "multi", "tests": "diagnosis + date + treatment linkage",
     "q": "Does he have dementia? If so, when was it diagnosed and is he being treated for it?",
     "expect": [["alzheimer"], ["2024-07-25"], ["galantamine"]]},
    {"id": "smoking_a1c", "kind": "multi", "tests": "social history + an implausible lab reported as recorded",
     "q": "Is he a smoker, and what was his most recent HbA1c?",
     "expect": [["ex-smoker", "former smoker", "quit"], ["2.43"], ["2026-08-12"]]},
    {"id": "alive_hospice", "kind": "multi", "tests": "deceased status + an old episode",
     "q": "Is he alive, and has he ever been in hospice?",
     "expect": [["alive", "no death", "not deceased", "living"], ["2019-10-16"]]},
]


def has(answer, phrase):
    """Numbers must match whole (104 does not match 104.55); text matches anywhere."""
    if re.fullmatch(r"[\d./ -]+", phrase):
        return re.search(rf"(?<![\d.]){re.escape(phrase)}(?!\.?\d)", answer) is not None
    return phrase in answer


# Details that appear nowhere in Cole's data (checked against the raw bundle,
# notes, and billing). Stating any of them as fact counts as an ungrounded
# inference. Labeled lines ("Interpretation (not in record): ...") are exempt.
UNGROUNDED = ["(pain)", "for pain", "pain)", "500 mg daily", "500 mg once daily", "hemodialysis", "haemodialysis",
              "antiplatelet"]


def recorded_part(answer):
    """The answer minus anything labeled as inferred, i.e. (inferred: ...) spans."""
    return re.sub(r"\(inferred:[^)]*\)", " ", answer, flags=re.I)


def ungrounded(answer):
    a = recorded_part(answer).lower()
    return [u for u in UNGROUNDED if u in a]


def statuses(trace):
    return [a.get("status") for a in (trace["result"] or {}).get("answers", []) if isinstance(a, dict)]


def check(item, answer, trace=None):
    """Correct = every expected fact present, nothing rejected, no unmarked
    ungrounded detail, and (for abstention questions) an abstaining status."""
    a = answer.lower().replace("\u2011", "-").replace("\u2010", "-")  # model sometimes uses non-ASCII hyphens
    if ungrounded(answer):
        return False
    if trace is not None and "expect_status" in item and not set(statuses(trace)) & set(item["expect_status"]):
        return False
    if any(has(a, r.lower()) for r in item.get("reject", [])):
        return False
    return all(any(has(a, alt.lower()) for alt in group) for group in item["expect"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--no-tools", action="store_true", help="context only, no run_sql or search_notes")
    ap.add_argument("--only", nargs="*", help="question ids to run")
    ap.add_argument("--list", action="store_true", help="list question ids and exit")
    args = ap.parse_args()
    if args.list:
        for q in QUESTIONS:
            print(f"{q['id']:<20} {q['kind']:<7} {q['q']}")
        return

    key = os.environ.get("OPENROUTER_API_KEY") or load_env().get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not found in environment or .env")
    out = Path(args.out)
    cole = find_patient("Cole Corwin", load_patients(Path("patients.csv"), out))[0]
    items = [q for q in QUESTIONS if not args.only or q["id"] in args.only]
    mode = "context only" if args.no_tools else "context + tools"

    results = []
    for item in items:
        t = ask(cole, item["q"], out, args.model, key, use_tools=not args.no_tools)
        answer = t["answer_text"]
        ok = check(item, answer, t)
        leaks = ungrounded(answer)
        n_inferred = sum(len(x.get("inferences") or []) for x in (t["result"] or {}).get("answers", [])
                         if isinstance(x, dict))
        results.append({**item, "pass": ok, "verified": t["verified"], "fixes": len(t["repairs"]),
                        "problems": t["problems"], "inferences": n_inferred, "leaks": leaks,
                        "answer": answer, "result": t["result"], "tool_calls": t["steps"],
                        "input_tokens": sum(u.get("prompt_tokens", 0) for u in t["usage"]),
                        "output_tokens": sum(u.get("completion_tokens", 0) for u in t["usage"]),
                        "finish_reason": t.get("finish_reason"), "seconds": t["seconds"]})
        print(f"\n{'=' * 70}\n[{'PASS' if ok else 'FAIL'}] {item['id']}  ({item['kind']}: {item['tests']})")
        print(f"Q: {item['q']}")
        print("  " + format_trace(t).replace("\n", "\n  "))
        print(f"  ({results[-1]['output_tokens']:,} output tokens, {t['seconds']}s, finish_reason: {t.get('finish_reason')})")
        if leaks:
            print(f"  UNGROUNDED DETAIL stated as fact (not marked inferred): {leaks}")

    n = len(results)
    print(f"\n{'=' * 70}")
    for kind in ("single", "multi"):
        sub = [r for r in results if r["kind"] == kind]
        if sub:
            print(f"{kind}: {sum(r['pass'] for r in sub)}/{len(sub)} correct")
    print(f"{mode}: {sum(r['pass'] for r in results)}/{n} correct, "
          f"{sum(r['verified'] for r in results)}/{n} passed the evidence checks "
          f"({sum(r['fixes'] for r in results)} fixes needed), "
          f"{sum(r['inferences'] for r in results)} labeled inferences, "
          f"{sum(bool(r['leaks']) for r in results)} unlabeled leaks")
    print(f"{sum(len(r['tool_calls']) for r in results)} tool calls, "
          f"{sum(r['input_tokens'] for r in results):,} input tokens total")
    rdir = out / "evals"
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"cole_{'context' if args.no_tools else 'tools'}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"mode": mode, "model": args.model, "results": results},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
