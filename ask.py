"""
The main pipeline: answer a clinician's question from one patient's record.

  python ask.py "Merlene Thompson" "What was her most recent HbA1c, and when?"
  python ask.py 5dae1e9e "What is he allergic to?"        # short patient ID works too
  python ask.py "Cole" "Any visit notes about chest pain?" --tools

How it works:
  1. The name is matched against patients.csv. If several patients match, it
     lists them and stops. The patient comes from this lookup only, never from
     the question text.
  2. The patient's reduced record (out/context/<patient>.txt) and the question
     go to the model in one call. The record is cached, so later questions about
     the same patient are cheap.
  3. The model replies in JSON: per question a status, the answer, an exact
     quote for every fact cited by line ID ([M4]), and any judgment or outside
     knowledge marked "(inferred: ...)" with its recorded premises and the
     assumption behind it.
  4. verify_answer.py checks the JSON with plain code: quotes must exist in the
     lines they cite, values and dates in the answer must be quoted, and
     inferences must be labeled. On failure the problems go back to the model,
     up to 2 times. An answer that still fails is shown as NOT VERIFIED.
  5. The answer prints with its evidence (--json prints the raw JSON a UI would
     receive). A full trace is saved to out/traces/.

--tools also gives the model SQL and visit-note search, and --sql-first sends
only an index and makes every value come from SQL. Both live in tools.py; the
evals found the record alone as accurate, faster, and cheaper.

Needs OPENROUTER_API_KEY in .env (same folder) or the environment.
Standard library only.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from common import (MAX_OUTPUT_TOKENS, CACHE, cache_latest, call_model, describe, find_patient,
                    finish_trace, format_trace, load_env, load_patients, new_trace, record_call)
from verify_answer import index_context, parse_json, verify

MAX_REPAIRS = 2  # times the model may fix an answer that fails the checks


# ---------------------------------------------------------------- prompt

# The system prompt has six labeled parts: role, task, input context, success,
# failure, and output. tools.py reuses them and swaps in its own input context
# and failure lines.

ROLE = """# Role
You are a clinical record assistant for physicians and other clinicians who are treating this patient. You report what the record says, accurately and with dates, and you keep anything you conclude yourself separate and labeled. You do not recommend treatment."""

TASK = """# Task
Answer the clinician's questions about this one patient using only their medical record. A message may contain one question or several. Answer every question asked: find the relevant facts, check them, and give a direct answer to each. Always give an answer. Never decline, and never reply only that you cannot judge: if the record cannot settle a question, the answer is what the record does show and what is not recorded.

Every statement in an answer is one of two kinds:
- Fact: something the record states. Report it as recorded and cite it.
- Inference: everything else. Any judgment call and any outside knowledge is an inference, for example:
  - a verdict on a value or trend (worse, improving, stable, controlled, normal, high, implausible) unless the record itself says it;
  - a diagnosis or cause that is not listed as a condition or written in a note;
  - calling a medication or condition current, stopped, or resolved when its recorded status does not say so;
  - linking two items (this drug is for that condition) when the record does not link them;
  - general medical knowledge: a drug's class, what a test measures, how a drug is usually taken;
  - anything you calculated: a count, an average, a difference, a time span.
If a question asks for a judgment (Is it getting worse? Is it controlled? What is the likely diagnosis?), answer it, but as an inference: state the recorded facts, then give the judgment marked (inferred: ...), and cite the facts it rests on as that inference's premises. If you are unsure whether something is a fact or an inference, treat it as an inference."""

INPUT_COMMON = """# Input context
- The patient record is between <record> tags. It was built by code from the patient's health information exchange (HIE) records in FHIR format. Nothing was summarized or dropped except billing, audit, and duplicate data, so every clinical fact is there.
- The legend at the top of the record explains its format. Lists run oldest to newest; "latest" is the most recent reading.
- Every item line starts with an ID in brackets, such as [M4] or [L45]. Cite facts by these IDs.
- The data comes from a source system that often leaves conditions and medications "active" after they ended, and some values may look clinically implausible.
- Use only this record{tool_results}. Do not search the web. Outside knowledge may appear only as a labeled inference."""

INPUT_NO_TOOLS = """- You have no tools in this conversation. Answer from the record alone."""

SUCCESS = """# Define success
A successful answer:
- answers exactly what was asked, with values, dates, and units as recorded;
- uses the right readings for the question: the most recent date for "latest", the full series on that item's line for trends, and every value when several share a date;
- backs every stated fact with an exact quote, so a clinician can check it against the source;
- reports what the record shows about current versus past (dates, statuses), and points out when a status and its dates disagree;
- states as fact only what the record says, and gives every judgment and every piece of outside knowledge as a labeled inference whose premises cite the facts it rests on;
- uses the status that matches what was found. An honest "not recorded" is a success; a fluent guess is a failure."""

FAILURE_COMMON = """# Define failure
If you cannot fully answer:
- If the record does not contain the answer, say "Not recorded" and say what you checked. Give the nearest relevant facts if they help (for example, the closest date with a reading).
- If two different values share a date, report both. Do not pick one.
- If the question is ambiguous, answer the most likely reading and state the assumption in one line.
- Never invent a value, date, count, query, or tool result. A wrong answer is worse than "Not recorded".
- Never state a judgment or outside knowledge as fact, even when it seems obvious. An unlabeled judgment is a failure even if it is correct.
- When a detail is missing from the record (a medication's reason, dose, frequency, or route; a procedure's type; a cause), write that it is not recorded, for example "reason not recorded". Before saying a detail is not recorded, check the item's full name: names often state it (for example strength, form, and route in "Oral Tablet"). Add an inferred detail only if it would genuinely help the clinician, and label it."""

FAILURE_NO_TOOLS = """- Do not write SQL or tool calls, and do not pretend to run a query.
- For counts, count from the record. A count you add up yourself is an inference: mark it (inferred: ...) and say how you counted in the assumption."""

OUTPUT = """# Define the output
Reply with one JSON object and nothing else: no prose before or after it, no code fences.
{"answers": [{"question": "...", "status": "...", "answer": "...", "evidence": [...], "inferences": [...], "checked": null}]}

- answers: one entry per question asked, in the order asked.
- question: the question this entry answers, in a few words.
- status: judge it by what the question asks, not by related details it did not ask for.
  - "answered": everything the question asks is answered from the record or tool results. Missing details the question did not ask for (a dose, frequency, or reason) do not change this; mention them in the answer as "not recorded" if useful. Example: "What is her latest potassium?" answered with the value and date is "answered" even though the reason the test was ordered is not recorded.
  - "partial": the question asks for two or more things, and at least one is answered while at least one is not recorded or cannot be determined.
  - "not_recorded": nothing the question asks for is in the record or tool results, after checking.
  - "cannot_determine": the record has the information but it conflicts or is too unclear to give one answer. For example, two different values for the same test on the same latest date; report both.
  - If the record explicitly states an absence (a line that reads "none recorded"), the question is answered with that absence, cited by the line's ID. Use "not_recorded" only when the record has nothing on the topic at all.
  - A judgment question answered with the recorded facts plus a labeled inference is "answered".
  - When one message holds several questions, give each its own entry and judge each separately.
- answer: the direct answer for a clinician in plain sentences, no markdown lists or tables, under about 150 words. Every value and date in it must appear in one of its evidence quotes. Put the ID after the facts it supports, and write every judgment or detail not stated in the record as (inferred: <detail>) where it appears. Example answer: "Ferritin was 22 on 2019-02-01 and 61 on 2021-03-03 [L7] (inferred: rising)." Its inference entry: {"claim": "rising", "premises": [{"id": "L7", "quote": "2019-02-01 22 · 2021-03-03 61"}], "assumption": "A later value higher than an earlier one is a rise."}
- evidence: a list of {"id": "<line ID such as M4, or tool result ID such as SQL1>", "quote": "<text copied exactly from that line or result>"}. Copy each quote character for character as one continuous piece of the source: never shorten, splice, or join pieces with "...", and do not fix spelling or spacing. Code compares every quote exactly against its source, so it must be identical. Quote only the part that supports the answer, usually under 200 characters; for a long line of readings, quote just the date and value. When a tool result has several rows, quote from within one row and include its date, so it is clear which row it came from.
- inferences: one entry per (inferred: ...) marker, or [] if nothing is inferred. Each entry: "claim" (the same words as in the marker), "premises" (a list of {"id", "quote"} for the recorded facts the inference starts from, quoted exactly), and "assumption" (the general-knowledge step from the premises to the claim, in one sentence). Keep each claim short, one judgment per inference; put the reasoning in the assumption.
- checked: for "not_recorded", "partial", or "cannot_determine", what you looked at or searched for the part you could not answer (line IDs, queries). Otherwise null.

Code checks your JSON: every quote must appear exactly in the line or result it cites, every value and date in an answer must appear in that answer's evidence quotes, and every (inferred: ...) marker must have a matching inference. If a check fails, you will be told what to fix."""


def build_system():
    """The system prompt for the record-only pipeline."""
    return "\n\n".join([
        ROLE,
        TASK,
        INPUT_COMMON.format(tool_results="") + "\n" + INPUT_NO_TOOLS,
        SUCCESS,
        FAILURE_COMMON + "\n" + FAILURE_NO_TOOLS,
        OUTPUT,
    ])


# ---------------------------------------------------------------- checking

def check_reply(text, trace, messages, context_lines, tool_results, question):
    """Check the model's JSON reply. Returns True when finished (it passed, or
    it is out of fix attempts); otherwise adds a fix request to messages."""
    trace["raw_answer"] = text
    if trace["finish_reason"] == "length":
        problems = [f"The reply was cut off at the {MAX_OUTPUT_TOKENS:,}-token limit. Shorten the answers and quotes."]
    else:
        try:
            obj = parse_json(text)
            problems = verify(obj, context_lines, tool_results, question)
            trace["result"] = obj
        except ValueError as e:
            problems = [str(e)]
    trace["problems"] = problems
    if not problems:
        trace["verified"] = True
        return True
    if len(trace["repairs"]) >= MAX_REPAIRS:
        return True
    trace["repairs"].append(problems)
    messages.append({"role": "assistant", "content": text})
    messages.append({"role": "user", "content": "Your JSON failed these checks:\n- " + "\n- ".join(problems)
                     + "\nReturn the full corrected JSON only."})
    return False


# ---------------------------------------------------------------- the pipeline

def ask(patient, question, out_dir, model, key, use_tools=False, sql_first=False):
    """Answer one message (which may hold several questions) about one patient
    from the record alone. Returns a trace: the answer, whether it passed the
    checks, any problems left, fixes needed, and usage.
    use_tools / sql_first hand the question to tools.py instead."""
    if use_tools or sql_first:
        from tools import ask_with_tools  # loaded only when tools are used
        return ask_with_tools(patient, question, out_dir, model, key, sql_first=sql_first)

    context = (out_dir / "context" / f"{patient['stem']}.txt").read_text(encoding="utf-8")
    context_lines = index_context(context)  # {'M4': 'the full line', ...} for checking quotes
    messages = [
        {"role": "system", "content": build_system()},
        {"role": "user", "content": [
            # cached, so later questions about the same patient reuse the record cheaply
            {"type": "text", "text": f"<record>\n{context}</record>", "cache_control": CACHE},
            {"type": "text", "text": f"Question: {question}"}]},
    ]
    trace = new_trace(patient, question, model, use_tools=False, sql_first=False)
    started = time.monotonic()
    while True:
        cache_latest(messages)
        msg = record_call(trace, call_model(messages, model, key))
        if check_reply(msg.get("content") or "", trace, messages, context_lines, {}, question):
            break
    return finish_trace(trace, started)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("patient", help="name (any words of it) or the start of the patient ID")
    ap.add_argument("question")
    ap.add_argument("--out", default="out")
    ap.add_argument("--patients-csv", default="patients.csv")
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--tools", action="store_true", help="also give the model SQL and visit-note search (tools.py)")
    ap.add_argument("--sql-first", action="store_true",
                    help="send only the index (names and codes, no values); every value comes from SQL (tools.py)")
    ap.add_argument("--full-results", action="store_true", help="print tool results in full")
    ap.add_argument("--json", action="store_true", help="also print the raw JSON answer (what a UI would receive)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    matches = find_patient(args.patient, load_patients(Path(args.patients_csv), out_dir))
    if not matches:
        raise SystemExit(f'No patient matches "{args.patient}".')
    if len(matches) > 1:
        print(f'{len(matches)} patients match "{args.patient}". Be more specific or use the ID:')
        for p in matches:
            print(f"  {describe(p)}")
        raise SystemExit(1)
    patient = matches[0]

    key = os.environ.get("OPENROUTER_API_KEY") or load_env().get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not found in environment or .env")

    print(f"Patient: {describe(patient)}")
    print(f"Question: {args.question}\n")
    trace = ask(patient, args.question, out_dir, args.model, key, use_tools=args.tools, sql_first=args.sql_first)
    print(format_trace(trace, args.full_results))
    if args.json and trace["result"] is not None:
        print("\n" + json.dumps(trace["result"], ensure_ascii=False, indent=1))
    tokens_in = sum(u.get("prompt_tokens", 0) for u in trace["usage"])
    tokens_out = sum(u.get("completion_tokens", 0) for u in trace["usage"])
    cached = sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for u in trace["usage"])
    cost = sum(u.get("cost") or 0 for u in trace["usage"])
    print(f"\n{len(trace['usage'])} model call(s), {tokens_in:,} input tokens ({cached:,} from cache), "
          f"{tokens_out:,} output tokens, ${cost:.3f}, {trace['seconds']}s")

    tdir = out_dir / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    path = tdir / f"{datetime.now():%Y%m%d-%H%M%S}_{patient['stem'][:30]}.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"trace: {path}")


if __name__ == "__main__":
    main()
