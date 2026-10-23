"""
A small web UI for the record-only pipeline in ask.py.

  python app.py            then open http://127.0.0.1:5000

Pick a patient, ask a question, and see:
  - each answer with its status and whether it passed the code checks,
  - the model's inferences in their own section, with the facts they rest on,
  - the evidence, which expands from the quote to the full record line, to the
    FHIR resource ID(s), to the original FHIR JSON for that resource.

Nothing here changes how answers are made: it calls ask.ask() and renders the
trace. Tables are drawn by code from the verbatim evidence quotes, never by the
model, so a table cannot misquote the record.
"""
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request
from markupsafe import Markup, escape

from ask import ask
from common import describe, find_patient, load_env, load_patients
from verify_answer import DATE, index_context

OUT = Path(os.environ.get("OUT_DIR", "out"))
MODEL = os.environ.get("MODEL", "anthropic/claude-opus-5")
PATIENTS = load_patients(Path("patients.csv"), OUT)
BY_ID = {p["patient_id"]: p for p in PATIENTS}
MAX_SOURCES = 25  # FHIR resources shown per evidence line

app = Flask(__name__)


# ------------------------------------------------------------ patient picker

def label(p):
    """What the autocomplete shows: 'Pat Owen Rutherford · male · 1975-10-22 · 57efda89'."""
    return describe(p)


def patient_from_form(text):
    """The picker sends the label; its last part is the start of the patient ID.
    Falls back to a name search if the text was typed by hand."""
    text = (text or "").strip()
    short_id = text.rsplit("·", 1)[-1].strip()
    exact = [p for p in PATIENTS if len(short_id) >= 6 and p["patient_id"].startswith(short_id)]
    if len(exact) == 1:
        return exact[0], None
    found = find_patient(text, PATIENTS)
    if len(found) == 1:
        return found[0], None
    if not found:
        return None, f'No patient matches "{text}".'
    return None, f'{len(found)} patients match "{text}". Pick one from the list.'


# ------------------------------------------------------------ tracing evidence

def source_rows(db_path, source_ids):
    """{fhir_id: (table, date, raw_json)} for the given FHIR IDs, from any table."""
    if not source_ids:
        return {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
              if "raw" in [c[1] for c in con.execute(f"PRAGMA table_info({r[0]})")]]
    found, wanted = {}, list(dict.fromkeys(source_ids))
    marks = ",".join("?" * len(wanted))
    for t in tables:
        date_col = "birth_date" if t == "patient" else "date"
        for fid, date, raw in con.execute(f"SELECT id, {date_col}, raw FROM {t} WHERE id IN ({marks})", wanted):
            found.setdefault(fid, (t, date, raw))
    con.close()
    return found


def trace_evidence(item, context_lines, ids_map, db_path):
    """One evidence quote, traced: full line, then the FHIR resources behind it.
    When the quote names dates, only resources from those dates are shown, so
    '2024-09-23 6.15' leads to that one observation, not all eight."""
    cid = str(item.get("id", "")).strip("[] ")
    quote = str(item.get("quote", ""))
    entry = ids_map.get(cid, {})
    rows = source_rows(db_path, entry.get("source_ids", []))
    dates = set(DATE.findall(quote))
    matched = {k: v for k, v in rows.items() if v[1] and v[1][:10] in dates} if dates else {}
    shown = matched or rows
    resources = [{"id": fid, "table": t, "date": (d or "")[:10],
                  "json": json.dumps(json.loads(raw), indent=2, ensure_ascii=False)}
                 for fid, (t, d, raw) in list(shown.items())[:MAX_SOURCES]]
    return {"id": cid, "quote": quote, "line": context_lines.get(cid, ""),
            "section": entry.get("section", ""), "table": readings_table(quote),
            "resources": resources, "n_resources": len(shown),
            "filtered": bool(matched) and len(matched) < len(rows), "n_all": len(rows)}


# ------------------------------------------------------------ rendering

READING = re.compile(r"(\d{4}-\d{2}-\d{2}) ([^·|]+?)\s*(?=·|\||$)")


def readings_table(quote):
    """'2016-12-05 88 · 2017-12-11 90 · ...' -> [(date, value), ...] when the quote
    holds two or more dated readings; otherwise None (shown as plain text)."""
    pairs = [(d, v.strip()) for d, v in READING.findall(quote)]
    return pairs if len(pairs) >= 2 else None


CITE = re.compile(r"\[((?:[A-Z]{1,2}\d+)(?:\]\[[A-Z]{1,2}\d+)*)\]")
INFERRED = re.compile(r"\(inferred:\s*([^)]*)\)", re.I)


def answer_html(text, n):
    """The answer with citation chips and highlighted (inferred: ...) spans."""
    html = str(escape(text))
    html = INFERRED.sub(lambda m: f'<span class="inf">inferred: {m.group(1)}</span>', html)
    html = CITE.sub(lambda m: "".join(
        f'<a class="chip" href="#ev-{n}-{c}" onclick="openEv(\'ev-{n}-{c}\')">{c}</a>'
        for c in m.group(1).split("][")), html)
    return Markup(html)


def view(trace, patient):
    """Turn a trace into what the page shows."""
    context = (OUT / "context" / f"{patient['stem']}.txt").read_text(encoding="utf-8")
    context_lines = index_context(context)
    ids_path = OUT / "context" / f"{patient['stem']}.ids.json"
    ids_map = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else {}
    db_path = OUT / "db" / f"{patient['stem']}.sqlite"
    answers = []
    for n, a in enumerate((trace["result"] or {}).get("answers", []), 1):
        if not isinstance(a, dict):
            continue
        answers.append({
            "n": n, "question": a.get("question", ""), "status": a.get("status", ""),
            "html": answer_html(a.get("answer", ""), n), "checked": a.get("checked"),
            "evidence": [trace_evidence(e, context_lines, ids_map, db_path)
                         for e in a.get("evidence") or [] if isinstance(e, dict)],
            "inferences": [{"claim": i.get("claim", ""), "assumption": i.get("assumption", ""),
                            "premises": [trace_evidence(p, context_lines, ids_map, db_path)
                                         for p in i.get("premises") or [] if isinstance(p, dict)]}
                           for i in a.get("inferences") or [] if isinstance(i, dict)]})
    u = trace["usage"]
    return {"answers": answers, "verified": trace["verified"], "problems": trace["problems"],
            "fixes": len(trace["repairs"]), "raw": trace.get("raw_answer", ""),
            "calls": len(u), "seconds": trace["seconds"],
            "cost": sum(x.get("cost") or 0 for x in u)}


# ------------------------------------------------------------ the page

@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"patients": [label(p) for p in PATIENTS], "picked": "", "query": "", "error": None,
           "patient": None, "result": None}
    if request.method == "POST":
        ctx["picked"] = request.form.get("patient", "")
        ctx["query"] = request.form.get("query", "").strip()
        patient, error = patient_from_form(ctx["picked"])
        key = os.environ.get("OPENROUTER_API_KEY") or load_env().get("OPENROUTER_API_KEY")
        if not key:
            error = "OPENROUTER_API_KEY not found in environment or .env"
        if error or not ctx["query"]:
            ctx["error"] = error or "Type a question."
            return render_template("index.html", **ctx)
        trace = ask(patient, ctx["query"], OUT, MODEL, key)
        tdir = OUT / "traces"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{datetime.now():%Y%m%d-%H%M%S}_{patient['stem'][:30]}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
        ctx.update(picked=label(patient), patient=label(patient), result=view(trace, patient))
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=False)
