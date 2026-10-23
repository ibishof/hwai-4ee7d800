"""
Plumbing shared by every pipeline: finding a patient, calling the model through
OpenRouter, prompt caching, and printing a trace. Nothing here decides what the
model sees or how an answer is checked; that is in ask.py (and tools.py).
"""
import csv
import json
import re
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

MAX_OUTPUT_TOKENS = 8000  # room for answers that show their counting
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NETWORK_RETRIES = 3  # attempts per model call when the connection drops
CACHE = {"type": "ephemeral"}  # Anthropic prompt-cache mark


# ----------------------------------------------------------- patient lookup

def plain(s):
    """Lowercase, accents removed, digits removed: 'Germán350' -> 'german'."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower())


def load_patients(csv_path, out_dir):
    """Rows of {patient_id, display_name, birth_date, sex, deceased_date, stem}.
    Uses patients.csv when present, otherwise the patient table of each database."""
    patients = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["stem"] = Path(r["file"]).stem
                patients.append(r)
        return patients
    for db in sorted((out_dir / "db").glob("*.sqlite")):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        pid, name, birth, sex, dead = con.execute(
            "SELECT id, name, birth_date, gender, deceased_date FROM patient").fetchone()
        con.close()
        patients.append({"patient_id": pid, "display_name": name, "birth_date": birth,
                         "sex": sex, "deceased_date": dead or "", "maiden_name": "", "stem": db.stem})
    return patients


def find_patient(query, patients):
    """Every word of the query must appear in the name (or maiden name),
    or the query must be the start of the patient ID."""
    q = query.strip().lower()
    by_id = [p for p in patients if p["patient_id"].lower().startswith(q)]
    if by_id and len(q) >= 6:
        return by_id
    words = plain(query).split()
    return [p for p in patients
            if words and all(w in plain(f"{p['display_name']} {p.get('maiden_name', '')}").split()
                             for w in words)]


def describe(p):
    dead = f" · died {p['deceased_date']}" if p.get("deceased_date") else ""
    return f"{p['display_name']} · {p.get('sex', '')} · {p['birth_date']}{dead} · {p['patient_id'][:8]}"


# ------------------------------------------------------------- model calls

def load_env(path=".env"):
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def call_model(messages, model, key, tools=None):
    """One chat call through OpenRouter. tools: tool definitions, or None for none.
    Retries dropped connections and 429/5xx errors."""
    payload = {"model": model, "messages": messages, "max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0,
               "usage": {"include": True}}  # OpenRouter then reports the dollar cost of each call
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(1, NETWORK_RETRIES + 1):
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < NETWORK_RETRIES:
                print(f"  (OpenRouter error {e.code}; retrying in {5 * attempt}s)", flush=True)
                time.sleep(5 * attempt)
                continue
            raise SystemExit(f"OpenRouter error {e.code}: {detail}")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
            # Dropped or garbled connection (for example "SSL: bad record mac").
            if attempt < NETWORK_RETRIES:
                print(f"  (connection problem: {e}; retrying in {5 * attempt}s)", flush=True)
                time.sleep(5 * attempt)
                continue
            raise SystemExit(f"Could not reach OpenRouter after {NETWORK_RETRIES} attempts: {e}")


def cache_latest(messages):
    """Mark the newest tool result or fix request as cacheable, and unmark older ones.
    Each call resends the whole conversation; with this, earlier turns are re-read
    from the cache instead of paid for again at full price. The record (message 1)
    keeps its own mark. Anthropic allows 4 marks per request; this uses at most 2."""
    for m in messages[2:]:
        if isinstance(m.get("content"), list):
            m["content"] = "".join(part.get("text", "") for part in m["content"])
    last = messages[-1]
    if len(messages) > 2 and last["role"] in ("tool", "user") and isinstance(last.get("content"), str):
        last["content"] = [{"type": "text", "text": last["content"], "cache_control": CACHE}]


# ------------------------------------------------------------------ traces

def new_trace(patient, question, model, use_tools, sql_first):
    return {"patient": patient["display_name"], "patient_id": patient["patient_id"],
            "question": question, "model": model, "use_tools": use_tools, "sql_first": sql_first,
            "steps": [], "usage": [], "repairs": [], "result": None, "verified": False, "problems": []}


def record_call(trace, resp):
    """Log one model response's usage and finish reason; return its message."""
    trace["usage"].append(resp.get("usage", {}))
    choice = resp["choices"][0]
    trace["finish_reason"] = choice.get("finish_reason")
    return choice["message"]


def finish_trace(trace, started):
    trace["seconds"] = round(time.monotonic() - started, 1)
    trace["answer_text"] = " ".join(a.get("answer", "") for a in (trace["result"] or {}).get("answers", [])
                                    if isinstance(a, dict))
    return trace


def format_trace(trace, full_results=False):
    """Readable rendering of a trace: answers, evidence, inferences, checks, tools."""
    out = []
    flag = "VERIFIED" if trace["verified"] else "NOT VERIFIED"
    fixes = f", after {len(trace['repairs'])} fix(es)" if trace["repairs"] else ""
    out.append(f"[{flag}{fixes}]")
    answers = (trace["result"] or {}).get("answers") or []
    for n, a in enumerate(answers, 1):
        if not isinstance(a, dict):
            continue
        out.append(f"\n{n}. {a.get('question', '')}  ({a.get('status', '')})")
        out.append(f"   {a.get('answer', '')}")
        for e in a.get("evidence") or []:
            out.append(f"   [{e.get('id')}] \"{e.get('quote')}\"")
        for inf in a.get("inferences") or []:
            out.append(f"   INFERRED: {inf.get('claim')}")
            for p_ in inf.get("premises") or []:
                out.append(f"     Recorded [{p_.get('id')}]: \"{p_.get('quote')}\"")
            out.append(f"     Assumption: {inf.get('assumption')}")
        if a.get("checked"):
            out.append(f"   Checked: {a['checked']}")
    if not answers:
        out.append(trace.get("raw_answer") or "(no answer)")
    if trace["problems"]:
        out.append("\nChecks still failing:")
        out += [f"  - {p_}" for p_ in trace["problems"]]
    if trace["steps"]:
        out.append(f"\n--- {len(trace['steps'])} tool call(s) ---")
        for s_ in trace["steps"]:
            out.append(f"[{s_['id']}] {s_['tool']}: {s_['input']}")
            shown = s_["result"] if full_results else s_["result"][:400]
            out.append("  " + shown.replace("\n", "\n  "))
            if not full_results and len(s_["result"]) > 400:
                out.append("  ... (shortened on screen; use --full-results, or see the trace file)")
    return "\n".join(out)
