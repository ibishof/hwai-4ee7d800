"""
Deterministic checks on the model's JSON answer. No LLM involved.

The model must reply with:
  {"answers": [{
      "question":   "...",
      "status":     "answered" | "partial" | "not_recorded" | "cannot_determine",
      "answer":     "plain text; inferred details written as (inferred: ...)",
      "evidence":   [{"id": "M4", "quote": "exact text from line M4"}],
      "inferences": [{"claim": "once daily",
                      "premises": [{"id": "M4", "quote": "exact text from line M4"}],
                      "assumption": "the general-knowledge step, in one sentence"}],
      "checked":    "what was searched (required for not_recorded / partial), else null"}]}

verify() returns a list of plain-English problems. Empty list = passed.
The problems are written so they can be sent straight back to the model to fix.

Checks:
  1. The JSON has the required shape and allowed status values.
  2. Every cited ID exists (a context line like M4, or a tool result like SQL1).
  3. Every quote appears exactly in what it cites (ignoring case, spacing,
     and dash style).
  4. Every date and number in the answer, outside (inferred: ...), appears in
     one of that answer's quotes (evidence or inference premises) or in the
     question itself. A number the model calculated (say 60 + 58 = 118) is in
     no quote, so it must be marked (inferred: ...).
     Limit: words cannot be checked this way. An unmarked inferred word such as
     "daily" passes; the prompt rule and the eval's leak list cover those.
  5. Every (inferred: ...) marker has a matching inference with premises and an
     assumption, and every inference has a marker.
  6. "answered" and "partial" need evidence; "not_recorded" and "partial" need "checked".
"""
import json
import re
import unicodedata

STATUSES = {"answered", "partial", "not_recorded", "cannot_determine"}
LINE_ID = re.compile(r"^\[([A-Z]{1,2}\d+)\] ")
INFERRED = re.compile(r"\(inferred:\s*([^)]*)\)", re.I)
CITATION = re.compile(r"\[?\b(?:[A-Z]{1,2}\d+|SQL\d+|NOTES\d+)\b\]?")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def norm(s):
    """Lowercase, plain dashes, single spaces: how quotes are compared."""
    s = unicodedata.normalize("NFKC", str(s or "")).translate(DASHES)
    return re.sub(r"\s+", " ", s).strip().lower()


def index_context(context):
    """{'M4': 'the full line text', ...} for every ID'd line in the context."""
    out = {}
    for line in context.splitlines():
        m = LINE_ID.match(line)
        if m:
            out[m.group(1)] = line
    return out


def parse_json(text):
    """The answer object, or raise ValueError. Tolerates code fences or stray prose."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found. Reply with one JSON object only.")
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def number_in(num, text):
    """Whole-number match: '104' is in '2021-10-06 104' but not in '104.55'."""
    return re.search(rf"(?<![\d.]){re.escape(num)}(?!\.?\d)", text) is not None


def check_quote(where, item, sources):
    """Problems with one {"id", "quote"} pair."""
    if not isinstance(item, dict):
        return [f"{where}: must be an object with 'id' and 'quote'."]
    cid, quote = str(item.get("id", "")).strip("[] "), item.get("quote")
    if cid not in sources:
        return [f"{where}: ID '{cid}' does not exist. Use a line ID from the record (like M4) or a tool result ID (like SQL1)."]
    if not quote or len(norm(quote)) < 2:
        return [f"{where}: quote for {cid} is empty."]
    if norm(quote) not in norm(sources[cid]):
        return [f"{where}: the quote is not in {cid}. Copy it exactly from {cid}: \"{str(quote)[:120]}\""]
    return []


def verify(obj, context_lines, tool_results, question=""):
    sources = {**context_lines, **tool_results}
    problems = []
    if not isinstance(obj, dict) or not isinstance(obj.get("answers"), list) or not obj["answers"]:
        return ['The reply must be {"answers": [ ... ]} with at least one answer.']

    for i, a in enumerate(obj["answers"]):
        w = f"answers[{i}]"
        if not isinstance(a, dict):
            problems.append(f"{w}: must be an object.")
            continue
        shape = [f"{w}: '{key}' is missing or not a {kind.__name__}."
                 for key, kind in (("question", str), ("status", str), ("answer", str),
                                   ("evidence", list), ("inferences", list))
                 if not isinstance(a.get(key), kind)]
        if shape:
            problems += shape
            continue
        status, answer = a["status"], a["answer"]
        if status not in STATUSES:
            problems.append(f"{w}: status must be one of {sorted(STATUSES)}.")
        if status in ("answered", "partial") and not a["evidence"]:
            problems.append(f"{w}: status '{status}' needs at least one evidence quote.")
        if status in ("not_recorded", "partial") and not str(a.get("checked") or "").strip():
            problems.append(f"{w}: status '{status}' needs 'checked': what you looked at or searched.")

        # 2-3. evidence quotes
        for j, ev in enumerate(a["evidence"]):
            problems += check_quote(f"{w}.evidence[{j}]", ev, sources)

        # 5. inference markers <-> inference entries
        markers = [norm(m) for m in INFERRED.findall(answer)]
        claims = []
        for j, inf in enumerate(a["inferences"]):
            iw = f"{w}.inferences[{j}]"
            if not isinstance(inf, dict):
                problems.append(f"{iw}: must be an object with 'claim', 'premises', 'assumption'.")
                continue
            claims.append(norm(inf.get("claim")))
            if not str(inf.get("assumption") or "").strip():
                problems.append(f"{iw}: 'assumption' is empty. State the general-knowledge step in one sentence.")
            premises = inf.get("premises")
            if not isinstance(premises, list) or not premises:
                problems.append(f"{iw}: needs at least one premise quoted from the record.")
                continue
            for k, p in enumerate(premises):
                problems += check_quote(f"{iw}.premises[{k}]", p, sources)
        for m in markers:
            if m not in claims:
                problems.append(f"{w}: '(inferred: {m})' has no inference entry with that exact claim.")
        for c in claims:
            if c and c not in markers:
                problems.append(f"{w}: inference claim '{c}' is not marked in the answer as (inferred: {c}).")

        # 4. dates and numbers in the answer must be in the evidence
        recorded = CITATION.sub(" ", INFERRED.sub(" ", answer))
        # Verified quotes in evidence and in inference premises both count as grounding.
        cited = list(a["evidence"]) + [p for inf in a["inferences"] if isinstance(inf, dict)
                                       for p in (inf.get("premises") or [])]
        quoted = " ".join(norm(e.get("quote")) for e in cited if isinstance(e, dict))
        q = norm(question) + " " + norm(a["question"])
        dates = DATE.findall(recorded)
        for d in dates:
            if d not in quoted and d not in q:
                problems.append(f"{w}: the date {d} in the answer is not in any evidence quote. "
                                "Quote the source that contains it, or mark it (inferred: ...).")
        for num in NUMBER.findall(DATE.sub(" ", recorded)):
            if not number_in(num, quoted) and not number_in(num, q):
                problems.append(f"{w}: the number {num} in the answer is not in any evidence quote. "
                                "Quote the source that contains it, or, if you calculated or assumed it, "
                                "mark it (inferred: ...) and explain it under inferences.")
    return problems
