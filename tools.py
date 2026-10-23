"""
The two alternative pipelines the evals compared against the record-only one
in ask.py:

  record + tools   the full record, plus run_sql and search_notes the model may use
  SQL-first        only an index of item names and codes; every value from run_sql

Both reuse ask.py's prompt sections and answer checks, swapping in their own
input-context and failure lines.

Safety on run_sql: the connection is read-only, opens only this patient's
file, cannot ATTACH another database, allows only reads (an authorizer
rejects everything else), times out after 5 seconds, and returns at most
200 rows.
"""
import json
import re
import sqlite3
import time

from ask import (FAILURE_COMMON, INPUT_COMMON, OUTPUT, ROLE, SUCCESS, TASK, check_reply)
from common import CACHE, cache_latest, call_model, finish_trace, new_trace, record_call
from verify_answer import index_context

MAX_ROUNDS = 5
MAX_ROUNDS_SQL_FIRST = 8  # SQL-first gets every value by query, so it needs more rounds
MAX_ROWS = 200
SQL_TIMEOUT_S = 5


# ------------------------------------------------------------ the SQL tool

# SELECT, reading a column, calling a function, and recursive CTEs (WITH RECURSIVE).
# Recursive queries are allowed because the timeout below stops runaway ones.
READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


def authorizer(action, arg1, arg2, db_name, trigger):
    # Allow plain reads only. Note: no SQLITE_ATTACH, PRAGMA, writes, or DDL.
    return sqlite3.SQLITE_OK if action in READ_ACTIONS else sqlite3.SQLITE_DENY


def open_patient_db(db_path, locked=True):
    """Read-only connection to one patient's file that cannot ATTACH another.
    locked=True also adds the read-only authorizer; that is the connection the
    model's SQL runs on. search_notes and the schema reader use an unlocked one
    because they run only fixed queries written here, never model-written SQL."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)  # cannot open any other database file
    if locked:
        con.set_authorizer(authorizer)
    return con


def run_sql(con, query, params=()):
    """Run one read-only query; return a text table or an error message."""
    deadline = time.monotonic() + SQL_TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(query, params)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        if "interrupted" in str(e):
            return f"SQL error: query stopped after {SQL_TIMEOUT_S} seconds; simplify it."
        return f"SQL error: {e}"
    finally:
        con.set_progress_handler(None, 0)
    if not rows:
        return "0 rows. (The query found nothing. That alone does not prove the item is absent from the record.)"
    more = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    lines = [" | ".join(cols)] + [" | ".join("" if v is None else str(v) for v in r) for r in rows]
    if more:
        lines.append(f"... more than {MAX_ROWS} rows; narrow the query or aggregate.")
    else:
        lines.append(f"({len(rows)} rows)")
    return "\n".join(lines)


def search_notes(con, text):
    """Keyword search over visit notes, best matches first, with a one-line excerpt each.
    The search words are passed as a parameter, never pasted into the SQL."""
    words = re.findall(r"\w+", text)
    if not words:
        return "No search words given."
    match = " OR ".join(f'"{w}"' for w in words)  # any word; best matches still surface first per date
    return run_sql(con, (
        "SELECT date, replace(snippet(notes_fts, 0, '', '', '...', 32), char(10), ' ') AS excerpt "
        "FROM notes_fts WHERE notes_fts MATCH ? ORDER BY rank LIMIT 20"), (match,))


def schema_text(con):
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
        "AND name NOT LIKE 'notes_fts%' AND name NOT LIKE 'sqlite_%' ORDER BY type DESC, name")]
    lines = []
    for t in tables:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        lines.append(f"{t}({', '.join(cols)})")
    return "\n".join(lines)


TOOLS = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run one read-only SQLite SELECT query on this patient's database and get the rows back (max 200).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_notes",
        "description": "Keyword search over this patient's visit notes. Returns up to 20 dated excerpts, best matches first.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Words to search for, e.g. 'shortness of breath'."}},
            "required": ["query"]}}},
]


# ---------------------------------------------------------------- prompt parts

INPUT_TOOLS = """- You have two tools, both limited to this patient:
  - run_sql: read-only SQL on the patient's database. Use it for exact counts, averages, date filters, or picking one value out of a long list. Most other questions can be answered from the record directly.
  - search_notes: keyword search over visit note text, which is not in the record. Use it for anything that might only be in notes.
  - Each tool result comes with its own ID (SQL1, SQL2, NOTES1, ...). Cite and quote it like a record line.
- Database (SQLite). Dates are TEXT 'YYYY-MM-DD'. Every table keeps the original FHIR resource in a `raw` JSON column (use json_extract for fields without their own column). The `facts` view is the easiest place to start: one row per dated fact with kind, code, display, date, value_num, value_text, unit, status. Multi-part observations (blood pressure, surveys) have one row per part, linked by id, with parent_code naming the panel. notes_fts cannot be queried through run_sql; use search_notes.
{schema}"""

INPUT_INDEX = """# Input context
- Between <index> tags is an index of the patient's record, built by code from the patient's health information exchange (HIE) records in FHIR format. It lists every item in the patient's database by name and code, grouped by section. Every item line starts with an ID in brackets, such as [M4] or [L45].
- The index holds no values, dates, statuses, or counts. Get all of those from the database with run_sql, and visit note text with search_notes. Cite an index line only for an item's name or code; cite every value, date, status, and count from the tool result it came from.
- The data comes from a source system that often leaves conditions and medications "active" after they ended, and some values may look clinically implausible.
- Use only the index and your tool results. Do not search the web. Outside knowledge may appear only as a labeled inference."""

FAILURE_TOOLS = """- A query that returns 0 rows only means that query found nothing. Before concluding something is absent, check the record and try a broader query (LIKE '%term%' on display, or the code). If still nothing, say what you searched.
- If a query errors, fix it and retry once; if it still fails, answer from the record and say the query failed."""


def build_system_tools(schema, sql_first=False):
    """The system prompt with tools: ask.py's sections with tool lines swapped in."""
    if sql_first:  # index only: every value comes from a query
        tools = INPUT_TOOLS.format(schema=schema).replace(
            "Use it for exact counts, averages, date filters, or picking one value out of a long list. "
            "Most other questions can be answered from the record directly.",
            "Use it for every value, date, status, and count.")
        return "\n\n".join([
            ROLE, TASK, INPUT_INDEX + "\n" + tools,
            SUCCESS.replace("the full series on that item's line for trends", "the full series of readings for trends"),
            FAILURE_COMMON + "\n" + FAILURE_TOOLS.replace("check the record and try", "check the index and try"),
            OUTPUT.replace("in the line or result it cites", "in the index line or tool result it cites")])
    return "\n\n".join([
        ROLE,
        TASK,
        INPUT_COMMON.format(tool_results=" (plus the results of your tool calls)") + "\n" + INPUT_TOOLS.format(schema=schema),
        SUCCESS,
        FAILURE_COMMON + "\n" + FAILURE_TOOLS,
        OUTPUT,
    ])


LINE_START = re.compile(r"^\[[A-Z]{1,2}\d+\] ")


def make_index(context):
    """The record with every value removed: section headings plus each item's
    name and code, under the same line IDs as the full record.
    '[L12] Hemoglobin A1c/Hemoglobin.total in Blood (LOINC 4548-4) % | n=8 | latest 6.31 ...'
    becomes '[L12] Hemoglobin A1c/Hemoglobin.total in Blood (LOINC 4548-4) %'."""
    out = []
    for line in context.splitlines():
        if line.startswith("# Patient record:"):
            out.append(line.replace("# Patient record:", "# Patient index:"))
        elif line.startswith("## "):
            out += ["", line]
        elif line.startswith("[N1] "):
            out.append("[N1] visit notes (search them with search_notes)")
        elif LINE_START.match(line):
            out.append(line.split(" | ")[0])
    return "\n".join(out).strip() + "\n"


# ---------------------------------------------------------------- the loop

def ask_with_tools(patient, question, out_dir, model, key, sql_first=False):
    """Like ask.ask(), but the model may call run_sql and search_notes (at most
    MAX_ROUNDS rounds) before answering. sql_first=True sends only the index."""
    context = (out_dir / "context" / f"{patient['stem']}.txt").read_text(encoding="utf-8")
    if sql_first:
        context = make_index(context)
    context_lines = index_context(context)
    db_path = out_dir / "db" / f"{patient['stem']}.sqlite"
    con = open_patient_db(db_path)                      # model-written SQL runs here
    notes_con = open_patient_db(db_path, locked=False)  # fixed queries only
    tag = "index" if sql_first else "record"
    max_rounds = MAX_ROUNDS_SQL_FIRST if sql_first else MAX_ROUNDS
    messages = [
        {"role": "system", "content": build_system_tools(schema_text(notes_con), sql_first)},
        {"role": "user", "content": [
            {"type": "text", "text": f"<{tag}>\n{context}</{tag}>", "cache_control": CACHE},
            {"type": "text", "text": f"Question: {question}"}]},
    ]
    trace = new_trace(patient, question, model, use_tools=True, sql_first=sql_first)
    tool_results, counts = {}, {"run_sql": 0, "search_notes": 0}
    started = time.monotonic()
    rounds = 0
    while True:
        cache_latest(messages)  # earlier query results are re-read from the cache
        msg = record_call(trace, call_model(messages, model, key, tools=TOOLS))
        calls = msg.get("tool_calls") or []
        if calls:
            rounds += 1
            if rounds > max_rounds:
                trace["problems"] = [f"Stopped after {max_rounds} tool rounds without a final answer."]
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                name = c["function"]["name"]
                arg = json.loads(c["function"].get("arguments") or "{}").get("query", "")
                counts[name] = counts.get(name, 0) + 1
                rid = f"{'SQL' if name == 'run_sql' else 'NOTES'}{counts[name]}"
                result = run_sql(con, arg) if name == "run_sql" else search_notes(notes_con, arg)
                tool_results[rid] = result
                trace["steps"].append({"id": rid, "tool": name, "input": arg, "result": result})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": f"Result ID: {rid}\n{result}"})
            continue
        if check_reply(msg.get("content") or "", trace, messages, context_lines, tool_results, question):
            break
    con.close()
    notes_con.close()
    return finish_trace(trace, started)
