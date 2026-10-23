"""The reduction loses nothing it should keep.

  reconciliation  every resource in every bundle is a database row, a logged
                  removal, or a Medication folded into its order
  round trip      every lab/vital reading survives the run compression exactly
  line IDs        every [ID] in a context has a source map entry, and every
                  FHIR ID in that map exists in the patient's database
"""
import json
import re
import sqlite3

import pytest

from conftest import DATA, OUT, need

FOLDED = {"Medication"}  # parse_fhir.FOLDED_TYPES: named inside the order that points to it


def stems():
    return sorted(p.stem for p in need(OUT / "db").glob("*.sqlite"))


def db_ids(stem):
    con = sqlite3.connect(f"file:{OUT / 'db' / (stem + '.sqlite')}?mode=ro", uri=True)
    ids = set()
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        if "id" in [c[1] for c in con.execute(f"PRAGMA table_info({t})")]:
            ids |= {r[0] for r in con.execute(f"SELECT id FROM {t}")}
    con.close()
    return ids


def test_reconciliation():
    need(DATA)
    problems = []
    for stem in stems():
        bundle = DATA / f"{stem}.json"
        if not bundle.exists():
            continue
        resources = [e["resource"] for e in json.loads(bundle.read_text(encoding="utf-8"))["entry"]]
        removed = OUT / "removed" / f"{stem}.jsonl"
        logged = {json.loads(line)["resource"].get("id") for line in removed.open(encoding="utf-8")} if removed.exists() else set()
        kept = db_ids(stem)
        lost = [r["id"] for r in resources
                if r["id"] not in kept and r["id"] not in logged and r["resourceType"] not in FOLDED]
        if lost:
            problems.append(f"{stem}: {len(lost)} resources neither kept nor logged, e.g. {lost[:2]}")
    assert not problems, "\n".join(problems)


def test_round_trip():
    from render_context import check_round_trip
    failures = []
    for f in sorted(need(OUT / "grouped").glob("*.json")):
        problems = check_round_trip(json.loads(f.read_text(encoding="utf-8")), collapse_runs=True)
        if problems:
            failures.append(f"{f.stem}: {problems[0]}")
    assert not failures, "\n".join(failures)


LINE_ID = re.compile(r"^\[([A-Z]{1,2}\d+)\] ", re.M)


@pytest.mark.parametrize("stem", stems() if (OUT / "db").exists() else [])
def test_line_ids_trace_to_source(stem):
    ctx = OUT / "context" / f"{stem}.txt"
    ids_path = OUT / "context" / f"{stem}.ids.json"
    if not ctx.exists() or not ids_path.exists():
        pytest.skip("context not built")
    line_ids = set(LINE_ID.findall(ctx.read_text(encoding="utf-8")))
    id_map = json.loads(ids_path.read_text(encoding="utf-8"))
    assert line_ids == set(id_map), f"IDs without a map entry: {sorted(line_ids - set(id_map))[:5]}"
    missing = {s for e in id_map.values() for s in e["source_ids"]} - db_ids(stem)
    assert not missing, f"{len(missing)} mapped FHIR IDs not in the database, e.g. {sorted(missing)[:2]}"
