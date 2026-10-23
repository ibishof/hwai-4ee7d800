"""
Step 3 of the pipeline: per-patient SQLite -> one grouped record per patient.

Every row-per-fact table is pivoted so each item appears once with its whole
history attached: all HbA1c readings together, all metformin orders together,
each condition with every episode's dates. Nothing is dropped; this is a
lossless regrouping of the database, not a summary.

Items are grouped by code (LOINC, SNOMED, RxNorm, CVX), not by name, so name
variants like "Urea Nitrogen" vs "Urea nitrogen" merge. Labs are also split by
unit, so readings in different units never share one range.

Outputs, per patient:
  out/grouped/<patient>.json   the grouped record (source for any context renderer)
  out/grouped/<patient>.txt    a readable rendering of it, every rule off (for inspection)
  out/grouped/_sizes.csv       characters and estimated tokens per patient

Usage:
  python group_record.py                  # every database in out/db
  python group_record.py --only Merlene Cole
"""
import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from render_context import fmt, render


def rows(con, sql):
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(sql)]


def ids(rows_):
    """FHIR resource IDs behind a grouped item, in order, without repeats."""
    return list(dict.fromkeys(r["id"] for r in rows_))


def group_by(items, key):
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups.values():
        g.sort(key=lambda r: (r.get("date") or "", r.get("datetime") or ""))
    return groups


def latest_display(group):
    """Name variants share a code; use the most recent spelling."""
    return next((r["display"] for r in reversed(group) if r.get("display")), None)


# ------------------------------------------------------------ per section

def g_conditions(con):
    out = []
    for (system, code), g in group_by(rows(con, "SELECT * FROM conditions"),
                                      lambda r: (r["system"], r["code"])).items():
        out.append({
            "code": code, "system": system, "display": latest_display(g),
            "episodes": [{"onset": r["onset_date"], "abatement": r["abatement_date"],
                          "status": r["status"]} for r in g],
            "latest_status": g[-1]["status"], "source_ids": ids(g)})
    return sorted(out, key=lambda c: c["episodes"][0]["onset"] or "")


def g_medications(con):
    admins = group_by(rows(con, "SELECT * FROM med_administrations"), lambda r: r["code"])
    out = []
    for code, g in group_by(rows(con, "SELECT * FROM med_requests"), lambda r: r["code"]).items():
        a = admins.pop(code, [])
        out.append({
            "code": code, "system": g[0]["system"], "display": latest_display(g),
            "orders": [{"date": r["date"], "status": r["status"]} for r in g],
            "latest_status": g[-1]["status"],
            "reasons": sorted({r["reason"] for r in g if r["reason"]}),
            "dosage": next((r["dosage"] for r in reversed(g) if r["dosage"]), None),
            "given_in_clinic": [r["date"] for r in a], "source_ids": ids(g) + ids(a)})
    for code, a in admins.items():  # given in clinic with no matching order
        out.append({"code": code, "system": a[0]["system"], "display": latest_display(a),
                    "orders": [], "latest_status": None,
                    "reasons": sorted({r["reason"] for r in a if r["reason"]}),
                    "dosage": None, "given_in_clinic": [r["date"] for r in a], "source_ids": ids(a)})
    return sorted(out, key=lambda m: (m["orders"] or [{"date": ""}])[-1]["date"] or "", reverse=True)


def g_observations(con):
    out = []
    obs = rows(con, "SELECT * FROM observations")
    for (code, unit), g in group_by(obs, lambda r: (r["code"], r["unit"])).items():
        nums = [r["value_num"] for r in g if r["value_num"] is not None]
        item = {
            "code": code, "system": g[0]["system"], "display": latest_display(g), "unit": unit,
            "category": g[-1]["category"], "part_of": g[-1]["parent_display"],
            "readings": [[r["date"], r["value_num"] if r["value_num"] is not None else r["value_text"]]
                         for r in g],
            "source_ids": ids(g)}
        if nums:
            item.update(min=min(nums), max=max(nums))
        out.append(item)
    return sorted(out, key=lambda o: (o["category"] or "", o["display"] or ""))


def g_simple(con, table, extra=()):
    """Group a table by code with all its dates (procedures, immunizations, imaging...)."""
    out = []
    for code, g in group_by(rows(con, f"SELECT * FROM {table}"), lambda r: r["code"]).items():
        item = {"code": code, "system": g[0]["system"], "display": latest_display(g),
                "dates": [r["date"] for r in g], "latest_status": g[-1]["status"], "source_ids": ids(g)}
        for col in extra:
            vals = sorted({str(r[col]) for r in g if r.get(col) is not None})
            if vals:
                item[col] = vals
        out.append(item)
    return sorted(out, key=lambda x: (x["dates"][-1] or "") if x["dates"] else "", reverse=True)


def g_encounters(con):
    out = []
    enc = rows(con, "SELECT * FROM encounters")
    for (cls, code), g in group_by(enc, lambda r: (r["class"], r["code"])).items():
        out.append({"class": cls, "code": code, "display": latest_display(g),
                    "dates": [r["date"] for r in g],
                    "reasons": sorted({r["reason"] for r in g if r["reason"]}), "source_ids": ids(g)})
    return sorted(out, key=lambda e: len(e["dates"]), reverse=True)


def build(db_path):
    con = sqlite3.connect(db_path)
    patient = rows(con, "SELECT * FROM patient")[0]
    patient.pop("raw")
    notes = rows(con, "SELECT count(*) n, min(date) first, max(date) last FROM notes")[0]
    record = {
        "patient": patient,
        "allergies": [{k: r[k] for k in ("id", "display", "code", "type", "category", "criticality",
                                          "reactions", "status", "date")}
                      for r in rows(con, "SELECT * FROM allergies ORDER BY date")],
        "conditions": g_conditions(con),
        "medications": g_medications(con),
        "careplans": [{k: r[k] for k in ("id", "display", "code", "status", "date", "end_date",
                                          "addresses", "activities")}
                      for r in rows(con, "SELECT * FROM careplans ORDER BY date")],
        "observations": g_observations(con),
        "procedures": g_simple(con, "procedures", extra=("reason",)),
        "immunizations": g_simple(con, "immunizations"),
        "imaging": g_simple(con, "imaging_studies", extra=("modality", "body_site")),
        "devices": g_simple(con, "devices"),
        "supplies": g_simple(con, "supply_deliveries", extra=("quantity",)),
        "panels": g_simple(con, "diagnostic_reports"),
        "careteams": [{k: r[k] for k in ("id", "display", "status", "date", "end_date", "members")}
                      for r in rows(con, "SELECT * FROM careteams ORDER BY date")],
        "encounters": g_encounters(con),
        "notes": notes,
    }
    con.close()
    return record


# ---------------------------------------------------------- text rendering
# The readable .txt written here is for inspecting the grouping. It uses the
# same renderer as the context (render_context.py) with every rule turned off.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--only", nargs="*", help="name prefixes, e.g. Merlene Cole")
    args = ap.parse_args()

    out = Path(args.out)
    gdir = out / "grouped"
    gdir.mkdir(parents=True, exist_ok=True)
    dbs = sorted((out / "db").glob("*.sqlite"))
    if args.only:
        dbs = [d for d in dbs if d.name.startswith(tuple(args.only))]

    sizes = []
    for db in dbs:
        record = build(db)
        (gdir / f"{db.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        text = render(record, collapse_runs=False, guide=False)
        (gdir / f"{db.stem}.txt").write_text(text, encoding="utf-8")
        sizes.append({"patient": db.stem, "lines": text.count("\n"), "chars": len(text),
                      "est_tokens": len(text) // 4})
        print(f"{len(text) // 4:>8,} est. tokens  {db.stem}")

    with open(gdir / "_sizes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sizes[0]))
        w.writeheader()
        w.writerows(sorted(sizes, key=lambda s: s["est_tokens"], reverse=True))
    toks = sorted(s["est_tokens"] for s in sizes)
    print(f"\n{len(sizes)} grouped records written to {gdir}")
    print(f"est. tokens of the full grouped text: median {toks[len(toks) // 2]:,}, max {toks[-1]:,}")


if __name__ == "__main__":
    main()
