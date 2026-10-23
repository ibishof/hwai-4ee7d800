"""
Step 4 of the pipeline: grouped record (JSON) -> context text for the model.

The grouped JSON is the lossless source and never changes here. This script
only decides how it is written out. Each formatting rule is a switch, and
every rule that claims to be lossless is proven by a round-trip check: the
rendered text is parsed back and compared to the JSON for every patient.

Rules available now:
  Rule 1, --collapse-runs (on by default): when a lab or survey value repeats
  back to back, write the value once, with a count, before all of its dates.
      before: 2016-09-25 Never smoked · 2017-10-01 Never smoked · 2018-10-07 Never smoked
      after:  Never smoked (3x): 2016-09-25, 2017-10-01, 2018-10-07
  A single reading is unchanged: "2016-09-25 5.99".

Layout, --no-guide to turn off (on by default):
  - a short legend at the top explaining the conventions
  - section headers (## Medications) instead of a type label on every line
  - "latest <value> on <date>" at the front of every multi-reading line
  - an ID such as [M4] at the start of every item line, plus
    out/context/<patient>.ids.json mapping each ID to its codes and the
    original FHIR resource IDs

Outputs:
  out/context/<patient>.txt    the context for that patient
  out/context/_sizes.csv       characters and estimated tokens per patient

Usage:
  python render_context.py                    # rule 1 on
  python render_context.py --no-collapse-runs # rule 1 off, same as group_record.py's .txt
  python render_context.py --only Merlene Cole
"""
import argparse
import csv
import json
import re
from pathlib import Path

DATE = r"\d{4}-\d{2}-\d{2}"
SEGMENT_SEP = " · "
DATE_SEP = ", "


def fmt(v):
    """Exact value as text: 3.0 -> '3', 0.088799 stays '0.088799'."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


# ------------------------------------------------------------------ rule 1

def format_readings(readings, collapse_runs=True):
    """[[date, value], ...] -> one line of readings.

    A single reading is written date first:      '2017-11-20 5.88'
    With collapse_runs, back-to-back readings with the same value become one
    segment, value first with a count:           'Never smoked (3x): 2016-09-25, 2017-10-01, 2018-10-07'
    Segments are joined with ' · '.
    """
    segments = []
    for d, v in readings:
        text = fmt(v)
        if collapse_runs and segments and segments[-1][1] == text:
            segments[-1][0].append(d)
        else:
            segments.append([[d], text])
    return SEGMENT_SEP.join(
        f"{ds[0]} {text}" if len(ds) == 1 else f"{text} ({len(ds)}x): {DATE_SEP.join(ds)}"
        for ds, text in segments)


RUN = re.compile(rf"(.*) \((\d+)x\): ((?:{DATE}{DATE_SEP})*{DATE})$", re.S)
SINGLE = re.compile(rf"({DATE}) (.*)$", re.S)


def parse_readings(text):
    """Inverse of format_readings. Used only by the round-trip check."""
    out = []
    for seg in text.split(SEGMENT_SEP):
        m = RUN.match(seg)
        if m:
            dates = m.group(3).split(DATE_SEP)
            if len(dates) != int(m.group(2)):
                raise ValueError(f"count {m.group(2)} but {len(dates)} dates: {seg[:80]!r}")
            out.extend((d, m.group(1)) for d in dates)
            continue
        m = SINGLE.match(seg)
        if not m:
            raise ValueError(f"cannot parse segment: {seg[:80]!r}")
        out.append((m.group(1), m.group(2)))
    return out


def check_round_trip(record, collapse_runs):
    """Return a list of problems; empty means every reading survived exactly."""
    problems = []
    for o in record["observations"]:
        expected = [(d, fmt(v)) for d, v in o["readings"]]
        try:
            got = parse_readings(format_readings(o["readings"], collapse_runs))
        except ValueError as e:
            problems.append(f"{o['display']}: {e}")
            continue
        if got != expected:
            first = next(i for i, (a, b) in enumerate(zip(got + [None] * len(expected), expected)) if a != b)
            problems.append(f"{o['display']}: reading {first} differs: "
                            f"got {got[first] if first < len(got) else None}, expected {expected[first]}")
    return problems


# --------------------------------------------------------------- rendering

def code_label(item):
    return f" ({item['system']} {item['code']})" if item.get("code") else ""


def dates_text(dates):
    return SEGMENT_SEP.join(d or "?" for d in dates)


LEGEND = """How to read this record:
- Dates are YYYY-MM-DD. Lists run oldest to newest.
- n = number of readings or events. "latest" is the most recent reading.
- "Never smoked (8x): date, date, ..." means that same value was recorded on each of the 8 dates listed.
- Codes in parentheses: LOINC (labs, vitals, surveys), SNOMED (conditions, procedures), RxNorm (medications), CVX (vaccines).
- Status comes from the source system, which often leaves conditions and medications "active" after they ended. Check the dates.
- Visit note text and fields not shown here are stored separately and can be looked up.
- Each item line starts with an ID in brackets, such as [M4] or [L45]. Cite these IDs."""

CATEGORY_TITLES = {"laboratory": "Laboratory", "vital-signs": "Vital signs", "survey": "Surveys",
                   "social-history": "Social history", "exam": "Exam findings", "imaging": "Imaging findings",
                   "procedure": "Procedure results", "therapy": "Therapy"}


def latest_text(readings):
    """'latest 6.31 on 2025-09-29'. If several different values share the last
    date (two panels drawn the same day), list them all instead of picking one."""
    last = readings[-1][0]
    vals = list(dict.fromkeys(fmt(v) for d, v in readings if d == last))
    return f"latest {' and '.join(vals)} on {last}"


# Prefix for each section's line IDs, e.g. [M4] is the 4th medication line.
ID_PREFIX = {"patient": "PT", "allergies": "A", "conditions": "C", "medications": "M",
             "careplans": "CP", "observations": "L", "procedures": "P", "immunizations": "V",
             "imaging": "I", "devices": "D", "supplies": "S", "panels": "PN",
             "careteams": "CT", "encounters": "E", "notes": "N"}


def render(record, collapse_runs=True, guide=True, id_map=None):
    """guide=True adds the legend, section headers (instead of a type label on
    every line), a latest value at the front of each multi-reading line, and an
    ID such as [M4] at the start of every item line. If id_map is a dict, it is
    filled with ID -> {section, display, code, system, source_ids} so every
    cited line can be traced back to the original FHIR resources."""
    out = []
    counters = {}

    def tag(section_key, item=None):
        """'[M4] ' in the guided layout (and record it in id_map); '' otherwise."""
        if not guide:
            return ""
        counters[section_key] = counters.get(section_key, 0) + 1
        line_id = f"{ID_PREFIX[section_key]}{counters[section_key]}"
        if id_map is not None:
            item = item or {}
            sources = item.get("source_ids") or ([item["id"]] if item.get("id") else [])
            id_map[line_id] = {"section": section_key, "display": item.get("display"),
                               "code": item.get("code"), "system": item.get("system"),
                               "source_ids": sources}
        return f"[{line_id}] "

    def section(title, label):
        """Start a section; return the per-line prefix to use in it."""
        out.append("")
        if guide:
            out.append(f"## {title}")
            return ""
        return f"{label} | "

    p = record["patient"]
    person = (f"{p['name']} | {p['gender']} | born {p['birth_date']}"
              + (f" | died {p['deceased_date']}" if p["deceased_date"] else " | alive")
              + f" | {p['race']}, {p['ethnicity']} | {p['language']} | {p['city']}, {p['state']}")
    if guide:
        out += [f"# Patient record: {p['name']}", "", LEGEND, "", "## Patient",
                tag("patient", {"display": p["name"], "id": p["id"]}) + person]
    else:
        out.append(f"PATIENT | {person}")

    pre = section("Allergies", "ALLERGY")
    if not record["allergies"]:
        out.append(f"{tag('allergies', {'display': 'none recorded'})}{pre}none recorded")
    for a in record["allergies"]:
        out.append(f"{tag('allergies', a)}{pre}{a['display']} | {a['category']} | criticality {a['criticality']}"
                   f" | reaction: {a['reactions'] or 'not recorded'} | {a['status']} since {a['date']}")

    pre = section("Conditions", "CONDITION")
    for c in record["conditions"]:
        eps = SEGMENT_SEP.join(f"{e['onset']}" + (f" to {e['abatement']}" if e["abatement"] else "")
                               for e in c["episodes"])
        n = len(c["episodes"])
        out.append(f"{tag('conditions', c)}{pre}{c['display']}{code_label(c)} | "
                   + (f"{n} episodes: " if n > 1 else "")
                   + f"{eps} | status {c['latest_status']}")

    pre = section("Medications", "MED")
    for m in record["medications"]:
        o = m["orders"]
        when = (f"{len(o)} orders, {o[0]['date']} to {o[-1]['date']}" if len(o) > 1
                else f"1 order, {o[0]['date']}" if o else "no orders")
        line = f"{tag('medications', m)}{pre}{m['display']}{code_label(m)} | {when} | status {m['latest_status']}"
        if m["reasons"]:
            line += f" | reason: {'; '.join(m['reasons'])}"
        if m["dosage"]:
            line += f" | dosage: {m['dosage']}"
        if m["given_in_clinic"]:
            line += f" | given in clinic {len(m['given_in_clinic'])}x"
        out.append(line)

    pre = section("Care plans", "CAREPLAN")
    for cp in record["careplans"]:
        out.append(f"{tag('careplans', cp)}{pre}{cp['display']} | {cp['status']} from {cp['date']}"
                   + (f" to {cp['end_date']}" if cp["end_date"] else "")
                   + (f" | for: {cp['addresses']}" if cp["addresses"] else "")
                   + (f" | activities: {cp['activities']}" if cp["activities"] else ""))

    if guide:
        out += ["", "## Labs, vitals, and surveys"]
    else:
        out.append("")
    current = object()
    for o in record["observations"]:
        cat = o["category"] or "other"
        if guide and cat != current:
            out.append(f"### {CATEGORY_TITLES.get(cat, cat.capitalize())}")
            current = cat
        pre = "" if guide else f"{cat.upper() if o['category'] else 'OBS'} | "
        unit = f" {o['unit']}" if o["unit"] else ""
        part = f" [part of {o['part_of']}]" if o["part_of"] else ""
        many = len(o["readings"]) > 1
        latest = f" | {latest_text(o['readings'])}" if guide and many else ""
        rng = f" | range {fmt(o['min'])}-{fmt(o['max'])}" if "min" in o and many else ""
        vals = format_readings(o["readings"], collapse_runs)
        out.append(f"{tag('observations', o)}{pre}{o['display']}{code_label(o)}{unit}{part} | n={len(o['readings'])}{latest}{rng} | {vals}")

    for key, title, label in (("procedures", "Procedures", "PROCEDURE"), ("immunizations", "Vaccines", "VACCINE"),
                              ("imaging", "Imaging studies", "IMAGING"), ("devices", "Devices", "DEVICE"),
                              ("supplies", "Supplies", "SUPPLY"), ("panels", "Lab panels ordered", "PANEL")):
        pre = section(title, label)
        for x in record[key]:
            extra = "".join(f" | {k}: {', '.join(x[k])}" for k in ("reason", "modality", "body_site", "quantity") if k in x)
            dates = [d for d in x["dates"] if d]
            when = f"n={len(x['dates'])} | {dates_text(dates)}" if dates else f"n={len(x['dates'])}"
            out.append(f"{tag(key, x)}{pre}{x['display']}{code_label(x)} | {when}{extra} | status {x['latest_status']}")

    pre = section("Care teams", "CARETEAM")
    for ct in record["careteams"]:
        out.append(f"{tag('careteams', ct)}{pre}{ct['display'] or 'care team'} | {ct['status']} from {ct['date']}"
                   + (f" to {ct['end_date']}" if ct["end_date"] else "") + f" | {ct['members']}")

    pre = section("Visits", "VISITS")
    for e in record["encounters"]:
        out.append(f"{tag('encounters', e)}{pre}{e['display']} ({e['class']}) | n={len(e['dates'])} | {dates_text(e['dates'])}"
                   + (f" | reasons: {'; '.join(e['reasons'])}" if e["reasons"] else ""))

    n = record["notes"]
    pre = section("Visit notes", "NOTES")
    where = "stored separately" if guide else "in the notes table"
    out.append(f"{tag('notes', {'display': 'visit notes'})}{pre}{n['n']} visit notes, {n['first']} to {n['last']} (full text {where})")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--only", nargs="*", help="name prefixes, e.g. Merlene Cole")
    ap.add_argument("--no-collapse-runs", action="store_true", help="turn rule 1 off")
    ap.add_argument("--no-guide", action="store_true",
                    help="no legend, section headers, or latest values (the plain inspection layout)")
    args = ap.parse_args()
    collapse = not args.no_collapse_runs

    out = Path(args.out)
    cdir = out / "context"
    cdir.mkdir(parents=True, exist_ok=True)
    files = sorted((out / "grouped").glob("*.json"))
    if args.only:
        files = [f for f in files if f.name.startswith(tuple(args.only))]

    sizes, failures = [], 0
    for f in files:
        record = json.loads(f.read_text(encoding="utf-8"))
        problems = check_round_trip(record, collapse)
        if problems:
            failures += 1
            print(f"ROUND-TRIP FAILED  {f.stem}")
            for p in problems[:5]:
                print(f"    {p}")
            continue
        id_map = {}
        text = render(record, collapse, guide=not args.no_guide, id_map=id_map)
        (cdir / f"{f.stem}.txt").write_text(text, encoding="utf-8")
        if id_map:
            (cdir / f"{f.stem}.ids.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
        before = len(render(record, collapse_runs=False, guide=False))
        sizes.append({"patient": f.stem, "chars": len(text), "est_tokens": len(text) // 4,
                      "chars_without_rules": before,
                      "saved_pct": round(100 * (before - len(text)) / before, 1)})
        print(f"{len(text) // 4:>8,} est. tokens  ({sizes[-1]['saved_pct']:>4}% smaller)  {f.stem}")

    if sizes:
        with open(cdir / "_sizes.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sizes[0]))
            w.writeheader()
            w.writerows(sorted(sizes, key=lambda s: s["est_tokens"], reverse=True))
        toks = sorted(s["est_tokens"] for s in sizes)
        print(f"\nrule 1 (collapse runs): {'on' if collapse else 'off'}")
        print(f"round trip: {len(sizes)} patients passed, {failures} failed")
        print(f"est. tokens (chars/4, undercounts number-heavy text): "
              f"median {toks[len(toks) // 2]:,}, max {toks[-1]:,}")
        print(f"wrote {cdir}")


if __name__ == "__main__":
    main()
