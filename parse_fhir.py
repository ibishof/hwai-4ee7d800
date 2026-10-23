"""
Step 1 and 2 of the pipeline: FHIR bundle -> one SQLite file per patient.

What it does, per patient bundle:
  1. Removes billing (Claim, ExplanationOfBenefit), audit (Provenance), and
     duplicates (each note's second copy, stored as a DiagnosticReport whose
     only content is presentedForm). Removed resources are written to
     out/removed/<patient>.jsonl so you can inspect them, plus a readable
     summary in out/removed/<patient>_summary.txt.
  2. Flattens every remaining resource into one row per fact (one reading,
     one order, one diagnosis) in out/db/<patient>.sqlite. Every row keeps
     the original resource as JSON in a `raw` column, so no clinical field
     is lost even if it has no column of its own.

No LLM is involved. Plain Python and the standard library only.

Usage:
  python parse_fhir.py PATH_TO_DATA_DIR            # all patients
  python parse_fhir.py PATH_TO_DATA_DIR --only Merlene Cole
"""
import argparse
import base64
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

SHARED_PREFIXES = ("hospitalInformation", "practitionerInformation")
REMOVED_TYPES = {
    "Claim": "billing",
    "ExplanationOfBenefit": "billing",
    "Provenance": "audit",
}
SYSTEM_NAMES = {
    "http://loinc.org": "LOINC",
    "http://snomed.info/sct": "SNOMED",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
    "http://hl7.org/fhir/sid/cvx": "CVX",
}
# Patient identifiers we never store (synthetic here, but real IDs in practice).
SENSITIVE_ID_CODES = {"SS", "DL", "PPN"}


# ---------------------------------------------------------------- helpers

def first_coding(concept):
    """(system, code, display) from a CodeableConcept, a list of them, or a str."""
    if not concept:
        return None, None, None
    if isinstance(concept, list):
        concept = concept[0] if concept else {}
    if isinstance(concept, str):  # e.g. AllergyIntolerance.category is ["food"]
        return None, None, concept
    coding = (concept.get("coding") or [{}])[0]
    system = coding.get("system") or ""
    system = SYSTEM_NAMES.get(system, system.rsplit("/", 1)[-1])
    return system or None, coding.get("code"), concept.get("text") or coding.get("display")


def display_of(concept):
    return first_coding(concept)[2]


def ref_id(reference):
    """'urn:uuid:abc' -> 'abc'. Shared-file references are returned unchanged."""
    if not reference:
        return None
    ref = reference.get("reference") if isinstance(reference, dict) else reference
    if ref and ref.startswith("urn:uuid:"):
        return ref[len("urn:uuid:"):]
    return ref


def day(dt):
    return dt[:10] if dt else None


def clinical_status(r):
    cs = r.get("clinicalStatus")
    return (cs.get("coding") or [{}])[0].get("code") if cs else r.get("status")


def strip_digits(name):
    return re.sub(r"\d+", "", name or "").strip()


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ------------------------------------------------------------ shared files

def load_practitioners(data_dir):
    """NPI -> (name, specialty) from the shared practitioner file."""
    out = {}
    for f in data_dir.glob("practitionerInformation*.json"):
        entries = json.loads(f.read_text(encoding="utf-8"))["entry"]
        names, specialty = {}, {}
        for e in entries:
            r = e["resource"]
            if r["resourceType"] == "Practitioner":
                npi = r["identifier"][0]["value"]
                n = (r.get("name") or [{}])[0]
                names[npi] = strip_digits(" ".join(n.get("prefix", []) + n.get("given", []) + [n.get("family", "")]))
            elif r["resourceType"] == "PractitionerRole":
                npi = r["practitioner"]["identifier"]["value"]
                specialty[npi] = display_of(r.get("specialty"))
        for npi, name in names.items():
            out[npi] = (name, specialty.get(npi))
    return out


def npi_of(reference):
    ref = (reference or {}).get("reference", "")
    return ref.split("|")[-1] if "us-npi|" in ref else None


# ------------------------------------------------------------------ schema

COMMON = "id TEXT PRIMARY KEY, encounter_id TEXT, date TEXT, datetime TEXT, " \
         "system TEXT, code TEXT, display TEXT, status TEXT"

SCHEMA = f"""
CREATE TABLE patient (id TEXT PRIMARY KEY, name TEXT, maiden_name TEXT, gender TEXT,
  birth_date TEXT, deceased_date TEXT, race TEXT, ethnicity TEXT, birth_sex TEXT,
  marital_status TEXT, language TEXT, city TEXT, state TEXT, raw TEXT);
CREATE TABLE encounters ({COMMON}, end_datetime TEXT, class TEXT, reason TEXT,
  practitioner TEXT, specialty TEXT, location TEXT, organization TEXT, raw TEXT);
CREATE TABLE conditions ({COMMON}, onset_date TEXT, abatement_date TEXT,
  verification TEXT, category TEXT, raw TEXT);
CREATE TABLE observations (id TEXT, part INTEGER, encounter_id TEXT, date TEXT, datetime TEXT,
  system TEXT, code TEXT, display TEXT, status TEXT, category TEXT,
  value_num REAL, value_text TEXT, unit TEXT,
  parent_code TEXT, parent_display TEXT, report_id TEXT, report_display TEXT, raw TEXT,
  PRIMARY KEY (id, part));
CREATE TABLE diagnostic_reports ({COMMON}, category TEXT, n_results INTEGER, raw TEXT);
CREATE TABLE med_requests ({COMMON}, intent TEXT, category TEXT, reason_condition_id TEXT,
  reason TEXT, dosage TEXT, as_needed INTEGER, requester TEXT, raw TEXT);
CREATE TABLE med_administrations ({COMMON}, reason_condition_id TEXT, reason TEXT, raw TEXT);
CREATE TABLE procedures ({COMMON}, end_datetime TEXT, reason_condition_id TEXT, reason TEXT,
  location TEXT, raw TEXT);
CREATE TABLE immunizations ({COMMON}, location TEXT, raw TEXT);
CREATE TABLE allergies ({COMMON}, type TEXT, category TEXT, criticality TEXT,
  verification TEXT, reactions TEXT, raw TEXT);
CREATE TABLE careplans ({COMMON}, end_date TEXT, addresses_condition_id TEXT,
  addresses TEXT, activities TEXT, raw TEXT);
CREATE TABLE careteams ({COMMON}, end_date TEXT, reason TEXT, members TEXT, raw TEXT);
CREATE TABLE devices ({COMMON}, raw TEXT);
CREATE TABLE imaging_studies ({COMMON}, modality TEXT, body_site TEXT,
  n_series INTEGER, n_instances INTEGER, raw TEXT);
CREATE TABLE supply_deliveries ({COMMON}, quantity REAL, raw TEXT);
CREATE TABLE notes (id TEXT PRIMARY KEY, encounter_id TEXT, date TEXT, datetime TEXT,
  type TEXT, status TEXT, author TEXT, text TEXT, raw TEXT);
CREATE VIRTUAL TABLE notes_fts USING fts5(text, note_id UNINDEXED, date UNINDEXED);

-- One place for most questions: every dated clinical fact as a single row.
CREATE VIEW facts AS
  SELECT 'condition' kind, id source_id, encounter_id, onset_date date, system, code, display,
         NULL value_num, NULL value_text, NULL unit, status FROM conditions
  UNION ALL SELECT 'observation', id, encounter_id, date, system, code, display,
         value_num, value_text, unit, status FROM observations
  UNION ALL SELECT 'med_request', id, encounter_id, date, system, code, display,
         NULL, reason, NULL, status FROM med_requests
  UNION ALL SELECT 'med_administration', id, encounter_id, date, system, code, display,
         NULL, reason, NULL, status FROM med_administrations
  UNION ALL SELECT 'procedure', id, encounter_id, date, system, code, display,
         NULL, reason, NULL, status FROM procedures
  UNION ALL SELECT 'immunization', id, encounter_id, date, system, code, display,
         NULL, NULL, NULL, status FROM immunizations
  UNION ALL SELECT 'allergy', id, encounter_id, date, system, code, display,
         NULL, reactions, NULL, status FROM allergies
  UNION ALL SELECT 'careplan', id, encounter_id, date, system, code, display,
         NULL, addresses, NULL, status FROM careplans
  UNION ALL SELECT 'device', id, encounter_id, date, system, code, display,
         NULL, NULL, NULL, status FROM devices
  UNION ALL SELECT 'imaging', id, encounter_id, date, system, code, display,
         NULL, modality, NULL, status FROM imaging_studies
  UNION ALL SELECT 'encounter', id, id, date, system, code, display,
         NULL, reason, NULL, status FROM encounters;
"""


# ---------------------------------------------------- one function per type
# Each returns (table, row dict) or a list of them. `ctx` holds lookups.

def row_common(r, date_field, concept, encounter_field="encounter"):
    system, code, disp = first_coding(concept)
    dt = date_field
    return {"id": r["id"], "encounter_id": ref_id(r.get(encounter_field)),
            "date": day(dt), "datetime": dt, "system": system, "code": code,
            "display": disp, "status": clinical_status(r), "raw": dumps(r)}


def parse_patient(r, ctx):
    ext = {x["url"].rsplit("/", 1)[-1]: x for x in r.get("extension", [])}

    def ext_text(key):
        sub = ext.get(key, {}).get("extension", [])
        return next((s.get("valueString") for s in sub if s["url"] == "text"), None)

    # Never store SSN, driver's license, or passport numbers.
    kept_ids, dropped_ids = [], []
    for ident in r.get("identifier", []):
        code = ((ident.get("type") or {}).get("coding") or [{}])[0].get("code")
        (dropped_ids if code in SENSITIVE_ID_CODES else kept_ids).append(ident)
    clean = dict(r, identifier=kept_ids)
    ctx["removed_patient_ids"] = dropped_ids

    names = r.get("name", [])
    official = next((n for n in names if n.get("use") == "official"), names[0] if names else {})
    maiden = next((n for n in names if n.get("use") == "maiden"), None)
    addr = (r.get("address") or [{}])[0]
    return "patient", {
        "id": r["id"],
        "name": strip_digits(" ".join(official.get("given", []) + [official.get("family", "")])),
        "maiden_name": strip_digits(maiden.get("family")) if maiden else None,
        "gender": r.get("gender"), "birth_date": r.get("birthDate"),
        "deceased_date": day(r.get("deceasedDateTime")),
        "race": ext_text("us-core-race"), "ethnicity": ext_text("us-core-ethnicity"),
        "birth_sex": ext.get("us-core-birthsex", {}).get("valueCode"),
        "marital_status": display_of(r.get("maritalStatus")),
        "language": display_of((r.get("communication") or [{}])[0].get("language")),
        "city": addr.get("city"), "state": addr.get("state"), "raw": dumps(clean)}


def parse_encounter(r, ctx):
    row = row_common(r, r["period"]["start"], r.get("type"))
    row["id"] = r["id"]
    row["encounter_id"] = r["id"]
    part = next((p for p in r.get("participant", []) if p.get("individual")), {})
    npi = npi_of(part.get("individual"))
    name, specialty = ctx["practitioners"].get(npi, (None, None))
    row.update(end_datetime=r["period"].get("end"), status=r.get("status"),
               **{"class": (r.get("class") or {}).get("code")},
               reason=display_of(r.get("reasonCode")),
               practitioner=name or strip_digits((part.get("individual") or {}).get("display")),
               specialty=specialty,
               location=((r.get("location") or [{}])[0].get("location") or {}).get("display"),
               organization=(r.get("serviceProvider") or {}).get("display"))
    return "encounters", row


def parse_condition(r, ctx):
    row = row_common(r, r.get("onsetDateTime") or r.get("recordedDate"), r.get("code"))
    row.update(onset_date=day(r.get("onsetDateTime")),
               abatement_date=day(r.get("abatementDateTime")),
               verification=first_coding(r.get("verificationStatus"))[1],
               category=display_of(r.get("category")) or first_coding(r.get("category"))[1])
    return "conditions", row


def observation_value(obs):
    """(value_num, value_text, unit) for an Observation or one of its components."""
    if "valueQuantity" in obs:
        q = obs["valueQuantity"]
        v = q.get("value")
        return (v if isinstance(v, (int, float)) else None), None, q.get("unit") or q.get("code")
    if "valueCodeableConcept" in obs:
        return None, display_of(obs["valueCodeableConcept"]), None
    for k in ("valueString", "valueBoolean", "valueInteger", "valueDateTime"):
        if k in obs:
            v = obs[k]
            return (v, None, None) if isinstance(v, (int, float)) and not isinstance(v, bool) else (None, str(v), None)
    return None, None, None


def parse_observation(r, ctx):
    """One row per value. Multi-value observations (blood pressure, surveys)
    become one row per component, linked back by parent_code."""
    base = row_common(r, r.get("effectiveDateTime"), r.get("code"))
    base["category"] = first_coding(r.get("category"))[1]
    report_id, report_display = ctx["obs_to_report"].get(r["id"], (None, None))
    base.update(report_id=report_id, report_display=report_display)
    rows = []
    if r.get("component"):
        for i, comp in enumerate(r["component"], start=1):
            system, code, disp = first_coding(comp.get("code"))
            num, text, unit = observation_value(comp)
            rows.append(("observations", dict(base, part=i, system=system, code=code, display=disp,
                                               value_num=num, value_text=text, unit=unit,
                                               parent_code=base["code"], parent_display=base["display"],
                                               raw=base["raw"] if i == 1 else None)))
    if not r.get("component") or any(k.startswith("value") for k in r):
        num, text, unit = observation_value(r)
        rows.append(("observations", dict(base, part=0, value_num=num, value_text=text, unit=unit,
                                           parent_code=None, parent_display=None)))
    return rows


def parse_diagnostic_report(r, ctx):
    row = row_common(r, r.get("effectiveDateTime"), r.get("code"))
    row.update(category=first_coding(r.get("category"))[1], n_results=len(r.get("result", [])))
    return "diagnostic_reports", row


def med_concept(r, ctx):
    if "medicationCodeableConcept" in r:
        return r["medicationCodeableConcept"]
    med = ctx["by_id"].get(ref_id(r.get("medicationReference")), {})
    return med.get("code")


def reason_of(r, ctx):
    """(condition_id, display) from reasonReference, else reasonCode."""
    rr = r.get("reasonReference")
    if rr:
        cid = ref_id(rr[0])
        cond = ctx["by_id"].get(cid, {})
        return cid, display_of(cond.get("code")) or rr[0].get("display")
    return None, display_of(r.get("reasonCode"))


def parse_med_request(r, ctx):
    row = row_common(r, r.get("authoredOn"), med_concept(r, ctx))
    cid, reason = reason_of(r, ctx)
    dose = (r.get("dosageInstruction") or [{}])[0]
    row.update(intent=r.get("intent"), category=display_of(r.get("category")),
               reason_condition_id=cid, reason=reason, dosage=dose.get("text"),
               as_needed=int(dose["asNeededBoolean"]) if "asNeededBoolean" in dose else None,
               requester=strip_digits((r.get("requester") or {}).get("display")))
    return "med_requests", row


def parse_med_administration(r, ctx):
    row = row_common(r, r.get("effectiveDateTime"), med_concept(r, ctx), encounter_field="context")
    cid, reason = reason_of(r, ctx)
    row.update(reason_condition_id=cid, reason=reason)
    return "med_administrations", row


def parse_procedure(r, ctx):
    period = r.get("performedPeriod") or {}
    row = row_common(r, period.get("start") or r.get("performedDateTime"), r.get("code"))
    cid, reason = reason_of(r, ctx)
    row.update(end_datetime=period.get("end"), reason_condition_id=cid, reason=reason,
               location=(r.get("location") or {}).get("display"))
    return "procedures", row


def parse_immunization(r, ctx):
    row = row_common(r, r.get("occurrenceDateTime"), r.get("vaccineCode"))
    row["location"] = (r.get("location") or {}).get("display")
    return "immunizations", row


def parse_allergy(r, ctx):
    row = row_common(r, r.get("recordedDate"), r.get("code"))
    reactions = []
    for rx in r.get("reaction", []):
        what = ", ".join(display_of(m) or "" for m in rx.get("manifestation", []))
        reactions.append(f"{what} ({rx['severity']})" if rx.get("severity") else what)
    row.update(type=r.get("type"), category=", ".join(r.get("category", [])),
               criticality=r.get("criticality"),
               verification=first_coding(r.get("verificationStatus"))[1],
               reactions="; ".join(reactions) or None)
    return "allergies", row


def parse_careplan(r, ctx):
    # The first category is a generic "assess-plan" tag; the plan itself is the second.
    cats = r.get("category", [])
    concept = next((c for c in cats if (c.get("coding") or [{}])[0].get("display")), cats[0] if cats else None)
    row = row_common(r, (r.get("period") or {}).get("start"), concept)
    addr = r.get("addresses") or []
    cid = ref_id(addr[0]) if addr else None
    acts = [display_of((a.get("detail") or {}).get("code")) for a in r.get("activity", [])]
    row.update(end_date=day((r.get("period") or {}).get("end")), addresses_condition_id=cid,
               addresses=display_of(ctx["by_id"].get(cid, {}).get("code")) if cid else None,
               activities="; ".join(a for a in acts if a) or None)
    return "careplans", row


def parse_careteam(r, ctx):
    row = row_common(r, (r.get("period") or {}).get("start"), r.get("reasonCode"))
    members = [strip_digits((p.get("member") or {}).get("display")) for p in r.get("participant", [])]
    row.update(end_date=day((r.get("period") or {}).get("end")), reason=row["display"],
               members="; ".join(m for m in members if m) or None)
    return "careteams", row


def parse_device(r, ctx):
    # Synthea gives no date the patient received the device, only a
    # manufacture date, so date is left empty rather than guessed.
    row = row_common(r, None, r.get("type"))
    return "devices", row


def parse_imaging(r, ctx):
    row = row_common(r, r.get("started"), r.get("procedureCode"))
    series = (r.get("series") or [{}])[0]
    row.update(modality=(series.get("modality") or {}).get("display"),
               body_site=(series.get("bodySite") or {}).get("display"),
               n_series=r.get("numberOfSeries"), n_instances=r.get("numberOfInstances"))
    return "imaging_studies", row


def parse_supply(r, ctx):
    item = r.get("suppliedItem") or {}
    row = row_common(r, r.get("occurrenceDateTime"), item.get("itemCodeableConcept"))
    row["quantity"] = (item.get("quantity") or {}).get("value")
    return "supply_deliveries", row


def parse_note(r, ctx):
    data = (r.get("content") or [{}])[0].get("attachment", {}).get("data", "")
    raw = json.loads(dumps(r))
    raw["content"][0]["attachment"]["data"] = "<decoded into the text column>"
    return "notes", {
        "id": r["id"], "encounter_id": ref_id((r.get("context") or {}).get("encounter", [None])[0]),
        "date": day(r.get("date")), "datetime": r.get("date"),
        "type": display_of(r.get("type")), "status": r.get("status"),
        "author": strip_digits(((r.get("author") or [{}])[0]).get("display")),
        "text": base64.b64decode(data).decode("utf-8", "replace"), "raw": dumps(raw)}


PARSERS = {
    "Patient": parse_patient, "Encounter": parse_encounter, "Condition": parse_condition,
    "Observation": parse_observation, "DiagnosticReport": parse_diagnostic_report,
    "MedicationRequest": parse_med_request, "MedicationAdministration": parse_med_administration,
    "Procedure": parse_procedure, "Immunization": parse_immunization,
    "AllergyIntolerance": parse_allergy, "CarePlan": parse_careplan, "CareTeam": parse_careteam,
    "Device": parse_device, "ImagingStudy": parse_imaging, "SupplyDelivery": parse_supply,
    "DocumentReference": parse_note,
}
# Medication only names the drug for requests that point to it; it is folded
# into med_requests.code/display rather than kept as its own table.
FOLDED_TYPES = {"Medication"}


# ------------------------------------------------------------ main routine

def removal_reason(r):
    t = r["resourceType"]
    if t in REMOVED_TYPES:
        return REMOVED_TYPES[t]
    if t == "DiagnosticReport" and "presentedForm" in r and not r.get("result"):
        return "duplicate: copy of the visit note already stored as a DocumentReference"
    return None


def parse_bundle(path, out_dir, practitioners):
    entries = json.loads(path.read_text(encoding="utf-8"))["entry"]
    resources = [e["resource"] for e in entries]
    ctx = {"practitioners": practitioners,
           "by_id": {r["id"]: r for r in resources},
           "obs_to_report": {}}
    for r in resources:
        if r["resourceType"] == "DiagnosticReport":
            for res in r.get("result", []):
                ctx["obs_to_report"][ref_id(res)] = (r["id"], display_of(r.get("code")))

    rows, removed, unhandled = defaultdict(list), [], Counter()
    for r in resources:
        t = r["resourceType"]
        reason = removal_reason(r)
        if reason:
            removed.append({"reason": reason, "resource": r})
            continue
        if t in FOLDED_TYPES:
            continue
        if t not in PARSERS:
            unhandled[t] += 1
            removed.append({"reason": "no parser for this resource type", "resource": r})
            continue
        result = PARSERS[t](r, ctx)
        for table, row in (result if isinstance(result, list) else [result]):
            rows[table].append(row)
    for ident in ctx.get("removed_patient_ids", []):
        removed.append({"reason": "sensitive identifier (SSN, license, or passport)",
                        "resource": {"resourceType": "Patient.identifier", **ident}})

    # Write SQLite
    db_path = out_dir / "db" / f"{path.stem}.sqlite"
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    for table, table_rows in rows.items():
        cols = list(table_rows[0])
        con.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [tuple(row.get(c) for c in cols) for row in table_rows])
    con.executemany("INSERT INTO notes_fts (text, note_id, date) VALUES (?, ?, ?)",
                    [(n["text"], n["id"], n["date"]) for n in rows["notes"]])
    for table in ("encounters", "conditions", "observations", "med_requests", "procedures"):
        con.execute(f"CREATE INDEX idx_{table}_code ON {table} (code, date)")
    con.commit()
    con.close()

    write_removed(path, out_dir, removed)
    return {t: len(v) for t, v in rows.items()}, Counter(x["reason"] for x in removed), unhandled


# --------------------------------------------------- removed-data reports

def walk_strings(obj, prefix=""):
    """Yield (field path, string) for every display/text/value string in a resource."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v, prefix)
    elif isinstance(obj, str) and prefix.split(".")[-1] in {"display", "text", "valueString"}:
        yield prefix, obj


def leaf_paths(obj, prefix=""):
    """Yield the dotted path of every leaf value, e.g. 'total.amount.value'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaf_paths(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaf_paths(v, prefix)
    else:
        yield prefix


def write_removed(path, out_dir, removed):
    removed_dir = out_dir / "removed"
    with open(removed_dir / f"{path.stem}.jsonl", "w", encoding="utf-8") as fh:
        for item in removed:
            fh.write(dumps(item) + "\n")

    lines = [f"Removed from {path.name}", ""]
    by_type = defaultdict(list)
    for item in removed:
        by_type[(item["resource"]["resourceType"], item["reason"])].append(item["resource"])
    for (t, reason), items in sorted(by_type.items()):
        lines.append(f"== {t}: {len(items)} removed ({reason})")
        strings, paths = defaultdict(Counter), Counter()
        for r in items:
            for field, s in walk_strings(r):
                strings[field][s] += 1
            paths.update(set(leaf_paths(r)))
        lines.append("   fields present: " + ", ".join(f"{p} [{n}]" for p, n in sorted(paths.items())))
        for field, counter in sorted(strings.items()):
            top = ", ".join(f"{s} [{n}]" for s, n in counter.most_common(8))
            more = f" ... +{len(counter) - 8} more distinct" if len(counter) > 8 else ""
            lines.append(f"   {field}: {top}{more}")
        lines.append("")
    (removed_dir / f"{path.stem}_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="out")
    ap.add_argument("--only", nargs="*", help="name prefixes to parse, e.g. Merlene Cole")
    args = ap.parse_args()

    data_dir, out = Path(args.data_dir), Path(args.out)
    for sub in ("db", "removed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    practitioners = load_practitioners(data_dir)

    files = [f for f in sorted(data_dir.glob("*.json")) if not f.name.startswith(SHARED_PREFIXES)]
    if args.only:
        files = [f for f in files if f.name.startswith(tuple(args.only))]
    totals, removed_totals, unhandled_totals = Counter(), Counter(), Counter()
    for f in files:
        counts, removed, unhandled = parse_bundle(f, out, practitioners)
        totals.update(counts)
        removed_totals.update(removed)
        unhandled_totals.update(unhandled)
        print(f"{sum(counts.values()):>7,} rows  {sum(removed.values()):>6,} removed  {f.stem}")

    print(f"\n{len(files)} patients parsed into {out / 'db'}")
    print("rows per table:", dict(totals.most_common()))
    print("removed:", dict(removed_totals))
    if unhandled_totals:
        print("WARNING, types with no parser (saved to removed files):", dict(unhandled_totals))


if __name__ == "__main__":
    main()
