"""
Profile the shape of the Synthea data across all patients.

Answers questions like:
  - Which resource types exist, how many, in how many patients?
  - Which fields are always filled vs. rare (per resource type)?
  - What status values appear (active, resolved, stopped...)?
  - What references point where (which links are real)?
  - How many distinct labs/vitals/conditions/meds exist? Units consistent?
  - What makes the big patients big (repetition vs. many distinct facts)?
  - How small does one-line-per-item summarizing make each patient?

Writes CSVs to out/profile/ and a draft summary per patient to
out/profile/summary/<file>.txt (one line per lab, condition, med, etc.).

Usage:
  python profile_data.py PATH_TO_DATA_DIR
"""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from measure_size import SHARED_PREFIXES, DROP, code_text, when, value, status

SKIP_ITEMS = DROP | {"Medication", "DocumentReference", "Patient"}


def value_kind(r):
    for k in ("valueQuantity", "valueCodeableConcept", "valueString",
              "valueBoolean", "valueInteger"):
        if k in r:
            return k[5:]
    return "component" if "component" in r else "none"


def name_of(r, by_url):
    concept = (r.get("code") or r.get("medicationCodeableConcept")
               or r.get("vaccineCode") or r.get("type"))
    if not concept and "medicationReference" in r:
        med = by_url.get(r["medicationReference"].get("reference"), {})
        concept = med.get("code")
    return code_text(concept)


def refs(r):
    """Yield (field, reference) for top-level reference fields."""
    for k, v in r.items():
        for it in (v if isinstance(v, list) else [v]):
            if isinstance(it, dict) and "reference" in it:
                yield k, it["reference"]


def target(ref, by_url):
    if ref.startswith("urn:uuid:"):
        return by_url.get(ref, {}).get("resourceType", "MISSING")
    if "?identifier=" in ref:
        return ref.split("?")[0] + " (shared file)"
    return ref.split("/")[0]


def render_summary(items):
    """One line per (type, name): count, first/last date, latest value, range."""
    lines = []
    for (t, name), rows in sorted(items.items()):
        rows.sort(key=lambda x: x[0])
        nums = [x[3] for x in rows if x[3] is not None]
        parts = [t, name, f"n={len(rows)}"]
        if len(rows) > 1 and rows[0][0]:
            parts.append(f"first {rows[0][0]}")
        if rows[-1][0]:
            parts.append(f"last {rows[-1][0]}")
        if rows[-1][1]:
            parts.append(f"latest {rows[-1][1]}")
        if len(nums) > 1:
            parts.append(f"range {min(nums):g}-{max(nums):g}")
        if rows[-1][2]:
            parts.append(rows[-1][2])
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def top(counter, n=5):
    return "; ".join(f"{k}:{v}" for k, v in counter.most_common(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    out = Path(args.out) / "profile"
    (out / "summary").mkdir(parents=True, exist_ok=True)

    type_count, type_pat = Counter(), defaultdict(set)
    field_count, status_count, ref_count, enc_class = Counter(), Counter(), Counter(), Counter()
    code_count, code_pat = Counter(), defaultdict(set)
    code_kind, code_unit, code_cat = (defaultdict(Counter) for _ in range(3))
    code_min, code_max = {}, {}
    patient_rows = []

    files = [f for f in sorted(Path(args.data_dir).glob("*.json"))
             if not f.name.startswith(SHARED_PREFIXES)]
    for f in files:
        entries = json.loads(f.read_text(encoding="utf-8")).get("entry", [])
        by_url = {e.get("fullUrl"): e["resource"] for e in entries}
        items = defaultdict(list)  # (type, name) -> [(date, value, status, number)]
        dates = []

        for e in entries:
            r = e["resource"]
            t = r["resourceType"]
            type_count[t] += 1
            type_pat[t].add(f.name)
            for k in r:
                field_count[(t, k)] += 1
            st = status(r)
            if st:
                status_count[(t, st)] += 1
            for k, ref in refs(r):
                ref_count[(t, k, target(ref, by_url))] += 1
            if t == "Encounter":
                enc_class[(r.get("class") or {}).get("code", "")] += 1

            if t in SKIP_ITEMS:
                continue
            name = name_of(r, by_url)
            if not name:
                continue
            key = (t, name)
            d = when(r)
            num = None
            if "valueQuantity" in r:
                q = r["valueQuantity"]
                code_unit[key][q.get("unit", "")] += 1
                if isinstance(q.get("value"), (int, float)):
                    num = q["value"]
                    code_min[key] = min(code_min.get(key, num), num)
                    code_max[key] = max(code_max.get(key, num), num)
            items[key].append((d, value(r), st, num))
            if d:
                dates.append(d)
            code_count[key] += 1
            code_pat[key].add(f.name)
            code_kind[key][value_kind(r)] += 1
            for c in r.get("category", []):
                code_cat[key][code_text(c)] += 1

        summary = render_summary(items)
        (out / "summary" / f"{f.stem}.txt").write_text(summary, encoding="utf-8")
        n_lines = sum(len(v) for v in items.values())
        top_key = max(items, key=lambda k: len(items[k])) if items else ("", "")
        years = (int(max(dates)[:4]) - int(min(dates)[:4])) if dates else 0
        patient_rows.append([
            f.name, n_lines, len(items), " / ".join(top_key),
            round(len(items[top_key]) / max(n_lines, 1), 3), years,
            len(summary) // 4,
        ])

    n_files = len(files)
    write_csv(out / "resource_types.csv", ["type", "count", "patients"],
              [[t, c, len(type_pat[t])] for t, c in type_count.most_common()])
    write_csv(out / "fields.csv", ["type", "field", "fill_rate"],
              [[t, k, round(c / type_count[t], 3)]
               for (t, k), c in sorted(field_count.items())])
    write_csv(out / "statuses.csv", ["type", "status", "count"],
              [[t, s, c] for (t, s), c in sorted(status_count.items())])
    write_csv(out / "references.csv", ["from_type", "field", "to_type", "count"],
              [[a, b, c, n] for (a, b, c), n in ref_count.most_common()])
    write_csv(out / "codes.csv",
              ["type", "name", "count", "patients", "value_kinds", "units",
               "categories", "min", "max"],
              [[t, n, c, len(code_pat[(t, n)]), top(code_kind[(t, n)]),
                top(code_unit[(t, n)]), top(code_cat[(t, n)]),
                code_min.get((t, n), ""), code_max.get((t, n), "")]
               for (t, n), c in code_count.most_common()])
    write_csv(out / "patients.csv",
              ["file", "item_lines", "distinct_items", "top_item",
               "top_item_share", "years_spanned", "summary_tok_est"],
              sorted(patient_rows, key=lambda r: r[1], reverse=True))

    # Highlights
    distinct = Counter(t for t, _ in code_count)
    print(f"{n_files} patients\n\nDistinct names per type:")
    for t, n in distinct.most_common():
        print(f"  {t:<22} {n}")
    multi_unit = [k for k, u in code_unit.items() if len(u) > 1]
    print(f"\nObservation names with more than one unit: {len(multi_unit)}")
    for k in multi_unit[:10]:
        print(f"  {k[1]}: {dict(code_unit[k])}")
    print(f"\nEncounter classes: {dict(enc_class.most_common())}")
    s = [r[6] for r in patient_rows]
    print(f"\nSummary tokens (est): median {statistics.median(s):,.0f}, max {max(s):,}")
    print("\nBiggest patients: what dominates?")
    for r in sorted(patient_rows, key=lambda r: r[1], reverse=True)[:5]:
        print(f"  {r[0][:35]:<35} {r[1]:>7,} lines, {r[2]:>4} distinct, "
              f"top '{r[3][:45]}' = {r[4]:.0%}, {r[5]} yrs, summary ~{r[6]:,} tok")
    print(f"\nWrote CSVs and summaries to {out}/")


if __name__ == "__main__":
    main()
