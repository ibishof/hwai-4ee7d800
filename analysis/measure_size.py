"""
Measure how big each patient's record is after dropping the junk.

For every patient bundle it reports:
  raw_mb          original file size
  kept_json_tok   approx tokens of the remaining resources as compact JSON
  flat_tok        approx tokens if each resource becomes one plain-text line
  note_tok        approx tokens of decoded visit notes (kept separate)
It also writes each patient's flattened text to out/flat/<file>.txt so you
can open the biggest one and see how complex it really is.

Token estimates use chars/4, which undercounts JSON (UUIDs tokenize badly).
Use --count N to get the official count for the N largest patients.

Usage:
  python measure_size.py PATH_TO_DATA_DIR
  python measure_size.py PATH_TO_DATA_DIR --count 5
"""
import argparse
import base64
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

SHARED_PREFIXES = ("hospitalInformation", "practitionerInformation")
DROP = {"Claim", "ExplanationOfBenefit", "Provenance"}


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def code_text(cc):
    """Display text for a CodeableConcept (or a list of them)."""
    if not cc:
        return ""
    if isinstance(cc, list):
        cc = cc[0] if cc else {}
    if isinstance(cc, str):  # e.g. AllergyIntolerance.category is ["food"]
        return cc
    codings = cc.get("coding") or [{}]
    return cc.get("text") or codings[0].get("display", "")


def when(r):
    for k in ("effectiveDateTime", "onsetDateTime", "authoredOn",
              "occurrenceDateTime", "performedDateTime", "recordedDate",
              "issued", "date"):
        if r.get(k):
            return r[k][:10]
    for k in ("period", "performedPeriod", "effectivePeriod"):
        start = (r.get(k) or {}).get("start")
        if start:
            return start[:10]
    return ""


def value(r):
    if "valueQuantity" in r:
        q = r["valueQuantity"]
        return f"{q.get('value')} {q.get('unit', '')}".strip()
    if "valueCodeableConcept" in r:
        return code_text(r["valueCodeableConcept"])
    if "valueString" in r:
        return r["valueString"]
    if "component" in r:
        return "; ".join(f"{code_text(c.get('code'))}={value(c)}"
                         for c in r["component"])
    return ""


def status(r):
    cs = r.get("clinicalStatus")
    if cs:
        return (cs.get("coding") or [{}])[0].get("code", "")
    return r.get("status", "")


def flat_line(r, by_url):
    t = r["resourceType"]
    if t == "Patient":
        n = (r.get("name") or [{}])[0]
        name = " ".join(n.get("given", []) + [n.get("family", "")])
        return f"Patient | {name} | {r.get('gender', '')} | {r.get('birthDate', '')}"
    concept = (r.get("code") or r.get("medicationCodeableConcept")
               or r.get("vaccineCode") or r.get("type"))
    if not concept and "medicationReference" in r:
        med = by_url.get(r["medicationReference"].get("reference"), {})
        concept = med.get("code")
    parts = (t, when(r), code_text(concept), value(r), status(r))
    return " | ".join(p for p in parts if p)


def measure(path, out_dir):
    bundle = json.loads(path.read_text(encoding="utf-8"))
    entries = bundle.get("entry", [])
    by_url = {e.get("fullUrl"): e["resource"] for e in entries}

    types = Counter()
    kept_chars = 0
    lines, notes = [], []
    for e in entries:
        r = e["resource"]
        t = r["resourceType"]
        types[t] += 1
        if t in DROP or t == "Medication":  # Medication is folded into its request
            continue
        if t == "DocumentReference":
            for c in r.get("content", []):
                data = (c.get("attachment") or {}).get("data")
                if data:
                    notes.append(base64.b64decode(data).decode("utf-8", "replace"))
            continue
        if t == "DiagnosticReport":
            r = {k: v for k, v in r.items() if k != "presentedForm"}
        kept_chars += len(compact(r))
        lines.append(flat_line(r, by_url))

    flat = "\n".join(lines)
    note_text = "\n\n".join(notes)
    (out_dir / f"{path.stem}.txt").write_text(flat, encoding="utf-8")

    row = {
        "file": path.name,
        "raw_mb": round(path.stat().st_size / 1e6, 1),
        "resources": sum(types.values()),
        "kept_json_tok": kept_chars // 4,
        "flat_lines": len(lines),
        "flat_words": len(flat.split()),
        "flat_tok": len(flat) // 4,
        "note_words": len(note_text.split()),
        "note_tok": len(note_text) // 4,
    }
    return row, flat, note_text


def load_env(path=".env"):
    """Tiny .env reader so we don't need python-dotenv."""
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def official_count(text, model):
    """
    OpenRouter has no count-tokens endpoint, so send the text with
    max_tokens=1 and read usage.prompt_tokens. This is a real (paid) call
    billed for the full input, so keep --count small.
    """
    import os
    import urllib.error
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY") or load_env().get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not found in environment or .env")
    body = json.dumps({
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": text}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return f"error {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
    return data.get("usage", {}).get("prompt_tokens", "no usage returned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="out")
    ap.add_argument("--count", type=int, default=0,
                    help="official token count for the N largest patients")
    ap.add_argument("--model", default="anthropic/claude-opus-5",
                    help="OpenRouter model slug")
    ap.add_argument("--with-notes", action="store_true",
                    help="also count flat+notes (a second, larger paid call)")
    args = ap.parse_args()

    out = Path(args.out)
    flat_dir = out / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)

    rows, texts = [], {}
    for f in sorted(Path(args.data_dir).glob("*.json")):
        if f.name.startswith(SHARED_PREFIXES):
            continue
        row, flat, notes = measure(f, flat_dir)
        rows.append(row)
        texts[f.name] = (flat, notes)
        print(f"{row['raw_mb']:>6} MB  flat~{row['flat_tok']:>8,} tok  "
              f"notes~{row['note_tok']:>8,} tok  {f.name}")

    rows.sort(key=lambda r: r["flat_tok"], reverse=True)

    with open(out / "sizes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if args.count:
        print(f"\nreal token counts via OpenRouter ({args.model})")
        for row in rows[:args.count]:
            flat, notes = texts[row["file"]]
            line = (f"  {row['file'][:40]:<40} flat {official_count(flat, args.model)}"
                    f" (est {row['flat_tok']:,})")
            if args.with_notes:
                line += f"   flat+notes {official_count(flat + chr(10) * 2 + notes, args.model)}"
            print(line, flush=True)

    flat_toks = [r["flat_tok"] for r in rows]
    note_toks = [r["note_tok"] for r in rows]
    print(f"\n{len(rows)} patients")
    print(f"flat tokens   median {statistics.median(flat_toks):,.0f}  max {max(flat_toks):,}")
    print(f"note tokens   median {statistics.median(note_toks):,.0f}  max {max(note_toks):,}")
    print(f"wrote {out / 'sizes.csv'} and {flat_dir}/")


if __name__ == "__main__":
    main()
