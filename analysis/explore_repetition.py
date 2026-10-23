"""
Find repetitive or information-sparse data in the grouped records, and test
whether "write a repeated value once" (rule 1) is safe.

Reads out/grouped/*.json (from group_record.py). Prints a report and writes
out/explore/repetition_report.txt.

Usage:
  python explore_repetition.py
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def line_len(readings):
    return len(" · ".join(f"{d} {fmt(v)}" for d, v in readings))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out)
    records = {p.stem: json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((out / "grouped").glob("*.json"))}
    report = []

    def say(s=""):
        print(s)
        report.append(s)

    # ------------------------------------------------ 1. rule 1 edge cases
    say("== 1. Edge cases for rule 1 (write a repeated value once)")
    same_day_conflict, same_day_dup, missing_date, mixed_types, fmt_collision = 0, 0, 0, 0, 0
    examples = defaultdict(list)
    for name, rec in records.items():
        for o in rec["observations"]:
            by_day = defaultdict(list)
            kinds = set()
            for d, v in o["readings"]:
                by_day[d].append(v)
                kinds.add(type(v).__name__)
                if d is None:
                    missing_date += 1
            for d, vals in by_day.items():
                if len(vals) > 1:
                    if len(set(map(str, vals))) > 1:
                        same_day_conflict += 1
                        if len(examples["conflict"]) < 4:
                            examples["conflict"].append(f"{name[:18]} | {o['display'][:45]} | {d}: {vals}")
                    else:
                        same_day_dup += 1
            if len(kinds - {"NoneType"}) > 1:
                mixed_types += 1
                if len(examples["mixed"]) < 3:
                    examples["mixed"].append(f"{name[:18]} | {o['display'][:45]} | {kinds}")
            # two different raw values that print identically would merge wrongly
            printed = defaultdict(set)
            for _, v in o["readings"]:
                printed[fmt(v)].add(repr(v))
            fmt_collision += sum(1 for s in printed.values() if len(s) > 1)
    say(f"  same lab, same day, different values: {same_day_conflict} cases")
    for e in examples["conflict"]:
        say(f"     e.g. {e}")
    say(f"  same lab, same day, identical value (true duplicates): {same_day_dup} cases")
    say(f"  readings with no date: {missing_date}")
    say(f"  labs mixing numbers and text: {mixed_types}")
    for e in examples["mixed"]:
        say(f"     e.g. {e}")
    say(f"  different stored values that print the same (would merge wrongly): {fmt_collision}")

    # --------------------------------------- 2. constant and near-constant series
    say("\n== 2. How repetitive are lab and survey series?")
    const_chars, total_obs_chars, repeat_share = 0, 0, Counter()
    const_names, n_readings_total, n_repeats_total = Counter(), 0, 0
    for name, rec in records.items():
        for o in rec["observations"]:
            r = o["readings"]
            ln = line_len(r)
            total_obs_chars += ln
            vals = [fmt(v) for _, v in r]
            repeats = sum(1 for a, b in zip(vals, vals[1:]) if a == b)
            n_readings_total += len(vals)
            n_repeats_total += repeats
            if len(r) >= 3 and len(set(vals)) == 1:
                const_chars += ln
                const_names[o["display"]] += 1
    say(f"  readings equal to the previous reading: {n_repeats_total:,} of {n_readings_total:,}"
        f" ({n_repeats_total / n_readings_total:.0%})")
    say(f"  series with 3+ readings that never change: {const_chars / total_obs_chars:.0%} of all lab/survey characters")
    say("  most common never-changing series (patients):")
    for n, c in const_names.most_common(10):
        say(f"     {c:>3}  {n}")

    # --------------------------------- 3. two codes carrying the same numbers
    say("\n== 3. Different lab codes carrying the same (date, value) pairs")
    pair_overlap = Counter()
    for name, rec in records.items():
        series = {o["display"]: {(d, fmt(v)) for d, v in o["readings"]}
                  for o in rec["observations"] if len(o["readings"]) >= 5}
        names = sorted(series)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = len(series[a] & series[b])
                if inter >= 5 and inter / min(len(series[a]), len(series[b])) >= 0.8:
                    pair_overlap[(a, b)] += 1
    for (a, b), n in pair_overlap.most_common(8):
        say(f"  {n:>3} patients: {a[:55]}  ==  {b[:55]}")
    if not pair_overlap:
        say("  none found")

    # ------------------------- 4. procedures that always happen together
    say("\n== 4. Procedures and other items that always occur on the same dates")
    bundles = Counter()
    for name, rec in records.items():
        by_dates = defaultdict(list)
        for p in rec["procedures"]:
            if len(p["dates"]) >= 2:
                by_dates[tuple(p["dates"])].append(p["display"])
        for dates, names in by_dates.items():
            if len(names) >= 3:
                bundles[tuple(sorted(names))] += 1
    for names, n in bundles.most_common(5):
        say(f"  {n:>3} patients, {len(names)} procedures always together: "
            + "; ".join(x.replace(' (procedure)', '')[:40] for x in names[:5])
            + (" ..." if len(names) > 5 else ""))

    # imaging that duplicates a procedure
    dup_img = tot_img = 0
    for rec in records.values():
        procs = {(p["code"], tuple(p["dates"])) for p in rec["procedures"]}
        for im in rec["imaging"]:
            tot_img += 1
            dup_img += (im["code"], tuple(im["dates"])) in procs
    say(f"  imaging lines whose code and dates exactly match a procedure line: {dup_img} of {tot_img}")

    # ---------------------------------------------- 5. where the characters go
    say("\n== 5. Where the text goes (all patients, share of characters)")
    share = Counter()
    for name in records:
        for line in (out / "grouped" / f"{name}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                share[line.split(" | ")[0]] += len(line)
    total = sum(share.values())
    for label, c in share.most_common(12):
        say(f"  {c / total:>5.1%}  {label}")
    # label text vs data: how much is the item name itself
    name_chars = 0
    for rec in records.values():
        for o in rec["observations"]:
            name_chars += len(o["display"] or "") + len(o.get("part_of") or "")
    say(f"  lab/survey names and 'part of' labels alone: {name_chars / total:.1%} of all characters")

    # ------------------------------------------ 6. long free text in values
    say("\n== 6. Longest coded or text values (sparse words per fact)")
    texts = Counter()
    for rec in records.values():
        for o in rec["observations"]:
            for _, v in o["readings"]:
                if isinstance(v, str) and len(v) > 40:
                    texts[v] += 1
    for t, n in texts.most_common(6):
        say(f"  {n:>5}x  {t[:100]}")

    (out / "explore").mkdir(parents=True, exist_ok=True)
    (out / "explore" / "repetition_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'explore' / 'repetition_report.txt'}")


if __name__ == "__main__":
    main()
