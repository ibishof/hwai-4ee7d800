"""
How much do answers vary between runs of the same system on the same questions?

  python variance.py out\\evals\\generic_questions_context_<t1>.json out\\evals\\generic_questions_context_<t2>.json out\\evals\\generic_questions_context_<t3>.json

For each question, compared across every pair of runs:
  cited IDs     Jaccard overlap of the record line IDs cited (evidence and
                premises). Same IDs = the same facts were used. Tool result IDs
                (SQL1, NOTES1) are left out: they are numbered per run.
  values        Jaccard overlap of the numbers and dates stated in the answer
                (outside inferences, 85.0 counted as 85). This includes extra
                context the model chose to add (a prior reading, a range), so it
                runs lower than the required facts, which "pass" covers.
  status        share of run pairs that gave the same status(es).
  pass          share of run pairs with the same code-graded result.
  cosine        similarity of the full answer text, wording and all, embedded
                with Voyage AI's voyage-4-large. Needs VOYAGE_API_KEY in .env;
                skipped with a note if missing. A first filter for "how
                differently is it worded", not a verdict.

1.0 everywhere = identical across runs. Writes out/evals/variance.csv.
"""
import csv
import itertools
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from common import load_env
from eval_generic import norm_answer, recorded_part, score
from verify_answer import CITATION, DATE, NUMBER

LINE_ID = re.compile(r"^[A-Z]{1,2}\d+$")
VOYAGE_MODEL = "voyage-4-large"
# Atlas model keys use ai.mongodb.com; keys from the older Voyage dashboard use api.voyageai.com.
VOYAGE_URLS = ["https://ai.mongodb.com/v1/embeddings", "https://api.voyageai.com/v1/embeddings"]


def embed(texts, key):
    """Voyage embeddings for texts (unit length, so a dot product is the cosine).
    Tries each endpoint until one accepts the key."""
    out = []
    for start in range(0, len(texts), 100):  # stay well under the per-request input limit
        batch = [t or " " for t in texts[start:start + 100]]
        body = json.dumps({"input": batch, "model": VOYAGE_MODEL, "input_type": "document"}).encode()
        last = None
        for url in VOYAGE_URLS:
            req = urllib.request.Request(url, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())["data"]
                out += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
                break
            except urllib.error.HTTPError as e:
                last = f"{url}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:200]}"
        else:
            raise RuntimeError(last)
    return out


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def cited_ids(result):
    ids = set()
    for a in (result or {}).get("answers", []):
        if not isinstance(a, dict):
            continue
        for e in (a.get("evidence") or []) + [p for i in (a.get("inferences") or []) if isinstance(i, dict)
                                              for p in (i.get("premises") or [])]:
            cid = str(e.get("id", "")).strip("[] ") if isinstance(e, dict) else ""
            if LINE_ID.match(cid) and not cid.startswith(("SQL", "NOTES")):
                ids.add(cid)
    return ids


def values(answer):
    text = CITATION.sub(" ", norm_answer(recorded_part(answer)))
    dates = set(DATE.findall(text))
    nums = set(NUMBER.findall(DATE.sub(" ", text)))
    return dates | nums


def jaccard(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def mean(xs):
    xs = [x for x in xs if x == x]  # skip NaN (not applicable)
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x):
    return f"{'-':>11}" if x != x else f"{x:>11.2f}"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    files = sys.argv[1:]
    if len(files) < 2:
        raise SystemExit("Give two or more results files from the same system.")
    runs = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        rows = {}
        for r in d["results"]:
            s = score(r, {"answer_text": r["answer"], "result": r["result"]})
            rows[(r["patient"], r["kind"])] = {"answer": r["answer"], "result": r["result"], "pass": s["pass"],
                                                "statuses": tuple(r.get("statuses") or [])}
        runs.append(rows)
    keys = [k for k in runs[0] if all(k in r for r in runs)]

    vectors = None
    key = os.environ.get("VOYAGE_API_KEY") or load_env().get("VOYAGE_API_KEY")
    if not key:
        print("(cosine skipped: VOYAGE_API_KEY not found in environment or .env)\n")
    else:
        texts = [r[k]["answer"] for k in keys for r in runs]
        try:
            flat = embed(texts, key)
            vectors = {k: flat[i * len(runs):(i + 1) * len(runs)] for i, k in enumerate(keys)}
        except (RuntimeError, urllib.error.URLError, OSError) as e:
            print(f"(cosine skipped: Voyage request failed: {e})\n")

    table = []
    for k in keys:
        answers = [r[k] for r in runs]
        pairs = list(itertools.combinations(answers, 2))
        row = {"patient": k[0], "kind": k[1],
               "cited IDs": mean(jaccard(cited_ids(a["result"]), cited_ids(b["result"])) for a, b in pairs),
               "values": mean(jaccard(values(a["answer"]), values(b["answer"])) for a, b in pairs),
               "status": mean(a["statuses"] == b["statuses"] for a, b in pairs),
               # general questions are not code-graded, so pass agreement does not apply
               "pass": float("nan") if k[1] == "general" else mean(a["pass"] == b["pass"] for a, b in pairs)}
        if vectors:
            row["cosine"] = mean(cosine(a, b) for a, b in itertools.combinations(vectors[k], 2))
        table.append(row)

    cols = ["cited IDs", "values", "status", "pass"] + (["cosine"] if vectors else [])
    print(f"{len(files)} runs, {len(keys)} questions, {len(files) * (len(files) - 1) // 2} run pairs per question\n")
    print(f"{'question':<22}" + "".join(f"{c:>11}" for c in cols))
    for row in table:
        print(f"{row['patient'].split()[0] + ' ' + row['kind']:<22}" + "".join(fmt(row[c]) for c in cols))
    print(f"{'ALL (mean)':<22}" + "".join(fmt(mean(r[c] for r in table)) for c in cols))
    print("\nBy question kind")
    for kind in dict.fromkeys(r["kind"] for r in table):
        sub = [r for r in table if r["kind"] == kind]
        print(f"  {kind:<20}" + "".join(fmt(mean(r[c] for r in sub)) for c in cols))

    out = Path(files[0]).parent / "variance.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient", "kind"] + cols)
        w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in table)
    print(f"\ntable: {out}")


if __name__ == "__main__":
    main()
