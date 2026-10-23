"""
Measure how many tokens dates cost in different formats, using OpenRouter.

OpenRouter has no free token counter, so each call sends the text with
max_tokens=1 and reads usage.prompt_tokens. Each format is measured as
(tokens with 500 dates) - (tokens with an empty message), divided by 500,
so the fixed overhead of a message cancels out. Total cost: well under $0.01.

Needs OPENROUTER_API_KEY in .env (same folder) or the environment.

Usage:
  python count_date_tokens.py
  python count_date_tokens.py --model anthropic/claude-opus-5
"""
import argparse
from datetime import date, timedelta

from measure_size import official_count

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="anthropic/claude-opus-5")
args = ap.parse_args()

dates = [date(2016, 8, 31) + timedelta(days=7 * i) for i in range(500)]
n = len(dates)

formats = {
    "date with value, ' · ' separator": " · ".join(f"{d.isoformat()} 9.31" for d in dates),
    "date only, ' · ' separator": " · ".join(d.isoformat() for d in dates),
    "date only, comma separator": ", ".join(d.isoformat() for d in dates),
    "grouped by year (2023: 01-04 · 01-11)": "\n".join(
        f"{y}: " + " · ".join(d.strftime("%m-%d") for d in dates if d.year == y)
        for y in sorted({d.year for d in dates})),
}

base = official_count("x", args.model)
if not isinstance(base, int):
    raise SystemExit(f"OpenRouter call failed: {base}")
print(f"model: {args.model}   (baseline message: {base} tokens)\n")
for label, text in formats.items():
    total = official_count(text, args.model)
    print(f"{(total - base) / n:5.2f} tokens per date   {label}")
