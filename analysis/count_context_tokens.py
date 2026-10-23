"""
Real token counts for the largest rendered contexts, via OpenRouter.

Sends each context once with max_tokens=1 and reads usage.prompt_tokens, then
subtracts a one-character baseline call so the fixed message overhead is not
counted. You are billed for the input tokens (roughly $0.50 to $1 for Cole on
Opus), so --top stays small by default.

Needs OPENROUTER_API_KEY in .env (same folder) or the environment.

Usage:
  python count_context_tokens.py                 # largest 1 (Cole)
  python count_context_tokens.py --top 4
  python count_context_tokens.py --only Merlene
"""
import argparse
from pathlib import Path

from measure_size import official_count

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--top", type=int, default=1, help="count the N largest contexts")
ap.add_argument("--only", nargs="*", help="name prefixes instead of the largest")
ap.add_argument("--model", default="anthropic/claude-opus-5")
args = ap.parse_args()

files = sorted((Path(args.out) / "context").glob("*.txt"), key=lambda f: f.stat().st_size, reverse=True)
if args.only:
    files = [f for f in files if f.name.startswith(tuple(args.only))]
else:
    files = files[:args.top]

base = official_count("x", args.model)
if not isinstance(base, int):
    raise SystemExit(f"OpenRouter call failed: {base}")
print(f"model: {args.model}\n")
for f in files:
    text = f.read_text(encoding="utf-8")
    total = official_count(text, args.model)
    if not isinstance(total, int):
        print(f"FAILED {f.stem}: {total}")
        continue
    real = total - base
    print(f"{real:>9,} real tokens  (chars/4 estimate {len(text) // 4:,}, "
          f"ratio {real / (len(text) // 4):.2f})  {f.stem}")
