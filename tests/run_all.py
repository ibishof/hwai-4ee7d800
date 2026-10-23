"""
Run every test with plain Python, no pytest needed:

  python tests\\run_all.py

(If pytest is installed, `python -m pytest tests` runs the same tests.)
Data tests need the built outputs in out/ and skip themselves if missing.
"""
import importlib
import inspect
import sys
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

try:
    import pytest
    Skip = pytest.skip.Exception
except ImportError:  # a minimal stand-in for the few pytest features the tests use
    class Skip(Exception):
        pass

    def _skip(msg=""):
        raise Skip(msg)

    def _fixture(f):
        f._fixture = True
        return f

    def _parametrize(name, values):
        def deco(f):
            f._params = (name, list(values))
            return f
        return deco

    pytest = types.ModuleType("pytest")
    pytest.skip, pytest.fixture = _skip, _fixture
    pytest.mark = types.SimpleNamespace(parametrize=_parametrize)
    sys.modules["pytest"] = pytest


class MonkeyPatch:
    def __init__(self):
        self.undo = []

    def setattr(self, obj, name, value):
        self.undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self):
        for obj, name, value in reversed(self.undo):
            setattr(obj, name, value)


def cases(f):
    """One run per parameter value (pytest or stand-in marks), else one run."""
    marks = getattr(f, "pytestmark", [])
    for m in marks:
        if m.name == "parametrize":
            name, values = m.args[0], m.args[1]
            return [{name: v} for v in values]
    if hasattr(f, "_params"):
        name, values = f._params
        return [{name: v} for v in values]
    return [{}]


def main():
    passed = failed = skipped = 0
    started = time.monotonic()
    for path in sorted(HERE.glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        fixtures = {n: getattr(f, "__wrapped__", f) for n, f in vars(mod).items()
                    if getattr(f, "_fixture", False) or getattr(f, "_pytestfixturefunction", None)
                    or type(f).__name__ == "FixtureFunctionDefinition"}
        tests = [(n, f) for n, f in list(vars(mod).items()) if n.startswith("test_") and callable(f)]
        for name, f in tests:
            for params in cases(f):
                mp = MonkeyPatch()
                label = f"{path.stem}::{name}" + (f"[{list(params.values())[0]}]" if params else "")
                try:
                    args = {}
                    for p in inspect.signature(f).parameters:
                        if p in params:
                            args[p] = params[p]
                        elif p == "monkeypatch":
                            args[p] = mp
                        elif p in fixtures:
                            args[p] = fixtures[p]()
                    f(**args)
                    passed += 1
                except Skip as e:
                    skipped += 1
                    print(f"SKIP  {label}: {e}")
                except Exception:
                    failed += 1
                    print(f"FAIL  {label}")
                    traceback.print_exc(limit=3)
                finally:
                    mp.restore()
        print(f"  done {path.name}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped in {time.monotonic() - started:.0f}s")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
