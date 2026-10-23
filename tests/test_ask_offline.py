"""The record-only pipeline in ask.py with a stand-in model: no API key, no cost.
Checks the prompt's six sections and the check-and-fix loop."""
import json

import pytest

import ask
from conftest import OUT, need


def test_prompt_has_the_six_sections_in_order():
    s = ask.build_system()
    heads = ["# Role", "# Task", "# Input context", "# Define success", "# Define failure", "# Define the output"]
    positions = [s.index(h) for h in heads]
    assert positions == sorted(positions)
    assert "You have no tools" in s


def scripted(replies):
    """A stand-in for call_model that returns the given replies in order."""
    sent = []

    def fake(messages, model, key, tools=None):
        sent.append(messages)
        return {"choices": [{"message": {"content": replies[len(sent) - 1]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    return fake, sent


@pytest.fixture
def patient():
    from common import load_patients
    need(OUT / "context")
    stem = sorted((OUT / "context").glob("*.ids.json"))[0].name.replace(".ids.json", "")
    return next(p for p in load_patients(OUT.parent / "patients.csv", OUT) if p["stem"] == stem)


def test_passes_first_time(monkeypatch, patient):
    ctx = (OUT / "context" / f"{patient['stem']}.txt").read_text(encoding="utf-8")
    line = next(l for l in ctx.splitlines() if l.startswith("[PT1] "))
    reply = json.dumps({"answers": [{"question": "who", "status": "answered", "answer": "See [PT1].",
                                     "evidence": [{"id": "PT1", "quote": line[6:26]}], "inferences": [],
                                     "checked": None}]})
    fake, sent = scripted([reply])
    monkeypatch.setattr(ask, "call_model", fake)
    t = ask.ask(patient, "Who is this?", OUT, "m", "k")
    assert t["verified"] and len(sent) == 1 and not t["repairs"]


def test_bad_json_gets_fixed(monkeypatch, patient):
    ctx = (OUT / "context" / f"{patient['stem']}.txt").read_text(encoding="utf-8")
    line = next(l for l in ctx.splitlines() if l.startswith("[PT1] "))
    reply = json.dumps({"answers": [{"question": "who", "status": "answered", "answer": "See [PT1].",
                                     "evidence": [{"id": "PT1", "quote": line[6:26]}], "inferences": [],
                                     "checked": None}]})
    fake, sent = scripted(["not json", reply])
    monkeypatch.setattr(ask, "call_model", fake)
    t = ask.ask(patient, "Who is this?", OUT, "m", "k")
    assert t["verified"] and len(t["repairs"]) == 1
    assert "failed these checks" in json.dumps(sent[1][-1])  # the fix request went back to the model


def test_gives_up_after_two_fixes(monkeypatch, patient):
    fake, sent = scripted(["not json"] * 3)
    monkeypatch.setattr(ask, "call_model", fake)
    t = ask.ask(patient, "Who is this?", OUT, "m", "k")
    assert not t["verified"] and len(sent) == 1 + ask.MAX_REPAIRS
