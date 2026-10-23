"""The answer checks in verify_answer.py, one rule at a time. No model, no data."""
from verify_answer import verify

LINES = {"L1": "[L1] Hemoglobin A1c (LOINC 4548-4) % | n=3 | latest 6.31 on 2025-09-29 | 2024-09-23 6.15 · 2025-09-29 6.31",
         "L2": "[L2] Chloride (LOINC 2075-0) mmol/L | 2021-10-06 104.55",
         "A1": "[A1] none recorded"}


def answer(text, evidence, status="answered", inferences=(), checked=None):
    return {"answers": [{"question": "q", "status": status, "answer": text, "evidence": list(evidence),
                         "inferences": list(inferences), "checked": checked}]}


def ev(cid, quote):
    return {"id": cid, "quote": quote}


def test_good_answer_passes():
    a = answer("Latest A1c 6.31% on 2025-09-29 [L1].", [ev("L1", "latest 6.31 on 2025-09-29")])
    assert verify(a, LINES, {}) == []


def test_quote_must_be_verbatim():
    a = answer("Latest A1c 6.31% on 2025-09-29 [L1].", [ev("L1", "latest A1c 6.31 on 2025-09-29")])
    assert any("not in L1" in p for p in verify(a, LINES, {}))


def test_spliced_quote_fails():
    a = answer("6.15 then 6.31 [L1].", [ev("L1", "2024-09-23 6.15 ... 6.31")])
    assert any("not in L1" in p for p in verify(a, LINES, {}))


def test_unknown_id_fails():
    a = answer("x [L9].", [ev("L9", "x")])
    assert any("does not exist" in p for p in verify(a, LINES, {}))


def test_number_must_be_quoted():
    a = answer("Latest A1c 6.4% [L1].", [ev("L1", "latest 6.31 on 2025-09-29")])
    assert any("number 6.4" in p for p in verify(a, LINES, {}))


def test_whole_number_does_not_match_inside_decimal():
    # 104 is not in "104.55": the model cannot round a value and pass
    a = answer("Chloride was 104 on 2021-10-06 [L2].", [ev("L2", "2021-10-06 104.55")])
    assert any("number 104" in p for p in verify(a, LINES, {}))


def test_date_must_be_quoted():
    a = answer("Latest A1c 6.31% on 2025-10-01 [L1].", [ev("L1", "latest 6.31 on 2025-09-29")])
    assert any("date 2025-10-01" in p for p in verify(a, LINES, {}))


def test_marked_inference_needs_entry():
    a = answer("A1c 6.15 then 6.31 [L1] (inferred: rising).", [ev("L1", "2024-09-23 6.15 · 2025-09-29 6.31")])
    assert any("no inference entry" in p for p in verify(a, LINES, {}))


def test_inference_with_premise_passes():
    inf = {"claim": "rising", "premises": [ev("L1", "2024-09-23 6.15 · 2025-09-29 6.31")],
           "assumption": "A later value higher than an earlier one is a rise."}
    a = answer("A1c 6.15 then 6.31 [L1] (inferred: rising).", [ev("L1", "2024-09-23 6.15 · 2025-09-29 6.31")],
               inferences=[inf])
    assert verify(a, LINES, {}) == []


def test_inference_entry_must_be_marked():
    inf = {"claim": "rising", "premises": [ev("L1", "2024-09-23 6.15")], "assumption": "x"}
    a = answer("A1c 6.15 then 6.31 [L1].", [ev("L1", "2024-09-23 6.15 · 2025-09-29 6.31")], inferences=[inf])
    assert any("not marked" in p for p in verify(a, LINES, {}))


def test_calculated_number_must_be_inferred():
    a = answer("It rose by 0.16 [L1].", [ev("L1", "2024-09-23 6.15 · 2025-09-29 6.31")])
    assert any("number 0.16" in p for p in verify(a, LINES, {}))


def test_answered_needs_evidence():
    a = answer("No allergies.", [])
    assert any("needs at least one evidence" in p for p in verify(a, LINES, {}))


def test_not_recorded_needs_checked():
    a = answer("No TSH is recorded.", [], status="not_recorded")
    assert any("needs 'checked'" in p for p in verify(a, LINES, {}))
    a = answer("No TSH is recorded.", [], status="not_recorded", checked="Lab lines L1-L2")
    assert verify(a, LINES, {}) == []


def test_bad_status_fails():
    a = answer("x [A1].", [ev("A1", "none recorded")], status="probably")
    assert any("status must be one of" in p for p in verify(a, LINES, {}))
