"""The eval scorer in eval_generic.py: number matching and the unlabeled-verdict check."""
from eval_generic import ABOUT_THE_RECORD, VERDICT, has, norm_answer, recorded_part


def leaks(text):
    t = recorded_part(norm_answer(text))
    return sorted({m.group(0) for m in VERDICT.finditer(t)
                   if not ABOUT_THE_RECORD.search(t[max(0, m.start() - 70):m.start()])})


def test_database_numbers_match_the_key():
    assert has(norm_answer("Heart rate 85.0 on 2026-01-26"), "85")
    assert has(norm_answer("Calcium 9.980 mg/dL"), "9.98")


def test_whole_number_match_only():
    assert not has("chloride 104.55", "104")


def test_labeled_verdict_is_not_a_leak():
    assert leaks("88 to 99 (inferred: rising)") == []


def test_unlabeled_verdict_is_a_leak():
    assert leaks("Systolic rose from 88 to 99.") == ["rose"]
    assert leaks("No clear worsening.") == ["worsening"]


def test_statements_about_the_record_are_not_leaks():
    assert leaks("The record does not label these as normal or abnormal.") == []
    assert leaks("No reading is recorded as abnormal in the record.") == []


def test_exemption_stays_in_its_clause():
    assert leaks("No reference range is given, values look normal.") == ["normal"]
