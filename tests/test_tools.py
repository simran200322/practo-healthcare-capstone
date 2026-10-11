"""Unit tests for check_appointment_status and escalation scoring."""
import pytest
from app.tools import check_appointment_status, calculate_escalation_score
from app.config import ESCALATION_THRESHOLD

def test_escalation_score_formula_bounds():
    # Min possible score: follow_up False, days=0 -> 0.0
    assert calculate_escalation_score(False, 0) == 0.0
    # Max possible score: follow_up True, days=30 -> 1.0
    assert calculate_escalation_score(True, 30) == 1.0
    # Intermediate cases
    score_fu_true = calculate_escalation_score(True, 0)
    assert score_fu_true == 0.60
    assert score_fu_true >= ESCALATION_THRESHOLD

    score_fu_false_high_days = calculate_escalation_score(False, 30)
    assert score_fu_false_high_days == 0.40
    assert score_fu_false_high_days < ESCALATION_THRESHOLD

def test_appointment_status_lookup_success():
    res = check_appointment_status("APP-0001")
    assert res["found"] is True
    assert res["record_id"] == "APP-0001"
    assert "status" in res
    assert res["consultation_fee_inr"] > 0
    assert 0.0 <= res["escalation_score"] <= 1.0
    assert isinstance(res["escalation_recommended"], bool)

def test_appointment_status_lookup_missing():
    res = check_appointment_status("APP-9999")
    assert res["found"] is False
    assert res["status"] == "Not Found"
    assert res["consultation_fee_inr"] == 0
    assert res["escalation_recommended"] is False
