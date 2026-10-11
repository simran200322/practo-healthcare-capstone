"""Healthcare domain tools: check_appointment_status with designed escalation scoring."""
from typing import Dict, Any, Optional
from app.config import ESCALATION_THRESHOLD, FOLLOW_UP_WEIGHT, RECENCY_WEIGHT
from app.dataset import load_dataset


def calculate_escalation_score(
    follow_up_required: bool,
    days_since_created: int
) -> float:
    """Calculate normalized clinical escalation score in [0, 1].
    
    Formula:
      escalation_score = (0.60 * follow_up_required) + (0.40 * (days_since_created / 30.0))
      
    Justification:
      Clinical follow-up necessity contributes 60% of escalation priority.
      Appointment staleness/recency contributes 40% (normalized over 30 days).
      In our dataset distribution, routine inquiries without follow-up needs peak at 0.40
      (80th percentile of days_since_created). Hence, any case with follow_up_required=True
      immediately scores >= 0.60, cleanly exceeding the 0.50 escalation threshold.
    """
    recency_norm = min(max(days_since_created / 30.0, 0.0), 1.0)
    score = (0.60 * float(follow_up_required)) + (0.40 * recency_norm)
    return round(score, 4)


def check_appointment_status(record_id: str) -> Dict[str, Any]:
    """Look up an appointment by its record ID, returning status, fee, and escalation score.
    
    Args:
        record_id: Synthetic appointment identifier formatted as 'APP-XXXX' (e.g. 'APP-0001').
        
    Returns:
        A dictionary containing record_id, category, status, consultation_fee_inr,
        days_since_created, follow_up_required, escalation_score, and escalation_recommended.
    """
    clean_id = record_id.strip().upper()
    # Normalize if user entered just integer like "1" or "0001"
    if clean_id.isdigit():
        clean_id = f"APP-{int(clean_id):04d}"

    records = load_dataset()
    match: Optional[Dict[str, Any]] = None
    for r in records:
        if r["record_id"].upper() == clean_id:
            match = r
            break

    if not match:
        return {
            "found": False,
            "record_id": clean_id,
            "status": "Not Found",
            "consultation_fee_inr": 0,
            "escalation_score": 0.0,
            "escalation_recommended": False,
            "message": f"Appointment record '{clean_id}' was not found in the clinic database."
        }

    score = calculate_escalation_score(
        follow_up_required=match["follow_up_required"],
        days_since_created=match["days_since_created"]
    )
    should_escalate = score >= ESCALATION_THRESHOLD

    msg = (
        f"Appointment {match['record_id']} is currently '{match['status']}' in {match['category']}. "
        f"Consultation fee: INR {match['consultation_fee_inr']}. "
        f"Follow-up required: {match['follow_up_required']}. "
        f"Escalation score: {score:.2f} ({'ESCALATE' if should_escalate else 'ROUTINE'})."
    )

    return {
        "found": True,
        "record_id": match["record_id"],
        "category": match["category"],
        "status": match["status"],
        "consultation_fee_inr": match["consultation_fee_inr"],
        "days_since_created": match["days_since_created"],
        "follow_up_required": match["follow_up_required"],
        "escalation_score": score,
        "escalation_recommended": should_escalate,
        "message": msg
    }
