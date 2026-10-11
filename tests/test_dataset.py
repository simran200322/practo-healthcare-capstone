"""Unit tests for appointment dataset generation and validation."""
import pytest
from app.dataset import (
    generate_appointments,
    validate_dataset,
    CATEGORIES,
    STATUSES,
    DATASET_SEED,
    TOTAL_RECORDS,
)

def test_dataset_determinism():
    """Ensure that identical seeds yield strictly identical datasets."""
    run1 = generate_appointments(seed=DATASET_SEED, n_records=TOTAL_RECORDS)
    run2 = generate_appointments(seed=DATASET_SEED, n_records=TOTAL_RECORDS)
    assert run1 == run2, "Dataset generator must be strictly deterministic"

def test_dataset_size_and_schema():
    """Verify record count is >= 40 and schema has all required keys."""
    records = generate_appointments(seed=DATASET_SEED, n_records=TOTAL_RECORDS)
    assert len(records) >= 40
    
    required_keys = {
        "record_id",
        "category",
        "status",
        "consultation_fee_inr",
        "days_since_created",
        "follow_up_required"
    }
    
    for r in records:
        assert set(r.keys()) == required_keys
        assert r["record_id"].startswith("APP-")
        assert r["category"] in CATEGORIES
        assert r["status"] in STATUSES
        assert isinstance(r["consultation_fee_inr"], int)
        assert 500 <= r["consultation_fee_inr"] <= 2000
        assert isinstance(r["days_since_created"], int)
        assert 0 <= r["days_since_created"] <= 30
        assert isinstance(r["follow_up_required"], bool)

def test_category_and_status_constraints():
    """Verify each category has >=3 records, each status has >=1 record."""
    records = generate_appointments(seed=DATASET_SEED, n_records=TOTAL_RECORDS)
    metrics = validate_dataset(records)
    
    for cat in CATEGORIES:
        assert metrics["category_counts"][cat] >= 3, f"Category {cat} should have >= 3 records"
        
    for st in STATUSES:
        assert metrics["status_counts"][st] >= 1, f"Status {st} should have >= 1 record"

def test_follow_up_percentage_band():
    """Verify follow-up percentage is strictly between 10% and 30%."""
    records = generate_appointments(seed=DATASET_SEED, n_records=TOTAL_RECORDS)
    metrics = validate_dataset(records)
    assert 10.0 <= metrics["follow_up_percentage"] <= 30.0

def test_validator_raises_on_invalid_data():
    """Ensure validate_dataset catches out-of-spec datasets."""
    # Under 40 records
    with pytest.raises(ValueError, match="Total records must be >= 40"):
        validate_dataset([])
        
    # High follow-up percentage
    bad_records = [
        {
            "record_id": f"APP-{i}",
            "category": "General Medicine",
            "status": "Scheduled",
            "consultation_fee_inr": 500,
            "days_since_created": 5,
            "follow_up_required": True
        }
        for i in range(45)
    ]
    with pytest.raises(ValueError):
        validate_dataset(bad_records)
