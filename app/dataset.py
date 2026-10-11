"""Deterministic appointment dataset generator and validator for Practo Healthcare Capstone."""
import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from app.config import APPOINTMENTS_FILE

# Fixed seed for perfect determinism and reproducibility
DATASET_SEED = 42
TOTAL_RECORDS = 50

# Given category vocabulary (all 5 required)
CATEGORIES = [
    "General Medicine",
    "Cardiology",
    "Dermatology",
    "Pediatrics",
    "Orthopedics"
]

# Given status vocabulary (all 5 required)
STATUSES = [
    "Scheduled",
    "Completed",
    "Cancelled",
    "No-Show",
    "Rescheduled"
]

# Realistic specialty-based consultation fee ranges in INR
# Reasoning: Consultation fees range from 500 to 2000 INR, mirroring verified private outpatient
# clinic and specialist consultation tariffs across metro and tier-1 Indian healthcare networks.
FEE_RANGES_INR = {
    "General Medicine": (500, 800, 50),
    "Pediatrics": (600, 900, 50),
    "Dermatology": (700, 1200, 100),
    "Orthopedics": (800, 1500, 100),
    "Cardiology": (1000, 2000, 100)
}

CATEGORY_WEIGHTS = [0.30, 0.15, 0.20, 0.20, 0.15]
STATUS_WEIGHTS = [0.35, 0.30, 0.12, 0.10, 0.13]
FOLLOW_UP_PROBABILITY = 0.20  # Calibrated to reliably produce 10%-30% follow-up rate


def generate_appointments(
    seed: int = DATASET_SEED,
    n_records: int = TOTAL_RECORDS
) -> List[Dict[str, Any]]:
    """Generate deterministic synthetic appointment records meeting all capstone constraints."""
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []

    for i in range(1, n_records + 1):
        record_id = f"APP-{i:04d}"
        
        # Sample category using realistic outpatient distribution
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        
        # Sample status
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        
        # Realistic fee sampling in 50/100 INR increments based on specialty
        min_fee, max_fee, step = FEE_RANGES_INR[category]
        possible_fees = list(range(min_fee, max_fee + 1, step))
        fee = rng.choice(possible_fees)
        
        # Days since creation (0-30 integer)
        days_since_created = rng.randint(0, 30)
        
        # Follow-up required boolean
        follow_up_required = rng.random() < FOLLOW_UP_PROBABILITY
        
        record = {
            "record_id": record_id,
            "category": category,
            "status": status,
            "consultation_fee_inr": fee,
            "days_since_created": days_since_created,
            "follow_up_required": follow_up_required
        }
        records.append(record)

    return records


def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate all structural criteria required by Capstone Part 1 Task 1."""
    total_count = len(records)
    if total_count < 40:
        raise ValueError(f"Constraint Failed: Total records must be >= 40, got {total_count}")

    # Category counts
    category_counts: Dict[str, int] = {c: 0 for c in CATEGORIES}
    # Status counts
    status_counts: Dict[str, int] = {s: 0 for s in STATUSES}
    follow_up_count = 0

    for r in records:
        category = r["category"]
        status = r["status"]
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts[status] = 1

        if r["follow_up_required"]:
            follow_up_count += 1

    # Check category constraint: each required category >= 3
    for cat in CATEGORIES:
        count = category_counts.get(cat, 0)
        if count < 3:
            raise ValueError(f"Constraint Failed: Category '{cat}' has {count} records (required >= 3)")

    # Check status constraint: each required status >= 1
    for st in STATUSES:
        count = status_counts.get(st, 0)
        if count < 1:
            raise ValueError(f"Constraint Failed: Status '{st}' has {count} records (required >= 1)")

    # Check follow-up percentage constraint: 10% to 30%
    follow_up_pct = (follow_up_count / total_count) * 100.0
    if not (10.0 <= follow_up_pct <= 30.0):
        raise ValueError(
            f"Constraint Failed: follow_up_required percentage is {follow_up_pct:.2f}% "
            f"(required strictly between 10.0% and 30.0%)"
        )

    return {
        "total_records": total_count,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "follow_up_count": follow_up_count,
        "follow_up_percentage": round(follow_up_pct, 2)
    }


def save_dataset(records: List[Dict[str, Any]], filepath: Path = APPOINTMENTS_FILE) -> None:
    """Save records to target JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_dataset(filepath: Path = APPOINTMENTS_FILE) -> List[Dict[str, Any]]:
    """Load records from target JSON file."""
    if not filepath.exists():
        records = generate_appointments()
        save_dataset(records, filepath)
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Pre-generated deterministic constant instance
APPOINTMENTS = generate_appointments()
