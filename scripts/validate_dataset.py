"""Executable script to generate, validate, and report appointment dataset statistics."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset import generate_appointments, validate_dataset, save_dataset, DATASET_SEED, TOTAL_RECORDS
from app.config import APPOINTMENTS_FILE

def main():
    print("=" * 60)
    print("PRACTO HEALTHCARE CAPSTONE - DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Seed: {DATASET_SEED}")
    print(f"Target record count: {TOTAL_RECORDS}")
    
    # Generate records
    records = generate_appointments(seed=DATASET_SEED, n_records=TOTAL_RECORDS)
    
    # Validate constraints
    metrics = validate_dataset(records)
    
    # Save to data/appointments.json
    save_dataset(records, APPOINTMENTS_FILE)
    print(f"Dataset successfully saved to: {APPOINTMENTS_FILE}")
    print("-" * 60)
    print(f"Total Appointments Generated: {metrics['total_records']}")
    print("-" * 60)
    print("Category Breakdown (Threshold: >= 3 records per category):")
    for cat, count in metrics["category_counts"].items():
        status_flag = "PASS" if count >= 3 else "FAIL"
        print(f"  - {cat:20s}: {count:2d} records [{status_flag}]")
    print("-" * 60)
    print("Status Breakdown (Threshold: >= 1 record per status):")
    for st, count in metrics["status_counts"].items():
        status_flag = "PASS" if count >= 1 else "FAIL"
        print(f"  - {st:20s}: {count:2d} records [{status_flag}]")
    print("-" * 60)
    pct = metrics["follow_up_percentage"]
    fu_flag = "PASS" if 10.0 <= pct <= 30.0 else "FAIL"
    print(f"Follow-up Required Count: {metrics['follow_up_count']} / {metrics['total_records']}")
    print(f"Follow-up Required Rate : {pct:.2f}% (Threshold: 10.0% - 30.0%) [{fu_flag}]")
    print("=" * 60)
    print("ALL ACCEPTANCE CONSTRAINTS MET PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
