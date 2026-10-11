"""Knowledge base loader, metadata registry, and coverage validator."""
from pathlib import Path
from typing import List, Dict, Any
from app.config import KB_DIR

REQUIRED_TOPICS = [
    "appointment-booking policy",
    "cancellation/rescheduling window",
    "consultation-fee structure by specialty",
    "insurance-claim process",
    "prescription-refill policy",
    "lab-test turnaround times",
    "telemedicine eligibility",
    "emergency-visit protocol",
    "patient-data privacy policy",
    "follow-up-visit discount policy",
    "second-opinion process",
    "home-visit eligibility",
]

DOC_METADATA_MAP = {
    "01_appointment_booking.txt": {
        "id": "KB-DOC-001",
        "topic": "appointment-booking policy",
        "title": "Appointment Booking Policy",
    },
    "02_cancellation_rescheduling.txt": {
        "id": "KB-DOC-002",
        "topic": "cancellation/rescheduling window",
        "title": "Cancellation and Rescheduling Window Policy",
    },
    "03_consultation_fee_structure.txt": {
        "id": "KB-DOC-003",
        "topic": "consultation-fee structure by specialty",
        "title": "Consultation Fee Structure by Specialty",
    },
    "04_insurance_claims.txt": {
        "id": "KB-DOC-004",
        "topic": "insurance-claim process",
        "title": "Insurance Claim and Reimbursement Process",
    },
    "05_prescription_refills.txt": {
        "id": "KB-DOC-005",
        "topic": "prescription-refill policy",
        "title": "Prescription Refill Policy",
    },
    "06_lab_test_turnaround.txt": {
        "id": "KB-DOC-006",
        "topic": "lab-test turnaround times",
        "title": "Diagnostic Lab Test Turnaround Times",
    },
    "07_telemedicine_eligibility.txt": {
        "id": "KB-DOC-007",
        "topic": "telemedicine eligibility",
        "title": "Telemedicine Consultation Eligibility",
    },
    "08_emergency_visit_protocol.txt": {
        "id": "KB-DOC-008",
        "topic": "emergency-visit protocol",
        "title": "Emergency Visit and Triage Protocol",
    },
    "09_patient_data_privacy.txt": {
        "id": "KB-DOC-009",
        "topic": "patient-data privacy policy",
        "title": "Patient Data Privacy and Security Policy",
    },
    "10_follow_up_discount.txt": {
        "id": "KB-DOC-010",
        "topic": "follow-up-visit discount policy",
        "title": "Follow-Up Visit Discount Policy",
    },
    "11_second_opinion_process.txt": {
        "id": "KB-DOC-011",
        "topic": "second-opinion process",
        "title": "Specialist Second Opinion Process",
    },
    "12_home_visit_eligibility.txt": {
        "id": "KB-DOC-012",
        "topic": "home-visit eligibility",
        "title": "Home Visit and Specimen Collection Eligibility",
    },
}


def load_knowledge_base(kb_dir: Path = KB_DIR) -> List[Dict[str, Any]]:
    """Load all knowledge base documents with structural metadata."""
    documents: List[Dict[str, Any]] = []

    for filename, meta in DOC_METADATA_MAP.items():
        filepath = kb_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required KB file missing: {filepath}")

        content = filepath.read_text(encoding="utf-8").strip()
        # Count sentences roughly by period/exclamation/question mark
        sentences = [s.strip() for s in content.replace("\n", " ").split(".") if s.strip()]

        documents.append({
            "doc_id": meta["id"],
            "filename": filename,
            "topic": meta["topic"],
            "title": meta["title"],
            "content": content,
            "sentence_count": len(sentences),
            "sentences": [s + "." for s in sentences]
        })

    return documents


def validate_knowledge_base(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that all 12 required topics are covered and each doc has 2-5 sentences."""
    if len(documents) < 12:
        raise ValueError(f"Constraint Failed: KB must contain >= 12 documents, got {len(documents)}")

    covered_topics = {d["topic"] for d in documents}
    for topic in REQUIRED_TOPICS:
        if topic not in covered_topics:
            raise ValueError(f"Constraint Failed: Required topic missing: '{topic}'")

    for doc in documents:
        sent_count = doc["sentence_count"]
        if not (2 <= sent_count <= 5):
            raise ValueError(
                f"Constraint Failed: Document '{doc['title']}' has {sent_count} sentences "
                f"(must be between 2 and 5 sentences)"
            )

    return {
        "total_documents": len(documents),
        "covered_topics_count": len(covered_topics),
        "topics": list(covered_topics)
    }
