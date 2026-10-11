"""Calibrate the RAG 'I don't know' threshold empirically on in-scope vs out-of-scope queries."""
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import retrieve_top_k
from app.rag import generate_grounded_response
from app.config import (
    COLLECTION_NAME_SENTENCE,
    CHROMA_SENTENCE_DIR,
    COLLECTION_NAME_FIXED,
    CHROMA_FIXED_DIR,
    EVALUATION_DIR
)

CALIBRATION_QUERIES = [
    {
        "id": "IN-01",
        "type": "in-scope",
        "query": "How much is the consultation fee for a Cardiologist?",
        "expected_topic": "consultation-fee structure by specialty"
    },
    {
        "id": "IN-02",
        "type": "in-scope",
        "query": "Can I cancel my appointment one hour before without penalty?",
        "expected_topic": "cancellation/rescheduling window"
    },
    {
        "id": "IN-03",
        "type": "in-scope",
        "query": "How long does it take to get diagnostic blood test results back?",
        "expected_topic": "lab-test turnaround times"
    },
    {
        "id": "IN-04",
        "type": "in-scope",
        "query": "What is the discount policy for follow-up doctor visits?",
        "expected_topic": "follow-up-visit discount policy"
    },
    {
        "id": "IN-05",
        "type": "in-scope",
        "query": "Are acute chest pain emergencies eligible for video telemedicine?",
        "expected_topic": "telemedicine eligibility"
    },
    {
        "id": "OOS-01",
        "type": "out-of-scope",
        "query": "What was the stock market closing price of Apple on NASDAQ yesterday?",
        "expected_topic": None
    },
    {
        "id": "OOS-02",
        "type": "out-of-scope",
        "query": "What is the best culinary recipe for baking chocolate chip cookies at home?",
        "expected_topic": None
    }
]


def run_calibration():
    print("=" * 70)
    print("EMPIRICAL 'I DON'T KNOW' THRESHOLD CALIBRATION (Task 4)")
    print("=" * 70)

    in_scope_scores = []
    oos_scores = []
    detailed_measurements = []

    print("\nMeasuring Top-1 Cosine Similarity across Sentence-Based Collection:")
    print("-" * 70)

    for item in CALIBRATION_QUERIES:
        query = item["query"]
        q_type = item["type"]
        chunks = retrieve_top_k(
            query=query,
            collection_name=COLLECTION_NAME_SENTENCE,
            persist_dir=CHROMA_SENTENCE_DIR,
            top_k=3
        )
        top_sim = chunks[0]["similarity"] if chunks else 0.0

        if q_type == "in-scope":
            in_scope_scores.append(top_sim)
        else:
            oos_scores.append(top_sim)

        top_match_title = chunks[0]["title"] if chunks else "N/A"
        print(f"[{item['id']}] ({q_type.upper():12s}) Top-1 Sim: {top_sim:.4f} | Query: {query}")
        print(f"      Matched: {top_match_title}")

        detailed_measurements.append({
            "id": item["id"],
            "type": q_type,
            "query": query,
            "top_1_similarity": top_sim,
            "matched_title": top_match_title,
            "matched_text": chunks[0]["text"] if chunks else None
        })

    min_in_scope = min(in_scope_scores)
    max_in_scope = max(in_scope_scores)
    avg_in_scope = sum(in_scope_scores) / len(in_scope_scores)

    min_oos = min(oos_scores)
    max_oos = max(oos_scores)
    avg_oos = sum(oos_scores) / len(oos_scores)

    # Calibrate threshold halfway between max out-of-scope and min in-scope
    calibrated_threshold = round((max_oos + min_in_scope) / 2.0, 4)

    print("-" * 70)
    print(f"In-Scope Cluster     : Min={min_in_scope:.4f}, Max={max_in_scope:.4f}, Mean={avg_in_scope:.4f}")
    print(f"Out-of-Scope Cluster : Min={min_oos:.4f}, Max={max_oos:.4f}, Mean={avg_oos:.4f}")
    print(f"Separation Margin    : {min_in_scope - max_oos:.4f}")
    print(f"CALIBRATED THRESHOLD : {calibrated_threshold:.4f}")
    print("=" * 70)

    print("\nDemonstrating Grounded Generation using Calibrated Threshold:")
    print("-" * 70)

    demo_results = []
    for item in CALIBRATION_QUERIES:
        chunks = retrieve_top_k(
            query=item["query"],
            collection_name=COLLECTION_NAME_SENTENCE,
            persist_dir=CHROMA_SENTENCE_DIR,
            top_k=3
        )
        gen = generate_grounded_response(
            query=item["query"],
            retrieved_chunks=chunks,
            similarity_threshold=calibrated_threshold
        )
        status_str = "FALLBACK TRIGGERED (CORRECT)" if gen["is_fallback"] else "GROUNDED ANSWER"
        print(f"\nQuery [{item['id']}]: {item['query']}")
        print(f"  Similarity: {gen['top_similarity']:.4f} | Outcome: {status_str}")
        print(f"  Answer    : {gen['answer']}")

        demo_results.append({
            "id": item["id"],
            "query": item["query"],
            "type": item["type"],
            "top_similarity": gen["top_similarity"],
            "grounded": gen["grounded"],
            "is_fallback": gen["is_fallback"],
            "answer": gen["answer"]
        })

    # Save artifact
    output_path = EVALUATION_DIR / "threshold_calibration_results.json"
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "calibrated_threshold": calibrated_threshold,
            "in_scope_cluster": {
                "min": min_in_scope,
                "max": max_in_scope,
                "mean": round(avg_in_scope, 4)
            },
            "oos_cluster": {
                "min": min_oos,
                "max": max_oos,
                "mean": round(avg_oos, 4)
            },
            "margin": round(min_in_scope - max_oos, 4),
            "measurements": detailed_measurements,
            "demo_runs": demo_results
        }, f, indent=2)

    print(f"\nSaved calibration results to: {output_path}")
    return calibrated_threshold


if __name__ == "__main__":
    run_calibration()
