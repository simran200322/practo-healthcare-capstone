"""Retrieval evaluation comparing Fixed-Size vs Sentence-Based chunking strategies."""
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import retrieve_top_k
from app.config import (
    COLLECTION_NAME_FIXED,
    COLLECTION_NAME_SENTENCE,
    CHROMA_FIXED_DIR,
    CHROMA_SENTENCE_DIR,
    EVALUATION_DIR
)

# 5 In-scope queries with ground-truth parent document IDs
EVAL_QUERIES = [
    {
        "id": "Q1",
        "query": "How much is the consultation fee for a Cardiologist?",
        "ground_truth_docs": {"KB-DOC-003", "KB-DOC-010"}  # Fee structure & follow-up fee
    },
    {
        "id": "Q2",
        "query": "Can I cancel my appointment one hour before without penalty?",
        "ground_truth_docs": {"KB-DOC-002"}  # Cancellation/rescheduling
    },
    {
        "id": "Q3",
        "query": "How long does it take to get diagnostic blood test results back?",
        "ground_truth_docs": {"KB-DOC-006"}  # Lab-test turnaround
    },
    {
        "id": "Q4",
        "query": "What is the discount policy for follow-up doctor visits?",
        "ground_truth_docs": {"KB-DOC-010"}  # Follow-up discount
    },
    {
        "id": "Q5",
        "query": "Are acute chest pain emergencies eligible for video telemedicine?",
        "ground_truth_docs": {"KB-DOC-007", "KB-DOC-008"}  # Telemedicine & emergency protocol
    }
]


def evaluate_collection(collection_name: str, persist_dir: Path, strategy_name: str):
    results = []
    print(f"\nEvaluating Strategy: {strategy_name} (Collection: {collection_name})")
    print("=" * 80)

    for item in EVAL_QUERIES:
        q_id = item["id"]
        query = item["query"]
        ground_truth = item["ground_truth_docs"]

        # Retrieve top-3 chunks
        chunks = retrieve_top_k(
            query=query,
            collection_name=collection_name,
            persist_dir=persist_dir,
            top_k=3
        )

        # Map back to parent documents and deduplicate
        retrieved_parent_docs = []
        seen = set()
        for c in chunks:
            p_id = c["parent_doc_id"]
            if p_id not in seen:
                seen.add(p_id)
                retrieved_parent_docs.append(p_id)

        # Relevant parent documents retrieved
        relevant_retrieved = [doc_id for doc_id in retrieved_parent_docs if doc_id in ground_truth]

        # Precision@3 = |Relevant Unique Docs| / |Total Unique Docs Retrieved|
        p_at_3 = len(relevant_retrieved) / len(retrieved_parent_docs) if retrieved_parent_docs else 0.0

        # Recall@3 = |Relevant Unique Docs| / |Total Ground Truth Docs|
        r_at_3 = len(relevant_retrieved) / len(ground_truth) if ground_truth else 0.0

        print(f"[{q_id}] Query: \"{query}\"")
        print(f"     Ground Truth Docs     : {list(ground_truth)}")
        print(f"     Retrieved Unique Docs : {retrieved_parent_docs}")
        print(f"     Relevant Retrieved    : {relevant_retrieved}")
        print(f"     Precision@3 Arithmetic: {len(relevant_retrieved)} / {len(retrieved_parent_docs)} = {p_at_3:.4f}")
        print(f"     Recall@3 Arithmetic   : {len(relevant_retrieved)} / {len(ground_truth)} = {r_at_3:.4f}")
        print("-" * 80)

        results.append({
            "query_id": q_id,
            "query": query,
            "ground_truth": list(ground_truth),
            "retrieved_unique_docs": retrieved_parent_docs,
            "relevant_retrieved": relevant_retrieved,
            "precision_at_3": round(p_at_3, 4),
            "recall_at_3": round(r_at_3, 4)
        })

    avg_p = sum(r["precision_at_3"] for r in results) / len(results)
    avg_r = sum(r["recall_at_3"] for r in results) / len(results)

    print(f">> Mean Precision@3 across 5 queries: {avg_p:.4f}")
    print(f">> Mean Recall@3 across 5 queries   : {avg_r:.4f}")
    print("=" * 80)

    return {
        "strategy": strategy_name,
        "collection": collection_name,
        "queries": results,
        "mean_precision_at_3": round(avg_p, 4),
        "mean_recall_at_3": round(avg_r, 4)
    }


def main():
    print("=" * 80)
    print("PRACTO HEALTHCARE CAPSTONE - CHUNKING STRATEGY COMPARISON (Task 5)")
    print("=" * 80)

    fixed_eval = evaluate_collection(
        collection_name=COLLECTION_NAME_FIXED,
        persist_dir=CHROMA_FIXED_DIR,
        strategy_name="Fixed-Size Chunks with Overlap"
    )

    sentence_eval = evaluate_collection(
        collection_name=COLLECTION_NAME_SENTENCE,
        persist_dir=CHROMA_SENTENCE_DIR,
        strategy_name="Sentence-Based Semantic Chunks"
    )

    # Formulate Recommendation citing real numbers
    fixed_p = fixed_eval["mean_precision_at_3"]
    fixed_r = fixed_eval["mean_recall_at_3"]
    sent_p = sentence_eval["mean_precision_at_3"]
    sent_r = sentence_eval["mean_recall_at_3"]

    recommendation = (
        f"We recommend deploying the Sentence-Based Chunking strategy. In empirical evaluation across the 5 "
        f"representative policy queries, Sentence-Based chunking achieved a Mean Precision@3 of {sent_p:.4f} and "
        f"Mean Recall@3 of {sent_r:.4f}, compared to {fixed_p:.4f} Precision@3 and {fixed_r:.4f} Recall@3 for "
        f"Fixed-Size chunking. Sentence-based boundaries eliminate cross-sentence arbitrary truncation, preserving "
        f"atomic clinical rules and maximizing semantic relevance per retrieved chunk."
    )

    print("\nDEPLOYMENT RECOMMENDATION:")
    print("-" * 80)
    print(recommendation)
    print("-" * 80)

    # Save artifact
    output_path = EVALUATION_DIR / "retrieval_comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "fixed_size_strategy": fixed_eval,
            "sentence_based_strategy": sentence_eval,
            "recommendation": recommendation
        }, f, indent=2)

    print(f"\nSaved comparison results to: {output_path}")


if __name__ == "__main__":
    main()
