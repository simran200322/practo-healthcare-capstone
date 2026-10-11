"""Script to index knowledge base using both chunking strategies into separate ChromaDB collections."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.knowledge_base import load_knowledge_base, validate_knowledge_base
from app.chunking import fixed_size_chunking, sentence_based_chunking
from app.retrieval import index_chunks
from app.config import (
    CHROMA_FIXED_DIR,
    CHROMA_SENTENCE_DIR,
    COLLECTION_NAME_FIXED,
    COLLECTION_NAME_SENTENCE,
)

def main():
    print("=" * 65)
    print("PRACTO HEALTHCARE CAPSTONE - KNOWLEDGE BASE INDEXING")
    print("=" * 65)
    
    # 1. Load and validate documents
    docs = load_knowledge_base()
    validation = validate_knowledge_base(docs)
    print(f"Loaded {validation['total_documents']} documents covering all {validation['covered_topics_count']} topics.")
    
    # 2. Strategy 1: Fixed-size with overlap
    fixed_chunks = fixed_size_chunking(docs, chunk_size=180, chunk_overlap=40)
    print(f"\n[Strategy 1: Fixed-Size (180 chars, 40 overlap)]")
    print(f"  Generated chunks: {len(fixed_chunks)}")
    print(f"  Indexing into ChromaDB collection '{COLLECTION_NAME_FIXED}' at: {CHROMA_FIXED_DIR}")
    index_chunks(fixed_chunks, COLLECTION_NAME_FIXED, CHROMA_FIXED_DIR, recreate=True)
    print("  Index built successfully!")

    # 3. Strategy 2: Sentence-based
    sentence_chunks = sentence_based_chunking(docs)
    print(f"\n[Strategy 2: Sentence-Based (semantic boundaries)]")
    print(f"  Generated chunks: {len(sentence_chunks)}")
    print(f"  Indexing into ChromaDB collection '{COLLECTION_NAME_SENTENCE}' at: {CHROMA_SENTENCE_DIR}")
    index_chunks(sentence_chunks, COLLECTION_NAME_SENTENCE, CHROMA_SENTENCE_DIR, recreate=True)
    print("  Index built successfully!")

    print("\n" + "=" * 65)
    print("BOTH CHUNKING STRATEGIES EMBEDDED & INDEXED IN SEPARATE COLLECTIONS!")
    print("=" * 65)

if __name__ == "__main__":
    main()
