"""Implementation of two distinct chunking strategies: Fixed-Size-with-Overlap and Sentence-Based."""
import re
from typing import List, Dict, Any


def split_into_sentences(text: str) -> List[str]:
    """Cleanly split text into individual sentences using punctuation boundary regex."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def fixed_size_chunking(
    documents: List[Dict[str, Any]],
    chunk_size: int = 180,
    chunk_overlap: int = 40
) -> List[Dict[str, Any]]:
    """Chunk documents using a fixed character size with sliding window overlap.
    
    Preserves parent document ID and metadata on every chunk.
    """
    chunks: List[Dict[str, Any]] = []

    for doc in documents:
        text = doc["content"]
        doc_id = doc["doc_id"]
        start = 0
        chunk_idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{doc_id}-fix-{chunk_idx:02d}",
                    "parent_doc_id": doc_id,
                    "title": doc["title"],
                    "topic": doc["topic"],
                    "strategy": "fixed_size",
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end
                })
                chunk_idx += 1

            if end == text_len:
                break
            start += (chunk_size - chunk_overlap)

    return chunks


def sentence_based_chunking(
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Chunk documents at semantic sentence boundaries.
    
    Preserves parent document ID and metadata on every chunk.
    """
    chunks: List[Dict[str, Any]] = []

    for doc in documents:
        doc_id = doc["doc_id"]
        sentences = split_into_sentences(doc["content"])
        
        for idx, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc_id}-sent-{idx:02d}",
                "parent_doc_id": doc_id,
                "title": doc["title"],
                "topic": doc["topic"],
                "strategy": "sentence_based",
                "text": sentence,
                "sentence_index": idx
            })

    return chunks
