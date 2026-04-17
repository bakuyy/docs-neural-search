"""
search.py — embed a query and retrieve the most similar chunks from pgvector.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List

import openai
import psycopg2
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "neural_search",
    "user": "postgres",
    "password": "postgres",
}

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class ChunkResult:
    url: str
    title: str
    heading: str
    heading_path: List[str]
    content: str
    similarity: float


def _embed_query(query: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


def similarity_search(query: str, k: int = 5) -> List[ChunkResult]:
    """Single-query search with smart post-processing based on intent."""
    # Always do single search - let the RAG system handle the intent
    return _enhanced_single_search(query, k)

def _extract_entities(query: str) -> list[str]:
    """Extract entities from comparative queries."""
    import re
    query_lower = query.lower()
    
    # Split on comparison words
    comparison_words = ['vs', 'versus', 'compared to', 'difference between', 'better than', 'compare']
    for comp_word in comparison_words:
        if comp_word in query_lower:
            parts = re.split(rf'\b{re.escape(comp_word)}\b', query_lower)
            entities = []
            for part in parts:
                # Extract potential entity names
                words = re.findall(r'\b[a-z]+(?:[a-z0-9]*[a-z])*\b', part.strip())
                entities.extend([w for w in words if len(w) > 2 and w not in ['the', 'and', 'for', 'with', 'what', 'are']])
            return list(dict.fromkeys(entities))  # Remove duplicates
    return []

def _enhanced_single_search(query: str, k: int) -> List[ChunkResult]:
    """Single-query search with smart post-processing for better diversity."""
    # Single embedding and single query - fast and simple
    vector = _embed_query(query)
    vector_str = "[" + ",".join(str(x) for x in vector) + "]"

    # Get more results than requested for post-processing diversity
    search_k = min(k * 3, 50)  # Get 3x results but cap at 50 for performance

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT url, title, heading, heading_path, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_str, vector_str, search_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    all_chunks = [
        ChunkResult(
            url=row[0],
            title=row[1],
            heading=row[2],
            heading_path=row[3] or [],
            content=row[4],
            similarity=float(row[5]),
        )
        for row in rows
    ]

    # Post-process for diversity based on query intent
    query_lower = query.lower()
    is_comparative = any(word in query_lower for word in ['vs', 'versus', 'compared to', 'difference between', 'better than', 'compare'])
    
    if is_comparative and len(all_chunks) > k:
        # Extract entities and diversify results
        entities = _extract_entities(query)
        if entities:
            all_chunks = _diversify_by_entities(all_chunks, entities, k)
    
    # Return top k results
    return all_chunks[:k]

def _diversify_by_entities(chunks: List[ChunkResult], entities: List[str], target_k: int) -> List[ChunkResult]:
    """Post-process to ensure diverse content covering different entities."""
    entity_buckets = {}
    unmatched = []
    
    # Categorize chunks by entity
    for chunk in chunks:
        content_lower = (chunk.content + " " + chunk.title + " " + chunk.heading).lower()
        matched_entity = None
        
        for entity in entities:
            if entity.lower() in content_lower:
                matched_entity = entity
                break
        
        if matched_entity:
            if matched_entity not in entity_buckets:
                entity_buckets[matched_entity] = []
            entity_buckets[matched_entity].append(chunk)
        else:
            unmatched.append(chunk)
    
    # Interleave diverse results
    result = []
    if entity_buckets:
        max_per_entity = max(2, target_k // len(entity_buckets))
        
        for entity, entity_chunks in entity_buckets.items():
            entity_chunks.sort(key=lambda x: x.similarity, reverse=True)
            result.extend(entity_chunks[:max_per_entity])
    
    # Add unmatched chunks to fill remaining slots
    remaining_slots = target_k - len(result)
    if remaining_slots > 0:
        result.extend(unmatched[:remaining_slots])
    
    # Sort final results by similarity 
    result.sort(key=lambda x: x.similarity, reverse=True)
    return result