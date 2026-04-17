"""
Enhanced search system with query expansion and multi-strategy retrieval for better comparative searches.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Set
import openai
import psycopg2
from dotenv import load_dotenv

from engine.search import ChunkResult, DB_CONFIG, _embed_query

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

@dataclass
class QueryAnalysis:
    is_comparative: bool
    entities: List[str]
    query_type: str  # "comparison", "definition", "how-to", "troubleshooting"
    expanded_queries: List[str]

def analyze_query(query: str) -> QueryAnalysis:
    """Analyze query to understand intent and extract entities."""
    query_lower = query.lower()
    
    # Detect comparative queries
    comparison_words = ['vs', 'versus', 'compared to', 'difference between', 'better than', 'compare']
    is_comparative = any(word in query_lower for word in comparison_words)
    
    # Extract entities (simple approach - could be enhanced with NER)
    entities = []
    if is_comparative:
        # Split on comparison words and extract entities
        for comp_word in comparison_words:
            if comp_word in query_lower:
                parts = re.split(rf'\b{re.escape(comp_word)}\b', query_lower)
                for part in parts:
                    # Extract potential entity names (words that look like products/services)
                    words = re.findall(r'\b[a-z]+(?:[a-z0-9]*[a-z])*\b', part.strip())
                    entities.extend([w for w in words if len(w) > 2 and w not in ['the', 'and', 'for', 'with']])
    
    # Remove duplicates while preserving order
    entities = list(dict.fromkeys(entities))
    
    # Determine query type
    if is_comparative:
        query_type = "comparison"
    elif any(word in query_lower for word in ['what is', 'define', 'definition']):
        query_type = "definition"
    elif any(word in query_lower for word in ['how to', 'how do', 'tutorial']):
        query_type = "how-to"
    elif any(word in query_lower for word in ['error', 'issue', 'problem', 'fix', 'troubleshoot']):
        query_type = "troubleshooting"
    else:
        query_type = "general"
    
    # Generate expanded queries
    expanded_queries = generate_expanded_queries(query, entities, query_type)
    
    return QueryAnalysis(
        is_comparative=is_comparative,
        entities=entities,
        query_type=query_type,
        expanded_queries=expanded_queries
    )

def generate_expanded_queries(original_query: str, entities: List[str], query_type: str) -> List[str]:
    """Generate expanded queries for better retrieval."""
    queries = [original_query]
    
    if query_type == "comparison" and len(entities) >= 2:
        # For comparisons, create individual queries for each entity
        for entity in entities[:4]:  # Limit to avoid too many queries
            queries.append(f"{entity} features benefits")
            queries.append(f"{entity} pros and cons")
            queries.append(f"{entity} use cases")
            queries.append(f"when to use {entity}")
        
        # Add direct comparison queries
        if len(entities) >= 2:
            queries.append(f"{entities[0]} {entities[1]} comparison")
            queries.append(f"difference between {entities[0]} and {entities[1]}")
    
    elif query_type == "definition":
        for entity in entities[:2]:
            queries.append(f"what is {entity}")
            queries.append(f"{entity} overview")
    
    elif query_type == "how-to":
        for entity in entities[:2]:
            queries.append(f"how to use {entity}")
            queries.append(f"{entity} tutorial guide")
    
    return queries

def multi_query_search(queries: List[str], k_per_query: int = 8) -> List[ChunkResult]:
    """Search with multiple queries and merge results."""
    all_chunks = []
    seen_content = set()
    
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        for query in queries:
            vector = _embed_query(query)
            vector_str = "[" + ",".join(str(x) for x in vector) + "]"
            
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT url, title, heading, heading_path, content,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_str, vector_str, k_per_query),
                )
                rows = cur.fetchall()
                
                for row in rows:
                    # Deduplicate by content hash
                    content_hash = hash(row[4])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        chunk = ChunkResult(
                            url=row[0],
                            title=row[1],
                            heading=row[2],
                            heading_path=row[3] or [],
                            content=row[4],
                            similarity=float(row[5]),
                        )
                        all_chunks.append(chunk)
    finally:
        conn.close()
    
    return all_chunks

def diversified_search(query: str, k: int = 20) -> List[ChunkResult]:
    """Enhanced search with query analysis and diversification."""
    analysis = analyze_query(query)
    
    # Get chunks from multiple expanded queries
    all_chunks = multi_query_search(analysis.expanded_queries, k_per_query=max(3, k // len(analysis.expanded_queries)))
    
    if not all_chunks:
        return []
    
    # For comparative queries, ensure we have content about different entities
    if analysis.is_comparative and analysis.entities:
        entity_chunks = diversify_by_entities(all_chunks, analysis.entities)
        if entity_chunks:
            all_chunks = entity_chunks
    
    # Sort by similarity and limit
    all_chunks.sort(key=lambda x: x.similarity, reverse=True)
    return all_chunks[:k]

def diversify_by_entities(chunks: List[ChunkResult], entities: List[str]) -> List[ChunkResult]:
    """Ensure we have diverse content covering different entities."""
    entity_buckets: Dict[str, List[ChunkResult]] = {}
    unmatched_chunks = []
    
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
            unmatched_chunks.append(chunk)
    
    # Interleave chunks from different entities
    result = []
    max_per_entity = max(3, 15 // len(entity_buckets)) if entity_buckets else 0
    
    for entity, entity_chunks in entity_buckets.items():
        # Sort by similarity within entity
        entity_chunks.sort(key=lambda x: x.similarity, reverse=True)
        result.extend(entity_chunks[:max_per_entity])
    
    # Add some unmatched chunks that might contain comparative content
    result.extend(unmatched_chunks[:5])
    
    return result

def hybrid_search_with_keywords(query: str, k: int = 20) -> List[ChunkResult]:
    """Hybrid approach combining semantic search with keyword matching."""
    # Get semantic results
    semantic_chunks = diversified_search(query, k)
    
    # For comparative queries, also do keyword search
    analysis = analyze_query(query)
    if analysis.is_comparative and analysis.entities:
        keyword_chunks = keyword_search_for_entities(analysis.entities, k // 2)
        
        # Merge and deduplicate
        all_chunks = semantic_chunks + keyword_chunks
        seen_content = set()
        unique_chunks = []
        
        for chunk in all_chunks:
            content_hash = hash(chunk.content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        # Sort by similarity
        unique_chunks.sort(key=lambda x: x.similarity, reverse=True)
        return unique_chunks[:k]
    
    return semantic_chunks

def keyword_search_for_entities(entities: List[str], limit: int = 10) -> List[ChunkResult]:
    """Search using SQL text matching for entities."""
    if not entities:
        return []
    
    # Create search conditions for entities
    entity_conditions = []
    for entity in entities[:3]:  # Limit to avoid complex queries
        entity_conditions.append(f"content ILIKE '%{entity}%'")
    
    where_clause = " OR ".join(entity_conditions)
    
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT url, title, heading, heading_path, content, 0.5 as similarity
                FROM chunks
                WHERE {where_clause}
                LIMIT %s
                """,
                (limit,)
            )
            rows = cur.fetchall()
            
            return [
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
    finally:
        conn.close()