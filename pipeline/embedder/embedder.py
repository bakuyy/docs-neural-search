"""
Embedder: takes Chunks, calls text-embedding-3-small, inserts into pgvector
"""

from __future__ import annotations
import os
from typing import List

import openai
import psycopg2
from psycopg2.extras import execute_values

from chunker.chunker import Chunk
from dotenv import load_dotenv

load_dotenv()

# --- config ---

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100 
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DB_CONFIG = {
    "host": "127.0.0.1",  
    "port": 5432,
    "dbname": "neural_search",
    "user": "postgres",
    "password": "postgres",
}

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API in batches, return list of vectors."""
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        # Response items are ordered to match input order
        vectors = [item.embedding for item in response.data]
        all_vectors.extend(vectors)
    return all_vectors


def insert_chunks(chunks: List[Chunk], conn) -> None:
    """Embed and insert a list of chunks into the chunks table."""
    if not chunks:
        return

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    rows = [
        (
            c.canonical_url,
            c.title,
            c.heading,
            c.heading_path,
            c.text,
            vector,
        )
        for c, vector in zip(chunks, vectors)
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO chunks (url, title, heading, heading_path, content, embedding)
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s::vector)",
        )
    conn.commit()
    print(f"Inserted {len(rows)} chunks")


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def embed_and_store(chunks: List[Chunk]) -> None:
    """Top-level function: embed chunks and store in pgvector."""
    conn = get_connection()
    try:
        insert_chunks(chunks, conn)
    finally:
        conn.close()