"""
rag.py — synthesize an answer from retrieved chunks using GPT-4o-mini.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List

import openai
from dotenv import load_dotenv

from engine.search import ChunkResult

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_MODEL = "gpt-4o-mini"

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class Source:
    url: str
    title: str
    heading: str


@dataclass
class RAGResponse:
    answer: str
    sources: List[Source]


def _build_prompt(query: str, chunks: List[ChunkResult]) -> str:
    # Simple, fast prompt building without complex logic
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        breadcrumb = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.heading
        context_blocks.append(
            f"[{i}] {breadcrumb}\n{chunk.content}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    # Check if comparative for simple prompt adjustment
    query_lower = query.lower()
    is_comparative = any(word in query_lower for word in ['vs', 'versus', 'compared to', 'difference between', 'better than', 'compare'])
    
    if is_comparative:
        return f"""You are an expert technical analyst. The user is asking for a comparison. Provide a comprehensive, balanced comparison addressing the key differences, strengths, and use cases.

Context:
{context}

Question: {query}

Answer:"""
    else:
        return f"""You are a technical documentation assistant. Answer the user's question using the context provided. Be precise and helpful.

Context:
{context}

Question: {query}

Answer:"""



def synthesize(query: str, chunks: List[ChunkResult]) -> RAGResponse:
    """Call the LLM with retrieved chunks as context and return a synthesized answer."""
    if not chunks:
        return RAGResponse(
            answer="No relevant documentation found for your query.",
            sources=[],
        )

    prompt = _build_prompt(query, chunks)

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,   # slightly higher for more natural comparative language
        max_tokens=1500,   # increased for comprehensive comparisons
    )

    answer = response.choices[0].message.content.strip()

    # Deduplicate sources by URL while preserving order
    seen = set()
    sources = []
    for chunk in chunks:
        if chunk.url not in seen:
            seen.add(chunk.url)
            sources.append(Source(
                url=chunk.url,
                title=chunk.title,
                heading=chunk.heading,
            ))

    return RAGResponse(answer=answer, sources=sources)