"""
Simplified API without RAG to debug the hanging issue.
"""

from __future__ import annotations
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.search import similarity_search, ChunkResult

app = FastAPI(title="Neural Search Engine - Debug")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SimpleSearchResponse(BaseModel):
    query: str
    chunks: List[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search", response_model=SimpleSearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Just return search results without RAG processing
    chunks = similarity_search(req.query, k=req.k)
    
    return SimpleSearchResponse(
        query=req.query,
        chunks=[
            {
                "url": c.url,
                "title": c.title,
                "heading": c.heading,
                "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                "similarity": round(c.similarity, 4),
            }
            for c in chunks
        ],
    )