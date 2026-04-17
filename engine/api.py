"""
Run with:
uvicorn engine.api:app --reload --port 8000
"""

from __future__ import annotations
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.search import similarity_search, ChunkResult
from engine.rag import synthesize, RAGResponse, Source


app = FastAPI(title="Neural Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request / response models ---

class SearchRequest(BaseModel):
    query: str
    k: int = 10        # balanced default for speed vs quality


class SourceResponse(BaseModel):
    url: str
    title: str
    heading: str


class SearchResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]
    chunks: List[dict]   # raw chunks, useful for debugging in the UI


# --- endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    chunks = similarity_search(req.query, k=req.k)
    rag_response = synthesize(req.query, chunks)

    return SearchResponse(
        answer=rag_response.answer,
        sources=[
            SourceResponse(url=s.url, title=s.title, heading=s.heading)
            for s in rag_response.sources
        ],
        chunks=[
            {
                "url": c.url,
                "heading": c.heading,
                "heading_path": c.heading_path,
                "content": c.content,
                "similarity": round(c.similarity, 4),
            }
            for c in chunks
        ],
    )