"""
Stage 3 — semantic search API over the enriched catalog.

Loads  : output/search_index.json  (full EnrichedContent with embeddings)
Serves : GET /search?q=<query>&top_k=<n>
         GET /health

Run with:
    uvicorn src.search_api:app --reload
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from src.models import EnrichedContent, EnrichStatus, SearchResponse, SearchResult
from src.services.openai_client import LLMClient

load_dotenv()

ROOT = Path(__file__).parent.parent
SEARCH_INDEX_PATH = ROOT / "output" / "search_index.json"

SIMILARITY_THRESHOLD = 0.25

catalog: list[EnrichedContent] = []
embeddings_matrix: np.ndarray = np.empty((0,))
llm_client: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and normalize the catalog once at startup. Never reload at runtime."""
    global catalog, embeddings_matrix, llm_client

    raw = json.loads(SEARCH_INDEX_PATH.read_text(encoding="utf-8"))
    all_items = [EnrichedContent(**item) for item in raw]

    valid = [
        item for item in all_items
        if item.enrich_status == EnrichStatus.OK and item.embedding
    ]

    if not valid:
        raise RuntimeError(
            f"No valid items in {SEARCH_INDEX_PATH}. Run src/enrich.py first."
        )

    catalog = valid

    vectors = np.array([item.embedding for item in catalog], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_matrix = vectors / norms

    llm_client = LLMClient()

    print(f"Catalog loaded: {len(catalog)} titles, embeddings normalized.")
    yield


app = FastAPI(
    title="AI Content Discovery",
    description="Semantic search over an enriched movie/TV catalog.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Quick check that the server started and the catalog loaded correctly."""
    return {"status": "ok", "catalog_size": len(catalog)}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=20, description="Maximum number of results to return"),
):
    """
    Search the catalog using a natural language query.

    Returns the most semantically relevant titles, ranked by similarity
    score. Returns an empty list if no title is a strong enough match.
    """
    if llm_client is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty or whitespace.")

    query_vector = np.array(llm_client.get_embedding(q), dtype=np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Cosine similarity as a single matrix dot product: both sides are
    # pre-normalized to unit length, so the dot product equals cosine
    # similarity directly -- computed for every catalog item at once
    # (vectorized), not via a Python loop over each item.
    similarities = embeddings_matrix @ query_vector

    # Filter weak matches BEFORE ranking, so a query with no strong match
    # returns an empty list instead of forcing top_k weak results.
    above_threshold = [
        (float(score), item)
        for score, item in zip(similarities, catalog)
        if float(score) >= SIMILARITY_THRESHOLD
    ]

    above_threshold.sort(key=lambda x: x[0], reverse=True)

    results = [
        SearchResult(
            content_id=item.content_id,
            title=item.title,
            year=item.year,
            score=round(score, 4),
        )
        for score, item in above_threshold[:top_k]
    ]

    return SearchResponse(query=q, results=results)