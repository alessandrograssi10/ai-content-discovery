"""
Pydantic data models for the content discovery pipeline.

The models follow an additive hierarchy: each pipeline stage extends the
previous one, so an object accumulates fields as it flows through
fetch -> enrich -> (search).

    SeedItem            -> raw input (content_id, title, year)
      FetchedContent    -> + description, tmdb_id, media_type fetched from TMDb
        EnrichedContent -> + LLM-generated metadata and embedding vector

SearchResult and SearchResponse are intentionally NOT part of this
hierarchy: they only exist at runtime while serving a search request and
are never persisted to disk, so they inherit from nothing.
"""

from enum import Enum

from pydantic import BaseModel, Field


class FetchStatus(str, Enum):
    """Outcome of a TMDb fetch attempt for a single title.

    OK        -> a matching title was found and a description was extracted.
    NOT_FOUND -> TMDb responded successfully but no matching title exists.
    ERROR     -> every fetch attempt failed due to a network/HTTP error
                 (e.g. TMDb unreachable) -- distinct from NOT_FOUND, since
                 it means we could not verify whether the title exists,
                 not that it doesn't.
    """

    OK = "ok"
    NOT_FOUND = "not_found"
    ERROR = "error"


class SeedItem(BaseModel):
    """A single row from the seed dataset (content_sample.csv/json).

    content_id stays a string because it is an identifier, not a value to
    compute on. year is typed as int: Pydantic coerces the CSV string
    "1994" into the integer 1994 during validation.
    """

    content_id: str
    title: str
    year: int


class FetchedContent(SeedItem):
    """A seed item extended with data fetched from TMDb (Stage 1).

    description holds the TMDb overview, or an empty string when the title
    was not found. fetch_status makes the outcome explicit so downstream
    stages can skip anything that is not OK. media_type ("movie" or "tv")
    is kept for traceability -- confirms at a glance that the movie/tv
    fallback logic resolved ambiguous titles correctly (e.g. "The Office"
    should show media_type="tv", not "movie").
    """

    description: str
    tmdb_id: int | None = None
    media_type: str | None = None
    fetch_status: FetchStatus


class EnrichStatus(str, Enum):
    """Outcome of an LLM enrichment attempt for a single title (Stage 2).

    Mirrors FetchStatus : enrichment is at least as
    likely to fail per-item as fetching (malformed JSON output, unexpected
    safety filtering, transient API errors).

    OK      -> enrichment succeeded; enriched_metadata and embedding are present.
    SKIPPED -> enrichment was never attempted, because the upstream
               fetch_status was not OK (avoiding enrichment without a real
               description to ground it in -- see FetchedContent).
    ERROR   -> enrichment was attempted but failed (malformed LLM output,
               API error) despite having a valid description to work from.
    """

    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


class EnrichedMetadata(BaseModel):
    """LLM-generated structured metadata for a single title (Stage 2).

    Nested inside EnrichedContent rather than part of the additive
    hierarchy, because it is a self-contained block produced in one stage,
    not an "is-a" extension of FetchedContent.
    """

    detailed_genres: list[str]
    mood: list[str]
    themes: list[str]
    target_audience: list[str]
    similar_content_suggestions: list[str]


class EnrichedContent(FetchedContent):
    """A fetched item extended with LLM metadata and an embedding (Stage 2).

    enriched_metadata and embedding are Optional because enrich_status may
    be SKIPPED or ERROR, in which case there is nothing to put there --
    the item still appears in the output (never silently dropped), just
    with these fields empty and the status explaining why.

    enrichment_model records which model produced the metadata (e.g.
    "gpt-5.4-mini"), for traceability if the pipeline is ever re-run with a
    different model.

    This is the shape persisted in output/enriched_content.json (the
    main output, embedding excluded) and output/search_index.json (internal,
    full model with embedding, loaded by Stage 3).
    """

    enrich_status: EnrichStatus
    enriched_metadata: EnrichedMetadata | None = None
    embedding: list[float] | None = None
    enrichment_model: str | None = None


class SearchResult(BaseModel):
    """A single ranked match returned by the search endpoint (Stage 3).

    Runtime-only, never persisted. Deliberately flat (a DTO, not a nested
    EnrichedContent) -- score is a property of the query, not of the
    content itself, and a search response should not carry internal fields
    like the embedding or fetch_status.
    """

    content_id: str
    title: str
    year: int
    score: float = Field(..., description="Cosine similarity, range [-1, 1]")


class SearchResponse(BaseModel):
    """The full payload returned by the search endpoint."""

    query: str
    results: list[SearchResult]