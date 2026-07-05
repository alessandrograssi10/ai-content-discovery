"""
Stage 2 — enrich fetched content with LLM-generated metadata and embeddings.

Reads  : output/fetched_content.json
Writes : output/enriched_content.json  (main output, all 21 items, no embeddings)
         output/search_index.json       (internal, only OK items with embeddings, used by Stage 3)

Run with:
    python -m src.enrich
"""

import json
from pathlib import Path

import openai
from dotenv import load_dotenv
from pydantic import ValidationError

from src.models import EnrichedContent, EnrichedMetadata, EnrichStatus, FetchedContent, FetchStatus
from src.services.openai_client import ENRICHMENT_MODEL, LLMClient

load_dotenv()

ROOT = Path(__file__).parent.parent
FETCHED_PATH = ROOT / "output" / "fetched_content.json"
ENRICHED_PATH = ROOT / "output" / "enriched_content.json"
SEARCH_INDEX_PATH = ROOT / "output" / "search_index.json"
PROMPT_PATH = ROOT / "prompts" / "enrichment_prompt.md"

# Exceptions we specifically expect and know how to handle:
#   - json.JSONDecodeError: the LLM response wasn't valid JSON. Rare with
#     response_format={"type": "json_object"} forcing JSON mode, but not
#     impossible, so still handled explicitly.
#   - ValidationError: the JSON was valid but didn't match our schema
#     (missing field, wrong type). The realistic failure mode here.
#   - openai.APIError: infrastructure-level failure (rate limit, timeout,
#     OpenAI outage) -- a data/API problem, not a bug in our code.
# Anything else (e.g. AttributeError from a typo, KeyError from a wrong
# dict access) is a genuine bug and is intentionally left uncaught, so it
# crashes loudly instead of being silently logged as "enrichment failed".
ENRICHMENT_EXPECTED_ERRORS = (json.JSONDecodeError, ValidationError, openai.APIError)


def build_embedding_text(item: EnrichedContent) -> str:
    """Combine title, description, and metadata into a single string for embedding.

    Embedding on combined text rather than description alone improves match
    quality for queries about tone or atmosphere (e.g. "something funny and
    real") that are captured in mood/themes metadata, not in the raw plot
    description.
    """
    meta = item.enriched_metadata
    parts = [
        item.title,
        item.description,
        ", ".join(meta.detailed_genres),
        ", ".join(meta.mood),
        ", ".join(meta.themes),
        ", ".join(meta.target_audience),
    ]
    return " | ".join(p for p in parts if p)


def main() -> None:
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8").strip()

    fetched_items = [
        FetchedContent(**item)
        for item in json.loads(FETCHED_PATH.read_text(encoding="utf-8"))
    ]

    all_results: list[EnrichedContent] = []

    client = LLMClient()

    for item in fetched_items:
        base = item.model_dump()

        if item.fetch_status != FetchStatus.OK or not item.description:
            all_results.append(EnrichedContent(
                **base,
                enrich_status=EnrichStatus.SKIPPED,
            ))
            print(f"[SKIPPED] {item.title} ({item.year}) — fetch_status={item.fetch_status.value}")
            continue

        try:
            raw_metadata = client.enrich_content(
                title=item.title,
                year=item.year,
                description=item.description,
                prompt_template=prompt_template,
            )
            metadata = EnrichedMetadata(**raw_metadata)
        except ENRICHMENT_EXPECTED_ERRORS as exc:
            all_results.append(EnrichedContent(
                **base,
                enrich_status=EnrichStatus.ERROR,
            ))
            print(f"[ERROR]   {item.title} ({item.year}) — enrichment failed ({type(exc).__name__}): {exc}")
            continue
        except Exception as exc:
            # Not one of the errors we anticipated -- likely a bug in our own
            # code, not a data/API problem. Still isolated to this single
            # item (the other 20 titles are not lost), but flagged distinctly
            # so it is not mistaken for a normal, expected failure.
            all_results.append(EnrichedContent(
                **base,
                enrich_status=EnrichStatus.ERROR,
            ))
            print(f"[BUG?]    {item.title} ({item.year}) — UNEXPECTED error ({type(exc).__name__}): {exc}")
            continue

        enriched = EnrichedContent(
            **base,
            enrich_status=EnrichStatus.OK,
            enriched_metadata=metadata,
            enrichment_model=ENRICHMENT_MODEL,
        )

        try:
            enriched.embedding = client.get_embedding(build_embedding_text(enriched))
        except openai.APIError as exc:
            enriched.enrich_status = EnrichStatus.ERROR
            enriched.enriched_metadata = None
            enriched.embedding = None
            all_results.append(enriched)
            print(f"[ERROR]   {item.title} ({item.year}) — embedding failed ({type(exc).__name__}): {exc}")
            continue
        except Exception as exc:
            enriched.enrich_status = EnrichStatus.ERROR
            enriched.enriched_metadata = None
            enriched.embedding = None
            all_results.append(enriched)
            print(f"[BUG?]    {item.title} ({item.year}) — UNEXPECTED error during embedding ({type(exc).__name__}): {exc}")
            continue

        print(f"[OK]      {item.title} ({item.year})")
        all_results.append(enriched)

    with open(ENRICHED_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump(exclude={"embedding"}) for r in all_results],
            f,
            indent=2,
            ensure_ascii=False,
        )

    search_index = [r for r in all_results if r.enrich_status == EnrichStatus.OK]
    with open(SEARCH_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in search_index],
            f,
            indent=2,
            ensure_ascii=False,
        )

    ok = sum(1 for r in all_results if r.enrich_status == EnrichStatus.OK)
    skipped = sum(1 for r in all_results if r.enrich_status == EnrichStatus.SKIPPED)
    errors = sum(1 for r in all_results if r.enrich_status == EnrichStatus.ERROR)

    print(f"\nDone. {ok} OK / {skipped} skipped / {errors} errors")
    print(f"Main output  : {ENRICHED_PATH} ({len(all_results)} items)")
    print(f"Search index : {SEARCH_INDEX_PATH} ({len(search_index)} items)")


if __name__ == "__main__":
    main()