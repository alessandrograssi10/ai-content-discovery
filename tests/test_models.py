import pytest
from pydantic import ValidationError

from src.models import EnrichedContent, EnrichedMetadata, EnrichStatus, FetchStatus, SeedItem


def test_seed_item_coerces_year_from_csv_string():
    item = SeedItem(content_id="1001", title="The Shawshank Redemption", year="1994")
    assert item.year == 1994


def test_seed_item_rejects_non_numeric_year():
    with pytest.raises(ValidationError):
        SeedItem(content_id="1001", title="The Shawshank Redemption", year="unknown")


def test_enriched_metadata_rejects_incomplete_llm_output():
    # The realistic LLM failure mode: valid JSON, but a required field is missing.
    incomplete = {
        "detailed_genres": ["drama"],
        "mood": ["hopeful"],
        "themes": ["friendship"],
        "target_audience": ["adults"],
    }
    with pytest.raises(ValidationError):
        EnrichedMetadata(**incomplete)


def test_skipped_item_needs_no_metadata_or_embedding():
    item = EnrichedContent(
        content_id="9999",
        title="Unknown Title",
        year=2020,
        description="",
        fetch_status=FetchStatus.NOT_FOUND,
        enrich_status=EnrichStatus.SKIPPED,
    )
    assert item.enriched_metadata is None
    assert item.embedding is None


def test_status_enums_serialize_to_lowercase_strings():
    item = EnrichedContent(
        content_id="9999",
        title="Unknown Title",
        year=2020,
        description="",
        fetch_status=FetchStatus.ERROR,
        enrich_status=EnrichStatus.SKIPPED,
    )
    dumped = item.model_dump(mode="json")
    assert dumped["fetch_status"] == "error"
    assert dumped["enrich_status"] == "skipped"
