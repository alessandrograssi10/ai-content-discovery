from src.enrich import build_embedding_text
from src.models import EnrichedContent, EnrichedMetadata, EnrichStatus, FetchStatus


def make_item(description="A banker is jailed for murder.", **meta_overrides):
    meta = {
        "detailed_genres": ["prison drama"],
        "mood": ["hopeful", "tense"],
        "themes": ["friendship"],
        "target_audience": ["adults"],
        "similar_content_suggestions": ["The Green Mile"],
        **meta_overrides,
    }
    return EnrichedContent(
        content_id="1001",
        title="The Shawshank Redemption",
        year=1994,
        description=description,
        fetch_status=FetchStatus.OK,
        enrich_status=EnrichStatus.OK,
        enriched_metadata=EnrichedMetadata(**meta),
    )


def test_embedding_text_combines_title_description_and_metadata():
    assert build_embedding_text(make_item()) == (
        "The Shawshank Redemption | A banker is jailed for murder. | prison drama"
        " | hopeful, tense | friendship | adults"
    )


def test_embedding_text_excludes_similar_content_suggestions():
    # Other titles' names would pull the vector toward those titles.
    assert "The Green Mile" not in build_embedding_text(make_item())


def test_embedding_text_skips_empty_parts():
    text = build_embedding_text(make_item(description="", themes=[]))
    assert text == "The Shawshank Redemption | prison drama | hopeful, tense | adults"
