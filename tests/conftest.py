"""Shared fixtures. No test in this suite calls TMDb or OpenAI."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.search_api as search_api

SEARCH_INDEX_PATH = Path(__file__).parent.parent / "output" / "search_index.json"


@pytest.fixture(scope="session")
def index_items() -> list[dict]:
    """The real search index committed in the repo (21 titles with embeddings)."""
    return json.loads(SEARCH_INDEX_PATH.read_text(encoding="utf-8"))


class FakeLLMClient:
    """Stands in for LLMClient: returns whatever vector the test sets as `next_embedding`."""

    def __init__(self):
        self.next_embedding: list[float] | None = None
        self.queries: list[str] = []

    def get_embedding(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.next_embedding


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def api_client(monkeypatch, fake_llm):
    """The real FastAPI app (lifespan included) with OpenAI replaced by FakeLLMClient."""
    monkeypatch.setattr(search_api, "LLMClient", lambda: fake_llm)
    with TestClient(search_api.app) as client:
        yield client
