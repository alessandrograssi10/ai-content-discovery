"""TMDbClient matching logic, with TMDb replaced by an in-memory HTTP transport."""

import httpx
import pytest

from src.models import FetchStatus
from src.services.tmdb_client import TMDB_BASE_URL, TMDbClient


def make_client(handler) -> TMDbClient:
    """A TMDbClient whose HTTP calls are answered by `handler` instead of the network."""
    client = TMDbClient(api_key="test-key")
    client._client.close()
    client._client = httpx.Client(base_url=TMDB_BASE_URL, transport=httpx.MockTransport(handler))
    return client


def movie(title, release_date, id=1, overview="A plot."):
    return {"id": id, "title": title, "release_date": release_date, "overview": overview}


def tv(name, first_air_date, id=2, overview="A plot."):
    return {"id": id, "name": name, "first_air_date": first_air_date, "overview": overview}


def respond(movie_results=(), tv_results=()):
    """Handler that returns fixed results per endpoint, ignoring the year filter."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search/movie"):
            return httpx.Response(200, json={"results": list(movie_results)})
        return httpx.Response(200, json={"results": list(tv_results)})

    return handler


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TMDbClient(api_key=None)


def test_exact_title_and_year_is_ok():
    client = make_client(respond(movie_results=[movie("Inception", "2010-07-15", id=27205)]))
    result = client.fetch("Inception", 2010)
    assert result["status"] == FetchStatus.OK
    assert result["tmdb_id"] == 27205
    assert result["media_type"] == "movie"
    assert result["description"] == "A plot."


def test_title_match_is_case_insensitive():
    client = make_client(respond(movie_results=[movie("INCEPTION", "2010-07-15")]))
    assert client.fetch("Inception", 2010)["status"] == FetchStatus.OK


def test_year_off_by_one_is_accepted():
    # Casablanca: premiere 1942, US release 1943.
    client = make_client(respond(movie_results=[movie("Casablanca", "1943-01-23")]))
    assert client.fetch("Casablanca", 1942)["status"] == FetchStatus.OK


def test_year_off_by_two_is_rejected():
    client = make_client(respond(movie_results=[movie("Casablanca", "1944-01-01")]))
    assert client.fetch("Casablanca", 1942)["status"] == FetchStatus.NOT_FOUND


def test_falls_back_to_tv_when_no_movie_matches():
    client = make_client(respond(tv_results=[tv("Breaking Bad", "2008-01-20", id=1396)]))
    result = client.fetch("Breaking Bad", 2008)
    assert result["status"] == FetchStatus.OK
    assert result["media_type"] == "tv"
    assert result["tmdb_id"] == 1396


def test_partial_title_does_not_match():
    client = make_client(respond(movie_results=[movie("Inception: The Cobol Job", "2010-12-07")]))
    assert client.fetch("Inception", 2010)["status"] == FetchStatus.NOT_FOUND


def test_only_top_five_results_are_considered():
    others = [movie(f"Other {i}", "2010-01-01", id=i) for i in range(5)]
    client = make_client(respond(movie_results=[*others, movie("Inception", "2010-07-15")]))
    assert client.fetch("Inception", 2010)["status"] == FetchStatus.NOT_FOUND


def test_no_results_anywhere_is_not_found():
    client = make_client(respond())
    result = client.fetch("Nonexistent Title", 2020)
    assert result == {
        "status": FetchStatus.NOT_FOUND,
        "tmdb_id": None,
        "description": "",
        "media_type": None,
    }


def test_network_errors_on_every_attempt_is_error_not_not_found():
    def handler(request):
        raise httpx.ConnectError("TMDb unreachable", request=request)

    result = make_client(handler).fetch("Inception", 2010)
    assert result["status"] == FetchStatus.ERROR


def test_http_500_is_error():
    client = make_client(lambda request: httpx.Response(500))
    assert client.fetch("Inception", 2010)["status"] == FetchStatus.ERROR


def test_year_filter_is_sent_with_the_right_parameter_name():
    seen = []

    def handler(request):
        seen.append((request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={"results": []})

    make_client(handler).fetch("Anything", 2010)
    (movie_path, movie_params), (tv_path, tv_params), (_, movie_no_year), (_, tv_no_year) = seen
    assert movie_path.endswith("/search/movie") and movie_params["primary_release_year"] == "2010"
    assert tv_path.endswith("/search/tv") and tv_params["first_air_date_year"] == "2010"
    assert "primary_release_year" not in movie_no_year
    assert "first_air_date_year" not in tv_no_year
