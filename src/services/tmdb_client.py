"""
TMDb client, isolated in its own module (services/).

Matching logic empirically validated against all 21 seed titles (21/21
correct) before being written here.

Alternatives considered and rejected (documented for the record, not
implemented):
  - /search/multi as the sole search mechanism: uses a different ranking
    algorithm ("ngram" search, confirmed by TMDb staff) than /search/movie
    /search/tv (which default to "phrase" search, a more literal match).
    Empirically scored 20/21 on the seed, missing "8½" -- ngram's
    autocomplete-style matching mishandles the special character. No
    documented way to force /search/multi into "phrase" mode.

  - /search/multi as a fast path with a targeted cascade as fallback:
    closes the coverage gap (21/21) but adds real complexity (two separate
    search code paths, an extra parameter to reconcile differing result
    shapes) for a benefit -- saving one API call per title in the common
    case -- that is irrelevant at this catalog size (TMDb has no
    meaningful rate limit here).

  - Fuzzy title scoring (SequenceMatcher) as a secondary check when no
    exact title match is found: initially built as a safety net, then
    empirically found to contribute ZERO accepted matches across all 21
    titles -- every title was resolved by an exact title match at some
    point in the movie/tv + year/no-year cascade (confirmed: every row of
    the validation run shows ratio 1.00). Removed for the same reason as
    the multi+cascade hybrid above: complexity without a demonstrated
    benefit on this dataset. A larger, messier real-world catalog would
    likely need this safety net back; this one didn't.

Matching logic, in attempt order:
  1. /search/movie with year filter
  2. /search/tv with year filter (titles like Breaking Bad, The Office,
     Fleabag, Chernobyl are series, not movies)
  3. /search/movie WITHOUT year filter (for cases where TMDb records a
     different year than the seed, e.g. premiere vs national release --
     historically confirmed for Casablanca: premiere Nov 26 1942 in New
     York, US national release Jan 23 1943)
  4. /search/tv WITHOUT year filter (symmetric fallback for the same kind
     of discrepancy on a TV show -- not needed by any of the 21 seed
     titles, but no principled reason to only allow this recovery path
     for movies)

For each attempt, among the top 5 results: EXACT title match
(case-insensitive) + year within YEAR_TOLERANCE -> accepted. No fuzzy
fallback (see above).

YEAR_TOLERANCE = 1 is calibrated on the real seed data: the 3
discrepancies observed (Casablanca, Funny Games, Coherence) are all
exactly 1 year off, due to festival/premiere dates vs official release
dates.

Known residual risks (documented, not hidden):
  - Exact-title homonym risk: two genuinely different productions sharing
    the literal same title within YEAR_TOLERANCE could be confused.
  - Only the first 5 results per search are scanned; a correct match
    ranked lower would be missed.
  - No fuzzy fallback: a title with a minor spelling/punctuation mismatch
    against TMDb's exact string (not present in this seed) would go
    unmatched and fall through to FetchStatus.NOT_FOUND rather than being
    approximately recovered.

Rate limit: TMDb has had no hard rate limit since 2019 (~40-50 requests/
second anti-abuse ceiling). With 21 titles this is never approached.
"""

import os

import httpx

from src.models import FetchStatus

TMDB_BASE_URL = "https://api.themoviedb.org/3"

YEAR_TOLERANCE = 1


class TMDbClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TMDB_API_KEY is missing. Set it in your .env file (see .env.example)."
            )
        self._client = httpx.Client(
            base_url=TMDB_BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "accept": "application/json",
            },
            timeout=10.0,
        )

    def fetch(self, title: str, expected_year: int) -> dict:
        """
        Search for a title on TMDb and return a dict describing the outcome,
        ready to be unpacked into a FetchedContent:
            {"status": FetchStatus, "tmdb_id": int|None,
             "description": str, "media_type": str|None}

        FetchStatus.ERROR is returned only if EVERY attempt failed due to a
        network/HTTP error -- distinct from FetchStatus.NOT_FOUND, which
        means TMDb responded successfully but no matching title was found.
        """
        attempts = [
            ("movie", expected_year),
            ("tv", expected_year),
            ("movie", None),
            ("tv", None),
        ]

        any_network_error = False

        for endpoint, year_used in attempts:
            try:
                results = self._search(endpoint, title, year_used)
            except httpx.HTTPError:
                any_network_error = True
                continue

            if not results:
                continue

            match = self._find_exact_match(results, endpoint, title, expected_year)
            if match is not None:
                return {
                    "status": FetchStatus.OK,
                    "tmdb_id": match.get("id"),
                    "description": (match.get("overview") or "").strip(),
                    "media_type": endpoint,
                }

        if any_network_error:
            return {
                "status": FetchStatus.ERROR,
                "tmdb_id": None,
                "description": "",
                "media_type": None,
            }

        return {
            "status": FetchStatus.NOT_FOUND,
            "tmdb_id": None,
            "description": "",
            "media_type": None,
        }

    def _search(self, endpoint: str, title: str, year: int | None) -> list[dict]:
        params = {"query": title, "include_adult": False, "language": "en-US"}
        if year is not None:
            year_param = "primary_release_year" if endpoint == "movie" else "first_air_date_year"
            params[year_param] = year

        response = self._client.get(f"/search/{endpoint}", params=params)
        response.raise_for_status()
        return response.json().get("results", [])

    def _find_exact_match(
        self, results: list[dict], endpoint: str, title: str, expected_year: int
    ) -> dict | None:
        """Exact title match (case-insensitive) + year within tolerance."""
        for candidate in results[:5]:
            tmdb_title = (candidate.get("title") or candidate.get("name") or "").strip().lower()
            if tmdb_title == title.strip().lower():
                found_year = self._extract_year(candidate, endpoint)
                if found_year is not None and abs(found_year - expected_year) <= YEAR_TOLERANCE:
                    return candidate
        return None

    @staticmethod
    def _extract_year(result: dict, endpoint: str) -> int | None:
        date = result.get("release_date") if endpoint == "movie" else result.get("first_air_date")
        return int(date[:4]) if date else None

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
