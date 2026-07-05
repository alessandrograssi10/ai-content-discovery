"""
Stage 1 — fetch description for all seed titles from TMDb.

Reads  : data/content_sample.csv
Writes : output/fetched_content.json

Run with:
    python -m src.fetch_data
"""

import csv
import json
from pathlib import Path

from dotenv import load_dotenv

from src.models import FetchedContent, FetchStatus, SeedItem
from src.services.tmdb_client import TMDbClient

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "content_sample.csv"
OUTPUT_PATH = ROOT / "output" / "fetched_content.json"


def load_seed_items(path: Path) -> list[SeedItem]:
    """Parse the seed CSV into validated SeedItem objects."""
    with open(path, newline="", encoding="utf-8") as f:
        return [SeedItem(**row) for row in csv.DictReader(f)]


def main() -> None:
    seed_items = load_seed_items(DATA_PATH)
    print(f"Loaded {len(seed_items)} seed titles. Starting fetch...\n")

    results: list[FetchedContent] = []

    with TMDbClient() as client:
        for item in seed_items:
            outcome = client.fetch(item.title, item.year)
            fetched = FetchedContent(
                content_id=item.content_id,
                title=item.title,
                year=item.year,
                description=outcome["description"],
                tmdb_id=outcome["tmdb_id"],
                media_type=outcome["media_type"],
                fetch_status=outcome["status"],
            )
            label = outcome["status"].value.upper()
            print(f"[{label:<9}] {item.title} ({item.year})")
            results.append(fetched)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results],
            f,
            indent=2,
            ensure_ascii=False,
        )

    ok = sum(1 for r in results if r.fetch_status == FetchStatus.OK)
    not_found = sum(1 for r in results if r.fetch_status == FetchStatus.NOT_FOUND)
    errors = sum(1 for r in results if r.fetch_status == FetchStatus.ERROR)

    print(f"\nDone. {ok} OK / {not_found} not found / {errors} errors")
    print(f"Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
