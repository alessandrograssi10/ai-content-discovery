"""
Precompute embeddings for the golden queries, so the evaluation can run in CI
without calling OpenAI.

Re-run only when eval/golden_queries.json or EMBEDDING_MODEL changes:
    python -m eval.build_query_embeddings
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from src.services.openai_client import EMBEDDING_MODEL, LLMClient

load_dotenv()

EVAL_DIR = Path(__file__).parent
GOLDEN_PATH = EVAL_DIR / "golden_queries.json"
QUERY_EMBEDDINGS_PATH = EVAL_DIR / "query_embeddings.json"


def main() -> None:
    queries = [entry["query"] for entry in json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))]
    client = LLMClient()

    embeddings = {query: client.get_embedding(query) for query in queries}

    QUERY_EMBEDDINGS_PATH.write_text(
        json.dumps({"model": EMBEDDING_MODEL, "embeddings": embeddings}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(embeddings)} query embeddings ({EMBEDDING_MODEL}) to {QUERY_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
