"""
OpenAI wrapper, isolated in its own module (services/).

Design decision (see project-wide rationale): a SINGLE provider for both
generation AND embeddings, instead of Anthropic (which has no native
embedding model -- Anthropic recommends Voyage AI as a third-party
partner, which would mean two providers instead of one). Consolidating on
OpenAI reduces dependencies for a time-boxed prototype.

Models used (verified current as of July 2026 -- the pre-GPT-5 family,
including gpt-4o-mini, was retired when the GPT-5.4 family launched in
March 2026):

  - gpt-5.4-mini            -> enrichment (structured JSON output).
    Chosen over the cheaper gpt-5.4-nano because enrichment requires some
    judgment (inferring mood/themes, not pure extraction), and at only 21
    titles the cost difference between the two tiers is a fraction of a
    cent -- irrelevant, so quality margin wins over marginal savings.

  - text-embedding-3-small  -> embeddings for semantic search. Still the
    current OpenAI default for this use case as of mid-2026.

No LangChain: direct SDK calls for full transparency instead of relying on a framework's abstraction.
"""

import json
import os

from openai import OpenAI

ENRICHMENT_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def enrich_content(self, title: str, year: int, description: str, prompt_template: str) -> dict:
        """
        Generate structured metadata (mood, themes, target audience, etc.)
        from the raw TMDb description.

        Uses OpenAI's JSON mode to force a parsable structured output every
        time -- the prompt alone does not guarantee valid JSON; prompt and
        structured output work together, not one instead of the other.

        temperature=0.3: low value chosen deliberately for consistency
        across the 21 items. Enrichment requires judgment, not creativity --
        the same title should produce the same metadata regardless of when
        it is processed.
        """
        prompt = prompt_template.format(title=title, year=year, description=description)

        response = self.client.chat.completions.create(
            model=ENRICHMENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        raw_content = response.choices[0].message.content
        return json.loads(raw_content)

    def get_embedding(self, text: str) -> list[float]:
        """
        Generate the embedding vector for a piece of text. Used both when
        indexing the catalog (Stage 2, once per title) and when serving a
        search query (Stage 3, once per request) -- always the same model
        in both cases, since embeddings from different models are not
        comparable to each other.
        """
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
