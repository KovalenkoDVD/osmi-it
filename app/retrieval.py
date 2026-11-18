"""Hybrid retrieval: pgvector semantic search + Postgres full-text search, fused with RRF.

Dense vectors catch paraphrases ("сколько стоит" ~ "тарифы"); full-text search catches
exact tokens embeddings are bad at: INN/OGRN numbers, e-mails, product and client names.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from app.db import Database, SearchHit
from app.embeddings import Embedder


def rrf_fuse(
    rankings: list[list[SearchHit]],
    k: int = 60,
    limit: int | None = None,
    max_per_url: int | None = None,
    weights: list[float] | None = None,
) -> list[SearchHit]:
    """Weighted Reciprocal Rank Fusion: score = sum(weight / (k + rank)).

    ``max_per_url`` keeps the context diverse: one long page can't take every slot.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    hits: dict[int, SearchHit] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=scores.get, reverse=True)
    result: list[SearchHit] = []
    per_url: dict[str, int] = {}
    for chunk_id in ordered:
        hit = hits[chunk_id]
        if max_per_url and per_url.get(hit.url, 0) >= max_per_url:
            continue
        per_url[hit.url] = per_url.get(hit.url, 0) + 1
        result.append(replace(hit, score=scores[chunk_id]))
        if limit and len(result) >= limit:
            break
    return result


def section_key(heading: str) -> tuple[str, ...]:
    """Top two heading levels, e.g. ("Коробочные ИИ-решения", "Вертикальные ИИ-решения")."""
    parts = [p for p in heading.split(" > ") if p]
    return tuple(parts[:2]) if len(parts) >= 2 else ()


@dataclass
class Retriever:
    db: Database
    embedder: Embedder
    top_k: int = 8
    candidates_k: int = 30
    rrf_k: int = 60
    max_per_url: int = 3
    expand_top: int = 3  # complete the sections of the best hits with neighbouring chunks

    async def search(self, queries: list[str]) -> list[SearchHit]:
        """Search with the question first and optional extra phrasings after it.

        Extra phrasings (question + previous question, for follow-ups) get half the weight,
        so the previous topic helps resolve "а сколько это стоит?" but does not take over.
        """
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return []
        tasks, weights = [], []
        for i, q in enumerate(queries):
            tasks += [self._vector(q), self.db.text_search(q, self.candidates_k)]
            weights += [1.0 if i == 0 else 0.5] * 2
        rankings = await asyncio.gather(*tasks)
        hits = rrf_fuse(list(rankings), k=self.rrf_k, limit=self.top_k, max_per_url=self.max_per_url, weights=weights)
        return await self._expand_sections(hits)

    async def _expand_sections(self, hits: list[SearchHit]) -> list[SearchHit]:
        """A list split over two chunks ("Вертикальные ИИ-решения": девелопмент | промышленность)
        is returned whole: neighbours of a top hit from the same section are added after it."""
        top = hits[: self.expand_top]
        have = {h.chunk_id for h in hits}
        extra: dict[int, list[SearchHit]] = {}
        for parent_id, nb in await self.db.neighbours([h.chunk_id for h in top]):
            parent = next(h for h in top if h.chunk_id == parent_id)
            key = section_key(parent.heading)
            if nb.chunk_id in have or not key or section_key(nb.heading) != key:
                continue
            have.add(nb.chunk_id)
            extra.setdefault(parent_id, []).append(replace(nb, score=parent.score))
        result: list[SearchHit] = []
        for hit in hits:
            group = sorted([hit, *extra.get(hit.chunk_id, [])], key=lambda h: h.chunk_index)
            result.extend(group)
        return result

    async def _vector(self, query: str) -> list[SearchHit]:
        embedding = await self.embedder.embed_query(query)
        return await self.db.vector_search(embedding, self.candidates_k)
