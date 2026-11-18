"""Retrieval quality check: does the fact needed for the answer reach the LLM context?

For each question we look for the expected strings in the top-k retrieved chunks and
report hit@k and MRR for vector-only, full-text-only and hybrid (RRF) retrieval.

    python -m eval.run_eval [--k 8] [--answers]   # --answers also generates LLM replies
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.db import Database, SearchHit
from app.embeddings import create_embedder
from app.retrieval import Retriever, rrf_fuse

QUESTIONS = Path(__file__).with_name("questions.json")


def first_hit_rank(hits: list[SearchHit], expect: list[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        haystack = f"{hit.title}\n{hit.heading}\n{hit.content}".lower()
        # every expected item must be present; "a | b" means either alternative is fine
        if all(any(alt.strip().lower() in haystack for alt in e.split(" | ")) for e in expect):
            return rank
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="top-k passed to the LLM (default: TOP_K)")
    parser.add_argument("--answers", action="store_true", help="also print LLM answers")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions for --answers")
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.top_k
    db = Database(settings.database_url)
    await db.connect()
    embedder = create_embedder(settings)
    retriever = Retriever(db, embedder, top_k=k, candidates_k=settings.candidates_k, rrf_k=settings.rrf_k)
    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    totals = {"vector": [], "fulltext": [], "hybrid": []}
    misses = []
    for case in cases:
        q, expect = case["q"], case["expect"]
        vec = await db.vector_search(await embedder.embed_query(q), settings.candidates_k)
        fts = await db.text_search(q, settings.candidates_k)
        hybrid = rrf_fuse([vec, fts], k=settings.rrf_k, limit=k, max_per_url=retriever.max_per_url)
        ranks = {
            "vector": first_hit_rank(vec[:k], expect),
            "fulltext": first_hit_rank(fts[:k], expect),
            "hybrid": first_hit_rank(hybrid, expect),
        }
        for name, rank in ranks.items():
            totals[name].append(rank)
        mark = "✓" if ranks["hybrid"] else "✗"
        cols = " ".join(f"{name}@{rank or '-':<2}" for name, rank in ranks.items())
        print(f"{mark} {cols} {q}")
        if not ranks["hybrid"]:
            misses.append((q, expect, hybrid[:3]))

    print(f"\n{len(cases)} questions, k={k}, embeddings={embedder.model_name}")
    for name, ranks in totals.items():
        hit = sum(r is not None for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        print(f"  {name:<9} hit@{k}={hit:.0%}  MRR={mrr:.3f}")

    for q, expect, top in misses:
        print(f"\nMISS: {q} (expected {expect})")
        for h in top:
            print(f"   - {h.url} | {h.header[:80]}")

    if args.answers:
        from app.llm import create_llm
        from app.rag import RAGService
        from app.tracing import Tracing

        rag = RAGService(db, retriever, create_llm(settings), Tracing(settings), history_turns=0)
        for i, case in enumerate(cases[: args.limit]):
            answer = await rag.answer(chat_id=-(10_000 + i), question=case["q"])
            print(f"\n### {case['q']}  ({answer.latency:.1f} s, {answer.model})\n{answer.text}")
            await db.clear_history(-(10_000 + i))
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
