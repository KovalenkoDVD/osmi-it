"""Build / refresh the knowledge base: crawl -> chunk -> embed -> store.

Incremental: a document is re-embedded only when its content hash changed, pages
that disappeared from the sites are deleted (unless the crawl looks broken).

    python -m app.ingest            # incremental refresh
    python -m app.ingest --full     # re-embed everything
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass

from app.chunking import build_chunks
from app.config import Settings, get_settings
from app.crawler import Crawler
from app.db import Database
from app.embeddings import create_embedder

log = logging.getLogger(__name__)

# Refuse to delete stale documents if the crawl returned much less than we already have:
# most likely the site was down or blocked us, not that half the pages were removed.
MIN_CRAWL_SHARE = 0.6


@dataclass
class IngestReport:
    crawled: int = 0
    documents: int = 0
    chunks: int = 0
    embedded_documents: int = 0
    unchanged_documents: int = 0
    deleted_documents: int = 0
    seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"страниц скачано: {self.crawled}, документов: {self.documents}, чанков: {self.chunks}; "
            f"обновлено: {self.embedded_documents}, без изменений: {self.unchanged_documents}, "
            f"удалено: {self.deleted_documents}; {self.seconds:.0f} с"
        )


async def run_ingest(db: Database, settings: Settings, embedder=None, *, full: bool = False) -> IngestReport:
    started = time.monotonic()
    report = IngestReport()
    embedder = embedder or create_embedder(settings)

    crawler = Crawler(
        settings.start_urls,
        settings.allowed_hosts,
        max_pages=settings.crawl_max_pages,
        concurrency=settings.crawl_concurrency,
        timeout=settings.crawl_timeout,
        user_agent=settings.crawl_user_agent,
    )
    pages = await crawler.crawl()
    report.crawled = len(pages)
    docs = build_chunks(pages, settings.chunk_size, settings.chunk_overlap)
    report.documents = len(docs)
    report.chunks = sum(len(d.chunks) for d in docs)
    log.info("Built %d documents / %d chunks", report.documents, report.chunks)

    existing = await db.document_hashes()
    todo = [d for d in docs if full or existing.get(d.url) != d.content_hash]
    report.unchanged_documents = len(docs) - len(todo)
    total_chunks = sum(len(d.chunks) for d in todo)
    log.info("Embedding %d changed documents (%d chunks)", len(todo), total_chunks)

    done_chunks = 0
    for i, doc in enumerate(todo, start=1):
        vectors = await embedder.embed_documents([c.embed_text for c in doc.chunks])
        await db.upsert_document(
            url=doc.url,
            host=doc.host,
            kind=doc.kind,
            title=doc.title,
            content_hash=doc.content_hash,
            chunks=[(c.index, c.heading, c.content, v) for c, v in zip(doc.chunks, vectors, strict=True)],
        )
        done_chunks += len(doc.chunks)
        if i % 20 == 0 or i == len(todo):
            log.info("  %d/%d documents, %d/%d chunks", i, len(todo), done_chunks, total_chunks)
    report.embedded_documents = len(todo)

    stale = sorted(set(existing) - {d.url for d in docs})
    if stale:
        if existing and len(docs) < MIN_CRAWL_SHARE * len(existing):
            log.warning(
                "Crawl returned %d documents vs %d stored: keeping %d stale documents just in case",
                len(docs),
                len(existing),
                len(stale),
            )
        else:
            report.deleted_documents = await db.delete_documents(stale)

    await db.set_meta("last_ingest", time.strftime("%Y-%m-%d %H:%M:%S"))
    report.seconds = time.monotonic() - started
    log.info("Ingest finished: %s", report)
    return report


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Crawl the sites and (re)build the vector knowledge base")
    parser.add_argument("--full", action="store_true", help="re-embed all documents, not only changed ones")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    db = Database(settings.database_url)
    await db.connect()
    try:
        embedder = create_embedder(settings)
        await db.init_schema(embedder.model_name, embedder.dim)
        report = await run_ingest(db, settings, embedder, full=args.full)
        print(report)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(_main())
