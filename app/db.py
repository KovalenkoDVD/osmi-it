"""PostgreSQL + pgvector storage: documents, chunks (vector + full-text), chat history."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import asyncpg
from pgvector.asyncpg import register_vector

log = logging.getLogger(__name__)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id           serial PRIMARY KEY,
    url          text NOT NULL UNIQUE,
    host         text NOT NULL,
    kind         text NOT NULL,
    title        text NOT NULL,
    content_hash text NOT NULL,
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document_id int  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index int  NOT NULL,
    heading     text NOT NULL DEFAULT '',
    content     text NOT NULL,
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('russian', heading), 'A') ||
                    to_tsvector('russian', content)
                ) STORED,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS messages (
    id         bigserial PRIMARY KEY,
    chat_id    bigint NOT NULL,
    role       text   NOT NULL CHECK (role IN ('user', 'assistant')),
    content    text   NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_chat_idx ON messages (chat_id, id DESC);
"""


@dataclass
class SearchHit:
    chunk_id: int
    url: str
    title: str
    heading: str
    content: str
    score: float = 0.0
    chunk_index: int = 0

    @property
    def header(self) -> str:
        return self.title if not self.heading else f"{self.title} > {self.heading}"


class Database:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        # The extension must exist before the pool registers the vector codec.
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=10, init=register_vector)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    # ------------------------------------------------------------------ schema
    async def init_schema(self, embedding_model: str, dim: int) -> None:
        """Create tables; wipe the vectors if the embedding model/dimension changed."""
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(4242)")
            meta = (
                {r["key"]: r["value"] for r in await conn.fetch("SELECT * FROM kb_meta")}
                if await conn.fetchval("SELECT to_regclass('kb_meta') IS NOT NULL")
                else {}
            )
            if meta and (meta.get("embedding_model") != embedding_model or meta.get("embedding_dim") != str(dim)):
                log.warning(
                    "Embedding model changed (%s/%s -> %s/%s): dropping the knowledge base, re-ingest required",
                    meta.get("embedding_model"),
                    meta.get("embedding_dim"),
                    embedding_model,
                    dim,
                )
                await conn.execute("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS documents;")
            await conn.execute(SCHEMA.format(dim=int(dim)))
            await conn.execute(
                """INSERT INTO kb_meta(key, value) VALUES ('embedding_model', $1), ('embedding_dim', $2)
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                embedding_model,
                str(dim),
            )

    # ------------------------------------------------------------------- stats
    async def stats(self) -> dict[str, int | str | None]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT (SELECT count(*) FROM documents) AS documents,
                          (SELECT count(*) FROM chunks) AS chunks,
                          (SELECT max(updated_at) FROM documents) AS updated_at"""
            )
            hosts = await conn.fetch("SELECT host, count(*) AS n FROM documents GROUP BY host ORDER BY host")
        return {
            "documents": row["documents"],
            "chunks": row["chunks"],
            "updated_at": row["updated_at"].strftime("%Y-%m-%d %H:%M UTC") if row["updated_at"] else None,
            "hosts": ", ".join(f"{r['host']} ({r['n']})" for r in hosts),
        }

    async def set_meta(self, key: str, value: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO kb_meta(key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = $2",
                key,
                value,
            )

    # ------------------------------------------------------------------ ingest
    async def document_hashes(self) -> dict[str, str]:
        async with self.pool.acquire() as conn:
            return {r["url"]: r["content_hash"] for r in await conn.fetch("SELECT url, content_hash FROM documents")}

    async def upsert_document(
        self,
        *,
        url: str,
        host: str,
        kind: str,
        title: str,
        content_hash: str,
        chunks: list[tuple[int, str, str, list[float]]],
    ) -> None:
        """Replace a document and all its chunks atomically. chunks: (index, heading, content, embedding)."""
        async with self.pool.acquire() as conn, conn.transaction():
            doc_id = await conn.fetchval(
                """INSERT INTO documents(url, host, kind, title, content_hash, updated_at)
                   VALUES ($1, $2, $3, $4, $5, now())
                   ON CONFLICT (url) DO UPDATE
                     SET host = EXCLUDED.host, kind = EXCLUDED.kind, title = EXCLUDED.title,
                         content_hash = EXCLUDED.content_hash, updated_at = now()
                   RETURNING id""",
                url,
                host,
                kind,
                title,
                content_hash,
            )
            await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
            await conn.executemany(
                """INSERT INTO chunks(document_id, chunk_index, heading, content, embedding)
                   VALUES ($1, $2, $3, $4, $5)""",
                [(doc_id, idx, heading, content, emb) for idx, heading, content, emb in chunks],
            )

    async def delete_documents(self, urls: list[str]) -> int:
        if not urls:
            return 0
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM documents WHERE url = ANY($1::text[])", urls)
        return int(result.split()[-1])

    # ------------------------------------------------------------------ search
    async def vector_search(self, embedding: list[float], limit: int) -> list[SearchHit]:
        async with self.pool.acquire() as conn:
            await conn.execute("SET hnsw.ef_search = 100")
            rows = await conn.fetch(
                """SELECT c.id, d.url, d.title, c.heading, c.content, c.chunk_index,
                          1 - (c.embedding <=> $1) AS score
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   ORDER BY c.embedding <=> $1
                   LIMIT $2""",
                embedding,
                limit,
            )
        return [_hit(r) for r in rows]

    async def text_search(self, query: str, limit: int) -> list[SearchHit]:
        """Lexical search scored like BM25-lite: sum of IDF of the matched query terms.

        Plain ``ts_rank`` over an OR-query favours long chunks that repeat frequent words
        ("платформа", "OSMI"); weighting each term by its rarity lets a short chunk with
        the rare, decisive word ("сотрудничество", an INN number) win.
        """
        terms = query_terms(query)
        if not terms:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """WITH n AS (SELECT count(*)::float AS total FROM chunks),
                        terms AS (
                            SELECT q,
                                   ln(1 + n.total / (1 + (SELECT count(*) FROM chunks c2 WHERE c2.tsv @@ q))) AS idf
                            FROM unnest($1::text[]) AS t, LATERAL to_tsquery('russian', t) AS q, n
                            WHERE numnode(q) > 0
                        )
                   SELECT c.id, d.url, d.title, c.heading, c.content, c.chunk_index,
                          sum(terms.idf) + ts_rank_cd(c.tsv, to_tsquery('russian', $2), 1) AS score
                   FROM chunks c
                   JOIN terms ON c.tsv @@ terms.q
                   JOIN documents d ON d.id = c.document_id
                   GROUP BY c.id, d.url, d.title
                   ORDER BY score DESC
                   LIMIT $3""",
                terms,
                " | ".join(terms),
                limit,
            )
        return [_hit(r) for r in rows]

    async def neighbours(self, chunk_ids: list[int]) -> list[tuple[int, SearchHit]]:
        """Adjacent chunks (index ± 1) of the same document: (source chunk id, neighbour)."""
        if not chunk_ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT c1.id AS parent, c2.id, d.url, d.title, c2.heading, c2.content, c2.chunk_index,
                          0.0::float AS score
                   FROM chunks c1
                   JOIN chunks c2 ON c2.document_id = c1.document_id
                                 AND c2.chunk_index IN (c1.chunk_index - 1, c1.chunk_index + 1)
                   JOIN documents d ON d.id = c1.document_id
                   WHERE c1.id = ANY($1::bigint[])""",
                chunk_ids,
            )
        return [(r["parent"], _hit(r)) for r in rows]

    # ----------------------------------------------------------------- history
    async def add_message(self, chat_id: int, role: str, content: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages(chat_id, role, content) VALUES ($1, $2, $3)", chat_id, role, content
            )

    async def history(self, chat_id: int, turns: int) -> list[tuple[str, str]]:
        if turns <= 0:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT role, content FROM messages
                   WHERE chat_id = $1 AND created_at > now() - interval '1 day'
                   ORDER BY id DESC LIMIT $2""",
                chat_id,
                turns * 2,
            )
        return [(r["role"], r["content"]) for r in reversed(rows)]

    async def clear_history(self, chat_id: int) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM messages WHERE chat_id = $1", chat_id)


def _hit(r) -> SearchHit:
    return SearchHit(r["id"], r["url"], r["title"], r["heading"], r["content"], float(r["score"]), r["chunk_index"])


# Words that carry no meaning for lexical search (questions are phrased in natural language).
_STOPWORDS = {
    "какой",
    "какая",
    "какие",
    "каким",
    "какое",
    "каких",
    "какую",
    "который",
    "которые",
    "что",
    "чем",
    "кто",
    "где",
    "когда",
    "как",
    "есть",
    "ли",
    "это",
    "этот",
    "эта",
    "эти",
    "для",
    "при",
    "про",
    "под",
    "над",
    "или",
    "так",
    "вас",
    "ваш",
    "ваша",
    "ваши",
    "вам",
    "вы",
    "мне",
    "меня",
    "можно",
    "нужно",
    "сколько",
    "почему",
    "зачем",
    "расскажи",
    "расскажите",
    "подскажи",
    "подскажите",
    "скажи",
    "скажите",
    "компания",
    "компании",
    "the",
    "and",
    "what",
    "how",
    "about",
    "был",
    "была",
    "были",
    "будет",
    "все",
    "всё",
    "его",
    "её",
    "она",
    "они",
    "оно",
    "там",
    "тут",
    "уже",
    "еще",
    "ещё",
    "также",
    "тоже",
    "очень",
    "без",
}
_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё\-]*")


def query_terms(text: str) -> list[str]:
    """Meaningful lowercase words of a natural-language question (stop words removed)."""
    words: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        for part in token.split("-"):
            if len(part) < 2 or part in _STOPWORDS or (len(part) < 3 and not part.isdigit()):
                continue
            if part not in words:
                words.append(part)
    return words


def build_tsquery(text: str) -> str:
    """OR-query of meaningful words: natural questions rarely match with AND semantics."""
    return " | ".join(query_terms(text))
