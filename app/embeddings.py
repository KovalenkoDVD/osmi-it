"""Embedding backends.

``fastembed`` (default) runs a multilingual ONNX model locally on CPU: no API key,
and the knowledge base never leaves the server. ``openai`` works with any
OpenAI-compatible ``/embeddings`` endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Protocol

from app.config import Settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


# Asymmetric models need instruction prefixes; fastembed does not add them itself.
_PREFIXES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
    "qwen3-embedding": (
        "Instruct: Given a question, retrieve passages from the company website that answer it\nQuery: ",
        "",
    ),
}


# Good multilingual (Russian) models that fastembed does not ship out of the box:
# registered on the fly from the ONNX weights published in their HF repos.
_CUSTOM_MODELS: dict[str, int] = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
}


def _register_custom_model(model_name: str) -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    supported = {m["model"].lower() for m in TextEmbedding.list_supported_models()}
    if model_name.lower() in supported or model_name not in _CUSTOM_MODELS:
        return
    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=model_name),
        dim=_CUSTOM_MODELS[model_name],
        model_file="onnx/model.onnx",
    )


def _prefixes(model: str) -> tuple[str, str]:
    name = model.lower()
    for key, value in _PREFIXES.items():
        if key in name:
            return value
    return "", ""


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dim: int, cache_dir: str | None, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.dim = dim
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = _prefixes(model_name)
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                log.info("Loading embedding model %s (first run downloads it)...", self.model_name)
                _register_custom_model(self.model_name)
                self._model = TextEmbedding(self.model_name, cache_dir=self.cache_dir)
            return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        with self._lock:  # onnxruntime session is not re-entrant friendly; keep it serial
            return [vec.tolist() for vec in model.embed(texts, batch_size=self.batch_size)]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed, [self.passage_prefix + t for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._embed, [self.query_prefix + text]))[0]

    async def warmup(self) -> None:
        await asyncio.to_thread(self._get_model)


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.model_name = settings.openai_embedding_model
        self.dim = settings.embedding_dim
        self.batch_size = 128

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = await self.client.embeddings.create(
                model=self.model_name, input=texts[i : i + self.batch_size], dimensions=self.dim
            )
            out.extend(item.embedding for item in resp.data)
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def warmup(self) -> None:
        return None


def create_embedder(settings: Settings) -> FastEmbedEmbedder | OpenAIEmbedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    return FastEmbedEmbedder(
        settings.embedding_model,
        settings.embedding_dim,
        settings.fastembed_cache_dir,
        settings.embedding_batch_size,
    )
