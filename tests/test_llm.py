"""LLM adapters against a local stub HTTP server: request shape + response parsing, no network."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from app.config import Settings
from app.db import SearchHit
from app.llm import AnthropicLLM, LLMResult, OpenAICompatibleLLM
from app.rag import RAGService
from app.tracing import Tracing


class _Stub(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    response: ClassVar[dict] = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Stub.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        payload = json.dumps(_Stub.response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_server():
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Stub.requests = []
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_anthropic_request_and_parsing(stub_server):
    _Stub.response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": "ИНН 6317139806 [1]"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    settings = Settings(_env_file=None, anthropic_api_key="test", anthropic_base_url=stub_server)
    llm = AnthropicLLM(settings)
    result = asyncio.run(llm.generate("system", [{"role": "user", "content": "ИНН?"}]))

    assert result.text == "ИНН 6317139806 [1]"
    assert result.usage == {"input_tokens": 100, "output_tokens": 10}
    req = _Stub.requests[0]
    assert req["path"].startswith("/v1/messages")
    assert req["body"]["model"] == "claude-opus-5-5"
    assert req["body"]["output_config"] == {"effort": "low"}
    assert req["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    assert req["body"]["system"] == "system"


def test_anthropic_refusal(stub_server):
    _Stub.response = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [],
        "stop_reason": "refusal",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }
    settings = Settings(_env_file=None, anthropic_api_key="test", anthropic_base_url=stub_server)
    result = asyncio.run(AnthropicLLM(settings).generate("s", [{"role": "user", "content": "x"}]))
    assert result.refused


def test_openai_compatible(stub_server):
    _Stub.response = {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "local-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Ответ [2]"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }
    settings = Settings(_env_file=None, llm_model="local-model", openai_base_url=f"{stub_server}/v1")
    result = asyncio.run(OpenAICompatibleLLM(settings).generate("sys", [{"role": "user", "content": "q"}]))
    assert result.text == "Ответ [2]"
    body = _Stub.requests[0]["body"]
    assert body["messages"][0] == {"role": "system", "content": "sys"}


# ----------------------------------------------------------------------------- RAG service
class FakeDB:
    def __init__(self):
        self.messages: list[tuple[int, str, str]] = []

    async def history(self, chat_id, turns):
        return [(r, c) for cid, r, c in self.messages if cid == chat_id][-turns * 2 :]

    async def add_message(self, chat_id, role, content):
        self.messages.append((chat_id, role, content))


class FakeRetriever:
    def __init__(self):
        self.queries: list[list[str]] = []

    async def search(self, queries):
        self.queries.append(queries)
        return [
            SearchHit(1, "https://osmi-it.ru/requisites/", "Реквизиты", "", "ИНН/КПП 6317139806/502701001"),
            SearchHit(2, "https://osmi-it.ru/contacts/", "Контакты", "", "Телефон для связи: +7 (495) 128-19-89"),
        ]


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.calls = []

    async def generate(self, system, messages):
        self.calls.append(messages)
        return LLMResult(text="ИНН компании — **6317139806** [1].", model="fake")


def test_rag_answer_cites_only_used_sources_and_keeps_history():
    db, retriever, llm = FakeDB(), FakeRetriever(), FakeLLM()
    rag = RAGService(db, retriever, llm, Tracing(Settings(_env_file=None)), history_turns=2)

    first = asyncio.run(rag.answer(1, "Какой ИНН?"))
    assert "<b>6317139806</b>" in first.text
    assert [s.url for s in first.cited] == ["https://osmi-it.ru/requisites/"]
    assert "contacts" not in first.text  # uncited source is not listed
    assert '<source id="1"' in llm.calls[0][-1]["content"]

    asyncio.run(rag.answer(1, "а КПП?"))
    assert retriever.queries[-1] == ["а КПП?", "Какой ИНН?\nа КПП?"]  # follow-up searched with context
    roles = [m["role"] for m in llm.calls[-1]]
    assert roles == ["user", "assistant", "user"]
    assert "Источники" not in llm.calls[-1][1]["content"]  # history stores the answer body only


def test_rag_without_llm_returns_quotes():
    rag = RAGService(FakeDB(), FakeRetriever(), None, Tracing(Settings(_env_file=None)))
    answer = asyncio.run(rag.answer(5, "ИНН"))
    assert "6317139806" in answer.text and "Источники" in answer.text


def test_sources_attributed_when_model_skips_citations():
    from app.rag import attribute_sources, group_sources

    hits = [
        SearchHit(
            1,
            "https://osmi-it.ru/decisions/",
            "Решения",
            "Горизонтальные ИИ-решения",
            "ИИ-хаб поддержки:\nАвтоматизация поддержки, которая работает без выходных\n"
            "ИИ-ассистент по закупкам:\nЗакупочный цикл — быстрее, точнее, без рутины",
        ),
        SearchHit(2, "https://osmi-it.ru/", "Главная", "", "Веб-разработка, дизайн, аналитика"),
    ]
    answer = (
        "• ИИ-хаб поддержки: Автоматизация поддержки без выходных\n"
        "• ИИ-ассистент по закупкам: Закупочный цикл — быстрее, точнее, без рутины"
    )
    cited = attribute_sources(answer, group_sources(hits))
    assert [s.url for s in cited] == ["https://osmi-it.ru/decisions/"]
