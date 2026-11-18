"""LLM backends: Anthropic Claude (default) or any OpenAI-compatible chat endpoint."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from app.config import Settings

log = logging.getLogger(__name__)

Message = dict[str, str]
# Local reasoning models (Qwen3, DeepSeek-R1 via Ollama) may inline their reasoning.
_THINK = re.compile(r"<think>.*?</think>", re.S)  # {"role": "user" | "assistant", "content": "..."}


class LLMError(RuntimeError):
    """Raised with a user-safe message when the model call fails."""


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    refused: bool = False


class LLM(Protocol):
    model: str

    async def generate(self, system: str, messages: list[Message]) -> LLMResult: ...


class AnthropicLLM:
    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key or None,  # None -> SDK resolves env / `ant auth` profile
            base_url=settings.anthropic_base_url or None,
            timeout=settings.llm_timeout,
            max_retries=3,
        )
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.effort = settings.llm_effort
        self.fallback = settings.llm_refusal_fallback

    async def generate(self, system: str, messages: list[Message]) -> LLMResult:
        anthropic = self._anthropic
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallback:
            # Server-side retry on another model if a safety classifier declines the request.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        try:
            response = await self.client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Неверный ключ Anthropic API (ANTHROPIC_API_KEY).") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("Ключ Anthropic API не имеет доступа к модели или региону.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Модель {self.model} не найдена.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Превышен лимит запросов к LLM, попробуйте через минуту.") from exc
        except anthropic.BadRequestError as exc:
            log.error("Anthropic bad request: %s", exc)
            raise LLMError("LLM отклонила запрос (400).") from exc
        except anthropic.APIStatusError as exc:
            log.error("Anthropic API error %s: %s", exc.status_code, exc)
            raise LLMError("Сервис LLM временно недоступен.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Нет соединения с Anthropic API.") from exc

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if response.stop_reason == "refusal":
            log.warning("Model refused: %s", getattr(response, "stop_details", None))
            return LLMResult(text="", model=response.model, usage=usage, refused=True)
        if response.stop_reason == "max_tokens":
            text += "…"
        return LLMResult(text=text, model=response.model, usage=usage)


class OpenAICompatibleLLM:
    """OpenAI, OpenRouter, DeepSeek, Ollama, vLLM, YandexGPT/GigaChat via compatible gateways."""

    def __init__(self, settings: Settings, *, api_key: str | None = None, base_url: str | None = None) -> None:
        import openai

        self._openai = openai
        self.client = openai.AsyncOpenAI(
            api_key=api_key or settings.openai_api_key or "not-needed",
            base_url=base_url or settings.openai_base_url,
            timeout=settings.llm_timeout,
            max_retries=2,
        )
        self.model = settings.llm_model
        self.fallback_model = settings.llm_fallback_model
        self.max_tokens = settings.llm_max_tokens

    async def generate(self, system: str, messages: list[Message]) -> LLMResult:
        openai = self._openai
        chat = [{"role": "system", "content": system}, *messages]
        try:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model, max_tokens=self.max_tokens, messages=chat
                )
            except (openai.RateLimitError, openai.InternalServerError, openai.NotFoundError) as exc:
                # Free tiers are often overloaded (503) or out of quota (429): try the backup model.
                if not self.fallback_model:
                    raise
                log.warning("%s failed (%s), falling back to %s", self.model, exc.status_code, self.fallback_model)
                response = await self.client.chat.completions.create(
                    model=self.fallback_model, max_tokens=self.max_tokens, messages=chat
                )
        except openai.AuthenticationError as exc:
            raise LLMError("Неверный ключ OPENAI_API_KEY.") from exc
        except openai.RateLimitError as exc:
            raise LLMError("Превышен лимит запросов к LLM, попробуйте через минуту.") from exc
        except openai.APIStatusError as exc:
            log.error("OpenAI-compatible API error %s: %s", exc.status_code, exc)
            raise LLMError("Сервис LLM вернул ошибку.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Нет соединения с LLM API.") from exc
        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        text = _THINK.sub("", choice.message.content or "").strip()
        return LLMResult(text=text, model=response.model, usage=usage)


def create_llm(settings: Settings) -> LLM | None:
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    if settings.llm_provider == "gemini":
        return OpenAICompatibleLLM(settings, api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(settings)
    return None  # retrieval-only mode: the bot answers with the most relevant excerpts
