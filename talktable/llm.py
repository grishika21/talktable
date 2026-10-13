"""Provider-agnostic LLM client.

Ollama (local), the Hugging Face router, and Groq all expose OpenAI-compatible chat
APIs, so one client covers all three. Only the base URL, API key, and a couple of
request options differ. Pick the provider with LLM_PROVIDER and the model with LLM_MODEL
(plus, optionally, LLM_QUERY_MODEL for the query-writing step).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError

from talktable.config import Settings

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

HF_BASE_URL = "https://router.huggingface.co/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class LLMError(Exception):
    """The LLM could not be reached or gave an unusable answer. The message is user-safe."""


class ChatModel(Protocol):
    """What the pipeline needs from an LLM. Tests substitute a scripted fake."""

    def complete(self, messages: list[Message], *, json_mode: bool = False, max_tokens: int = 512) -> str: ...


class LLMClient:
    """Chat completions against Ollama, Hugging Face, or Groq."""

    def __init__(self, settings: Settings, *, model: str | None = None) -> None:
        self.provider = settings.llm_provider
        self.model = model or settings.llm_model
        if self.provider == "ollama":
            base_url, api_key = f"{settings.ollama_base_url.rstrip('/')}/v1", "ollama"  # key is ignored
        elif self.provider == "hf":
            if settings.hf_token is None:
                raise LLMError("LLM_PROVIDER=hf needs HF_TOKEN to be set.")
            base_url, api_key = HF_BASE_URL, settings.hf_token.get_secret_value()
        else:
            if settings.groq_api_key is None:
                raise LLMError("LLM_PROVIDER=groq needs GROQ_API_KEY to be set.")
            base_url, api_key = GROQ_BASE_URL, settings.groq_api_key.get_secret_value()
        self._client = OpenAI(base_url=base_url, api_key=api_key,
                              timeout=settings.llm_timeout_seconds, max_retries=1)
        # Hosted providers don't all support JSON mode for every model; we remember a
        # rejection and fall back to plain text (we parse the JSON ourselves anyway).
        self._json_mode_supported = True

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def complete(self, messages: list[Message], *, json_mode: bool = False, max_tokens: int = 512) -> str:
        """One chat completion at temperature 0. Returns the reply text (reasoning removed)."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if self.provider == "ollama":
            # Thinking models (e.g. qwen3) otherwise spend hundreds of tokens reasoning
            # before answering, which is very slow on a laptop. Other models ignore it.
            request["extra_body"] = {"reasoning_effort": "none"}
        if json_mode and self._json_mode_supported:
            request["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except APIStatusError as exc:
            if "response_format" in request and exc.status_code in (400, 422):
                logger.info("%s rejected JSON mode; retrying without it", self.label)
                self._json_mode_supported = False
                return self.complete(messages, json_mode=json_mode, max_tokens=max_tokens)
            raise LLMError(self._status_message(exc)) from None
        except APITimeoutError:
            raise LLMError("The language model took too long to answer.") from None
        except APIConnectionError:
            hint = " Is Ollama running (ollama serve)?" if self.provider == "ollama" else ""
            raise LLMError(f"Could not reach the language model ({self.provider}).{hint}") from None
        except OpenAIError as exc:
            raise LLMError(f"The language model request failed: {type(exc).__name__}") from None

        text = (response.choices[0].message.content or "") if response.choices else ""
        usage = response.usage
        logger.info("llm %s: %.1fs, %s prompt + %s completion tokens", self.label,
                    time.perf_counter() - started,
                    usage.prompt_tokens if usage else "?", usage.completion_tokens if usage else "?")
        text = _THINK_BLOCK.sub("", text).strip()
        if not text:
            raise LLMError("The language model returned an empty answer.")
        return text

    def _status_message(self, exc: APIStatusError) -> str:
        if exc.status_code in (401, 403):
            return f"The {self.provider} API rejected the credentials (check the API token)."
        if exc.status_code == 404:
            return f"Model '{self.model}' was not found on {self.provider}."
        if exc.status_code == 429:
            return f"The {self.provider} API rate limit was reached. Try again in a minute."
        return f"The language model API returned an error (HTTP {exc.status_code})."


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the JSON object in an LLM reply, tolerating code fences and chatter around it.

    Raises:
        LLMError: no JSON object could be found.
    """
    cleaned = _CODE_FENCE.sub("", _THINK_BLOCK.sub("", text).strip()).strip()
    try:
        value = json.loads(cleaned, strict=False)  # strict=False: allow raw newlines in strings
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("The language model did not return JSON.") from None
        try:
            value = json.loads(cleaned[start:end + 1], strict=False)
        except json.JSONDecodeError as exc:
            raise LLMError(f"The language model returned invalid JSON ({exc.msg}).") from None
    if not isinstance(value, dict):
        raise LLMError("The language model returned JSON, but not an object.")
    return value
