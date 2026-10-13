"""Test doubles for the pipeline: a scripted LLM and a cheap deterministic embedder."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

import numpy as np

from talktable.llm import LLMError, Message

Handler = Callable[[list[Message], bool], str]


class FakeLLM:
    """Answers each call with `handler(messages, json_mode)` and records every call.

    `step(messages)` tells which pipeline step is calling, from its system prompt.
    """

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, list[Message]]] = []

    @staticmethod
    def step(messages: list[Message]) -> str:
        system = messages[0]["content"]
        if system.startswith("You route questions"):
            return "router"
        if system.startswith("You explain"):
            return "explainer"
        if system.startswith("You rewrite"):
            return "condenser"
        return "generator"

    def complete(self, messages: list[Message], *, json_mode: bool = False, max_tokens: int = 512) -> str:
        self.calls.append((self.step(messages), [dict(m) for m in messages]))
        return self.handler(messages, json_mode)

    def count(self, step: str) -> int:
        return sum(1 for name, _ in self.calls if name == step)


def scripted(router: str | None = None, generator: list[str] | None = None,
             explainer: str | Callable[[list[Message]], str] | None = None,
             condenser: str | None = None) -> FakeLLM:
    """A FakeLLM that replies per step. `generator` replies are used in order (the last
    one repeats). The default explainer quotes the first result row, so it is grounded.
    `condenser` is the standalone question a follow-up is rewritten to."""
    generator_replies = list(generator or [])

    def handler(messages: list[Message], json_mode: bool) -> str:
        step = FakeLLM.step(messages)
        if step == "router":
            if router is None:
                raise LLMError("no router reply scripted")
            return router
        if step == "condenser":
            if condenser is None:
                raise LLMError("no condenser reply scripted")
            return json.dumps({"question": condenser})
        if step == "generator":
            if not generator_replies:
                raise LLMError("no generator reply scripted")
            return generator_replies.pop(0) if len(generator_replies) > 1 else generator_replies[0]
        if callable(explainer):
            return explainer(messages)
        if explainer is not None:
            return explainer
        return quote_first_row(messages)

    return FakeLLM(handler)


def quote_first_row(messages: list[Message]) -> str:
    """An explainer reply that restates the first row of the results table it was given."""
    table = messages[1]["content"].split(":\n", 1)[1].splitlines()
    header, first = table[0].split(" | "), table[1].split(" | ")
    return "The top result is " + ", ".join(f"{h} {v}" for h, v in zip(header, first)) + "."


def sql_reply(sql: str) -> str:
    return json.dumps({"sql": sql})


class HashEmbedder:
    """Bag-of-words hashing vectors: deterministic, instant, and good enough to rank
    texts that share words. Used where the real model's quality doesn't matter."""

    dim = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)
