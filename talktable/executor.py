"""Executor: run a generated query, and self-correct when it fails.

If the guardrails reject the query or the database returns an error, the error and the
failed query go back to the generator for a corrected version, at most MAX_RETRIES
times. Every retry is a brand-new query, so it goes through the guardrails again
(`adapter.run()` always validates).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import BaseModel

from talktable.adapters import DatabaseAdapter, DatabaseUnavailableError, QueryError
from talktable.generator import GeneratedQuery, Generator, QueryContext, query_to_text
from talktable.guardrails import GuardrailError
from talktable.llm import LLMError
from talktable.models import QueryResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class Attempt(BaseModel):
    query: str  # as generated (or, if it ran, as executed after the guardrails)
    error: str | None = None


class ExecutionOutcome(BaseModel):
    result: QueryResult | None
    attempts: list[Attempt]
    error: str | None = None  # set when no attempt succeeded

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)


def execute_with_retries(
    adapter: DatabaseAdapter,
    generator: Generator,
    question: str,
    context: QueryContext,
    query: GeneratedQuery,
    *,
    max_retries: int = MAX_RETRIES,
    on_step: Callable[[str], None] | None = None,
) -> ExecutionOutcome:
    """Run `query`; on failure ask the generator for a fix, up to `max_retries` times.
    `on_step` is told about each retry (for progress displays)."""
    attempts: list[Attempt] = []
    while True:
        try:
            result = adapter.run(query)
        except DatabaseUnavailableError as exc:  # a new query won't help
            attempts.append(Attempt(query=query_to_text(query), error=str(exc)))
            return ExecutionOutcome(result=None, attempts=attempts, error=str(exc))
        except (GuardrailError, QueryError) as exc:
            kind = "rejected by guardrails" if isinstance(exc, GuardrailError) else "failed"
            logger.info("%s: attempt %d %s: %s", adapter.name, len(attempts) + 1, kind, exc)
            attempts.append(Attempt(query=query_to_text(query), error=str(exc)))
            if len(attempts) > max_retries:
                return ExecutionOutcome(result=None, attempts=attempts, error=str(exc))
            if on_step is not None:
                on_step(f"The query {kind}; fixing it (retry {len(attempts)} of {max_retries})")
            try:
                query = generator.correct(question, context, query, str(exc))
            except LLMError as llm_exc:
                return ExecutionOutcome(result=None, attempts=attempts, error=str(llm_exc))
            logger.info("%s: retry %d with: %s", adapter.name, len(attempts), query_to_text(query))
            continue
        attempts.append(Attempt(query=result.query))
        return ExecutionOutcome(result=result, attempts=attempts)
