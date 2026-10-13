"""`ask()`: question in, plain-English answer + query + rows out.

    1. router        -> which database
    2. schema_index  -> the relevant tables/collections
    3. generator     -> SQL or a Mongo pipeline
    4. guardrails    -> validated inside adapter.run()
    5. executor      -> run it, self-correct on errors (max 2 retries)
    6. explainer     -> a 1-3 sentence answer grounded in the rows

`TalkTable` is built once at start-up: it connects to whatever databases are reachable,
loads the embedding model, and indexes the schemas. Databases that are down are
reported as offline and simply not used.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence

from talktable.adapters import ALL_DATABASES, DatabaseAdapter, DatabaseUnavailableError, QueryError, create_adapters
from talktable.cache import AnswerCache, rows_fingerprint
from talktable.config import DESCRIPTIONS_PATH, Settings, get_settings
from talktable.conversation import Condenser, Turn, looks_like_follow_up
from talktable.descriptions import Descriptions, describe_tables, load_descriptions
from talktable.executor import execute_with_retries
from talktable.explainer import Explainer
from talktable.generator import Generator, QueryContext, query_to_text
from talktable.guardrails import GuardrailError
from talktable.llm import ChatModel, LLMClient, LLMError
from talktable.models import AskResult, DatabaseStatus, TableInfo
from talktable.router import RouteDecision, Router, canonical_name, is_small_talk
from talktable.schema_index import RETRIEVE_K, Embedder, SchemaIndex, SentenceEmbedder
from talktable.vector_store import create_vector_store

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
_URL_SETTING = {"postgres": "POSTGRES_URL", "mongo": "MONGO_URL", "sqlite": "SQLITE_PATH"}

ProgressFn = Callable[[str], None]  # receives short step messages like "Writing the query"

_FROM_SETTINGS = object()  # TalkTable(cache=...) default: build the cache from the settings


def cache_version(settings: Settings) -> str:
    """Cached queries are only valid for the same schema descriptions and models."""
    import hashlib

    digest = hashlib.sha256(DESCRIPTIONS_PATH.read_bytes())
    for part in (settings.llm_provider, settings.llm_model, settings.llm_query_model or "", settings.embedding_model):
        digest.update(b"\0" + part.encode())
    return digest.hexdigest()[:16]


class TalkTable:
    """The TalkTable engine: adapters, schema index, and the LLM-backed pipeline steps."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        adapters: dict[str, DatabaseAdapter] | None = None,
        llm: ChatModel | None = None,
        query_llm: ChatModel | None = None,
        embedder: Embedder | None = None,
        descriptions: Descriptions | None = None,
        cache: AnswerCache | None | object = _FROM_SETTINGS,
    ) -> None:
        """`llm` routes and explains; `query_llm` (default: `llm`) writes and corrects
        queries. Both are built from the settings when not given, and so is the answer
        `cache` (pass None for none)."""
        self.settings = settings or get_settings()
        self.descriptions = descriptions or load_descriptions()
        self.adapters = adapters if adapters is not None else create_adapters(self.settings)
        self.index = SchemaIndex(embedder or SentenceEmbedder(self.settings.embedding_model))
        self._llm_error: str | None = None
        if llm is None:
            try:
                llm = LLMClient(self.settings)
                if query_llm is None and self.settings.llm_query_model:
                    query_llm = LLMClient(self.settings, model=self.settings.llm_query_model)
            except LLMError as exc:  # e.g. missing API key: report it on every question
                self._llm_error = str(exc)
        self._llm = llm
        if llm is not None:
            self.router = Router(llm, self.index, self.descriptions, min_relevance=self.settings.min_relevance)
            self.generator = Generator(query_llm or llm)
            self.explainer = Explainer(llm)
            self.condenser = Condenser(llm)
        if cache is _FROM_SETTINGS:
            cache = (AnswerCache(self.settings.cache_file, self.index.embed, version=cache_version(self.settings),
                                 similarity=self.settings.cache_similarity, ttl_hours=self.settings.cache_ttl_hours)
                     if self.settings.cache_enabled else None)
        self.cache: AnswerCache | None = cache  # type: ignore[assignment]
        self._status: dict[str, DatabaseStatus] = {}
        self.refresh()

    # --- databases ------------------------------------------------------------

    def refresh(self) -> list[DatabaseStatus]:
        """(Re)connect to every database, re-read schemas, and rebuild the index."""
        schemas: dict[str, list[TableInfo]] = {}
        for name in ALL_DATABASES:
            description = self.descriptions.databases.get(name)
            info = {
                "name": name,
                "title": description.title if description else name,
                "kind": description.kind if description else ("mongo" if name == "mongo" else "sql"),
                "description": " ".join(description.summary.split()) if description else "",
            }
            adapter = self.adapters.get(name)
            if adapter is None:
                self.index.remove_database(name)
                self._status[name] = DatabaseStatus(**info, online=False,
                                                    detail=f"not configured ({_URL_SETTING[name]} is not set)")
                continue
            try:
                if not adapter.ping():
                    raise QueryError("not reachable")
                schemas[name] = describe_tables(adapter.get_schema(), description)
                self._status[name] = DatabaseStatus(**info, online=True, tables=[t.name for t in schemas[name]])
            except QueryError as exc:
                self.index.remove_database(name)
                self._status[name] = DatabaseStatus(**info, online=False, detail=str(exc))
        self._choose_vector_store(sum(len(tables) for tables in schemas.values()))
        for name, tables in schemas.items():
            self.index.add_database(name, tables, self.descriptions.databases.get(name))
        logger.info("databases: %s", ", ".join(f"{s.name}={'online' if s.online else 'offline'}"
                                               for s in self._status.values()))
        return self.status()

    def _choose_vector_store(self, table_count: int) -> None:
        """NumPy for small schemas, Qdrant for large ones (or as configured)."""
        backend = self.settings.vector_backend(table_count)
        if backend == self.index.store.backend:
            return
        s = self.settings
        self.index.use_store(create_vector_store(
            backend, collection=s.qdrant_collection, url=s.qdrant_url, path=s.qdrant_dir,
            api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None))

    def status(self) -> list[DatabaseStatus]:
        return [self._status[name] for name in ALL_DATABASES if name in self._status]

    @property
    def online(self) -> list[str]:
        return [s.name for s in self.status() if s.online]

    def resolve_database(self, name: str) -> str:
        """Our name for an online database ("PostgreSQL" -> "postgres").

        Raises:
            ValueError: the database is unknown or offline (the message is user-safe).
        """
        canonical = canonical_name(name)
        status = self._status.get(canonical)
        if status is None:
            raise ValueError(f"Unknown database '{name}'. Choose one of: {', '.join(ALL_DATABASES)}.")
        if not status.online:
            online = ", ".join(self.online) or "none"
            raise ValueError(f"The {canonical} database is not available: {status.detail}. Online: {online}.")
        return canonical

    def schema(self, database: str) -> list[TableInfo]:
        """The tables (or collections) of an online database, with their descriptions.

        Raises:
            ValueError: the database is unknown or offline.
        """
        return self.index.tables(self.resolve_database(database))

    # --- the pipeline -----------------------------------------------------------

    def ask(self, question: str, *, database: str | None = None, on_step: ProgressFn | None = None,
            history: Sequence[Turn] = (), use_cache: bool = True) -> AskResult:
        """Answer `question`. Never raises for user-facing problems: they come back as
        an AskResult with `error` set and a friendly `answer`.

        `database` skips routing. `on_step` is called with a short message as each
        step starts, for progress displays. `history` holds earlier turns of a chat, so
        follow-ups like "and in 2024?" can be answered. `use_cache=False` ignores the
        answer cache (the fresh answer still replaces the cached one).
        """
        started = time.perf_counter()
        notify = on_step or (lambda _message: None)
        question = original = " ".join((question or "").split())
        logger.info("question: %s", question)

        def failure(message: str, *, databases: list[str] | None = None, queries: list[str] | None = None,
                    retries: int = 0) -> AskResult:
            logger.warning("no answer: %s", message)
            return AskResult(question=original, databases_used=databases or [], queries=queries or [],
                             rows=[], answer=message, retries=retries, error=message,
                             interpreted_as=question if question != original else None,
                             latency_ms=round((time.perf_counter() - started) * 1000))

        if not question:
            return failure("Please type a question.")
        if len(question) > MAX_QUESTION_CHARS:
            return failure(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
        if self._llm is None:
            return failure(f"The language model isn't configured: {self._llm_error}")
        online = self.online
        if not online:
            return failure("No databases are online right now.")

        if database is not None:
            try:
                database = self.resolve_database(database)
            except ValueError as exc:
                return failure(str(exc))
        scope = database or "auto"  # cache entries are separate for forced databases

        # 0. A follow-up in a chat becomes a standalone question
        if history and not is_small_talk(question) and looks_like_follow_up(question):
            notify("Reading the conversation")
            question = self.condenser.standalone(question, history)
            if question != original:
                logger.info("follow-up rewritten as: %s", question)

        # 0b. Asked before? Re-run the cached query instead of writing a new one
        if self.cache is not None and use_cache and not is_small_talk(question):
            cached = self._answer_from_cache(question, original, scope, online, notify, started)
            if cached is not None:
                return cached

        # 1. Route
        if database is not None:
            route = RouteDecision(databases=[database], reason="chosen by the caller", method="forced")
        else:
            if len(online) > 1:
                notify("Choosing a database")
            route = self.router.route(question, online)
        logger.info("route: %s (%s: %s)", route.databases, route.method, route.reason)
        if not route.databases:  # a greeting or an unrelated question: say what we can do
            return AskResult(question=original, databases_used=[], queries=[], rows=[],
                             answer=self.about_message(original, off_topic=route.method == "off_topic"), retries=0,
                             latency_ms=round((time.perf_counter() - started) * 1000))
        db = route.databases[0]  # cross-database questions come in Phase 6

        # 2. Retrieve the relevant schema
        kind = self.adapters[db].kind
        selection = self.index.retrieve(question, db, k=RETRIEVE_K, kind=kind,
                                        token_budget=self.settings.schema_token_budget)
        logger.info("schema: %s", ", ".join(f"{name} ({role})" for name, role in selection.roles.items()))
        description = self.descriptions.databases.get(db)
        context = QueryContext(
            database=db,
            kind=kind,
            title=description.title if description else db,
            tables=selection.tables,
            few_shot=selection.examples,
            glossary=self.descriptions.glossary_for(question, db),
            joins=selection.joins,
            compact=selection.compact,
        )
        if context.glossary:
            logger.info("glossary: %s", ", ".join(entry.term for entry in context.glossary))

        # 3. Generate
        notify(f"Writing the query ({context.title}, {db})")
        try:
            query = self.generator.generate(question, context)
        except LLMError as exc:
            return failure(str(exc), databases=[db])
        logger.info("query: %s", query_to_text(query))

        # 4 + 5. Guardrails, execute, self-correct
        notify("Running the query")
        outcome = execute_with_retries(self.adapters[db], self.generator, question, context, query, on_step=notify)
        queries = [a.query for a in outcome.attempts]
        if outcome.result is None:
            return failure(f"I couldn't get a working query for that question. Last error: {outcome.error}",
                           databases=[db], queries=queries[-1:], retries=outcome.retries)
        logger.info("rows: %d (retries: %d)", outcome.result.row_count, outcome.retries)

        # 6. Explain
        if outcome.result.rows:
            notify("Explaining the results")
        answer = self.explainer.explain(question, outcome.result)
        latency_ms = round((time.perf_counter() - started) * 1000)
        logger.info("answer (%d ms): %s", latency_ms, answer)
        if self.cache is not None:
            self.cache.store(question, database=db, query=outcome.result.query, answer=answer, scope=scope,
                             rows_hash=rows_fingerprint(outcome.result.columns, outcome.result.rows))
        return AskResult(
            question=original,
            interpreted_as=question if question != original else None,
            databases_used=[db],
            queries=[outcome.result.query],
            rows=outcome.result.rows,
            columns=outcome.result.columns,
            answer=answer,
            retries=outcome.retries,
            latency_ms=latency_ms,
        )

    def _answer_from_cache(self, question: str, original: str, scope: str, online: list[str],
                           notify: ProgressFn, started: float) -> AskResult | None:
        """The answer from a cached query, re-run on the current data; None on a miss."""
        assert self.cache is not None
        hit = self.cache.lookup(question, scope)
        if hit is None or hit.database not in online:
            return None
        logger.info("cache: %s question (%s); re-running its query", hit.match, hit.question)
        notify("Asked before: re-running the saved query")
        try:
            result = self.adapters[hit.database].run(hit.query)  # through the guardrails again
        except DatabaseUnavailableError as exc:
            logger.warning("cache: %s is unavailable (%s)", hit.database, exc)
            return None
        except (GuardrailError, QueryError) as exc:  # e.g. the schema changed: write a new query
            logger.warning("cache: the saved query no longer works (%s); answering afresh", exc)
            self.cache.forget(hit.question, scope)
            return None
        rows_hash = rows_fingerprint(result.columns, result.rows)
        if rows_hash == hit.rows_hash:
            answer = hit.answer
        else:  # the data changed since: explain the new rows
            notify("Explaining the results")
            answer = self.explainer.explain(question, result)
            self.cache.store(question, database=hit.database, query=result.query, rows_hash=rows_hash,
                             answer=answer, scope=scope)
        latency_ms = round((time.perf_counter() - started) * 1000)
        logger.info("answer from cache (%d ms): %s", latency_ms, answer)
        return AskResult(question=original, interpreted_as=question if question != original else None,
                         databases_used=[hit.database], queries=[result.query], rows=result.rows,
                         columns=result.columns, answer=answer, retries=0, latency_ms=latency_ms, cached=True)

    def about_message(self, message: str = "", *, off_topic: bool = False) -> str:
        """What TalkTable can answer, for greetings and questions about something else."""
        first_word = message.strip().lower().split(" ")[0].strip("!.,?") if message.strip() else ""
        if off_topic:
            intro = "That doesn't look like a question about the bookstore's data, which is all I can answer."
        elif first_word in {"thanks", "thank", "thx", "ty", "cheers"}:
            intro = "You're welcome! Ask me anything else about the bookstore's data."
        elif first_word in {"bye", "goodbye", "see"}:
            intro = "Goodbye! Come back any time with a question about the bookstore's data."
        else:
            intro = "Hi! I'm TalkTable. I answer questions about an online bookstore's data."
        lines = [intro, "", "I can look things up in:"]
        example = None
        for status in self.status():
            if not status.online:
                continue
            description = self.descriptions.databases.get(status.name)
            lines.append(f"- **{status.title}**: {', '.join(status.tables)}")
            if example is None and description and description.examples:
                example = description.examples[0]
        if example:
            lines += ["", f'Try, for example: "{example}"']
        return "\n".join(lines)

    def close(self) -> None:
        for adapter in self.adapters.values():
            adapter.close()
        self.index.store.close()


_engine: TalkTable | None = None
_engine_lock = threading.Lock()


def get_talktable() -> TalkTable:
    """The shared engine, built on first use. Safe to call from several threads at once
    (the MCP server and the Gradio app both call it from worker threads)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = TalkTable()
        return _engine


def ask(question: str, database: str | None = None) -> AskResult:
    """Answer a natural-language question using the shared engine."""
    return get_talktable().ask(question, database=database)
