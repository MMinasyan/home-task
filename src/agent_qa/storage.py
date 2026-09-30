"""Durable session and turn storage on one single-worker SQLite connection."""

import asyncio
import json
import secrets
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NamedTuple, cast

Status = Literal["success", "failure", "interrupted"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    current_turn_id TEXT,
    usage_total TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn_id TEXT,
    request_id TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_session ON entries(session_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS user_request
    ON entries(session_id, request_id) WHERE kind = 'user';
"""


class Snapshot(NamedTuple):
    """One committed session revision: its running pointer and ordered entries."""

    current_turn_id: str | None
    entries: tuple[tuple[str, str], ...]


def _encode(payload) -> str:
    return json.dumps(payload, allow_nan=False)


class Storage:
    """Durable sessions and append-only entries behind one SQLite connection.

    The connection is created, used, and closed exclusively on the single
    worker thread of an owned ``ThreadPoolExecutor(max_workers=1)``, keeping
    SQLite's normal thread affinity. Async methods submit one complete
    synchronous operation through ``run_in_executor`` and await its result;
    the executor's FIFO queue also drains recovery before close.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._connection: sqlite3.Connection | None = None

    @classmethod
    async def open(cls, path: Path) -> "Storage":
        """Construct the worker, initialize its connection and schema, and return it.

        Running turns are not settled here; startup calls ``interrupt_all``
        explicitly before serving. Initialization failure, including a
        cancelled await, closes the storage through the ordinary FIFO close
        after the initialization work and re-raises the original outcome.
        """
        storage = cls()
        try:
            await asyncio.get_running_loop().run_in_executor(
                storage._executor, storage._initialize, path
            )
        except BaseException:
            await storage.close()
            raise
        return storage

    async def create_session(self) -> str:
        """Insert one session with a random ID and return it only after commit."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._create_session)

    async def snapshot(self, session_id: str) -> Snapshot | None:
        """Return one committed session revision, or None when it does not exist."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._snapshot, session_id)

    async def admit(
        self, session_id: str, request_id: str, text: str
    ) -> tuple[str, bool] | Literal["unknown", "busy"]:
        """Admit one user turn and return ``(turn_id, fresh)``.

        An existing user row for the request ID wins over every other state
        and returns ``(turn_id, False)`` without writing. A set running
        pointer on a new request returns ``"busy"``; an unknown session
        returns ``"unknown"``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._admit, session_id, request_id, text
        )

    async def settle(
        self, session_id: str, turn_id: str, status: Status, assistant: dict | None
    ) -> bool:
        """Apply one guarded terminal transition.

        Returns False, writing nothing, unless the session's running pointer
        still names the supplied turn ID. On a match, appends the assistant
        payload when supplied, a signal for non-success, exactly one
        settlement, merges the assistant usage into the cumulative total, and
        clears the pointer last.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._settle, session_id, turn_id, status, assistant
        )

    async def interrupt_all(self) -> None:
        """Settle every set pointer as interrupted with no assistant payload.

        Used before serving and at the shutdown tail; idempotent, no model call.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._interrupt_all)

    async def close(self) -> None:
        """Close the connection when one exists, then join the executor.

        Callers must stop submitting work before closing.
        """
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._close)
        finally:
            await asyncio.to_thread(self._executor.shutdown, wait=True)

    def _initialize(self, path):
        connection = sqlite3.connect(path, isolation_level=None)
        self._connection = connection
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(SCHEMA)

    def _close(self):
        if self._connection is not None:
            connection = cast(sqlite3.Connection, self._connection)
            connection.close()
            self._connection = None

    def _transaction(self, work):
        connection = cast(sqlite3.Connection, self._connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            result = work(connection)
            connection.execute("COMMIT")
            return result
        except BaseException:
            # SQLite may already have rolled the transaction back itself, as
            # on disk-full; a failed COMMIT can also leave it open. Roll back
            # only when the transaction is still active.
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _create_session(self):
        def work(connection):
            session_id = secrets.token_urlsafe(24)
            connection.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
            return session_id

        return self._transaction(work)

    def _snapshot(self, session_id):
        def work(connection):
            row = connection.execute(
                "SELECT current_turn_id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            entries = tuple(
                connection.execute(
                    "SELECT kind, payload FROM entries WHERE session_id = ? ORDER BY id",
                    (session_id,),
                )
            )
            return Snapshot(row[0], entries)

        return self._transaction(work)

    def _admit(self, session_id, request_id, text):
        def work(connection):
            session = connection.execute(
                "SELECT current_turn_id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session is None:
                return "unknown"
            existing = connection.execute(
                "SELECT turn_id FROM entries"
                " WHERE session_id = ? AND request_id = ? AND kind = 'user'",
                (session_id, request_id),
            ).fetchone()
            if existing is not None:
                return (existing[0], False)
            if session[0] is not None:
                return "busy"
            turn_id = secrets.token_urlsafe(24)
            connection.execute(
                "INSERT INTO entries (session_id, turn_id, request_id, kind, payload)"
                " VALUES (?, ?, ?, 'user', ?)",
                (session_id, turn_id, request_id, _encode({"text": text})),
            )
            connection.execute(
                "UPDATE sessions SET current_turn_id = ? WHERE id = ?",
                (turn_id, session_id),
            )
            return (turn_id, True)

        return self._transaction(work)

    def _settle(self, session_id, turn_id, status, assistant):
        def work(connection):
            return self._guarded_settle(connection, session_id, turn_id, status, assistant)

        return self._transaction(work)

    def _guarded_settle(self, connection, session_id, turn_id, status, assistant):
        """Apply one settlement inside the caller's transaction; False on guard miss."""
        session = connection.execute(
            "SELECT current_turn_id, usage_total FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None or session[0] != turn_id:
            return False
        if assistant is not None:
            connection.execute(
                "INSERT INTO entries (session_id, turn_id, kind, payload)"
                " VALUES (?, ?, 'assistant', ?)",
                (session_id, turn_id, _encode(assistant)),
            )
            usage = assistant.get("usage")
            if usage is not None:
                total = json.loads(session[1])
                for key, value in usage.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    total[key] = total.get(key, 0) + value
                connection.execute(
                    "UPDATE sessions SET usage_total = ? WHERE id = ?",
                    (_encode(total), session_id),
                )
        if status != "success":
            connection.execute(
                "INSERT INTO entries (session_id, turn_id, kind, payload)"
                " VALUES (?, ?, 'signal', ?)",
                (session_id, turn_id, _encode({"status": status})),
            )
        connection.execute(
            "INSERT INTO entries (session_id, turn_id, kind, payload)"
            " VALUES (?, ?, 'turn_settlement', ?)",
            (session_id, turn_id, _encode({"status": status})),
        )
        connection.execute(
            "UPDATE sessions SET current_turn_id = NULL WHERE id = ?", (session_id,)
        )
        return True

    def _interrupt_all(self):
        def work(connection):
            pointers = connection.execute(
                "SELECT id, current_turn_id FROM sessions WHERE current_turn_id IS NOT NULL"
            ).fetchall()
            for session_id, turn_id in pointers:
                self._guarded_settle(connection, session_id, turn_id, "interrupted", None)

        self._transaction(work)
