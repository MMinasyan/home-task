"""Durable storage: atomic admission, guarded settlement, and worker lifetime."""

import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import agent_qa.storage
from agent_qa.storage import Storage


def dump(path):
    """Complete durable rows for exact before/after comparison."""
    connection = sqlite3.connect(path)
    try:
        sessions = connection.execute(
            "SELECT id, current_turn_id, usage_total FROM sessions ORDER BY rowid"
        ).fetchall()
        entries = connection.execute(
            "SELECT session_id, turn_id, request_id, kind, payload FROM entries ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    return sessions, entries


def kinds(snapshot):
    """The entry kinds of one snapshot in entry-ID order."""
    return [kind for kind, _ in snapshot.entries]


RECORDED = {}


class RecordedConnection(sqlite3.Connection):
    """A real connection that records the thread closing it."""

    def close(self):
        RECORDED["closed_by"] = threading.get_ident()
        super().close()


class BrokenSchemaConnection(RecordedConnection):
    """A real connection whose schema script fails."""

    def executescript(self, script):
        raise sqlite3.OperationalError("injected schema failure")


class CommitFailureConnection(RecordedConnection):
    """A real connection whose COMMIT statement fails once, as on disk-full."""

    armed = False

    def execute(self, sql, parameters=()):
        if sql == "COMMIT" and type(self).armed:
            type(self).armed = False
            raise sqlite3.OperationalError("database or disk is full")
        return super().execute(sql, parameters)


class RecordedExecutor(ThreadPoolExecutor):
    """The production executor recording the thread that joins it."""

    def shutdown(self, wait=True, *, cancel_futures=False):
        RECORDED["shutdown_by"] = threading.get_ident()
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


def patch_recording(monkeypatch, factory=RecordedConnection):
    """Route storage through the recording executor and connection types."""
    RECORDED.clear()
    monkeypatch.setattr(agent_qa.storage, "ThreadPoolExecutor", RecordedExecutor)
    real_connect = sqlite3.connect

    def spy(path, *args, **kwargs):
        RECORDED["opened_by"] = threading.get_ident()
        return real_connect(path, *args, factory=factory, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)


def capture_initialization(monkeypatch, hook):
    """Wrap Storage._initialize so ``hook`` runs before its real work."""
    original = Storage._initialize

    def wrapper(self, path):
        hook(self)
        return original(self, path)

    monkeypatch.setattr(Storage, "_initialize", wrapper)


def test_open_creates_schema_and_snapshots(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            assert await storage.snapshot("missing") is None
            session_id = await storage.create_session()
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert snapshot.entries == ()
        finally:
            await storage.close()
        sessions, entries = dump(path)
        assert len(sessions) == 1 and entries == []
        assert sessions[0][2] == "{}"

    asyncio.run(run())


def test_admission_writes_user_entry_and_pointer(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            turn_id, fresh = await storage.admit(session_id, "request-1", "What?")
            assert fresh
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id == turn_id
            assert kinds(snapshot) == ["user"]
            assert json.loads(snapshot.entries[0][1]) == {"text": "What?"}
        finally:
            await storage.close()
        _, entries = dump(path)
        assert entries[0][2] == "request-1"

    asyncio.run(run())


def test_unknown_session_is_uniformly_rejected(tmp_path):
    async def run():
        storage = await Storage.open(tmp_path / "db.sqlite3")
        try:
            assert await storage.admit("missing", "r", "t") == "unknown"
            assert await storage.snapshot("missing") is None
            assert await storage.settle("missing", "t", "success", None) is False
        finally:
            await storage.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", ["success", "failure", "interrupted"])
def test_duplicate_beats_busy_and_survives_terminal_states(tmp_path, status):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            turn_id, _ = await storage.admit(session_id, "r1", "first text")
            # While running, and with replacement text, the first writer wins.
            assert await storage.admit(session_id, "r1", "replacement") == (turn_id, False)
            assert await storage.admit(session_id, "r2", "second") == "busy"
            assert await storage.settle(session_id, turn_id, status, None)
            # After the terminal state the duplicate still resolves without writing.
            before = dump(path)
            assert await storage.admit(session_id, "r1", "again") == (turn_id, False)
            assert dump(path) == before
            turn_id_2, fresh = await storage.admit(session_id, "r2", "second")
            assert fresh and turn_id_2 != turn_id
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert json.loads(snapshot.entries[0][1]) == {"text": "first text"}
        finally:
            await storage.close()

    asyncio.run(run())


def test_concurrent_same_request_id_admits_once(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            results = await asyncio.gather(
                storage.admit(session_id, "r", "one"),
                storage.admit(session_id, "r", "two"),
            )
            admitted = [result for result in results if isinstance(result, tuple)]
            assert len({result[0] for result in admitted}) == 1
            assert [result[1] for result in admitted].count(True) == 1
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id == admitted[0][0]
            assert kinds(snapshot) == ["user"]
            assert json.loads(snapshot.entries[0][1])["text"] in ("one", "two")
        finally:
            await storage.close()

    asyncio.run(run())


def test_concurrent_different_request_ids_single_flight(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            results = await asyncio.gather(
                storage.admit(session_id, "a", "one"),
                storage.admit(session_id, "b", "two"),
            )
            admitted = [result for result in results if isinstance(result, tuple)]
            busy = [result for result in results if result == "busy"]
            assert len(admitted) == 1 and len(busy) == 1
            assert admitted[0][1] is True
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id == admitted[0][0]
            assert kinds(snapshot) == ["user"]
        finally:
            await storage.close()

    asyncio.run(run())


def test_settlement_guard_rejects_mismatches_without_writes(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            turn_id, _ = await storage.admit(session_id, "r1", "q")
            assistant = {"message": {"role": "assistant", "content": "a"},
                         "usage": {"input": 7}}
            before = dump(path)
            # Wrong turn, payload-free and with nonzero usage: no effect anywhere.
            assert await storage.settle(session_id, "other", "success", None) is False
            assert await storage.settle(session_id, "other", "failure", assistant) is False
            assert dump(path) == before
            assert await storage.settle(session_id, turn_id, "success", None) is True
            settled = dump(path)
            # Repeated settlement with an assistant payload: still no effect.
            assert await storage.settle(session_id, turn_id, "success", assistant) is False
            assert dump(path) == settled
            assert json.loads(settled[0][0][2]) == {}
        finally:
            await storage.close()

    asyncio.run(run())


def test_settlement_writes_assistant_signal_and_settlement(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            turn_id, _ = await storage.admit(session_id, "r1", "q")
            assistant = {"message": {"role": "assistant", "content": "a"}, "usage": None}
            assert await storage.settle(session_id, turn_id, "success", assistant)
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert kinds(snapshot) == ["user", "assistant", "turn_settlement"]
            assert json.loads(snapshot.entries[2][1]) == {"status": "success"}

            turn_id, _ = await storage.admit(session_id, "r2", "q")
            assert await storage.settle(session_id, turn_id, "failure", None)
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert kinds(snapshot) == [
                "user", "assistant", "turn_settlement",
                "user", "signal", "turn_settlement",
            ]
            assert json.loads(snapshot.entries[4][1]) == {"status": "failure"}
        finally:
            await storage.close()
        _, entries = dump(path)
        assert json.loads(entries[1][4]) == assistant
        assert [row[2] for row in entries[1:]] == [None, None, "r2", None, None]

    asyncio.run(run())


def test_usage_accumulates_counters_and_keeps_payloads_verbatim(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            first = {"message": {"role": "assistant", "content": "a"},
                     "usage": {"input": 2, "output": 1.5, "flag": True,
                               "note": "x", "detail": {"a": 1}, "absent": None}}
            turn_id, _ = await storage.admit(session_id, "r1", "q")
            assert await storage.settle(session_id, turn_id, "success", first)
            second = {"message": {"role": "assistant", "content": "b"},
                      "usage": {"input": 3, "output": 0.25, "new": 4}}
            turn_id, _ = await storage.admit(session_id, "r2", "q")
            assert await storage.settle(session_id, turn_id, "success", second)
            third = {"message": {"role": "assistant", "content": "c"}, "usage": None}
            turn_id, _ = await storage.admit(session_id, "r3", "q")
            assert await storage.settle(session_id, turn_id, "success", third)
            turn_id, _ = await storage.admit(session_id, "r4", "q")
            assert await storage.settle(session_id, turn_id, "success", None)

            sessions, entries = dump(path)
            assert json.loads(sessions[0][2]) == {"input": 5, "output": 1.75, "new": 4}
            assert json.loads(entries[1][4]) == first
            assert json.loads(entries[4][4]) == second
            assert json.loads(entries[7][4]) == third
            assert [row[3] for row in entries] == [
                "user", "assistant", "turn_settlement",
                "user", "assistant", "turn_settlement",
                "user", "assistant", "turn_settlement",
                "user", "turn_settlement",
            ]
        finally:
            await storage.close()

    asyncio.run(run())


def test_non_finite_and_overflow_usage_rolls_back(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            turn_id, _ = await storage.admit(session_id, "r1", "q")
            infinite = {"message": {"role": "assistant", "content": "a"},
                        "usage": {"input": float("inf")}}
            with pytest.raises(ValueError):
                await storage.settle(session_id, turn_id, "success", infinite)
            sessions, entries = dump(path)
            assert sessions[0][1] == turn_id
            assert [row[3] for row in entries] == ["user"]

            overflowing = {"message": {"role": "assistant", "content": "a"},
                           "usage": {"x": 1.7e308}}
            assert await storage.settle(session_id, turn_id, "success", overflowing)
            turn_id, _ = await storage.admit(session_id, "r2", "q")
            before = dump(path)
            with pytest.raises(ValueError):
                await storage.settle(session_id, turn_id, "success", overflowing)
            assert dump(path) == before
            valid = {"message": {"role": "assistant", "content": "b"}, "usage": None}
            assert await storage.settle(session_id, turn_id, "success", valid)
        finally:
            await storage.close()

    asyncio.run(run())


def test_injected_assistant_insert_failure_rolls_back(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        connection = sqlite3.connect(path, isolation_level=None)
        try:
            session_id = await storage.create_session()
            turn_id, _ = await storage.admit(session_id, "r1", "q")
            connection.execute(
                "CREATE TRIGGER fail_assistant AFTER INSERT ON entries"
                " WHEN new.kind = 'assistant' BEGIN"
                " SELECT RAISE(ABORT, 'injected failure'); END"
            )
            before = dump(path)
            assistant = {"message": {"role": "assistant", "content": "a"}, "usage": None}
            with pytest.raises(sqlite3.Error):
                await storage.settle(session_id, turn_id, "success", assistant)
            assert dump(path) == before
            connection.execute("DROP TRIGGER fail_assistant")
            assert await storage.settle(session_id, turn_id, "success", assistant)
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert kinds(snapshot) == ["user", "assistant", "turn_settlement"]
        finally:
            connection.close()
            await storage.close()

    asyncio.run(run())


def test_injected_user_insert_failure_rolls_back(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        connection = sqlite3.connect(path, isolation_level=None)
        try:
            session_id = await storage.create_session()
            connection.execute(
                "CREATE TRIGGER fail_user AFTER INSERT ON entries"
                " WHEN new.kind = 'user' BEGIN"
                " SELECT RAISE(ABORT, 'injected failure'); END"
            )
            before = dump(path)
            with pytest.raises(sqlite3.Error):
                await storage.admit(session_id, "r1", "q")
            assert dump(path) == before
            connection.execute("DROP TRIGGER fail_user")
            turn_id, fresh = await storage.admit(session_id, "r1", "q")
            assert fresh
            snapshot = await storage.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id == turn_id
        finally:
            connection.close()
            await storage.close()

    asyncio.run(run())


def test_interrupt_all_recovers_running_sessions_idempotently(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            running = await storage.create_session()
            idle = await storage.create_session()
            turn_id, _ = await storage.admit(running, "r1", "q")
            # Restart boundary: reopen the same database before recovery.
            await storage.close()
            storage = await Storage.open(path)
            snapshot = await storage.snapshot(running)
            assert snapshot is not None
            assert snapshot.current_turn_id == turn_id
            assert kinds(snapshot) == ["user"]
            await storage.interrupt_all()
            snapshot = await storage.snapshot(running)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert kinds(snapshot) == ["user", "signal", "turn_settlement"]
            assert json.loads(snapshot.entries[1][1]) == {"status": "interrupted"}
            assert json.loads(snapshot.entries[2][1]) == {"status": "interrupted"}
            assert await storage.snapshot(idle) == (None, ())
            after_first = dump(path)
            await storage.interrupt_all()
            assert dump(path) == after_first
        finally:
            await storage.close()

    asyncio.run(run())


def test_cancelled_settlement_await_still_commits_and_fifo_scan_follows(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        first = await storage.create_session()
        second = await storage.create_session()
        first_turn, _ = await storage.admit(first, "r1", "q1")
        await storage.admit(second, "r2", "q2")

        entered = threading.Event()
        release = threading.Event()
        original = storage._guarded_settle

        def blocked(connection, session_id, turn_id, status, assistant):
            entered.set()
            release.wait()
            return original(connection, session_id, turn_id, status, assistant)

        storage._guarded_settle = blocked

        try:
            settle_task = asyncio.create_task(storage.settle(first, first_turn, "success", None))
            assert await asyncio.to_thread(entered.wait, 5)
            settle_task.cancel()
            # Enqueue the interrupt scan and close behind the still-running job.
            scan_task = asyncio.create_task(storage.interrupt_all())
            close_task = asyncio.create_task(storage.close())
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await settle_task
            await scan_task
            await close_task
        finally:
            release.set()

        reopened = await Storage.open(path)
        try:
            snapshot = await reopened.snapshot(first)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert kinds(snapshot) == ["user", "turn_settlement"]
            assert json.loads(snapshot.entries[1][1]) == {"status": "success"}
            snapshot = await reopened.snapshot(second)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert kinds(snapshot) == ["user", "signal", "turn_settlement"]
            assert json.loads(snapshot.entries[1][1]) == {"status": "interrupted"}
            _, entries = dump(path)
            assert [row[3] for row in entries].count("turn_settlement") == 2
        finally:
            await reopened.close()

    asyncio.run(run())


def test_cancelled_queued_job_writes_nothing(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        session_id = await storage.create_session()

        entered = threading.Event()
        release = threading.Event()
        original = storage._transaction

        def blocked(work):
            entered.set()
            release.wait()
            return original(work)

        storage._transaction = blocked

        try:
            holder = asyncio.create_task(storage.snapshot(session_id))
            assert await asyncio.to_thread(entered.wait, 5)
            admit_task = asyncio.create_task(storage.admit(session_id, "r1", "q"))
            await asyncio.sleep(0)
            assert not admit_task.done()
            admit_task.cancel()
            release.set()
            await holder
            with pytest.raises(asyncio.CancelledError):
                await admit_task
            await storage.close()
        finally:
            release.set()

        reopened = await Storage.open(path)
        try:
            snapshot = await reopened.snapshot(session_id)
            assert snapshot is not None
            assert snapshot.current_turn_id is None
            assert snapshot.entries == ()
        finally:
            await reopened.close()

    asyncio.run(run())


def test_open_failure_before_connection_creation_cleans_up(tmp_path, monkeypatch):
    holder = []
    patch_recording(monkeypatch)
    capture_initialization(monkeypatch, holder.append)

    async def run():
        with pytest.raises(sqlite3.OperationalError):
            await Storage.open(tmp_path / "missing-dir" / "db.sqlite3")

    asyncio.run(run())
    assert "closed_by" not in RECORDED
    assert RECORDED["shutdown_by"] != threading.get_ident()
    with pytest.raises(RuntimeError):
        holder[0]._executor.submit(lambda: None)


def test_open_failure_after_schema_error_closes_on_worker_thread(tmp_path, monkeypatch):
    patch_recording(monkeypatch, factory=BrokenSchemaConnection)
    holder = []
    capture_initialization(monkeypatch, holder.append)

    async def run():
        with pytest.raises(sqlite3.OperationalError, match="injected schema failure"):
            await Storage.open(tmp_path / "db.sqlite3")

    asyncio.run(run())
    assert RECORDED["closed_by"] == RECORDED["opened_by"]
    assert RECORDED["shutdown_by"] != threading.get_ident()
    with pytest.raises(RuntimeError):
        holder[0]._executor.submit(lambda: None)


def test_cancelled_initialization_still_cleans_up(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    holder = []

    def barrier(storage_instance):
        holder.append(storage_instance)
        RECORDED["worker"] = threading.get_ident()
        entered.set()
        release.wait()

    patch_recording(monkeypatch)
    capture_initialization(monkeypatch, barrier)

    async def run():
        task = asyncio.create_task(Storage.open(tmp_path / "db.sqlite3"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            outcome = dict(RECORDED)
            # The initialization still completed; the database is usable.
            reopened = await Storage.open(tmp_path / "db.sqlite3")
            try:
                session_id = await reopened.create_session()
                snapshot = await reopened.snapshot(session_id)
                assert snapshot is not None
            finally:
                await reopened.close()
            return outcome
        finally:
            release.set()

    outcome = asyncio.run(run())
    assert outcome["closed_by"] == outcome["opened_by"] == outcome["worker"]
    assert outcome["shutdown_by"] != threading.get_ident()
    with pytest.raises(RuntimeError):
        holder[0]._executor.submit(lambda: None)


def test_disk_full_auto_rollback_preserves_the_original_error(tmp_path):
    async def run():
        path = tmp_path / "db.sqlite3"
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            turn_id, _ = await storage.admit(session_id, "r1", "q")
            original = storage._guarded_settle
            state = {"fault": True}

            def disk_full(connection, session_id, turn_id, status, assistant):
                if state["fault"]:
                    state["fault"] = False
                    limit = connection.execute("PRAGMA page_count").fetchone()[0]
                    connection.execute(f"PRAGMA max_page_count = {limit}")
                    try:
                        return original(connection, session_id, turn_id, status, assistant)
                    finally:
                        connection.execute("PRAGMA max_page_count = 1000000")
                return original(connection, session_id, turn_id, status, assistant)

            storage._guarded_settle = disk_full
            assistant = {"message": {"role": "assistant", "content": "x" * 100000},
                         "usage": None}
            with pytest.raises(sqlite3.OperationalError, match="database or disk is full"):
                await storage.settle(session_id, turn_id, "success", assistant)
            sessions, entries = dump(path)
            assert sessions[0][1] == turn_id
            assert [row[3] for row in entries] == ["user"]
            assert await storage.settle(session_id, turn_id, "success", None) is True
        finally:
            await storage.close()

    asyncio.run(run())


def test_commit_failure_rolls_back_and_recovers(tmp_path, monkeypatch):
    async def run():
        path = tmp_path / "db.sqlite3"
        patch_recording(monkeypatch, factory=CommitFailureConnection)
        storage = await Storage.open(path)
        try:
            session_id = await storage.create_session()
            turn_id, _ = await storage.admit(session_id, "r1", "q")
            assistant = {"message": {"role": "assistant", "content": "a"},
                         "usage": {"input": 3}}
            # Arm the one-shot COMMIT fault for this settlement only.
            monkeypatch.setattr(CommitFailureConnection, "armed", True)
            with pytest.raises(sqlite3.OperationalError, match="database or disk is full"):
                await storage.settle(session_id, turn_id, "success", assistant)
            sessions, entries = dump(path)
            assert sessions[0][1] == turn_id
            assert sessions[0][2] == "{}"
            assert [row[3] for row in entries] == ["user"]
            # The one-shot fault disarmed itself; the next operation is usable.
            assert await storage.settle(session_id, turn_id, "success", assistant) is True
            sessions, entries = dump(path)
            assert json.loads(sessions[0][2]) == {"input": 3}
            assert [row[3] for row in entries] == ["user", "assistant", "turn_settlement"]
        finally:
            await storage.close()

    asyncio.run(run())
