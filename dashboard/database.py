"""Optional PostgreSQL-backed semantic index for the dashboard."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

LOGGER = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class RunIndex(Base):
    __tablename__ = "dashboard_runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(Text)
    artifact_name: Mapped[str] = mapped_column(Text)
    task: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    stage: Mapped[str | None] = mapped_column(String(128), nullable=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    percent_complete: Mapped[float | None] = mapped_column(Float, nullable=True)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    episode_length: Mapped[float | None] = mapped_column(Float, nullable=True)
    kl: Mapped[float | None] = mapped_column(Float, nullable=True)
    entropy: Mapped[float | None] = mapped_column(Float, nullable=True)
    repository_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    run_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    modified_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    curriculum: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)


class DashboardEvent(Base):
    __tablename__ = "dashboard_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time, index=True)


class CheckpointIndex(Base):
    __tablename__ = "dashboard_checkpoints"

    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    relative_path: Mapped[str] = mapped_column(Text)
    iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stable: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    value = _float(value)
    return int(value) if value is not None else None


class DashboardDatabase:
    """Small resilient index. Filesystem monitoring still works when it is unavailable."""

    def __init__(self, url: str | None, repo_root: Path) -> None:
        self.url = url
        self.repo_root = repo_root
        self.engine = None
        self.sessions: sessionmaker[Session] | None = None
        self.error: str | None = None
        self._reconnect_lock = threading.Lock()
        self._failure_count = 0
        self._next_retry_at = 0.0
        self._retry_base_s = 0.5
        self._retry_max_s = 30.0
        self.last_successful_sync_at: float | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    @property
    def available(self) -> bool:
        return self.engine is not None and self.error is None

    def initialize(self, *, attempts: int = 10, retry_delay_s: float = 0.5) -> None:
        """Migrate/connect, but never make the filesystem control plane unavailable."""
        if not self.url:
            return
        attempts = max(1, attempts)
        for attempt in range(attempts):
            try:
                from alembic import command
                from alembic.config import Config

                config = Config(str(self.repo_root / "alembic.ini"))
                config.set_main_option("sqlalchemy.url", self.url)
                command.upgrade(config, "head")
                self._connect()
                return
            except Exception as error:
                self._mark_unavailable(error)
                if attempt + 1 < attempts:
                    time.sleep(max(0.0, retry_delay_s))

    def _connect(self) -> None:
        if not self.url:
            return
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if self.url.startswith("sqlite:"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        engine = create_engine(self.url, **kwargs)
        try:
            with engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
        except Exception:
            engine.dispose()
            raise
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        self.error = None
        self._failure_count = 0
        self._next_retry_at = 0.0

    def _ensure_available(self) -> bool:
        if not self.enabled:
            return False
        if self.available:
            return True
        if time.monotonic() < self._next_retry_at:
            return False
        if not self._reconnect_lock.acquire(blocking=False):
            return False
        try:
            if self.available:
                return True
            try:
                self._connect()
                LOGGER.info("Dashboard database connection recovered")
                return self.available
            except Exception as error:
                self._mark_unavailable(error)
                return False
        finally:
            self._reconnect_lock.release()

    def _mark_unavailable(self, error: Exception) -> None:
        self.error = str(error)
        self._failure_count += 1
        delay = min(
            self._retry_base_s * (2 ** (self._failure_count - 1)),
            self._retry_max_s,
        )
        self._next_retry_at = time.monotonic() + delay
        engine = self.engine
        self.engine = None
        self.sessions = None
        if engine is not None:
            engine.dispose()
        LOGGER.warning("Dashboard database unavailable; retry in %.1fs: %s", delay, error)

    def dispose(self) -> None:
        engine = self.engine
        self.engine = None
        self.sessions = None
        if engine is not None:
            engine.dispose()

    def status(self) -> dict[str, Any]:
        self._ensure_available()
        backend = None
        if self.url:
            backend = self.url.split(":", 1)[0].split("+", 1)[0]
        source_conflicts = None
        if self.available:
            try:
                with self._session() as session:
                    source_conflicts = session.scalar(
                        select(func.count())
                        .select_from(DashboardEvent)
                        .where(DashboardEvent.event_type == "run_source_conflict")
                    )
            except SQLAlchemyError as error:
                self._mark_unavailable(error)
        return {
            "enabled": self.enabled,
            "available": self.available,
            "backend": backend,
            "error": self.error,
            "source_conflicts": source_conflicts,
            "last_successful_sync_at": self.last_successful_sync_at,
        }

    def _session(self) -> Session:
        if self.sessions is None:
            raise RuntimeError("dashboard database is unavailable")
        return self.sessions()

    def record_event(
        self,
        event_type: str,
        message: str,
        *,
        run_id: str | None = None,
        payload: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> None:
        if not self._ensure_available():
            return
        try:
            with self._session() as session:
                session.add(
                    DashboardEvent(
                        run_id=run_id,
                        event_type=event_type,
                        message=message,
                        payload=payload,
                        created_at=created_at or time.time(),
                    )
                )
                session.commit()
        except SQLAlchemyError as error:
            self._mark_unavailable(error)

    def _insert_if_missing(
        self,
        session: Session,
        model: type[Base],
        values: dict[str, Any],
        key_column: str,
    ) -> bool:
        dialect = session.get_bind().dialect.name
        insert_factory = {
            "postgresql": postgresql_insert,
            "sqlite": sqlite_insert,
        }.get(dialect)
        if insert_factory is None:
            key = values[key_column]
            if session.get(model, key) is not None:
                return False
            session.add(model(**values))
            session.flush()
            return True
        key = getattr(model, key_column)
        statement = insert_factory(model).values(**values)
        statement = statement.on_conflict_do_nothing(index_elements=[key]).returning(key)
        return session.execute(statement).scalar_one_or_none() is not None

    @staticmethod
    def _insert_for(session: Session, model: type[Base]):
        dialect = session.get_bind().dialect.name
        insert_factory = {
            "postgresql": postgresql_insert,
            "sqlite": sqlite_insert,
        }.get(dialect)
        return insert_factory(model) if insert_factory else None

    def _record_source_conflict(
        self,
        session: Session,
        *,
        run_id: str,
        stored_artifact: str,
        incoming_artifact: str,
    ) -> bool:
        payload = {
            "stored_artifact": stored_artifact,
            "incoming_artifact": incoming_artifact,
        }
        existing = session.scalars(
            select(DashboardEvent)
            .where(
                DashboardEvent.run_id == run_id,
                DashboardEvent.event_type == "run_source_conflict",
            )
            .order_by(DashboardEvent.created_at.desc())
            .limit(100)
        ).all()
        if any(event.payload == payload for event in existing):
            return False
        session.add(
            DashboardEvent(
                run_id=run_id,
                event_type="run_source_conflict",
                message=(
                    f"Run {run_id} resolved to conflicting artifact paths; "
                    f"kept {stored_artifact} and rejected {incoming_artifact}"
                ),
                payload=payload,
            )
        )
        return True

    def run_artifact(self, run_id: str) -> str | None:
        """Return the indexed artifact path without scanning the artifact tree."""
        if not self._ensure_available():
            return None
        try:
            with self._session() as session:
                record = session.get(RunIndex, run_id)
                return record.artifact_name if record is not None else None
        except SQLAlchemyError as error:
            self._mark_unavailable(error)
            return None

    def list_runs(self, *, limit: int = 5000) -> list[dict[str, Any]] | None:
        """Read the last indexed run snapshot without walking artifact mounts."""
        if not self._ensure_available():
            return None
        try:
            with self._session() as session:
                records = session.scalars(
                    select(RunIndex)
                    .order_by(RunIndex.updated_at.desc())
                    .limit(max(1, min(int(limit), 5000)))
                ).all()
            now = time.time()
            rows = []
            for record in records:
                freshness = (
                    max(0.0, now - record.modified_at)
                    if record.modified_at is not None
                    else None
                )
                rows.append(
                    {
                        "id": record.id,
                        "name": record.artifact_name,
                        "display_name": record.display_name,
                        "task": record.task,
                        "stage": record.stage,
                        "state": record.state,
                        "stale": (
                            record.state in {"starting", "running"}
                            and freshness is not None
                            and freshness > 90.0
                        ),
                        "freshness_seconds": freshness,
                        "modified_at": record.modified_at,
                        "updated_at": record.updated_at,
                        "tags": [],
                        "purpose": "",
                        "lineage": {},
                        "repository_version": {
                            "status": record.repository_status,
                            "is_outdated": record.repository_status == "outdated",
                            "run_commit": record.run_commit,
                            "current_commit": None,
                        },
                        "iteration": record.iteration,
                        "total_iterations": record.total_iterations,
                        "percent_complete": record.percent_complete,
                        "eta_seconds": None,
                        "throughput": None,
                        "reward": record.reward,
                        "episode_length": record.episode_length,
                        "kl": record.kl,
                        "entropy": record.entropy,
                        "ppo_loss": None,
                        "clip_fraction": None,
                        "invalid_update": None,
                        "curriculum": record.curriculum,
                    }
                )
            return rows
        except SQLAlchemyError as error:
            self._mark_unavailable(error)
            return None

    def sync_run(
        self,
        row: dict[str, Any],
        curriculum: dict[str, Any] | None = None,
    ) -> bool:
        if not row.get("id") or not self._ensure_available():
            return False
        run_id = str(row["id"])
        repository = row.get("repository_version") or {}
        display_name = str(row.get("display_name") or row.get("name") or run_id)
        artifact_name = str(row.get("name") or "")
        state = str(row.get("state") or "unknown")
        values: dict[str, Any] = {
            "id": run_id,
            "display_name": display_name,
            "artifact_name": artifact_name,
            "task": row.get("task"),
            "stage": row.get("stage"),
            "state": state,
            "iteration": _int(row.get("iteration")),
            "total_iterations": _int(row.get("total_iterations")),
            "percent_complete": _float(row.get("percent_complete")),
            "reward": _float(row.get("reward")),
            "episode_length": _float(row.get("episode_length")),
            "kl": _float(row.get("kl")),
            "entropy": _float(row.get("entropy")),
            "repository_status": repository.get("status"),
            "run_commit": repository.get("run_commit"),
            "modified_at": _float(row.get("modified_at")),
            "updated_at": time.time(),
        }
        if curriculum is not None:
            values["curriculum"] = curriculum
        try:
            with self._session() as session:
                inserted = self._insert_if_missing(
                    session,
                    RunIndex,
                    values,
                    "id",
                )
                record = session.scalar(
                    select(RunIndex).where(RunIndex.id == run_id).with_for_update()
                )
                if record is None:
                    raise RuntimeError(f"run {run_id} disappeared during index synchronization")
                old_state = None if inserted else record.state
                old_curriculum = None if inserted else record.curriculum

                if record.artifact_name and artifact_name and record.artifact_name != artifact_name:
                    created = self._record_source_conflict(
                        session,
                        run_id=run_id,
                        stored_artifact=record.artifact_name,
                        incoming_artifact=artifact_name,
                    )
                    session.commit()
                    self.last_successful_sync_at = time.time()
                    if created:
                        LOGGER.error(
                            "Run ID %s maps to both %s and %s; preserving the indexed source",
                            run_id,
                            record.artifact_name,
                            artifact_name,
                        )
                    return False

                record.display_name = display_name
                record.artifact_name = artifact_name or record.artifact_name
                record.task = values["task"]
                record.stage = values["stage"]
                record.state = state
                record.iteration = values["iteration"]
                record.total_iterations = values["total_iterations"]
                record.percent_complete = values["percent_complete"]
                record.reward = values["reward"]
                record.episode_length = values["episode_length"]
                record.kl = values["kl"]
                record.entropy = values["entropy"]
                record.repository_status = values["repository_status"]
                record.run_commit = values["run_commit"]
                record.modified_at = values["modified_at"]
                if curriculum is not None:
                    record.curriculum = curriculum
                record.updated_at = values["updated_at"]

                if old_state is not None and old_state != record.state:
                    session.add(
                        DashboardEvent(
                            run_id=run_id,
                            event_type="run_state",
                            message=f"{record.display_name}: {old_state} → {record.state}",
                            payload={"from": old_state, "to": record.state},
                        )
                    )
                old_stage = (
                    old_curriculum.get("stage") if isinstance(old_curriculum, dict) else None
                )
                new_stage = curriculum.get("stage") if isinstance(curriculum, dict) else None
                if old_stage is not None and new_stage is not None and old_stage != new_stage:
                    session.add(
                        DashboardEvent(
                            run_id=run_id,
                            event_type="curriculum_transition",
                            message=(
                                f"{record.display_name}: curriculum stage "
                                f"{old_stage} → {new_stage}"
                            ),
                            payload={"from": old_stage, "to": new_stage},
                        )
                    )
                session.commit()
                self.last_successful_sync_at = time.time()
                return True
        except SQLAlchemyError as error:
            self._mark_unavailable(error)
            return False

    def recent_events(self, *, run_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if not self._ensure_available():
            return []
        limit = max(1, min(limit, 100))
        try:
            with self._session() as session:
                statement = select(DashboardEvent)
                if run_id is not None:
                    statement = statement.where(DashboardEvent.run_id == run_id)
                statement = statement.order_by(DashboardEvent.created_at.desc()).limit(limit)
                items = session.scalars(statement).all()
                return [
                    {
                        "id": item.id,
                        "run_id": item.run_id,
                        "type": item.event_type,
                        "message": item.message,
                        "payload": item.payload,
                        "created_at": item.created_at,
                    }
                    for item in items
                ]
        except SQLAlchemyError as error:
            self._mark_unavailable(error)
            return []

    def sync_checkpoints(self, run_id: str, checkpoints: list[dict[str, Any]]) -> None:
        if not self._ensure_available():
            return
        try:
            with self._session() as session:
                for checkpoint in checkpoints:
                    relative = str(checkpoint.get("relative_path") or "")
                    if not relative:
                        continue
                    key = f"{run_id}:{relative}"
                    values = {
                        "id": key,
                        "run_id": run_id,
                        "relative_path": relative,
                        "iteration": _int(checkpoint.get("iteration")),
                        "stable": bool(checkpoint.get("stable", True)),
                        "created_at": time.time(),
                    }
                    statement = self._insert_for(session, CheckpointIndex)
                    if statement is None:
                        record = session.get(CheckpointIndex, key)
                        if record is None:
                            session.add(CheckpointIndex(**values))
                        else:
                            record.iteration = values["iteration"]
                            record.stable = values["stable"]
                        continue
                    excluded = statement.excluded
                    session.execute(
                        statement.values(**values).on_conflict_do_update(
                            index_elements=[CheckpointIndex.id],
                            set_={
                                "iteration": excluded.iteration,
                                "stable": excluded.stable,
                            },
                        )
                    )
                session.commit()
                if checkpoints:
                    self.last_successful_sync_at = time.time()
        except SQLAlchemyError as error:
            self._mark_unavailable(error)

