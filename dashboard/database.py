"""Optional PostgreSQL-backed semantic index for the dashboard."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, Float, Integer, String, Text, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


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
                kwargs: dict[str, Any] = {"pool_pre_ping": True}
                if self.url.startswith("sqlite:"):
                    kwargs["connect_args"] = {"check_same_thread": False}
                engine = create_engine(self.url, **kwargs)
                with engine.connect() as connection:
                    connection.exec_driver_sql("SELECT 1")
                self.engine = engine
                self.sessions = sessionmaker(engine, expire_on_commit=False)
                self.error = None
                return
            except Exception as error:
                self._mark_unavailable(error)
                if attempt + 1 < attempts:
                    time.sleep(max(0.0, retry_delay_s))

    def _mark_unavailable(self, error: Exception) -> None:
        self.error = str(error)
        engine = self.engine
        self.engine = None
        self.sessions = None
        if engine is not None:
            engine.dispose()

    def dispose(self) -> None:
        engine = self.engine
        self.engine = None
        self.sessions = None
        if engine is not None:
            engine.dispose()

    def status(self) -> dict[str, Any]:
        backend = None
        if self.url:
            backend = self.url.split(":", 1)[0].split("+", 1)[0]
        return {
            "enabled": self.enabled,
            "available": self.available,
            "backend": backend,
            "error": self.error,
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
        if not self.available:
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

    def sync_run(self, row: dict[str, Any], curriculum: dict[str, Any] | None = None) -> None:
        if not self.available or not row.get("id"):
            return
        run_id = str(row["id"])
        repository = row.get("repository_version") or {}
        try:
            with self._session() as session:
                record = session.get(RunIndex, run_id)
                old_state = record.state if record is not None else None
                old_curriculum = record.curriculum if record is not None else None
                if record is None:
                    record = RunIndex(
                        id=run_id,
                        display_name=str(row.get("display_name") or row.get("name") or run_id),
                        artifact_name=str(row.get("name") or ""),
                        state=str(row.get("state") or "unknown"),
                    )
                    session.add(record)

                record.display_name = str(row.get("display_name") or row.get("name") or run_id)
                record.artifact_name = str(row.get("name") or "")
                record.task = row.get("task")
                record.stage = row.get("stage")
                record.state = str(row.get("state") or "unknown")
                record.iteration = _int(row.get("iteration"))
                record.total_iterations = _int(row.get("total_iterations"))
                record.percent_complete = _float(row.get("percent_complete"))
                record.reward = _float(row.get("reward"))
                record.episode_length = _float(row.get("episode_length"))
                record.kl = _float(row.get("kl"))
                record.entropy = _float(row.get("entropy"))
                record.repository_status = repository.get("status")
                record.run_commit = repository.get("run_commit")
                record.modified_at = _float(row.get("modified_at"))
                if curriculum is not None:
                    record.curriculum = curriculum
                record.updated_at = time.time()

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
        except SQLAlchemyError as error:
            self._mark_unavailable(error)

    def recent_events(self, *, run_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if not self.available:
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
        if not self.available:
            return
        try:
            with self._session() as session:
                for checkpoint in checkpoints:
                    relative = str(checkpoint.get("relative_path") or "")
                    if not relative:
                        continue
                    key = f"{run_id}:{relative}"
                    record = session.get(CheckpointIndex, key)
                    if record is None:
                        record = CheckpointIndex(
                            id=key,
                            run_id=run_id,
                            relative_path=relative,
                        )
                        session.add(record)
                    record.iteration = _int(checkpoint.get("iteration"))
                    record.stable = bool(checkpoint.get("stable", True))
                session.commit()
        except SQLAlchemyError as error:
            self._mark_unavailable(error)

