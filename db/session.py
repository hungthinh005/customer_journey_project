"""SQLAlchemy engine + session management (SQLAlchemy 1.4, Airflow-compatible)."""

import sys
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import settings

Base = declarative_base()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session():
    """Return a new session. Caller is responsible for closing it."""
    return SessionLocal()


@contextmanager
def session_scope():
    """Transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
