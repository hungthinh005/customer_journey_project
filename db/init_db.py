"""Create all serving-store tables. Idempotent (CREATE TABLE IF NOT EXISTS)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import models  # noqa: F401  (registers models on Base.metadata)
from db.session import Base, engine


def init_db():
    print("Creating serving-store tables...")
    Base.metadata.create_all(bind=engine)
    print("  [OK] Tables ready:")
    for table in Base.metadata.sorted_tables:
        print(f"    - {table.name}")


if __name__ == "__main__":
    init_db()
