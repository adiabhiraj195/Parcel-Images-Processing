"""Database engine, session factory, and SQLite initialization."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.engine import Engine

from app.config import settings

# SQLite configuration
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints and WAL mode for SQLite."""
    if settings.DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for scoped database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables defined in models and ensure migrations are applied."""
    from app.models import Batch, Image, Extraction  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # SQLite migration helper to ensure newly added columns exist in existing DB
    if settings.DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import text
        with engine.connect() as conn:
            try:
                result = conn.execute(text("PRAGMA table_info(extractions);")).fetchall()
                col_names = [row[1] for row in result]
                if col_names and "parcel_label_outcome" not in col_names:
                    conn.execute(text("ALTER TABLE extractions ADD COLUMN parcel_label_outcome VARCHAR(50);"))
                    conn.commit()
            except Exception:
                pass

            try:
                img_result = conn.execute(text("PRAGMA table_info(images);")).fetchall()
                img_cols = [row[1] for row in img_result]
                if img_cols:
                    if "current_stage" not in img_cols:
                        conn.execute(text("ALTER TABLE images ADD COLUMN current_stage VARCHAR(50) DEFAULT 'PENDING';"))
                    if "stage_message" not in img_cols:
                        conn.execute(text("ALTER TABLE images ADD COLUMN stage_message VARCHAR(255) DEFAULT 'Waiting in queue';"))
                    if "stage_progress" not in img_cols:
                        conn.execute(text("ALTER TABLE images ADD COLUMN stage_progress INTEGER DEFAULT 0;"))
                    if "stage_started_at" not in img_cols:
                        conn.execute(text("ALTER TABLE images ADD COLUMN stage_started_at DATETIME;"))
                    conn.commit()
            except Exception:
                pass
