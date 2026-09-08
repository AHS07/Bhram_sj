"""SQLite engine + session helper.

WAL mode: enables concurrent reads during writes, eliminating "database is
locked" errors when the UI polls /facts or /relations while an upload or
reconciliation is running.

busy_timeout: SQLite waits up to 30 s for a lock to clear before raising
OperationalError, instead of failing immediately.

FK enforcement: SQLite does not enforce foreign keys unless explicitly
enabled per connection; we do so in the connect_args.
"""
from contextlib import contextmanager
from sqlalchemy import event, text
from sqlmodel import SQLModel, Session, create_engine

engine = create_engine(
    "sqlite:///bhram.db",
    connect_args={
        "check_same_thread": False,
        "timeout": 30,           # busy_timeout in seconds
    },
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")   # concurrent reads + writes
    cursor.execute("PRAGMA foreign_keys=ON")    # enforce FK constraints
    cursor.execute("PRAGMA synchronous=NORMAL") # safe + faster than FULL under WAL
    cursor.close()


def init_db():
    SQLModel.metadata.create_all(engine)
    with engine.connect() as conn:
        try:
            cursor = conn.exec_driver_sql("PRAGMA table_info(document)")
            columns = [row[1] for row in cursor.fetchall()]
            if "content_hash" not in columns:
                conn.exec_driver_sql("ALTER TABLE document ADD COLUMN content_hash TEXT DEFAULT ''")
                conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_document_content_hash ON document (content_hash)")
                conn.commit()
        except Exception:
            pass


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session
