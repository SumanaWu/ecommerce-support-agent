import os
import sqlite3
from datetime import date
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

from sqlalchemy import create_engine, BigInteger, CheckConstraint, Date, Float, ForeignKey, Integer, String, Text, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(100))
    item: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20))  # placed/shipped/delivered/cancelled
    order_date: Mapped[date] = mapped_column(Date)
    amount_minor: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(String(3))
    __table_args__ = (CheckConstraint("amount_minor >= 0 AND typeof(amount_minor) = 'integer'"),
                      CheckConstraint("currency IN ('CNY', 'USD')"))
    customer_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default='1')


class Payment(Base):
    __tablename__ = "payments"

    payment_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"), unique=True)
    status: Mapped[str] = mapped_column(String(20))  # pending/paid/refunded
    method: Mapped[str] = mapped_column(String(30))
    amount_minor: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(String(3))
    __table_args__ = (CheckConstraint("amount_minor >= 0 AND typeof(amount_minor) = 'integer'"),
                      CheckConstraint("currency IN ('CNY', 'USD')"))


class Refund(Base):
    __tablename__ = "refunds"

    refund_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="none")  # none/requested/approved/completed
    reason: Mapped[str | None] = mapped_column(String(300), nullable=True)


def get_engine(db_url: str | None = None):
    default_path = Path(os.environ.get('SUPPORT_DATA_DIR', str(PROJECT_ROOT / 'data'))) / 'business.sqlite'
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_url = f"sqlite:///{default_path.as_posix()}"
    url = db_url or os.environ.get("SUPPORT_AGENT_DB_URL") or default_url
    return create_engine(url, echo=False)


def init_db(engine) -> None:
    inspector = inspect(engine)
    if inspector.has_table('orders'):
        columns = {column['name'] for column in inspector.get_columns('orders')}
        with engine.connect() as connection:
            migrated = inspector.has_table('schema_migrations') and connection.execute(
                text('SELECT 1 FROM schema_migrations WHERE version=2')).first()
        if not migrated:
            if engine.dialect.name != 'sqlite':
                raise RuntimeError('第二阶段旧库迁移目前仅支持 SQLite。')
            database = engine.url.database
            if database and database != ':memory:':
                backup = str(Path(database)) + '.before-v2-' + uuid4().hex + '.bak'
                with sqlite3.connect(database) as source, sqlite3.connect(backup) as target:
                    source.backup(target)
            with engine.begin() as connection:
                if 'customer_id' not in columns:
                    connection.execute(text('ALTER TABLE orders ADD COLUMN customer_id VARCHAR(40)'))
                if 'version' not in columns:
                    connection.execute(text('ALTER TABLE orders ADD COLUMN version INTEGER NOT NULL DEFAULT 1'))
                # Only known sample IDs have a defined mapping; unknown legacy
                # orders stay unowned instead of guessing from customer names.
                for order_id in range(1001, 1011):
                    connection.execute(text('UPDATE orders SET customer_id=:owner WHERE order_id=:id AND customer_id IS NULL'),
                                       {'owner': sample_owner(order_id), 'id': order_id})
    from src.backend.migrations import migrate_money
    migrate_money(engine)
    inspector = inspect(engine)
    if inspector.has_table('orders') and not inspector.has_table('execution_budgets'):
        database = engine.url.database
        if database and database != ':memory:':
            with sqlite3.connect(database) as source, sqlite3.connect(str(Path(database)) + '.before-v4-' + uuid4().hex + '.bak') as target:
                source.backup(target)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        for version in (1, 2, 3, 4):
            if connection.execute(SchemaMigration.__table__.select().where(SchemaMigration.version == version)).first() is None:
                connection.execute(SchemaMigration.__table__.insert().values(version=version))


def sample_owner(order_id):
    if 1001 <= order_id <= 1007:
        return 'customer_a'
    if 1008 <= order_id <= 1010:
        return 'customer_b'
    return None


def get_session_factory(engine):
    return sessionmaker(bind=engine)



class Ticket(Base):
    __tablename__ = "tickets"
    ticket_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    reason: Mapped[str] = mapped_column(String(300))
    context: Mapped[str] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(String(20), default="pending")


class RequestRecord(Base):
    __tablename__ = "request_records"
    user_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    mode: Mapped[str] = mapped_column(String(10), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)


class ActionRecord(Base):
    __tablename__ = "action_records"
    user_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    mode: Mapped[str] = mapped_column(String(10), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    action_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    tool: Mapped[str] = mapped_column(String(50))
    args: Mapped[str] = mapped_column(Text)
    result: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)


class PendingAction(Base):
    __tablename__ = 'pending_actions'
    action_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(40))
    mode: Mapped[str] = mapped_column(String(10))
    session_id: Mapped[str] = mapped_column(String(40))
    source_request_id: Mapped[str] = mapped_column(String(64))
    tool: Mapped[str] = mapped_column(String(50))
    order_id: Mapped[int] = mapped_column(ForeignKey('orders.order_id'))
    args: Mapped[str] = mapped_column(Text)
    business_version: Mapped[str] = mapped_column(String(64))
    amount_minor: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(String(3))
    __table_args__ = (CheckConstraint("amount_minor >= 0 AND typeof(amount_minor) = 'integer'"),
                      CheckConstraint("currency IN ('CNY', 'USD')"))
    status: Mapped[str] = mapped_column(String(20), default='pending')
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float)


class PolicyDocument(Base):
    __tablename__ = 'policy_documents'
    policy_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    version: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(Text)
    effective_from: Mapped[date] = mapped_column(Date)
    keywords: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    revision: Mapped[int] = mapped_column(Integer, unique=True)


class PolicyChunk(Base):
    __tablename__ = 'policy_chunks'
    policy_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    version: Mapped[str] = mapped_column(String(40), primary_key=True)
    chunk_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    section: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    search_text: Mapped[str] = mapped_column(Text)


class ExecutionBudgetRecord(Base):
    __tablename__ = 'execution_budgets'
    user_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    mode: Mapped[str] = mapped_column(String(10), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_calls: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    max_model_calls: Mapped[int] = mapped_column(Integer)
    max_tool_calls: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[float] = mapped_column(Float)
    deadline_at: Mapped[float] = mapped_column(Float)
