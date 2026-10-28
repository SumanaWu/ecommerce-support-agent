"""SQLite v3: validate first, convert atomically, preserve auditable old snapshots."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from sqlalchemy import inspect

from src.money import major_to_minor
from src.rules import business_fingerprint


def migrate_money(engine):
    inspector = inspect(engine)
    monetary = {'orders': 'order_id', 'payments': 'payment_id', 'pending_actions': 'action_id'}
    legacy = [table for table in monetary if inspector.has_table(table) and
              'amount' in {column['name'] for column in inspector.get_columns(table)}]
    if not legacy:
        return
    if engine.dialect.name != 'sqlite' or sqlite3.sqlite_version_info < (3, 35, 0):
        raise RuntimeError('整数金额旧库迁移需要 SQLite 3.35 或以上；请先升级运行环境。')
    currency = os.environ.get('SUPPORT_LEGACY_CURRENCY') or 'CNY'
    major_to_minor(0, currency)
    database = engine.url.database
    if database and database != ':memory:':
        with sqlite3.connect(database) as source, sqlite3.connect(str(Path(database)) + '.before-v3-' + uuid4().hex + '.bak') as backup:
            source.backup(backup)
    with engine.connect() as connection:
        connection.exec_driver_sql('BEGIN IMMEDIATE')
        try:
            # Read only fixed table names. No SQL identifier comes from a document/client.
            names = set(inspect(connection).get_table_names())
            facts = {table: [dict(row) for row in connection.exec_driver_sql(f'SELECT * FROM {table}').mappings()]
                     for table in (*monetary, 'refunds') if table in names}
            converted = {}
            for table in legacy:
                converted[table] = {}
                for row in facts[table]:
                    row_currency = row.get('currency') or currency
                    minor = major_to_minor(row['amount'], row_currency)
                    converted[table][row[monetary[table]]] = (minor, row_currency)
            # Only rebase proposals that were genuinely current before conversion.
            orders = {row['order_id']: row for row in facts.get('orders', [])}
            payments = {row['order_id']: row for row in facts.get('payments', [])}
            refunds = {row['order_id']: row for row in facts.get('refunds', [])}
            current = []
            for action in facts.get('pending_actions', []):
                order = orders.get(action['order_id'])
                if action['status'] != 'pending' or order is None:
                    continue
                payment, refund = payments.get(order['order_id']), refunds.get(order['order_id'])
                old = [order['customer_id'], order['version'], order['status'], order['amount'], order['item'], order['order_date'],
                       [payment['payment_id'], payment['status'], payment['method'], payment['amount']] if payment else None,
                       [refund['refund_id'], refund['status'], refund['reason']] if refund else None]
                digest = hashlib.sha256(json.dumps(old, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                price = payment if action['tool'] == 'process_payment' else order
                price_table = 'payments' if action['tool'] == 'process_payment' else 'orders'
                if (digest == action['business_version'] and price and
                        converted['pending_actions'][action['action_id']] == converted[price_table][price[monetary[price_table]]]):
                    current.append(action)
            for table in legacy:
                columns = {column['name'] for column in inspect(connection).get_columns(table)}
                if 'amount_minor' not in columns:
                    connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN amount_minor BIGINT NOT NULL DEFAULT 0 CHECK(amount_minor >= 0 AND typeof(amount_minor) = 'integer')")
                if 'currency' not in columns:
                    connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN currency VARCHAR(3) NOT NULL DEFAULT '{currency}' CHECK(currency IN ('CNY', 'USD'))")
                for key, (minor, row_currency) in converted[table].items():
                    connection.exec_driver_sql(f'UPDATE {table} SET amount_minor=?,currency=? WHERE {monetary[table]}=?', (minor, row_currency, key))
                connection.exec_driver_sql(f'ALTER TABLE {table} DROP COLUMN amount')
            for action in current:
                order, payment, refund = orders[action['order_id']], payments.get(action['order_id']), refunds.get(action['order_id'])
                for table, row in [('orders', order), ('payments', payment)]:
                    if row is not None:
                        row['amount_minor'], row['currency'] = converted[table][row[monetary[table]]]
                digest = business_fingerprint(order, payment, refund)
                connection.exec_driver_sql('UPDATE pending_actions SET business_version=? WHERE action_id=?', (digest, action['action_id']))
            # Register the money migration in the same transaction as the conversion.
            connection.exec_driver_sql('CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY)')
            connection.exec_driver_sql('INSERT OR IGNORE INTO schema_migrations VALUES(3)')
            connection.commit()
        except Exception:
            connection.rollback()
            raise
