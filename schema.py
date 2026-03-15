"""
schema.py – Backtest-DB Schema (PostgreSQL).

Erstellt Kline-Tabellen für alle Timeframes.
Composite PK (timestamp, symbol) — identisch zur Live-DB.
"""

import psycopg2
from config import PG_DSN, TIMEFRAMES, log


def get_connection():
    """Erstellt PostgreSQL-Verbindung zur Backtest-DB."""
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    return conn


def init_db(conn):
    """Erstellt alle Kline-Tabellen."""
    cur = conn.cursor()

    for interval, table in TIMEFRAMES.items():
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                timestamp   BIGINT NOT NULL,
                symbol      TEXT   NOT NULL,
                open        DOUBLE PRECISION NOT NULL,
                high        DOUBLE PRECISION NOT NULL,
                low         DOUBLE PRECISION NOT NULL,
                close       DOUBLE PRECISION NOT NULL,
                volume      DOUBLE PRECISION NOT NULL,
                turnover    DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (timestamp, symbol)
            )
        """)

        # Index für schnelle Symbol-Queries (Backtests laden immer per Symbol)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_symbol
            ON {table} (symbol, timestamp)
        """)

    conn.commit()
    log.info(f"Backtest-DB initialisiert: {len(TIMEFRAMES)} Tabellen "
             f"({', '.join(TIMEFRAMES.values())})")


def get_row_counts(conn, symbol=None):
    """Row-Counts aller Tabellen."""
    cur = conn.cursor()
    counts = {}
    for table in TIMEFRAMES.values():
        try:
            if symbol:
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE symbol = %s", (symbol,))
            else:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
        except Exception:
            counts[table] = 0
    return counts
