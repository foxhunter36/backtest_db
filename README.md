# Backtest Database

Separate PostgreSQL-Datenbank (`bybit_backtest`) für historische Kline-Daten.
Unabhängig von der Live-DB (`bybit_market`) — kein Einfluss auf laufende Bots.

## Daten

- **55 Coins** (53 Momentum + NEARUSDT + SOLUSDT)
- **5 Timeframes**: 5m, 15m, 1h, 4h, 1d
- **365 Tage** Historie (konfigurierbar)

## Usage

```bash
# Initialer Backfill (alle Coins, 365 Tage, alle TFs)
python backfill.py

# Nur 180 Tage
python backfill.py --days 180

# Nur ein Symbol
python backfill.py --symbol BTCUSDT

# Nur ein Timeframe
python backfill.py --timeframe 5

# Delta: nur fehlende Daten nachfüllen (vor Backtests)
python backfill.py --delta

# Status: Row-Counts + Datenbereich
python backfill.py --status
```

## Zugriff von der Trading Station

```python
import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host="100.96.116.110",  # Tailscale IP
    port=5432,
    dbname="bybit_backtest",
    user="collector",
    password="bybit2026",
)

df = pd.read_sql_query(
    "SELECT * FROM kline_5m WHERE symbol = 'BTCUSDT' ORDER BY timestamp",
    conn,
)
```

## Struktur

```
backtest_db/
├── config.py          # DB-Connection, Symbols, Timeframes
├── schema.py          # CREATE TABLE + Indizes
├── backfill.py        # Backfill-Script (Initial + Delta)
├── requirements.txt
└── README.md
```
