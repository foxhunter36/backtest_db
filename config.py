"""
config.py – Backtest-Datenbank Konfiguration.

Separate DB (bybit_backtest) für historische Daten.
Kein Einfluss auf die Live-DB (bybit_market).

55 Coins × 5 Timeframes (5m, 15m, 1h, 4h, 1d) × 365 Tage.
"""

import os

# ── PostgreSQL ─────────────────────────────────────────────────────────
PG_DSN = os.getenv(
    "BACKTEST_PG_DSN",
    "host=localhost port=5432 dbname=bybit_backtest user=collector password=bybit2026"
)

# ── Symbols (55: gleich wie Collector) ─────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "TRUMPUSDT", "XRPUSDT", "HYPEUSDT",
    "RIVERUSDT", "DOGEUSDT", "LYNUSDT", "TAOUSDT", "PIXELUSDT", "SUIUSDT",
    "CUSDT", "XANUSDT", "MNTUSDT", "1000PEPEUSDT", "ZECUSDT", "ADAUSDT",
    "FARTCOINUSDT", "AXSUSDT", "BANANAS31USDT", "APRUSDT", "SAHARAUSDT",
    "NEARUSDT", "DOTUSDT", "LINKUSDT", "LAUSDT", "HBARUSDT", "WIFUSDT",
    "RENDERUSDT", "RESOLVUSDT", "MYXUSDT", "AVAXUSDT", "AAVEUSDT", "BNBUSDT",
    "FLOWUSDT", "VIRTUALUSDT", "BCHUSDT", "PIPPINUSDT", "KITEUSDT", "MBOXUSDT",
    "ENAUSDT", "XPLUSDT", "ICPUSDT", "LTCUSDT", "BERAUSDT", "ASTERUSDT",
    "TOWNSUSDT", "GALAUSDT", "DEXEUSDT", "CRVUSDT", "XMRUSDT", "VVVUSDT",
    "BEATUSDT", "PUMPFUNUSDT",
]

CATEGORY = "linear"

# ── Timeframes ─────────────────────────────────────────────────────────
# Bybit REST interval → DB-Tabellenname
TIMEFRAMES = {
    "5":   "kline_5m",
    "15":  "kline_15m",
    "60":  "kline_1h",
    "240": "kline_4h",
    "D":   "kline_1d",
}

# ── Backfill ───────────────────────────────────────────────────────────
DEFAULT_DAYS = 365
REQUEST_DELAY = 0.12  # Sekunden zwischen API-Calls (Rate Limit)
BATCH_SIZE = 200      # Candles pro API-Call (Bybit Max)

# ── Logging ────────────────────────────────────────────────────────────
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("backtest_db")
