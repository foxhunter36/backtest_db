"""
backfill.py – Backtest-DB Backfill Script.

Holt historische Kline-Daten von Bybit REST API
und schreibt sie in die Backtest-DB (bybit_backtest).

Kann vom Server ODER der Trading Station ausgeführt werden
(solange PostgreSQL erreichbar ist).

Usage:
    python backfill.py                      # Alle 55 Coins, 365 Tage, alle TFs
    python backfill.py --days 180           # Nur 180 Tage
    python backfill.py --symbol BTCUSDT     # Nur ein Symbol
    python backfill.py --timeframe 5        # Nur 5m
    python backfill.py --status             # Row-Counts anzeigen
    python backfill.py --delta              # Nur fehlende Daten nachfüllen
"""

import argparse
import time
import signal
import sys
from pybit.unified_trading import HTTP

from config import (
    SYMBOLS, CATEGORY, TIMEFRAMES, DEFAULT_DAYS,
    REQUEST_DELAY, BATCH_SIZE, log,
)
from schema import get_connection, init_db, get_row_counts

session = HTTP()

_running = True


def _signal_handler(sig, frame):
    global _running
    log.info("Abbruch angefordert...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ═══════════════════════════════════════════════════════════════════════
#  FETCH + INSERT
# ═══════════════════════════════════════════════════════════════════════

def fetch_kline(conn, symbol, interval, end_ms=None, limit=BATCH_SIZE):
    """Holt Klines von Bybit und schreibt in DB. Returns: Anzahl neue Rows."""
    table = TIMEFRAMES.get(interval)
    if not table:
        return 0

    params = dict(category=CATEGORY, symbol=symbol, interval=interval, limit=limit)
    if end_ms:
        params["end"] = end_ms

    try:
        r = session.get_kline(**params)
    except Exception as e:
        log.error(f"API-Fehler {symbol}/{interval}: {e}")
        return 0

    if r.get("retCode") != 0:
        log.debug(f"Bybit retCode={r.get('retCode')} bei {symbol}/{interval}")
        return 0

    rows = r.get("result", {}).get("list", [])
    if not rows:
        return 0

    now_ms = int(time.time() * 1000)
    cur = conn.cursor()
    inserted = 0

    for candle in rows:
        try:
            ts, o, h, l, c, vol, turn = candle
            if int(ts) >= now_ms:
                continue
            cur.execute(
                f"""INSERT INTO {table} VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING""",
                (int(ts), symbol, float(o), float(h), float(l),
                 float(c), float(vol), float(turn))
            )
            inserted += cur.rowcount
        except (ValueError, IndexError) as e:
            log.debug(f"Parse-Fehler {symbol}/{interval}: {e}")

    conn.commit()
    return inserted


def backfill_symbol_tf(conn, symbol, interval, days):
    """Backfill für ein Symbol/Timeframe-Paar."""
    table = TIMEFRAMES.get(interval)
    if not table:
        return 0

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 86_400_000)
    end_ms = now_ms
    total = 0
    calls = 0

    while end_ms > start_ms and _running:
        n = fetch_kline(conn, symbol, interval, end_ms=end_ms, limit=BATCH_SIZE)
        total += n
        calls += 1

        if n == 0:
            break

        # Ältesten Timestamp finden für nächste Iteration
        cur = conn.cursor()
        cur.execute(
            f"SELECT MIN(timestamp) FROM {table} WHERE symbol = %s AND timestamp <= %s",
            (symbol, end_ms)
        )
        oldest = cur.fetchone()[0]
        if oldest is None or oldest >= end_ms:
            break
        end_ms = oldest - 1

        time.sleep(REQUEST_DELAY)

        if calls % 100 == 0:
            log.info(f"    {symbol} {table}: {total} Rows, {calls} Calls...")

    return total


def delta_backfill(conn, symbol, interval):
    """Füllt nur fehlende Daten nach (ab letztem Timestamp)."""
    table = TIMEFRAMES.get(interval)
    if not table:
        return 0

    cur = conn.cursor()
    cur.execute(
        f"SELECT MAX(timestamp) FROM {table} WHERE symbol = %s",
        (symbol,)
    )
    row = cur.fetchone()
    last_ts = row[0] if row and row[0] else None

    if last_ts is None:
        log.info(f"  {symbol} {table}: keine Daten → überspringe Delta (nutze --days)")
        return 0

    # Hole alles ab dem letzten Timestamp
    total = 0
    n = fetch_kline(conn, symbol, interval, limit=BATCH_SIZE)
    total += n

    if n > 0:
        log.info(f"  {symbol} {table}: {n} neue Candles (Delta)")

    return total


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def run_backfill(conn, symbols, timeframes, days):
    """Voller Backfill für alle Symbols × Timeframes."""
    t0 = time.time()
    total = 0
    n_pairs = len(symbols) * len(timeframes)
    done = 0

    log.info(f"{'='*60}")
    log.info(f"BACKFILL START")
    log.info(f"  Symbols:    {len(symbols)}")
    log.info(f"  Timeframes: {list(timeframes.keys())}")
    log.info(f"  Tage:       {days}")
    log.info(f"  Paare:      {n_pairs}")
    log.info(f"{'='*60}")

    # Größere TFs zuerst (weniger Calls, schneller)
    tf_order = ["D", "240", "60", "15", "5"]
    sorted_tfs = [tf for tf in tf_order if tf in timeframes]

    for i, symbol in enumerate(symbols, 1):
        if not _running:
            break

        symbol_total = 0
        for interval in sorted_tfs:
            if not _running:
                break

            n = backfill_symbol_tf(conn, symbol, interval, days)
            symbol_total += n
            total += n
            done += 1

        pct = (done / n_pairs) * 100
        elapsed = time.time() - t0
        log.info(f"[{i}/{len(symbols)}] {symbol}: {symbol_total} Rows "
                 f"({pct:.0f}% | {elapsed:.0f}s)")

    elapsed = time.time() - t0
    log.info(f"{'='*60}")
    log.info(f"BACKFILL FERTIG: {total:,} Rows in {elapsed:.0f}s")
    log.info(f"{'='*60}")


def run_delta(conn, symbols, timeframes):
    """Delta-Backfill: nur fehlende Daten nachfüllen."""
    t0 = time.time()
    total = 0

    log.info(f"DELTA-BACKFILL: {len(symbols)} Symbols × {len(timeframes)} TFs")

    for i, symbol in enumerate(symbols, 1):
        if not _running:
            break

        for interval in timeframes:
            if not _running:
                break
            n = delta_backfill(conn, symbol, interval)
            total += n

        if (i % 10 == 0):
            log.info(f"  Delta: {i}/{len(symbols)} Symbols verarbeitet...")

    elapsed = time.time() - t0
    log.info(f"DELTA FERTIG: {total} neue Rows in {elapsed:.0f}s")


def print_status(conn):
    """Zeigt Row-Counts."""
    counts = get_row_counts(conn)
    total = sum(counts.values())

    log.info(f"─── BACKTEST-DB Status ────────────────────────")
    for table, count in counts.items():
        log.info(f"  {table:.<20s} {count:>12,} Rows")
    log.info(f"  {'TOTAL':.<20s} {total:>12,} Rows")
    log.info(f"───────────────────────────────────────────────")

    # Stichprobe: Älteste + neueste Candle pro Timeframe
    cur = conn.cursor()
    for table in TIMEFRAMES.values():
        try:
            cur.execute(f"SELECT MIN(timestamp), MAX(timestamp), COUNT(DISTINCT symbol) FROM {table}")
            row = cur.fetchone()
            if row and row[0]:
                from datetime import datetime
                oldest = datetime.utcfromtimestamp(row[0] / 1000).strftime("%Y-%m-%d")
                newest = datetime.utcfromtimestamp(row[1] / 1000).strftime("%Y-%m-%d")
                n_sym = row[2]
                log.info(f"  {table}: {oldest} → {newest} ({n_sym} Symbols)")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Backtest-DB Backfill (Bybit REST API → PostgreSQL)"
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Backfill-Tiefe in Tagen (default: {DEFAULT_DAYS})")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Nur ein Symbol backfillen")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Nur ein Timeframe (5, 15, 60, 240, D)")
    parser.add_argument("--status", action="store_true",
                        help="Row-Counts anzeigen")
    parser.add_argument("--delta", action="store_true",
                        help="Nur fehlende Daten nachfüllen")
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)

    if args.status:
        print_status(conn)
        conn.close()
        return

    # Filter
    symbols = SYMBOLS
    if args.symbol:
        if args.symbol not in SYMBOLS:
            log.error(f"Symbol {args.symbol} nicht in der Liste!")
            conn.close()
            return
        symbols = [args.symbol]

    timeframes = TIMEFRAMES
    if args.timeframe:
        if args.timeframe not in TIMEFRAMES:
            log.error(f"Timeframe {args.timeframe} nicht gültig! "
                      f"Optionen: {list(TIMEFRAMES.keys())}")
            conn.close()
            return
        timeframes = {args.timeframe: TIMEFRAMES[args.timeframe]}

    try:
        if args.delta:
            run_delta(conn, symbols, timeframes)
        else:
            run_backfill(conn, symbols, timeframes, args.days)

        print_status(conn)

    except Exception as e:
        log.critical(f"Fehler: {e}", exc_info=True)
    finally:
        conn.close()
        log.info("Done.")


if __name__ == "__main__":
    main()
