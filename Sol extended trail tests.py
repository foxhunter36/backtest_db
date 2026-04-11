# Sol extended trail tests

#!/usr/bin/env python3
"""
SOL Supertrend — Erweiterte Trailing Stop Tests
════════════════════════════════════════════════
Test A: Hybrid Exit (Trail + Flip-Only Fallback, kein fixer SL)
Test B: SL nur als Katastrophen-Stop (sehr weit, 5-8×ATR)
Test C: Breakeven-Stop (SL auf Entry wenn +X% erreicht, dann Trail)
Test D: Multi-Coin Trail Validation (9 Coins mit Best-Trail Params)

Läuft auf Trading Station, DB via Tailscale.
"""

import psycopg2
import pandas as pd
import numpy as np
from itertools import product

PG_DSN = "host=100.96.116.110 port=5432 dbname=bybit_backtest user=collector password=bybit2026"

COINS = ["PIXELUSDT", "XRPUSDT", "MBOXUSDT", "ICPUSDT", "SOLUSDT",
         "RENDERUSDT", "DEXEUSDT", "AXSUSDT", "HBARUSDT"]

ST_ATR_PERIOD = 14
ST_MULTIPLIER = 2.0
AO_FAST = 7
AO_SLOW = 34
FEE_PCT = 0.055
FUNDING_LONG_MAX = 0.0001
FUNDING_SHORT_MIN = -0.0001


# ═══════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def calc_atr(df, period=ST_ATR_PERIOD):
    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift(1)).abs(),
                    (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()

def calc_supertrend(df):
    df = df.copy()
    atr = calc_atr(df); df["atr"] = atr
    hl2 = (df["high"]+df["low"])/2.0
    upper = (hl2+ST_MULTIPLIER*atr).values; lower = (hl2-ST_MULTIPLIER*atr).values
    close = df["close"].values; n = len(df)
    fu=np.full(n,np.nan); fl=np.full(n,np.nan); d=np.zeros(n,dtype=int)
    s = ST_ATR_PERIOD
    if s >= n:
        df["st_direction"]=0; df["st_long"]=False; df["st_short"]=False; return df
    fu[s]=upper[s]; fl[s]=lower[s]; d[s]=-1
    for i in range(s+1,n):
        fu[i]=upper[i] if (upper[i]<fu[i-1] or close[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lower[i] if (lower[i]>fl[i-1] or close[i-1]<fl[i-1]) else fl[i-1]
        if d[i-1]==1:
            if close[i]<fl[i]: d[i]=-1
            else: d[i]=1
        else:
            if close[i]>fu[i]: d[i]=1
            else: d[i]=-1
    df["st_direction"]=d
    ds=pd.Series(d,index=df.index)
    df["st_long"]=(ds==1)&(ds.shift(1)==-1); df["st_short"]=(ds==-1)&(ds.shift(1)==1)
    return df

def calc_ao(df):
    df = df.copy()
    mid=(df["high"]+df["low"])/2.0
    df["ao"]=mid.rolling(AO_FAST).mean()-mid.rolling(AO_SLOW).mean()
    return df


# ═══════════════════════════════════════════════════════════════════════
#  SIGNAL CHECK (shared)
# ═══════════════════════════════════════════════════════════════════════

def check_signal(row):
    sig = None
    if row["st_long"]: sig = "long"
    elif row["st_short"]: sig = "short"
    if not sig: return None
    if sig == "long" and row["ao"] <= 0: return None
    if sig == "short" and row["ao"] >= 0: return None
    fr = row.get("funding_rate", np.nan)
    if pd.notna(fr):
        if sig == "long" and fr > FUNDING_LONG_MAX: return None
        if sig == "short" and fr < FUNDING_SHORT_MIN: return None
    if row["atr"] <= 0: return None
    return sig


# ═══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINES
# ═══════════════════════════════════════════════════════════════════════

def backtest_standard_trail(df, sl_mult, trail_act, trail_off):
    """Standard: SL fix + Trail (wie vorheriger Test)."""
    trades = []; pos = None
    for i in range(1, len(df)):
        row = df.iloc[i]; c,h,l = row["close"],row["high"],row["low"]
        if pos:
            pnl_now = (c-pos["entry"])/pos["entry"]*100 if pos["side"]=="long" else (pos["entry"]-c)/pos["entry"]*100
            if not pos["trail_active"] and pnl_now >= trail_act:
                off = row["atr"]*trail_off
                pos["trail_price"] = (c-off) if pos["side"]=="long" else (c+off)
                pos["trail_active"]=True; pos["trail_off_abs"]=off
            if pos["trail_active"]:
                if pos["side"]=="long":
                    nt=h-pos["trail_off_abs"]
                    if nt>pos["trail_price"]: pos["trail_price"]=nt
                else:
                    nt=l+pos["trail_off_abs"]
                    if nt<pos["trail_price"]: pos["trail_price"]=nt
            ex=False
            if pos["side"]=="long":
                if l<=pos["sl"]: trades.append({"pnl":(pos["sl"]-pos["entry"])/pos["entry"]*100,"exit":"SL","bars":i-pos["bar"]}); ex=True
                elif pos["trail_active"] and l<=pos["trail_price"]: trades.append({"pnl":(pos["trail_price"]-pos["entry"])/pos["entry"]*100,"exit":"TRAIL","bars":i-pos["bar"]}); ex=True
                elif row["st_short"]: trades.append({"pnl":(c-pos["entry"])/pos["entry"]*100,"exit":"FLIP","bars":i-pos["bar"]}); ex=True
            else:
                if h>=pos["sl"]: trades.append({"pnl":(pos["entry"]-pos["sl"])/pos["entry"]*100,"exit":"SL","bars":i-pos["bar"]}); ex=True
                elif pos["trail_active"] and h>=pos["trail_price"]: trades.append({"pnl":(pos["entry"]-pos["trail_price"])/pos["entry"]*100,"exit":"TRAIL","bars":i-pos["bar"]}); ex=True
                elif row["st_long"]: trades.append({"pnl":(pos["entry"]-c)/pos["entry"]*100,"exit":"FLIP","bars":i-pos["bar"]}); ex=True
            if ex: pos=None
            else: continue
        if pos: continue
        sig=check_signal(row)
        if not sig: continue
        atr=row["atr"]
        if sig=="long": pos={"side":"long","entry":c,"sl":c-sl_mult*atr,"bar":i,"trail_active":False,"trail_price":0,"trail_off_abs":0}
        else: pos={"side":"short","entry":c,"sl":c+sl_mult*atr,"bar":i,"trail_active":False,"trail_price":0,"trail_off_abs":0}
    return analyze(trades)


def backtest_hybrid(df, trail_act, trail_off, catastrophe_sl_mult=None):
    """
    Hybrid: Kein normaler SL. Trail bei +X%. Vor Trail-Aktivierung nur ST-Flip als Exit.
    Optional: Katastrophen-SL (sehr weit, z.B. 8×ATR) als Sicherheitsnetz.
    """
    trades = []; pos = None
    for i in range(1, len(df)):
        row = df.iloc[i]; c,h,l = row["close"],row["high"],row["low"]
        if pos:
            pnl_now = (c-pos["entry"])/pos["entry"]*100 if pos["side"]=="long" else (pos["entry"]-c)/pos["entry"]*100
            if not pos["trail_active"] and pnl_now >= trail_act:
                off = row["atr"]*trail_off
                pos["trail_price"] = (c-off) if pos["side"]=="long" else (c+off)
                pos["trail_active"]=True; pos["trail_off_abs"]=off
            if pos["trail_active"]:
                if pos["side"]=="long":
                    nt=h-pos["trail_off_abs"]
                    if nt>pos["trail_price"]: pos["trail_price"]=nt
                else:
                    nt=l+pos["trail_off_abs"]
                    if nt<pos["trail_price"]: pos["trail_price"]=nt
            ex=False
            if pos["side"]=="long":
                if catastrophe_sl_mult and l<=pos["cat_sl"]:
                    trades.append({"pnl":(pos["cat_sl"]-pos["entry"])/pos["entry"]*100,"exit":"CAT_SL","bars":i-pos["bar"]}); ex=True
                elif pos["trail_active"] and l<=pos["trail_price"]:
                    trades.append({"pnl":(pos["trail_price"]-pos["entry"])/pos["entry"]*100,"exit":"TRAIL","bars":i-pos["bar"]}); ex=True
                elif row["st_short"]:
                    trades.append({"pnl":(c-pos["entry"])/pos["entry"]*100,"exit":"FLIP","bars":i-pos["bar"]}); ex=True
            else:
                if catastrophe_sl_mult and h>=pos["cat_sl"]:
                    trades.append({"pnl":(pos["entry"]-pos["cat_sl"])/pos["entry"]*100,"exit":"CAT_SL","bars":i-pos["bar"]}); ex=True
                elif pos["trail_active"] and h>=pos["trail_price"]:
                    trades.append({"pnl":(pos["entry"]-pos["trail_price"])/pos["entry"]*100,"exit":"TRAIL","bars":i-pos["bar"]}); ex=True
                elif row["st_long"]:
                    trades.append({"pnl":(pos["entry"]-c)/pos["entry"]*100,"exit":"FLIP","bars":i-pos["bar"]}); ex=True
            if ex: pos=None
            else: continue
        if pos: continue
        sig=check_signal(row)
        if not sig: continue
        atr=row["atr"]
        cat_sl_long = c - catastrophe_sl_mult*atr if catastrophe_sl_mult else 0
        cat_sl_short = c + catastrophe_sl_mult*atr if catastrophe_sl_mult else 999999
        if sig=="long": pos={"side":"long","entry":c,"cat_sl":cat_sl_long,"bar":i,"trail_active":False,"trail_price":0,"trail_off_abs":0}
        else: pos={"side":"short","entry":c,"cat_sl":cat_sl_short,"bar":i,"trail_active":False,"trail_price":0,"trail_off_abs":0}
    return analyze(trades)


def backtest_breakeven(df, sl_mult, be_activation_pct, trail_act, trail_off):
    """
    Breakeven + Trail:
    1. Entry mit SL
    2. Bei +be_activation_pct: SL auf Entry (Breakeven)
    3. Bei +trail_act: Trail aktivieren mit trail_off×ATR
    """
    trades = []; pos = None
    for i in range(1, len(df)):
        row = df.iloc[i]; c,h,l = row["close"],row["high"],row["low"]
        if pos:
            pnl_now = (c-pos["entry"])/pos["entry"]*100 if pos["side"]=="long" else (pos["entry"]-c)/pos["entry"]*100
            # Breakeven
            if not pos["be_active"] and pnl_now >= be_activation_pct:
                pos["sl"] = pos["entry"]  # SL auf Entry
                pos["be_active"] = True
            # Trail
            if not pos["trail_active"] and pnl_now >= trail_act:
                off = row["atr"]*trail_off
                pos["trail_price"] = (c-off) if pos["side"]=="long" else (c+off)
                pos["trail_active"]=True; pos["trail_off_abs"]=off
            if pos["trail_active"]:
                if pos["side"]=="long":
                    nt=h-pos["trail_off_abs"]
                    if nt>pos["trail_price"]: pos["trail_price"]=nt
                else:
                    nt=l+pos["trail_off_abs"]
                    if nt<pos["trail_price"]: pos["trail_price"]=nt
            ex=False
            if pos["side"]=="long":
                if l<=pos["sl"]:
                    label = "BE" if pos["be_active"] and not pos["trail_active"] else "SL"
                    trades.append({"pnl":(pos["sl"]-pos["entry"])/pos["entry"]*100,"exit":label,"bars":i-pos["bar"]}); ex=True
                elif pos["trail_active"] and l<=pos["trail_price"]:
                    trades.append({"pnl":(pos["trail_price"]-pos["entry"])/pos["entry"]*100,"exit":"TRAIL","bars":i-pos["bar"]}); ex=True
                elif row["st_short"]:
                    trades.append({"pnl":(c-pos["entry"])/pos["entry"]*100,"exit":"FLIP","bars":i-pos["bar"]}); ex=True
            else:
                if h>=pos["sl"]:
                    label = "BE" if pos["be_active"] and not pos["trail_active"] else "SL"
                    trades.append({"pnl":(pos["entry"]-pos["sl"])/pos["entry"]*100,"exit":label,"bars":i-pos["bar"]}); ex=True
                elif pos["trail_active"] and h>=pos["trail_price"]:
                    trades.append({"pnl":(pos["entry"]-pos["trail_price"])/pos["entry"]*100,"exit":"TRAIL","bars":i-pos["bar"]}); ex=True
                elif row["st_long"]:
                    trades.append({"pnl":(pos["entry"]-c)/pos["entry"]*100,"exit":"FLIP","bars":i-pos["bar"]}); ex=True
            if ex: pos=None
            else: continue
        if pos: continue
        sig=check_signal(row)
        if not sig: continue
        atr=row["atr"]
        if sig=="long": pos={"side":"long","entry":c,"sl":c-sl_mult*atr,"bar":i,"be_active":False,"trail_active":False,"trail_price":0,"trail_off_abs":0}
        else: pos={"side":"short","entry":c,"sl":c+sl_mult*atr,"bar":i,"be_active":False,"trail_active":False,"trail_price":0,"trail_off_abs":0}
    return analyze(trades)


def analyze(trades):
    if not trades: return None
    pnls = np.array([t["pnl"]-FEE_PCT*2 for t in trades])
    bars = np.array([t["bars"] for t in trades])
    wins=(pnls>0).sum(); total=len(pnls)
    cum=np.cumsum(pnls); max_dd=(cum-np.maximum.accumulate(cum)).min()
    sharpe=pnls.mean()/pnls.std()*np.sqrt(total) if pnls.std()>0 else 0
    gp=pnls[pnls>0].sum(); gl=abs(pnls[pnls<=0].sum())
    pf=gp/gl if gl>0 else float("inf")
    exits={}
    for t in trades: exits[t["exit"]]=exits.get(t["exit"],0)+1
    return {"trades":total,"wins":wins,"winrate":wins/total*100,"total_pnl":pnls.sum(),
            "avg_pnl":pnls.mean(),"max_dd":max_dd,"sharpe":sharpe,"pf":pf,
            "avg_bars":bars.mean(),"exits":exits}


def fmt(r):
    if not r: return "(keine Trades)"
    ex=" ".join(f"{k}:{v}" for k,v in sorted(r["exits"].items()))
    return (f"Tr={r['trades']:>4d} WR={r['winrate']:>4.0f}% PnL={r['total_pnl']:>+7.1f}% "
            f"Sh={r['sharpe']:>5.2f} PF={r['pf']:>5.2f} DD={r['max_dd']:>6.1f}% "
            f"Bars={r['avg_bars']:>5.1f} | {ex}")


def load_coin(conn, symbol):
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM kline_1h "
        "WHERE symbol=%s ORDER BY timestamp", conn, params=(symbol,))
    fr = pd.read_sql_query(
        "SELECT timestamp, funding_rate FROM funding_rate "
        "WHERE symbol=%s ORDER BY timestamp", conn, params=(symbol,))
    if len(fr)>0:
        df = pd.merge_asof(df.sort_values("timestamp"), fr.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    else:
        df["funding_rate"]=np.nan
    df = calc_supertrend(df)
    df = calc_ao(df)
    return df.dropna(subset=["atr","ao"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("="*100)
    print("SOL SUPERTREND — ERWEITERTE EXIT-TESTS")
    print("="*100)

    conn = psycopg2.connect(PG_DSN)
    df_sol = load_coin(conn, "SOLUSDT")
    print(f"SOL Data: {len(df_sol)} bars ({len(df_sol)/24:.0f} Tage)\n")

    # ══════════════════════════════════════════════════════════════
    #  TEST A: HYBRID (kein SL, nur Flip + Trail)
    # ══════════════════════════════════════════════════════════════
    print("="*100)
    print("TEST A: HYBRID EXIT (kein normaler SL, Flip + Trail)")
    print("="*100)

    configs_a = [
        ("Flip Only (kein Trail, kein SL)", None, 999, 0),
        ("Hybrid: Trail +2% / 0.5x", None, 2.0, 0.5),
        ("Hybrid: Trail +3% / 0.5x", None, 3.0, 0.5),
        ("Hybrid: Trail +3% / 1.0x", None, 3.0, 1.0),
        ("Hybrid: Trail +4% / 0.5x", None, 4.0, 0.5),
        ("Hybrid + CatSL 5x: Trail +3% / 0.5x", 5.0, 3.0, 0.5),
        ("Hybrid + CatSL 6x: Trail +3% / 0.5x", 6.0, 3.0, 0.5),
        ("Hybrid + CatSL 8x: Trail +3% / 0.5x", 8.0, 3.0, 0.5),
        ("Standard BEST: SL 3.0 / Trail +3% / 0.5x", "STD", 3.0, 0.5),
    ]

    print(f"\n{'Config':<45s} | Results")
    print("-"*130)
    for name, cat_sl, t_act, t_off in configs_a:
        if cat_sl == "STD":
            r = backtest_standard_trail(df_sol, sl_mult=3.0, trail_act=t_act, trail_off=t_off)
        else:
            r = backtest_hybrid(df_sol, trail_act=t_act, trail_off=t_off,
                               catastrophe_sl_mult=cat_sl)
        print(f"{name:<45s} | {fmt(r)}")

    # ══════════════════════════════════════════════════════════════
    #  TEST B: BREAKEVEN + TRAIL
    # ══════════════════════════════════════════════════════════════
    print("\n"+"="*100)
    print("TEST B: BREAKEVEN + TRAIL")
    print("Logik: SL fix → bei +X% SL auf Entry → bei +Y% Trail aktivieren")
    print("="*100)

    configs_b = [
        # (name, sl, be_act, trail_act, trail_off)
        ("SL3 → BE+1.5% → Trail+3%/0.5x", 3.0, 1.5, 3.0, 0.5),
        ("SL3 → BE+2% → Trail+3%/0.5x", 3.0, 2.0, 3.0, 0.5),
        ("SL3 → BE+2% → Trail+4%/0.5x", 3.0, 2.0, 4.0, 0.5),
        ("SL3 → BE+1% → Trail+2%/0.5x", 3.0, 1.0, 2.0, 0.5),
        ("SL3 → BE+1% → Trail+3%/0.5x", 3.0, 1.0, 3.0, 0.5),
        ("SL2 → BE+1.5% → Trail+3%/0.5x", 2.0, 1.5, 3.0, 0.5),
        ("SL2 → BE+2% → Trail+3%/0.5x", 2.0, 2.0, 3.0, 0.5),
        ("Standard (kein BE): SL3/Trail+3%/0.5x", "NOBR", 0, 3.0, 0.5),
    ]

    print(f"\n{'Config':<45s} | Results")
    print("-"*130)
    for item in configs_b:
        if item[1] == "NOBR":
            name = item[0]
            r = backtest_standard_trail(df_sol, sl_mult=3.0, trail_act=item[3], trail_off=item[4])
        else:
            name, sl, be, ta, to_ = item
            r = backtest_breakeven(df_sol, sl_mult=sl, be_activation_pct=be,
                                   trail_act=ta, trail_off=to_)
        print(f"{name:<45s} | {fmt(r)}")

    # ══════════════════════════════════════════════════════════════
    #  TEST C: WALK-FORWARD für Top Configs
    # ══════════════════════════════════════════════════════════════
    print("\n"+"="*100)
    print("TEST C: WALK-FORWARD (70/30) — Top Configs")
    print("="*100)

    split = int(len(df_sol)*0.7)
    train = df_sol.iloc[:split].reset_index(drop=True)
    test = df_sol.iloc[split:].reset_index(drop=True)
    print(f"Train: {len(train)/24:.0f}d | Test: {len(test)/24:.0f}d\n")

    wf_configs = [
        ("Standard: SL3/Trail+3%/0.5x", lambda d: backtest_standard_trail(d, 3.0, 3.0, 0.5)),
        ("Hybrid: Trail+3%/0.5x (kein SL)", lambda d: backtest_hybrid(d, 3.0, 0.5)),
        ("Hybrid+CatSL6: Trail+3%/0.5x", lambda d: backtest_hybrid(d, 3.0, 0.5, 6.0)),
        ("BE: SL3→BE+1.5%→Trail+3%/0.5x", lambda d: backtest_breakeven(d, 3.0, 1.5, 3.0, 0.5)),
        ("BE: SL3→BE+2%→Trail+3%/0.5x", lambda d: backtest_breakeven(d, 3.0, 2.0, 3.0, 0.5)),
    ]

    print(f"{'Config':<40s} | {'Set':<6s} | {'Sharpe':>7s} {'PnL%':>8s} {'WR%':>5s} {'MaxDD':>7s} {'Trades':>6s}")
    print("-"*90)
    for name, fn in wf_configs:
        for label, data in [("Train", train), ("Test", test)]:
            r = fn(data)
            if r:
                print(f"{name:<40s} | {label:<6s} | {r['sharpe']:>7.2f} {r['total_pnl']:>+7.1f}% "
                      f"{r['winrate']:>4.0f}% {r['max_dd']:>6.1f}% {r['trades']:>6d}")
            else:
                print(f"{name:<40s} | {label:<6s} | (keine Trades)")

    # ══════════════════════════════════════════════════════════════
    #  TEST D: MULTI-COIN TRAIL VALIDATION
    # ══════════════════════════════════════════════════════════════
    print("\n"+"="*100)
    print("TEST D: MULTI-COIN TRAILING STOP VALIDATION (9 Coins)")
    print("Best Trail: SL 3.0 / Trail +3% / 0.5×ATR")
    print("="*100)

    print(f"\n{'Symbol':<16s} | {'Std Trail':<45s} | {'Hybrid+Cat6':<45s}")
    print("-"*115)

    std_sharpes = []
    hyb_sharpes = []

    for sym in COINS:
        df_coin = load_coin(conn, sym)

        r_std = backtest_standard_trail(df_coin, sl_mult=3.0, trail_act=3.0, trail_off=0.5)
        r_hyb = backtest_hybrid(df_coin, trail_act=3.0, trail_off=0.5, catastrophe_sl_mult=6.0)

        std_str = f"Sh={r_std['sharpe']:>5.2f} PnL={r_std['total_pnl']:>+6.1f}% WR={r_std['winrate']:>3.0f}% DD={r_std['max_dd']:>5.1f}%" if r_std else "(none)"
        hyb_str = f"Sh={r_hyb['sharpe']:>5.2f} PnL={r_hyb['total_pnl']:>+6.1f}% WR={r_hyb['winrate']:>3.0f}% DD={r_hyb['max_dd']:>5.1f}%" if r_hyb else "(none)"

        if r_std: std_sharpes.append(r_std["sharpe"])
        if r_hyb: hyb_sharpes.append(r_hyb["sharpe"])

        better = "STD" if (r_std and r_hyb and r_std["sharpe"]>r_hyb["sharpe"]) else "HYB"
        print(f"{sym:<16s} | {std_str:<45s} | {hyb_str:<45s}  {better}")

    print(f"\n  Standard Trail Avg Sharpe:   {np.mean(std_sharpes):.2f}")
    print(f"  Hybrid+CatSL6 Avg Sharpe:   {np.mean(hyb_sharpes):.2f}")
    print(f"  Profitable (Std):  {sum(1 for s in std_sharpes if s>0)}/{len(std_sharpes)}")
    print(f"  Profitable (Hyb):  {sum(1 for s in hyb_sharpes if s>0)}/{len(hyb_sharpes)}")

    conn.close()

    print("\n"+"="*100)
    print("FAZIT")
    print("="*100)


if __name__ == "__main__":
    main()