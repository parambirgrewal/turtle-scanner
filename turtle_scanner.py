#!/usr/bin/env python3
"""
Trinity Trilogy Turtle Breakouts - NSE scanner
------------------------------------------------
* Every ~15 min in market hours (09:20-15:35 IST): live reading on today's partial bar.
  Sends ONLY names not already sent today.
* After the close (15:40+ IST): one confirmed scan on sealed daily bars + market pulse.
* Rules: close above prior 20-day high + Minervini trend template.
         Stop = entry - 2N (N = 20-day ATR), then trails up with the 10-day low.
         Target = entry + 3N (reference only; winners are left to run on the trailing stop).
* Data: Yahoo Finance via yfinance (free, may be delayed). Delivery: Telegram bot.
Educational only. Not investment advice.
"""
import io
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

IST = ZoneInfo("Asia/Kolkata")

# ----------------------------------------------------------------- settings
ENTRY_LEN = 20          # breakout lookback (days)
EXIT_LEN = 10           # trailing-stop lookback (days)
ATR_LEN = 20            # N
STOP_N = 2.0            # stop distance in N
TARGET_N = 3.0          # reference target in N
MIN_BARS = 230          # need ~200 SMA + slope history
MIN_TURNOVER = float(os.getenv("MIN_TURNOVER_CR", "5")) * 1e7   # avg daily value traded, rupees (5 crore)

STATE_FILE = os.getenv("STATE_FILE", "state.json")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
MODE = (os.getenv("MODE", "auto") or "auto").lower()      # auto | intraday | final
DRY_RUN = os.getenv("DRY_RUN", "") == "1"

DISCLAIMER = ("⚠️ DISCLAIMER: This platform is for personal research and educational purposes only. "
              "It does not constitute investment advice, a research report, or a recommendation to buy or sell any security. "
              "The author is not a SEBI-registered Research Analyst.")

FALLBACK_UNIVERSE = """RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL SBIN LT ITC HINDUNILVR KOTAKBANK AXISBANK
BAJFINANCE MARUTI SUNPHARMA TITAN ASIANPAINT ULTRACEMCO NTPC POWERGRID ONGC COALINDIA TATASTEEL JSWSTEEL HCLTECH
WIPRO TECHM M&M ADANIENT ADANIPORTS BAJAJFINSV NESTLEIND HINDALCO GRASIM CIPLA DRREDDY EICHERMOT HEROMOTOCO
BRITANNIA APOLLOHOSP TATACONSUM INDUSINDBK SBILIFE HDFCLIFE BPCL TRENT BEL SHRIRAMFIN BAJAJ-AUTO JIOFIN""".split()


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- universe
def load_universe():
    """universe.txt (one NSE symbol per line) wins; else Nifty 500 from NSE; else a small fallback list."""
    if os.path.exists("universe.txt"):
        syms = [s.strip().upper() for s in open("universe.txt") if s.strip() and not s.startswith("#")]
        if syms:
            log(f"Universe: universe.txt ({len(syms)} symbols)")
            return syms
    urls = ["https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
            "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"]
    for u in urls:
        try:
            r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            r.raise_for_status()
            syms = [s.strip() for s in pd.read_csv(io.StringIO(r.text))["Symbol"].dropna().astype(str)]
            if len(syms) > 100:
                log(f"Universe: Nifty 500 ({len(syms)} symbols)")
                return syms
        except Exception as e:  # noqa: BLE001
            log(f"Universe download failed ({u}): {e}")
    log("Universe: built-in fallback list (add a universe.txt to use your own)")
    return FALLBACK_UNIVERSE


# ----------------------------------------------------------------- data
def clean(df):
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["High", "Low", "Close"]).sort_index()


def pick(raw, ticker):
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.get_level_values(0):
            return None
        df = raw[ticker]
    else:
        df = raw
    if df is None or df.dropna(how="all").empty:
        return None
    return clean(df.dropna(how="all"))


def download(tickers, period="2y"):
    import yfinance as yf
    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                              auto_adjust=True, threads=True, progress=False)
            if raw is not None and not raw.empty:
                return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(3 * (attempt + 1))
    log(f"Download failed for batch starting {tickers[0]}: {last_err}")
    return None


def fetch_history(symbols):
    out = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        tickers = [s + ".NS" for s in batch]
        raw = download(tickers)
        if raw is None:
            continue
        for s, t in zip(batch, tickers):
            try:
                df = pick(raw, t)
            except Exception:  # noqa: BLE001
                df = None
            if df is not None and len(df):
                out[s] = df
        log(f"Fetched {len(out)} / {min(i + 100, len(symbols))} symbols")
    return out


# ----------------------------------------------------------------- indicators
def wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def prep(df):
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    out = pd.DataFrame(index=df.index)
    out["N"] = wilder(tr, ATR_LEN)

    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr14 = wilder(tr, 14)
    pdi = 100 * wilder(plus_dm, 14) / atr14
    mdi = 100 * wilder(minus_dm, 14) / atr14
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi)).replace([np.inf, -np.inf], np.nan)
    out["adx"] = wilder(dx, 14)

    out["sma50"] = c.rolling(50).mean()
    out["sma150"] = c.rolling(150).mean()
    out["sma200"] = c.rolling(200).mean()
    out["sma200_lag"] = out["sma200"].shift(22)
    out["hi52"] = h.rolling(252, min_periods=200).max()
    out["lo52"] = l.rolling(252, min_periods=200).min()
    out["vol_avg"] = v.rolling(20).mean().shift(1)
    out["level"] = h.shift(1).rolling(ENTRY_LEN).max()       # prior 20 highs (excludes this bar)
    out["exit_low"] = l.shift(1).rolling(EXIT_LEN).min()     # prior 10 lows (excludes this bar)
    return out


def trend_mask(c, r):
    """Minervini trend template, vectorised."""
    return ((c > r["sma150"]) & (c > r["sma200"]) & (r["sma150"] > r["sma200"]) &
            (r["sma200"] > r["sma200_lag"]) & (r["sma50"] > r["sma150"]) & (r["sma50"] > r["sma200"]) &
            (c > r["sma50"]) & (c >= 1.30 * r["lo52"]) & (c >= 0.75 * r["hi52"]))


def template(px, r):
    """Minervini trend template for a single (live) price against the last sealed row."""
    try:
        return bool(px > r.sma150 and px > r.sma200 and r.sma150 > r.sma200 and r.sma200 > r.sma200_lag and
                    r.sma50 > r.sma150 and r.sma50 > r.sma200 and px > r.sma50 and
                    px >= 1.30 * r.lo52 and px >= 0.75 * r.hi52)
    except Exception:  # noqa: BLE001
        return False


def simulate(df, ind):
    """Walk the sealed bars and return (open position or None, breakout-signal array)."""
    brk = ((df["Close"] > ind["level"]) & trend_mask(df["Close"], ind)).values
    low = df["Low"].values
    close = df["Close"].values
    n = ind["N"].values
    xl = ind["exit_low"].values
    pos = None
    for i in range(len(df)):
        if pos is not None:
            if not np.isnan(xl[i]):
                pos["stop"] = max(pos["stop"], xl[i])          # trail up with the 10-day low
            if low[i] <= pos["stop"]:
                pos = None                                      # stopped out; no same-bar re-entry
            continue
        if brk[i] and not np.isnan(n[i]):
            pos = {"date": df.index[i], "entry": float(close[i]), "stop": float(close[i] - STOP_N * n[i])}
    return pos, brk


def tier_of(vol, adx, near_high):
    pts = 0
    if vol >= 1.5:
        pts += 2
    elif vol >= 1.2:
        pts += 1
    if adx >= 30:
        pts += 2
    elif adx >= 25:
        pts += 1
    if near_high:
        pts += 1
    return "HIGH" if pts >= 4 else "MEDIUM" if pts >= 2 else "LOW"


def make_signal(sym, px, n, vol, adx, near_high, pos, is_new):
    sig = {"sym": sym, "entry": px, "stop": px - STOP_N * n, "target": px + TARGET_N * n,
           "vol": vol, "adx": adx, "tier": tier_of(vol, adx, near_high), "open": None}
    if pos is not None and not is_new:
        sig["open"] = {"date": pos["date"], "entry": pos["entry"], "stop": pos["stop"]}
    return sig


# ----------------------------------------------------------------- scan
def scan(hist, today, mode, frac=1.0):
    """mode 'final': last bar is sealed. mode 'intraday': last bar (dated today) is partial."""
    signals, scanned = [], 0
    br = {"n": 0, "a200": 0, "a50": 0, "hi": 0, "lo": 0}
    for sym, df in hist.items():
        if mode == "intraday":
            if df.index[-1] != today:
                continue
            live, base = df.iloc[-1], df.iloc[:-1]
        else:
            live, base = None, df
        if len(base) < MIN_BARS:
            continue
        if not (base["Close"] * base["Volume"]).tail(20).mean() >= MIN_TURNOVER:
            continue
        scanned += 1
        ind = prep(base)
        pos, brk = simulate(base, ind)
        last = ind.iloc[-1]
        if pd.isna(last["N"]) or pd.isna(last["sma200_lag"]) or pd.isna(last["hi52"]):
            continue

        if mode == "final":
            c = float(base["Close"].iloc[-1])
            br["n"] += 1
            br["a200"] += c > last["sma200"]
            br["a50"] += c > last["sma50"]
            br["hi"] += float(base["High"].iloc[-1]) >= last["hi52"]
            br["lo"] += float(base["Low"].iloc[-1]) <= last["lo52"]
            if brk[-1] and pos is not None:
                va = last["vol_avg"]
                vol = float(base["Volume"].iloc[-1]) / va if va and va > 0 else np.nan
                is_new = pos["date"] == base.index[-1]
                signals.append(make_signal(sym, c, float(last["N"]), vol, float(last["adx"]),
                                           c >= 0.95 * last["hi52"], pos, is_new))
        else:
            px = float(live["Close"])
            level = float(base["High"].iloc[-ENTRY_LEN:].max())
            if px <= level or not template(px, last):
                continue
            if pos is not None:
                stop_now = max(pos["stop"], float(base["Low"].iloc[-EXIT_LEN:].min()))
                if float(live["Low"]) <= stop_now:
                    continue                                    # already stopped out today
                pos["stop"] = stop_now
            va = float(base["Volume"].iloc[-20:].mean())
            vol = float(live["Volume"]) / max(frac, 0.05) / va if va > 0 else np.nan
            signals.append(make_signal(sym, px, float(last["N"]), vol, float(last["adx"]),
                                       px >= 0.95 * last["hi52"], pos, pos is None))

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    signals.sort(key=lambda s: (s["open"] is not None, order[s["tier"]], -(s["vol"] if s["vol"] == s["vol"] else 0)))
    return signals, scanned, br


# ----------------------------------------------------------------- message
def rupee(x):
    return f"₹{x:,.2f}"


def fnum(x, d=1):
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def format_signals(signals, mode, now, scanned, total):
    stamp = now.strftime("%d %b %Y, %H:%M IST")
    if mode == "final":
        head = (f"🇮🇳 IN  ✅ CONFIRMED BREAKOUTS — {stamp}\n"
                "Post-close scan on sealed daily bars. These closed above their breakout level.")
    else:
        head = (f"🇮🇳 IN  ⏱ INTRADAY BREAKOUTS — {stamp}\n"
                "Live reading on today's partial bar. Not confirmed until the close; volume is pace-adjusted.")
    pct = round(100 * scanned / total) if total else 0
    lines = [head, f"Scanned {scanned} of {total} names ({pct}%).", ""]
    for s in signals:
        lines.append(f"• {s['sym']} — Trinity Trilogy Turtle Breakouts · {s['tier']}")
        lines.append(f"   Entry {rupee(s['entry'])} · Stop {rupee(s['stop'])} · Target {rupee(s['target'])}")
        lines.append(f"   {fnum(s['vol'])}x vol · ADX {fnum(s['adx'])}")
        if s["open"]:
            o = s["open"]
            lines.append(f"   ⚠️ ALREADY OPEN since {o['date']:%d %b} @ {rupee(o['entry'])} — not a new entry")
            lines.append(f"      Stop on THAT position: {rupee(o['stop'])} (trailed) — the line above is for a new entry today")
    n_open = sum(1 for s in signals if s["open"])
    if signals:
        tail = f"{len(signals)} name(s) — {len(signals) - n_open} new, {n_open} already open."
    else:
        tail = "No breakouts."
    if mode == "final":
        tail += " Earlier alerts today were intraday readings; this list is the one taken on final prices."
    else:
        tail += " The confirmed list follows after the close."
    lines += ["", tail]
    return "\n".join(lines)


def _pct(a, b):
    return (a / b - 1) * 100 if b else float("nan")


def market_pulse(br, scanned):
    """Index levels + breadth. Display-only. Returns '' on any failure."""
    try:
        raw = download(["^NSEI", "^NSEBANK", "^INDIAVIX"])
        if raw is None:
            return ""
        rows = []
        for t, name in (("^NSEI", "Nifty 50"), ("^NSEBANK", "Bank Nifty")):
            d = pick(raw, t)
            if d is None or len(d) < 2:
                continue
            c = d["Close"]
            chg = _pct(c.iloc[-1], c.iloc[-2])
            arrow = "▲" if chg >= 0 else "▼"
            extra = ""
            if len(c) >= 200:
                m200 = c.rolling(200).mean().iloc[-1]
                extra = f" · {_pct(c.iloc[-1], m200):+.1f}% vs 200-DMA ({m200:,.0f})"
            rows.append(f"  {name:<11} {c.iloc[-1]:,.1f} {arrow}{abs(chg):.2f}%{extra}")
        v = pick(raw, "^INDIAVIX")
        if v is not None and len(v) >= 2:
            chg = _pct(v["Close"].iloc[-1], v["Close"].iloc[-2])
            rows.append(f"  India VIX   {v['Close'].iloc[-1]:.2f} {'▲' if chg >= 0 else '▼'}{abs(chg):.2f}%")
        if br["n"]:
            rows.append(f"  Breadth: {100 * br['a200'] / br['n']:.0f}% above 200-DMA · {100 * br['a50'] / br['n']:.0f}% above 50-DMA · "
                        f"52-wk highs {br['hi']} / lows {br['lo']} ({br['n']} priced)")
        if not rows:
            return ""
        return "🌡 Market Pulse\n" + "\n".join(rows) + "\n  Display-only. Not a tested signal."
    except Exception as e:  # noqa: BLE001
        log(f"Market pulse skipped: {e}")
        return ""


# ----------------------------------------------------------------- telegram + state
def send(text):
    if DRY_RUN or not BOT_TOKEN or not CHAT_ID:
        log("---- (not sent: DRY_RUN or Telegram not configured) ----")
        log(text)
        return DRY_RUN
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    for ch in chunks:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          json={"chat_id": CHAT_ID, "text": ch, "disable_web_page_preview": True}, timeout=30)
        if not r.ok:
            log(f"Telegram error {r.status_code}: {r.text}")
            return False
    return True


def load_state(day):
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:  # noqa: BLE001
        s = {}
    if s.get("date") != str(day):
        s = {"date": str(day), "seen": [], "final_sent": False}
    return s


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)


# ----------------------------------------------------------------- main
def main():
    now = datetime.now(IST)
    today = pd.Timestamp(now.date())
    minutes = now.hour * 60 + now.minute
    forced = MODE in ("intraday", "final")

    if MODE == "auto":
        if now.weekday() >= 5:
            log("Weekend - nothing to do.")
            return
        if 9 * 60 + 20 <= minutes < 15 * 60 + 35:
            mode = "intraday"
        elif minutes >= 15 * 60 + 40:
            mode = "final"
        else:
            log("Outside market window - nothing to do.")
            return
    else:
        mode = MODE

    state = load_state(today.date())
    if mode == "final" and state["final_sent"] and not forced:
        log("Final scan already sent today.")
        save_state(state)
        return

    universe = load_universe()
    hist = fetch_history(universe)
    if not hist:
        log("No data downloaded - aborting.")
        save_state(state)
        return

    latest = max(df.index[-1] for df in hist.values())
    if latest < today and not forced:
        log(f"No bar for {today.date()} (market holiday or data not ready). Latest bar: {latest.date()}")
        save_state(state)
        return

    frac = min(max((minutes - (9 * 60 + 15)) / 375.0, 0.05), 1.0)
    signals, scanned, br = scan(hist, today, mode, frac)
    log(f"Mode {mode}: scanned {scanned}, signals {len(signals)}")

    if mode == "intraday":
        fresh = [s for s in signals if s["sym"] not in state["seen"]]
        if fresh:
            msg = format_signals(fresh, mode, now, scanned, len(universe))
            msg += "\n\n" + DISCLAIMER
            ok = send(msg)
            if ok and not forced:
                state["seen"] += [s["sym"] for s in fresh]
        else:
            log("No new intraday names - staying quiet.")
    else:
        msg = format_signals(signals, mode, now, scanned, len(universe))
        pulse = market_pulse(br, scanned)
        if pulse:
            msg += "\n\n" + pulse
        msg += "\n\n" + DISCLAIMER
        ok = send(msg)
        if ok and not forced:
            state["final_sent"] = True
    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log(f"Scanner error: {e!r}")
        sys.exit(1)
