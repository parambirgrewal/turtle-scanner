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

DISCLAIMER = ("⚠️ Educational signal only. Not investment advice. Not a SEBI-registered Research Analyst. "
              "Trading decisions are solely your responsibility.")

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
        return No
