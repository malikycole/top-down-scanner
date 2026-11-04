#!/usr/bin/env python3
"""Top down strength scanner: strongest index > strongest sector > strongest stocks.

Scans every U.S. common stock with market cap >= $1B and price >= $10 (about 2,400
names), scores each one against its own sector ETF, and writes site/data.json for
the dashboard in site/index.html.

Strength = blended return: 40% of 1 month + 35% of 3 months + 25% of 6 months.
Same math as the Thinkorswim column, scan and chart dashboard.

Educational research tool only. Not investment advice.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from true_momentum import true_momentum

ROOT = Path(__file__).parent
SITE = ROOT / "site"
UNIVERSE_FILE = ROOT / "universe.csv"

MIN_CAP = 1_000_000_000
MIN_PRICE = 10.0
MIN_AVG_VOLUME = 1_000_000
MIN_CORRELATION = 0.50
MAX_OFF_HIGH = 25.0
# The Std Dev Channel tab uses the wider thinkorswim CustStdDevScan filters, so the universe
# is downloaded at these looser limits; everything else only uses the core (MIN_CAP, MIN_PRICE).
WIDE_MIN_CAP = 350_000_000
WIDE_MIN_PRICE = 1.25
MIN_SETUP_TO_LIST = 5  # stocks below this many checks are left out of the JSON
OPTIONS_CACHE = ROOT / "options_cache.json"
MAX_OI_FETCH = 400     # open interest lookups per run; the rest wait for the next run (cache)
MIN_OPT_DTE = 20      # option liquidity is read at the first expiration this many days out
MIN_OPT_OI = 500       # open interest at that expiration for "liquid options"
CHANNEL_BARS = 251    # std dev channel length: a 1 year daily chart, as in thinkorswim
EMA21_ZONE = 0.01      # current price within 1% of the 21 EMA counts as at the 21 EMA
MIN_PULLBACK = 0.04    # or a pattern low at least 4% under the prior 10 bar high

INDEXES = ["SPY", "QQQ", "DIA", "IWM"]
SECTORS = {
    "XLK": "Technology", "XLC": "Communication", "XLY": "Discretionary",
    "XLF": "Financials", "XLI": "Industrials", "XLV": "Health Care",
    "XLE": "Energy", "XLB": "Materials", "XLP": "Staples",
    "XLU": "Utilities", "XLRE": "Real Estate",
}
GICS_TO_ETF = {
    "Information Technology": "XLK", "Communication Services": "XLC",
    "Consumer Discretionary": "XLY", "Financials": "XLF", "Industrials": "XLI",
    "Health Care": "XLV", "Energy": "XLE", "Materials": "XLB",
    "Consumer Staples": "XLP", "Utilities": "XLU", "Real Estate": "XLRE",
}
NASDAQ_TO_ETF = {
    "Technology": "XLK", "Telecommunications": "XLC", "Consumer Discretionary": "XLY",
    "Finance": "XLF", "Industrials": "XLI", "Health Care": "XLV", "Energy": "XLE",
    "Basic Materials": "XLB", "Consumer Staples": "XLP", "Utilities": "XLU",
    "Real Estate": "XLRE",
}
CHECKS = ["Stronger than sector", "Beats sector 1M", "Beats sector 3M", "RS line rising",
          "Trend stack", "Moves with sector", "Near 52w high", "Liquid"]
# Whole word patterns, so names like UnitedHealth, Brighthouse or Fundamental are not caught.
# The screener lists stocks only; this drops the non stock securities mixed into it.
_NON_EQUITY = re.compile(
    r"\bwarrants?\b|\brights?\b|\bunits?\b|\betfs?\b|\bfunds?\b|\bpreferred\b|\bnotes?\b|\bdebentures?\b"
    r"|\bzones\b|contingent value|%|\bsubordinated\b|\bbaby bonds?\b")
_ADR = re.compile(r"american deposit[ao]ry|\bads\b|\badss\b|\badr\b|registry shares")
_PREF_DEPOSITARY = re.compile(r"deposit[ao]ry shares?")


# ---------------------------------------------------------------- universe

def _num(value) -> float:
    try:
        return float(str(value or "0").replace(",", "").replace("$", ""))
    except ValueError:
        return 0.0


def _get(url: str) -> str:
    from curl_cffi import requests
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, impersonate="chrome", timeout=30)
    response.raise_for_status()
    return response.text


def build_universe() -> pd.DataFrame:
    """Nasdaq screener (all U.S. listings) filtered to common stock, cap and price,
    with S&P 500 GICS sectors where known and Nasdaq sectors for everything else."""
    rows = json.loads(_get("https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"))["data"]["rows"]
    sp = pd.read_html(StringIO(_get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")))[0]
    gics = {s.replace(".", "-"): GICS_TO_ETF.get(g, "") for s, g in zip(sp["Symbol"], sp["GICS Sector"])}
    out = []
    for row in rows:
        symbol, name = row.get("symbol", ""), row.get("name", "").lower()
        if not symbol or "^" in symbol:
            continue
        # Foreign ADRs count as stocks ("right to receive" in their names is not a stock right).
        # Other depositary shares are U.S. preferreds, skipped with warrants, units, notes and funds.
        adr = bool(_ADR.search(name))
        if adr:
            if re.search(r"\bwarrants?\b|\bpreferred\b|\bnotes?\b|\bunits?\b", name):
                continue
        elif _NON_EQUITY.search(name) or _PREF_DEPOSITARY.search(name):
            continue
        cap, price = _num(row.get("marketCap")), _num(row.get("lastsale"))
        if cap < WIDE_MIN_CAP or price < WIDE_MIN_PRICE:
            continue
        ticker = symbol.replace(".", "-").replace("/", "-")
        etf = gics.get(ticker) or NASDAQ_TO_ETF.get(row.get("sector", ""), "")
        clean = row.get("name", "")
        for junk in (" Common Stock", " Ordinary Shares", " Common Shares", " Class A", " Class B", " Class C",
                     " American Depositary", " American depositary", " Sponsored ADR", " ADS"):
            clean = clean.split(junk)[0]
        out.append({"ticker": ticker, "name": clean.strip()[:60], "etf": etf, "cap": cap,
                    "sp500": ticker in gics, "core": cap >= MIN_CAP and price >= MIN_PRICE})
    return pd.DataFrame(out).drop_duplicates("ticker").sort_values("cap", ascending=False)


def load_universe() -> pd.DataFrame:
    """Fresh universe when the feeds answer; the saved copy when they don't."""
    try:
        universe = build_universe()
        if len(universe) > 1000:
            universe.to_csv(UNIVERSE_FILE, index=False)
            return universe
    except Exception as exc:  # noqa: BLE001  feeds can block cloud runners
        print(f"universe refresh failed, using saved copy: {exc}")
    saved = pd.read_csv(UNIVERSE_FILE, keep_default_na=False)
    if "core" not in saved:
        saved["core"] = True
    return saved


# ---------------------------------------------------------------- data

def download(symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for start in range(0, len(symbols), 300):
        batch = symbols[start:start + 300]
        for attempt in range(3):
            try:
                # 2 years so the yearly Tr3ndy edge always has all of last year
                raw = yf.download(batch, period="2y", interval="1d", auto_adjust=True,
                                  group_by="ticker", threads=True, progress=False)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"batch {start} attempt {attempt + 1} failed: {exc}")
                time.sleep(5)
        else:
            continue
        for ticker in batch:
            try:
                frame = raw[ticker].dropna(subset=["Close", "High", "Volume"])
            except KeyError:
                continue
            if len(frame) >= 200:
                frames[ticker] = frame
    return frames


def download_weekly(symbols: list[str], interval: str = "1wk", period: str = "10y") -> dict[str, pd.Series]:
    """Long weekly (or monthly) closes, the live bar included, for the True Momentum Trend Oscillator,
    whose 120 week and 21 month EMAs need a long history to settle."""
    out: dict[str, pd.Series] = {}
    for start in range(0, len(symbols), 400):
        batch = symbols[start:start + 400]
        try:
            raw = yf.download(batch, period=period, interval=interval, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001
            print(f"weekly batch {start} failed: {exc}")
            continue
        for t in batch:
            try:
                wc = raw[t]["Close"].dropna()
            except KeyError:
                continue
            if len(wc) >= 30:
                out[t] = wc
    return out


# ---------------------------------------------------------------- math

def blend(c: np.ndarray) -> float:
    return 0.40 * (c[-1] / c[-22] - 1) + 0.35 * (c[-1] / c[-64] - 1) + 0.25 * (c[-1] / c[-127] - 1)


def ema(c: np.ndarray, span: int) -> float:
    return float(pd.Series(c).ewm(span=span, adjust=False).mean().iloc[-1])


def trend_up(c: np.ndarray) -> bool:
    e21 = ema(c, 21)
    return bool(c[-1] > e21 > c[-50:].mean())


def pullback_triggers(frame: pd.DataFrame) -> dict:
    """Entry triggers on the latest bar (age 0) or the bar before it (age 1).

    21 EMA: the current price is within EMA21_ZONE of the 21 EMA, above or below it.
    Minor low: a bar whose low is under the lows on both sides, then a green bar that
        closes above that low bar's high.
    Inside and up: an inside bar (high and low inside the bar before), then a bar that
        closes above the inside bar's high, with the current price within EMA21_ZONE
        of the 21 EMA.
    Minor low only counts after a real pullback: the pattern low is within EMA21_ZONE
        of the 21 EMA, or at least MIN_PULLBACK below the 10 bar high before it.
    """
    o, h, l, c = (frame[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    e21 = pd.Series(c).ewm(span=21, adjust=False).mean().to_numpy()
    touch = abs(c[-1] / e21[-1] - 1) <= EMA21_ZONE

    def pulled_back(i: int) -> bool:
        return l[i] <= e21[i] * (1 + EMA21_ZONE) or l[i] <= h[i - 10:i].max() * (1 - MIN_PULLBACK)

    patterns = []
    for age in (0, 1):
        up = -1 - age              # the confirming bar
        mid, before = up - 1, up - 2
        if l[mid] < l[before] and l[up] > l[mid] and c[up] > o[up] and c[up] > h[mid] and pulled_back(mid):
            patterns.append(("Minor low", age, l[mid]))
        if h[mid] <= h[before] and l[mid] >= l[before] and c[up] > h[mid] and touch:
            patterns.append(("Inside and up", age, l[mid]))
    triggers = (["21 EMA"] if touch else []) + sorted({p[0] for p in patterns})
    return {
        "trig": triggers,
        "age": min((p[1] for p in patterns), default=0),
        "e21d": round((c[-1] / e21[-1] - 1) * 100, 2),
        "pb": round((1 - min(l[-3:]) / h[-13:-1].max()) * 100, 1),
        "stop": round(float(min((p[2] for p in patterns), default=min(l[-2:]))), 2),
    }


def jt_trends(frame: pd.DataFrame) -> dict:
    """Port of Jeff Thaw's JT_Trends thinkorswim study.
    MOMO: 8 EMA > 13 EMA > 21 EMA bullish, 8 < 13 < 21 bearish, else slop.
    Trend: MOMO bullish and 21 EMA > 34 EMA of highs = bullish; MOMO bearish and
    21 EMA < 34 EMA of lows = bearish; else sideways. Double green = Trend bullish."""
    c = frame["Close"]
    e8, e13, e21 = (float(c.ewm(span=n, adjust=False).mean().iloc[-1]) for n in (8, 13, 21))
    e34h = float(frame["High"].ewm(span=34, adjust=False).mean().iloc[-1])
    e34l = float(frame["Low"].ewm(span=34, adjust=False).mean().iloc[-1])
    momo = 1 if e8 > e13 > e21 else -1 if e8 < e13 < e21 else 0
    wave = 1 if momo == 1 and e21 > e34h else -1 if momo == -1 and e21 < e34l else 0
    return {"momo": momo, "wave": wave}


def big3(frame: pd.DataFrame) -> dict:
    """Port of Simpler Trading's Big 3 Squeeze studies (ST_Big3_Squeeze, ST_Big3SqueezeLabels,
    ST_Big3_APlus_Signals, ST_Big3_Slingshot_Signals, ATR trailing stop) on daily bars."""
    o, h, l, c = (frame[k] for k in ("Open", "High", "Low", "Close"))
    prev_c = c.shift()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    # Squeeze: Bollinger (20, 2.0, population stdev) upper band inside Keltner (20, 1.75 x SMA of TR)
    sma20 = c.rolling(20).mean()
    sqz = (sma20 + 2.0 * c.rolling(20).std(ddof=0)) < (sma20 + 1.75 * tr.rolling(20).mean())

    ema_ = lambda n: c.ewm(span=n, adjust=False).mean()
    macd = ema_(24) - ema_(52)
    diff = macd - macd.ewm(span=9, adjust=False).mean()          # MACDHistogram(24, 52, 9) = MD = Diff
    e5, e8, e21, e34 = ema_(5), ema_(8), ema_(21), ema_(34)
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    sp = (e5 > e8) & (e8 > e21) & (e21 > e34) & (e21 > s50)
    sn = (e5 < e8) & (e8 < e21) & (e21 < e34) & (e21 < s50)

    # Modified ATR trailing stop (period 9, factor 2.4, exponential, first trade long)
    hl = h - l
    hilo = np.minimum(hl, 1.5 * hl.rolling(9).mean())
    href = np.where(l <= h.shift(), h - prev_c, (h - prev_c) - 0.5 * (l - h.shift()))
    lref = np.where(h >= l.shift(), prev_c - l, (prev_c - l) - 0.5 * (l.shift() - h))
    mtr = pd.concat([hilo, pd.Series(href, index=c.index), pd.Series(lref, index=c.index)], axis=1).max(axis=1, skipna=False)
    loss = (2.4 * mtr.ewm(span=9, adjust=False, ignore_na=True).mean()).to_numpy()
    cv = c.to_numpy()
    trail, state = np.full(len(cv), np.nan), 0                     # 0 init, 1 long, -1 short
    for i in range(len(cv)):
        if state == 0:
            if not np.isnan(loss[i]):
                state, trail[i] = 1, cv[i] - loss[i]
        elif state == 1:
            if cv[i] > trail[i - 1]:
                trail[i] = max(trail[i - 1], cv[i] - loss[i])
            else:
                state, trail[i] = -1, cv[i] + loss[i]
        else:
            if cv[i] < trail[i - 1]:
                trail[i] = min(trail[i - 1], cv[i] + loss[i])
            else:
                state, trail[i] = 1, cv[i] - loss[i]

    x = -1                      # latest bar
    C, D, D1 = cv[x], float(diff.iloc[x]), float(diff.iloc[x - 1])
    E8, E21, S50, S200, T = float(e8.iloc[x]), float(e21.iloc[x]), float(s50.iloc[x]), float(s200.iloc[x]), float(trail[x])
    up50, up21, up200 = s50.iloc[x] > s50.iloc[x - 1], e21.iloc[x] > e21.iloc[x - 1], s200.iloc[x] > s200.iloc[x - 1]
    dn50, dn21, dn200 = s50.iloc[x] < s50.iloc[x - 1], e21.iloc[x] < e21.iloc[x - 1], s200.iloc[x] < s200.iloc[x - 1]
    SP, SN, SQ = bool(sp.iloc[x]), bool(sn.iloc[x]), bool(sqz.iloc[x])
    bull = [D > 0, up50, up21, up200, SP, C > S50, C > E21, C > S200, E21 > S50, S50 > S200, E8 > E21, C > T]
    bear = [D < 0, SN, C < S50, C < E21, C < S200, E21 < S50, S50 < S200, dn50, dn21, dn200, E8 < E21, C < T]
    bull_score, bear_score = int(sum(bull)), int(sum(bear))
    score = bull_score - bear_score
    bull_ts, bear_ts = up50 and up21 and up200, dn50 and dn21 and dn200

    signals = []
    if SP and S50 > S200 and D > 0 and C > S50 and SQ and C > T and score > 6 and bull_ts:
        signals.append("A+ Bull")
    if SQ and bull_score == 12:
        signals.append("+12 Bull")
    if SP and D > 0 and C > S50 and SQ:
        signals.append("Bull squeeze")
    if SQ and SP and D < 0 and D > D1 and C > S50:
        signals.append("Slingshot bull")
    if SN and S50 < S200 and D < 0 and C < S50 and SQ and C < T and score < -6 and bear_ts:
        signals.append("A+ Bear")
    if SN and D < 0 and C < S50 and SQ:
        signals.append("Bear squeeze")
    if SQ and SN and D > 0 and D < D1 and C < S50:
        signals.append("Slingshot bear")

    # Labels study: TREND, STRUCTURE, MOMENTUM, TRAILING STOP as 1 bull, -1 bear, 0 neutral
    trend = 1 if bull_ts and E21 > S50 > S200 else -1 if bear_ts and E21 < S50 < S200 else 0
    structure = 1 if SP and E21 > S50 > S200 else -1 if SN and E21 < S50 < S200 else 0
    momentum = 1 if D > 0 else -1 if D < 0 else 0
    trail_lab = 1 if C > T else -1 if C < T else 0

    atr_sma = float(tr.rolling(14).mean().iloc[x])                 # targets use Average(TrueRange, 14)
    atr_ema = float(tr.ewm(span=14, adjust=False).mean().iloc[x])  # the ATR label uses an exponential 14
    atr_wilder = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[x])
    sq = sqz.to_numpy()
    bars = 0
    while bars < len(sq) and sq[len(sq) - 1 - bars]:
        bars += 1
    return {
        "sqz": SQ, "sqzn": bars, "bull": bull_score, "bear": bear_score, "score": score, "sig": signals,
        "lab": [trend, structure, momentum, trail_lab],
        "hist": round(D, 4), "hist_up": bool(D > D1),
        "e21": round(E21, 2), "atrs21": round((C - E21) / atr_wilder, 2) if atr_wilder else 0.0,
        "trail": round(T, 2), "tgt_up": round(E21 + 3 * atr_sma, 2), "tgt_dn": round(E21 - 3 * atr_sma, 2),
        "atr": round(atr_ema, 2), "buyz": bool(C <= E21 + 1.25 * atr_sma), "sellz": bool(C >= E21 - 1.25 * atr_sma),
        "vol1": int(frame["Volume"].iloc[x]),
    }


def std_channel(frame: pd.DataFrame) -> dict:
    """Port of the thinkorswim StandardDeviationChannel (fullRange = yes) on a 1 year daily
    chart: regression line over the last CHANNEL_BARS closes (InertiaAll) plus or minus
    the population standard deviation of those closes (StDevAll)."""
    c = frame["Close"].to_numpy()[-CHANNEL_BARS:]
    x = np.arange(len(c))
    slope, intercept = np.polyfit(x, c, 1)
    mean = intercept + slope * x[-1]
    sd = float(np.std(c))
    if sd == 0:
        return {"dev": 0.0}
    dev = (c[-1] - mean) / sd
    angle = slope * 20 / sd    # playbook slope: deviations per 20 bars
    regime = ("Steep up" if angle >= 0.22 else "Mild up" if angle >= 0.15 else
              "Steep down" if angle <= -0.22 else "Mild down" if angle <= -0.15 else "Flat")
    o = frame["Open"].to_numpy()
    return {
        "dev": round(float(dev), 2), "dev1": round(float((c[-2] - (intercept + slope * x[-2])) / sd), 2),
        "mean": round(float(mean), 2), "sd": round(sd, 2), "angle": round(float(angle), 3), "regime": regime,
        "b25": round(float(mean - 2.5 * sd), 2), "b3": round(float(mean - 3 * sd), 2), "b4": round(float(mean - 4 * sd), 2),
        "u25": round(float(mean + 2.5 * sd), 2), "u4": round(float(mean + 4 * sd), 2),
        "green": bool(c[-1] > o[-1]), "to_mean": round(float((mean / c[-1] - 1) * 100), 1),
    }


EDGE_TOL = {"D": 0.0, "W": 0.01, "M": 0.015, "Y": 0.02}   # how close to the lower edge counts as "at" it


def _edge_levels(h: float, l: float) -> dict:
    """Tr3ndy Edges levels off one prior bar (range r = h - l)."""
    r = h - l
    return {"le": h - r * 0.887, "ue": l + r * 0.887, "mid": l + r * 0.5,
            "sbu": h - r * 0.236, "bu": h - r * 0.382, "be": l + r * 0.382, "sbe": l + r * 0.236}


def _edge_label(c: float, lv: dict) -> int:
    """Tr3ndy banner: 3 extreme bullish, 2 super bullish, 1 bullish, 0 neutral, -1 bearish,
    -2 super bearish, -3 extreme bearish."""
    if c >= lv["ue"]:
        return 3
    if c >= lv["sbu"]:
        return 2
    if c >= lv["bu"]:
        return 1
    if c <= lv["le"]:
        return -3
    if c <= lv["sbe"]:
        return -2
    if c <= lv["be"]:
        return -1
    return 0


def tr3ndy_edges(frame: pd.DataFrame) -> dict:
    """Port of the Tr3ndy Edges study (@beezy, trendytrading.co) with Malik's edge rules, long side.
    Levels come from the prior daily, weekly, monthly and yearly (blue) bar: lower edge 11.3% of
    the range, decelerator 38.2%, mid 50%, accelerator 61.8%, upper edge 88.7%. Setups, confirmed on
    this daily bar or the one before:
      Edge reversal     a minor low or inside and up whose low reached the lower edge; target the
                        upper edge, stop the pattern low
      Edge rejection    price traded under the lower edge and closed back above it; target the
                        decelerator, stop under the wick low
      Accelerator hold  banner bullish (not extreme), price pulled back to the accelerator and closed
                        above it; target the upper edge, stop a close under the mid
      Breakout retest   the prior high was broken earlier this period, price came back to it and held;
                        stop back under the upper edge, target trail (weekly and up)"""
    o, h, l, c = (frame[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    agg = {"High": "max", "Low": "min", "Close": "last"}
    prior = {}
    for tf, rule in (("W", "W-FRI"), ("M", "ME"), ("Y", "YE")):
        bars = frame.resample(rule).agg(agg).dropna()
        if len(bars) < 2:
            continue
        if tf == "Y" and (bars.index[-2].year != frame.index[-1].year - 1
                          or frame.index[0] > pd.Timestamp(bars.index[-2].year, 1, 10, tz=frame.index[0].tz)):
            continue    # need the whole of last year
        hi, lo = float(bars["High"].iloc[-2]), float(bars["Low"].iloc[-2])
        prior[tf] = {**_edge_levels(hi, lo), "hi": hi}

    def patterns(age: int) -> list[tuple[str, int, float]]:
        """Reversal patterns confirmed by bar -1-age: (name, index of the low bar, pattern low)."""
        up = -1 - age
        mid, before = up - 1, up - 2
        found = []
        if l[mid] < l[before] and l[up] > l[mid] and c[up] > o[up] and c[up] > h[mid]:
            found.append(("Minor low", mid, float(l[mid])))
        if h[mid] <= h[before] and l[mid] >= l[before] and c[up] > h[mid]:
            found.append(("Inside and up", before, float(min(l[before], l[mid]))))
        return found

    period = {"W": frame.index.to_period("W-FRI"), "M": frame.index.to_period("M"), "Y": frame.index.to_period("Y")}
    banners, setups = {}, {}
    for tf in ("D", "W", "M", "Y"):
        if tf == "D":
            lv = {**_edge_levels(h[-2], l[-2]), "hi": float(h[-2])}
        elif tf in prior:
            lv = prior[tf]
        else:
            continue
        lbl = _edge_label(c[-1], lv)
        banners[tf] = lbl
        tol = EDGE_TOL[tf]
        same = (lambda i: True) if tf == "D" else (lambda i: period[tf][i] == period[tf][-1])
        found = []

        # Edge reversal: minor low or inside and up at the lower edge
        for a in (0, 1):
            names, plow = [], None
            for name, low_bar, pl in patterns(a):
                if tf != "D" and not same(low_bar):
                    continue        # the low has to be made against this period's edge
                edge = _edge_levels(h[low_bar - 1], l[low_bar - 1])["le"] if tf == "D" else lv["le"]
                if pl <= edge * (1 + tol):
                    names.append(name)
                    plow = pl if plow is None else min(plow, pl)
            if names:
                found.append({"n": "Edge reversal", "pat": names, "age": a, "stop": plow, "tgt": lv["ue"]})
                break
        # Edge rejection: under the lower edge, closed back above it (today or yesterday)
        for a in (0, 1):
            i = -1 - a
            # the daily edge for bar i comes from the bar before it; longer edges from the prior period
            bl = _edge_levels(h[i - 1], l[i - 1]) if tf == "D" else lv
            if same(i) and l[i] < bl["le"] and c[i] > bl["le"] and c[-1] > bl["le"]:
                found.append({"n": "Edge rejection", "age": a, "stop": float(l[i]), "tgt": bl["be"]})
                break
        # Accelerator hold: bullish banner, pulled back to the accelerator and closed above it
        if lbl in (1, 2) and l[-1] <= lv["bu"] * (1 + max(tol, 0.003)) and c[-1] >= lv["bu"]:
            found.append({"n": "Accelerator hold", "age": 0, "stop": lv["mid"], "tgt": lv["ue"]})
        # Breakout retest: prior high broken earlier this period, back to it and holding
        if tf != "D":
            idx = [i for i in range(-len(c), -1) if same(i)]
            if idx and max(h[i] for i in idx) > lv["hi"] and l[-1] <= lv["hi"] * (1 + max(tol, 0.005)) and c[-1] > lv["hi"]:
                found.append({"n": "Breakout retest", "age": 0, "stop": lv["ue"], "tgt": None})
        # do not chase: drop longs already at their target or in the extreme bullish zone
        # (the breakout retest lives above the edge by design)
        found = [f for f in found if f["n"] == "Breakout retest"
                 or (lbl != 3 and (f["tgt"] is None or c[-1] < f["tgt"]))]
        if found:
            for f in found:
                f["stop"] = round(f["stop"], 2)
                f["tgt"] = round(f["tgt"], 2) if f["tgt"] is not None else None
            setups[tf] = {"set": found, "le": round(lv["le"], 2), "be": round(lv["be"], 2), "mid": round(lv["mid"], 2),
                          "bu": round(lv["bu"], 2), "ue": round(lv["ue"], 2), "hi": round(lv["hi"], 2)}
    return {"bn": banners, "tf": setups}


def _macd_hist(closes: pd.Series) -> pd.Series:
    """Moxie core: EMA12 - EMA26 of the higher timeframe closes, minus its EMA9."""
    vc = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    return vc - vc.ewm(span=9, adjust=False).mean()


def trampoline(moxie_up: np.ndarray, close: np.ndarray, ma_mid: np.ndarray, clear: int = 1) -> tuple[int, int]:
    """Port of the ST_MoxiePriceSignal trampoline state machine. Returns (state, bars in state) for the
    last bar: 0 idle, 1 trampoline potential (Moxie up, close under the 50 SMA), 2 trampoline confirmed
    (closed back over the 50 SMA), 3 inverse potential, 4 inverse confirmed."""
    code = 0
    for up, c, m in zip(moxie_up, close, ma_mid):
        if np.isnan(m):
            code = 0
            continue
        dn = not up
        bull_setup, bear_setup = up and c < m, dn and c > m
        if 1000 <= code < 2000:
            code = (3000 if bear_setup else 0) if not up else (2001 if c > m else 1000)
        elif 2000 <= code < 3000:
            if not up:
                code = 3000 if bear_setup else 0
            elif c <= m:
                code = 1000
            elif clear > 0 and code - 2000 >= clear:
                code = 0
            else:
                code = 2000 + min(code - 2000 + 1, 999)
        elif 3000 <= code < 4000:
            code = (1000 if bull_setup else 0) if not dn else (4001 if c < m else 3000)
        elif code >= 4000:
            if not dn:
                code = 1000 if bull_setup else 0
            elif c >= m:
                code = 3000
            elif clear > 0 and code - 4000 >= clear:
                code = 0
            else:
                code = 4000 + min(code - 4000 + 1, 999)
        else:
            code = 1000 if bull_setup else 3000 if bear_setup else 0
    return code // 1000, code % 1000


def moxie_daily(frame: pd.DataFrame) -> dict:
    """Moxie on a daily chart (ST_MoxieIndicator / ST_MoxiePriceSignal): the higher timeframe is the
    week, so Moxie = MACD(12, 26, 9) histogram of weekly closes (x3 for the indicator line). Every daily
    bar carries its week's value. Also the daily trampoline, the zero line cross and the trend stack."""
    c = frame["Close"]
    weeks = c.resample("W-FRI").last().dropna()
    hist_w = _macd_hist(weeks)
    per = frame.index.to_period("W-FRI")
    mox = per.map(pd.Series(hist_w.values, index=hist_w.index.to_period("W-FRI"))).to_numpy(float) * 3
    sma50 = c.rolling(50).mean().to_numpy()
    sma200 = c.rolling(200).mean().to_numpy()
    cv = c.to_numpy()
    st, bars = trampoline(mox[-300:] > 0, cv[-300:], sma50[-300:], clear=1)
    # weekly Moxie only changes at a new week, so the cross is this week's value over zero after last week's under
    cross = 0 if len(hist_w) > 1 and hist_w.iloc[-2] < 0 < hist_w.iloc[-1] else None
    e8, e21 = c.ewm(span=8, adjust=False).mean().iloc[-1], c.ewm(span=21, adjust=False).mean().iloc[-1]
    trend = 1 if e8 > e21 > sma50[-1] else -1 if e8 < e21 < sma50[-1] else 0
    return {"mox": round(float(mox[-1]), 4), "mox_up": bool(mox[-1] > 0),
            "mox_rising": bool(mox[-1] > mox[-1 - 5]) if len(mox) > 6 else False,
            "d_state": st, "d_bars": bars, "cross": cross, "trend": trend,
            "s50": round(float(sma50[-1]), 2), "s200": round(float(sma200[-1]), 2),
            "d50": round(float((cv[-1] / sma50[-1] - 1) * 100), 2)}


def moxie_hourly(tickers: list[str], daily: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Hourly Moxie trampoline (the thinkorswim "Moxie Trampoline Hourly Scan"): on an hourly chart the
    higher timeframe is the day, so Moxie = MACD(12, 26, 9) histogram of daily closes. Scan: hourly close
    under the hourly 50 SMA and over the hourly 200 SMA."""
    out = {}
    for start in range(0, len(tickers), 300):
        batch = tickers[start:start + 300]
        try:
            raw = yf.download(batch, period="60d", interval="1h", auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001
            print(f"hourly batch {start} failed: {exc}")
            continue
        for t in batch:
            try:
                hf = raw[t].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(hf) < 210:
                continue
            hd = _macd_hist(daily[t]["Close"])
            day_of = hf.index.tz_convert(daily[t].index.tz).normalize() if daily[t].index.tz else hf.index.tz_localize(None).normalize()
            mox = pd.Series(hd.values, index=hd.index.normalize()).reindex(day_of).to_numpy(float)
            hc = hf["Close"].to_numpy(float)
            h50 = hf["Close"].rolling(50).mean().to_numpy()
            h200 = hf["Close"].rolling(200).mean().to_numpy()
            st, bars = trampoline(np.nan_to_num(mox) > 0, hc, h50, clear=1)
            out[t] = {"h_mox": round(float(mox[-1]), 4) if not np.isnan(mox[-1]) else None,
                      "h_scan": bool(hc[-1] < h50[-1] and hc[-1] > h200[-1]),
                      "h_state": st, "h50": round(float(h50[-1]), 2), "h200": round(float(h200[-1]), 2),
                      "h_d50": round(float((hc[-1] / h50[-1] - 1) * 100), 2)}
    return out


def squeeze_pro(frame: pd.DataFrame) -> dict:
    """Port of ST_SqueezePRO (Simpler Trading): Bollinger (20, 2.0) upper band against Keltner upper bands
    at 1.0 (high squeeze, orange), 1.5 (mid, red) and 2.0 (low, black). Momentum = TTM Squeeze histogram:
    20 bar linear regression of close minus the average of the Donchian mid and the 20 SMA."""
    o, h, l, c = (frame[k] for k in ("Open", "High", "Low", "Close"))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    sma20 = c.rolling(20).mean()
    bb_up = sma20 + 2.0 * c.rolling(20).std(ddof=0)
    atr20 = tr.rolling(20).mean()
    lvl = np.select([bb_up <= sma20 + 1.0 * atr20, bb_up <= sma20 + 1.5 * atr20, bb_up <= sma20 + 2.0 * atr20],
                    [3, 2, 1], 0)                      # 3 high, 2 mid, 1 low, 0 no squeeze
    delta = c - ((h.rolling(20).max() + l.rolling(20).min()) / 2 + sma20) / 2
    x = np.arange(20)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    dv = delta.to_numpy()
    reg_end = lambda y: y.mean() + 9.5 * (w * y).sum()      # 20 bar regression line value at its last bar
    m, m1 = float(reg_end(dv[-20:])), float(reg_end(dv[-21:-1]))
    color = ("cyan" if m >= 0 else "yellow") if m > m1 else ("blue" if m >= 0 else "red")
    on = lvl > 0
    bars = 0
    while bars < len(on) and on[len(on) - 1 - bars]:
        bars += 1
    fired = bool(lvl[-1] == 0 and lvl[-2] > 0)
    cv, hv = c.to_numpy(), h.to_numpy()
    hi252 = float(hv[-253:-1].max())                    # the 252 day high before today
    e21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
    s50_all = c.rolling(50).mean().to_numpy()
    s50, s50_3 = float(s50_all[-1]), s50_all[-3:]
    # respects the 50 SMA: a low reached it (within 1%) in the last 3 bars and every close held above it
    respect50 = bool((l.to_numpy()[-3:] <= s50_3 * 1.01).any() and (cv[-3:] >= s50_3).all())
    return {"lvl": int(lvl[-1]), "lvl1": int(lvl[-2]), "sq_bars": bars, "fired": fired,
            "mom": round(m, 3), "mom_c": color, "hi252": round(hi252, 2),
            "off_hi": round((1 - cv[-1] / hi252) * 100, 2), "new_hi": bool(hv[-1] >= hi252),
            "e21": round(e21, 2), "e21d": round((cv[-1] / e21 - 1) * 100, 2),
            "s50": round(s50, 2), "s50d": round((cv[-1] / s50 - 1) * 100, 2), "resp50": respect50,
            "avgvol": int(frame["Volume"].iloc[-50:].mean())}


def squeeze_ultra(frame: pd.DataFrame) -> dict:
    """Port of ST_SqueezeUltra (daily settings, length 21). Bollinger upper = 21 EMA + 2 population
    std devs; Keltner upper = 21 SMA + 1.0 / 1.5 / 2.0 times the 21 EMA of true range (3 high, 2 mid,
    1 low). Momentum = TTM Squeeze histogram (length 21). Ideal squeeze (the IdealSqueezeScan plot):
    8 > 13 > 21 EMA, the 13 and 21 EMA both rising, and a mid squeeze."""
    n = 21
    h, l, c = (frame[k] for k in ("High", "Low", "Close"))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    e8, e13, e21 = (c.ewm(span=k, adjust=False).mean() for k in (8, 13, 21))
    bb_up = e21 + 2.0 * c.rolling(n).std(ddof=0)
    sma = c.rolling(n).mean()
    shift = tr.ewm(span=n, adjust=False).mean()
    lvl = np.select([bb_up <= sma + 1.0 * shift, bb_up <= sma + 1.5 * shift, bb_up <= sma + 2.0 * shift], [3, 2, 1], 0)
    ideal = ((e8 > e13) & (e13 > e21) & (e13 > e13.shift()) & (e21 > e21.shift())).to_numpy() & (lvl == 2)
    delta = (c - ((h.rolling(n).max() + l.rolling(n).min()) / 2 + e21) / 2).to_numpy()
    x = np.arange(n)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    reg_end = lambda y: y.mean() + (n - 1) / 2 * (w * y).sum()   # regression line value at its last bar
    m, m1 = float(reg_end(delta[-n:])), float(reg_end(delta[-n - 1:-1]))
    color = ("cyan" if m >= 0 else "yellow") if m > m1 else ("blue" if m >= 0 else "red")
    run = lambda arr: next((k for k in range(len(arr)) if not arr[len(arr) - 1 - k]), len(arr))
    cv = c.to_numpy()
    e21v = float(e21.iloc[-1])
    s50_all = c.rolling(50).mean().to_numpy()
    s50, s50_3 = float(s50_all[-1]), s50_all[-3:]
    respect50 = bool((l.to_numpy()[-3:] <= s50_3 * 1.01).any() and (cv[-3:] >= s50_3).all())
    return {"lvl": int(lvl[-1]), "lvl1": int(lvl[-2]), "sq_bars": run(lvl > 0), "fired": bool(lvl[-1] == 0 and lvl[-2] > 0),
            "ideal": bool(ideal[-1]), "ideal_bars": run(ideal), "mom": round(m, 3), "mom_c": color,
            "e21": round(e21v, 2), "e21d": round((cv[-1] / e21v - 1) * 100, 2),
            "s50": round(s50, 2), "s50d": round((cv[-1] / s50 - 1) * 100, 2), "resp50": respect50,
            "avgvol": int(frame["Volume"].iloc[-50:].mean())}


def _fib_pocket(bar: pd.Series) -> tuple[float, float, float]:
    """AC/DC levels off one bar: upper = high - 38.2% of range, mid = 50%, lower = 61.8%."""
    h, l = float(bar["High"]), float(bar["Low"])
    return h - (h - l) * 0.382, h - (h - l) * 0.5, h - (h - l) * 0.618


def acdc(frame: pd.DataFrame) -> dict:
    """Port of the AC/DC BULL thinkorswim scan. Monthly study: this month's close inside the
    prior month's 38.2 to 61.8 pocket. Weekly study: this week's close at or above the prior
    week's 38.2 line (upper). Bars are built from daily data, the current bar is the live one."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    m = frame.resample("ME").agg(agg).dropna()
    w = frame.resample("W-FRI").agg(agg).dropna()
    if len(m) < 2 or len(w) < 2:
        return {"pocket": False, "wbull": False}
    c = float(frame["Close"].iloc[-1])
    mu, mm, ml = _fib_pocket(m.iloc[-2])
    wu, wm, wl = _fib_pocket(w.iloc[-2])

    # Trend strength 0 to 6, strongest bull trends first (same idea as the Morning Star AC/DC scan)
    cl, hi, lo = frame["Close"], frame["High"], frame["Low"]
    e = {n: float(cl.ewm(span=n, adjust=False).mean().iloc[-1]) for n in (8, 21, 50, 200)}
    up, dn = hi.diff(), -lo.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    rma = lambda s: pd.Series(s, index=cl.index).ewm(alpha=1 / 14, adjust=False).mean()
    atr = rma(tr)
    pdi, mdi = 100 * rma(plus_dm) / atr, 100 * rma(minus_dm) / atr
    adx = float(rma((100 * (pdi - mdi).abs() / (pdi + mdi)).fillna(0)).iloc[-1])
    r1m = c / float(cl.iloc[-22]) - 1
    strength = int(sum([c > e[21], e[8] > e[21], e[21] > e[50], e[50] > e[200], adx >= 20, r1m > 0]))
    stacked = c > e[21] > e[50] > e[200]
    return {
        "pocket": bool(ml <= c <= mu), "wbull": bool(c >= wu),
        "m_up": round(mu, 2), "m_mid": round(mm, 2), "m_lo": round(ml, 2),
        "m_hi": round(float(m.iloc[-2]["High"]), 2), "m_low": round(float(m.iloc[-2]["Low"]), 2),
        "w_up": round(wu, 2), "w_hi": round(float(w.iloc[-2]["High"]), 2),
        "pos": round((c - ml) / (mu - ml), 2) if mu > ml else 0.0,   # 0 = at 61.8 line, 1 = at 38.2 line
        "adx": round(adx, 1), "tstr": strength,
        "tlabel": "Strong" if strength >= 5 and stacked else "Building" if strength >= 3 else "Weak",
    }


def optionable_symbols() -> tuple[set, set]:
    """Every optionable symbol and the ones with weekly options, from Cboe's symbol directory."""
    def symbols(url: str) -> set:
        text = _get(url)
        return {line.split('","')[1].strip().replace(".", "-") for line in text.splitlines()[1:] if '","' in line}
    try:
        return (symbols("https://www.cboe.com/us/options/symboldir/equity_index_options/?download=csv"),
                symbols("https://www.cboe.com/us/options/symboldir/weeklys_options/?download=csv"))
    except Exception as exc:  # noqa: BLE001
        print(f"optionable directory failed: {exc}")
        return set(), set()


def attach_options(*row_lists: list, limit: int = MAX_OI_FETCH) -> None:
    """Optionable (directory) on every row; open interest from Cboe quotes, fetched slowly to stay
    under Cboe's rate limit and cached for the day (open interest only updates once a day)."""
    listed, weekly = optionable_symbols()
    cache = {}
    if OPTIONS_CACHE.exists():
        saved = json.loads(OPTIONS_CACHE.read_text())
        if saved.get("date") == str(pd.Timestamp.now().date()):
            cache = saved["data"]
    fetched, blocked = 0, False
    for rows in row_lists:
        for row in rows:
            t = row["t"]
            if t not in cache and listed and t in listed and fetched < limit and not blocked:
                info = options_info(t)
                time.sleep(0.4)
                fetched += 1
                if info:
                    cache[t] = info
                else:
                    blocked = True     # Cboe is rate limiting: stop asking, keep what is cached
            row.update(cache.get(t, {}))
            if listed:
                row["opt"] = t in listed
                row["wk"] = t in weekly
    OPTIONS_CACHE.write_text(json.dumps({"date": str(pd.Timestamp.now().date()), "data": cache}))
    print(f"options: {fetched} fetched, {len(cache)} cached{', stopped by Cboe rate limit' if blocked else ''}")


def options_info(ticker: str) -> dict:
    """Optionable and option liquidity from Cboe's free delayed quotes. Liquidity follows the
    thinkorswim scans: open interest MIN_OPT_OI+ at the first expiration MIN_OPT_DTE+ days out.
    The bid/ask spread of the call closest to the money there is reported for reference."""
    from curl_cffi import requests
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker.replace('-', '.')}.json"
        for wait in (0, 5, 15):        # Cboe answers 429 when asked too fast: back off and retry
            time.sleep(wait)
            r = requests.get(url, impersonate="chrome", timeout=20)
            if r.status_code != 429:
                break
        if r.status_code == 429:
            return {}                  # unknown this run; optionable still comes from the directory
        if r.status_code != 200:
            return {"opt": False}
        data = r.json()["data"]
        px = float(data.get("current_price") or 0)
        today = pd.Timestamp.now().normalize()
        rows = []
        for o in data.get("options", []):
            code = o["option"][-15:]           # YYMMDD + C/P + strike x 1000
            rows.append((pd.Timestamp("20" + code[:6]), code[6], int(code[7:]) / 1000,
                         o.get("open_interest") or 0, o.get("volume") or 0, o.get("bid") or 0, o.get("ask") or 0))
        if not rows:
            return {"opt": False}
        chain = pd.DataFrame(rows, columns=["exp", "cp", "strike", "oi", "vol", "bid", "ask"])
        far = chain[(chain["exp"] - today).dt.days >= MIN_OPT_DTE]
        if far.empty:
            return {"opt": True, "oi": 0, "ovol": int(chain["vol"].sum()), "spr": None, "liq": False}
        exp = far["exp"].min()
        leg = far[far["exp"] == exp]
        oi = int(leg["oi"].sum())
        calls = leg[leg["cp"] == "C"]
        atm = calls.iloc[(calls["strike"] - px).abs().argsort()[:1]] if len(calls) else calls
        spr = None
        if len(atm):
            bid, ask = float(atm["bid"].iloc[0]), float(atm["ask"].iloc[0])
            if ask > 0 and bid > 0:
                spr = round((ask - bid) / ((ask + bid) / 2) * 100, 1)
        return {"opt": True, "oi": oi, "oi_all": int(chain["oi"].sum()), "ovol": int(chain["vol"].sum()),
                "exp": str(exp.date()), "spr": spr,
                "liq": bool(oi >= MIN_OPT_OI)}
    except Exception as exc:  # noqa: BLE001  a bad feed for one name must not sink the scan
        print(f"options {ticker}: {exc}")
        return {"opt": None}


def beta_vs(frame: pd.DataFrame, bench: pd.DataFrame, n: int = 252) -> float | None:
    """Beta against SPY from the last year of daily returns: cov(stock, SPY) / var(SPY)."""
    sc, bc = aligned_closes(frame, bench)
    if len(sc) < n + 1:
        return None
    rs, rb = returns(sc, n), returns(bc, n)
    vb = float(np.var(rb))
    return round(float(np.cov(rs, rb, ddof=0)[0, 1] / vb), 2) if vb > 0 else None


def returns(c: np.ndarray, n: int = 60) -> np.ndarray:
    return c[-n:] / c[-n - 1:-1] - 1


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def aligned_closes(stock: pd.DataFrame, ref: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    joined = pd.concat([stock["Close"], ref["Close"]], axis=1, join="inner").dropna()
    return joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()


# ---------------------------------------------------------------- scan

def scan() -> dict:
    started = time.time()
    universe = load_universe()
    etfs = INDEXES + list(SECTORS)
    frames = download(etfs + universe["ticker"].tolist())
    closes = {t: frames[t]["Close"].to_numpy() for t in etfs if t in frames}

    spy = closes["SPY"]
    sma50, sma200 = spy[-50:].mean(), spy[-200:].mean()
    regime = "UPTREND" if spy[-1] > sma50 > sma200 else "CHOP" if spy[-1] > sma200 else "DOWNTREND"

    index_rows = []
    for t in INDEXES:
        c = closes[t]
        r, up = blend(c), trend_up(c)
        index_rows.append({"ticker": t, "strength": round(r * 100, 2), "trend_up": up,
                           "state": int(r > 0) + int(up), "day": round((c[-1] / c[-2] - 1) * 100, 2),
                           **jt_trends(frames[t])})
    index_rows.sort(key=lambda x: x["strength"], reverse=True)
    leader_index = index_rows[0]["ticker"]
    leader_ret = blend(closes[leader_index])

    sector_rows = []
    for t, label in SECTORS.items():
        c = closes[t]
        score, up = blend(c) - leader_ret, trend_up(c)
        sector_rows.append({"ticker": t, "name": label, "vs_index": round(score * 100, 2),
                            "strength": round(blend(c) * 100, 2), "trend_up": up,
                            "state": (1 + int(up)) if score > 0 else 0,
                            "day": round((c[-1] / c[-2] - 1) * 100, 2), **jt_trends(frames[t])})
    sector_rows.sort(key=lambda x: x["vs_index"], reverse=True)
    sector_state = {s["ticker"]: s["state"] for s in sector_rows}
    sector_rank = {s["ticker"]: i + 1 for i, s in enumerate(sector_rows)}

    etf_returns = {t: frames[t] for t in SECTORS}
    stocks, all_blends = [], []
    for rec in universe.itertuples(index=False):
        frame = frames.get(rec.ticker)
        if frame is None or len(frame) < 253:
            continue
        c = frame["Close"].to_numpy()
        core = str(rec.core) in ("True", "true", "1")
        if core:
            all_blends.append(blend(c))
        etf = rec.etf if rec.etf in SECTORS else ""
        if not etf:  # unknown sector: assign the sector ETF it moves with most
            fits = {e: corr(*map(returns, aligned_closes(frame, etf_returns[e]))) for e in SECTORS}
            etf = max(fits, key=fits.get)
        sc, ec = aligned_closes(frame, etf_returns[etf])
        if len(sc) < 253:
            continue
        stock_r, etf_r = blend(sc), blend(ec)
        rs_line = sc / ec
        e21, s50, s200 = ema(sc, 21), sc[-50:].mean(), sc[-200:].mean()
        high52 = float(frame["High"].to_numpy()[-252:].max())
        off_high = (1 - c[-1] / high52) * 100
        avg_vol = float(frame["Volume"].to_numpy()[-50:].mean())
        r_corr = corr(returns(sc), returns(ec))
        checks = [
            stock_r > etf_r,
            sc[-1] / sc[-22] > ec[-1] / ec[-22],
            sc[-1] / sc[-64] > ec[-1] / ec[-64],
            rs_line[-1] > rs_line[-50:].mean(),
            sc[-1] > e21 > s50 > s200,
            r_corr >= MIN_CORRELATION,
            off_high <= MAX_OFF_HIGH,
            c[-1] >= MIN_PRICE and avg_vol >= MIN_AVG_VOLUME,
        ]
        passed = int(sum(checks))
        base_trend = bool(sc[-1] > s50 > s200)  # pullbacks may dip under the 21 EMA, not the 50
        stocks.append({
            **pullback_triggers(frame), **jt_trends(frame), "base": base_trend, "b3": big3(frame), "sdc": std_channel(frame), "acdc": acdc(frame), "edges": tr3ndy_edges(frame), "mx": moxie_daily(frame), "sqp": {**squeeze_pro(frame), "beta": beta_vs(frame, frames["SPY"])}, "squ": squeeze_ultra(frame),
            "cap": round(float(rec.cap) / 1e9, 2), "core": core,
            "t": rec.ticker, "n": rec.name, "etf": etf, "sp": bool(rec.sp500),
            "px": round(float(c[-1]), 2), "day": round((c[-1] / c[-2] - 1) * 100, 2),
            "vs_sec": round((stock_r - etf_r) * 100, 2), "str": round(stock_r * 100, 2),
            "r1m": round((c[-1] / c[-22] - 1) * 100, 1), "r3m": round((c[-1] / c[-64] - 1) * 100, 1),
            "off": round(off_high, 1), "corr": round(r_corr, 2), "vol": int(avg_vol),
            "chk": [int(x) for x in checks], "pass": passed,
            "sec_state": sector_state[etf], "sec_rank": sector_rank[etf],
        })

    # RS rating 1 to 99: percentile of each stock's strength across the whole universe
    ranked = np.sort(np.array(all_blends))
    for s in stocks:
        pct = np.searchsorted(ranked, s["str"] / 100, side="right") / len(ranked)
        s["rs"] = int(min(99, max(1, round(pct * 99))))
    # Big 3 tab: every stock in a squeeze or carrying any Big 3 signal, with its top down context
    keep = ("t", "n", "etf", "px", "day", "vs_sec", "rs", "pass", "sec_state", "momo", "wave", "e21d", "cap", "vol")
    big3_rows = [{**{k: s[k] for k in keep}, **s["b3"]} for s in stocks
                 if s["core"] and (s["b3"]["sqz"] or s["b3"]["sig"])]
    big3_rows.sort(key=lambda s: (s["bull"], s["score"], s["rs"]), reverse=True)
    # Std Dev Channel tab: every stock 2.5 or more deviations from its 1 year regression line
    sd_rows = [{**{k: s[k] for k in keep}, **s["sdc"], "vol1": s["b3"]["vol1"]}
               for s in stocks if abs(s["sdc"]["dev"]) >= 2.5]
    sd_rows.sort(key=lambda s: s["dev"])
    # Trendy Edge tab: AC/DC BULL, monthly pocket and weekly push, market cap $1B+ (no price floor)
    acdc_rows = [{**{k: s[k] for k in keep}, **s["acdc"], "vol1": s["b3"]["vol1"]} for s in stocks
                 if s["cap"] >= 1 and s["acdc"]["pocket"] and s["acdc"]["wbull"]]
    acdc_rows.sort(key=lambda s: (s["tstr"], s["adx"], s["rs"]), reverse=True)
    # Tr3ndy Edges views: stocks at a daily, weekly, monthly or yearly lower edge, cap $1B+
    keep_e = ("t", "n", "etf", "px", "day", "rs", "pass", "sec_state", "momo", "wave", "cap")
    edge_rows = [{**{k: s[k] for k in keep_e}, "ed": s["edges"], "vol1": s["b3"]["vol1"]} for s in stocks
                 if s["cap"] >= 0.35 and s["edges"]["tf"]]
    # options: std dev and AC/DC first, then edge setups from the yearly down to the daily
    edge_rows.sort(key=lambda r: [tf in r["ed"]["tf"] for tf in "YMWD"], reverse=True)
    attach_options(sd_rows, acdc_rows, edge_rows)
    edge_rows.sort(key=lambda r: r["rs"], reverse=True)
    # Moxie tab: daily chart Moxie up (ST_MoxieScan on D), price $1.25+, volume 500K+, then the hourly check
    keep_m = ("t", "n", "etf", "px", "day", "rs", "pass", "sec_state", "momo", "wave", "cap")
    mox_rows = [{**{k: s[k] for k in keep_m}, **s["mx"], "vol1": s["b3"]["vol1"]} for s in stocks
                if s["mx"]["mox_up"] or s["mx"]["d_state"] in (1, 2) or s["mx"]["cross"] is not None]
    hourly = moxie_hourly([r["t"] for r in mox_rows if r["mox_up"] and r["px"] >= 1.25 and r["vol1"] >= 500_000],
                          frames)
    for r in mox_rows:
        r.update(hourly.get(r["t"], {}))
    mox_rows = [r for r in mox_rows if r.get("h_scan") or r["d_state"] in (1, 2) or r.get("h_state") in (1, 2)
                or r["cross"] is not None]
    print(f"moxie: {len(hourly)} hourly checked, {len(mox_rows)} rows")
    # Squeeze Pro tab: in a squeeze (any level) or fired today, within 15% of the 252 day high
    sqp_rows = [{**{k: s[k] for k in keep_m}, **s["sqp"], "vol1": s["b3"]["vol1"]} for s in stocks
                if (s["sqp"]["lvl"] > 0 or s["sqp"]["fired"]) and s["sqp"]["off_hi"] <= 15]
    attach_options(sqp_rows)
    # Squeeze Ultra tab: in any squeeze or fired today, $10+ and $1B+ (the page filters down to the scan)
    squ_rows = [{**{k: s[k] for k in keep_m}, **s["squ"], "beta": s["sqp"]["beta"], "vol1": s["b3"]["vol1"]} for s in stocks
                if (s["squ"]["lvl"] > 0 or s["squ"]["fired"]) and s["px"] >= 10 and s["cap"] >= 1]
    # options for the ones that match the ToS scan first (ideal, $20+, $2B+), the rest as the cache allows
    squ_rows.sort(key=lambda r: (r["ideal"] and r["px"] >= 20 and r["cap"] >= 2, r["ideal"], r["cap"]), reverse=True)
    attach_options(squ_rows, limit=150)
    print(f"squeeze ultra: {len(squ_rows)} rows, {sum(r['ideal'] for r in squ_rows)} ideal")
    # True Momentum tab: HiLo PRO X, weekly Trend Oscillator, Structure Score, swing levels. Cap $2B+, $5+
    tm_t = [s for s in stocks if s["cap"] >= 2 and s["px"] >= 5]
    t0 = time.time()
    weekly = download_weekly([s["t"] for s in tm_t])
    monthly = download_weekly([s["t"] for s in tm_t], "1mo", "20y")
    tm_rows = []
    for s in tm_t:
        if s["t"] not in weekly:
            continue
        try:
            tm = true_momentum(frames[s["t"]], weekly[s["t"]], monthly.get(s["t"]))
        except Exception as exc:  # noqa: BLE001
            print(f"true momentum {s['t']}: {exc}")
            continue
        if tm["buy10"] or tm["sell10"] or abs(tm["tm_score"]) >= 40:
            tm_rows.append({**{k: s[k] for k in keep_m}, **tm, "e21d": s["sqp"]["e21d"], "resp50": s["sqp"]["resp50"],
                            "s50d": s["sqp"]["s50d"], "vol1": s["b3"]["vol1"], "beta": s["sqp"]["beta"],
                            "sqp": s["sqp"]["lvl"], "squ": s["squ"]["lvl"], "squ_ideal": s["squ"]["ideal"]})
    print(f"true momentum: {len(weekly)} weekly series in {time.time() - t0:.0f}s, {len(tm_rows)} rows")
    for s in stocks:
        del s["b3"], s["sdc"], s["acdc"], s["edges"], s["mx"], s["sqp"], s["squ"]
    wide_count = len(stocks)
    stocks = [s for s in stocks if s.pop("core")]
    listed = [s for s in stocks if s["pass"] >= MIN_SETUP_TO_LIST]
    listed.sort(key=lambda s: (s["pass"], s["sec_state"], s["vs_sec"]), reverse=True)

    last_bar = max(f.index[-1] for f in frames.values())
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_date": str(pd.Timestamp(last_bar).date()),
        "seconds": round(time.time() - started, 1),
        "universe_size": int(universe["core"].astype(str).isin(["True", "true", "1"]).sum()), "scanned": len(stocks),
        "scanned_wide": wide_count,
        "settings": {"min_cap": MIN_CAP, "min_price": MIN_PRICE, "min_avg_volume": MIN_AVG_VOLUME,
                     "min_corr": MIN_CORRELATION, "max_off_high": MAX_OFF_HIGH},
        "checks": CHECKS,
        "market": {"regime": regime, "spy": round(float(spy[-1]), 2),
                   "sma50": round(float(sma50), 2), "sma200": round(float(sma200), 2)},
        "indexes": index_rows, "leader_index": leader_index,
        "sectors": sector_rows, "leader_sector": sector_rows[0]["ticker"],
        "stocks": listed, "big3": big3_rows, "stddev": sd_rows, "acdc": acdc_rows, "edges": edge_rows, "moxie": mox_rows, "sqpro": sqp_rows, "squltra": squ_rows, "truemo": tm_rows,
    }


if __name__ == "__main__":
    result = scan()
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"scanned {result['scanned']} of {result['universe_size']} in {result['seconds']}s, "
          f"listed {len(result['stocks'])}, market {result['market']['regime']}, "
          f"#1 index {result['leader_index']}, #1 sector {result['leader_sector']}")
