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
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

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
MIN_OPT_DTE = 20       # option liquidity is read at the first expiration this many days out
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
_NON_EQUITY = ("warrant", "right", "unit", "depositary", "etf", "fund", "preferred", "notes")


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
    equity = ("common stock", "ordinary share", "common share", "class a", "class b", "class c")
    out = []
    for row in rows:
        symbol, name = row.get("symbol", ""), row.get("name", "").lower()
        if not symbol or "^" in symbol or "/" in symbol:
            continue
        if not any(k in name for k in equity) or any(w in name for w in _NON_EQUITY):
            continue
        cap, price = _num(row.get("marketCap")), _num(row.get("lastsale"))
        if cap < WIDE_MIN_CAP or price < WIDE_MIN_PRICE:
            continue
        ticker = symbol.replace(".", "-")
        etf = gics.get(ticker) or NASDAQ_TO_ETF.get(row.get("sector", ""), "")
        clean = row.get("name", "")
        for junk in (" Common Stock", " Ordinary Shares", " Common Shares", " Class A", " Class B", " Class C"):
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
                raw = yf.download(batch, period="15mo", interval="1d", auto_adjust=True,
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


def options_info(ticker: str) -> dict:
    """Optionable and option liquidity from Cboe's free delayed quotes. Liquidity follows the
    thinkorswim scans: open interest MIN_OPT_OI+ at the first expiration MIN_OPT_DTE+ days out.
    The bid/ask spread of the call closest to the money there is reported for reference."""
    from curl_cffi import requests
    try:
        r = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker.replace('-', '.')}.json",
                         impersonate="chrome", timeout=20)
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
            **pullback_triggers(frame), **jt_trends(frame), "base": base_trend, "b3": big3(frame), "sdc": std_channel(frame),
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
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        for row, info in zip(sd_rows, pool.map(options_info, [r["t"] for r in sd_rows])):
            row.update(info)
    for s in stocks:
        del s["b3"], s["sdc"]
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
        "stocks": listed, "big3": big3_rows, "stddev": sd_rows,
    }


if __name__ == "__main__":
    result = scan()
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"scanned {result['scanned']} of {result['universe_size']} in {result['seconds']}s, "
          f"listed {len(result['stocks'])}, market {result['market']['regime']}, "
          f"#1 index {result['leader_index']}, #1 sector {result['leader_sector']}")
