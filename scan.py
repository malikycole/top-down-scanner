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
MIN_SETUP_TO_LIST = 5  # stocks below this many checks are left out of the JSON

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
        if cap < MIN_CAP or price < MIN_PRICE:
            continue
        ticker = symbol.replace(".", "-")
        etf = gics.get(ticker) or NASDAQ_TO_ETF.get(row.get("sector", ""), "")
        clean = row.get("name", "")
        for junk in (" Common Stock", " Ordinary Shares", " Common Shares", " Class A", " Class B", " Class C"):
            clean = clean.split(junk)[0]
        out.append({"ticker": ticker, "name": clean.strip()[:60], "etf": etf, "cap": cap,
                    "sp500": ticker in gics})
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
    return pd.read_csv(UNIVERSE_FILE, keep_default_na=False)


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
                           "state": int(r > 0) + int(up), "day": round((c[-1] / c[-2] - 1) * 100, 2)})
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
                            "day": round((c[-1] / c[-2] - 1) * 100, 2)})
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
        stocks.append({
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
    listed = [s for s in stocks if s["pass"] >= MIN_SETUP_TO_LIST]
    listed.sort(key=lambda s: (s["pass"], s["sec_state"], s["vs_sec"]), reverse=True)

    last_bar = max(f.index[-1] for f in frames.values())
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_date": str(pd.Timestamp(last_bar).date()),
        "seconds": round(time.time() - started, 1),
        "universe_size": len(universe), "scanned": len(stocks),
        "settings": {"min_cap": MIN_CAP, "min_price": MIN_PRICE, "min_avg_volume": MIN_AVG_VOLUME,
                     "min_corr": MIN_CORRELATION, "max_off_high": MAX_OFF_HIGH},
        "checks": CHECKS,
        "market": {"regime": regime, "spy": round(float(spy[-1]), 2),
                   "sma50": round(float(sma50), 2), "sma200": round(float(sma200), 2)},
        "indexes": index_rows, "leader_index": leader_index,
        "sectors": sector_rows, "leader_sector": sector_rows[0]["ticker"],
        "stocks": listed,
    }


if __name__ == "__main__":
    result = scan()
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"scanned {result['scanned']} of {result['universe_size']} in {result['seconds']}s, "
          f"listed {len(result['stocks'])}, market {result['market']['regime']}, "
          f"#1 index {result['leader_index']}, #1 sector {result['leader_sector']}")
