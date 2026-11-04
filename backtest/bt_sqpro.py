"""Backtest the Squeeze Pro tab on the same 1,755 stocks, 2008 to 2026, and merge into results.json.

Signal (first day all true, no lookahead): in a squeeze (Bollinger 20/2 upper inside Keltner 20 at 2.0
or tighter), price $20+, volume 150K+, close within 10% of the 252 day high before that day, and that
day's high still under it (no new high yet). Variants add the tab's filters: entry at the 21 EMA,
50 SMA hold, either, cyan momentum, high squeeze only, beta 1+ vs SPY.

Exit (common Squeeze Pro practice, the study has none): entry next open; out the next open after a
close under the 50 SMA; once the squeeze has fired, out after the second falling momentum bar above
zero (dark blue) or momentum dropping below zero; 60 day time limit.
"""
import json
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from bt_scanner import HERE, PRICES, summarize


def indicators(df: pd.DataFrame):
    o_, h_, l_, c_ = (df[k] for k in ("Open", "High", "Low", "Close"))
    tr = pd.concat([h_ - l_, (h_ - c_.shift()).abs(), (l_ - c_.shift()).abs()], axis=1).max(axis=1)
    sma20 = c_.rolling(20).mean()
    bb = sma20 + 2.0 * c_.rolling(20).std(ddof=0)
    atr = tr.rolling(20).mean()
    lvl = np.select([bb <= sma20 + atr, bb <= sma20 + 1.5 * atr, bb <= sma20 + 2.0 * atr], [3, 2, 1], 0)
    delta = (c_ - ((h_.rolling(20).max() + l_.rolling(20).min()) / 2 + sma20) / 2).to_numpy()
    x = np.arange(20)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    d0 = np.nan_to_num(delta)
    dot = np.convolve(d0, w[::-1], mode="full")[:len(d0)]          # sum(w * window) ending at each bar
    mean = pd.Series(delta).rolling(20).mean().to_numpy()
    mom = mean + 9.5 * dot
    mom[:40] = np.nan
    return lvl, mom


def run(df: pd.DataFrame, spy_ret: pd.Series) -> dict:
    df = df[~df.index.duplicated()].dropna()
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    if n < 400 or np.any(c <= 0):
        return {}
    c_ = df["Close"]
    lvl, mom = indicators(df)
    mom1 = np.roll(mom, 1)
    e21 = c_.ewm(span=21, adjust=False).mean().to_numpy()
    s50 = c_.rolling(50).mean().to_numpy()
    hi_prev = df["High"].shift().rolling(252).max().to_numpy()
    l3 = df["Low"].rolling(3).min().to_numpy()
    s50_3min = (pd.Series(c) - pd.Series(s50)).rolling(3).min().to_numpy()     # every close of 3 above the 50
    touch50 = (pd.Series(l) <= pd.Series(s50) * 1.01).rolling(3).max().to_numpy() > 0
    resp50 = touch50 & (s50_3min >= 0)
    at21 = np.abs(c / e21 - 1) <= 0.01
    cyan = (mom >= 0) & (mom > mom1)
    r = c_.pct_change()
    sr = spy_ret.reindex(c_.index)
    beta = (r.rolling(252).cov(sr) / sr.rolling(252).var()).to_numpy()
    base = (lvl > 0) & (c >= 20) & (v >= 150_000) & (c >= hi_prev * 0.9) & (h < hi_prev)
    variants = {
        "Squeeze Pro: any squeeze": base,
        "Squeeze Pro: at 21 EMA": base & at21,
        "Squeeze Pro: 50 SMA hold": base & resp50,
        "Squeeze Pro: 21 EMA or 50 SMA": base & (at21 | resp50),
        "Squeeze Pro: entry + cyan momentum": base & (at21 | resp50) & cyan,
        "Squeeze Pro: high squeeze + entry": base & (lvl == 3) & (at21 | resp50),
        "Squeeze Pro: entry + beta 1+": base & (at21 | resp50) & (beta >= 1),
    }
    blue = (mom >= 0) & (mom < mom1)
    out = {}
    for name, sig in variants.items():
        first = sig & ~np.roll(sig, 1)
        trades, i = [], 260
        while i < n - 2:
            if not first[i]:
                i += 1
                continue
            e = o[i + 1]
            fired, x, j = False, None, i + 1
            while j < n - 1:
                if c[j] < s50[j]:
                    x = o[j + 1]; j += 1; break
                if lvl[j] == 0 and lvl[j - 1] > 0:
                    fired = True
                if fired and ((blue[j] and blue[j - 1]) or mom[j] < 0):
                    x = o[j + 1]; j += 1; break
                if j - i >= 60:
                    x = c[j]; break
                j += 1
            if x is None:
                break
            trades.append((df.index[i + 1].year, x / e - 1, j - i))
            i = j + 1
        out[name] = trades
    return out


if __name__ == "__main__":
    frames = pickle.loads(PRICES.read_bytes())
    spy = yf.download("SPY", start="2007-01-01", end="2026-11-01", auto_adjust=True, progress=False)["Close"].squeeze()
    spy_ret = spy.pct_change()
    allt = {}
    for k, (t, df) in enumerate(frames.items()):
        for name, tr in run(df, spy_ret).items():
            allt.setdefault(name, []).extend(tr)
        if k % 300 == 0:
            print(f"{k}/{len(frames)}", flush=True)
    res = json.loads((HERE / "results.json").read_text())
    for name, tr in allt.items():
        res["setups"][name] = summarize(tr)
        s = res["setups"][name]
        if s["n"]:
            print(f"{name:38s} n={s['n']:6d} win={s['win']}% avg={s['avg']}% pf={s['pf']} bars={s['bars']} worst={s['worst']}% "
                  f"old {s['old']['win']}%/{s['old']['avg']}% new {s['new']['win']}%/{s['new']['avg']}%")
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
