"""Squeeze Pro with longer exits. Same entry as the tab (in a squeeze, $20+, 150K+ volume, within 10%
of the 252 day high with no new high, at the 21 EMA or holding the 50 SMA), tested with and without
beta 1+. Entry next open. Every exit keeps the 50 SMA close stop and a 120 day limit, then adds:

  50 SMA only        nothing else, ride the trend
  21 EMA after fire  once the squeeze fires, out on a close under the 21 EMA
  momentum < 0       once the squeeze fires, out when momentum drops below zero (no dark blue rule)
  breakout trail     once price closes above the old 252 day high, out on a close under the 21 EMA
  20% target         sell at +20% (intraday high touches it)
Also a control: hold 60 days with no stop, to see if the signal itself has an edge.
"""
import json
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from bt_scanner import HERE, PRICES, summarize
from bt_sqpro import indicators

EXITS = ["hold 60 days, no stop", "50 SMA only", "21 EMA after fire", "momentum < 0", "breakout trail", "20% target"]
LIMIT = 120


def run(df: pd.DataFrame, spy_ret: pd.Series) -> dict:
    df = df[~df.index.duplicated()].dropna()
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    if n < 400 or np.any(c <= 0):
        return {}
    c_ = df["Close"]
    lvl, mom = indicators(df)
    e21 = c_.ewm(span=21, adjust=False).mean().to_numpy()
    s50 = c_.rolling(50).mean().to_numpy()
    hi_prev = df["High"].shift().rolling(252).max().to_numpy()
    s50_3min = (pd.Series(c) - pd.Series(s50)).rolling(3).min().to_numpy()
    touch50 = (pd.Series(l) <= pd.Series(s50) * 1.01).rolling(3).max().to_numpy() > 0
    resp50 = touch50 & (s50_3min >= 0)
    at21 = np.abs(c / e21 - 1) <= 0.01
    r = c_.pct_change()
    sr = spy_ret.reindex(c_.index)
    beta = (r.rolling(252).cov(sr) / sr.rolling(252).var()).to_numpy()
    entry = (lvl > 0) & (c >= 20) & (v >= 150_000) & (c >= hi_prev * 0.9) & (h < hi_prev) & (at21 | resp50)
    entries = {"": entry, " + beta 1+": entry & (beta >= 1)}
    out = {}
    for tag, sig in entries.items():
        first = sig & ~np.roll(sig, 1)
        for ex in EXITS:
            trades, i = [], 260
            while i < n - 2:
                if not first[i]:
                    i += 1
                    continue
                e, old_hi = o[i + 1], hi_prev[i]
                fired = broke = False
                x, j = None, i + 1
                while j < n - 1:
                    if ex == "20% target" and h[j] >= e * 1.2:
                        x = max(e * 1.2, o[j]); break
                    if ex == "hold 60 days, no stop":
                        if j - i >= 60:
                            x = c[j]; break
                        j += 1
                        continue
                    if c[j] < s50[j]:
                        x = o[j + 1]; j += 1; break
                    if lvl[j] == 0 and lvl[j - 1] > 0:
                        fired = True
                    if c[j] > old_hi:
                        broke = True
                    if ex == "21 EMA after fire" and fired and c[j] < e21[j]:
                        x = o[j + 1]; j += 1; break
                    if ex == "momentum < 0" and fired and mom[j] < 0:
                        x = o[j + 1]; j += 1; break
                    if ex == "breakout trail" and broke and c[j] < e21[j]:
                        x = o[j + 1]; j += 1; break
                    if j - i >= LIMIT:
                        x = c[j]; break
                    j += 1
                if x is None:
                    break
                trades.append((df.index[i + 1].year, x / e - 1, j - i))
                i = j + 1
            out[f"Squeeze Pro exit: {ex}{tag}"] = trades
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
    base = res["baseline"]
    Ns = [int(x) for x in base]
    for name, tr in allt.items():
        s = res["setups"][name] = summarize(tr)
        if s["n"]:
            k = min(Ns, key=lambda b: abs(b - s["bars"]))
            rnd = base[str(k)]["per_day"] * s["bars"]
            print(f"{name:48s} n={s['n']:6d} win={s['win']}% avg={s['avg']}% pf={s['pf']} bars={s['bars']} "
                  f"rnd={rnd:.2f}% edge={s['avg'] - rnd:+.2f}% old {s['old']['avg']}% new {s['new']['avg']}%")
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
