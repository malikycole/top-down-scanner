"""Backtest the Moxie tab setups on the same 1,755 stocks, 2008 to 2026, and merge them into results.json.

Daily chart Moxie = MACD (12, 26, 9) histogram of WEEKLY closes. To avoid lookahead, each day uses the
week as it stood that day: last week's finished EMAs updated with today's close (what thinkorswim shows
live), never the week's final close.

Setups (entry next open, one trade at a time):
  Moxie: daily trampoline        weekly Moxie up, close back over the 50 SMA after closing under it
                                 (study state machine). Stop: lowest low of the dip. Exit: next open after
                                 weekly Moxie closes a day below zero. Time limit 40 days.
  Moxie: trampoline, 50 SMA stop same entry, stop a close back under the 50 SMA instead of the dip low.
  Moxie: trampoline + trend      same as the first, with the trend label green (8 EMA > 21 EMA > 50 SMA).
  Moxie: cross up                weekly Moxie crosses above zero. Exit when it closes back under zero. 60 days.
The hourly trampoline is not tested: free hourly history only goes back about 2 years.
"""
import json
import pickle

import numpy as np
import pandas as pd

from bt_scanner import HERE, PRICES, simulate, summarize


def live_weekly_moxie(c: pd.Series) -> np.ndarray:
    weeks = c.resample("W-FRI").last().dropna()
    a12, a26, a9 = 2 / 13, 2 / 27, 2 / 10
    e12 = weeks.ewm(span=12, adjust=False).mean()
    e26 = weeks.ewm(span=26, adjust=False).mean()
    sig = (e12 - e26).ewm(span=9, adjust=False).mean()
    per = c.index.to_period("W-FRI")
    wk = weeks.index.to_period("W-FRI")
    prev = lambda s: per.map(pd.Series(s.shift().values, index=wk)).to_numpy(float)   # last week's finished value
    p12, p26, psig = prev(e12), prev(e26), prev(sig)
    cv = c.to_numpy(float)
    l12 = a12 * cv + (1 - a12) * p12
    l26 = a26 * cv + (1 - a26) * p26
    vc = l12 - l26
    return (vc - (a9 * vc + (1 - a9) * psig)) * 3


def tramp_signals(up, c, l, mid):
    """Bar by bar trampoline state (study logic, bull side) -> first bar of each confirmed trampoline and
    the lowest low of the dip before it."""
    n = len(c)
    sig, dip = np.zeros(n, bool), np.full(n, np.nan)
    code, low = 0, np.nan
    for i in range(n):
        m = mid[i]
        if np.isnan(m) or np.isnan(up[i]):
            code, low = 0, np.nan
            continue
        u = up[i] > 0
        if 1000 <= code < 2000:
            if not u:
                code = 0
            elif c[i] > m:
                code = 2001
                sig[i], dip[i] = True, min(low, l[i])
            else:
                low = min(low, l[i])
        elif 2000 <= code < 3000:
            if not u:
                code = 0
            elif c[i] <= m:
                code, low = 1000, l[i]
            else:
                code = 0 if code - 2000 >= 1 else code + 1
        elif u and c[i] < m:
            code, low = 1000, l[i]
        else:
            code = 0
    return sig, dip


def run(df: pd.DataFrame) -> dict:
    df = df[~df.index.duplicated()].dropna()
    c_ = df["Close"]
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    if len(c) < 400 or np.any(c <= 0):
        return {}
    mox = live_weekly_moxie(c_)
    s50 = c_.rolling(50).mean().to_numpy()
    e8, e21 = c_.ewm(span=8, adjust=False).mean().to_numpy(), c_.ewm(span=21, adjust=False).mean().to_numpy()
    trend = (e8 > e21) & (e21 > s50)
    nan = np.full(len(c), np.nan)
    below0 = np.where(mox < 0, c + 1e9, np.nan)           # "close under X" that is true whenever Moxie < 0
    exit_mox_or_50 = np.where(mox < 0, c + 1e9, s50)     # out on Moxie < 0 or a close under the 50 SMA
    sig, dip = tramp_signals(mox, c, l, s50)
    res = {
        "Moxie: daily trampoline": simulate(o, h, l, c, sig, dip, nan, close_stop=below0, max_bars=40, moving=True),
        "Moxie: trampoline, 50 SMA stop": simulate(o, h, l, c, sig, nan, nan, close_stop=exit_mox_or_50, max_bars=40, moving=True),
        "Moxie: trampoline + trend": simulate(o, h, l, c, sig & trend, dip, nan, close_stop=below0, max_bars=40, moving=True),
    }
    cross = (mox > 0) & (np.roll(mox, 1) < 0)
    res["Moxie: cross up"] = simulate(o, h, l, c, cross, nan, nan, close_stop=below0, max_bars=60, moving=True)
    idx = df.index
    return {k: [(idx[i].year, r, b) for i, r, b in v] for k, v in res.items()}


if __name__ == "__main__":
    frames = pickle.loads(PRICES.read_bytes())
    allt = {}
    for k, (t, df) in enumerate(frames.items()):
        for name, tr in run(df).items():
            allt.setdefault(name, []).extend(tr)
        if k % 200 == 0:
            print(f"{k}/{len(frames)}", flush=True)
    res = json.loads((HERE / "results.json").read_text())
    for name, tr in allt.items():
        res["setups"][name] = summarize(tr)
        s = res["setups"][name]
        print(f"{name:34s} n={s['n']:6d} win={s['win']}% avg={s['avg']}% pf={s['pf']} bars={s['bars']} worst={s['worst']}% "
              f"old {s['old']['win']}%/{s['old']['avg']}% new {s['new']['win']}%/{s['new']['avg']}%")
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
