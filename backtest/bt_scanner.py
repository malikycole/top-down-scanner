"""Backtest every scanner setup on 1,755 U.S. stocks worth $2B+ (today's list), 2008 to 2026.

Signals use only data up to the signal bar (no lookahead). Entry = next bar's open. One trade
at a time per stock per setup. Stops checked on the low (exit at the stop, or the open if it gaps
through), targets on the high (exit at the target, or the open if it gaps over); if a bar hits
both, the stop is assumed first. "Close under X" stops exit at the next open after that close.
Every trade also has a time limit. Results: trades, win rate, average return, profit factor,
average bars held, worst trade, split 2008 to 2017 vs 2018 to 2026.

Caveat: the stock list is today's $2B+ names (survivorship), so long results lean optimistic.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PRICES = Path(r"C:\Users\malik\OneDrive\Desktop\Claude\ToS Deviation Buy Sell Scanner\backtest\big_prices.pkl")
EDGE_TOL = {"D": 0.0, "W": 0.01, "M": 0.015, "Y": 0.02}
MAX_BARS = {"D": 10, "W": 20, "M": 40, "Y": 120}


# ---------------------------------------------------------------- trade simulator

def simulate(o, h, l, c, sig, stop, target, close_stop=None, max_bars=40, start=260, moving=False):
    """sig: bool array. stop/target: price arrays fixed at the signal bar (nan = none).
    close_stop: price array, exit next open after a close below it. Fixed at the signal bar
    unless moving=True (then read each bar, like a 21 EMA stop).
    Returns list of (entry_index, return, bars)."""
    out, n, i = [], len(c), start
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = o[i + 1]
        if not (e > 0):
            i += 1
            continue
        sp, tg = stop[i], target[i]
        cs = close_stop[i] if close_stop is not None else np.nan
        if not np.isnan(sp) and sp >= e:      # gapped under the stop: skip
            i += 1
            continue
        if not np.isnan(tg) and tg <= e:      # gapped to the target: skip
            i += 1
            continue
        x, j = None, i + 1
        while j < n:
            if not np.isnan(sp) and l[j] <= sp:
                x = min(sp, o[j])
                break
            if not np.isnan(tg) and h[j] >= tg:
                x = max(tg, o[j])
                break
            lvl = close_stop[j] if (moving and close_stop is not None) else cs
            if not np.isnan(lvl) and c[j] < lvl and j + 1 < n:
                x = o[j + 1]
                j += 1
                break
            if j - i >= max_bars:
                x = c[j]
                break
            j += 1
        if x is None:
            break
        out.append((i + 1, x / e - 1, j - i))
        i = j + 1
    return out


# ---------------------------------------------------------------- indicators

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def true_range(h, l, c):
    pc = c.shift()
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr_trail(o, h, l, c):
    """Modified ATR trailing stop, period 9, factor 2.4, exponential (Big 3 labels study)."""
    hl = h - l
    hilo = np.minimum(hl, 1.5 * hl.rolling(9).mean())
    pc, ph, pl = c.shift(), h.shift(), l.shift()
    href = pd.Series(np.where(l <= ph, h - pc, (h - pc) - 0.5 * (l - ph)), index=c.index)
    lref = pd.Series(np.where(h >= pl, pc - l, (pc - l) - 0.5 * (pl - h)), index=c.index)
    mtr = pd.concat([hilo, href, lref], axis=1).max(axis=1, skipna=False)
    loss = (2.4 * mtr.ewm(span=9, adjust=False, ignore_na=True).mean()).to_numpy()
    cv = c.to_numpy()
    trail, state = np.full(len(cv), np.nan), 0
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
    return trail


def prior_period(df: pd.DataFrame, tf: str):
    """Prior completed period high/low mapped onto each daily bar, plus the period label."""
    if tf == "D":
        return df["High"].shift().to_numpy(), df["Low"].shift().to_numpy(), np.arange(len(df))
    per = df.index.to_period({"W": "W-FRI", "M": "M", "Y": "Y"}[tf])
    g = df.groupby(per).agg(H=("High", "max"), L=("Low", "min")).shift()
    return per.map(g["H"]).to_numpy(float), per.map(g["L"]).to_numpy(float), per


# ---------------------------------------------------------------- setups per stock

def run_stock(df: pd.DataFrame) -> dict[str, list]:
    df = df[~df.index.duplicated()].dropna()
    o_, h_, l_, c_ = (df[k] for k in ("Open", "High", "Low", "Close"))
    o, h, l, c = (x.to_numpy(float) for x in (o_, h_, l_, c_))
    n = len(c)
    res: dict[str, list] = {}
    if n < 300 or np.any(c <= 0):
        return res
    nan = np.full(n, np.nan)
    e5, e8, e13, e21, e34 = (ema(c_, k) for k in (5, 8, 13, 21, 34))
    s50, s200 = c_.rolling(50).mean(), c_.rolling(200).mean()
    E21, S50, S200 = e21.to_numpy(), s50.to_numpy(), s200.to_numpy()
    tr = true_range(h_, l_, c_)

    def add(name, trades):
        res.setdefault(name, []).extend(trades)

    # ---- Home tab pullbacks (base trend: close > 50 SMA > 200 SMA), stop as on the page, target 2R
    base = (c > S50) & (S50 > S200)
    lm1, lm2 = np.roll(l, 1), np.roll(l, 2)
    hm1, hm2 = np.roll(h, 1), np.roll(h, 2)
    near21 = np.abs(c / E21 - 1) <= 0.01
    hi10 = h_.shift(2).rolling(10).max().to_numpy()      # 10 bars before the low bar
    ml = (lm1 < lm2) & (l > lm1) & (c > o) & (c > hm1) & ((lm1 <= np.roll(E21, 1) * 1.01) | (lm1 <= hi10 * 0.96))
    iu = (hm1 <= hm2) & (lm1 >= lm2) & (c > hm1) & near21
    stop2 = np.minimum(l, lm1)
    for name, sig, stp in (("Pullback: 21 EMA", base & near21, stop2),
                           ("Pullback: Minor low", base & ml, lm1),
                           ("Pullback: Inside and up", base & iu, np.minimum(lm1, lm2))):
        risk = c - stp
        tgt = np.where(risk > 0, c + 2 * risk, np.nan)
        add(name, simulate(o, h, l, c, sig & (risk > 0), stp.astype(float), tgt, max_bars=20))

    # ---- Big 3 Squeeze (stop: close under the 21 EMA, target 21 EMA + 3 ATR)
    sma20 = c_.rolling(20).mean()
    sqz = ((sma20 + 2 * c_.rolling(20).std(ddof=0)) < (sma20 + 1.75 * tr.rolling(20).mean())).to_numpy()
    macd = ema(c_, 24) - ema(c_, 52)
    D = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy()
    D1 = np.roll(D, 1)
    sp = ((e5 > e8) & (e8 > e21) & (e21 > e34) & (e21 > s50)).to_numpy()
    trail = atr_trail(o_, h_, l_, c_)
    up50, up21, up200 = (s50.diff() > 0).to_numpy(), (e21.diff() > 0).to_numpy(), (s200.diff() > 0).to_numpy()
    dn50, dn21, dn200 = (s50.diff() < 0).to_numpy(), (e21.diff() < 0).to_numpy(), (s200.diff() < 0).to_numpy()
    sn = ((e5 < e8) & (e8 < e21) & (e21 < e34) & (e21 < s50)).to_numpy()
    E8 = e8.to_numpy()
    bull = (D > 0).astype(int) + up50 + up21 + up200 + sp + (c > S50) + (c > E21) + (c > S200) + (E21 > S50) + (S50 > S200) + (E8 > E21) + (c > trail)
    bear = (D < 0).astype(int) + sn + (c < S50) + (c < E21) + (c < S200) + (E21 < S50) + (S50 < S200) + dn50 + dn21 + dn200 + (E8 < E21) + (c < trail)
    score = bull - bear
    tgt3 = E21 + 3 * tr.rolling(14).mean().to_numpy()
    b3 = {
        "Big 3: +12 Bull Squeeze": sqz & (bull == 12),
        "Big 3: A+ Bull": sp & (S50 > S200) & (D > 0) & (c > S50) & sqz & (c > trail) & (score > 6) & up50 & up21 & up200,
        "Big 3: Bull squeeze": sp & (D > 0) & (c > S50) & sqz,
        "Big 3: Slingshot bull": sqz & sp & (D < 0) & (D > D1) & (c > S50),
    }
    for name, sig in b3.items():
        first = sig & ~np.roll(sig, 1)            # first day the signal appears
        add(name, simulate(o, h, l, c, first, nan, tgt3, close_stop=E21, max_bars=40, moving=True))

    # ---- Std Dev Channel buyers (1 year rolling regression, target the line, 20% stop)
    N = 251
    s = pd.Series(c); t = pd.Series(np.arange(n, dtype=float))
    S1, Sty = s.rolling(N).sum(), (s * t).rolling(N).sum()
    xm, sxx = (N - 1) / 2, N * (N * N - 1) / 12
    b = (Sty - (t - (N - 1)) * S1 - xm * S1) / sxx
    mean = (S1 / N + b * (N - 1 - xm)).to_numpy()
    sd = s.rolling(N).std(ddof=0).to_numpy()
    dev = (c - mean) / sd
    angle = (b * 20).to_numpy() / sd
    zone = (dev <= -2.5) & (dev >= -4)
    first = zone & ~np.roll(zone, 1)
    mean_prev = np.roll(mean, 1)
    for name, sig in (("Std Dev: buyer -2.5 to -4", first),
                      ("Std Dev: buyer, flat channel", first & (np.abs(angle) < 0.15)),
                      ("Std Dev: buyer, flat + green candle", zone & (np.abs(angle) < 0.15) & (c > o))):
        add(name, simulate(o, h, l, c, sig, c * 0.8, mean_prev, max_bars=120))   # target: the line (prior bar)

    # ---- Trendy Edge on each timeframe
    for tf in ("D", "W", "M", "Y"):
        PH, PL, per = prior_period(df, tf)
        r = PH - PL
        le, ue, mid = PH - r * 0.887, PL + r * 0.887, PL + r * 0.5
        bu, be, sbu = PH - r * 0.382, PL + r * 0.382, PH - r * 0.236
        tol = EDGE_TOL[tf]
        lbl3 = c >= ue
        lbl_bull = (c >= bu) & ~lbl3
        if tf == "D":
            same1 = same2 = np.ones(n, bool)
            le_low1, le_low2 = np.roll(le, 1), np.roll(le, 2)     # the low bar's own daily edge
        else:
            pa = np.asarray(per)
            same1 = np.r_[False, pa[1:] == pa[:-1]]
            same2 = np.r_[False, False, pa[2:] == pa[:-2]]
            le_low1 = le_low2 = le
        mlp = (lm1 < lm2) & (l > lm1) & (c > o) & (c > hm1) & same1 & (lm1 <= le_low1 * (1 + tol))
        iup = (hm1 <= hm2) & (lm1 >= lm2) & (c > hm1) & same2 & (lm2 <= le_low2 * (1 + tol))
        ok = ~lbl3 & (c < ue) & ~np.isnan(le)
        rev = (mlp | iup) & ok
        rev_stop = np.where(mlp, lm1, lm2)
        mb = MAX_BARS[tf]
        add(f"Edge {tf}: Edge reversal", simulate(o, h, l, c, rev, rev_stop, ue, max_bars=mb))
        add(f"Edge {tf}: Edge reversal (minor low)", simulate(o, h, l, c, mlp & ok, lm1, ue, max_bars=mb))
        add(f"Edge {tf}: Edge reversal (inside and up)", simulate(o, h, l, c, iup & ok, lm2, ue, max_bars=mb))
        le_today = le if tf != "D" else le
        rej = (l < le_today) & (c > le_today) & ~lbl3 & (c < be)
        add(f"Edge {tf}: Edge rejection", simulate(o, h, l, c, rej, l.copy(), be, max_bars=mb))
        acc = lbl_bull & (l <= bu * (1 + max(tol, 0.003))) & (c >= bu) & (c < ue)
        add(f"Edge {tf}: Accelerator hold", simulate(o, h, l, c, acc, nan, ue, close_stop=mid, max_bars=mb))
        if tf != "D":
            prev_max = pd.Series(h).groupby(np.asarray(per)).cummax().groupby(np.asarray(per)).shift().to_numpy()
            brk = (prev_max > PH) & (l <= PH * (1 + max(tol, 0.005))) & (c > PH)
            first_b = brk & ~np.roll(brk, 1)
            add(f"Edge {tf}: Breakout retest", simulate(o, h, l, c, first_b, nan, nan, close_stop=ue, max_bars=mb))

    # ---- AC/DC Bull (my trade plan: stop close under the monthly 61.8 line, target last month's high)
    PHm, PLm, _ = prior_period(df, "M")
    PHw, PLw, _ = prior_period(df, "W")
    m_up, m_lo = PHm - (PHm - PLm) * 0.382, PHm - (PHm - PLm) * 0.618
    w_up = PHw - (PHw - PLw) * 0.382
    ac = (c >= m_lo) & (c <= m_up) & (c >= w_up)
    first = ac & ~np.roll(ac, 1)
    add("AC/DC Bull", simulate(o, h, l, c, first, nan, PHm, close_stop=m_lo, max_bars=40))

    # keep only dated results
    idx = df.index
    return {k: [(idx[i].year, r_, b_) for i, r_, b_ in v] for k, v in res.items()}


def summarize(trades: list) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t[1] for t in trades])
    bars = np.array([t[2] for t in trades])
    yrs = np.array([t[0] for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")

    def part(mask):
        rr = r[mask]
        return {"n": int(mask.sum()), "win": round(float((rr > 0).mean() * 100), 1) if len(rr) else None,
                "avg": round(float(rr.mean() * 100), 2) if len(rr) else None}
    return {
        "n": int(len(r)), "win": round(float((r > 0).mean() * 100), 1), "avg": round(float(r.mean() * 100), 2),
        "avg_win": round(float(wins.mean() * 100), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean() * 100), 2) if len(losses) else 0.0,
        "pf": round(float(pf), 2) if np.isfinite(pf) else None, "bars": round(float(bars.mean()), 1),
        "worst": round(float(r.min() * 100), 1), "best": round(float(r.max() * 100), 1),
        "old": part(yrs <= 2017), "new": part(yrs >= 2018),
    }


if __name__ == "__main__":
    frames = pickle.loads(PRICES.read_bytes())
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    names = list(frames)[:limit]
    allt: dict[str, list] = {}
    for k, t in enumerate(names):
        for name, tr in run_stock(frames[t]).items():
            allt.setdefault(name, []).extend(tr)
        if k % 100 == 0:
            print(f"{k}/{len(names)}", flush=True)
    out = {name: summarize(tr) for name, tr in allt.items()}
    first = min(f.index[0] for f in frames.values()).date()
    last = max(f.index[-1] for f in frames.values()).date()
    meta = {"stocks": len(names), "from": str(first), "to": str(last)}
    (HERE / "results.json").write_text(json.dumps({"meta": meta, "setups": out}, indent=1))
    for name, s_ in sorted(out.items(), key=lambda kv: -(kv[1].get("pf") or 0)):
        if s_["n"]:
            print(f"{name:42s} n={s_['n']:6d} win={s_['win']:5.1f}% avg={s_['avg']:6.2f}% pf={s_['pf']} bars={s_['bars']} worst={s_['worst']}%")
