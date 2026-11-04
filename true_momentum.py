"""True Momentum system (Simpler Trading, Eric Purdy) ported from the course scan studies:
ST_HiLoPRO_X_Scan, ST_TrendOscillatorPRO_X_Scan, ST_StructureScoreX and ST_SwingLevelsPRO.
Daily settings. The Trend Oscillator runs on weekly bars ("Daily-Use Wk Aggregation": 30 / 120),
which is also what the daily chart study and the Structure Score show."""
import numpy as np
import pandas as pd

# HiLo PRO X daily thresholds: SlowD crossing above any OS line is a buy arrow, below any OB a sell
_OB = [63.38, 65.35, 68.71, 72.5, 76.04, 58.35, 50.898, 44.8, 40.12, 34.95, 32.1]
_OS = [69.1, 66.9, 61.4, 36, 40.8, 70.3, 42, 75.3, 48.2, 55.7, 46]


def _stoch_d(h: pd.Series, l: pd.Series, c: pd.Series, k: int, d: int, slow: int = 3) -> pd.Series:
    """thinkorswim StochasticFull FullD. The study passes `if AverageType.EXPONENTIAL == 1 then SIMPLE`,
    which resolves to SIMPLE, so both smoothings are simple averages."""
    lo = l.rolling(k).min()
    rng = h.rolling(k).max() - lo
    fast_k = ((c - lo) / rng * 100).where(rng != 0, 0.0)
    full_k = fast_k.rolling(slow).mean()
    return full_k.rolling(d).mean()


def hilo(frame: pd.DataFrame) -> dict:
    h, l, c = frame["High"], frame["Low"], frame["Close"]
    sd = _stoch_d(h, l, c, 99, 3).to_numpy()
    sdx = _stoch_d(h, l, c, 13, 34).to_numpy()
    p = np.roll(sd, 1)
    p[0] = np.nan
    ch = sd - p
    up = np.zeros(len(sd), bool)
    dn = np.zeros(len(sd), bool)
    for lv in _OS:
        up |= (sd > lv) & (p <= lv)
    for lv in _OB:
        dn |= (sd < lv) & (p >= lv)
    up |= ((sd > 78.9) & (sd < 94.5) & (ch > 2.12)) | ((sd > 70) & (sd < 100) & (ch > 3.75))
    up |= ((sd < 40) & (sd > 25.6) & (ch > 3.1)) | ((sd < 40) & (sd > 18.4) & (ch > 4.4))
    up |= ((sd < 50) & (sd > 17.6) & (ch > 5)) | ((sd < 27) & (sd > 24) & (ch > 2.6))
    dn |= ((sd > 78.6) & (sd < 87.4) & (ch < -2.1)) | ((sd > 65) & (sd < 100) & (ch < -4.15))
    dn |= ((sd > 70.6) & (sd < 100) & (ch < -3.56)) | ((sd > 84.9) & (sd < 92.9) & (ch < -3))
    dn |= ((sd < 29) & (sd > 16) & (ch < -3)) | ((sd < 50) & (sd > 10) & (ch < -5))
    thrust, last, since = np.zeros(len(sd), int), 0, None
    for i in range(len(sd)):
        if up[i]:
            last = 1
        elif dn[i]:
            last = -1
        thrust[i] = last
    # bars since the last arrow in the current thrust direction
    k = len(sd) - 1
    while k >= 0 and not (up[k] if thrust[-1] == 1 else dn[k]):
        k -= 1
    since = len(sd) - 1 - k if k >= 0 else None
    return {"buy10": bool(up[-10:].any()), "sell10": bool(dn[-10:].any()), "buy": bool(up[-1]), "sell": bool(dn[-1] and not up[-1]), "thrust": int(thrust[-1]),
            "hl_d": round(float(sd[-1]), 1), "hl_x": round(float(sdx[-1]), 1),
            "hl_conf": bool((thrust[-1] == 1 and sd[-1] > sdx[-1]) or (thrust[-1] == -1 and sd[-1] < sdx[-1])),
            "hl_since": since, "hl_ext": bool(sdx[-1] > 79 or sdx[-1] < 21)}


def trend_osc(weekly_close: pd.Series, l1: int = 30, l2: int = 120) -> dict:
    d = weekly_close.diff()
    a1 = d.ewm(span=l1, adjust=False).mean()
    a2 = d.abs().ewm(span=l1, adjust=False).mean()
    to = 50 * ((a1 / a2).where(a2 != 0, 0.0) + 1)
    em = to.ewm(span=l2, adjust=False).mean()
    t, e, t1, e1 = float(to.iloc[-1]), float(em.iloc[-1]), float(to.iloc[-2]), float(em.iloc[-2])
    cross_up, cross_dn = t > e and t1 <= e1, t < e and t1 >= e1
    result = 1 if cross_up else -1 if cross_dn else 2 if t > e else -2 if t < e else 0
    delta = abs(t - e)
    score = 0
    if delta > 2.4:
        score += 3 if t > e else -3
        score += 7 if (t > 52 and e > 51.5) else -7 if (t < 48 and e < 48.5) else 0
        score += 2 if (t > 52 and e < 51.5) else -2 if (t < 48 and e > 48.5) else 0
    return {"to": round(t, 2), "to_ema": round(e, 2), "to_res": result, "to_up": bool(e > e1),
            "to_score": score, "to_ext": bool(t > 69 or t < 31)}


def atr_trail(frame: pd.DataFrame, period: int = 9, factor: float = 2.9) -> np.ndarray:
    """thinkorswim ATRTrailingStop, modified, exponential average, first trade long."""
    h, l, c = frame["High"], frame["Low"], frame["Close"]
    prev_c = c.shift()
    hl = h - l
    hilo_ = np.minimum(hl, 1.5 * hl.rolling(period).mean())
    href = np.where(l <= h.shift(), h - prev_c, (h - prev_c) - 0.5 * (l - h.shift()))
    lref = np.where(h >= l.shift(), prev_c - l, (prev_c - l) - 0.5 * (l.shift() - h))
    mtr = pd.concat([hilo_, pd.Series(href, index=c.index), pd.Series(lref, index=c.index)], axis=1).max(axis=1, skipna=False)
    loss = (factor * mtr.ewm(span=period, adjust=False, ignore_na=True).mean()).to_numpy()
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


def swing_levels(frame: pd.DataFrame, n: int = 25) -> dict:
    """ST_SwingLevelsPRO: the last swing low / high with n lower (higher) bars on both sides."""
    lo, hi = frame["Low"].to_numpy(), frame["High"].to_numpy()
    sl = sh = None
    for i in range(n, len(lo) - n):
        if lo[i] < lo[i + 1:i + 1 + n].min() and lo[i] <= lo[i - n:i].min():
            sl = float(lo[i])
        if hi[i] > hi[i + 1:i + 1 + n].max() and hi[i] >= hi[i - n:i].max():
            sh = float(hi[i])
    return {"sw_lo": round(sl, 2) if sl else None, "sw_hi": round(sh, 2) if sh else None}


def true_momentum(frame: pd.DataFrame, weekly_close: pd.Series, monthly_close: pd.Series | None = None) -> dict:
    c = frame["Close"]
    e21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
    s50, s200 = float(c.rolling(50).mean().iloc[-1]), float(c.rolling(200).mean().iloc[-1])
    px = float(c.iloc[-1])
    if np.isnan(s200):
        ma = 15 if e21 > s50 else -15
    else:
        ma = (15 if e21 > s50 > s200 else 5 if e21 > s200 > s50 else 3 if s50 > e21 > s200 else 0) + \
             (-15 if s200 > s50 > e21 else -5 if s50 > s200 > e21 else -3 if s200 > e21 > s50 else 0)
    trail = float(atr_trail(frame)[-1])
    atr_v = 5 if px > trail else -5
    hl, to = hilo(frame), trend_osc(weekly_close)
    hl_score = (5 if hl["thrust"] == 1 else -5 if hl["thrust"] == -1 else 0) + \
               (5 if hl["thrust"] == 1 and hl["hl_conf"] else -5 if hl["thrust"] == -1 and hl["hl_conf"] else 0)
    total = to["to_score"] + hl_score + ma + atr_v
    label = "Max Bull" if total >= 40 else "Bull" if total >= 20.1 else "Max Bear" if total <= -40 else "Bear" if total <= -20.1 else "Neutral"
    vol = frame["Volume"]
    # Max Bull scan pieces: monthly Trend Oscillator ("Weekly-Use Mo Aggregation": 10 / 21) on monthly bars
    tom = trend_osc(monthly_close, 10, 21) if monthly_close is not None and len(monthly_close) >= 30 else None
    return {**hl, **to, **swing_levels(frame),
            "tm_score": int(total), "tm_label": label, "tm_price": ma + atr_v, "tm_trend": to["to_score"], "tm_thrust": hl_score,
            "atr_trail": round(trail, 2), "to_m_res": tom["to_res"] if tom else None,
            "to_m": tom["to"] if tom else None, "to_m_ema": tom["to_ema"] if tom else None,
            "stack": 1 if e21 > s50 > s200 else -1 if e21 < s50 < s200 else 0, "above_trail": bool(px > trail), "vol_ema20": int(vol.ewm(span=20, adjust=False).mean().iloc[-1])}
