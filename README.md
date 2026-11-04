# Top Down Scanner

Live web version of the top down strength system: strongest index, then strongest sector, then the strongest stocks inside it. Same math as the Thinkorswim files in `ToS Top Down Strength Scanner`.

## How it runs

* GitHub Actions runs `scan.py` every 15 minutes during market hours, Monday to Friday. GitHub skips many scheduled runs, so `ticker.yml` starts once at the open, stays running through the session and starts a scan every 15 minutes itself. `scan.yml` also has 3 backup scheduled runs a day.
* Each run downloads daily prices from Yahoo Finance for about 2,000 U.S. stocks (market cap $1B+, price $10+), plus SPY, QQQ, DIA, IWM and the 11 sector ETFs.
* It writes `site/data.json` and publishes the `site` folder to GitHub Pages. The page reloads the data every 5 minutes.
* Your PC does not need to be on.

## What it checks

Strength = 40% of 1 month return + 35% of 3 months + 25% of 6 months.

1. **Market:** SPY above 50 SMA above 200 SMA = uptrend, above 200 only = chop, else downtrend.
2. **Indexes:** ranked by strength, #1 is the leader.
3. **Sectors:** scored against the #1 index. Green = beating it and in its own uptrend. Yellow = beating it but only falling less.4. **Stocks:** each one is scored against its own sector ETF with 8 checks:
   stronger than sector, beats sector over 1 month, beats sector over 3 months, RS line above its 50 day average, close > 21 EMA > 50 SMA > 200 SMA, daily moves correlate 0.50+ with the sector ETF, within 25% of the 52 week high, price $10+ with 1M+ average volume.

5. **Pullback entries:** stocks with 6+ checks, still above the 50 SMA with the 50 above the 200, that triggered on this bar or the one before:
   * **21 EMA:** the current price is within 1% of the 21 EMA, above or below it.
   * **Minor low:** a bar with a lower low than the bars on both sides, then a green bar that closes above that low bar's high.
   * **Inside and up:** an inside bar, then a bar that closes above the inside bar's high, with the current price within 1% of the 21 EMA.
   * Minor low only counts after a real pullback: the pattern low is within 1% of the 21 EMA, or at least 4% under the prior 10 bar high (`EMA21_ZONE`, `MIN_PULLBACK` in `scan.py`).
   * Stop = the pattern low, or the lowest low of the last 2 bars for a 21 EMA entry.

6. **JT Trends** (port of Jeff Thaw's thinkorswim study): MOMO bullish = 8 EMA > 13 EMA > 21 EMA; Trend bullish = MOMO bullish and 21 EMA above the 34 EMA of highs (bearish mirrors it with the 34 EMA of lows). Double green = Trend bullish. Shown as two squares on every stock, a Double green filter on both tables, a trend badge (UPTREND, SIDEWAYS, DOWNTREND) inside every index and sector chip, and the sector Trend view.

## Big 3 Squeeze tab

Open it with the **Big 3 Squeeze** tab or the link `#big3`. It is a port of the Simpler Trading Big 3 studies (squeeze, labels, A+ signals, slingshot, ATR trailing stop) on daily bars, run on every stock scanned:

* **Squeeze:** Bollinger Band (20, 2.0) upper band inside the Keltner Channel (20, 1.75) upper band.
* **+12 Bull Squeeze:** the default, matching the thinkorswim scan: squeeze on and Bull Score 12/12, price $20+, volume 200K+, market cap $10B+.
* **A+ Bull:** 5 > 8 > 21 > 34 EMA stack with 21 EMA above the 50 SMA, 50 SMA above the 200, MACD histogram (24, 52, 9) above zero, close above the 50 SMA and the trailing stop, squeeze on, B3 Score 7+, and 50 SMA, 21 EMA, 200 SMA all rising. A+ Bear is the mirror.
* **Bull squeeze, Slingshot** and their bear versions as in the studies.
* Columns: Bull score, B3 score (bull minus bear), squeeze length, Trend / Structure / Momentum / Trailing stop labels, histogram, distance from the 21 EMA in ATRs, buy zone (within 1.25 ATR above the 21 EMA), stop (21 EMA), target (21 EMA + 3 ATR), R:R, trailing stop, top down setup score, RS and JT.
* A How to trade it section at the bottom explains entry, stop, target and exits.

## Std Dev Channel tab

Open it with the **Std Dev Channel** tab or the link `#stddev`. It ports the thinkorswim StandardDeviationChannel with full range on a 1 year daily chart: a regression line over the last 251 closes, plus or minus the standard deviation of those closes (`CHANNEL_BARS` in `scan.py`).

* Uses the CustStdDevScan filters: price $1.25+, market cap $350M+, volume 500K+ (the scan downloads this wider set; Home and Big 3 still use $1B+ and $10+).
* One table with a Show filter: **Buyers** (-2.5 to -4 deviations, the default), **Sellers** (+2.5 to +4) or Both.
* Options from Cboe delayed quotes: Optionable, and Liquid options = open interest 500+ at the first expiration 20+ days out (MIN_OPT_OI, MIN_OPT_DTE). ATM call spread shown for reference.
* Filters for channel slope (flat, rising, falling, not falling), green candle, price, cap, volume and sector. Columns: deviation today and yesterday, channel slope, bands, middle line (target), distance to it, 20% stop, top down score, RS and JT.
* How to trade it at the bottom uses the deviation channel backtest results.

The universe is every U.S. listed stock in the Nasdaq screener, foreign ADRs included, minus warrants, units, rights, preferreds, notes, funds and ETFs (whole word matching, so names like UnitedHealth are kept).

Sectors come from the S&P 500 list (GICS) where the stock is in it, the Nasdaq screener sector otherwise, and the best correlated sector ETF if neither has one.

## Change the settings

Edit the constants at the top of `scan.py` (`MIN_CAP`, `MIN_PRICE`, `MIN_AVG_VOLUME`, `MIN_CORRELATION`, `MAX_OFF_HIGH`) and push. The next run uses them.

## Run it on your PC

```bash
python scan.py
python -m http.server 5070 --directory site
```

Then open http://localhost:5070

## Notes

* Prices during market hours include today's live bar, so strength moves through the day.
* GitHub's scheduler can run a few minutes late when it is busy.
* Research tool only, not investment advice.
