# Top Down Scanner

Live web version of the top down strength system: strongest index, then strongest sector, then the strongest stocks inside it. Same math as the Thinkorswim files in `ToS Top Down Strength Scanner`.

## How it runs

* GitHub Actions runs `scan.py` every 15 minutes, Monday to Friday, 9 AM to 5 PM ET.
* Each run downloads daily prices from Yahoo Finance for about 2,000 U.S. stocks (market cap $1B+, price $10+), plus SPY, QQQ, DIA, IWM and the 11 sector ETFs.
* It writes `site/data.json` and publishes the `site` folder to GitHub Pages. The page reloads the data every 5 minutes.
* Your PC does not need to be on.

## What it checks

Strength = 40% of 1 month return + 35% of 3 months + 25% of 6 months.

1. **Market:** SPY above 50 SMA above 200 SMA = uptrend, above 200 only = chop, else downtrend.
2. **Indexes:** ranked by strength, #1 is the leader.
3. **Sectors:** scored against the #1 index. Green = beating it and in its own uptrend. Yellow = beating it but only falling less.
4. **Stocks:** each one is scored against its own sector ETF with 8 checks:
   stronger than sector, beats sector over 1 month, beats sector over 3 months, RS line above its 50 day average, close > 21 EMA > 50 SMA > 200 SMA, daily moves correlate 0.50+ with the sector ETF, within 25% of the 52 week high, price $10+ with 1M+ average volume.

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
