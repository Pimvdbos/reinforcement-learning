# Data

Raw market data are intentionally not committed to this repository.

The real-market experiment downloads adjusted daily closing prices with `yfinance` for:

`XOM, DAL, NFLX, MCD, CMG, BKNG, DG, LULU, NVDA, INTC, TSLA, CVX, UAL, JPM`

and uses the S&P 500 (`^GSPC`) for market-level features.

Study window: **2010-01-01 to 2025-01-01**.

The downloader explicitly reorders the returned columns to the configured asset list before returns or portfolio actions are constructed. This avoids silent ticker/action mismatches when an upstream data provider changes or alphabetizes its column order.

Yahoo Finance data are fetched at runtime and remain subject to Yahoo's terms and possible historical adjustments.
