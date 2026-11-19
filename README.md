# Cloud Backtest (Method 2 – Ichimoku cloud scale-in)

Same strategies and conditions as the live paper-trading app, but run on **historical candles** for a **From date → To date**.
No live feed, no orders – Angel One login is used only to download past candles.

Run: `python -m streamlit run backtest_m2.py`

1. **Backtest Builder** – folder, exchange, **From date / To date**, symbols, timeframe, capital, lots, entry/exit, method → *Save backtest files* (one file per combination).
2. **My Backtests** – open a folder → *Run backtest* → summary table (trades, win %, realized / open P&L, return %, max drawdown) + CSV downloads. Click a file for equity curve, cloud chart, signals, closed trades, bar log.

A position still open on the To date is shown as "Open at end date" (marked to the last close).
Backtest files are stored in `backtests/<ClientID>/<folder>/` – back them up from *Account* on Streamlit Cloud.
