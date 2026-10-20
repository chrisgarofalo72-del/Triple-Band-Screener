# Triple EMA Band Stock Screener (reconstruction)

This is a preliminary, configurable **interpretation** of the Facebook trading clip. It is NOT the presenter's verified strategy.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or upload `app.py` and `requirements.txt` to a new GitHub repository, then deploy `app.py` on Streamlit Community Cloud.

## Current interpretation
- Top = 20-period EMA of daily high
- Center = 20-period EMA of daily close
- Bottom = 20-period EMA of daily low
- Search for close crossing the selected EMA within the last N **trading sessions**
- Optional filter: price above a **rising** 150-day SMA
- Filter for share price and 20-day average share volume

**Caution:** The source recording displays a dynamic EMA ribbon but does not establish the precise input series or signal rules. Do not assume this is identical to the speaker's system, and backtest before trading. yfinance data may be delayed or missing and does not constitute live execution quotes.
