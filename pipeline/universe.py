"""Asset universe and factor definitions shared by every pipeline step."""

# Factors: the "levers" a user can pull in a scenario.
# Each factor is a liquid ETF, so its history is real market data.
FACTORS = {
    "SPY": "US stock market (S&P 500)",
    "XLK": "Technology sector",
    "XLF": "Financials sector",
    "XLE": "Energy sector",
    "XLV": "Healthcare sector",
    "XLY": "Consumer discretionary sector",
    "XLP": "Consumer staples sector",
    "XLI": "Industrials sector",
    "XLU": "Utilities sector",
    "TLT": "Long-term US Treasury bonds (rates)",
    "GLD": "Gold",
}

# Assets a user can put in a portfolio: ticker -> (name, sector factor)
ASSETS = {
    "AAPL": ("Apple", "XLK"),
    "MSFT": ("Microsoft", "XLK"),
    "NVDA": ("NVIDIA", "XLK"),
    "AMD": ("AMD", "XLK"),
    "INTC": ("Intel", "XLK"),
    "GOOGL": ("Alphabet (Google)", "XLK"),
    "META": ("Meta Platforms", "XLK"),
    "NFLX": ("Netflix", "XLY"),
    "AMZN": ("Amazon", "XLY"),
    "TSLA": ("Tesla", "XLY"),
    "MCD": ("McDonald's", "XLY"),
    "DIS": ("Disney", "XLY"),
    "JPM": ("JPMorgan Chase", "XLF"),
    "BAC": ("Bank of America", "XLF"),
    "V": ("Visa", "XLF"),
    "XOM": ("ExxonMobil", "XLE"),
    "CVX": ("Chevron", "XLE"),
    "JNJ": ("Johnson & Johnson", "XLV"),
    "PFE": ("Pfizer", "XLV"),
    "UNH": ("UnitedHealth", "XLV"),
    "KO": ("Coca-Cola", "XLP"),
    "PG": ("Procter & Gamble", "XLP"),
    "WMT": ("Walmart", "XLP"),
    "BA": ("Boeing", "XLI"),
    "CAT": ("Caterpillar", "XLI"),
    "NEE": ("NextEra Energy", "XLU"),
    # Funds / other asset classes
    "SPY": ("S&P 500 ETF", "SPY"),
    "QQQ": ("Nasdaq-100 ETF", "XLK"),
    "TLT": ("20+ Year Treasury Bond ETF", "TLT"),
    "GLD": ("Gold ETF", "GLD"),
    "BTC-USD": ("Bitcoin", "SPY"),
}

# Training window for honest evaluation (model never sees anything after this)
TRAIN_START = "2010-01-01"
TRAIN_END = "2019-12-31"

# Held-out stress episodes, all AFTER the training window.
EPISODES = {
    "covid_2020": ("COVID-19 crash", "2020-02-19", "2020-03-23"),
    "bear_2022": ("2022 rate-hike bear market", "2022-01-03", "2022-10-12"),
    "carry_2024": ("Aug 2024 yen carry-trade unwind", "2024-07-16", "2024-08-05"),
    "tariff_2025": ("April 2025 tariff shock", "2025-04-02", "2025-04-08"),
}

DATA_START = "2010-01-01"
DATA_END = "2026-11-14"
