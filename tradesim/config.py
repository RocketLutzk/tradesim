"""Що торгуємо і скільки коштує угода."""

# key: (клас активу, назва для людей, символ Binance або None, тікер Yahoo)
ASSETS = {
    # Крипта (торгується 24/7)
    "BTC": ("crypto", "Bitcoin", "BTCUSDT", "BTC-USD"),
    "ETH": ("crypto", "Ethereum", "ETHUSDT", "ETH-USD"),
    "SOL": ("crypto", "Solana", "SOLUSDT", "SOL-USD"),
    "BNB": ("crypto", "BNB", "BNBUSDT", "BNB-USD"),
    "XRP": ("crypto", "XRP", "XRPUSDT", "XRP-USD"),
    "DOGE": ("crypto", "Dogecoin", "DOGEUSDT", "DOGE-USD"),
    # Акції та фонди США (торгуються ~7 год на день у будні)
    "SPY": ("stocks", "S&P 500 (SPY)", None, "SPY"),
    "QQQ": ("stocks", "Nasdaq 100 (QQQ)", None, "QQQ"),
    "AAPL": ("stocks", "Apple", None, "AAPL"),
    "NVDA": ("stocks", "Nvidia", None, "NVDA"),
    "TSLA": ("stocks", "Tesla", None, "TSLA"),
    "MSFT": ("stocks", "Microsoft", None, "MSFT"),
    # Сировина
    "GOLD": ("commodities", "Золото (ф'ючерс)", None, "GC=F"),
    "OIL": ("commodities", "Нафта WTI (ф'ючерс)", None, "CL=F"),
    # Форекс
    "EURUSD": ("forex", "EUR/USD", None, "EURUSD=X"),
    "GBPUSD": ("forex", "GBP/USD", None, "GBPUSD=X"),
    "USDJPY": ("forex", "USD/JPY", None, "USDJPY=X"),
}

CLASS_NAMES = {
    "crypto": "Крипта",
    "stocks": "Акції",
    "commodities": "Сировина",
    "forex": "Форекс",
}

# Витрати на ОДНУ сторону угоди (купівля або продаж): комісія + спред/проковзування.
# Свідомо трохи песимістично: краще недооцінити стратегію, ніж переоцінити.
COSTS = {
    "crypto": 0.0010 + 0.0005,   # 0.10% комісія біржі + 0.05% проковзування
    "stocks": 0.0003 + 0.0002,
    "commodities": 0.0003 + 0.0002,
    "forex": 0.0001 + 0.0001,
}

TRAIN_SHARE = 0.7        # 70% історії — на підбір параметрів, 30% — на чесну перевірку
START_CAPITAL = 1000.0   # віртуальний депозит на кожну пару «стратегія × актив»
