from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st

# ==========================
# KONFIGURATION
# ==========================
VIX_THRESHOLD = 20
MA_WINDOW = 200

REFRESH_DEFAULT_SECONDS = 15 * 60
REFRESH_OPTIONS = {
    "15 min (standard)": 15 * 60,
    "10 min": 10 * 60,
}

CACHE_TTL_SECONDS = 10 * 60  # moderat pga Yahoo throttling

SIGNALS = {
    "EUNL": "EUNL.DE",
    "CNDX": "CNDX.L",
    "SMH":  "SMH.L",
    "WSML": "WSML.L",
    "URNU": "URNU.L",
    "BTC":  "BTC-USD",
    "VIX":  "^VIX",
}

BASE_WEIGHTS = {
    "EUNL (MSCI World)": 0.26,
    "EIMI (EM IMI)": 0.09,          # proxy via EUNL-signal
    "Europa": 0.05,                 # proxy via EUNL-signal
    "US (cap+eqw)": 0.10,           # proxy via EUNL-signal
    "WSML (Small Cap)": 0.07,
    "CNDX (Nasdaq)": 0.08,
    "SMH (Semis)": 0.07,
    "URNU (Uranium)": 0.02,
    "BTC (Crypto proxy)": 0.02,
    "Kontant": 0.02,
}

BUCKET_SIGNAL = {
    "EUNL (MSCI World)": "EUNL",
    "EIMI (EM IMI)": "EUNL",
    "Europa": "EUNL",
    "US (cap+eqw)": "EUNL",
    "WSML (Small Cap)": "WSML",
    "CNDX (Nasdaq)": "CNDX",
    "SMH (Semis)": "SMH",
    "URNU (Uranium)": "URNU",
    "BTC (Crypto proxy)": "BTC",
    "Kontant": None,
}

HIGH_BETA_BUCKETS = {
    "WSML (Small Cap)",
    "CNDX (Nasdaq)",
    "SMH (Semis)",
    "URNU (Uranium)",
    "BTC (Crypto proxy)",
}

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"


# ==========================
# YAHOO V8 CHART (uofficiel)
# ==========================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def yahoo_chart(symbol: str, range_: str = "1y", interval: str = "1d") -> dict:
    # Yahoo v8 chart endpoint omtales som fungerende i praksis uden API-key, men er uofficiel. [1](https://siepr.stanford.edu/publications/policy-brief/us-economy-2026-what-watch)[2](https://www.goldmansachs.com/insights/outlooks/2026-outlooks)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=10)
    r.raise_for_status()
    return r.json()


def parse_close_series(payload: dict) -> pd.DataFrame:
    result = payload.get("chart", {}).get("result")
    if not result:
        return pd.DataFrame(columns=["date", "close"]).set_index("date")

    r0 = result[0]
    ts = r0.get("timestamp", [])
    quote = r0.get("indicators", {}).get("quote", [{}])[0]
    close = quote.get("close", [])

    dates = [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in ts]
    df = pd.DataFrame({"date": dates, "close": close}).dropna()
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
