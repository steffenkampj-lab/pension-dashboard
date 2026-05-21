from datetime import datetime, timezone
import pandas as pd
import requests
import streamlit as st

# ============================================================
# Konfiguration (ingen globals der ændres!)
# ============================================================
VIX_THRESHOLD = 20
MA_WINDOW = 200

# Standard refresh (sekunder)
REFRESH_DEFAULT_SECONDS = 15 * 60
REFRESH_OPTIONS = {
    "15 min (standard)": 15 * 60,
    "10 min": 10 * 60,
}

# Cache TTL: hold moderat (Yahoo kan rate-limite ved for hyppige kald)
# Vi rydder cache manuelt når brugeren skifter 10/15 min.
CACHE_TTL_SECONDS = 10 * 60

# Yahoo tickers (som du har valgt)
SIGNALS = {
    "EUNL": "EUNL.DE",
    "CNDX": "CNDX.L",
    "SMH": "SMH.L",
    "WSML": "WSML.L",
    "URNU": "URNU.L",
    "BTC": "BTC-USD",
    "VIX": "^VIX",
}

# Strategiske base-vægte (bucket-setup som i din Excel)
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

# Hvilket signal driver hver bucket
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

# Yahoo v8 chart (uofficiel). User-Agent hjælper ofte mod blokering. [1](https://siepr.stanford.edu/publications/policy-brief/us-economy-2026-what-watch)[2](https://www.goldmansachs.com/insights/outlooks/2026-outlooks)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"


# ============================================================
# Yahoo client (v8 chart endpoint)
# ============================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def yahoo_chart(symbol: str, range_: str = "2y", interval: str = "1d") -> dict:
    """
    Henter data fra Yahoo v8 chart endpoint (uofficiel).
    Brug moderat refresh for at reducere risiko for throttling. [1](https://siepr.stanford.edu/publications/policy-brief/us-economy-2026-what-watch)[2](https://www.goldmansachs.com/insights/outlooks/2026-outlooks)
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_close_series(payload: dict) -> pd.DataFrame:
    """Returnerer daglig close-serie med date index."""
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
    return df


def parse_meta_snapshot(payload: dict) -> dict:
    """Trækker live-ish regularMarketPrice fra meta (hvis tilgængeligt). [1](https://siepr.stanford.edu/publications/policy-brief/us-economy-2026-what-watch)[3](https://www.stlouisfed.org/on-the-economy/2025/dec/professional-forecasters-past-performance-outlook-2026)"""
    result = payload.get("chart", {}).get("result")
    if not result:
        return {"regularMarketPrice": None, "currency": None, "exchangeName": None, "regularMarketTime": None}
    meta = result[0].get("meta", {})
    return {
        "regularMarketPrice": meta.get("regularMarketPrice"),
        "currency": meta.get("currency"),
        "exchangeName": meta.get("exchangeName"),
        "regularMarketTime": meta.get("regularMarketTime"),
    }


# ============================================================
# Beregninger
# ============================================================
def ma(series: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    return series.rolling(window=window).mean()


def signal_from_close(close: pd.Series) -> pd.Series:
    """Signal = 1.0 hvis close > MA200, ellers 0.5."""
    m = ma(close, MA_WINDOW)
    return (close > m).astype(float).replace({0.0: 0.5})


def latest_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[-1] if len(s2) else None


def compute_buckets(latest_signals: dict, latest_vix: float | None) -> pd.DataFrame:
    """
    Aggressiv timing:
    - high beta: 0.35x ved trend ned, 1.05x ved trend op
    - VIX>20 => 0.55x ekstra for high beta
    - cap: 0.4x–1.4x af base
    - kontant 2% fast, resterende normaliseres
    """
    trend_down = 0.35
    trend_up = 1.05
    vix_mult = 0.55

    rows = []
    for bucket, base_w in BASE_WEIGHTS.items():
