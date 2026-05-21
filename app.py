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

# Cache TTL (moderat pga Yahoo throttling)
CACHE_TTL_SECONDS = 10 * 60

# Yahoo tickers (som du bruger)
SIGNALS = {
    "EUNL": "EUNL.DE",
    "CNDX": "CNDX.L",
    "SMH":  "SMH.L",
    "WSML": "WSML.L",
    "URNU": "URNU.L",
    "BTC":  "BTC-USD",
    "VIX":  "^VIX",
}

# Base-vægte (bucket setup)
BASE_WEIGHTS = {
    "EUNL (MSCI World)": 0.26,
    "EIMI (EM IMI)": 0.09,          # proxy via EUNL signal
    "Europa": 0.05,                 # proxy via EUNL signal
    "US (cap+eqw)": 0.10,           # proxy via EUNL signal
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
# DATAHENTNING (Yahoo v8 chart)
# ==========================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def yahoo_chart(symbol: str, range_: str = "1y", interval: str = "1d") -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
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
    return df


def parse_meta_snapshot(payload: dict) -> dict:
    result = payload.get("chart", {}).get("result")
    if not result:
        return {"regularMarketPrice": None, "currency": None, "exchangeName": None}
    meta = result[0].get("meta", {})
    return {
        "regularMarketPrice": meta.get("regularMarketPrice"),
        "currency": meta.get("currency"),
        "exchangeName": meta.get("exchangeName"),
    }


# ==========================
# BEREGNINGER
# ==========================
def ma(series: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    return series.rolling(window=window).mean()


def signal_from_close(close: pd.Series) -> pd.Series:
    m = ma(close, MA_WINDOW)
    return (close > m).astype(float).replace({0.0: 0.5})


def latest_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if len(s) else None


def compute_buckets(latest_signals: dict, latest_vix: float | None) -> pd.DataFrame:
    trend_down = 0.35
    trend_up = 1.05
    vix_mult = 0.55

    rows = []
    for bucket, base_w in BASE_WEIGHTS.items():
        if bucket == "Kontant":
            rows.append({
                "Bucket": bucket, "Base": base_w, "Signal": None,
                "TrendMult": 1.0, "VIXMult": 1.0, "Capped": base_w
            })
            continue

        sig_name = BUCKET_SIGNAL[bucket]
        sig_val = latest_signals.get(sig_name)

        if bucket in HIGH_BETA_BUCKETS:
            t_mult = 1.0 if sig_val is None else (trend_up if sig_val >= 1 else trend_down)
        else:
            t_mult = 1.0 if (sig_val is None or sig_val >= 1) else 0.7

        if bucket in HIGH_BETA_BUCKETS and latest_vix is not None:
            v_mult = vix_mult if latest_vix > VIX_THRESHOLD else 1.0
        else:
            v_mult = 1.0

        capped = max(base_w * 0.4, min(base_w * 1.4, base_w * t_mult * v_mult))

        rows.append({
            "Bucket": bucket, "Base": base_w, "Signal": sig_val,
            "TrendMult": t_mult, "VIXMult": v_mult, "Capped": capped
        })

    df = pd.DataFrame(rows)

    cash = float(df.loc[df["Bucket"] == "Kontant", "Base"].iloc[0])
    invested_share = 1 - cash
    sum_capped_ex_cash = df["Capped"].sum() - cash

    df["Tactical"] = df.apply(
        lambda r: cash if r["Bucket"] == "Kontant" else invested_share * r["Capped"] / sum_capped_ex_cash,
        axis=1
    )
    df["Deviation"] = df["Tactical"] - df["Base"]
    df["Action"] = df["Deviation"].apply(lambda x: "Ingen" if abs(x) < 0.005 else ("Øg" if x > 0 else "Reducér"))
    return df


# ==========================
# STREAMLIT UI (vis noget med det samme)
# ==========================
st.set_page_config(page_title="Pension Dashboard", layout="wide")
st.title("Executive Dashboard – MA200 + VIX (daglige closes) + Yahoo live snapshot")

st.caption(f"App started: {datetime.utcnow().isoformat()} UTC")

# Refresh state (ingen globals)
if "refresh_seconds" not in st.session_state:
    st.session_state["refresh_seconds"] = REFRESH_DEFAULT_SECONDS

with st.sidebar:
    st.header("Indstillinger")
    choice = st.selectbox("Refresh interval", list(REFRESH_OPTIONS.keys()), index=0)
    new_seconds = REFRESH_OPTIONS[choice]
    if new_seconds != st.session_state["refresh_seconds"]:
        st.session_state["refresh_seconds"] = new_seconds
        st.cache_data.clear()

