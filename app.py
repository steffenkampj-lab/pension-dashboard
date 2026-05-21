import time
from datetime import datetime, timezone
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Konfiguration
# -----------------------------
REFRESH_SECONDS = 15 * 60  # 15 min standard
VIX_THRESHOLD = 20
MA_WINDOW = 200

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

HIGH_BETA_BUCKETS = {"WSML (Small Cap)", "CNDX (Nasdaq)", "SMH (Semis)", "URNU (Uranium)", "BTC (Crypto proxy)"}

# -----------------------------
# Yahoo v8 chart client
# -----------------------------
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def yahoo_chart(symbol: str, range_: str = "2y", interval: str = "1d") -> dict:
    """Henter data fra Yahoo v8 chart endpoint (uofficiel)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_ohlc_close(payload: dict) -> pd.DataFrame:
    result = payload.get("chart", {}).get("result")
    if not result:
        return pd.DataFrame(columns=["date", "close"]).set_index("date")

    r0 = result[0]
    ts = r0.get("timestamp", [])
    quote = r0.get("indicators", {}).get("quote", [{}])[0]
    close = quote.get("close", [])

    # Konverter unix seconds -> date
    dates = [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in ts]
    df = pd.DataFrame({"date": dates, "close": close}).dropna()
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return df


def parse_meta_snapshot(payload: dict) -> dict:
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


# -----------------------------
# Beregninger
# -----------------------------

def ma(series: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    return series.rolling(window=window).mean()


def signal_from_close(close: pd.Series) -> pd.Series:
    m = ma(close)
    return pd.Series((close > m).astype(float)).replace({0.0: 0.5})


def latest_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[-1] if len(s2) else None


def compute_buckets(latest_signals: dict, latest_vix: float) -> pd.DataFrame:
    # multipliers
    trend_down = 0.35
    trend_up = 1.05
    vix_mult = 0.55

    rows = []
    for bucket, base_w in BASE_WEIGHTS.items():
        if bucket == "Kontant":
            rows.append({
                "Bucket": bucket,
                "Base": base_w,
                "Signal": None,
                "TrendMult": 1.0,
                "VIXMult": 1.0,
                "Capped": base_w,
            })
            continue

        sig_name = BUCKET_SIGNAL[bucket]
        sig_val = latest_signals.get(sig_name)

        # Trend multiplier
        if bucket in HIGH_BETA_BUCKETS:
            if sig_val is None:
                t_mult = 1.0
            else:
                t_mult = trend_up if sig_val >= 1 else trend_down
        else:
            # lav-beta proxy: mild nedskalering
            t_mult = 1.0 if (sig_val is None or sig_val >= 1) else 0.7

        # VIX multiplier (kun high beta)
        if bucket in HIGH_BETA_BUCKETS and latest_vix is not None:
            v_mult = vix_mult if latest_vix > VIX_THRESHOLD else 1.0
        else:
            v_mult = 1.0

        capped = max(base_w * 0.4, min(base_w * 1.4, base_w * t_mult * v_mult))

        rows.append({
            "Bucket": bucket,
            "Base": base_w,
            "Signal": sig_val,
            "TrendMult": t_mult,
            "VIXMult": v_mult,
            "Capped": capped,
        })

    df = pd.DataFrame(rows)

    cash = float(df.loc[df["Bucket"] == "Kontant", "Base"].iloc[0])
    invested_share = 1 - cash
    sum_capped_ex_cash = df["Capped"].sum() - cash

    df["Tactical"] = df.apply(
        lambda r: cash if r["Bucket"] == "Kontant" else invested_share * r["Capped"] / sum_capped_ex_cash,
        axis=1,
    )
    df["Deviation"] = df["Tactical"] - df["Base"]
    df["Action"] = df["Deviation"].apply(lambda x: "Ingen" if abs(x) < 0.005 else ("Øg" if x > 0 else "Reducér"))
    return df


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Pension Dashboard", layout="wide")

# Auto-refresh
st.markdown("""<script>
setTimeout(function(){ window.location.reload(); }, %d);
</script>""" % (REFRESH_SECONDS*1000), unsafe_allow_html=True)

st.title("Executive Dashboard – MA200 + VIX (daglige closes) + Yahoo live snapshot")

with st.sidebar:
    st.header("Indstillinger")
    refresh = st.selectbox("Refresh interval", ["15 min (standard)", "10 min"], index=0)
    if refresh.startswith("10"):
        st.cache_data.clear()
        global REFRESH_SECONDS
        REFRESH_SECONDS = 10*60
    st.caption("Bemærk: Yahoo endpoint er uofficiel. Hold refresh moderat.")

# Fetch data
prices = {}
signals_latest = {}
snapshots = {}

for name, sym in SIGNALS.items():
    try:
        payload = yahoo_chart(sym, range_="2y", interval="1d")
        df = parse_ohlc_close(payload)
        prices[name] = df
        sig_series = signal_from_close(df["close"]) if len(df) else pd.Series(dtype=float)
        signals_latest[name] = latest_non_null(sig_series)
        snap = parse_meta_snapshot(payload)
        snapshots[name] = snap
    except Exception as e:
        prices[name] = pd.DataFrame(columns=["close"])  # empty
        signals_latest[name] = None
        snapshots[name] = {"regularMarketPrice": None, "currency": None, "exchangeName": None, "regularMarketTime": None}

latest_vix = None
if len(prices.get("VIX", pd.DataFrame())):
    latest_vix = latest_non_null(prices["VIX"]["close"])

regime = "" if latest_vix is None else ("RED (risk-off)" if latest_vix > VIX_THRESHOLD else "GREEN (risk-on)")

bucket_df = compute_buckets(signals_latest, latest_vix)

# Executive KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Regime", regime if regime else "(mangler VIX)")
with col2:
    st.metric("Seneste VIX (close)", f"{latest_vix:.2f}" if latest_vix is not None else "-")
with col3:
    eunl_sig = signals_latest.get("EUNL")
    st.metric("Global trend (EUNL signal)", f"{eunl_sig:.1f}" if eunl_sig is not None else "-")
with col4:
    need_trade = (regime == "RED (risk-off)") or (bucket_df["Deviation"].abs().max() > 0.015)
    st.metric("SKAL JEG HANDLE?", "JA" if need_trade else "NEJ")

st.divider()

# Live snapshot table
snap_rows = []
for k in ["EUNL", "CNDX", "SMH", "WSML", "URNU", "BTC", "VIX"]:
    sym = SIGNALS[k]
    snap = snapshots.get(k, {})
    snap_rows.append({
        "Signal": k,
        "YahooSymbol": sym,
        "Live pris (regularMarketPrice)": snap.get("regularMarketPrice"),
        "Valuta": snap.get("currency"),
        "Exchange": snap.get("exchangeName"),
    })

st.subheader("Live snapshot (Yahoo meta.regularMarketPrice)")
st.dataframe(pd.DataFrame(snap_rows), use_container_width=True)

# Bucket table + top deviations
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Buckets: strategisk vs taktisk")
    df_show = bucket_df[["Bucket","Base","Tactical","Deviation","Action"]].copy()
    st.dataframe(df_show.style.format({"Base":"{:.2%}","Tactical":"{:.2%}","Deviation":"{:+.2%}"}), use_container_width=True)

with c2:
    st.subheader("Top 5 afvigelser")
    top = bucket_df.loc[bucket_df["Bucket"] != "Kontant"].copy()
    top["AbsDev"] = top["Deviation"].abs()
    top = top.sort_values("AbsDev", ascending=False).head(5)
    st.dataframe(top[["Bucket","Deviation","Action"]].style.format({"Deviation":"{:+.2%}"}), use_container_width=True)

# Charts
st.subheader("Pris + MA200 (seneste 260 handelsdage)")
chart_cols = st.columns(3)
plot_list = ["EUNL","CNDX","SMH","WSML","URNU","BTC","VIX"]
for i, name in enumerate(plot_list):
    with chart_cols[i % 3]:
        df = prices.get(name)
        if df is None or df.empty:
            st.write(f"{name}: ingen data")
            continue
        df2 = df.copy()
        df2["MA200"] = ma(df2["close"], MA_WINDOW)
        df2 = df2.tail(260)
        st.line_chart(df2[["close","MA200"]], height=220)
        st.caption(f"{name} ({SIGNALS[name]})")

st.caption("⚠️ Bemærk: Yahoo v8 chart endpoint er uofficiel; brug moderat refresh og hav evt. backup-kilde.")
