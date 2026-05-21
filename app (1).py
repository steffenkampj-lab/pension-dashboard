from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, Tuple, Optional

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Pension Executive Dashboard (MA200 + VIX) — med konkret ordreliste
# ============================================================
# Designmål:
# 1) UI vises med det samme (ingen "blank skærm")
# 2) Ingen global-variabler der ændres (ingen global/syntax fejl)
# 3) Parallel datahentning + tydelig status
# 4) KONKRET ORDRELISTE i DKK baseret på:
#    - Taktiske vægte fra modellen
#    - Din porteføljeværdi (default 600.000 kr)
#    - Dine nuværende vægte (default = Base, men kan redigeres i app)
# ============================================================

# ----------------------------
# Modelparametre
# ----------------------------
MA_WINDOW = 200
VIX_THRESHOLD = 20

REFRESH_OPTIONS = {
    "15 min (standard)": 15 * 60,
    "10 min": 10 * 60,
}
DEFAULT_REFRESH = REFRESH_OPTIONS["15 min (standard)"]

# Caching TTL: moderat (Yahoo kan rate-limite ved hyppige kald)
CACHE_TTL_SECONDS = 10 * 60

# Yahoo tickers (signal inputs)
SIGNALS = {
    "EUNL": "EUNL.DE",
    "CNDX": "CNDX.L",
    "SMH": "SMH.L",
    "WSML": "WSML.L",
    "URNU": "URNU.L",
    "BTC": "BTC-USD",
    "VIX": "^VIX",
}

# Strategiske base-vægte (bucket setup)
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

# Ticker/produkt mapping for ordreliste (kan ændres i UI)
DEFAULT_ORDER_MAPPING = {
    "EUNL (MSCI World)": "EUNL.DE (iShares Core MSCI World)",
    "EIMI (EM IMI)": "EIMI (iShares Core EM IMI)",
    "Europa": "MSCI Europe ETF (din) / IE…",  # placeholder
    "US (cap+eqw)": "SXR8+EWSP (US cap+equal weight)",
    "WSML (Small Cap)": "WSML.L (iShares World Small Cap)",
    "CNDX (Nasdaq)": "CNDX.L (iShares Nasdaq 100)",
    "SMH (Semis)": "SMH.L (VanEck Semiconductors)",
    "URNU (Uranium)": "URNU.L (Global X Uranium)",
    "BTC (Crypto proxy)": "BTC-USD proxy/ETP",
    "Kontant": "Kontant",
}

# Yahoo v8 chart: User-Agent hjælper ofte, da endpoint kan afvise default clients.
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

# Vi bruger ~400d daglige data: nok til MA200 og mindre payload end 2y.
RANGE_FOR_MA200 = "400d"
INTERVAL = "1d"


# ----------------------------
# Yahoo client (v8 chart)
# ----------------------------

def _requests_get_json(url: str, timeout: int = 12, retries: int = 2, backoff: float = 0.8) -> dict:
    last_exc = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            if r.status_code in (403, 429):
                raise RuntimeError(f"Yahoo HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if i < retries:
                time.sleep(backoff * (i + 1))
            else:
                raise last_exc


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def yahoo_chart(symbol: str, range_: str = RANGE_FOR_MA200, interval: str = INTERVAL) -> dict:
    """Henter JSON fra Yahoo v8 chart endpoint (uofficiel)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}"
    return _requests_get_json(url)


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


# ----------------------------
# Beregninger
# ----------------------------

def ma(series: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    return series.rolling(window=window).mean()


def signal_from_close(close: pd.Series) -> pd.Series:
    m = ma(close, MA_WINDOW)
    return (close > m).astype(float).replace({0.0: 0.5})


def latest_non_null(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def compute_buckets(latest_signals: dict, latest_vix: Optional[float]) -> pd.DataFrame:
    # Aggressiv profil
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

        # Trend-multiplier
        if bucket in HIGH_BETA_BUCKETS:
            t_mult = 1.0 if sig_val is None else (trend_up if sig_val >= 1 else trend_down)
        else:
            t_mult = 1.0 if (sig_val is None or sig_val >= 1) else 0.7

        # VIX-multiplier
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


def compute_orders(bucket_df: pd.DataFrame, current_weights: pd.Series, portfolio_value: float, mapping: dict, min_trade_dkk: float) -> pd.DataFrame:
    """Ordreliste i DKK: (taktisk vægt - nuværende vægt) * porteføljeværdi."""
    df = bucket_df.copy()
    df["Current"] = df["Bucket"].map(current_weights.to_dict()).fillna(df["Base"])
    df["TargetDKK"] = (df["Tactical"] * portfolio_value).round(0)
    df["CurrentDKK"] = (df["Current"] * portfolio_value).round(0)
    df["OrderDKK"] = (df["TargetDKK"] - df["CurrentDKK"]).round(0)
    df["Produkt"] = df["Bucket"].map(mapping).fillna(df["Bucket"])

    def _side(x):
        if abs(x) < 1:
            return ""
        return "KØB" if x > 0 else "SÆLG"

    df["Side"] = df["OrderDKK"].apply(_side)
    df["AbsOrder"] = df["OrderDKK"].abs()
    df = df.sort_values("AbsOrder", ascending=False)

    # filter små handler
    df = df[df["AbsOrder"] >= min_trade_dkk]

    return df[["Side", "Produkt", "Bucket", "Current", "Tactical", "OrderDKK"]]


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Pension Dashboard", layout="wide")

if "refresh_seconds" not in st.session_state:
    st.session_state["refresh_seconds"] = DEFAULT_REFRESH

if "order_mapping" not in st.session_state:
    st.session_state["order_mapping"] = DEFAULT_ORDER_MAPPING.copy()

st.title("Executive Dashboard – MA200 + VIX (daglige closes) + Yahoo live snapshot")
st.caption(f"App started: {datetime.now(timezone.utc).isoformat()} UTC")

with st.sidebar:
    st.header("Indstillinger")
    choice = st.selectbox("Refresh interval", list(REFRESH_OPTIONS.keys()), index=0)
    new_seconds = REFRESH_OPTIONS[choice]
    if new_seconds != st.session_state["refresh_seconds"]:
        st.session_state["refresh_seconds"] = new_seconds
        st.cache_data.clear()

    st.subheader("Ordreliste")
    portfolio_value = st.number_input("Porteføljeværdi (DKK)", min_value=0.0, value=600000.0, step=10000.0)
    min_trade = st.number_input("Min. handel (DKK)", min_value=0.0, value=2000.0, step=500.0)

    st.caption("Tip: Ordreliste bliver mest præcis, hvis du indtaster dine nuværende vægte i tabellen.")

# Auto-refresh (browser reload)
st.markdown(
    f"""
    <script>
    setTimeout(function(){{ window.location.reload(); }}, {st.session_state['refresh_seconds']*1000});
    </script>
    """,
    unsafe_allow_html=True,
)

st.info("Loader data… (hvis Yahoo blokerer/rate-limiter, vises fejl pr. ticker)")

status = st.empty()
progress = st.progress(0)

prices: Dict[str, pd.DataFrame] = {}
signals_latest: Dict[str, Optional[float]] = {}
snapshots: Dict[str, dict] = {}
errors: Dict[str, str] = {}

items = list(SIGNALS.items())


def fetch_one(name: str, sym: str) -> Tuple[str, pd.DataFrame, Optional[float], dict]:
    payload = yahoo_chart(sym, range_=RANGE_FOR_MA200, interval=INTERVAL)
    df = parse_close_series(payload)
    snap = parse_meta_snapshot(payload)
    sig_val = None
    if not df.empty:
        sig_val = latest_non_null(signal_from_close(df["close"]))
    return name, df, sig_val, snap


with ThreadPoolExecutor(max_workers=5) as ex:
    futures = {ex.submit(fetch_one, name, sym): (name, sym) for name, sym in items}
    done = 0
    total = len(futures)

    for fut in as_completed(futures):
        name, sym = futures[fut]
        try:
            nm, df, sig_val, snap = fut.result()
            prices[nm] = df
            signals_latest[nm] = sig_val
            snapshots[nm] = snap
        except Exception as e:
            prices[name] = pd.DataFrame(columns=["close"])
            signals_latest[name] = None
            snapshots[name] = {"regularMarketPrice": None, "currency": None, "exchangeName": None}
            errors[name] = str(e)

        done += 1
        progress.progress(int(done / total * 100))
        status.info(f"Henter data… {done}/{total} færdig")

progress.empty()
status.success("Opdatering færdig.")

if errors:
    st.warning("Nogle tickere kunne ikke hentes (midlertidigt).")
    st.json(errors)

# VIX regime
latest_vix = None
if "VIX" in prices and not prices["VIX"].empty:
    latest_vix = latest_non_null(prices["VIX"]["close"])

regime = "" if latest_vix is None else ("RED (risk-off)" if latest_vix > VIX_THRESHOLD else "GREEN (risk-on)")

bucket_df = compute_buckets(signals_latest, latest_vix)

# KPI
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Regime", regime if regime else "(mangler VIX)")
with k2:
    st.metric("Seneste VIX (close)", f"{latest_vix:.2f}" if latest_vix is not None else "-")
with k3:
    eunl_sig = signals_latest.get("EUNL")
    st.metric("Global trend (EUNL signal)", f"{eunl_sig:.1f}" if eunl_sig is not None else "-")
with k4:
    need_trade = (regime == "RED (risk-off)") or (bucket_df["Deviation"].abs().max() > 0.015)
    st.metric("SKAL JEG HANDLE?", "JA" if need_trade else "NEJ")

st.divider()

# Live snapshot
st.subheader("Live snapshot (Yahoo meta.regularMarketPrice)")
snap_rows = []
for key in ["EUNL", "CNDX", "SMH", "WSML", "URNU", "BTC", "VIX"]:
    snap = snapshots.get(key, {})
    snap_rows.append({
        "Signal": key,
        "YahooSymbol": SIGNALS[key],
        "Live pris": snap.get("regularMarketPrice"),
        "Valuta": snap.get("currency"),
        "Exchange": snap.get("exchangeName"),
    })

st.dataframe(pd.DataFrame(snap_rows), width='stretch')

# Buckets
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Buckets: strategisk vs taktisk")
    show = bucket_df[["Bucket", "Base", "Tactical", "Deviation", "Action"]].copy()
    st.dataframe(show.style.format({"Base": "{:.2%}", "Tactical": "{:.2%}", "Deviation": "{:+.2%}"}), width='stretch')

with c2:
    st.subheader("Top 5 afvigelser")
    top = bucket_df[bucket_df["Bucket"] != "Kontant"].copy()
    top["AbsDev"] = top["Deviation"].abs()
    top = top.sort_values("AbsDev", ascending=False).head(5)
    st.dataframe(top[["Bucket", "Deviation", "Action"]].style.format({"Deviation": "{:+.2%}"}), width='stretch')

st.divider()

# ----------------------------
# Ordreliste
# ----------------------------
st.subheader("Konkret ordreliste (DKK)")

st.markdown(
    """
**Sådan bruges ordreliste:**
- Appen beregner din **taktiske mål-vægt** pr. bucket.
- Du indtaster dine **nuværende vægte** (hvis du vil have præcis ordreliste). Default = Base.
- Ordre (DKK) = (taktisk vægt − nuværende vægt) × porteføljeværdi.

> Hvis du ikke indtaster nuværende vægte, antager appen at du ligger på **Base** (strategisk). Så bliver ordreliste en “delta fra base til taktisk”.
"""
)

# Editor til nuværende vægte
current_df = pd.DataFrame({
    "Bucket": list(BASE_WEIGHTS.keys()),
    "CurrentWeight": [BASE_WEIGHTS[b] for b in BASE_WEIGHTS.keys()],
})

st.caption("Redigér kun CurrentWeight (som decimal, fx 0.08 = 8%).")
current_edit = st.data_editor(current_df, num_rows='fixed', hide_index=True, width='stretch')

# Mapping editor
with st.expander("(Valgfrit) Ret produktnavne i ordreliste"):
    map_df = pd.DataFrame({
        "Bucket": list(DEFAULT_ORDER_MAPPING.keys()),
        "Produkt": [st.session_state["order_mapping"].get(b, b) for b in DEFAULT_ORDER_MAPPING.keys()],
    })
    map_edit = st.data_editor(map_df, num_rows='fixed', hide_index=True, width='stretch')
    # gem mapping
    st.session_state["order_mapping"] = dict(zip(map_edit["Bucket"], map_edit["Produkt"]))

# beregn ordreliste
current_weights = pd.Series(current_edit["CurrentWeight"].values, index=current_edit["Bucket"].values)
orders = compute_orders(bucket_df, current_weights, float(portfolio_value), st.session_state["order_mapping"], float(min_trade))

if orders.empty:
    st.success("Ingen handler over min. handel – du kan evt. sænke min. handel eller tjekke afvigelser.")
else:
    # Format
    orders_show = orders.copy()
    orders_show["Current"] = orders_show["Current"].map(lambda x: f"{x:.2%}")
    orders_show["Tactical"] = orders_show["Tactical"].map(lambda x: f"{x:.2%}")
    orders_show["OrderDKK"] = orders_show["OrderDKK"].map(lambda x: f"{int(x):,}".replace(",", "."))
    st.dataframe(orders_show, width='stretch')

st.caption("⚠️ Yahoo v8 chart endpoint er uofficiel og kan ændre sig. Hvis Yahoo blokerer, vil du se fejl pr. ticker ovenfor.")
