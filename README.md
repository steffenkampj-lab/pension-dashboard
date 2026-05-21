# Streamlit – Pension Executive Dashboard (MA200 + VIX + Yahoo live snapshot)

Denne app bygger et dashboard til din model:
- Daglige closes (interval=1d) fra Yahoo v8 chart
- MA200 + signal (1 / 0,5)
- VIX-regime (GREEN/RED ved tærskel 20)
- Executive KPI: “SKAL JEG HANDLE?”

## Kør lokalt
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (enkelt)
- Streamlit Community Cloud (hurtigst)
- Azure App Service / Container Apps (enterprise)
- Render/Fly.io

## Tickere
Konfigureres i `SIGNALS` i `app.py`.

## Bemærk
Yahoo v8 chart endpoint er uofficiel og kan ændre sig. Brug moderat refresh (10–15 min) og overvej backup-kilde.
