# GDP-2030

A small Flask API that pulls a country's latest GDP indicators from the World Bank and IMF APIs and runs them through a scikit-learn random forest to produce a GDP estimate.

## Context

I started this in 2023 as a learning project on Replit and reworked it in 2025: a training script that fetches World Bank indicators for 2010-2030, engineers growth features, and trains a `RandomForestRegressor` with time-series cross-validation, plus a Flask service that serves the saved model.

## Stack

- Python 3.10+, Flask, pandas, NumPy, scikit-learn, joblib, requests, matplotlib
- External data: World Bank API v2 (`NY.GDP.MKTP.CD` and related indicators) and IMF DataMapper (`NGDP_RPCH`), fetched live at request time
- `models/gdp_forecast_model.pkl` (7.6 MB, 200 trees) and `models/gdp_forecast_scaler.pkl` (1.3 KB), both saved with scikit-learn 1.6.1 and NumPy 2.x

## Layout

- `python.py` - fetches training data, builds features, trains the model and writes both `.pkl` files to `models/`
- `main.py` - Flask app with `POST /predict`, `GET /health` and `GET /` (matplotlib chart)
- `train_model.py` - empty placeholder; training lives in `python.py`

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install flask pandas "numpy>=2" "scikit-learn==1.6.1" joblib requests matplotlib
python main.py                  # listens on 0.0.0.0, $PORT or 5001
curl localhost:5001/health
curl -X POST localhost:5001/predict -H 'Content-Type: application/json' -d '{"country_code": "USA"}'
```

`requirements.txt` and `pyproject.toml` also list PyTorch, which no code path uses; it is safe to skip. To retrain: `python python.py` (needs network access, writes to `models/`). Accepted country codes are the 15 in `VALID_COUNTRY_CODES` in `main.py`.

## Status

Not working end to end at the current commit. `/health` returns healthy, but `/predict` and the chart on `/` fail as described below. There are no tests.

## Known gaps

- `main.py` `EXPECTED_COLUMNS` lists the nine features in a different order from the one the scaler was fitted on (`GDP_growth, GDP_pcap, GDP_pcap_growth, GNI, GNI_pcap, hist_growth_mean, hist_growth_std, growth_ma_5, growth_volatility`), so `scaler.transform` raises and `/predict` returns 500.
- `GET /` calls `render_template('index.html')`, but there is no `templates/` directory; the old root-level `index.html` was removed in 2025. The route returns 500.
- The `year` field in `/predict` is echoed back but not used; the model predicts the same value for any year. The model maps current indicators to current GDP, so it is not a 2030 forecast.
- Feature engineering differs between training and serving: `python.py` uses World Bank `GDP_pcap_growth` and `GNI_pcap`, while `main.py` computes them as `GDP_pcap / GDP` and `GNI / GDP_pcap`, and fills the rolling-window features with the latest single value.
- The confidence interval in `GET /` refits 100 one-sample random forests on the point prediction, so the interval collapses to that value. `/predict` perturbs inputs by 1 % instead.
- `python.py` labels years after 2023 as forecasts, and training uses `country/all`, which includes regional aggregates alongside countries.
- Dependencies disagree: `pyproject.toml` pins pandas `^1.5.3` while `requirements.txt` asks for `>=2.0.0`; there is no lock file, PyTorch is unused, and the `.pkl` files need NumPy 2.x and scikit-learn 1.6.x (on NumPy 1.x, `load_model()` silently swaps in a dummy model trained on one row).
- `pytest`, `black` and `flake8` are listed as dev dependencies, but there is no test suite or lint config.

## What I'd do next

- Read the feature order from `scaler.feature_names_in_`, add a `templates/index.html`, and cover `/predict` and `/health` with pytest using recorded API responses.
- Share one feature-building module between training and serving, and either use `year` (forecast growth forward) or drop it.
- Choose one dependency file, drop PyTorch, pin scikit-learn to the version the model was saved with, and consider retraining on country rows only.
