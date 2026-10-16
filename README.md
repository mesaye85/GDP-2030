# GDP-2030

A small Flask service that projects a country's real GDP (constant 2015 US$) to a target year up to 2050. It starts from the latest World Bank level, compounds IMF World Economic Outlook growth forecasts where they exist, and uses the country's trailing 10-year average growth after the IMF horizon. Every projected year says which of the two it used.

It is an extrapolation with stated rules, not a model of the economy. See [Limitations](#limitations).

## Method

Inputs, fetched at request time:

| Series | Source | Use |
|---|---|---|
| `NY.GDP.MKTP.KD` | World Bank API v2 | GDP level, constant 2015 US$ |
| `NY.GDP.MKTP.KD.ZG` | World Bank API v2 | Annual real growth history, % |
| `NGDP_RPCH` | IMF DataMapper (WEO) | Annual real growth forecasts, % |

With `b` the latest year that has a World Bank level:

```
V_b    = World Bank NY.GDP.MKTP.KD for year b
m, s   = mean and sample standard deviation of NY.GDP.MKTP.KD.ZG over b-9 ... b
g_t    = IMF NGDP_RPCH for year t if published ("imf"), otherwise m ("trend")

V_t    = V_(t-1)    * (1 + g_t / 100)
low_t  = low_(t-1)  * (1 + (g_t - s) / 100)
high_t = high_(t-1) * (1 + (g_t + s) / 100)        with low_b = high_b = V_b
```

- On trend years the band is exactly mean minus/plus one standard deviation. On IMF years the same spread sits around the IMF rate, so the band always contains the central path.
- If the IMF request fails or has no series for the country, every year uses the trend and the response says so in `notes`.
- The trend needs at least 5 of the 10 years to have growth data; otherwise the request returns 422.
- The target year must be after `b` and no later than 2050. The country code must be three letters; the World Bank decides whether it exists.
- The current WEO series runs to 2031, so a 2030 projection today is mostly IMF rates.

## Running it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python main.py                # http://localhost:5001, or $PORT
```

- `GET /` shows a country/year form, a table of the projected path and an inline SVG chart.
- `GET /health` returns `{"status": "ok"}`.
- `POST /predict` with JSON, or `GET /predict?country_code=KEN&year=2035`.

```bash
curl -s localhost:5001/predict -H 'Content-Type: application/json' \
  -d '{"country_code": "KEN", "year": 2035}'
```

Errors come back as `{"error": "..."}` with 400 (bad input or year), 404 (unknown country or no GDP data), 422 (too little growth history) or 502 (World Bank unavailable). An IMF failure is not an error; it switches to the trend.

Example response, trimmed, for the synthetic country used in the tests (mocked data, not real figures):

```json
{
  "country": "TST",
  "country_name": "Testland",
  "units": "constant 2015 US$",
  "base_year": 2025,
  "base_value": 1000000000000.0,
  "target_year": 2033,
  "projected_value": 1137618371805.92,
  "band": {
    "low": 1046576340445.67,
    "high": 1235516255423.42,
    "label": "spread from past growth volatility, not a statistical confidence interval"
  },
  "trend": {"mean_growth": 2.0, "std_growth": 1.0541, "from_year": 2016, "to_year": 2025, "observations": 10},
  "imf_available": true,
  "path": [
    {"year": 2026, "growth_rate": 1.5, "source": "imf", "value": 1015000000000.0,
     "low": 1004459074466.11, "high": 1025540925533.89},
    "... 2027-2031 imf ...",
    {"year": 2032, "growth_rate": 2.0, "source": "trend", "value": 1115312129221.49,
     "low": 1036769460910.73, "high": 1198900717876.23},
    {"year": 2033, "growth_rate": 2.0, "source": "trend", "value": 1137618371805.92,
     "low": 1046576340445.67, "high": 1235516255423.42}
  ],
  "sources": {
    "level": {"indicator": "NY.GDP.MKTP.KD", "url": "https://api.worldbank.org/v2/country/TST/indicator/NY.GDP.MKTP.KD?..."},
    "history_growth": {"indicator": "NY.GDP.MKTP.KD.ZG", "url": "..."},
    "forecast_growth": {"indicator": "NGDP_RPCH", "url": "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/TST"}
  },
  "method": "Rule-based extrapolation, not an economic model. ...",
  "notes": ["IMF forecasts end at 2031; later years use the 10-year trend."]
}
```

## Layout

- `gdp_projection/data.py`: World Bank and IMF requests (with timeouts) and response parsers
- `gdp_projection/projection.py`: pure functions for validation, the trend, compounding and the band
- `gdp_projection/service.py`: fetch, project and build the JSON response
- `gdp_projection/chart.py`: geometry for the SVG chart
- `main.py`: Flask routes; `templates/index.html`: the page

## Tests

```bash
pip install -r requirements-dev.txt
pytest
flake8
```

Every HTTP call in the suite goes through the `responses` library; an unmocked request fails the test. The suite covers compounding against closed-form results, the IMF-to-trend switch at the horizon, the no-IMF fallback, band ordering, year and country validation, both parsers on the real response shapes (World Bank `[meta, rows]` with nulls, DataMapper `values -> NGDP_RPCH -> ISO3 -> year`), and the Flask routes. CI runs flake8 and pytest on Python 3.12.

## Limitations

- It is an extrapolation, not an economic model. It knows nothing about shocks, policy, demographics, exchange rates or inflation beyond what the IMF has already built into its forecasts.
- The trend assumes the last decade repeats. For a 2025 base year the window is 2016-2025, which includes the 2020 contraction and the 2021 rebound; both move the mean and widen the band.
- The band is not a probability interval. It is one standard deviation of past annual growth applied every year and compounded, so it widens mechanically with distance. It says nothing about how often outcomes fall inside it, and on IMF years it does not reflect the IMF's own forecast error.
- Constant 2015 US$ is not the current-dollar (nominal) GDP that is usually quoted, and the two can differ a lot. The numbers are useful for comparing real growth paths, not for matching headline figures.
- IMF growth rates are spliced onto a World Bank level. The two sources can disagree on recent years and revise on different schedules, so the splice is approximate.
- World Bank and IMF revisions change the results. The same request can give different numbers after a WEO release (April and October) or a World Bank data update.
- Country codes mostly match between the two sources. Kosovo and West Bank and Gaza are mapped; regional aggregates such as `WLD` or `EUU` are accepted by the World Bank but not found at the IMF, so they fall back to the trend.
- The parsers were written against live World Bank and IMF responses (October 2026), but the test suite uses mocked responses only, and the service has not been run end to end against the live APIs.

## History

I started this in 2023 as a random forest that predicted GDP from World Bank indicators, and reworked it in 2025. I removed it in 2026 because the approach did not hold up:

- Target leakage. The features included GNI (close to GDP by definition), GDP per capita and GNI per capita, so the model was mostly reconstructing an accounting identity. High scores did not mean forecasting skill.
- Year was not a feature, so the model mapped current indicators to current GDP. The target year in requests had no effect, and the IMF forecasts were merged only onto rows that were never used for training.
- The saved model could not be reproduced: `train_model.py` was empty, the training script pulled live data that has since been revised, and the pickle only loaded with scikit-learn 1.6.1 and NumPy 2.
- The serving code fabricated inputs the model was trained on (for example, growth volatility as 10% of the latest growth rate, and a "GNI per capita" that was actually population), and the feature order did not match the scaler, so `/predict` returned 500.

The old code, model files and the README describing their problems are at [dbed810](https://github.com/mesaye85/GDP-2030/tree/dbed810ea17313b0a73c37ab76174d39a03e092c).
