"""Builders for World Bank and IMF payloads in their real response shapes, plus a synthetic country."""

import re

import responses

from gdp_projection import data


def wb_payload(country_iso3, country_name, indicator, indicator_name, values):
    """A World Bank API v2 response: ``[meta, rows]``, newest year first, nulls allowed."""
    rows = [
        {
            "indicator": {"id": indicator, "value": indicator_name},
            "country": {"id": country_iso3[:2], "value": country_name},
            "countryiso3code": country_iso3,
            "date": str(year),
            "value": values[year],
            "unit": "",
            "obs_status": "",
            "decimal": 1,
        }
        for year in sorted(values, reverse=True)
    ]
    meta = {"page": 1, "pages": 1, "per_page": 100, "total": len(rows), "sourceid": "2", "lastupdated": "2026-10-08"}
    return [meta, rows]


# What the World Bank returns for an unrecognised country code.
WB_INVALID = [
    {"message": [{"id": "120", "key": "Invalid value", "value": "The provided parameter value is not valid"}]}
]


def imf_payload(series_by_country, indicator=data.IMF_GROWTH):
    """An IMF DataMapper v1 response: ``{"values": {indicator: {ISO3: {year: value}}}}``."""
    return {
        "values": {
            indicator: {
                code: {str(year): value for year, value in series.items()}
                for code, series in series_by_country.items()
            }
        },
        "api": {"version": "1", "output-method": "json"},
    }


def wb_url_re(country, indicator):
    return re.compile(rf"^{re.escape(data.WB_BASE)}/country/{country}/indicator/{re.escape(indicator)}\?.*$")


# A synthetic country with round numbers. Base year 2025 (the 2026 row is null,
# as the live API returns for the current year). Growth over 2016-2025 alternates
# 1 % and 3 %: mean 2.0, sample sd sqrt(10/9). IMF forecasts run to 2031.
BASE_YEAR = 2025
BASE_VALUE = 1_000_000_000_000.0
LEVELS = {year: BASE_VALUE / (1.02 ** (BASE_YEAR - year)) for year in range(1996, BASE_YEAR + 1)}
LEVELS[2026] = None
GROWTH = {year: (1.0 if year % 2 else 3.0) for year in range(1997, BASE_YEAR + 1)}
GROWTH[2026] = None
IMF_RATES = {year: 1.5 for year in range(1980, 2032)}


def register_country(rsps, country="TST", name="Testland", levels=None, growth=None, imf=None, imf_status=200):
    """Mock the three requests a projection makes for ``country``."""
    levels = LEVELS if levels is None else levels
    growth = GROWTH if growth is None else growth
    rsps.add(responses.GET, wb_url_re(country, data.WB_LEVEL),
             json=wb_payload(country, name, data.WB_LEVEL, "GDP (constant 2015 US$)", levels))
    rsps.add(responses.GET, wb_url_re(country, data.WB_GROWTH),
             json=wb_payload(country, name, data.WB_GROWTH, "GDP growth (annual %)", growth))
    imf_body = imf_payload({country: IMF_RATES if imf is None else imf, "KEN": {2030: 5.0}})
    rsps.add(responses.GET, data.imf_url(country), json=imf_body, status=imf_status)
