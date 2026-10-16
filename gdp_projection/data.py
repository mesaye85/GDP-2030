"""Fetching and parsing World Bank (API v2) and IMF DataMapper series.

World Bank responses look like ``[meta, [row, ...]]``: each row has a
``date`` (the year, as a string) and a ``value`` that may be null. An unknown
country code comes back as ``[{"message": [{"id": "120", ...}]}]``.

IMF DataMapper responses look like
``{"values": {"NGDP_RPCH": {"USA": {"2025": 1.9, ...}, ...}}}``. The parser
looks up the requested ISO3 key, so it works whether the response carries
one country or all of them.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import requests

from .errors import NotFound, UpstreamError

log = logging.getLogger(__name__)

WB_BASE = "https://api.worldbank.org/v2"
IMF_BASE = "https://www.imf.org/external/datamapper/api/v1"

WB_LEVEL = "NY.GDP.MKTP.KD"  # GDP, constant 2015 US$
WB_GROWTH = "NY.GDP.MKTP.KD.ZG"  # GDP growth, annual %, constant prices
IMF_GROWTH = "NGDP_RPCH"  # WEO real GDP growth, annual %

WB_INVALID_VALUE = "120"  # World Bank message id for an unrecognised country code

# Places where the World Bank and IMF DataMapper use different three-letter codes,
# as candidate keys in order. The DataMapper country list has both KOS and UVK for
# Kosovo, so both are tried. Any other code is assumed to match; a miss falls back
# to the trend and says so.
IMF_CODE_ALIASES = {
    "XKX": ("UVK", "KOS"),  # Kosovo
    "PSE": ("WBG",),  # West Bank and Gaza
}
TIMEOUT = (5, 20)  # seconds: connect, read
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "GDP-2030 (https://github.com/mesaye85/GDP-2030)",
}


@dataclass(frozen=True)
class Series:
    indicator: str
    url: str
    values: Dict[int, float]
    country_name: Optional[str] = None


@dataclass(frozen=True)
class ImfSeries:
    url: str
    values: Dict[int, float] = field(default_factory=dict)
    problem: Optional[str] = None  # why no forecasts are available, if none are


def world_bank_url(country: str, indicator: str, start: int, end: int) -> str:
    return f"{WB_BASE}/country/{country}/indicator/{indicator}?format=json&date={start}:{end}&per_page=100"


def imf_url(country: str, indicator: str = IMF_GROWTH) -> str:
    return f"{IMF_BASE}/{indicator}/{country}"


def _is_wb_message(payload) -> bool:
    return (
        isinstance(payload, list)
        and len(payload) >= 1
        and isinstance(payload[0], dict)
        and "message" in payload[0]
    )


def parse_world_bank(payload, country: str):
    """Return ``(values_by_year, country_name)`` from a World Bank v2 payload.

    Null values are skipped. Raises NotFound for an unrecognised country code
    and UpstreamError for any other error message or an unexpected shape.
    """
    if _is_wb_message(payload):
        messages = [m for m in payload[0].get("message") or [] if isinstance(m, dict)]
        if any(str(m.get("id")) == WB_INVALID_VALUE for m in messages):
            raise NotFound(f"Unknown country code '{country}': the World Bank API does not recognise it.")
        text = "; ".join(str(m.get("value", "")) for m in messages) or "unknown error"
        raise UpstreamError(f"World Bank API error: {text}")

    if not (isinstance(payload, list) and len(payload) == 2 and isinstance(payload[0], dict)):
        raise UpstreamError("Unexpected World Bank response shape.")

    meta, rows = payload
    if isinstance(meta.get("pages"), int) and meta["pages"] > 1:
        log.warning("World Bank response for %s has %s pages; only the first was read", country, meta["pages"])

    values: Dict[int, float] = {}
    name = None
    for row in rows or []:  # rows is null when nothing matches the query
        if not isinstance(row, dict):
            continue
        if name is None:
            name = (row.get("country") or {}).get("value")
        try:
            year = int(row["date"])
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("value") is None:
            continue
        values[year] = float(row["value"])
    return values, name


def parse_imf(payload, country: str, indicator: str = IMF_GROWTH) -> Dict[int, float]:
    """Return ``{year: growth_percent}`` for one country from a DataMapper payload.

    Returns an empty dict when the indicator or the country is absent.
    """
    try:
        series = payload["values"][indicator][country]
    except (KeyError, TypeError):
        return {}
    if not isinstance(series, dict):
        return {}

    values: Dict[int, float] = {}
    for year, value in series.items():
        if value is None:
            continue
        try:
            values[int(year)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _get_json(url: str, source: str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise UpstreamError(f"{source} request failed ({exc.__class__.__name__}).") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise UpstreamError(f"{source} returned a non-JSON response (HTTP {response.status_code}).") from exc
    return response.status_code, payload


def fetch_world_bank(country: str, indicator: str, start: int, end: int) -> Series:
    url = world_bank_url(country, indicator, start, end)
    status, payload = _get_json(url, "World Bank")
    if status >= 400 and not _is_wb_message(payload):
        raise UpstreamError(f"World Bank returned HTTP {status}.")
    values, name = parse_world_bank(payload, country)
    return Series(indicator=indicator, url=url, values=values, country_name=name)


def fetch_imf(country: str) -> ImfSeries:
    """Fetch IMF WEO real growth for one country. Never raises: the forecasts are optional."""
    codes = IMF_CODE_ALIASES.get(country, (country,))
    url = imf_url(codes[0])
    try:
        status, payload = _get_json(url, "IMF")
    except UpstreamError as exc:
        log.warning("IMF fetch failed for %s: %s", country, exc)
        return ImfSeries(url=url, problem=str(exc))
    if status >= 400:
        return ImfSeries(url=url, problem=f"IMF returned HTTP {status}.")
    for code in codes:
        values = parse_imf(payload, code)
        if values:
            return ImfSeries(url=url, values=values)
    return ImfSeries(url=url, problem=f"IMF DataMapper has no {IMF_GROWTH} series for {'/'.join(codes)}.")
