"""Parsers against the real response shapes, and the fetchers against mocked HTTP."""

from urllib.parse import parse_qs, urlparse

import pytest
import requests
import responses

from gdp_projection import data
from gdp_projection.errors import NotFound, UpstreamError

from tests.fakes import WB_INVALID, imf_payload, wb_payload, wb_url_re

# Trimmed from a live https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.KD.ZG response.
WB_USA_GROWTH = [
    {"page": 1, "pages": 1, "per_page": 3, "total": 3, "sourceid": "2", "lastupdated": "2026-10-08"},
    [
        {"indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"},
         "country": {"id": "US", "value": "United States"}, "countryiso3code": "USA",
         "date": "2026", "value": None, "unit": "", "obs_status": "", "decimal": 1},
        {"indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"},
         "country": {"id": "US", "value": "United States"}, "countryiso3code": "USA",
         "date": "2025", "value": 2.16138195623856, "unit": "", "obs_status": "", "decimal": 1},
        {"indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"},
         "country": {"id": "US", "value": "United States"}, "countryiso3code": "USA",
         "date": "2024", "value": 2.79318715363841, "unit": "", "obs_status": "", "decimal": 1},
    ],
]


def test_parse_world_bank_real_shape_skips_nulls():
    values, name = data.parse_world_bank(WB_USA_GROWTH, "USA")
    assert values == {2025: 2.16138195623856, 2024: 2.79318715363841}
    assert name == "United States"


def test_parse_world_bank_null_rows_list():
    meta = {"page": 0, "pages": 0, "per_page": 100, "total": 0, "sourceid": None, "lastupdated": "2026-10-08"}
    assert data.parse_world_bank([meta, None], "USA") == ({}, None)


def test_parse_world_bank_unknown_country_is_not_found():
    with pytest.raises(NotFound, match="XYZ"):
        data.parse_world_bank(WB_INVALID, "XYZ")


def test_parse_world_bank_other_message_is_upstream_error():
    payload = [{"message": [{"id": "175", "key": "Invalid format", "value": "The indicator was not found."}]}]
    with pytest.raises(UpstreamError, match="indicator was not found"):
        data.parse_world_bank(payload, "USA")


@pytest.mark.parametrize("payload", [{}, [], [1, 2], "oops", None])
def test_parse_world_bank_unexpected_shape(payload):
    with pytest.raises(UpstreamError):
        data.parse_world_bank(payload, "USA")


def test_parse_imf_real_datamapper_shape():
    # The live endpoint nests values -> indicator -> ISO3 -> year and may return many countries.
    payload = {
        "values": {
            "NGDP_RPCH": {
                "SDN": {"2024": -19.3, "2025": 0, "2026": 7},
                "USA": {"1980": -0.3, "2025": 1.9, "2026": 2.0, "2031": 2.1},
            }
        },
        "api": {"version": "1", "output-method": "json"},
    }
    assert data.parse_imf(payload, "USA") == {1980: -0.3, 2025: 1.9, 2026: 2.0, 2031: 2.1}
    assert data.parse_imf(payload, "SDN")[2025] == 0.0


@pytest.mark.parametrize("payload", [
    {"values": {"NGDP_RPCH": {"SDN": {"2025": 1.0}}}},  # country absent
    {"values": {"OTHER": {"USA": {"2025": 1.0}}}},  # indicator absent
    {"values": {}},
    {},
    [],
    None,
])
def test_parse_imf_missing_returns_empty(payload):
    assert data.parse_imf(payload, "USA") == {}


def test_parse_imf_skips_null_and_bad_values():
    payload = imf_payload({"USA": {2025: 1.9, 2026: None, 2027: "n/a"}})
    assert data.parse_imf(payload, "USA") == {2025: 1.9}


def test_old_flat_imf_shape_is_not_misread():
    # What the previous code assumed: values[country][year]. It must not parse as data.
    assert data.parse_imf({"values": {"USA": {"2025": 1.9}}}, "USA") == {}


def test_fetch_world_bank_builds_series(rsps):
    rsps.add(responses.GET, wb_url_re("USA", data.WB_GROWTH), json=WB_USA_GROWTH)
    series = data.fetch_world_bank("USA", data.WB_GROWTH, 1996, 2026)
    assert series.values[2025] == pytest.approx(2.1613819)
    assert series.url == data.world_bank_url("USA", data.WB_GROWTH, 1996, 2026)
    query = parse_qs(urlparse(rsps.calls[0].request.url).query)
    assert query["format"] == ["json"]
    assert query["date"] == ["1996:2026"]


def test_fetch_world_bank_unknown_country(rsps):
    rsps.add(responses.GET, wb_url_re("XYZ", data.WB_LEVEL), json=WB_INVALID)
    with pytest.raises(NotFound):
        data.fetch_world_bank("XYZ", data.WB_LEVEL, 1996, 2026)


def test_fetch_world_bank_http_error(rsps):
    rsps.add(responses.GET, wb_url_re("USA", data.WB_LEVEL), body="<html>busy</html>", status=503)
    with pytest.raises(UpstreamError):
        data.fetch_world_bank("USA", data.WB_LEVEL, 1996, 2026)


def test_fetch_world_bank_timeout(rsps):
    rsps.add(responses.GET, wb_url_re("USA", data.WB_LEVEL), body=requests.exceptions.ConnectTimeout())
    with pytest.raises(UpstreamError, match="ConnectTimeout"):
        data.fetch_world_bank("USA", data.WB_LEVEL, 1996, 2026)


def test_fetch_passes_a_timeout(monkeypatch):
    seen = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return wb_payload("USA", "United States", data.WB_LEVEL, "GDP", {2025: 1.0})

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(data.requests, "get", fake_get)
    data.fetch_world_bank("USA", data.WB_LEVEL, 1996, 2026)
    assert seen["timeout"] == data.TIMEOUT


def test_fetch_imf_ok(rsps):
    rsps.add(responses.GET, data.imf_url("USA"), json=imf_payload({"USA": {2026: 2.0}}))
    imf = data.fetch_imf("USA")
    assert imf.values == {2026: 2.0} and imf.problem is None


@pytest.mark.parametrize("imf_key", ["UVK", "KOS"])
def test_fetch_imf_maps_world_bank_codes(rsps, imf_key):
    rsps.add(responses.GET, data.imf_url("UVK"), json=imf_payload({imf_key: {2026: 3.5}}))
    imf = data.fetch_imf("XKX")
    assert imf.values == {2026: 3.5}
    assert imf.url.endswith("/NGDP_RPCH/UVK")


@pytest.mark.parametrize("kwargs", [
    {"body": requests.exceptions.ReadTimeout()},
    {"status": 500, "json": {"error": "boom"}},
    {"body": "not json"},
    {"json": imf_payload({"SDN": {2026: 7.0}})},
])
def test_fetch_imf_failures_never_raise(rsps, kwargs):
    rsps.add(responses.GET, data.imf_url("USA"), **kwargs)
    imf = data.fetch_imf("USA")
    assert imf.values == {}
    assert imf.problem
