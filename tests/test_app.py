"""Flask endpoints via the test client, with World Bank and IMF mocked."""

import math

import pytest
import responses

from gdp_projection import data

from tests.fakes import BASE_VALUE, WB_INVALID, register_country, wb_url_re

SD = math.sqrt(10 / 9)  # sample sd of the synthetic 1 %/3 % growth history


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_imf_then_trend(client, rsps):
    register_country(rsps)
    response = client.post("/predict", json={"country_code": "tst", "year": 2033})
    assert response.status_code == 200
    body = response.get_json()

    assert body["country"] == "TST"
    assert body["country_name"] == "Testland"
    assert body["base_year"] == 2025
    assert body["base_value"] == pytest.approx(BASE_VALUE)
    assert body["target_year"] == 2033
    assert body["imf_available"] is True

    sources = [(p["year"], p["source"]) for p in body["path"]]
    assert sources == [(y, "imf") for y in range(2026, 2032)] + [(2032, "trend"), (2033, "trend")]
    assert [p["growth_rate"] for p in body["path"]] == [1.5] * 6 + [2.0, 2.0]

    expected = BASE_VALUE * 1.015 ** 6 * 1.02 ** 2
    assert body["projected_value"] == pytest.approx(expected, rel=1e-9)
    assert body["path"][-1]["value"] == body["projected_value"]
    assert body["band"]["low"] == pytest.approx(BASE_VALUE * (1.015 - SD / 100) ** 6 * (1.02 - SD / 100) ** 2)
    assert body["band"]["high"] == pytest.approx(BASE_VALUE * (1.015 + SD / 100) ** 6 * (1.02 + SD / 100) ** 2)
    assert body["band"]["low"] <= body["projected_value"] <= body["band"]["high"]
    assert "not a statistical confidence interval" in body["band"]["label"]

    assert body["trend"]["mean_growth"] == pytest.approx(2.0)
    assert body["trend"]["observations"] == 10
    assert body["sources"]["level"]["indicator"] == "NY.GDP.MKTP.KD"
    assert body["sources"]["history_growth"]["indicator"] == "NY.GDP.MKTP.KD.ZG"
    assert body["sources"]["forecast_growth"]["url"] == data.imf_url("TST")
    assert "IMF forecasts end at 2031" in " ".join(body["notes"])
    assert "not an economic model" in body["method"]


def test_predict_year_changes_the_answer(client, rsps):
    register_country(rsps)
    a = client.post("/predict", json={"country_code": "TST", "year": 2028}).get_json()
    b = client.post("/predict", json={"country_code": "TST", "year": 2040}).get_json()
    assert a["projected_value"] < b["projected_value"]
    assert len(b["path"]) == 15


def test_predict_get_query_string(client, rsps):
    register_country(rsps)
    response = client.get("/predict?country_code=TST&year=2027")
    assert response.status_code == 200
    assert response.get_json()["projected_value"] == pytest.approx(BASE_VALUE * 1.015 ** 2)


def test_predict_defaults_to_2030(client, rsps):
    register_country(rsps)
    assert client.post("/predict", json={"country_code": "TST"}).get_json()["target_year"] == 2030


@pytest.mark.parametrize("imf_kwargs", [
    {"imf_status": 500},
    {"imf": {}},  # IMF answers but has no series for this country
])
def test_predict_without_imf_uses_trend_everywhere(client, rsps, imf_kwargs):
    register_country(rsps, **imf_kwargs)
    body = client.post("/predict", json={"country_code": "TST", "year": 2030}).get_json()
    assert body["imf_available"] is False
    assert {p["source"] for p in body["path"]} == {"trend"}
    assert body["projected_value"] == pytest.approx(BASE_VALUE * 1.02 ** 5)
    assert "IMF forecasts unavailable" in body["notes"][0]


def test_predict_imf_timeout_falls_back(client, rsps):
    import requests

    register_country(rsps)
    rsps.replace(responses.GET, data.imf_url("TST"), body=requests.exceptions.ReadTimeout())
    body = client.post("/predict", json={"country_code": "TST", "year": 2030}).get_json()
    assert body["imf_available"] is False
    assert {p["source"] for p in body["path"]} == {"trend"}


def test_predict_unknown_country_404(client, rsps):
    rsps.add(responses.GET, wb_url_re("XYZ", data.WB_LEVEL), json=WB_INVALID)
    response = client.post("/predict", json={"country_code": "XYZ", "year": 2030})
    assert response.status_code == 404
    assert "XYZ" in response.get_json()["error"]


def test_predict_country_without_data_404(client, rsps):
    register_country(rsps, country="XXA", levels={2025: None, 2024: None})
    response = client.post("/predict", json={"country_code": "XXA"})
    assert response.status_code == 404


@pytest.mark.parametrize("year,fragment", [(2025, "after 2025"), (2010, "after 2025"), (2051, "2050")])
def test_predict_rejects_out_of_range_years(client, rsps, year, fragment):
    register_country(rsps)
    response = client.post("/predict", json={"country_code": "TST", "year": year})
    assert response.status_code == 400
    assert fragment in response.get_json()["error"]


def test_predict_past_year_stops_before_extra_requests(client, rsps):
    register_country(rsps)
    client.post("/predict", json={"country_code": "TST", "year": 2020})
    assert [c.request.url.split("?")[0].rsplit("/", 1)[-1] for c in rsps.calls] == ["NY.GDP.MKTP.KD"]


@pytest.mark.parametrize("body", [{}, {"year": 2030}, {"country_code": "US"}, {"country_code": 12}])
def test_predict_bad_input_400(client, body):
    assert client.post("/predict", json=body).status_code == 400


def test_predict_non_json_400(client):
    assert client.post("/predict", data="country_code=USA").status_code == 400


def test_predict_short_history_422(client, rsps):
    register_country(rsps, growth={2024: 2.0, 2025: 3.0})
    assert client.post("/predict", json={"country_code": "TST"}).status_code == 422


def test_predict_upstream_failure_502(client, rsps):
    rsps.add(responses.GET, wb_url_re("TST", data.WB_LEVEL), body="busy", status=503)
    assert client.post("/predict", json={"country_code": "TST"}).status_code == 502


def test_index_form_only(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "<form" in html and 'name="country"' in html and 'name="year"' in html
    assert "<svg" not in html


def test_index_renders_table_and_chart(client, rsps):
    register_country(rsps)
    response = client.get("/?country=tst&year=2033")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Testland" in html
    assert "<svg" in html and "<polygon" in html
    assert "<table" in html and "IMF WEO" in html and "Trend" in html
    assert html.count("<tr>") == 1 + 1 + 8  # header, base year, eight projected years
    assert "not a statistical confidence interval" in html


def test_index_shows_error(client, rsps):
    rsps.add(responses.GET, wb_url_re("XYZ", data.WB_LEVEL), json=WB_INVALID)
    response = client.get("/?country=XYZ&year=2030")
    assert response.status_code == 404
    assert "Could not project" in response.get_data(as_text=True)
