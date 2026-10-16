"""Pure projection math on synthetic series."""

import math
import statistics

import pytest

from gdp_projection import projection as pj
from gdp_projection.errors import InsufficientData, InvalidInput, NotFound


def flat_growth(rate, start=2010, end=2025):
    return {year: rate for year in range(start, end + 1)}


def test_compound_known_values():
    assert pj.compound(100.0, [10.0, 10.0]) == pytest.approx([110.0, 121.0])
    assert pj.compound(100.0, [-50.0, 100.0]) == pytest.approx([50.0, 100.0])
    assert pj.compound(100.0, []) == []


def test_compound_floor_at_zero():
    assert pj.compound(100.0, [-150.0, 5.0]) == [0.0, 0.0]


def test_trailing_trend_uses_last_ten_years_only():
    growth = flat_growth(50.0, 2000, 2015)  # outside the window, must be ignored
    growth.update({year: (1.0 if year % 2 else 3.0) for year in range(2016, 2026)})
    trend = pj.trailing_trend(growth, 2025)
    assert (trend.first_year, trend.last_year, trend.observations) == (2016, 2025, 10)
    assert trend.mean == pytest.approx(2.0)
    assert trend.std == pytest.approx(math.sqrt(10 / 9))  # sample sd of five 1s and five 3s
    assert trend.std == pytest.approx(statistics.stdev([1.0, 3.0] * 5))


def test_trailing_trend_skips_missing_years_but_needs_enough():
    growth = {2021: 1.0, 2022: 2.0, 2023: 3.0, 2024: 4.0, 2025: 5.0}
    trend = pj.trailing_trend(growth, 2025)
    assert trend.observations == 5
    assert trend.mean == pytest.approx(3.0)
    del growth[2021]
    with pytest.raises(InsufficientData):
        pj.trailing_trend(growth, 2025)


def test_trend_only_projection_matches_closed_form():
    levels = {2024: 900.0, 2025: 1000.0}
    p = pj.project(levels, flat_growth(2.0), imf={}, target_year=2030)
    assert p.base_year == 2025 and p.base_value == 1000.0
    assert [pt.year for pt in p.path] == list(range(2026, 2031))
    assert {pt.source for pt in p.path} == {pj.SOURCE_TREND}
    assert p.projected_value == pytest.approx(1000.0 * 1.02 ** 5)
    # constant history means zero spread: the band collapses onto the path
    assert p.low == pytest.approx(p.projected_value)
    assert p.high == pytest.approx(p.projected_value)


def test_imf_years_then_trend_after_horizon():
    levels = {2025: 1000.0}
    growth = flat_growth(2.0)
    imf = {2024: 9.9, 2026: 5.0, 2027: 4.0}  # 2024 is before the base year and must not be used
    p = pj.project(levels, growth, imf, target_year=2029)
    assert [(pt.year, pt.source, pt.growth_rate) for pt in p.path] == [
        (2026, pj.SOURCE_IMF, 5.0),
        (2027, pj.SOURCE_IMF, 4.0),
        (2028, pj.SOURCE_TREND, 2.0),
        (2029, pj.SOURCE_TREND, 2.0),
    ]
    assert p.path[1].value == pytest.approx(1000.0 * 1.05 * 1.04)
    assert p.projected_value == pytest.approx(1000.0 * 1.05 * 1.04 * 1.02 * 1.02)


def test_band_is_rate_minus_and_plus_one_sd_compounded():
    levels = {2025: 1000.0}
    growth = {year: (1.0 if year % 2 else 3.0) for year in range(2016, 2026)}
    sd = math.sqrt(10 / 9)
    p = pj.project(levels, growth, imf={2026: 4.0}, target_year=2028)
    assert p.low == pytest.approx(1000.0 * (1.04 - sd / 100) * (1.02 - sd / 100) ** 2)
    assert p.high == pytest.approx(1000.0 * (1.04 + sd / 100) * (1.02 + sd / 100) ** 2)
    for pt in p.path:
        assert pt.low <= pt.value <= pt.high


def test_band_ordering_with_volatile_and_negative_history():
    levels = {2025: 5.0e11}
    growth = {2016: 6.0, 2017: -8.0, 2018: 12.0, 2019: -3.0, 2020: -20.0,
              2021: 15.0, 2022: 4.0, 2023: -1.0, 2024: 2.0, 2025: 0.5}
    p = pj.project(levels, growth, imf={2026: -2.0, 2027: 1.0}, target_year=2050)
    assert len(p.path) == 25
    for pt in p.path:
        assert 0.0 <= pt.low <= pt.value <= pt.high


@pytest.mark.parametrize("year", [2025, 2020, 1999])
def test_target_year_must_follow_latest_data(year):
    with pytest.raises(InvalidInput, match="after 2025"):
        pj.project({2025: 1000.0}, flat_growth(2.0), {}, target_year=year)


@pytest.mark.parametrize("value", [2051, "2051", 3000])
def test_target_year_capped_at_2050(value):
    with pytest.raises(InvalidInput, match="2050"):
        pj.parse_target_year(value)


@pytest.mark.parametrize("value", ["20x0", 2030.5, True, [2030], "-5"])
def test_target_year_rejects_non_integers(value):
    with pytest.raises(InvalidInput):
        pj.parse_target_year(value)


def test_target_year_defaults_and_accepts_strings():
    assert pj.parse_target_year(None) == pj.DEFAULT_TARGET_YEAR
    assert pj.parse_target_year("") == pj.DEFAULT_TARGET_YEAR
    assert pj.parse_target_year(" 2040 ") == 2040
    assert pj.parse_target_year(2050) == 2050


@pytest.mark.parametrize("code,expected", [("usa", "USA"), (" Ken ", "KEN"), ("DEU", "DEU")])
def test_country_code_normalised(code, expected):
    assert pj.validate_country_code(code) == expected


@pytest.mark.parametrize("code", ["US", "USAA", "U5A", "", None, 840])
def test_country_code_must_be_three_letters(code):
    with pytest.raises(InvalidInput):
        pj.validate_country_code(code)


def test_no_levels_is_not_found():
    with pytest.raises(NotFound):
        pj.latest_level({})
