"""Glue between the data layer and the pure projection: fetch, project, serialise."""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

from . import data
from .errors import NotFound
from .projection import (
    BAND_LABEL,
    SOURCE_IMF,
    SOURCE_TREND,
    TREND_WINDOW,
    Projection,
    latest_level,
    parse_target_year,
    project,
    validate_country_code,
    validate_target_year,
)

HISTORY_SPAN = 30  # years of World Bank history requested
UNITS = "constant 2015 US$"

METHOD_NOTE = (
    "Rule-based extrapolation, not an economic model. Starting from the latest World Bank real GDP level "
    "(constant 2015 US$), each year is grown by the IMF WEO real growth forecast where one exists (source 'imf') "
    f"and by the mean of the previous {TREND_WINDOW} years of World Bank real growth otherwise (source 'trend'). "
    "value_t = value_(t-1) * (1 + g_t / 100). The low/high band subtracts/adds one standard deviation of that same "
    f"{TREND_WINDOW}-year growth history to every year's rate and compounds it; it is a {BAND_LABEL}."
)


@dataclass(frozen=True)
class Result:
    country: str
    country_name: str
    projection: Projection
    history: Dict[int, float]
    level: data.Series
    growth: data.Series
    imf: data.ImfSeries


def run(country_code, year=None, today: Optional[date] = None) -> Result:
    """Validate the inputs, fetch the series and project. Raises ProjectionError subclasses."""
    country = validate_country_code(country_code)
    target_year = parse_target_year(year)

    end = (today or date.today()).year
    start = end - HISTORY_SPAN

    level = data.fetch_world_bank(country, data.WB_LEVEL, start, end)
    if not level.values:
        raise NotFound(f"The World Bank has no real GDP data ({data.WB_LEVEL}, {UNITS}) for '{country}'.")
    base_year, _ = latest_level(level.values)
    validate_target_year(target_year, base_year)  # fail before the remaining two requests

    growth = data.fetch_world_bank(country, data.WB_GROWTH, start, end)
    imf = data.fetch_imf(country)
    projection = project(level.values, growth.values, imf.values, target_year)

    return Result(
        country=country,
        country_name=level.country_name or country,
        projection=projection,
        history=level.values,
        level=level,
        growth=growth,
        imf=imf,
    )


def notes(result: Result) -> list:
    p = result.projection
    sources = {pt.source for pt in p.path}
    out = []
    if result.imf.problem:
        out.append(f"IMF forecasts unavailable ({result.imf.problem}); every year uses the {TREND_WINDOW}-year trend.")
    elif SOURCE_IMF not in sources:
        out.append(f"The IMF series has no values after {p.base_year}; every year uses the {TREND_WINDOW}-year trend.")
    elif SOURCE_TREND in sources:
        last_imf = max(pt.year for pt in p.path if pt.source == SOURCE_IMF)
        out.append(f"IMF forecasts end at {last_imf}; later years use the {TREND_WINDOW}-year trend.")
    if p.trend.observations < TREND_WINDOW:
        out.append(
            f"Only {p.trend.observations} of the last {TREND_WINDOW} years have growth data; "
            "the trend and band rest on those."
        )
    return out


def to_json(result: Result) -> dict:
    p = result.projection
    return {
        "country": result.country,
        "country_name": result.country_name,
        "units": UNITS,
        "base_year": p.base_year,
        "base_value": round(p.base_value, 2),
        "target_year": p.target_year,
        "projected_value": round(p.projected_value, 2),
        "band": {
            "low": round(p.low, 2),
            "high": round(p.high, 2),
            "label": BAND_LABEL,
        },
        "trend": {
            "mean_growth": round(p.trend.mean, 4),
            "std_growth": round(p.trend.std, 4),
            "from_year": p.trend.first_year,
            "to_year": p.trend.last_year,
            "observations": p.trend.observations,
        },
        "imf_available": not result.imf.problem,
        "path": [
            {
                "year": pt.year,
                "growth_rate": round(pt.growth_rate, 4),
                "source": pt.source,
                "value": round(pt.value, 2),
                "low": round(pt.low, 2),
                "high": round(pt.high, 2),
            }
            for pt in p.path
        ],
        "sources": {
            "level": {"indicator": result.level.indicator, "url": result.level.url},
            "history_growth": {"indicator": result.growth.indicator, "url": result.growth.url},
            "forecast_growth": {"indicator": data.IMF_GROWTH, "url": result.imf.url},
        },
        "method": METHOD_NOTE,
        "notes": notes(result),
    }
