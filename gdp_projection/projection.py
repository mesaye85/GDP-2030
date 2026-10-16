"""Pure projection functions. No network access, no Flask.

Growth rates are percentages (2.5 means 2.5 %). Levels are constant 2015 US$.
"""

import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .errors import InsufficientData, InvalidInput, NotFound

MAX_TARGET_YEAR = 2050
DEFAULT_TARGET_YEAR = 2030
TREND_WINDOW = 10  # years of growth history behind the trend and the band
MIN_TREND_OBS = 5  # fewest non-null growth years accepted inside that window
SOURCE_IMF = "imf"
SOURCE_TREND = "trend"
BAND_LABEL = "spread from past growth volatility, not a statistical confidence interval"

_COUNTRY_RE = re.compile(r"[A-Za-z]{3}")
_YEAR_RE = re.compile(r"[0-9]{1,4}")


@dataclass(frozen=True)
class Trend:
    mean: float
    std: float
    first_year: int
    last_year: int
    observations: int


@dataclass(frozen=True)
class PathPoint:
    year: int
    growth_rate: float
    source: str
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class Projection:
    base_year: int
    base_value: float
    target_year: int
    trend: Trend
    path: List[PathPoint]

    @property
    def projected_value(self) -> float:
        return self.path[-1].value

    @property
    def low(self) -> float:
        return self.path[-1].low

    @property
    def high(self) -> float:
        return self.path[-1].high


def validate_country_code(code) -> str:
    if not isinstance(code, str) or not _COUNTRY_RE.fullmatch(code.strip()):
        raise InvalidInput("country_code must be a three-letter ISO 3166-1 alpha-3 code, such as USA or KEN.")
    return code.strip().upper()


def parse_target_year(value) -> int:
    """Coerce the requested year and check the upper bound.

    The lower bound (after the latest data year) needs the data, so it is
    checked separately by ``validate_target_year``.
    """
    if value is None or value == "":
        return DEFAULT_TARGET_YEAR
    if isinstance(value, int) and not isinstance(value, bool):
        year = value
    elif isinstance(value, str) and _YEAR_RE.fullmatch(value.strip()):
        year = int(value.strip())
    else:
        raise InvalidInput("year must be a whole number, such as 2030.")
    if year > MAX_TARGET_YEAR:
        raise InvalidInput(f"year must be {MAX_TARGET_YEAR} or earlier.")
    return year


def validate_target_year(year: int, base_year: int) -> None:
    if year <= base_year:
        raise InvalidInput(
            f"year must be after {base_year}, the latest year with World Bank GDP data for this country."
        )
    if year > MAX_TARGET_YEAR:
        raise InvalidInput(f"year must be {MAX_TARGET_YEAR} or earlier.")


def latest_level(levels: Dict[int, float]) -> Tuple[int, float]:
    if not levels:
        raise NotFound("No real GDP data (constant 2015 US$) is available for this country.")
    year = max(levels)
    return year, levels[year]


def trailing_trend(growth: Dict[int, float], end_year: int,
                   window: int = TREND_WINDOW, min_obs: int = MIN_TREND_OBS) -> Trend:
    """Mean and sample standard deviation of growth over the ``window`` years ending at ``end_year``."""
    years = [y for y in range(end_year - window + 1, end_year + 1) if y in growth]
    if len(years) < min_obs:
        raise InsufficientData(
            f"Only {len(years)} of the {window} years {end_year - window + 1}-{end_year} have real GDP growth "
            f"data; at least {min_obs} are needed for a trend."
        )
    rates = [growth[y] for y in years]
    return Trend(
        mean=statistics.fmean(rates),
        std=statistics.stdev(rates),
        first_year=years[0],
        last_year=years[-1],
        observations=len(years),
    )


def growth_schedule(base_year: int, target_year: int, imf: Dict[int, float],
                    trend_mean: float) -> List[Tuple[int, float, str]]:
    """``(year, rate, source)`` for every year after the base year up to the target.

    A year with an IMF value uses it; every other year (in practice, the years
    past the IMF horizon) uses the trend.
    """
    schedule = []
    for year in range(base_year + 1, target_year + 1):
        if year in imf:
            schedule.append((year, imf[year], SOURCE_IMF))
        else:
            schedule.append((year, trend_mean, SOURCE_TREND))
    return schedule


def compound(base_value: float, rates: List[float]) -> List[float]:
    """Apply each annual rate in turn; returns the level after each year.

    A rate at or below -100 % takes the level to zero rather than negative.
    """
    values = []
    level = base_value
    for rate in rates:
        level *= max(0.0, 1 + rate / 100)
        values.append(level)
    return values


def project(levels: Dict[int, float], growth: Dict[int, float], imf: Dict[int, float],
            target_year: int) -> Projection:
    """Project real GDP from the latest level to ``target_year``.

    Central path: IMF rate where available, trailing mean otherwise.
    Band: the same rates minus/plus one standard deviation of the trailing
    growth history, compounded every year. On trend years that is exactly
    mean -/+ sd; on IMF years it centres the spread on the IMF rate, so the
    band always contains the central path.
    """
    base_year, base_value = latest_level(levels)
    validate_target_year(target_year, base_year)
    trend = trailing_trend(growth, base_year)
    schedule = growth_schedule(base_year, target_year, imf, trend.mean)

    rates = [rate for _, rate, _ in schedule]
    central = compound(base_value, rates)
    low = compound(base_value, [r - trend.std for r in rates])
    high = compound(base_value, [r + trend.std for r in rates])

    path = [
        PathPoint(year=year, growth_rate=rate, source=source, value=v, low=lo, high=hi)
        for (year, rate, source), v, lo, hi in zip(schedule, central, low, high)
    ]
    return Projection(base_year=base_year, base_value=base_value, target_year=target_year, trend=trend, path=path)
