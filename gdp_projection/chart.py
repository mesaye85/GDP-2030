"""Server-side geometry for the inline SVG chart on the index page (no plotting library)."""

import math
from typing import Dict, List

from .projection import SOURCE_IMF, Projection

WIDTH, HEIGHT = 720, 320
LEFT, RIGHT, TOP, BOTTOM = 76, 20, 16, 36
HISTORY_YEARS = 15


def format_usd(value: float) -> str:
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"${value / size:,.2f} {suffix}"
    return f"${value:,.0f}"


def _axis_label(value: float, scale: float, suffix: str) -> str:
    if value == 0:
        return "$0"
    text = f"{value / scale:,.2f}".rstrip("0").rstrip(".")
    return f"${text} {suffix}".strip()


def _nice_step(span: float, target: int) -> float:
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * magnitude:
            return mult * magnitude
    return 10 * magnitude


def _y_ticks(lo: float, hi: float, target: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + abs(lo) * 0.1 + 1
    step = _nice_step(hi - lo, target)
    first = math.floor(lo / step)
    last = math.ceil(hi / step)
    return [i * step for i in range(first, last + 1)]


def _x_step(span: int) -> int:
    for step in (1, 2, 5, 10):
        if span / step <= 8:
            return step
    return 10


def build(projection: Projection, history: Dict[int, float]) -> dict:
    p = projection
    hist_years = [y for y in sorted(history) if p.base_year - HISTORY_YEARS < y <= p.base_year]
    first_year = hist_years[0] if hist_years else p.base_year
    last_year = p.target_year

    all_values = [history[y] for y in hist_years] + [pt.low for pt in p.path] + [pt.high for pt in p.path]
    all_values.append(p.base_value)
    ticks = _y_ticks(min(all_values), max(all_values))
    y_lo, y_hi = ticks[0], ticks[-1]

    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM

    def x(year: float) -> float:
        if last_year == first_year:
            return LEFT + plot_w / 2
        return round(LEFT + (year - first_year) / (last_year - first_year) * plot_w, 1)

    def y(value: float) -> float:
        return round(TOP + (y_hi - value) / (y_hi - y_lo) * plot_h, 1)

    top_value = max(abs(y_lo), abs(y_hi))
    scale, suffix = next(((s, n) for s, n in ((1e12, "T"), (1e9, "B"), (1e6, "M")) if top_value >= s), (1, ""))

    def pts(pairs) -> str:
        return " ".join(f"{x(a)},{y(b)}" for a, b in pairs)

    base = (p.base_year, p.base_value)
    central = [base] + [(pt.year, pt.value) for pt in p.path]
    upper = [base] + [(pt.year, pt.high) for pt in p.path]
    lower = [base] + [(pt.year, pt.low) for pt in p.path]

    # Round years on the step, plus the target year; drop a round year that would crowd it.
    x_step = _x_step(last_year - first_year)
    x_ticks = [yr for yr in range(first_year, last_year + 1) if yr % x_step == 0]
    if last_year not in x_ticks:
        if x_ticks and last_year - x_ticks[-1] < x_step / 2:
            x_ticks.pop()
        x_ticks.append(last_year)

    return {
        "width": WIDTH,
        "height": HEIGHT,
        "plot": {"left": LEFT, "right": WIDTH - RIGHT, "top": TOP, "bottom": HEIGHT - BOTTOM},
        "y_ticks": [{"y": y(t), "label": _axis_label(t, scale, suffix)} for t in ticks],
        "x_ticks": [{"x": x(yr), "label": str(yr)} for yr in x_ticks],
        "history": pts((yr, history[yr]) for yr in hist_years),
        "history_points": [
            {"x": x(yr), "y": y(history[yr]), "title": f"{yr}: {format_usd(history[yr])} (World Bank)"}
            for yr in hist_years
        ],
        "central": pts(central),
        "band": pts(upper + lower[::-1]),
        "base_x": x(p.base_year),
        "points": [
            {
                "x": x(pt.year),
                "y": y(pt.value),
                "imf": pt.source == SOURCE_IMF,
                "title": f"{pt.year}: {format_usd(pt.value)} at {pt.growth_rate:.2f}% ({pt.source}); "
                         f"band {format_usd(pt.low)} to {format_usd(pt.high)}",
            }
            for pt in p.path
        ],
    }
