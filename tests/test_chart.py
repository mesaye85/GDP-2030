"""SVG chart geometry helpers."""

from gdp_projection import chart
from gdp_projection import projection as pj


def test_y_ticks_cover_the_data():
    for lo, hi in [(7.5e11, 1.24e12), (0.0, 6.7e11), (3.1e9, 3.3e9), (5.0, 5.0)]:
        ticks = chart._y_ticks(lo, hi)
        assert ticks[0] <= lo and ticks[-1] >= hi
        assert len(ticks) >= 2


def test_format_usd():
    assert chart.format_usd(2.32e13) == "$23.20 T"
    assert chart.format_usd(4.5e10) == "$45.00 B"
    assert chart.format_usd(7.1e6) == "$7.10 M"
    assert chart.format_usd(950.0) == "$950"


def test_build_keeps_points_inside_the_plot():
    levels = {year: 1000.0 * 1.02 ** (year - 2000) for year in range(2000, 2026)}
    growth = {year: 2.0 + (year % 3) for year in range(2010, 2026)}
    projection = pj.project(levels, growth, imf={2026: 1.0, 2027: 1.5}, target_year=2040)
    geometry = chart.build(projection, levels)
    plot = geometry["plot"]
    for point in geometry["points"] + geometry["history_points"]:
        assert plot["left"] <= point["x"] <= plot["right"]
        assert plot["top"] <= point["y"] <= plot["bottom"]
    assert [p["imf"] for p in geometry["points"][:3]] == [True, True, False]
    assert geometry["x_ticks"][-1]["label"] == "2040"
