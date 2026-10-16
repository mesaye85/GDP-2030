"""Flask entry point: ``python main.py`` serves on $PORT (default 5001)."""

import logging
import os

from flask import Flask, jsonify, render_template, request

from gdp_projection import chart, service
from gdp_projection.errors import ProjectionError
from gdp_projection.projection import DEFAULT_TARGET_YEAR, MAX_TARGET_YEAR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.json.sort_keys = False  # keep the documented field order in /predict


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Send a JSON object such as {\"country_code\": \"USA\", \"year\": 2030}."}), 400
    else:
        body = request.args
    country = body.get("country_code") or body.get("country")
    if not country:
        return jsonify({"error": "country_code is required."}), 400
    try:
        result = service.run(country, body.get("year"))
    except ProjectionError as exc:
        return jsonify({"error": str(exc)}), exc.status
    return jsonify(service.to_json(result))


@app.get("/")
def index():
    country = (request.args.get("country") or "").strip()
    year = (request.args.get("year") or "").strip()
    context = {
        "country": country.upper(),
        "year": year or DEFAULT_TARGET_YEAR,
        "max_year": MAX_TARGET_YEAR,
        "result": None,
        "chart": None,
        "error": None,
        "format_usd": chart.format_usd,
    }
    status = 200
    if country:
        try:
            result = service.run(country, year or None)
        except ProjectionError as exc:
            context["error"] = str(exc)
            status = exc.status
        else:
            context["result"] = service.to_json(result)
            context["chart"] = chart.build(result.projection, result.history)
    return render_template("index.html", **context), status


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    log.info("Starting on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
