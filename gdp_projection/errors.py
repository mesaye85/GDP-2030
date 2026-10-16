"""Exceptions shared by the data, projection and web layers.

Each carries the HTTP status the Flask app should return for it.
"""


class ProjectionError(Exception):
    status = 500


class InvalidInput(ProjectionError):
    """The request itself is malformed or out of range."""

    status = 400


class NotFound(ProjectionError):
    """The World Bank does not recognise the country code, or has no GDP data for it."""

    status = 404


class InsufficientData(ProjectionError):
    """The country exists but has too little growth history to compute a trend."""

    status = 422


class UpstreamError(ProjectionError):
    """The World Bank API failed or answered with something unexpected."""

    status = 502
