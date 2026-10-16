"""Shared fixtures.

Every test runs inside a ``responses`` mock, so any HTTP request that a test
has not registered raises ConnectionError instead of reaching the network.
"""

import pytest
import responses


@pytest.fixture(autouse=True)
def rsps():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


@pytest.fixture
def client():
    from main import app

    app.config.update(TESTING=True)
    return app.test_client()
