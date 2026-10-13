import pytest

from studio import server


@pytest.fixture(autouse=True)
def _testclient_is_local(monkeypatch):
    """FastAPI's test client calls itself `testserver`: for the Studio's host check it is this machine."""

    monkeypatch.setattr(server, "LOOPBACK_NAMES", server.LOOPBACK_NAMES | {"testserver"})
