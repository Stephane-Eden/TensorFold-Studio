import os

import pytest

from studio import neighbours, server


@pytest.fixture(autouse=True)
def _testclient_is_local(monkeypatch):
    """FastAPI's test client calls itself `testserver`: for the Studio's host check it is this machine."""

    monkeypatch.setattr(server, "LOOPBACK_NAMES", server.LOOPBACK_NAMES | {"testserver"})


@pytest.fixture(autouse=True)
def _no_real_home(tmp_path_factory, monkeypatch):
    """Living copies under a test's own folder unless the test says where, and no TensorFold Web of the machine's."""

    if not os.environ.get("TENSORFOLD_STUDIO_LIVING"):
        monkeypatch.setenv("TENSORFOLD_STUDIO_LIVING", str(tmp_path_factory.mktemp("living")))
    away = tmp_path_factory.mktemp("no-web") / "TensorFold Web"
    monkeypatch.setattr(neighbours, "config_dir", lambda: away)
