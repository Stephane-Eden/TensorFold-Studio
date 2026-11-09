"""TensorFold Web began as a copy of the Studio and kept its defaults: the two on one machine must not share the living
copies' folder (each recovers it when it starts) and must say so when one holds the other's port."""

import socket
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from studio import cli, instance, living, neighbours
from studio import settings as settings_mod
from studio.catalog import Catalog
from studio.engine import Engine
from studio.server import create_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TENSORFOLD_STUDIO_LIVING", raising=False)
    web = tmp_path / "web-config"
    monkeypatch.setattr(neighbours, "config_dir", lambda: web)
    return tmp_path


def test_a_new_studio_keeps_its_copies_apart_from_web(home):
    assert living.root() == home / "tensorfold-studio-living"
    (home / "tensorfold-living" / "nemotron-living").mkdir(parents=True)      # Web's copy, made with Web's defaults
    (home / "tensorfold-living" / "nemotron-living" / "config.json").write_text("{}")
    (home / "tensorfold-living" / ".studio").mkdir()
    (home / "web-config").mkdir()
    assert living.root() == home / "tensorfold-studio-living"
    assert living.left_behind() == home / "tensorfold-living"                # said on the Teach page
    assert not (home / "tensorfold-living" / ".studio" / "owner").exists()   # nothing of Web's is touched


def test_an_earlier_studio_keeps_its_folder_even_once_web_comes(home):
    (home / "tensorfold-living" / ".studio" / "eden").mkdir(parents=True)    # a Studio's copy, before Web was here
    assert living.root() == home / "tensorfold-living"
    assert (home / "tensorfold-living" / ".studio" / "owner").is_file()      # claimed, once
    (home / "web-config").mkdir()                                            # Web installed afterwards
    assert living.root() == home / "tensorfold-living" and living.left_behind() is None
    assert neighbours.shares_living(living.root())                           # and that is said at start
    (home / "web-config" / "web.env").write_text("HF_HUB_CACHE=/x\nTENSORFOLD_WEB_LIVING='~/web-copies'\n")
    assert neighbours.living_root() == home / "web-copies"
    assert not neighbours.shares_living(living.root())


def test_the_variable_wins(home, monkeypatch):
    (home / "tensorfold-living" / ".studio").mkdir(parents=True)
    monkeypatch.setenv("TENSORFOLD_STUDIO_LIVING", str(home / "mine"))
    assert living.root() == home / "mine" and living.left_behind() is None
    assert not (home / "tensorfold-living" / ".studio" / "owner").exists()


def test_web_is_running_while_it_holds_its_lock(home):
    assert not neighbours.running()                                          # not installed: nothing is made
    assert not (home / "web-config").exists()
    (home / "web-config").mkdir()
    lock = home / "web-config" / "web.lock"
    lock.write_text("123")
    assert not neighbours.running()
    with open(lock, "a+") as held:
        assert instance._try_lock(held)
        assert neighbours.running()
    assert not neighbours.running()


def test_a_studio_beside_a_running_web_leaves_the_shared_folder_alone(home, monkeypatch):
    shared = home / "tensorfold-living"
    (shared / ".studio").mkdir(parents=True)
    (shared / ".studio" / "owner").write_text("tensorfold-studio\n")
    (shared / ".nemotron-living.restore").mkdir()                            # Web restoring a version right now
    (home / "web-config").mkdir()
    monkeypatch.setattr(neighbours, "running", lambda: True)
    monkeypatch.setattr(settings_mod, "config_dir", lambda: home / "conf")
    s = settings_mod.Settings(engine_url="http://127.0.0.1:1", remote=False)
    eng = Engine(s)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    c = TestClient(create_app(settings=s, engine=eng, catalog=cat))
    assert (shared / ".nemotron-living.restore").is_dir()                    # not swept away as a leftover
    assert any("TensorFold Web keeps its living copies" in line and "nothing left halfway" in line for line in eng.log)
    view = c.get("/api/living").json()
    assert view["shared_with"] == "TensorFold Web" and view["left_behind"] is None

    monkeypatch.setattr(neighbours, "running", lambda: False)                # Web stopped: leftovers are recovered
    create_app(settings=s, engine=Engine(s), catalog=cat)
    assert not (shared / ".nemotron-living.restore").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="not probed on Windows")
def test_a_taken_port_is_said_before_the_server_starts(monkeypatch):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        monkeypatch.setattr(neighbours, "running", lambda: False)
        assert cli.port_taken("127.0.0.1", port) == " by another program"
        monkeypatch.setattr(neighbours, "running", lambda: True)
        assert "TensorFold Web" in cli.port_taken("127.0.0.1", port)
    assert cli.port_taken("127.0.0.1", port) == ""


def test_an_engine_port_held_by_web_is_named(monkeypatch):
    s = settings_mod.Settings(engine_url="http://127.0.0.1:8080", remote=False)
    eng = Engine(s)
    eng._client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"status": "ok", "model": "nemotron", "data": []})))
    monkeypatch.setattr(neighbours, "running", lambda: True)
    with pytest.raises(RuntimeError, match="TensorFold Web's engine.*8081"):
        eng.start("nvidia/x")
