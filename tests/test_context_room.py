"""TensorFold 1.2 on a Mac sizes its streams before loading and refuses a context the memory has no room for: the
Studio starts again at half, for that start only, and says so."""

from fastapi.testclient import TestClient

from studio import native
from studio import settings as settings_mod
from studio.catalog import Catalog
from studio.engine import Engine
from studio.server import create_app

MODEL = "TensorFold/Qwen3.8-27B-MLX-4bit"


def refusal(context: int) -> str:
    return (f"tensorfold: --parallel 8 needs 23.4 GiB for its streams and 512-row prompt scratch at a {context}-token "
            "context; 4.1 GiB remains beside the weights and an 8 GiB margin: no wired-limit raise fits one stream and "
            "leaves 4 GiB of this Mac's 32.0 GiB of RAM to the system: lower --context")


def test_the_refusal_line():
    assert native.context_refusal(["[tensorfold] opening", refusal(32768)]) == 32768
    assert native.context_refusal(["tensorfold: cannot load /x (OverMemoryLimit)"]) is None


def run(tmp_path, monkeypatch, room: int):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "conf")
    monkeypatch.setattr(native, "find_binary", lambda: "/bin/tf")
    monkeypatch.setattr(Engine, "owned", property(lambda self: getattr(self, "_up", False)))
    s = settings_mod.Settings(engine_url="http://127.0.0.1:1", remote=False)
    eng = Engine(s)
    starts = []

    def start(model, **k):
        starts.append(s.context)
        eng.log.clear()
        asked = s.context or 32768
        if asked > room:
            eng.log.append(refusal(asked))
            eng.exit_code, eng._up = 1, False
        else:
            eng.exit_code, eng._up = None, True
        return {"model": model}
    monkeypatch.setattr(eng, "start", start)
    monkeypatch.setattr(eng, "wait_stopped", lambda _s: True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    monkeypatch.setattr(cat, "native_here", lambda e: True)
    c = TestClient(create_app(settings=s, engine=eng, catalog=cat))
    r = c.post("/api/engine/start", json={"model": MODEL})
    return r, starts, s, eng


def test_a_context_without_room_is_halved_for_this_start(tmp_path, monkeypatch):
    r, starts, s, eng = run(tmp_path, monkeypatch, room=16384)
    assert r.status_code == 200, r.text
    assert starts == [None, 16384] and s.context is None             # Settings keep their own
    assert any("no room for 32,768 tokens" in line and "started at 16,384" in line for line in eng.log)


def test_no_room_at_all_is_said(tmp_path, monkeypatch):
    _r, starts, _s, eng = run(tmp_path, monkeypatch, room=1000)
    assert starts == [None, 16384, 8192, 4096]
    assert any("not even 4,096" in line for line in eng.log)
