"""TensorFold 1.1.1 to 1.2.5: what changed under the Studio, one test each."""

import asyncio
import contextlib

import httpx
from fastapi.testclient import TestClient

from studio import living, native
from studio import settings as settings_mod
from studio.catalog import Catalog
from studio.engine import Engine
from studio.proxy import Proxy
from studio.server import create_app


def test_the_relay_keeps_the_page_origin_from_the_engine():
    """1.1.1: the engine answers 403 to an Origin that is not its own address, and the page's never is."""

    seen = []
    p = Proxy(lambda: "http://127.0.0.1:8080", lambda: "KEY")
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: seen.append(dict(r.headers)) or httpx.Response(200, json={})))

    class Req:
        method, query_params = "POST", httpx.QueryParams()
        headers = httpx.Headers({"origin": "http://studio.example.ts.net", "content-type": "application/json",
                                 "x-studio-token": "T", "host": "studio.example.ts.net"})

        async def stream(self):
            yield b"{}"
    asyncio.run(p.handle(Req(), "v1/chat/completions"))
    sent = seen[0]
    assert "origin" not in sent and "x-studio-token" not in sent
    assert sent["host"] == "127.0.0.1:8080" and sent["authorization"] == "Bearer KEY"


def test_the_serving_line_of_1_2_names_its_draft_sources():
    """1.2: `drafts:` names the sources (`MTP + copies`, `DFlash2 (…)`, `none`), and the context may be unlimited."""

    head = "info: serving nemotron at http://127.0.0.1:8080/v1 (sampling: temperature 0.6, top-p 0.95; "
    assert native.parse_startup(head + "drafts: MTP + copies; context: 262144; loaded in 21.4s)") == \
        {"context_tokens": 262144.0, "loaded_s": 21.4, "drafts": 1.0}
    assert native.parse_startup(head + "drafts: DFlash2 (z-lab/Qwen3.8-27B-DFlash2) + copies; context: 32768; "
                                       "loaded in 9.0s)")["drafts"] == 1.0
    assert native.parse_startup(head + "drafts: none; context: unlimited; loaded in 3.2s)") == \
        {"loaded_s": 3.2, "drafts": 0.0}
    assert native.parse_startup(head + "drafts: off; context: 4096; loaded in 1.0s)")["drafts"] == 0.0


def test_a_lesson_whose_save_failed_is_not_kept(monkeypatch):
    """1.2.1: after the facts, the engine writes the shards and says `saved`, or `failed` with its error."""

    events = ["event: fact", 'data: {"id": 4, "text": "My cat is called Pixel."}',
              "event: learning", 'data: {"id": 4}',
              "event: learned", 'data: {"id": 4, "recalled": true, "steps": 12}',
              "event: failed", 'data: {"phase": "saving", "message": "No space left on device"}',
              "data: [DONE]"]

    class Resp:
        status_code = 200

        def iter_lines(self):
            return iter(events)

        def close(self):
            pass

    @contextlib.contextmanager
    def stream(*_a, **_k):
        yield Resp()
    monkeypatch.setattr(living.httpx, "stream", stream)
    teacher = living.Teacher(lambda: None, lambda: None, lambda: None, graph=lambda: {"facts": []})
    it = living.Item(id="1", fact="My cat is called Pixel.")
    teacher._learn(living.Target("http://127.0.0.1:8080", "K", "m"), it)
    assert it.state == "failed" and "saving" in it.message and "No space left" in it.message

    events[-3:-1] = ["event: saved", 'data: {"phase": "saving"}']
    it = living.Item(id="2", fact="My cat is called Pixel.")
    teacher._learn(living.Target("http://127.0.0.1:8080", "K", "m"), it)
    assert it.state == "kept"


def test_a_copy_that_learns_gets_an_engine_key(tmp_path, monkeypatch):
    """1.2.5: `--slide` without a key stops the engine; the Studio makes one, as it does for sharing."""

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "conf")
    monkeypatch.setenv("TENSORFOLD_STUDIO_LIVING", str(tmp_path / "living"))
    (living.root() / "eden").mkdir(parents=True)
    (living.root() / "eden" / "config.json").write_text('{"model_type": "nemotron_h"}')
    monkeypatch.setattr(living, "teachable_here", lambda _b: None)
    monkeypatch.setattr(native, "find_binary", lambda: "/bin/tf")
    s = settings_mod.Settings(engine_url="http://127.0.0.1:1", remote=False)
    eng = Engine(s)
    started = []
    monkeypatch.setattr(eng, "start", lambda model, **k: started.append((model, s.api_key, k.get("slide"))) or {})
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    c = TestClient(create_app(settings=s, engine=eng, catalog=cat))
    r = c.post("/api/engine/start", json={"model": living.PREFIX + "eden"})
    assert r.status_code == 200, r.text
    model, key, slide = started[0]
    assert model == living.PREFIX + "eden" and slide and key and key.startswith("tf-")
    assert settings_mod.load().api_key == key                    # kept: the relay and Connect use the same key

    s.api_key = "tf-mine"                                        # a key already there is never replaced
    c.post("/api/engine/start", json={"model": living.PREFIX + "eden"})
    assert started[-1][1] == "tf-mine"


def test_the_engine_says_when_thinking_cannot_be_turned_off():
    """1.2.5: /v1/models carries the model's capabilities; an explicit off it cannot honour is a 400."""

    switch = {"value": False}

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "gemma", "capabilities": {"thinking_switch": switch["value"], "thinking_applicable": True}}]})
        return httpx.Response(200, json={"status": "ok", "model": "gemma", "live": {}})
    eng = Engine(settings_mod.Settings(engine_url="http://e.test:8080", remote=True))
    eng._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert eng.info()["thinking_switch"] is False
    switch["value"] = True                                       # read once per engine: the same target keeps it
    assert eng.info()["thinking_switch"] is False
    with eng.target_lock:
        eng.retargeted()
    assert eng.info()["thinking_switch"] is True

    plain = Engine(settings_mod.Settings(engine_url="http://e.test:8080", remote=True))
    plain._client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"data": [{"id": "x"}]} if r.url.path == "/v1/models" else
                                 {"status": "ok", "model": "x"})))
    assert plain.info()["thinking_switch"] is None              # an engine that does not say


def test_the_cuda_sizing_line_with_the_slide_trainer():
    """1.2: a copy that learns on CUDA names the memory kept for its trainer in the sizing line."""

    line = ("[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 18.49 GiB and 25.54 GiB "
            "kept for --slide's trainer; 8 streams at once, 0.18 GiB each at a 16384-token window, of 36.8 GiB left "
            "after a 12.2 GiB reserve; prompts in passes of up to 2048 rows")
    assert native.parse_startup(line) == {"resident_gib": 18.49, "streams": 8.0, "stream_gib": 0.18, "left_gib": 36.8,
                                          "reserve_gib": 12.2, "trainer_gib": 25.54}


def test_a_checkpoint_the_family_does_not_serve_is_a_refusal():
    """1.2.5 lists Gemma 4's MLX 4-bit format but serves only the dense 31B and 12B: the 26B-A4B is refused by name."""

    line = ("tensorfold: the native Gemma 4 engine cannot load this checkpoint (GemmaMoeNotServed): it serves Gemma 4 "
            "31B and 12B as MLX affine 2-6 or 8 bits in groups of 64, uniform or per module")
    assert native.refusal_in(["[tensorfold] opening", line]) == "GemmaMoeNotServed"
    assert native.refusal_in(["tensorfold: the native Qwen3.8-27B engine cannot load /x (QwenMetalQualificationRequired)"]) \
        == "QwenMetalQualificationRequired"
    assert native.refusal_in(["tensorfold: cannot load /x (OverMemoryLimit)", "cannot load m (OpenFailed)"]) is None
