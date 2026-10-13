"""Other machines' Studios in one view: the machines added here, their engines and memory, read by this Studio.

Each machine keeps its own Studio and token. This Studio reads the others' `/api/machine` and `/api/engine` with
their tokens, so the page never holds another machine's token: a token is written once, kept in `fleet.json`
(readable by this user only, like the settings) and never sent back to a page.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import settings as settings_mod

TOKEN_HEADER = "x-studio-token"
TIMEOUT = httpx.Timeout(4.0, connect=3.0)
MAX_PEERS = 16


def normalize_url(value: object) -> str:
    """`http(s)://host[:port]` with no path; raises ValueError otherwise."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("an address is needed, such as http://spark.example.ts.net")
    raw = value.strip()
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("the address must be http:// or https:// and name a machine")
    if parts.username or parts.password:
        raise ValueError("the address must not carry a user or password: the token goes in its own field")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("give the Studio's address only, without a path")
    return f"{parts.scheme}://{parts.netloc}"


def summary(machine: dict | None, engine: dict | None) -> dict:
    """What the fleet view shows of one machine, the same for this one and the others."""

    m, e = machine or {}, engine or {}
    findings = e.get("findings") or []
    return {
        "hostname": m.get("hostname"), "chip": m.get("chip"), "os": m.get("os"), "backend": m.get("backend"),
        "ram_bytes": m.get("ram_bytes"), "unified_memory": m.get("unified_memory"),
        "tensorfold_version": m.get("tensorfold_version"), "studio_version": m.get("studio_version"),
        "engine": None if not e else {
            "status": e.get("status"), "model": e.get("model"), "model_name": e.get("model_name"), "decode_tps": e.get("decode_tps"),
            "prefill_tps": e.get("prefill_tps"), "connections": e.get("connections"), "waiting": e.get("waiting"),
            "kv_fill": e.get("kv_fill"), "context_tokens": (e.get("startup") or {}).get("context_tokens"),
            "uptime_s": e.get("uptime_s"), "last_served": e.get("last_served"), "memory": e.get("machine_memory"), "prompt_cache": e.get("prompt_cache"),
            "findings": [{"level": f.get("level"), "title": f.get("title")} for f in findings
                         if f.get("level") in ("bad", "warn")],
        },
    }


class Peers:
    """The machines added to this Studio's fleet, kept in `fleet.json`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        return [p for p in data.get("peers", []) if isinstance(p, dict) and p.get("id") and p.get("url")]

    def _write(self, peers: list[dict]) -> None:
        settings_mod.write_private(self.path, json.dumps({"peers": peers}, indent=1))   # it holds the other Studios' tokens

    def all(self) -> list[dict]:
        with self._lock:
            return self._read()

    def public(self) -> list[dict]:
        """The machines as a page may see them: no token."""

        return [{"id": p["id"], "name": p["name"], "url": p["url"], "token_set": bool(p.get("token")),
                 "added": p.get("added")} for p in self.all()]

    def add(self, name: str, url: str, token: str | None) -> dict:
        with self._lock:
            peers = self._read()
            if any(p["url"] == url for p in peers):
                raise ValueError(f"{url} is already in the fleet")
            if len(peers) >= MAX_PEERS:
                raise ValueError(f"a fleet holds at most {MAX_PEERS} machines")
            peer = {"id": uuid.uuid4().hex[:12], "name": name, "url": url, "token": token or None,
                    "added": time.time()}
            self._write([*peers, peer])
            return peer

    def update(self, pid: str, *, name: str | None = None, token: str | None = None) -> dict:
        with self._lock:
            peers = self._read()
            peer = next((p for p in peers if p["id"] == pid), None)
            if peer is None:
                raise KeyError(pid)
            if name is not None:
                peer["name"] = name
            if token is not None:
                peer["token"] = token or None
            self._write(peers)
            return peer

    def remove(self, pid: str) -> None:
        with self._lock:
            peers = self._read()
            left = [p for p in peers if p["id"] != pid]
            if len(left) == len(peers):
                raise KeyError(pid)
            self._write(left)


def validate_name(value: object, fallback: str) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return fallback
    if not isinstance(value, str) or len(value.strip()) > 60:
        raise ValueError("a name is text of at most 60 characters")
    return value.strip()


def client() -> httpx.Client:
    """One client per probe or test; tests put another Studio's app behind it."""

    return httpx.Client(timeout=TIMEOUT, follow_redirects=False)


def _get(client: httpx.Client, url: str, path: str, token: str | None) -> httpx.Response:
    return client.get(url + path, headers={TOKEN_HEADER: token} if token else {})


def probe(peer: dict) -> dict:
    """One machine's summary, read from its Studio, with how long it took and why it failed if it did."""

    t0 = time.monotonic()
    out = {"id": peer["id"], "name": peer["name"], "url": peer["url"], "token_set": bool(peer.get("token")),
           "reachable": False, "error": None, "ms": None, "self": False}
    try:
        with client() as c:
            rm = _get(c, peer["url"], "/api/machine", peer.get("token"))
            if rm.status_code == 401:
                out["error"] = "its Studio refuses the token" if peer.get("token") else "its Studio needs its token"
                return out
            rm.raise_for_status()
            re = _get(c, peer["url"], "/api/engine", peer.get("token"))
            re.raise_for_status()
            out.update(summary(rm.json(), re.json()), reachable=True)
    except httpx.HTTPStatusError as exc:
        out["error"] = f"its Studio answered {exc.response.status_code}"
    except (httpx.HTTPError, ValueError) as exc:
        out["error"] = f"not reachable: {type(exc).__name__}"
    finally:
        out["ms"] = round((time.monotonic() - t0) * 1000)
    return out


def probe_all(peers: list[dict]) -> list[dict]:
    if not peers:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(peers))) as pool:
        return list(pool.map(probe, peers))


def test(url: str, token: str | None, *, own_hostname: str, own_version: str) -> dict:
    """What passed, failed or was skipped when reaching a Studio at ``url``, step by step."""

    checks: list[dict] = []

    def add(cid: str, label: str, status: str, detail: str, t0: float | None = None) -> None:
        checks.append({"id": cid, "label": label, "status": status, "detail": detail,
                       "seconds": round(time.monotonic() - t0, 2) if t0 else 0.0})

    def skip_rest(after: str) -> None:
        for cid, label in STEPS[[c for c, _ in STEPS].index(after) + 1:]:
            add(cid, label, "skip", "skipped: an earlier check failed")

    machine: dict | None = None
    step = "reach"
    try:
        with client() as c:
            t0 = time.monotonic()
            r = c.get(url + "/api/machine")
            add("reach", "The address answers", "pass", f"HTTP {r.status_code}", t0)
            step = "studio"

            if r.status_code == 421:
                add("studio", "It is a TensorFold Studio", "fail", "a Studio started without a token answers on its own "
                    "machine only: start it with --token-file, as the Mac and Spark setups do")
                skip_rest("studio")
                return {"ok": False, "checks": checks}
            is_studio = r.status_code == 401 and "token" in r.text.lower() or \
                (r.status_code == 200 and "studio_version" in r.text)
            add("studio", "It is a TensorFold Studio", "pass" if is_studio else "fail",
                "it asks for a Studio token" if r.status_code == 401 else
                "it answers as a Studio without a token" if is_studio else
                f"HTTP {r.status_code}, not a Studio's answer (an engine's address? give the Studio's, port 8090 or "
                "the tailscale serve name)")
            if not is_studio:
                skip_rest("studio")
                return {"ok": False, "checks": checks}

            step = "token"
            t0 = time.monotonic()
            if r.status_code == 200 and not token:
                machine = r.json()
                add("token", "The token opens it", "skip", "this Studio asks for no token (loopback only, or open)")
            else:
                r = _get(c, url, "/api/machine", token)
                if r.status_code == 200:
                    machine = r.json()
                    add("token", "The token opens it", "pass", "accepted", t0)
                else:
                    add("token", "The token opens it", "fail", "no token given: paste the one in its studio-token file"
                        if not token else "refused: the token is in ~/.config/tensorfold-studio/studio-token on Linux, "
                        "~/Library/Application Support/TensorFold Studio/studio-token on a Mac", t0)
                    skip_rest("token")
                    return {"ok": False, "checks": checks}

            same = bool(machine) and machine.get("hostname") == own_hostname
            add("other", "It is another machine", "fail" if same else "pass",
                "it is this Studio's own machine" if same else f"{machine.get('hostname')}, {machine.get('chip')}")
            theirs = str(machine.get("studio_version") or "?")
            add("version", "Same Studio version", "pass" if theirs == own_version else "warn",
                theirs if theirs == own_version else f"it runs Studio {theirs}, this one {own_version}: figures may be "
                "missing until both are updated")

            step = "engine"
            t0 = time.monotonic()
            r = _get(c, url, "/api/engine", token)
            if r.status_code == 200:
                e = r.json()
                add("engine", "Its engine", "pass" if e.get("status") == "running" else "warn",
                    f"serving {e.get('model')}" if e.get("status") == "running" else f"{e.get('status')}: nothing served now",
                    t0)
            else:
                add("engine", "Its engine", "fail", f"HTTP {r.status_code}", t0)
    except (httpx.HTTPError, ValueError) as exc:          # a stall, a reset, a body that is not JSON, a bad header
        label = dict(STEPS)[step]
        add(step, label, "fail", f"{type(exc).__name__}: " + ("check the name, the port and that the machine is on "
            "(Tailscale names end in .ts.net)" if step == "reach" else "the machine stopped answering, or answered "
            "something that is not a Studio's"), t0)
        skip_rest(step)
        return {"ok": False, "checks": checks}
    ok = all(ch["status"] in ("pass", "warn", "skip") for ch in checks)
    return {"ok": ok, "checks": checks, "machine": summary(machine, None) if machine else None}


STEPS = [("reach", "The address answers"), ("studio", "It is a TensorFold Studio"), ("token", "The token opens it"),
         ("other", "It is another machine"), ("version", "Same Studio version"), ("engine", "Its engine")]
