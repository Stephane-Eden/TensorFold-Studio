"""`tensorfold-studio`: start the local server and open the page."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from . import __version__
from . import settings as settings_mod
from .engine import Engine
from .server import create_app


def port_taken(host: str, port: int) -> str:
    """Whether something listens on the Studio's port already: "" if free, else what is said of it. TensorFold Web,
    begun as a copy of the Studio, listens on the same 8090. Windows is not asked: a probe there could share a port
    uvicorn then fails on anyway, with its own message."""

    if sys.platform == "win32":
        return ""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # as uvicorn does: a closing socket is no owner
        try:
            probe.bind((host, port))
        except OSError:
            from . import neighbours

            return f" ({neighbours.NAME} runs here and uses it by default)" if neighbours.running() else \
                " by another program"
    return ""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["service"]:                    # `tensorfold-studio service install|uninstall|status`
        from . import service

        return service.main(argv[1:])
    if argv[:1] == ["token"]:                      # `tensorfold-studio token [--copy] [--new]`
        from . import access

        return access.main(argv[1:])
    p = argparse.ArgumentParser(prog="tensorfold-studio", description="A local web interface for TensorFold",
                                epilog="`tensorfold-studio service install` keeps it running: at login on macOS, "
                                "with systemd on Linux. `tensorfold-studio token` shows the token its pages ask for.")
    p.add_argument("--host", default="127.0.0.1", help="address the Studio listens on (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8090, help="port the Studio listens on (default 8090)")
    p.add_argument("--engine-url", default=None, help="a TensorFold already running, here or on another machine")
    p.add_argument("--mock-engine", action="store_true", help="start a stand-in engine for development")
    p.add_argument("--mock-port", type=int, default=8080, help="the stand-in engine's port (default 8080, else a free one)")
    p.add_argument("--no-browser", action="store_true", help="do not open the page")
    p.add_argument("--token", default=None, help="required when --host is not loopback: the token the page and API calls "
                   "must carry; also worth setting when a proxy such as `tailscale serve` exposes the Studio")
    p.add_argument("--token-file", default=None, help="read the token from this file instead (keeps it out of the process list)")
    p.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                   help="another name the Studio may be reached under without a token (repeatable)")
    p.add_argument("--version", action="version", version=f"tensorfold-studio {__version__}")
    args = p.parse_args(argv)

    if args.token_file:
        try:
            args.token = Path(args.token_file).read_text().strip() or None
        except OSError as exc:
            p.error(f"cannot read --token-file: {exc}")
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not args.token:
        p.error("binding on a non-loopback address exposes the Studio's controls: pass --token <secret>, or keep "
                "127.0.0.1 and reach the Studio through an SSH tunnel")
    from . import instance

    lock = instance.hold()                         # before any recovery, adoption or store: one Studio at a time
    if lock is None:
        pid = instance.holder()
        print(f"[studio] another Studio is running for this user{f' (pid {pid})' if pid else ''}; it keeps the "
              "living copies, versions and settings. Stop it first (`tensorfold-studio service uninstall` for the "
              "service).", file=sys.stderr, flush=True)
        # systemd does not restart on 4 (3 is uvicorn's own failed start); launchd restarts any non-zero exit, so
        # under it the exit is clean and the agent stays down until the next login, as the message says
        return 0 if os.environ.get("XPC_SERVICE_NAME", "").startswith("com.tensorfold") else 4
    taken = port_taken(args.host, args.port)
    if taken:
        print(f"[studio] port {args.port} is taken{taken}: start the Studio on another, `--port {args.port + 1}` "
              f"(`tensorfold-studio service install --port {args.port + 1}` for the service)", file=sys.stderr,
              flush=True)
        return 0 if os.environ.get("XPC_SERVICE_NAME", "").startswith("com.tensorfold") else 4
    s = settings_mod.load()
    eng = Engine(s, state_dir=settings_mod.config_dir())
    eng.adopt()                                    # before any override: an engine of ours keeps its address and key
    if (args.engine_url or args.mock_engine) and eng.owned:   # the same refusal as Settings
        print(f"[studio] this Studio still runs {eng.model} from an earlier start; stop it (Monitor, or without "
              "--engine-url) before pointing the Studio at another engine", file=sys.stderr, flush=True)
        return 1
    if args.engine_url:
        try:                                           # the same checks as Settings: no credentials in the address
            s.engine_url = settings_mod.validate({"engine_url": args.engine_url})["engine_url"]
        except ValueError as exc:
            p.error(f"--engine-url: {exc}")
        s.remote = True
    if args.mock_engine:
        from .mock_engine import serve as mock

        server = mock(port=args.mock_port)
        port = server.server_address[1]
        s.engine_url, s.remote = f"http://127.0.0.1:{port}", True
        print(f"[studio] mock engine on {s.engine_url}")
    app = create_app(settings=s, engine=eng, token=args.token, bind_host=args.host, allowed_hosts=args.allow_host)
    shown = f"[{args.host}]" if ":" in args.host else args.host       # an IPv6 address needs its brackets in a URL
    url = f"http://{shown}:{args.port}/"
    # the address is printed without the token: this line ends up in service logs; only the browser opened here
    # gets the token, in its address, once
    note = " (its token is required: open it once with ?token=<the token>)" if args.token else ""
    print(f"[studio] TensorFold Studio {__version__} on {url}{note}", flush=True)
    if not args.no_browser:
        opened = url + (f"?token={args.token}" if args.token else "")
        threading.Timer(1.0, lambda: webbrowser.open(opened)).start()
    # an open page keeps a log stream that never ends by itself: without a bound, stopping the Studio waits for it
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
