# TensorFold Studio

A web interface for [TensorFold](https://github.com/ashhart/TensorFold): pick a model that fits the machine, run it,
watch it work, chat with it, teach it (Living Weights), measure it and connect tools to it, from a browser tab. In daily
use on a Mac mini M6 (Metal) and two DGX Spark GB10 (CUDA) with TensorFold 1.0.9. Working name: the TensorFold name
and logo are yours, and their use here is for you to agree to (`docs/questions-for-ash.md`, A1).

`docs/features.md` describes every screen and the module behind it; `docs/questions-for-ash.md` lists what the Studio
would like from the engine, with the bugs found on 1.0.7 and checked again on 1.0.9.

## Run it

```bash
uv venv .venv && . .venv/bin/activate          # or python -m venv
uv pip install -e '.[dev]'
tensorfold-studio                               # http://127.0.0.1:8090, opens the page
tensorfold-studio --mock-engine                 # without a model: a stand-in engine on port 8080
tensorfold-studio --engine-url http://spark.local:8080   # watch a TensorFold running elsewhere
tensorfold-studio --host 0.0.0.0 --token <secret>        # reachable from other machines, the controls behind a token
tensorfold-studio service install               # keep it running (launchd on macOS, a systemd user unit on Linux)
```

Tests: `pytest` (426) and `cd ui && npm test`; lint: `ruff check`. The page is prebuilt in `studio/ui_dist/`, so running
the Studio needs no Node; to work on the page: `cd ui && npm install && npm run dev` (proxies `/api` and `/engine` to
the Studio on 8090), then `npm run build` to refresh `studio/ui_dist/`.

## How it is built

- `studio/`: a Python package (FastAPI, Python 3.11+). It detects the machine (`machine.py`), computes for each catalog
  entry whether it fits the memory (`fit.py`, `catalog.py`), starts, adopts and stops the engine as a child process that
  outlives the Studio (`engine.py`, `native.py`, `engines.py`), downloads checkpoints into the Hugging Face cache
  (`pull.py`), relays the engine's HTTP under `/engine/` so the browser talks to one origin (`proxy.py`), and serves the
  page. Conversations, Bench runs and usage are kept in SQLite under the user's config folder.
- `ui/`: TypeScript, React and Vite, built to static files shipped inside the Python package.
- `tests/`: pytest against a stand-in engine (`studio/mock_engine.py`) that speaks the same HTTP as TensorFold.

The modules larger than CONTRIBUTING's 600 lines (`server.py`, `bench.py`, `living.py`, `engine.py`, `report.py`) split
naturally (the server by route group); left as they are until you say how the Studio should live in your repository.

## What it uses from TensorFold

The native binary (`tensorfold-native` or `tensorfold`, found on PATH, `TENSORFOLD_BIN`, Homebrew and the archives'
usual places):

- `capabilities --json`: the families and formats it serves on this chip, the `serve` flags;
- `info <model>`: a downloaded checkpoint's weights, memory floor and context;
- `serve <model> --host … --port …` with `--context`, `--thinking`/`--no-thinking`, `--slide` for a copy that learns,
  and the API key through `TENSORFOLD_API_KEY` (never on the command line); the startup lines are read for the admitted
  context, the load time and the CUDA sizing.

Its HTTP API: `/health`, `/metrics`, `/v1/models`, `/v1/chat/completions` (streamed, with the `tensorfold` and
`speculative` blocks, `"draft": false` for the exactness check, log-probabilities for KL), `/v1/messages`,
`/v1/responses`, and Living Weights' `/v1/slide/learn` and `/v1/slide/graph`. The frozen Python line (0.6.6) is used
only for the families the native engine does not serve yet, when it is installed in the Studio's environment.

## License

Apache License 2.0, as TensorFold. The logos in `ui/src/assets/` belong to their owners; each folder's README says
where each one comes from.
