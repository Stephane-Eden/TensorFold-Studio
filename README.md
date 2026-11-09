# TensorFold Studio

A web interface for [TensorFold](https://github.com/ashhart/TensorFold): pick a model that fits the machine, run it,
watch it work, chat with it, teach it (Living Weights), measure it and connect tools to it, from a browser tab.

**An independent project, not part of TensorFold.** TensorFold ships its own web interface, TensorFold Web
(`tensorfold web`), which began from this Studio; the two have gone their own ways since. The Studio is for those who
prefer its screens. TensorFold is Ash Hart's work; questions about the engine belong in
[its repository](https://github.com/ashhart/TensorFold), questions about the Studio here.

In daily use on a Mac mini M6 (Metal) and two DGX Spark GB10 (CUDA), with TensorFold 1.2.5. The Studio reads what the
installed engine offers (`capabilities --json`) and takes the newest native release it finds; a daily check
(`.github/workflows/engine-canary.yml`) runs it against the latest TensorFold release, so a change in the engine shows
up here first. `docs/features.md` describes every screen and the module behind it.

## Install

With [uv](https://docs.astral.sh/uv/), from this repository (Python 3.11 or later; the page is prebuilt, no Node):

```bash
uv tool install git+https://github.com/Stephane-Eden/TensorFold-Studio     # the `tensorfold-studio` command
uv tool upgrade tensorfold-studio                                           # later, to the latest main
uvx --from git+https://github.com/Stephane-Eden/TensorFold-Studio tensorfold-studio   # or try it without installing
```

`git+https://…@<tag>` installs a given release instead of the latest `main`. The Studio is not on PyPI.

## Run it

```bash
tensorfold-studio                               # http://127.0.0.1:8090, opens the page
tensorfold-studio --mock-engine                 # without a model: a stand-in engine on port 8080
tensorfold-studio --engine-url http://spark.local:8080   # watch a TensorFold running elsewhere
tensorfold-studio --host 0.0.0.0 --token <secret>        # reachable from other machines, the controls behind a token
tensorfold-studio service install               # keep it running (launchd on macOS, a systemd user unit on Linux)
```

TensorFold itself: `curl -fsSL https://tensorfold.dev/install.sh | sh` (see its README).

## Work on it

```bash
git clone https://github.com/Stephane-Eden/TensorFold-Studio && cd TensorFold-Studio
uv venv .venv && . .venv/bin/activate          # or python -m venv
uv pip install -e '.[dev]'
```

Tests: `pytest` and `cd ui && npm test`; lint: `ruff check`. The page is prebuilt in `studio/ui_dist/`, so running the
Studio needs no Node; to work on the page: `cd ui && npm install && npm run dev` (proxies `/api` and `/engine` to the
Studio on 8090), then `npm run build` to refresh `studio/ui_dist/`.

## Beside TensorFold Web

TensorFold Web kept the Studio's defaults, so on one machine:

- **Ports.** Both listen on 8090 and start their engine on 8080. The Studio says so when Web holds them: start it with
  `--port 8091` (`tensorfold-studio service install --port 8091`), and give its engine another address in Settings,
  such as `http://127.0.0.1:8081`.
- **Living copies.** The Studio keeps its own in `~/tensorfold-studio-living` (`TENSORFOLD_STUDIO_LIVING` to choose);
  `~/tensorfold-living` is Web's. A Studio that used `~/tensorfold-living` before Web was installed keeps it. Each one
  recovers its folder when it starts, so two of them must never share one: the Studio warns when they do.
- **Models.** Both use the Hugging Face cache: a model downloaded by one is there for the other, and a model removed by
  one is gone for both.

## How it is built

- `studio/`: a Python package (FastAPI, Python 3.11+). It detects the machine (`machine.py`), computes for each catalog
  entry whether it fits the memory (`fit.py`, `catalog.py`), starts, adopts and stops the engine as a child process that
  outlives the Studio (`engine.py`, `native.py`, `engines.py`), downloads checkpoints into the Hugging Face cache
  (`pull.py`), relays the engine's HTTP under `/engine/` so the browser talks to one origin (`proxy.py`), and serves the
  page. Conversations, Bench runs and usage are kept in SQLite under the user's config folder.
- `ui/`: TypeScript, React and Vite, built to static files shipped inside the Python package.
- `tests/`: pytest against a stand-in engine (`studio/mock_engine.py`) that speaks the same HTTP as TensorFold.

## What it uses from TensorFold

The native binary (`tensorfold-native` or `tensorfold`, found on PATH, `TENSORFOLD_BIN`, the installer's `~/.tensorfold`,
Homebrew and the archives' usual places):

- `capabilities --json`: the families and formats it serves on this chip, the `serve` flags;
- `info <model>`: a downloaded checkpoint's weights, memory floor and context;
- `serve <model> --host … --port …` with `--context`, `--thinking`/`--no-thinking`, `--slide` and `--slide-graph` for a
  copy that learns, `--drafter` for a separate draft model, `--cortheon` when Settings' Web research is on (off by
  default), and the API key through `TENSORFOLD_API_KEY` (never on the command line); the startup lines are read for
  the admitted context, the load time and the memory sizing.

Its HTTP API: `/health`, `/metrics`, `/v1/models` (and the model's capabilities, such as whether thinking can be
turned off), `/v1/chat/completions` (streamed, with the `tensorfold` and `speculative` blocks, `"draft": false` for the
exactness check, log-probabilities for KL), `/v1/completions` with `prompt_logprobs` (KL over every position of a fixed
text), `/v1/messages`, `/v1/responses`, and Living Weights' `/v1/slide/learn` and `/v1/slide/graph`. The frozen Python
line (0.6.6) is used only for the families the native engine does not serve, when it is installed in the Studio's
environment.

## License

Apache License 2.0, as TensorFold. "TensorFold" names Ash Hart's engine, which this Studio drives; the Studio is not
endorsed by it. The logos in `ui/src/assets/` belong to their owners; each folder's README says where each one comes
from.
