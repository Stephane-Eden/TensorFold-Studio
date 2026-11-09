"""The Studio against a TensorFold release: what it reads of the engine, checked on the binary and on the source of
the same tag. Skipped unless the release is given (the `engine-canary` workflow downloads the latest one):

    TENSORFOLD_CANARY_BIN=/path/to/tensorfold-native   the release's binary (version, capabilities, `serve --help`)
    TENSORFOLD_CANARY_SOURCE=/path/to/TensorFold       its source at the same tag (the log lines the Studio parses)

The log lines are not copied here: each format string is read from the source, filled with sample values, and given
to the Studio's own parser. A line the engine changes, or a value it adds, fails here before it fails on a machine.
"""

import os
import re
from pathlib import Path

import pytest

from studio import native

BIN = os.environ.get("TENSORFOLD_CANARY_BIN")
SOURCE = os.environ.get("TENSORFOLD_CANARY_SOURCE")
needs_bin = pytest.mark.skipif(not BIN, reason="TENSORFOLD_CANARY_BIN is not set")
needs_source = pytest.mark.skipif(not SOURCE, reason="TENSORFOLD_CANARY_SOURCE is not set")

# the flags the Studio's command line may carry (native.launch), and whether each takes a value
SERVE_FLAGS = {"--host": True, "--port": True, "--context": True, "--drafter": True, "--slide-graph": True,
               "--slide": False, "--thinking": False, "--no-thinking": False, "--dashboard": False, "--cortheon": False}
LEARN_EVENTS = ("fact", "learning", "learned", "saved", "failed")     # living.Teacher reads these from /v1/slide/learn
PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def _zig() -> list[tuple[Path, str]]:
    root = Path(SOURCE) / "zig" / "src"
    return [(p, p.read_text(errors="replace")) for p in root.rglob("*.zig") if not p.name.endswith("_test.zig")]


def _format(pattern: str) -> str:
    """The one string literal of the engine's source the pattern finds (a format string)."""

    found = {m.group(1) for _, text in _zig() for m in re.finditer(r'"((?:[^"\\\n]|\\.)*)"', text)
             if re.search(pattern, m.group(1))}
    assert found, f"no format string matches {pattern!r}: the engine changed this line"
    assert len(found) == 1, f"{pattern!r} matches several format strings: {sorted(found)}"
    return found.pop()


def _fill(fmt: str, *values: str) -> str:
    holes = PLACEHOLDER.findall(fmt)
    assert len(holes) == len(values), f"{len(holes)} values in {fmt!r} now, {len(values)} known to the Studio"
    it = iter(values)
    return PLACEHOLDER.sub(lambda _m: next(it), fmt).replace("\\n", "")


@needs_source
def test_the_serving_line():
    fmt = _format(r"^serving \{s\} at http")
    line = _fill(fmt, "nemotron", "127.0.0.1", "8080", "temperature 0.6, top-p 0.95", "MTP + copies", "262144", "21.4")
    assert native.parse_startup(line) == {"context_tokens": 262144.0, "loaded_s": 21.4, "drafts": 1.0}
    assert native.parse_startup(_fill(fmt, "m", "127.0.0.1", "8080", "greedy", "none", "unlimited", "3.2")) == \
        {"loaded_s": 3.2, "drafts": 0.0}


@needs_source
def test_the_cuda_sizing_line():
    fmt = _format(r"^CUDA sm_\{d\} device")
    line = _fill(fmt, "121", "0", "NVIDIA GB10", ", memory shared with the host", "18.49",
                 " and 25.54 GiB kept for --slide's trainer", "8", "s", "0.18 GiB each at a 16384-token window", "36.8",
                 "12.2", "2048", "", "", "")
    assert native.parse_startup(line) == {"resident_gib": 18.49, "streams": 8.0, "stream_gib": 0.18, "left_gib": 36.8,
                                          "reserve_gib": 12.2, "trainer_gib": 25.54}
    assert "kept for --slide's trainer" in _format(r"GiB kept for --slide's trainer")


@needs_source
def test_the_metal_prompt_cache_line():
    fmt = _format(r"^prompt cache: \{d:\.1\} GiB from")
    line = _fill(fmt, "1.8", "1.8", "70% cap", "18.6", "", "64", "2", "", "", "")
    assert native.parse_startup(line) == {"prompt_cache_gib": 1.8, "prompt_cache_free_gib": 1.8, "in_use_gib": 18.6}


@needs_source
def test_refusals_keep_their_names():
    names = {m.group(1) for _, text in _zig() for m in re.finditer(r"error\.(\w+)", text)}
    refusals = {n for n in names if n.endswith(("QualificationRequired", "NotServed"))}
    assert "GemmaMoeNotServed" in refusals or any(n.endswith("NotServed") for n in refusals)
    line = _fill(_format(r"^the native Gemma 4 engine cannot load this checkpoint"), "GemmaMoeNotServed")
    assert native.refusal_in([line]) == "GemmaMoeNotServed"


@needs_source
def test_info_lines():
    fam = _fill(_format(r"^family: \{s\} \(\{s\}\)"), "Nemotron 3", "nemotron_h")
    ctx = _fill(_format(r"^context: \{d\}"), "262144")
    weights = _fill(_format(r"^weights: \{d:\.2\} GiB"), "18.49", "19853394944")
    floor = _fill(_format(r"^memory floor: \{d:\.2\} GiB"), "21.00", "2.50")
    assert native.parse_info(f"{fam}\n{ctx}\n{weights}\n{floor}") == {
        "family": "Nemotron 3 (nemotron_h)", "context": 262144, "weights_bytes": 19853394944,
        "floor_bytes": int(21.0 * 1024**3)}


@needs_source
def test_the_learning_stream_and_the_models_capabilities():
    slide = "".join(text for p, text in _zig() if p.name.startswith("slide"))
    for event in LEARN_EVENTS:
        assert f'event("{event}"' in slide, f"/v1/slide/learn no longer sends `{event}`"
    assert any("thinking_switch" in text for _, text in _zig()), "/v1/models no longer says thinking_switch"


@needs_bin
def test_the_binary_says_what_the_studio_asks():
    native.forget()
    assert native.version_of(BIN), "`--version` is not `tensorfold-native X` any more"
    caps = native.capabilities(BIN)
    assert caps.get("version") and "chip" in caps          # null on a machine without a GPU it knows (the CI)
    families = caps.get("families")
    assert isinstance(families, dict) and families, "capabilities: `families` is not a map of family to formats"
    assert all(isinstance(v, list) for v in families.values())
    offered = native.serve_flags(BIN)
    missing = sorted(set(SERVE_FLAGS) - set(offered))
    assert not missing, f"`serve` no longer offers {missing}"
    for flag, takes_value in SERVE_FLAGS.items():
        assert offered[flag]["value"] in (takes_value, None), f"{flag} {'needs' if takes_value else 'takes no'} value"
