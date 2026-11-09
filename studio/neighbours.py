"""TensorFold Web, the engine's own web interface, began as a copy of the Studio and kept its defaults: the same ports
(8090 for the page, 8080 for the engine) and the same folder of living copies, `~/tensorfold-living`, with the same
`.studio` records. Both recover that folder when they start (a restore left halfway is undone, staging folders go),
so the two must never share it while both run. What the Studio knows of TensorFold Web on this machine, for this user:
read only, nothing of Web's is ever changed."""

from __future__ import annotations

import os
import platform
from pathlib import Path

NAME = "TensorFold Web"


def config_dir() -> Path:
    """TensorFold Web's settings folder (its settings.config_dir())."""

    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "TensorFold Web"
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "TensorFold Web"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tensorfold-web"


def installed() -> bool:
    """Has TensorFold Web ever run for this user: it makes its settings folder at its first start."""

    return config_dir().is_dir()


def running() -> bool:
    """Is a TensorFold Web running for this user now: it holds its `web.lock` for its whole life. The lock is tried
    and let go at once; a missing file is never made."""

    from . import instance

    path = config_dir() / "web.lock"
    if not path.is_file():
        return False
    try:
        with open(path, "r+") as f:
            return not instance._try_lock(f)       # closing the file lets the lock go
    except OSError:
        return False


def _env_value(key: str) -> str | None:
    """A KEY=value from Web's `web.env`, the environment its service starts with."""

    try:
        lines = (config_dir() / "web.env").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() == key:
            return v.strip().strip("'\"") or None
    return None


def living_root() -> Path:
    """The folder TensorFold Web keeps its living copies in, as its service would start."""

    return Path(_env_value("TENSORFOLD_WEB_LIVING") or Path.home() / "tensorfold-living").expanduser()


def shares_living(root: Path) -> bool:
    """Does TensorFold Web, set up here, keep its copies in the Studio's own folder?"""

    if not installed():
        return False
    try:
        return living_root().resolve() == root.resolve()
    except OSError:
        return False
