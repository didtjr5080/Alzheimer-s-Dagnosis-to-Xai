"""Test-session setup: keep every cache/bytecode side effect inside this
extension's own outputs folder (never in clip_xai_app/ or the repo root), and
make `graph_xai` importable without requiring the caller to set PYTHONPATH.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_EXTENSION_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _EXTENSION_ROOT / "outputs" / ".cache"

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# The Windows default temp dir on this machine is a third-party (ESTsoft)
# redirector with restrictive permissions, which breaks pytest's tmp_path
# fixture cleanup. Point TMP/TEMP at a writable folder inside this extension.
_PYTEST_TMP_ROOT = _CACHE_ROOT / "pytest_tmp"
_PYTEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(_PYTEST_TMP_ROOT)
os.environ["TEMP"] = str(_PYTEST_TMP_ROOT)

for _env_var, _subdir in (
    ("HF_HOME", "hf"),
    ("TRANSFORMERS_CACHE", "transformers"),
    ("TORCH_HOME", "torch"),
    ("MPLCONFIGDIR", "mpl"),
    ("XDG_CACHE_HOME", "xdg"),
):
    _path = _CACHE_ROOT / _subdir
    _path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_env_var, str(_path))

if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))


def pytest_configure(config):
    # Force pytest's own tmp_path/tmp_path_factory basetemp into the same
    # writable folder, regardless of the invocation cwd -- the env vars above
    # are not enough because tempfile.gettempdir() may already be cached by
    # the time this conftest is imported.
    if not config.option.basetemp:
        config.option.basetemp = str(_PYTEST_TMP_ROOT)

