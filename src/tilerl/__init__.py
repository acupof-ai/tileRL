"""tileRL: TileLang inference + training (CPU/CUDA/Metal).

One TileLang backend, torch as the tensor container only, a hand-written
reverse-mode autograd tape. Import submodules directly::

    from tilerl import config, model, engine, train, server, cli
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pyproject.toml [project].version is the one source; server.py reads this
    __version__ = version("tilerl")
except PackageNotFoundError:  # source tree with no install (e.g. bare PYTHONPATH)
    __version__ = "0.0.0+unknown"
