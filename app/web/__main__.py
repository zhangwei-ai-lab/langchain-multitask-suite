"""`python -m app.web` 的入口：等价于 `python serve.py`。"""

from __future__ import annotations

from .launcher import main

if __name__ == "__main__":
    raise SystemExit(main(prog="python -m app.web"))
