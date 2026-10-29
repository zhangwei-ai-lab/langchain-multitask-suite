"""项目入口。所有能力都收敛在 `app.cli`，这里只做分发。"""

from __future__ import annotations

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())
