"""pytest 公共配置：把项目根目录放进 sys.path，测试里直接 `from app...` 导入。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
