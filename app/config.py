"""配置层：环境变量、目录定位、运行产物落盘。

设计取舍
--------
1. 不依赖 python-dotenv：一个 20 行的解析器足够用，少一个依赖就少一个安装失败点。
2. 一切可调项都走环境变量，代码里不出现任何密钥、模型名、超时数字。
3. 相对路径统一以「项目根目录」为基准，保证从任何工作目录启动都不会找不到文件。
4. 读不到配置不抛异常，退回默认值 —— 配置缺失只是能力降级，不该让进程直接死。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ENV_FILE = PROJECT_ROOT / ".env"
ENV_LOCAL_FILE = PROJECT_ROOT / ".env.local"


def _parse_env(path: Path, override: bool) -> None:
    """把 KEY=VALUE 逐行写进 os.environ。支持 # 注释与成对引号。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def load_env() -> None:
    """.env 打底，.env.local 覆盖（本地私有配置优先于入库模板）。"""
    _parse_env(ENV_FILE, override=False)
    _parse_env(ENV_LOCAL_FILE, override=True)


load_env()


def get(name: str, default: str | None = None) -> str | None:
    """读字符串。空字符串按「未设置」处理，避免 .env 里留空的项被当成有效值。"""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def get_bool(name: str, default: bool = False) -> bool:
    value = get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_int(name: str, default: int) -> int:
    try:
        return int(get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def get_float(name: str, default: float) -> float:
    try:
        return float(get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def save_report(prefix: str, content: str) -> Path:
    """把运行结果写到 outputs/<prefix>-<时间戳>.md，文件名带秒级时间戳避免互相覆盖。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"{prefix}-{stamp}.md"
    path.write_text(content, encoding="utf-8")
    return path
