"""启动逻辑的唯一实现，`serve.py` 与 `main.py web` 都转调这里。

放在包里而不是只写在 serve.py：CLI 也需要「一条命令起界面」，
两边各抄一份 argparse 迟早会漂移（端口默认值对不上之类）。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser


def build_parser(prog: str = "serve.py") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="启动 LangChain 多任务控制台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只本机可访问")
    parser.add_argument("--port", type=int, default=8000, help="端口，默认 8000")
    parser.add_argument("--reload", action="store_true", help="改代码自动重启（开发用）")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    return parser


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, open_browser: bool = False) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "缺少 Web 依赖。先安装：\n  pip install -r requirements.txt\n"
            "（本项目只多依赖 fastapi / uvicorn / langchain-openai 三个包）",
            file=sys.stderr,
        )
        return 1

    url = f"http://{'127.0.0.1' if host in {'0.0.0.0', '::'} else host}:{port}"
    print(f"\n  控制台地址： {url}")
    print(f"  接口文档：   {url}/docs")
    print("  停止服务：   Ctrl+C\n")

    if open_browser:
        # 等 uvicorn 把端口绑好再开浏览器，否则打开的是一个「无法访问」的页面
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()

    uvicorn.run("app.web.server:app", host=host, port=port, reload=reload, log_level="info")
    return 0


def main(argv: list[str] | None = None, prog: str = "serve.py") -> int:
    args = build_parser(prog).parse_args(argv)
    return run(args.host, args.port, args.reload, args.open)
