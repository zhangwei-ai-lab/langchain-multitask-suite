"""一键启动 Web 控制台：`python serve.py`

为什么单独放一个文件而不是只给 `python -m app.web`
--------------------------------------------------
README 里第一条命令必须是「复制粘贴就能跑」的形状。`serve.py` 在项目根目录，
路径最短、最好记；真正干活的是 `app/web/launcher.py`，这里只做转发。

常用参数
--------
python serve.py                  # http://127.0.0.1:8000
python serve.py --port 8010      # 换端口
python serve.py --open           # 起来后自动打开浏览器
python serve.py --reload         # 改代码自动重启（开发时用）
python serve.py --host 0.0.0.0   # 局域网内其他设备也能访问（注意：没有鉴权）
"""

from __future__ import annotations

from app.web.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
