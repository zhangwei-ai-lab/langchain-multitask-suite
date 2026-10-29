"""Web 界面层：浏览器里的同一个多任务应用。

模块分工
--------
tasks.py    任务注册表：能力 → 流式事件生成器（业务装配都在这）
server.py   FastAPI 路由：请求 → 事件 → SSE，只做传输
static/     单文件前端，无需 npm、无需打包

启动方式（任选其一）
--------------------
python serve.py
python main.py web --port 8000
python -m app.web
"""

from __future__ import annotations

__all__ = ["server", "tasks"]
