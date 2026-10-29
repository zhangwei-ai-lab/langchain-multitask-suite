"""Web 界面层：FastAPI 路由 + SSE 流式输出，把同一套能力搬到浏览器里。

设计要点
--------
1. **只做传输，不做业务**：路由负责「解析请求 → 调 tasks.py 拿事件生成器 →
   编码成 SSE」，业务逻辑全部留在 agents / chains 里，命令行与界面共用同一份实现。
2. **失败要可读**：模型没配好、密钥没额度、工具报错，都以 `{"type":"error"}` 事件
   推给前端展示，绝不让浏览器收到一屏 Python traceback。
3. **状态接口不碰模型**：`/api/status` 只做探测（Ollama 有没有起、密钥有没有配），
   不建立任何模型连接，所以页面刷新再多次也不会额外烧 token。
4. **单文件前端**：`static/` 下没有构建步骤，`python serve.py` 就是全部安装流程。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import config
from ..llm import (
    LLMUnavailable,
    explain_error,
    get_llm,
    probe,
    set_runtime_provider,
    VALID_PROVIDERS,
)
from ..tools import catalog
from . import tasks as tasklib

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app = FastAPI(
    title="LangChain 多任务应用 · 可视化控制台",
    description="本地 Ollama 与云端 API 双后端，六类任务统一入口。",
    version="1.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------- 请求模型
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default="default", max_length=64)


class TaskRequest(BaseModel):
    task: str = Field(min_length=1, max_length=32)
    input: str = Field(default="", max_length=8000)


class ProviderRequest(BaseModel):
    provider: str = Field(default="auto", max_length=16)
    model: str = Field(default="", max_length=128)


# ---------------------------------------------------------------- SSE 封装
def _sse(events: Iterator[dict]) -> Iterator[str]:
    """事件字典 → SSE 文本帧。

    统一用 `data: {json}` 单通道而不是 `event:` 分类型：
    浏览器侧只需一个 `onmessage` 回调 + `switch (payload.type)`，
    少一套事件名注册，也不会出现「加了新事件类型前端静默丢弃」。
    """
    try:
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except LLMUnavailable as exc:
        payload = {"type": "error", "message": str(exc), "kind": "llm_unavailable"}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except KeyError as exc:
        payload = {"type": "error", "message": str(exc).strip("'\""), "kind": "bad_request"}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except Exception as exc:  # noqa: BLE001
        # 厂商原始错误（`RateLimitError: Error code: 429 - {...}`）用户看不懂，
        # 能翻译就翻成「原因 + 下一步」，翻不出才退回类型名，且不泄露服务器路径。
        payload = {
            "type": "error",
            "message": explain_error(exc) or f"{type(exc).__name__}: {exc}",
            "kind": "runtime",
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 关掉 nginx 之类的代理缓冲，否则前端会等到整轮结束才一次性收到
    "X-Accel-Buffering": "no",
}


# ---------------------------------------------------------------- 页面
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="前端文件缺失：app/web/static/index.html")
    return FileResponse(str(INDEX_FILE))


# ---------------------------------------------------------------- 状态
@app.get("/api/status")
def status(cloud: bool = Query(default=False)) -> dict:
    """环境与后端口径：页面顶部状态条、模型下拉框都读这里。"""
    info = probe(cloud=cloud)
    info["tasks"] = tasklib.task_catalog()
    info["providers"] = list(VALID_PROVIDERS)
    info["chat_tools"] = tasklib.CHAT_TOOL_NAMES
    info["chat_turns"] = {sid: len(h) for sid, h in tasklib._CHAT_HISTORY.items()}
    return info


@app.get("/api/tools")
def tools() -> dict:
    return {"tools": [{"name": name, "desc": desc} for name, desc in catalog()]}


@app.post("/api/provider")
def switch_provider(req: ProviderRequest) -> dict:
    """运行时切换后端：不重启进程、不改 .env，界面上点一下就生效。"""
    try:
        set_runtime_provider(req.provider, req.model or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    info = probe()
    return {
        "ok": True,
        "runtime_provider": info["runtime_provider"],
        "runtime_model": info["runtime_model"],
        "backend": info.get("backend_info"),
        "backend_banner": info["backend"],
    }


@app.post("/api/provider/test")
def test_provider(req: ProviderRequest) -> JSONResponse:
    """真发一次最小请求验证连通性 —— 「配置看起来对」和「真的能出字」是两件事。

    故意不复用当前缓存实例：测试要能验证「即将切换到的后端」，
    而不是正在用的那一个。
    """
    started = time.perf_counter()
    try:
        llm = get_llm(temperature=0.0, provider=req.provider, model=req.model or None)
        backend = getattr(llm, "_backend", None)
        reply = llm.invoke("只回复两个字：收到")
        text = (getattr(reply, "content", "") or "").strip()
        return JSONResponse(
            {
                "ok": bool(text),
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "reply": text[:120],
                "backend": backend.as_dict() if backend else None,
                "error": "" if text else "模型返回了空内容",
            }
        )
    except LLMUnavailable as exc:
        return JSONResponse(
            {
                "ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "reply": "",
                "backend": None,
                "error": str(exc),
            }
        )
    except Exception as exc:  # noqa: BLE001
        # 云端常见的 401/403/429 都会走到这里，原样带出去让用户知道该改哪里
        return JSONResponse(
            {
                "ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "reply": "",
                "backend": None,
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_code=200,
        )


# ---------------------------------------------------------------- 对话与任务
@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _sse(tasklib.run_chat(req.message, req.session_id)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/chat/reset")
def chat_reset(session_id: str = Query(default="default")) -> dict:
    tasklib.reset_chat(session_id)
    return {"ok": True}


@app.post("/api/task")
def run_task(req: TaskRequest) -> StreamingResponse:
    return StreamingResponse(
        _sse(tasklib.run_task(req.task, req.input)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ---------------------------------------------------------------- 报告
def _report_path(name: str) -> Path:
    """只允许读 outputs 目录下的 .md 文件：`../` 之类的路径在拼接前就被拒掉。"""
    if "/" in name or "\\" in name or not name.endswith(".md"):
        raise HTTPException(status_code=400, detail="只允许读取 outputs 目录下的 .md 文件")
    path = (config.OUTPUT_DIR / name).resolve()
    if path.parent != config.OUTPUT_DIR.resolve():
        raise HTTPException(status_code=400, detail="非法的报告名")
    if not path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    return path


@app.get("/api/reports")
def reports() -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(config.OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            }
        )
    return {"dir": str(config.OUTPUT_DIR), "reports": items}


@app.get("/api/reports/{name}", response_class=PlainTextResponse)
def report_content(name: str) -> PlainTextResponse:
    return PlainTextResponse(_report_path(name).read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "output_dir": str(config.OUTPUT_DIR)}


__all__ = ["app"]
