"""模型工厂：把「用哪个模型、怎么连」这件事从业务代码里彻底剥离。

为什么需要这一层
----------------
业务代码里如果散落着 `ChatTongyi(model_name="qwen-turbo", api_key=...)`，
换模型就要改 N 个文件，换台机器没密钥就整条链跑不动。这里做三件事：

1. **后端抽象**：本地 Ollama / 云端 DashScope 走同一个 `get_llm()` 出口，
   业务代码只拿到一个「可 invoke、可 bind_tools 的 ChatModel」，不关心它是什么。
2. **自动探测 + 明确回退**：`LLM_PROVIDER=auto` 时先探 `OLLAMA_BASE_URL/api/tags`，
   有可用对话模型就用本地（零密钥零费用），否则才找云端密钥。
3. **把不可用变成可读的提示**：全都连不上时抛 `LLMUnavailable`，
   错误信息里直接列出三条自救路径，而不是让上层拿到一个 urllib 的 URLError。

环境变量
--------
LLM_PROVIDER     auto | ollama | dashscope          默认 auto
LLM_MODEL        指定模型名，留空则按内置优先级挑
LLM_TEMPERATURE  默认温度，默认 0.2
OLLAMA_BASE_URL  默认 http://127.0.0.1:11434
OLLAMA_TIMEOUT   探测超时（秒），默认 2
DASHSCOPE_API_KEY / DASHSCOPE_MODEL
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import config

# 自动挑选时的优先级：优先「通用对话 + 支持 tool calling」的模型。
# 只影响自动挑选；显式设置 LLM_MODEL 时以用户配置为准。
PREFERRED_LOCAL_MODELS = (
    "qwen3:8b",
    "qwen3-8b-local:latest",
    "qwen2.5:7b",
    "qwen2.5-coder-7b-local:latest",
    "llama3.1:8b",
    "gemma3:4b",
)


class LLMUnavailable(RuntimeError):
    """没有任何可用后端时抛出，携带给用户看的自救指引。"""


@dataclass
class Backend:
    provider: str
    model: str
    base_url: str = ""
    extra: dict = field(default_factory=dict)

    def banner(self) -> str:
        if self.provider == "ollama":
            return f"[LLM] 本地 Ollama | {self.model} | {self.base_url}"
        return f"[LLM] 云端 DashScope | {self.model}"


def ollama_base_url() -> str:
    return (config.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "").rstrip("/")


def list_local_models(timeout: float | None = None) -> list[str]:
    """读 Ollama 已下载模型清单。服务没起、超时、返回体畸形，一律当「没有」处理。"""
    timeout = timeout if timeout is not None else config.get_float("OLLAMA_TIMEOUT", 2.0)
    try:
        with urllib.request.urlopen(f"{ollama_base_url()}/api/tags", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []

    names: list[str] = []
    for item in payload.get("models") or []:
        name = item.get("name") or item.get("model")
        if not name:
            continue
        # 向量模型能列出来但不能对话，必须排除，否则会被误选成默认模型
        families = " ".join((item.get("details") or {}).get("families") or []).lower()
        if "embedding" in name.lower() or "embedding" in families or "bert" in families:
            continue
        names.append(name)
    return names


def pick_local_model(available: list[str], preferred: str | None = None) -> str | None:
    """挑选顺序：用户指定 > 内置优先级（精确匹配）> 内置优先级（同名前缀）> 列表第一个。"""
    if not available:
        return None
    if preferred:
        for name in available:
            if name == preferred:
                return name
        prefix = preferred.split(":")[0]
        for name in available:
            if name.split(":")[0] == prefix:
                return name
    for want in PREFERRED_LOCAL_MODELS:
        if want in available:
            return want
    for want in PREFERRED_LOCAL_MODELS:
        prefix = want.split(":")[0]
        for name in available:
            if name.split(":")[0] == prefix:
                return name
    return available[0]


def resolve_backend(provider: str | None = None, model: str | None = None) -> Backend:
    """决定本次用哪个后端，只做决策不建立连接。"""
    provider = (provider or config.get("LLM_PROVIDER", "auto") or "auto").lower()
    model = model or config.get("LLM_MODEL")
    problems: list[str] = []

    if provider in {"auto", "ollama"}:
        available = list_local_models()
        chosen = pick_local_model(available, model)
        if chosen:
            return Backend("ollama", chosen, ollama_base_url(), {"available": available})
        if available:
            problems.append(f"本地 Ollama 可连通，但没有可用的对话模型（已装：{', '.join(available)}）")
        else:
            problems.append(f"本地 Ollama（{ollama_base_url()}）未响应或未安装任何模型")

    if provider in {"auto", "dashscope"}:
        key = config.get("DASHSCOPE_API_KEY")
        if key:
            return Backend("dashscope", model or config.get("DASHSCOPE_MODEL", "qwen-turbo"))
        problems.append("未配置 DASHSCOPE_API_KEY")

    raise LLMUnavailable(
        "没有可用的模型后端：\n  - "
        + "\n  - ".join(problems)
        + "\n三条自救路径（任选其一）：\n"
        "  A. 本地跑（推荐，零费用）：装好 Ollama 后 `ollama pull qwen3:8b`，"
        "确认 `curl http://127.0.0.1:11434/api/tags` 有输出；\n"
        "  B. 走云端：在阿里云百炼申请 DASHSCOPE_API_KEY，写进 .env；\n"
        "  C. 先只验工程：`python main.py doctor` 检查环境，`python -m pytest -q` 跑离线单测。"
    )


_CACHE: dict[tuple, object] = {}


def get_llm(
    temperature: float | None = None,
    model: str | None = None,
    provider: str | None = None,
    streaming: bool = False,
):
    """返回 ChatModel。参数相同的调用复用同一实例（避免重复探测、重复建连）。"""
    temperature = (
        temperature if temperature is not None else config.get_float("LLM_TEMPERATURE", 0.2)
    )
    key = (temperature, model, provider, streaming)
    if key in _CACHE:
        return _CACHE[key]

    backend = resolve_backend(provider=provider, model=model)

    if backend.provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict = {
            "model": backend.model,
            "base_url": backend.base_url,
            "temperature": temperature,
        }
        if streaming:
            kwargs["streaming"] = True
        try:
            # qwen3 这类带思考链的模型默认会把 <｜end▁of▁thinking｜> 段吐进正文，
            # 关掉后输出形态才和普通对话模型一致（老版本没有这个参数，降级即可）
            llm = ChatOllama(reasoning=False, **kwargs)
        except TypeError:
            llm = ChatOllama(**kwargs)
    else:
        try:
            from langchain_community.chat_models import ChatTongyi
        except ImportError as exc:  # 可选依赖缺失时给出可执行的安装命令
            raise LLMUnavailable(
                "已选择 DashScope 后端，但缺少可选依赖。执行：\n"
                "  pip install langchain-community dashscope\n"
                f"（原始错误：{exc}）"
            ) from exc

        kwargs = {
            "model_name": backend.model,
            "dashscope_api_key": config.get("DASHSCOPE_API_KEY"),
            "temperature": temperature,
        }
        if streaming:
            kwargs["streaming"] = True
        llm = ChatTongyi(**kwargs)

    # 挂后端信息，脚本/报告里可以打出「这次到底走了哪条路」
    llm._backend = backend  # type: ignore[attr-defined]
    _CACHE[key] = llm
    return llm


def describe(llm=None) -> str:
    llm = llm or get_llm()
    backend = getattr(llm, "_backend", None)
    return backend.banner() if backend else "[LLM] 未知后端"


def probe() -> dict:
    """给 `main.py doctor` 用的环境体检：不建立模型连接，只回答「能不能用」。"""
    available = list_local_models()
    info = {
        "ollama_url": ollama_base_url(),
        "ollama_reachable": bool(available),
        "local_models": available,
        "picked": pick_local_model(available, config.get("LLM_MODEL")),
        "dashscope_key": bool(config.get("DASHSCOPE_API_KEY")),
        "provider": config.get("LLM_PROVIDER", "auto"),
    }
    try:
        info["backend"] = resolve_backend().banner()
    except LLMUnavailable as exc:
        info["backend"] = f"不可用：{exc}".splitlines()[0]
    return info
