"""模型工厂：把「用哪个模型、怎么连」这件事从业务代码里彻底剥离。

为什么需要这一层
----------------
业务代码里如果散落着 `ChatTongyi(model_name="qwen-turbo", api_key=...)`，
换模型就要改 N 个文件，换台机器没密钥就整条链跑不动。这里做三件事：

1. **后端抽象**：本地 Ollama / 云端 OpenAI 兼容 API 走同一个 `get_llm()` 出口，
   业务代码只拿到一个「可 invoke、可 bind_tools、可 stream 的 ChatModel」，不关心它是什么。
2. **自动探测 + 明确回退**：`LLM_PROVIDER=auto` 时先探 `OLLAMA_BASE_URL/api/tags`，
   有可用对话模型就用本地（零密钥零费用），否则才找云端密钥。
3. **把不可用变成可读的提示**：全都连不上时抛 `LLMUnavailable`，
   错误信息里直接列出三条自救路径，而不是让上层拿到一个 urllib 的 URLError。

两条接入路径（都能跑通，二选一）
--------------------------------
A. 本地 Ollama   —— 零密钥零费用，装好 `ollama pull qwen3:8b` 即可；
B. 云端 API Key  —— 任何 OpenAI 兼容端点：DashScope（阿里云百炼）、DeepSeek、
   Kimi、智谱、硅基流动、OpenAI 官方…… 只改 `CLOUD_BASE_URL` 一行就能换厂商。

云端为什么走 OpenAI 兼容协议而不是各家 SDK
--------------------------------------------
各家 SDK 的入参、流式事件、工具调用格式都不一样，接一家就要多一个可选依赖。
而国内主流厂商都提供了 `/v1/chat/completions` 兼容端点，一套 `langchain-openai`
覆盖全部，「换厂商」从改代码降级成改一个环境变量。

环境变量
--------
LLM_PROVIDER       auto | ollama | cloud           默认 auto（cloud 是 dashscope 的新名字）
LLM_MODEL          指定模型名，留空则按内置优先级挑
LLM_TEMPERATURE    默认温度，默认 0.2

OLLAMA_BASE_URL    默认 http://127.0.0.1:11434
OLLAMA_TIMEOUT     探测超时（秒），默认 2

CLOUD_BASE_URL     云端端点，默认阿里云百炼兼容模式
CLOUD_API_KEY      云端密钥；留空时回退读 DASHSCOPE_API_KEY
CLOUD_MODEL        云端模型，留空时回退 DASHSCOPE_MODEL，再退 qwen-turbo
CLOUD_TIMEOUT      云端单次请求超时（秒），默认 60
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

# 阿里云百炼的 OpenAI 兼容端点。换成别的厂商只改 CLOUD_BASE_URL，代码不用动。
DEFAULT_CLOUD_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CLOUD_MODEL = "qwen-turbo"

# 历史名字与新名字都接受：老用户 .env 里写的 dashscope 不该因为改名而失效
PROVIDER_ALIASES = {
    "dashscope": "cloud",
    "openai": "cloud",
    "api": "cloud",
    "cloud": "cloud",
    "ollama": "ollama",
    "auto": "auto",
}

VALID_PROVIDERS = ("auto", "ollama", "cloud")


class LLMUnavailable(RuntimeError):
    """没有任何可用后端时抛出，携带给用户看的自救指引。"""


@dataclass
class Backend:
    provider: str  # ollama | cloud
    model: str
    base_url: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "本地" if self.provider == "ollama" else "云端"

    def banner(self) -> str:
        if self.provider == "ollama":
            return f"[LLM] 本地 Ollama | {self.model} | {self.base_url}"
        return f"[LLM] 云端 API | {self.model} | {self.base_url}"

    def as_dict(self) -> dict:
        """给 Web 界面用：不含任何密钥，只回答「现在用的是谁、连在哪」。"""
        return {
            "provider": self.provider,
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "banner": self.banner(),
        }


def ollama_base_url() -> str:
    return (config.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "").rstrip("/")


def cloud_settings() -> dict:
    """聚合云端配置，处理「新变量没写就回退老变量」的兼容逻辑。"""
    return {
        "base_url": (config.get("CLOUD_BASE_URL") or DEFAULT_CLOUD_BASE_URL).rstrip("/"),
        "api_key": config.get("CLOUD_API_KEY") or config.get("DASHSCOPE_API_KEY") or "",
        "model": config.get("CLOUD_MODEL") or config.get("DASHSCOPE_MODEL") or DEFAULT_CLOUD_MODEL,
    }


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


def list_cloud_models(timeout: float | None = None) -> tuple[list[str], str]:
    """读云端 `/models` 清单。返回 (模型 id 列表, 错误说明)。

    AI 厂商的模型 id 每月都在变，写死在代码里的下拉框很快就过时；
    能拉到就按真实清单给用户选，拉不到就退回手填 + 一条可读的原因。
    """
    settings = cloud_settings()
    if not settings["api_key"]:
        return [], "未配置云端密钥（CLOUD_API_KEY / DASHSCOPE_API_KEY）"
    timeout = timeout if timeout is not None else 8.0
    req = urllib.request.Request(
        f"{settings['base_url']}/models",
        headers={"Authorization": f"Bearer {settings['api_key']}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return [], f"云端返回 HTTP {exc.code}（密钥无效或额度受限）"
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        return [], f"连不上云端端点：{type(exc).__name__}"

    items = payload.get("data") if isinstance(payload, dict) else None
    ids = [str(item.get("id")) for item in (items or []) if item.get("id")]
    return ids, "" if ids else "云端端点没有返回任何模型"


# ---------------------------------------------------------------- 运行时切换
# Web 界面允许在不动 .env、不重启进程的前提下换后端，所以「当前选择」必须
# 存在进程内的变量里，并且优先级高于环境变量。
_RUNTIME: dict[str, str | None] = {"provider": None, "model": None}


def set_runtime_provider(provider: str | None, model: str | None = None) -> None:
    """切换当前进程使用的后端。传 None 表示回到 .env 的默认值。"""
    normalized = None
    if provider:
        normalized = PROVIDER_ALIASES.get(provider.strip().lower())
        if normalized is None:
            raise ValueError(f"未知的后端：{provider}；可选：{', '.join(VALID_PROVIDERS)}")
    _RUNTIME["provider"] = normalized
    _RUNTIME["model"] = (model or "").strip() or None
    # 缓存里放着已经建好连的 ChatModel，不清理就会出现「切了后端还在用旧的」
    _CACHE.clear()


def runtime_provider() -> str | None:
    return _RUNTIME["provider"]


def runtime_model() -> str | None:
    return _RUNTIME["model"]


def resolve_backend(provider: str | None = None, model: str | None = None) -> Backend:
    """决定本次用哪个后端，只做决策不建立连接。"""
    raw = provider or _RUNTIME["provider"] or config.get("LLM_PROVIDER", "auto") or "auto"
    provider = PROVIDER_ALIASES.get(str(raw).strip().lower(), str(raw).strip().lower())
    model = model or _RUNTIME["model"] or config.get("LLM_MODEL")
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

    if provider in {"auto", "cloud"}:
        settings = cloud_settings()
        if settings["api_key"]:
            return Backend(
                "cloud",
                model or settings["model"],
                settings["base_url"],
                {"api_key": settings["api_key"]},
            )
        problems.append("未配置 DASHSCOPE_API_KEY（或通用的 CLOUD_API_KEY）")

    raise LLMUnavailable(
        "没有可用的模型后端：\n  - "
        + "\n  - ".join(problems)
        + "\n三条自救路径（任选其一）：\n"
        "  A. 本地跑（推荐，零费用）：装好 Ollama 后 `ollama pull qwen3:8b`，"
        "确认 `curl http://127.0.0.1:11434/api/tags` 有输出；\n"
        "  B. 走云端：在阿里云百炼（或其他 OpenAI 兼容厂商）申请密钥，"
        "写进 .env 的 CLOUD_API_KEY，必要时同时改 CLOUD_BASE_URL；\n"
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
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # 可选依赖缺失时给出可执行的安装命令
            raise LLMUnavailable(
                "已选择云端后端，但缺少可选依赖。执行：\n"
                "  pip install langchain-openai\n"
                f"（原始错误：{exc}）"
            ) from exc

        kwargs = {
            "model": backend.model,
            "api_key": backend.extra.get("api_key") or "",
            "base_url": backend.base_url,
            "temperature": temperature,
            # 云端失败要快速暴露：默认 SDK 会静默重试两三次，界面会像卡死一样
            "timeout": config.get_float("CLOUD_TIMEOUT", 60.0),
            "max_retries": 1,
        }
        if streaming:
            kwargs["streaming"] = True
        llm = ChatOpenAI(**kwargs)

    # 挂后端信息，脚本/报告里可以打出「这次到底走了哪条路」
    llm._backend = backend  # type: ignore[attr-defined]
    _CACHE[key] = llm
    return llm


def describe(llm=None) -> str:
    llm = llm or get_llm()
    backend = getattr(llm, "_backend", None)
    return backend.banner() if backend else "[LLM] 未知后端"


def probe(cloud: bool = False) -> dict:
    """给 `main.py doctor` / Web 状态面板用的环境体检：不建立模型连接，只回答「能不能用」。

    `cloud=True` 时额外拉一次云端模型清单（有网络往返，默认不拉，避免拖慢 doctor）。
    """
    available = list_local_models()
    settings = cloud_settings()
    info = {
        "ollama_url": ollama_base_url(),
        "ollama_reachable": bool(available),
        "local_models": available,
        "picked": pick_local_model(available, config.get("LLM_MODEL")),
        "dashscope_key": bool(settings["api_key"]),
        "cloud": {
            "base_url": settings["base_url"],
            "model": settings["model"],
            "has_key": bool(settings["api_key"]),
        },
        "provider": config.get("LLM_PROVIDER", "auto"),
        "runtime_provider": _RUNTIME["provider"],
        "runtime_model": _RUNTIME["model"],
    }
    if cloud:
        models, error = list_cloud_models()
        info["cloud"]["models"] = models
        info["cloud"]["error"] = error
    try:
        info["backend"] = resolve_backend().banner()
        info["backend_info"] = resolve_backend().as_dict()
    except LLMUnavailable as exc:
        info["backend"] = f"不可用：{exc}".splitlines()[0]
        info["backend_info"] = None
    return info


# ---------------------------------------------------------------- 错误翻译
# 厂商原始错误对用户毫无帮助（`RateLimitError: Error code: 429 - {...}`），
# 但里面藏着「下一步该做什么」。这里按状态码/错误码翻译成一句可操作的话。
# 返回 None 表示认不出 —— 认不出就照原样抛，不做过度猜测。
_HTTP_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        # 注意别用裸 "quota"：403 的 AllocationQuota.FreeTierOnly 也含这个词，
        # 会把「没有权限」误报成「频率超限」。
        ("429", "limit_requests", "rate limit", "exceeded", "insufficient_quota"),
        "云端拒绝了本次请求：额度或调用频率超限。"
        "处理：① 到厂商控制台看余额/免费额度是否用尽；"
        "② 换一个更便宜的模型（--model qwen-turbo、deepseek-chat 等）；"
        "③ 切回本地（--provider ollama）。",
    ),
    (
        ("401", "invalid_api_key", "authentication"),
        "云端拒绝了密钥（401）。检查 .env 里的 CLOUD_API_KEY："
        "是否复制完整、有没有多余空格或引号。",
    ),
    (
        ("403", "free tier", "allocationquota", "access denied"),
        "云端返回 403：该密钥没有此模型的权限，或免费额度已用尽。"
        "到控制台开通/充值，或换模型、切回本地。",
    ),
    (
        ("404", "model_not_found", "does not exist", "invalid_model"),
        "模型名不存在（404）。用 --model 指定这个密钥确实可用的模型名，"
        "或 `python main.py doctor` 看可用模型列表。",
    ),
    (
        ("connect", "timeout", "timed out", "connection", "unreachable"),
        "连不上模型服务。本地路由：确认 `ollama serve` 在跑"
        "（curl http://127.0.0.1:11434/api/tags 有输出）；"
        "云端路由：确认网络/代理能访问 CLOUD_BASE_URL。",
    ),
)


def explain_error(exc: BaseException) -> str | None:
    """把模型调用的异常翻成用户能照着做的一句话；认不出返回 None。"""
    raw = f"{type(exc).__name__}: {exc}"
    low = raw.lower()
    for needles, hint in _HTTP_HINTS:
        if any(n in low for n in needles):
            return hint
    return None
