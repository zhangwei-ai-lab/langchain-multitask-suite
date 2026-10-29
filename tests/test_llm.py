"""模型工厂与搜索词归一化：都是纯逻辑，不需要真的连模型。

云端相关的用例只验证「决策」和「构造」，不发网络请求：
密钥有没有被读到、端点有没有被拼对、运行时切换有没有盖住环境变量 ——
这些判断错了，用户看到的就是「配置明明写了却不生效」。
"""

from __future__ import annotations

import json

import pytest

from app.llm import (
    DEFAULT_CLOUD_BASE_URL,
    Backend,
    LLMUnavailable,
    cloud_settings,
    explain_error,
    get_llm,
    pick_local_model,
    resolve_backend,
    set_runtime_provider,
)
from app.tools.search import filter_relevant, normalize_query, plan_query


def test_prefers_known_good_model():
    available = ["llama3.1:8b", "qwen3:8b", "qwen2.5-coder:7b"]
    assert pick_local_model(available) == "qwen3:8b"


def test_user_choice_wins():
    available = ["qwen3:8b", "qwen2.5-coder:7b"]
    assert pick_local_model(available, preferred="qwen2.5-coder:7b") == "qwen2.5-coder:7b"


def test_user_choice_matches_by_name_prefix():
    available = ["qwen2.5-coder-7b-local:latest"]
    assert pick_local_model(available, preferred="qwen2.5-coder-7b-local") == (
        "qwen2.5-coder-7b-local:latest"
    )


def test_falls_back_to_first_available():
    assert pick_local_model(["my-finetune:v3"]) == "my-finetune:v3"


def test_empty_means_no_model():
    assert pick_local_model([]) is None


def test_dashscope_without_key_raises_readable_error(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable) as excinfo:
        resolve_backend(provider="dashscope")
    message = str(excinfo.value)
    assert "DASHSCOPE_API_KEY" in message
    # 报错必须自带自救路径，而不是只丢一句「不可用」
    assert "ollama pull" in message
    assert "pytest" in message


# ------------------------------------------------------------------ 云端后端
def test_cloud_backend_reads_explicit_settings(monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-cloud")
    monkeypatch.setenv("CLOUD_BASE_URL", "https://example.com/v1/")
    monkeypatch.setenv("CLOUD_MODEL", "qwen-plus")
    backend = resolve_backend(provider="cloud")
    assert backend.provider == "cloud"
    assert backend.model == "qwen-plus"
    # 结尾斜杠要统一去掉，否则拼 /chat/completions 会出现双斜杠
    assert backend.base_url == "https://example.com/v1"


def test_dashscope_alias_maps_to_cloud_and_falls_back_to_old_key(monkeypatch):
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("CLOUD_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-legacy")
    backend = resolve_backend(provider="dashscope")
    assert backend.provider == "cloud"
    assert backend.extra["api_key"] == "sk-legacy"
    assert backend.base_url == DEFAULT_CLOUD_BASE_URL


def test_cloud_settings_prefers_new_names(monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-new")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-old")
    monkeypatch.setenv("CLOUD_MODEL", "deepseek-chat")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen-turbo")
    settings = cloud_settings()
    assert settings["api_key"] == "sk-new"
    assert settings["model"] == "deepseek-chat"


def test_cloud_without_any_key_raises_readable_error(monkeypatch):
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable) as excinfo:
        resolve_backend(provider="cloud")
    message = str(excinfo.value)
    assert "CLOUD_API_KEY" in message
    assert "ollama pull" in message


def test_runtime_switch_overrides_env(monkeypatch):
    """界面上的切换必须盖住 .env：否则用户点了「云端」却还在跑本地。"""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("CLOUD_API_KEY", "sk-cloud")
    set_runtime_provider("cloud", "qwen-max")
    try:
        backend = resolve_backend()
        assert backend.provider == "cloud"
        assert backend.model == "qwen-max"
    finally:
        set_runtime_provider(None, None)


def test_runtime_switch_rejects_unknown_provider():
    with pytest.raises(ValueError):
        set_runtime_provider("skynet")


def test_provider_alias_openai_and_api_map_to_cloud(monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-cloud")
    for alias in ("openai", "api", "CLOUD"):
        assert resolve_backend(provider=alias).provider == "cloud"


def test_banner_distinguishes_local_and_cloud():
    assert "本地 Ollama" in Backend("ollama", "m", "http://127.0.0.1:11434").banner()
    assert "云端 API" in Backend("cloud", "m", "http://fake/v1").banner()


def test_backend_dict_never_leaks_api_key():
    backend = Backend("cloud", "m", "http://fake/v1", {"api_key": "sk-secret"})
    assert "sk-secret" not in json.dumps(backend.as_dict(), ensure_ascii=False)


def test_get_llm_builds_openai_compatible_client(monkeypatch):
    """只构造客户端、不发请求：验证「配置 → ChatOpenAI」这条装配链是通的。"""
    monkeypatch.setenv("CLOUD_API_KEY", "sk-cloud")
    monkeypatch.setenv("CLOUD_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("CLOUD_MODEL", "qwen-turbo")
    llm = get_llm(temperature=0.0, provider="cloud")
    assert type(llm).__name__ == "ChatOpenAI"
    assert getattr(llm, "_backend").provider == "cloud"


def test_normalize_query_removes_spaces_between_chinese_only():
    assert normalize_query("10月28日 名人 出生") == "10月28日名人出生"
    # 英文单词之间的空格必须保留，否则 "windows 10" 会被拼成 "windows10"
    assert normalize_query("windows 10 release") == "windows 10 release"
    assert normalize_query("北京时间  apple event") == "北京时间 apple event"


def test_plan_query_quotes_date_token_first():
    """日期必须被引号锁成短语，否则搜索引擎会把它切碎（实测返回「Windows 10」）。"""
    plan = plan_query("10月29日 历史上的今天 重大事件")
    assert plan.variants[0].startswith('"10月29日"')
    assert plan.must_include == ["10月29日"]
    assert len(plan.variants) >= 2


def test_plan_query_without_date_keeps_single_variant():
    plan = plan_query("LangChain 1.0 发布")
    assert plan.variants == ["LangChain 1.0 发布"]
    assert plan.must_include == []


def test_filter_relevant_drops_obvious_noise():
    noisy = [
        {"title": "10 （自然数之一）_百度百科", "snippet": "10 是位于 9 与 11 之间的自然数"},
        {"title": "下载 Windows 10 光盘映像", "snippet": "2026年6月10日 ..."},
        {"title": "10月29日 - 百度百科", "snippet": "1914年土耳其加入一战"},
    ]
    kept = filter_relevant(noisy, ["10月29日"])
    assert [item["title"] for item in kept] == ["10月29日 - 百度百科"]


def test_filter_relevant_no_constraint_passes_through():
    rows = [{"title": "任意", "snippet": "内容"}]
    assert filter_relevant(rows, []) is rows


# ---------------------------------------------------------------- 错误翻译
# 这些用例的输入全部来自本次实测抓到的真实报错原文，不是编的字符串。
@pytest.mark.parametrize(
    "raw, must_have",
    [
        (
            "RateLimitError: Error code: 429 - {'error': {'code': 'limit_requests'}}",
            "额度或调用频率超限",
        ),
        (
            "APIStatusError: Error code: 403 - {'code': 'AllocationQuota.FreeTierOnly'}",
            "403",
        ),
        (
            "AuthenticationError: Error code: 401 - {'code': 'invalid_api_key'}",
            "401",
        ),
        (
            "NotFoundError: Error code: 404 - model qwen-x does not exist",
            "模型名不存在",
        ),
        (
            "APIConnectionError: Connection error.",
            "连不上模型服务",
        ),
        (
            "ConnectError: All connection attempts failed",
            "ollama serve",
        ),
    ],
)
def test_explain_error_maps_real_messages(raw, must_have):
    exc = RuntimeError(raw)
    hint = explain_error(exc)
    assert hint is not None, f"没认出来：{raw}"
    assert must_have in hint
    # 提示必须给出下一步动作，否则用户仍然卡住
    assert any(word in hint for word in ("处理", "检查", "确认", "指定", "到控制台"))


def test_explain_error_returns_none_for_unknown():
    """认不出来就返回 None，让上层保留原始异常 —— 宁可难看也别猜错。"""
    assert explain_error(ValueError("一个和模型无关的错误")) is None


# ---------------------------------------------------------------- 命令行接线
# 界面靠 /api/provider 切后端，命令行靠全局 --provider。
# 两条路都必须落到同一个 set_runtime_provider，否则「命令行指定了却没生效」。
def test_cli_rejects_unknown_provider(capsys):
    from app.cli import main

    assert main(["--provider", "foo", "tools"]) == 2
    assert "未知的后端" in capsys.readouterr().err


def test_cli_provider_flag_sets_runtime(monkeypatch, capsys):
    from app import llm
    from app.cli import main

    monkeypatch.setattr(llm, "_RUNTIME", {"provider": None, "model": None})
    assert main(["--provider", "cloud", "--model", "qwen-turbo", "tools"]) == 0
    assert llm._RUNTIME == {"provider": "cloud", "model": "qwen-turbo"}
    capsys.readouterr()
