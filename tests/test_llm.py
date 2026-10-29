"""模型工厂与搜索词归一化：都是纯逻辑，不需要真的连模型。"""

from __future__ import annotations

import pytest

from app.llm import LLMUnavailable, pick_local_model, resolve_backend
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
