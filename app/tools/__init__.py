"""工具注册表：所有 `@tool` 在这里集中登记，供 Agent 按名字取用。

集中登记而不是各处 import 一堆函数，好处是「有哪些工具、各自能干什么」有唯一答案：
`python main.py tools` 打印的清单、Agent 装配用的工具集、单测覆盖的范围，读的是同一份定义。
"""

from __future__ import annotations

from .calc import calculator
from .clock import current_datetime
from .netdiag import (
    check_http,
    check_port,
    dns_lookup,
    local_network_info,
    ping_host,
    traceroute,
)
from .search import web_search
from .textdata import count_lines, data_conversion, find_text, replace_text, text_analysis
from .weather import get_weather

REGISTRY = {
    tool.name: tool
    for tool in (
        web_search,
        get_weather,
        calculator,
        current_datetime,
        text_analysis,
        data_conversion,
        count_lines,
        find_text,
        replace_text,
        ping_host,
        dns_lookup,
        check_port,
        check_http,
        traceroute,
        local_network_info,
    )
}

# 按用途分组的工具集，Agent 装配时直接引用
WEB_TOOLS = [web_search]
RESEARCH_TOOLS = [web_search, current_datetime]
CALC_TOOLS = [get_weather, web_search, calculator]
TEXT_TOOLS = [text_analysis, data_conversion, count_lines, find_text, replace_text]
NET_TOOLS = [ping_host, dns_lookup, check_port, check_http, traceroute, local_network_info]


def by_names(names: list[str]) -> list:
    """按名字取工具，名字写错时立刻报错，避免 Agent 静默少一个工具。"""
    missing = [name for name in names if name not in REGISTRY]
    if missing:
        raise KeyError(f"未注册的工具：{', '.join(missing)}；可用：{', '.join(REGISTRY)}")
    return [REGISTRY[name] for name in names]


def catalog() -> list[tuple[str, str]]:
    """(工具名, 一句话说明) 列表，给 `main.py tools` 用。"""
    out = []
    for name, tool in REGISTRY.items():
        first_line = (tool.description or "").strip().splitlines()[0]
        out.append((name, first_line))
    return out


__all__ = [
    "REGISTRY",
    "WEB_TOOLS",
    "RESEARCH_TOOLS",
    "CALC_TOOLS",
    "TEXT_TOOLS",
    "NET_TOOLS",
    "by_names",
    "catalog",
]
