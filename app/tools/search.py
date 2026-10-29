"""联网搜索工具：查询改写 + 三级渠道降级 + 相关性过滤。

为什么要三层，而不是「调一个 API 就完事」
----------------------------------------
`load_tools(["serpapi"])` 这类写法在缺 API Key 时是 import 阶段直接抛异常，
整个进程起不来。搜索本质是「尽力而为」：拿不到就明确告诉模型拿不到，
让模型自己决定如实回答，而不是把异常抛给调用方。

三级渠道顺序是实测排的：
  * SerpAPI —— 有 key 时最稳，返回结构化 JSON；
  * Bing 网页 —— 免密钥，可达性最好，摘要质量一般；
  * DuckDuckGo —— 免密钥，部分网络（尤其走代理）会 502，所以排最后。

比渠道更麻烦的是**查询写法**。实测（2026-10，Bing）：
  「10月29日 历史上的今天 重大事件」→ 返回「10（自然数之一）」「Windows 10 光盘映像」
  「"10月29日" 历史上的今天」      → 返回「10月29日历史上的今天」「10月29日-百度百科」
也就是说，中文关键词连写会被搜索引擎切碎，"10" 这个 token 权重过高，
把真正相关的结果挤掉。加引号把日期锁成短语后，命中率才有保证。

结论做成两步，而不是一次「聪明改写」：
  1. `plan_query` 生成有序的查询变体（加引号日期 → 仅日期短语 → 原写法）；
  2. `filter_relevant` 在结果进模型之前按「必须命中日期」过滤一遍 ——
     宁可让模型看到「没搜到」，也不要让它看到「Windows 10」这种伪证据。
     Agent 拿到噪声证据比拿到空结果更危险：它会一本正经地据此推理。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from langchain_core.tools import tool

from .. import config
from .netutil import UA, http_get, strip_tags

_CJK = r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]"
DATE_TOKEN = re.compile(r"(?<!\d)(\d{1,2}月\d{1,2}[日号])")


def normalize_query(query: str) -> str:
    """只删掉「中文与中文之间」的空格，英文单词之间的空格必须保留。"""
    cleaned = re.sub(rf"(?<={_CJK})\s+(?={_CJK})", "", query or "")
    return re.sub(r"\s{2,}", " ", cleaned).strip()


@dataclass
class QueryPlan:
    variants: list[str]
    must_include: list[str]

    def __str__(self) -> str:  # 方便在 trace 里打印
        return " | ".join(self.variants)


def plan_query(query: str) -> QueryPlan:
    """把用户/模型的自然语言查询改写成「按成功率排序」的变体列表。

    含日期时，第一顺位是「日期加引号 + 其余关键词」，第二顺位是「只留引号日期」，
    原写法兜底；同时把日期登记为 must_include，供结果过滤使用。
    """
    query = (query or "").strip()
    dates = list(dict.fromkeys(DATE_TOKEN.findall(query)))
    if not dates:
        return QueryPlan([normalize_query(query)], [])

    quoted = " ".join(f'"{d}"' for d in dates)
    rest = normalize_query(DATE_TOKEN.sub(" ", query))
    variants = []
    if rest:
        variants.append(f"{quoted} {rest}")
    variants.append(quoted)
    variants.append(normalize_query(query))
    return QueryPlan(list(dict.fromkeys(variants)), dates)


def filter_relevant(results: list[dict], must_include: list[str]) -> list[dict]:
    """按「必须命中的关键词」过滤。没有约束条件时原样返回。"""
    if not must_include:
        return results
    kept = []
    for item in results:
        haystack = f"{item.get('title', '')} {item.get('snippet', '')}"
        if any(token in haystack for token in must_include):
            kept.append(item)
    return kept



def _serpapi(query: str, top_k: int) -> list[dict]:
    key = config.get("SERPAPI_API_KEY")
    if not key:
        return []
    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(
        {"q": query, "api_key": key, "hl": "zh-cn", "num": top_k}
    )
    payload = json.loads(http_get(url, timeout=20))
    results = [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
            "source": "SerpAPI",
        }
        for item in (payload.get("organic_results") or [])[:top_k]
    ]
    answer = (payload.get("answer_box") or {}).get("answer")
    if answer:
        results.insert(0, {"title": "直接答案", "snippet": answer, "link": "", "source": "SerpAPI"})
    return results


def _bing(query: str, top_k: int) -> list[dict]:
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "setlang": "zh-CN", "count": top_k}
    )
    page = http_get(url, timeout=15)
    results = []
    for block in re.findall(r'<li class="b_algo".*?</li>', page, re.S)[:top_k]:
        title_match = re.search(r"<h2[^>]*>(.*?)</h2>", block, re.S)
        if not title_match:
            continue
        body_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        results.append(
            {
                "title": strip_tags(title_match.group(1)),
                "snippet": (strip_tags(body_match.group(1)) if body_match else strip_tags(block))[
                    :300
                ],
                "link": "",
                "source": "Bing",
            }
        )
    return results


def _duckduckgo(query: str, top_k: int) -> list[dict]:
    request = urllib.request.Request(
        "https://html.duckduckgo.com/html/",
        data=urllib.parse.urlencode({"q": query}).encode(),
        headers={"User-Agent": UA},
    )
    with urllib.request.urlopen(request, timeout=15) as resp:
        page = resp.read().decode("utf-8", "replace")
    results = []
    for block in re.findall(r'<div class="result__body">(.*?)</div>\s*</div>', page, re.S)[:top_k]:
        title_match = re.search(r'class="result__a"[^>]*>(.*?)</a>', block, re.S)
        if not title_match:
            continue
        snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        results.append(
            {
                "title": strip_tags(title_match.group(1)),
                "snippet": strip_tags(snippet_match.group(1)) if snippet_match else "",
                "link": "",
                "source": "DuckDuckGo",
            }
        )
    return results


def search(query: str, top_k: int = 5) -> list[dict]:
    """按「渠道 × 查询变体」依次尝试，返回结构化结果；全失败返回空列表（不抛异常）。"""
    plan = plan_query(query)
    errors: list[str] = []
    fallback: list[dict] = []  # 有结果但没通过相关性过滤，留着当最后兜底

    for name, fn, needs_norm in (
        ("SerpAPI", _serpapi, False),
        ("Bing", _bing, True),
        ("DuckDuckGo", _duckduckgo, True),
    ):
        for variant in plan.variants:
            q = normalize_query(variant) if needs_norm else variant
            try:
                results = fn(q, top_k)
            except urllib.error.HTTPError as exc:
                errors.append(f"{name}:HTTP {exc.code}")
                break  # 渠道级失败，换渠道，不换变体重试
            except Exception as exc:  # noqa: BLE001 —— 抓取解析异常都不该打断 Agent
                errors.append(f"{name}:{type(exc).__name__}")
                break

            if not results:
                errors.append(f"{name}:空结果")
                continue
            relevant = filter_relevant(results, plan.must_include)
            if relevant:
                if config.get_bool("DEBUG_TOOLS", False):
                    print(f"[search] {name} 命中，查询={q!r}")
                return relevant[:top_k]
            fallback = fallback or results
            errors.append(f"{name}:无相关结果")

    if fallback:
        # 所有变体都没通过相关性检查：宁可少给，也不给噪声 —— 但保留一条也不要空手而归
        return fallback[: min(2, top_k)]
    if config.get_bool("DEBUG_TOOLS", False):
        print(f"[search] 全部失败：{'; '.join(errors)}")
    return []


@tool
def web_search(query: str, top_k: int = 5) -> str:
    """联网搜索实时信息。用于新闻、行情、赛事、人物近况等模型知识之外、或可能过期的信息。

    参数:
        query: 搜索关键词，越具体越好，例如 "2026年诺贝尔物理学奖 得主"
        top_k: 返回条数，默认 5
    返回:
        若干条结果的标题+摘要+来源；检索失败时返回一段说明文字
    """
    results = search(query, top_k=max(1, min(int(top_k), 8)))
    if not results:
        return (
            f"检索「{query}」失败：当前环境无法访问任何搜索渠道，也未配置 SERPAPI_API_KEY。"
            "请如实告知用户「这一点我没有查到依据」，不要编造。"
        )
    lines = []
    for idx, item in enumerate(results, 1):
        lines.append(f"{idx}. {item['title']}\n   {item['snippet']} [来源:{item['source']}]")
    return "\n".join(lines)
