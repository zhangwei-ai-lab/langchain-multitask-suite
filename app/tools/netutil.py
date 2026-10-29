"""HTTP 小工具：带重试的 GET + HTML 去标签。

工具层自己实现一次 GET，是为了不引入 requests/httpx 这类额外依赖 ——
标准库 urllib 足够覆盖「抓网页摘要」这种轻量场景，装依赖越少，别人 clone 下来越容易跑通。
"""

from __future__ import annotations

import html as html_mod
import re
import time
import urllib.error
import urllib.request

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def http_get(url: str, timeout: int = 15, retries: int = 1, headers: dict | None = None) -> str:
    """GET 一个 URL 返回文本。失败重试 retries 次后抛最后一个异常。

    重试不是「保险起见」而是实测需要：经过代理/公司网关时偶发 502，
    同一 URL 隔 1 秒重试基本都能成功，退避一次的成本远小于让整个 Agent 失败。
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                **(headers or {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, "replace")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    assert last is not None
    raise last


def strip_tags(fragment: str) -> str:
    """把一段 HTML 压成纯文本。脚本/样式整块丢掉，避免把 JS 当正文读进上下文。"""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    return re.sub(r"\s+", " ", text).strip()
