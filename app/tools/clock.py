"""时间工具：让「今天几号」「现在几点」这类问题有确定答案。

小模型对当前日期几乎必错（训练数据截止后就没有时间概念），
而日期又是一切「最近/今年/几天后」推理的前提。
把它做成工具，并在提示词里要求「涉及时间必须先调用」，比在 prompt 里写死日期可靠得多。
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


def now_text() -> str:
    now = datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    return (
        f"{now:%Y-%m-%d %H:%M:%S} 星期{weekday}"
        f"（{now.year} 年第 {now.timetuple().tm_yday} 天）"
    )


@tool
def current_datetime() -> str:
    """获取当前日期与时间（含星期、年内第几天）。凡涉及「今天/现在/最近」的问题都要先调用。"""
    return now_text()
