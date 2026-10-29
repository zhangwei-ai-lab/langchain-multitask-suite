"""文本与数据工具：统计、JSON↔CSV 互转、查找替换。

这一组工具对应「让模型做确定性的事」这个思路：
字符数、列顺序、CSV 转义这类工作不该交给模型猜，
把它们压成确定性函数，模型只负责决定「该调哪个、参数填什么」。

实现上有两个坑值得记住：
1. 中文没有空格，`len(text.split())` 统计出来的「词数」恒等于 1；
2. CSV 不能 `split(",")` —— 字段里带逗号或引号时整行列都会错位，必须用标准库 csv。
"""

from __future__ import annotations

import csv
import io
import json
import re

from langchain_core.tools import tool

POSITIVE_WORDS = ("好", "优秀", "喜欢", "快乐", "成功", "美好", "推荐", "满意", "棒")
NEGATIVE_WORDS = ("差", "糟糕", "讨厌", "悲伤", "失败", "痛苦", "失望", "垃圾", "慢")


def analyze_text(text: str) -> dict:
    """四项统计 + 一个极简情感倾向（关键词计数，不是模型判断）。"""
    positive = sum(text.count(w) for w in POSITIVE_WORDS)
    negative = sum(text.count(w) for w in NEGATIVE_WORDS)
    if positive > negative:
        sentiment = "积极"
    elif negative > positive:
        sentiment = "消极"
    else:
        sentiment = "中性"
    return {
        "char_count": len(text),
        "cjk_count": len(re.findall(r"[\u4e00-\u9fff]", text)),
        "en_words": len(re.findall(r"[A-Za-z]+", text)),
        "lines": len(text.splitlines()),
        "positive_hits": positive,
        "negative_hits": negative,
        "sentiment": sentiment,
    }


def json_to_csv_text(payload: str) -> str:
    """JSON（对象或对象数组）→ CSV 文本。列顺序按首次出现顺序，不排序。"""
    data = json.loads(payload)
    rows = data if isinstance(data, list) else [data]
    if not rows:
        raise ValueError("输入是空数组，没有可转换的行")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("只支持「对象」或「对象数组」结构的 JSON")

    headers: list[str] = []
    for row in rows:
        for key in row:  # dict 保序，天然得到稳定的列顺序
            if key not in headers:
                headers.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().strip()


def csv_to_json_text(payload: str) -> str:
    """CSV 文本 → JSON 数组。用 DictReader 处理引号、逗号、换行。"""
    reader = csv.DictReader(io.StringIO(payload.strip()))
    rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("CSV 至少需要一行标题行和一行数据")
    return json.dumps(rows, ensure_ascii=False, indent=2)


@tool
def text_analysis(text: str) -> str:
    """分析文本：总字符数、中文字数、英文词数、行数、情感倾向。

    参数:
        text: 待分析文本
    """
    s = analyze_text(text)
    return (
        f"文本分析结果：\n"
        f"- 总字符数: {s['char_count']}\n"
        f"- 中文字数: {s['cjk_count']}\n"
        f"- 英文词数: {s['en_words']}\n"
        f"- 行数: {s['lines']}\n"
        f"- 情感倾向: {s['sentiment']}"
        f"（正向词 {s['positive_hits']} / 负向词 {s['negative_hits']}）"
    )


@tool
def data_conversion(input_data: str, input_format: str, output_format: str) -> str:
    """在 JSON 与 CSV 之间互转，正确处理字段内的逗号、引号与换行。

    参数:
        input_data: 待转换的数据文本
        input_format: 输入格式，json 或 csv
        output_format: 输出格式，json 或 csv
    """
    src = (input_format or "").strip().lower()
    dst = (output_format or "").strip().lower()
    if src == dst:
        return "输入与输出格式相同，无需转换"
    try:
        if src == "json" and dst == "csv":
            return json_to_csv_text(input_data)
        if src == "csv" and dst == "json":
            return csv_to_json_text(input_data)
        return f"不支持的转换方向：{input_format} → {output_format}（当前支持 json↔csv）"
    except Exception as exc:  # noqa: BLE001
        return f"转换失败：{type(exc).__name__}: {exc}"


@tool
def count_lines(content: str) -> str:
    """统计文本行数（同时给出有效行数与总行数）。参数 content 为文本内容。"""
    total = len(content.splitlines())
    effective = len([line for line in content.splitlines() if line.strip()])
    return f"共 {effective} 行有效内容（总行数 {total}，其中空行 {total - effective}）"


@tool
def find_text(content: str, search_text: str) -> str:
    """在文本中查找字符串，返回命中的行号与整行内容。

    参数:
        content: 文本内容
        search_text: 要查找的字符串（区分大小写）
    """
    if not search_text:
        return "请提供要查找的文本"
    hits = [
        f"第 {idx} 行: {line}"
        for idx, line in enumerate(content.splitlines(), 1)
        if search_text in line
    ]
    if not hits:
        return f"未找到「{search_text}」"
    return f"找到 {len(hits)} 处匹配：\n" + "\n".join(hits)


@tool
def replace_text(content: str, old_text: str, new_text: str) -> str:
    """把文本中的旧内容替换为新内容，返回替换次数与替换后的全文。

    参数:
        content: 原文
        old_text: 被替换的旧文本
        new_text: 替换成的新文本
    """
    if not old_text:
        return "请提供要替换的旧文本"
    count = content.count(old_text)
    if count == 0:
        return f"未找到「{old_text}」，内容未改动"
    return f"替换完成，共 {count} 处。\n新内容：\n{content.replace(old_text, new_text)}"
