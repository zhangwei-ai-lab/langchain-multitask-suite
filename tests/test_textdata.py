"""文本/数据工具：中文计数、CSV 转义、列顺序 —— 三个最容易写错又最容易忽略的点。"""

from __future__ import annotations

import json

from app.tools.textdata import (
    analyze_text,
    count_lines,
    csv_to_json_text,
    data_conversion,
    find_text,
    json_to_csv_text,
    replace_text,
)


def test_analyze_text_counts_chinese_chars_not_words():
    stats = analyze_text("这个产品非常好用\n价格合理")
    assert stats["cjk_count"] == 12
    assert stats["lines"] == 2
    # 中文没有空格，split() 会得到 1；这里必须是按字统计
    assert stats["cjk_count"] > 1


def test_analyze_text_sentiment():
    assert analyze_text("这个产品很好，我很喜欢，推荐")["sentiment"] == "积极"
    assert analyze_text("服务太差，体验糟糕，很失望")["sentiment"] == "消极"
    assert analyze_text("今天下雨了")["sentiment"] == "中性"


def test_json_to_csv_keeps_first_seen_column_order():
    payload = json.dumps([{"b": 1, "a": 2}, {"a": 3, "c": 4}], ensure_ascii=False)
    csv_text = json_to_csv_text(payload)
    assert csv_text.splitlines()[0] == "b,a,c"


def test_csv_to_json_handles_quoted_comma():
    """字段里带逗号是 CSV 最经典的坑：split(',') 会把一行切成两列错位。"""
    csv_text = 'name,age,comment\n张三,25,"产品很好, 物流很快"\n李四,30,服务太慢'
    rows = json.loads(csv_to_json_text(csv_text))
    assert rows[0]["comment"] == "产品很好, 物流很快"
    assert rows[1]["age"] == "30"


def test_data_conversion_roundtrip():
    csv_text = 'a,b\n1,"x,y"'
    js = data_conversion.invoke(
        {"input_data": csv_text, "input_format": "csv", "output_format": "json"}
    )
    back = data_conversion.invoke(
        {"input_data": js, "input_format": "json", "output_format": "csv"}
    )
    assert back.strip().endswith('"x,y"')


def test_data_conversion_rejects_unsupported_direction():
    result = data_conversion.invoke(
        {"input_data": "x", "input_format": "yaml", "output_format": "csv"}
    )
    assert "不支持的转换方向" in result


def test_count_lines_separates_blank_lines():
    out = count_lines.invoke({"content": "第一行\n\n第二行\n"})
    assert "2 行有效" in out


def test_find_and_replace_text():
    assert "第 2 行" in find_text.invoke({"content": "甲\n乙关键词\n丙", "search_text": "关键词"})
    assert "未找到" in find_text.invoke({"content": "甲", "search_text": "zzz"})
    replaced = replace_text.invoke({"content": "a-b-a", "old_text": "a", "new_text": "c"})
    assert "共 2 处" in replaced and "c-b-c" in replaced
