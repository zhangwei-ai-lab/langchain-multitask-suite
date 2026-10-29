"""天气工具：Open-Meteo 免费接口（无需 Key），返回可直接引用的结构化数据。

为什么不用「搜索天气」
----------------------
最初版本是让搜索工具搜「北京今天气温」，再从网页摘要里抠数字。
实测下来摘要里几乎没有实时温度（多为「今天多云转晴，18~24℃」这种日内区间），
模型拿不到确定值就只能自己编 —— 这是「看起来能用、结论不可信」的典型。

换成结构化接口后，温度、体感、湿度、风速都是接口直出的数字，
模型只负责组织语言，数据可核验。Open-Meteo 免注册、免密钥，适合做默认数据源。
"""

from __future__ import annotations

import json
import urllib.parse

from langchain_core.tools import tool

from .netutil import http_get

# WMO 天气现象编码 → 中文（Open-Meteo 沿用该编码）
WMO_CODES = {
    0: "晴",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴",
    45: "有雾",
    48: "雾凇",
    51: "轻微毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def geocode(city: str) -> dict | None:
    """城市名 → 经纬度与时区。查不到返回 None。"""
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": city, "count": 1, "language": "zh", "format": "json"}
    )
    payload = json.loads(http_get(url, timeout=15))
    hits = payload.get("results") or []
    return hits[0] if hits else None


def current_weather(city: str) -> dict:
    """取某城市实时天气。返回字段化的 dict，方便上层复用与落盘。"""
    place = geocode(city)
    if not place:
        raise LookupError(f"未找到城市「{city}」，请换一个写法（如「北京」「上海」）")
    params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
        "weather_code,wind_speed_10m",
        "timezone": place.get("timezone", "auto"),
    }
    payload = json.loads(
        http_get(
            "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params),
            timeout=15,
        )
    )
    cur = payload.get("current") or {}
    if cur.get("temperature_2m") is None:
        raise LookupError("天气接口没有返回当前温度")
    return {
        "city": "".join(
            filter(None, [place.get("country"), place.get("admin1"), place.get("name")])
        ),
        "observed_at": cur.get("time", ""),
        "weather": WMO_CODES.get(int(cur.get("weather_code", -1)), "未知"),
        "temperature_c": cur.get("temperature_2m"),
        "apparent_c": cur.get("apparent_temperature"),
        "humidity": cur.get("relative_humidity_2m"),
        "wind_kmh": cur.get("wind_speed_10m"),
    }


@tool
def get_weather(city: str) -> str:
    """查询城市实时天气，温度单位为摄氏度。数据源 Open-Meteo，无需 API Key。

    参数:
        city: 城市名，支持中英文，例如 "北京" / "Shanghai"
    返回:
        天气现象、气温、体感温度、湿度、风速；失败时返回可读原因
    """
    try:
        w = current_weather(city)
    except LookupError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return f"天气接口调用失败（{type(exc).__name__}: {exc}），可改用 web_search 兜底"
    return (
        f"{w['city']} 实时天气（{w['observed_at']}）：{w['weather']}，"
        f"气温 {w['temperature_c']}°C，体感 {w['apparent_c']}°C，"
        f"相对湿度 {w['humidity']}%，风速 {w['wind_kmh']} km/h。"
    )
