"""换算型 Agent：实时天气 + 联网兜底 + 安全计算器，做「取证 → 计算 → 结论」。

这个用例专门用来暴露小模型的一个通病：**能心算就绝不调工具**。
8B 模型看到「10.1°C 是多少华氏度」会直接给一个近似值，
数字看着像对，但换算过程不可复现、无法审计。

对策不是「命令它别算」，而是把「任何涉及数字的步骤都必须调 calculator」写死进提示词，
并在工具描述里重复一遍 —— 模型对工具描述（description）的服从度往往高于 system prompt。
"""

from __future__ import annotations

from ..llm import get_llm
from ..tools import CALC_TOOLS
from ..tracing import AgentRun, build_agent, run_agent

SYSTEM_PROMPT = (
    "你是一个会使用工具的换算助理。执行纪律：\n"
    "1. 先取数：气温用 get_weather 查实时数据（查不到再用 web_search）；\n"
    "2. 再计算：任何换算、比例、四则运算都必须调用 calculator，包括「除以 4」这种简单除法；\n"
    "3. 不允许自己心算给出数字，也不允许编造气温；\n"
    "4. 最后用中文回答，并逐一说明每个数字来自哪一步工具。"
)

DEFAULT_QUESTION = "北京现在气温是多少华氏度？这个华氏温度的 1/4 是多少？"


def build():
    return build_agent(get_llm(temperature=0.1), CALC_TOOLS, SYSTEM_PROMPT)


def run(question: str = DEFAULT_QUESTION, trace: bool = True) -> AgentRun:
    return run_agent(build(), question, trace=trace)
