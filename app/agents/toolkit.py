"""工具调用型 Agent：5 个文本/数据工具交给模型自选。

工程上唯一值得注意的是 **Agent 的构建成本**。
`create_agent(...)` 会做工具 schema 绑定、中间件装配，在接云端模型时还隐含一次连接初始化；
如果每次处理任务都重新建一次，批量跑任务时会看到「模型没变、但每轮都慢一截」。
所以这里用模块级懒加载单例：第一次用时构建，后续复用。
"""

from __future__ import annotations

from ..llm import describe, get_llm
from ..tools import TEXT_TOOLS
from ..tracing import AgentRun, build_agent, run_agent

SYSTEM_PROMPT = (
    "你是文本与数据助手。规则：\n"
    "1. 必须调用工具完成统计、转换、查找、替换，不要凭印象改写数据；\n"
    "2. 用户给的数据原样传给工具，不要自行增删字段或「顺手美化」；\n"
    "3. 回答用中文，先给结论，再给工具产出的细节。"
)

SAMPLES = {
    "analysis": (
        "用 text_analysis 分析这段文本，然后告诉我它的情感倾向和行数："
        "「这个产品非常好用，我很喜欢它的设计，使用体验非常棒！\n"
        "价格也很合理，推荐大家购买。\n"
        "客服态度也很好，解答问题很及时。」"
    ),
    "convert": (
        "把下面这段 CSV 转成 JSON：\n"
        "name,age,comment\n张三,25,\"产品很好, 物流很快\"\n李四,30,服务太慢"
    ),
}

DEFAULT_TASK = SAMPLES["analysis"]

_agent = None


def build():
    """懒加载单例：同一个进程内复用同一个 Agent 实例。"""
    global _agent
    if _agent is None:
        _agent = build_agent(get_llm(temperature=0.0), TEXT_TOOLS, SYSTEM_PROMPT)
    return _agent


def run(task: str = DEFAULT_TASK, trace: bool = True) -> AgentRun:
    return run_agent(build(), task, trace=trace)


def banner() -> str:
    return describe(get_llm())
