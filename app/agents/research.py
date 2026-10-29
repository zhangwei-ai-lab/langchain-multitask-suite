"""研究型 Agent：会自己决定「先查时间、再查资料」的检索助理。

第一版提示词写的是原则（「事实类问题必须取证」），跑出来的结果是：
模型调了一次 current_datetime 拿到日期，然后**完全跳过搜索**，凭记忆罗列了三条
「历史上的今天」，其中时间、人物、事件全对不上 —— 它没有撒谎的意图，
只是把「我有相关知识」当成了「我已经取证」。

修法不是把话说得更重，而是给**动作序列 + 反例**（和网络诊断 Agent 同一条经验）：
明确写出「先 current_datetime → 再 web_search(具体关键词) → 不够就换词再搜」，
并加一条硬性约束：除纯时间问题外，正文之前必须至少调用一次 web_search。
"""

from __future__ import annotations

from ..llm import get_llm
from ..tools import RESEARCH_TOOLS
from ..tracing import AgentRun, build_agent, run_agent

SYSTEM_PROMPT = (
    "你是一个严谨的检索助理。\n\n"
    "【动作序列】\n"
    "1. 先调 current_datetime 拿到今天的年月日；\n"
    "2. 把问题里的时间/主体拼成具体关键词，调 web_search 检索（例如「10月29日 历史上的今天 大事」）；\n"
    "3. 检索结果不足以回答时，换一组更具体的词再搜一次，而不是改用记忆作答；\n"
    "4. 基于检索结果组织答案，逐条说明依据来自哪一条结果的标题。\n\n"
    "【硬性要求】\n"
    "- 除「今天几号 / 现在几点」这类纯时间问题外，正文回答之前必须至少调用一次 web_search；\n"
    "- 检索结果与你的记忆冲突时，以检索结果为准；\n"
    "- 检索失败或结果不相干，就回答「我没有查到可靠依据」，不得用记忆填补。\n\n"
    "【反例，不要这样】\n"
    "只调 current_datetime，然后凭记忆写出「某年某月某日发生了某某事件」——这是编造。\n\n"
    "【正例，要这样】\n"
    "current_datetime → web_search(\"10月29日 历史上的今天 重大事件\") → "
    "从返回的标题和摘要里挑出可核实的条目，注明「据 XX 报道」。"
)

DEFAULT_QUESTION = "今天是几月几号？历史上的今天有哪些重要事件？"


def build():
    return build_agent(get_llm(temperature=0.1), RESEARCH_TOOLS, SYSTEM_PROMPT)


def run(question: str = DEFAULT_QUESTION, trace: bool = True) -> AgentRun:
    return run_agent(build(), question, trace=trace)

