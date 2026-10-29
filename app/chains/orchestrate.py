"""LCEL 编排：串行、并行、注入中间结果，以及「工具取数 + 模型组织语言」。

和 Agent 的分工
---------------
Agent 由模型决定调什么工具、调几次；LCEL 由代码写死顺序。
前者适合开放任务，后者适合确定性流水线 —— 而且成本、延迟、可测试性都更好。
真实项目里两者常常混用：用 LCEL 把数据算准，用一次模型调用生成人话。

本模块把三个层次的编排都摆出来，用同一份文本跑一遍，差异一目了然。
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from ..tools.textdata import analyze_text, count_lines

SAMPLE_TEXT = (
    "这个产品非常好用，我很喜欢它的设计，使用体验非常棒！\n"
    "价格也很合理，推荐大家购买。\n"
    "客服态度也很好，解答问题很及时。"
)


def _format_analysis(text: str) -> str:
    s = analyze_text(text)
    return (
        f"字符数 {s['char_count']}，中文字数 {s['cjk_count']}，"
        f"英文词数 {s['en_words']}，情感倾向 {s['sentiment']}"
    )


# ---- 分支 1：单步工具调用（RunnableLambda 包装普通函数）----
analysis_branch = RunnableLambda(lambda x: _format_analysis(x["text"]))
lines_branch = RunnableLambda(lambda x: count_lines.invoke({"content": x["text"]}))


# ---- 分支 2：串行 + assign（保留原始入参，同时注入中间结果）----
def build_assign_chain():
    return RunnablePassthrough.assign(analysis=analysis_branch) | RunnableLambda(
        lambda x: {
            "analysis": x["analysis"],
            "lines": lines_branch.invoke({"text": x["text"]}),
        }
    )


# ---- 分支 3：并行（两个分支吃同一份入参，互不阻塞）----
def build_parallel_chain():
    return RunnableParallel(analysis=analysis_branch, lines=lines_branch)


def _summary_for_model(text: str) -> str:
    """喂给模型的摘要只保留「对文案判断有用」的三项。

    第一版把「英文词数 0」也塞了进去，模型据此给出「建议增加英文词汇」——
    数据没错，结论跑偏。给模型的中间结果要按目标裁剪，不是越多越好。
    """
    s = analyze_text(text)
    return (
        f"字符数 {s['char_count']}，行数 {s['lines']}，情感倾向 {s['sentiment']}"
        f"（正向词 {s['positive_hits']} / 负向词 {s['negative_hits']}）"
    )


# ---- 分支 4：并行取数 → 汇总 → 模型给建议 ----
def build_advice_chain(llm):
    summarize = (
        ChatPromptTemplate.from_template(
            "以下是一段产品文案的客观分析，请给出一句可执行的改进建议"
            "（针对情感表达或篇幅节奏），不超过 40 字：\n{data}"
        )
        | llm
        | StrOutputParser()
    )
    return RunnableLambda(lambda x: {"data": _summary_for_model(x["text"])}) | summarize


def run_all(llm, text: str = SAMPLE_TEXT) -> dict:
    """跑完四种编排并返回结果，供 CLI 打印与验收报告引用。"""
    return {
        "single": analysis_branch.invoke({"text": text}),
        "assign": build_assign_chain().invoke({"text": text}),
        "parallel": build_parallel_chain().invoke({"text": text}),
        "advice": build_advice_chain(llm).invoke({"text": text}).strip(),
    }
