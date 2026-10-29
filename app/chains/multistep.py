"""多段流水线：中文提问 → 英译 → 英文要点分析 → 回译中文。

这个模块的核心不是「翻译」，而是**数据形状对齐**。

LCEL 的 `A | B` 语义是「A 的输出原样作为 B 的输入」。第三段模板要的是
`{"output": ...}`，而第二段输出的是纯字符串。这里有一个容易误判的细节：

* 如果下游模板**只有一个变量**，langchain-core 会把字符串直接当成那个变量的值，
  于是 `... | translate_to_cn` 看起来能跑通 —— 这不是正确，是运气；
* 一旦模板多出一个变量（比如加一句「风格：{style}」），同样的链立刻抛
  `TypeError: Expected mapping type as input to ChatPromptTemplate`。

所以本项目显式插入 `to_output_dict` 做适配：链上的每一步输入输出形状都写在代码里，
不依赖某个版本的隐式兜底。这个适配只有一行，但它是「链式编程能跑」和「改一行就崩」
之间的分界线，因此做成具名函数并配单测（tests/test_chain_shapes.py）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

DEFAULT_QUESTION = "北京有哪些值得去的地方？简略回答，200 字以内。"


def to_output_dict(value) -> dict:
    """形状适配：把上一步的字符串包装成下一步模板需要的 `{"output": ...}`。

    上一步若已经是字典则原样透传，保证这个函数可以安全地放在任意两段之间。
    """
    if isinstance(value, dict):
        return value
    return {"output": value}


@dataclass
class Stage:
    name: str
    text: str
    seconds: float


def build_stages(llm):
    """三个子任务，各自是「模板 | 模型 | 解析器」的小链，可独立调用也可拼接。"""
    translate_to_en = (
        ChatPromptTemplate.from_template(
            "Translate the following Chinese text into concise English:\n{input}"
        )
        | llm
        | StrOutputParser()
    )
    analyze_en = (
        ChatPromptTemplate.from_template(
            "Summarize the key points of this text in one short English paragraph:\n{text}"
        )
        | llm
        | StrOutputParser()
    )
    translate_to_cn = (
        ChatPromptTemplate.from_template(
            "把下面这段英文翻译成简洁的中文，保留关键信息：\n{output}"
        )
        | llm
        | StrOutputParser()
    )
    return translate_to_en, analyze_en, translate_to_cn


def build_workflow(llm):
    """完整链。注意第三段前面的 `RunnableLambda(to_output_dict)` —— 形状适配层。"""
    translate_to_en, analyze_en, translate_to_cn = build_stages(llm)
    return (
        {"text": translate_to_en}
        | analyze_en
        | RunnableLambda(to_output_dict)
        | translate_to_cn
    )


def run_stepwise(llm, question: str) -> tuple[list[Stage], float]:
    """分步执行并记录每段耗时。链路越长延迟越累加，分段计时是优化的前提。"""
    translate_to_en, analyze_en, translate_to_cn = build_stages(llm)
    stages: list[Stage] = []
    started = time.perf_counter()

    for name, chain, payload in (
        ("中文 → 英文", translate_to_en, {"input": question}),
        ("英文要点提炼", analyze_en, None),
        ("英文 → 中文", translate_to_cn, None),
    ):
        t0 = time.perf_counter()
        if payload is None:
            previous = stages[-1].text
            payload = {"text": previous} if name == "英文要点提炼" else to_output_dict(previous)
        text = chain.invoke(payload).strip()
        stages.append(Stage(name, text, time.perf_counter() - t0))
    return stages, time.perf_counter() - started


def run_stream(workflow, question: str) -> tuple[str, float, float]:
    """流式执行，返回 (完整文本, 首字延迟, 总耗时)。"""
    started = time.perf_counter()
    first_chunk_at = None
    pieces: list[str] = []
    for chunk in workflow.stream({"input": question}):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter() - started
        pieces.append(chunk)
        print(chunk, end="", flush=True)
    print()
    return "".join(pieces), first_chunk_at or 0.0, time.perf_counter() - started
