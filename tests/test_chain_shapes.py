"""链的形状：用假 Runnable 复现「字符串喂给模板」的失败场景，并验证适配层能救回来。

这条测试的价值在于它**不需要模型**就能锁住一个曾经导致整条链崩溃的 bug：
`A | B` 里 B 需要 `{"output": ...}`，A 却给出字符串，运行时报
`Input to PromptTemplate is missing variables`。
"""

from __future__ import annotations

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.chains.multistep import to_output_dict
from app.chains.orchestrate import build_assign_chain, build_parallel_chain


def test_to_output_dict_wraps_string():
    assert to_output_dict("结果") == {"output": "结果"}


def test_to_output_dict_passes_dict_through():
    payload = {"output": "结果", "meta": 1}
    assert to_output_dict(payload) is payload


def test_single_variable_prompt_silently_coerces_string():
    """单变量模板会把字符串当变量值 —— 这是「看似能跑」的隐式兜底，不是正确性保证。"""
    template = ChatPromptTemplate.from_template("翻译成中文：{output}")
    rendered = template.invoke("纯字符串")
    assert "纯字符串" in rendered.messages[0].content


def test_second_variable_breaks_the_same_chain():
    """多一个变量，隐式兜底立刻失效：同样的数据流直接抛 TypeError。

    这就是为什么本项目要显式插 to_output_dict —— 不依赖「恰好只有一个变量」。
    """
    template = ChatPromptTemplate.from_template("翻译成中文（风格：{style}）：{output}")
    chain = RunnableLambda(lambda _x: "纯字符串") | template
    with pytest.raises(TypeError) as excinfo:
        chain.invoke({"input": "x"})
    assert "Expected mapping type" in str(excinfo.value)


def test_adapter_fixes_chain():
    """加了适配层后，整条链能跑通（这里用假模型，断言的是数据流而不是模型输出）。"""
    fake_translate = RunnableLambda(lambda x: "EN:" + x["input"])
    fake_analyze = RunnableLambda(lambda x: "AN:" + x["text"])
    fake_back = RunnableLambda(lambda x: "CN:" + x["output"])

    workflow = (
        {"text": fake_translate}
        | fake_analyze
        | RunnableLambda(to_output_dict)
        | fake_back
    )
    assert workflow.invoke({"input": "你好"}) == "CN:AN:EN:你好"


def test_assign_chain_keeps_original_input():
    result = build_assign_chain().invoke({"text": "这个产品很好，我很喜欢"})
    assert set(result) == {"analysis", "lines"}
    assert "积极" in result["analysis"]


def test_parallel_chain_runs_both_branches():
    result = build_parallel_chain().invoke({"text": "第一行\n第二行"})
    assert "字符数" in result["analysis"]
    assert "2 行有效" in result["lines"]
