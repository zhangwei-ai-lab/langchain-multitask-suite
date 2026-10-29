"""执行轨迹：把 Agent 的「思考→调用工具→看结果→再决定」变成可读、可落盘的结构。

为什么单独抽出这一层
--------------------
Agent 调试最痛苦的不是「答案不对」，而是「不知道它为什么不对」。
把每次工具调用、每次工具返回、每一步耗时都收集成结构化 Step 列表：

* 终端里能直接看到 `→ 调用 / ← 返回`，一眼判断模型有没有乱调工具；
* 同一条轨迹能原样写进 Markdown 诊断报告，事后复盘；
* 上层用例（网络诊断、研究助理）共用一套采集逻辑，不各写一遍 stream 解析。

LangChain 1.x 的 `agent.stream(..., stream_mode="updates")` 每次 yield
`{节点名: {"messages": [...]}}`，节点名通常是 `model` / `tools`。
这里不假设节点名，只按消息类型判断，换 Agent 实现也不会失效。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, ToolMessage


@dataclass
class Step:
    kind: str  # tool_call | tool_result | assistant
    name: str
    detail: str


@dataclass
class AgentRun:
    question: str
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def tool_calls(self) -> list[str]:
        return [s.name for s in self.steps if s.kind == "tool_call"]

    def to_markdown(self, title: str = "诊断报告") -> str:
        lines = [
            f"# {title}",
            "",
            f"- 问题：{self.question}",
            f"- 耗时：{self.seconds:.1f}s",
            f"- 工具调用序列：{' → '.join(self.tool_calls) or '（无）'}",
            "",
            "## 排查轨迹",
            "",
        ]
        for idx, step in enumerate(self.steps, 1):
            icon = {"tool_call": "↳", "tool_result": "←", "assistant": "✎"}.get(step.kind, "·")
            lines.append(f"{idx}. {icon} **{step.name}** {step.detail}")
        lines += ["", "## 最终结论", "", self.answer.strip() or "（模型没有返回文本）", ""]
        return "\n".join(lines)


def run_agent(
    agent, question: str, trace: bool = True, preview: int = 160, history: list | None = None
) -> AgentRun:
    """跑一次 Agent，收集轨迹并返回最终文本。

    `history` 是可选的既有对话（`[("user", ...), ("assistant", ...)]` 形式），
    传进来即实现多轮；不传就是单轮。
    """
    run = AgentRun(question=question)
    started = time.perf_counter()
    messages = list(history or []) + [("user", question)]

    for chunk in agent.stream({"messages": messages}, stream_mode="updates"):
        for _node, payload in chunk.items():
            for message in (payload or {}).get("messages") or []:
                for call in getattr(message, "tool_calls", None) or []:
                    args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
                    detail = f"({args})"
                    run.steps.append(Step("tool_call", call.get("name", "?"), detail))
                    if trace:
                        print(f"  ↳ 调用 {call.get('name')}({args})")

                if isinstance(message, ToolMessage):
                    text = str(message.content).replace("\n", " ")
                    run.steps.append(
                        Step("tool_result", message.name or "tool", text[:preview])
                    )
                    if trace:
                        print(f"  ← 返回 {text[:preview]}")

                if isinstance(message, AIMessage) and not (message.tool_calls or []):
                    content = (message.content or "").strip()
                    if content:
                        run.answer = content
                        run.steps.append(Step("assistant", "回答", f"{len(content)} 字"))
                        if trace:
                            print(f"  ✎ 给出回答（{len(content)} 字）")

    run.seconds = time.perf_counter() - started
    return run


def build_agent(llm, tools, system_prompt: str):
    """统一封装 create_agent，业务侧不用再关心版本导入路径。"""
    from langchain.agents import create_agent

    return create_agent(llm, tools, system_prompt=system_prompt)
