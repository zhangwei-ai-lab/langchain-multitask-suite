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

import json
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


def _text_of(chunk) -> str:
    """把消息内容统一成字符串。

    同一份代码在不同厂商下拿到的 content 形状不一样：
    本地 Ollama 是纯字符串，云端 OpenAI 兼容端点在有工具调用时可能给 list[dict] 分块。
    这里只取文本片段，图片/音频块直接跳过 —— 本项目不用多模态。
    """
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


@dataclass
class StreamEvents:
    """流式事件的类型常量。Web 层把它原样转成 SSE，前端按 type 分发。"""

    TOKEN = "token"  # {"type":"token","text":"..."}
    TOOL_CALL = "tool_call"  # {"type":"tool_call","name":..,"args":..}
    TOOL_RESULT = "tool_result"  # {"type":"tool_result","name":..,"preview":..}
    FINAL = "final"  # {"type":"final","answer":..,"steps":[..],"seconds":..}
    ERROR = "error"  # {"type":"error","message":..}


def _accumulate_call(pending: list[dict], call: dict) -> None:
    """把一个 `tool_call_chunk` 并进对应的调用槽。

    这里有个真实的坑：`tool_call_chunks` 的分片粒度各家不同。
    本地 Ollama 一次并行调 5 个工具时，5 组分片**都不带 id**、`index` 也给 `None`，
    如果按 `index or 0` 分槽，5 个不同工具会被拼成一条
    `{}{"target":...}{"hostname":...}` 的畸形参数，轨迹栏上只显示 1 次调用，
    而实际返回了 5 条结果 —— 界面看起来像「模型乱调工具」，实际是采集层丢信息。

    分槽规则：
    * 有 id 就按 id 找（OpenAI 兼容端点走这条）；
    * 没 id 时并进「上一个还没攒完」的槽；
    * 一旦参数已经能被 `json.loads` 解析完整，就标记 done，下一个分片必须开新槽。
    """
    cid = call.get("id")
    name = call.get("name") or ""
    args = call.get("args") or ""

    slot = None
    if cid:
        slot = next((item for item in pending if item.get("id") == cid), None)
    if slot is None:
        prev = pending[-1] if pending else None
        if prev is not None and not prev["done"] and (not name or name == prev["name"]):
            slot = prev
    if slot is None:
        slot = {"id": cid, "name": "", "args": "", "done": False}
        pending.append(slot)

    if name:
        slot["name"] = name
    slot["args"] += args

    stripped = slot["args"].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            slot["done"] = True
        except ValueError:
            pass  # 还没攒完，继续等下一片


def stream_agent(agent, question: str, history: list | None = None, preview: int = 300):
    """流式跑一次 Agent，逐块 yield 事件字典。

    与 `run_agent` 的区别只在 stream_mode：
    * `updates` 一次给一整个节点输出，只能拿到「完整回答」，适合终端打印；
    * `messages` 把模型吐出的每个 chunk 都给你，能实现打字机效果，适合界面。

    工具调用在 `messages` 模式下是分片到达的（tool_call_chunks），
    所以要按 index 累积参数，攒够了再作为一条 `tool_call` 事件发出去，
    否则同一个工具会被前端画成好几行半截参数。
    """
    from langchain_core.messages import AIMessageChunk, ToolMessage

    messages = list(history or []) + [("user", question)]
    started = time.perf_counter()
    run = AgentRun(question=question)
    pending: list[dict] = []
    answer_parts: list[str] = []

    def flush_pending():
        for item in pending:
            if not item.get("name"):
                continue
            detail = item.get("args") or ""
            run.steps.append(Step("tool_call", item["name"], f"({detail})"))
            events.append(
                {"type": StreamEvents.TOOL_CALL, "name": item["name"], "args": detail}
            )
        pending.clear()

    events: list[dict] = []

    for chunk, _meta in agent.stream({"messages": messages}, stream_mode="messages"):
        if isinstance(chunk, ToolMessage):
            flush_pending()
            text = _text_of(chunk).replace("\n", " ")
            run.steps.append(Step("tool_result", chunk.name or "tool", text[:preview]))
            events.append(
                {
                    "type": StreamEvents.TOOL_RESULT,
                    "name": chunk.name or "tool",
                    "preview": text[:preview],
                }
            )
        elif isinstance(chunk, AIMessageChunk):
            for call in getattr(chunk, "tool_call_chunks", None) or []:
                _accumulate_call(pending, call)
            piece = _text_of(chunk)
            if piece:
                answer_parts.append(piece)
                events.append({"type": StreamEvents.TOKEN, "text": piece})

        # 攒到一批就发出去，前端不必等整轮结束
        while events:
            yield events.pop(0)

    flush_pending()
    while events:
        yield events.pop(0)

    run.answer = "".join(answer_parts).strip()
    run.seconds = time.perf_counter() - started
    if run.answer:
        run.steps.append(Step("assistant", "回答", f"{len(run.answer)} 字"))
    yield {
        "type": StreamEvents.FINAL,
        "answer": run.answer,
        "seconds": round(run.seconds, 2),
        "steps": [{"kind": s.kind, "name": s.name, "detail": s.detail} for s in run.steps],
        "tool_calls": len(run.tool_calls),
    }


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
